#!/usr/bin/env python3
"""Opt-in CUDA integration regressions for CPU decode and GPU frame uploads.

Requires a CUDA GPU with H.264 NVDEC support, the built aji_encode/TensorRT
runtime, and ffmpeg/ffprobe with libx264 and FFV1. No models or engines are
needed: the inference chain performs a 50% resize on the GPU.

Run after rebuilding aji_encode:
    AJI_ENCODE_BIN=build/aji_encode python3 sanity/test_encode_decode.py

On WSL, include /usr/lib/wsl/lib and the CUDA runtime directory in
LD_LIBRARY_PATH if they are not already available to the dynamic loader.
"""

from array import array
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


WIDTH, HEIGHT, FRAMES = 128, 96, 6


def command(args):
    result = subprocess.run([str(arg) for arg in args], capture_output=True,
                            timeout=60)
    if result.returncode:
        raise AssertionError(
            f"Command failed ({result.returncode}): {args}\n"
            f"{result.stderr.decode(errors='replace')}"
        )
    return result.stdout


class EncoderDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        executable = os.environ.get('AJI_ENCODE_BIN')
        if not executable:
            raise unittest.SkipTest('set AJI_ENCODE_BIN to opt in to GPU tests')
        cls.binary = Path(executable).resolve()
        if not cls.binary.is_file():
            raise AssertionError(f'aji_encode does not exist: {cls.binary}')
        for tool in ('ffmpeg', 'ffprobe'):
            if not shutil.which(tool):
                raise unittest.SkipTest(f'{tool} is required')
        cls.temporary = tempfile.TemporaryDirectory(prefix='aji-decode-test-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.config = cls.directory / 'resize.conf'
        cls.config.write_text(
            '[global]\nconfig_version=2\nbackend=TensorRT\n'
            '[slot_1]\nchain_1_model_1_name=\n'
            'chain_1_model_1_resize_factor_before_upscale=50\n'
        )
        cls.source8 = cls.directory / 'source8.mkv'
        command(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                 f'testsrc2=size={WIDTH}x{HEIGHT}:rate=5', '-frames:v', FRAMES,
                 '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-color_range', 'tv',
                 '-colorspace', 'bt709', '-color_primaries', 'bt709',
                 '-color_trc', 'bt709', cls.source8])

        # Each plane is spatially constant, so the independently generated
        # reference does not depend on the GPU resampling filter. Distinct
        # chroma planes and frame values detect packing/depth/stale-frame bugs.
        raw = cls.directory / 'source10.yuv'
        expected = array('H')
        with raw.open('wb') as output:
            for frame in range(FRAMES):
                planes = (384 + 32 * frame, 448 + 8 * frame, 608 - 8 * frame)
                for plane, value in enumerate(planes):
                    count = WIDTH * HEIGHT // (4 if plane else 1)
                    pixels = array('H', [value]) * count
                    if sys.byteorder != 'little':
                        pixels.byteswap()
                    output.write(pixels.tobytes())
                    expected.extend(array('H', [value]) * (count // 4))
        cls.expected10 = expected
        cls.source10 = cls.directory / 'source10.mkv'
        command(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pixel_format',
                 'yuv420p10le', '-video_size', f'{WIDTH}x{HEIGHT}', '-framerate',
                 '5', '-color_range', 'tv', '-colorspace', 'bt709',
                 '-color_primaries', 'bt709', '-color_trc', 'bt709',
                 '-i', raw, '-c:v', 'ffv1', '-color_range', 'tv',
                 '-colorspace', 'bt709', '-color_primaries', 'bt709',
                 '-color_trc', 'bt709', cls.source10])
        decoded = command(['ffmpeg', '-v', 'error', '-i', cls.source10,
                           '-f', 'rawvideo', '-pix_fmt', 'yuv420p10le', '-'])
        if decoded != raw.read_bytes():
            raise AssertionError('FFV1 fixture changed the reference pixel values')

    def encode(self, name, source, decoder, ten_bit=False, extra=()):
        output = self.directory / f'{name}.mkv'
        command([self.binary, '--input', source, '--output', output,
                 '--conf', self.config, '--decoder', decoder, '--vcodec', 'ffv1',
                 '--pix-fmt', 'yuv420p10' if ten_bit else 'yuv420p',
                 '--progress', 'none', '--overwrite', *extra])
        stream = json.loads(command(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams',
             '-of', 'json', output]))['streams'][0]
        self.assertEqual((stream['width'], stream['height']),
                         (WIDTH // 2, HEIGHT // 2))
        self.assertEqual(stream['pix_fmt'], 'yuv420p10le' if ten_bit else 'yuv420p')
        pixels = command(['ffmpeg', '-v', 'error', '-i', output, '-map', '0:v:0',
                          '-fps_mode', 'passthrough', '-f', 'rawvideo',
                          '-pix_fmt', stream['pix_fmt'], '-'])
        expected_bytes = FRAMES * (WIDTH // 2) * (HEIGHT // 2) * 3 // 2
        self.assertEqual(len(pixels), expected_bytes * (2 if ten_bit else 1))
        return pixels

    def test_cpu_and_no_zerocopy_match_nvdec_pixels(self):
        gpu = self.encode('nvdec8', self.source8, 'nvdec')
        cpu = self.encode('cpu8', self.source8, 'cpu')
        no_zerocopy = self.encode('no_zerocopy8', self.source8, 'auto',
                                 extra=('--no-zerocopy',))
        gpu_hash = hashlib.sha256(gpu).hexdigest()
        self.assertEqual(hashlib.sha256(cpu).hexdigest(), gpu_hash)
        self.assertEqual(hashlib.sha256(no_zerocopy).hexdigest(), gpu_hash)

    def test_cpu_10bit_upload_preserves_plane_values(self):
        pixels = self.encode('cpu10', self.source10, 'cpu', ten_bit=True)
        actual = array('H')
        actual.frombytes(pixels)
        if sys.byteorder != 'little':
            actual.byteswap()
        self.assertEqual(len(actual), len(self.expected10))
        max_error = max(abs(a - b) for a, b in zip(actual, self.expected10))
        # The resize chain traverses FP16 RGB; allow up to two 10-bit code
        # values of rounding, while detecting planar/P010 packing mistakes.
        self.assertLessEqual(max_error, 2, f'10-bit maximum error: {max_error}')


if __name__ == '__main__':
    unittest.main(verbosity=2)
