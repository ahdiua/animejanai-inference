#!/usr/bin/env python3
"""Opt-in CUDA integration regression for aji_encode's muxer interleaving.

A sparse subtitle track (one event at the start) must not make the muxer
hold the whole encoded video until the trailer. Requires a CUDA GPU, the
built aji_encode with its TensorRT runtime, and ffmpeg with libx264 and
FFV1. No models are needed: the chain only performs a 50% resize.

    AJI_ENCODE_BIN=build/aji_encode python3 sanity/test_encode_mux.py

On WSL, put /usr/lib/wsl/lib first in LD_LIBRARY_PATH if a native
libcuda.so.1 is also installed.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


def run(args):
    result = subprocess.run([str(a) for a in args], capture_output=True,
                            text=True, timeout=300)
    if result.returncode:
        raise AssertionError(f'{args[0]} failed ({result.returncode}):\n'
                             f'{result.stderr[-2000:]}')
    return result


def size_mid_encode(args, output, threshold=60):
    """Output file size when PROGRESS first reaches `threshold` percent."""
    with subprocess.Popen([str(a) for a in args], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True) as process:
        size = None
        for line in process.stdout:
            fields = dict(f.split('=', 1) for f in line.split()[1:] if '=' in f)
            if (size is None and fields.get('phase') == 'encode' and
                    int(fields.get('pct', 0)) >= threshold):
                size = output.stat().st_size if output.exists() else 0
        error = process.stderr.read()
    if process.returncode:
        raise AssertionError(f'aji_encode failed ({process.returncode}):\n'
                             f'{error[-2000:]}')
    if size is None:
        raise AssertionError('no PROGRESS line reached '
                             f'{threshold}%:\n{error[-2000:]}')
    return size


class EncoderMuxTests(unittest.TestCase):
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
        cls.temporary = tempfile.TemporaryDirectory(prefix='aji-mux-test-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.config = cls.directory / 'resize.conf'
        cls.config.write_text(
            '[global]\nconfig_version=2\nbackend=TensorRT\n'
            '[slot_1]\nchain_1_model_1_name=\n'
            'chain_1_model_1_resize_factor_before_upscale=50\n')
        subtitle = cls.directory / 'sparse.srt'
        subtitle.write_text('1\n00:00:00,000 --> 00:00:00,100\nSparse\n')
        cls.with_subtitle = cls.directory / 'sub.mkv'
        run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
             'testsrc2=size=1280x720:rate=24:duration=40', '-i', subtitle,
             '-map', '0:v', '-map', '1:s', '-c:v', 'libx264',
             '-preset', 'ultrafast', '-crf', '10', '-c:s', 'srt',
             cls.with_subtitle])

    def test_sparse_subtitle_does_not_buffer_the_encode(self):
        output = self.directory / 'out.mkv'
        mid = size_mid_encode([self.binary, '--input', self.with_subtitle,
                               '--output', output, '--conf', self.config,
                               '--slot', 1, '--model-dir', self.directory,
                               '--vcodec', 'ffv1', '--progress', 'text',
                               '--overwrite'], output)
        final = output.stat().st_size
        streams = run(['ffprobe', '-v', 'error', '-show_entries',
                       'stream=codec_type', '-of', 'csv=p=0', output]).stdout
        self.assertEqual(streams.split(), ['video', 'subtitle'])
        # 60% into a 40 s clip, a 10 s interleave bound has written well
        # over a quarter of the file; a muxer waiting on the subtitle track
        # has written only its header.
        self.assertGreater(mid, final // 4,
                           f'{mid} of {final} bytes written at 60%: the '
                           'muxer is holding the encoded video')


if __name__ == '__main__':
    unittest.main(verbosity=2)
