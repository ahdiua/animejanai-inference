#!/usr/bin/env python3
"""Check real encoder RIFE scheduling with host-backed inference substitutes.

Requires a C compiler, CUDA headers, FFmpeg libraries, ffmpeg and ffprobe.
No CUDA GPU or model is required: python3 sanity/test_encode_rife.py
"""
import json
from itertools import product
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
HARNESS = r'''
#include <assert.h>
#define main aji_encode_main
#include "ENCODE_SOURCE"
#undef main

static int upscale_calls, rife_calls, scene_mode;
static struct {
    aji_frame a, b, out;
    double t;
    int rife;
} pending[2];
static int pending_count;

/* Publish queued GPU writes only at a completion wait. A RIFE crop followed
 * by upscale executes in stream order, while an early CPU encode sees the
 * original buffer contents and fails the decoded-pixel assertions. */
static void complete_pending(void)
{
    for (int i = 0; i < pending_count; i++) {
        const aji_frame *a = &pending[i].a, *b = &pending[i].b;
        const aji_frame *out = &pending[i].out;
        for (int p = 0; p < 2; p++) {
            for (int y = 0; y < a->height / (p ? 2 : 1); y++) {
                const unsigned char *pa = (unsigned char *)a->plane[p] + y * a->stride[p];
                unsigned char *po = (unsigned char *)out->plane[p] + y * out->stride[p];
                if (!pending[i].rife) {
                    memcpy(po, pa, a->width);
                    continue;
                }
                const unsigned char *pb = (unsigned char *)b->plane[p] + y * b->stride[p];
                double t = pending[i].t;
                for (int x = 0; x < a->width; x++)
                    po[x] = (unsigned char)((1 - t) * pa[x] + t * pb[x]);
            }
        }
    }
    pending_count = 0;
}

/* Supply ordinary ref-counted host frames instead of a CUDA frame pool. */
int av_hwframe_get_buffer(AVBufferRef *pool, AVFrame *f, int flags)
{
    f->width = f->height = 64;
    f->format = AV_PIX_FMT_NV12;
    int r = av_frame_get_buffer(f, 0);
    if (r < 0) return r;
    memset(f->data[0], 16, f->linesize[0] * f->height);
    memset(f->data[1], 128, f->linesize[1] * f->height / 2);
    return 0;
}

int aji_infer(aji_ctx *c, const aji_frame *in, const aji_frame *out, void *s)
{
    upscale_calls++;
    assert(in->format == AJI_FMT_NV12 && out->format == AJI_FMT_NV12);
    assert(in->width == out->width && in->height == out->height);
    assert(pending_count < 2);
    pending[pending_count].a = *in;
    pending[pending_count].out = *out;
    pending[pending_count].rife = 0;
    pending_count++;
    return AJI_OK;
}

int aji_infer_rife(aji_ctx *c, const aji_frame *a, const aji_frame *b,
                   double t, const aji_frame *out, void *s)
{
    rife_calls++;
    assert(pending_count == 0);
    assert(a->format == AJI_FMT_NV12 && b->format == AJI_FMT_NV12 &&
           out->format == AJI_FMT_NV12);
    /* In mixed mode the first pair is a cut, the second is continuous. */
    if (scene_mode && *(unsigned char *)a->plane[0] == 48)
        return AJI_SCENE;
    pending[0].a = *a;
    pending[0].b = *b;
    pending[0].out = *out;
    pending[0].t = t;
    pending[0].rife = 1;
    pending_count = 1;
    return AJI_OK;
}

int aji_resize(aji_ctx *c, const aji_frame *in, const aji_frame *out, void *s)
{ abort(); }
uint64_t aji_flush(aji_ctx *c, void *s) { return 1; }
int aji_wait(aji_ctx *c, uint64_t t) { complete_pending(); return AJI_OK; }
const char *aji_last_error(aji_ctx *c) { return "unexpected test failure"; }
cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t s) { abort(); }


int main(int argc, char **argv)
{
    if (argc != 5) return 2;
    enc_ctx c = {0};
    c.o.output = argv[3];
    c.o.overwrite = 1;
    c.o.vcodec = "ffv1";
    c.o.pix_fmt = OUT_420P8;
    c.src_w = c.src_h = c.up_w = c.up_h = c.out_w = c.out_h = 64;
    c.src_aji_fmt = AJI_FMT_NV12;
    c.src_sar = (AVRational){1, 1};
    c.color_range = AVCOL_RANGE_MPEG;
    c.color_space = AVCOL_SPC_BT709;
    c.color_pri = AVCOL_PRI_BT709;
    c.color_trc = AVCOL_TRC_BT709;
    c.chroma_loc = AVCHROMA_LOC_LEFT;
    c.rife = 1;
    c.rife_first = !strcmp(argv[1], "before");
    c.rnum = 4;
    c.rden = 1;
    c.out_fps = (AVRational){100, 1};
    scene_mode = strcmp(argv[2], "normal") != 0;
    int count = !strcmp(argv[2], "mixed") ? 3 : 2;
    g_progress = PROG_NONE;
    resolve_pixfmt(&c);
    c.passthrough = atoi(argv[4]);
    if (c.passthrough)
        c.out_aji_fmt = AJI_FMT_P010;  /* RIFE still consumes native NV12. */
    c.ifmt = avformat_alloc_context();
    if (!c.ifmt) return 1;
    AVStream *source = avformat_new_stream(c.ifmt, NULL);
    if (!source) return 1;
    source->time_base = (AVRational){1, 1000};
    source->sample_aspect_ratio = (AVRational){1, 1};
    if (open_output(&c) < 0) return 1;

    const int64_t pts[] = {5000, 5080, 5280};
    const int64_t durations[] = {80, 200, 120};
    const int levels[] = {48, 192, 208};
    for (int i = 0; i < count; i++) {
        AVFrame *f = av_frame_alloc();
        if (!f || av_hwframe_get_buffer(NULL, f, 0) < 0) return 1;
        memset(f->data[0], levels[i], f->linesize[0] * f->height);
        f->best_effort_timestamp = pts[i];
        f->duration = durations[i];
        set_frame_timing(&c, f);
        c.src_frames_seen++;
        if (push_rife(&c, f) < 0) return 1;
        assert(pending_count == 0);
    }
    if (push_rife(&c, NULL) < 0 || drain_encoder(&c, 1) < 0 ||
        av_write_trailer(c.ofmt) < 0) return 1;
    printf("%d %d %lld %lld\n", upscale_calls, rife_calls,
           (long long)c.scene_dupes, (long long)c.out_frames);
    av_frame_free(&c.up_prev);
    av_frame_free(&c.up_cur);
    avcodec_free_context(&c.enc);
    sws_freeContext(c.sws);
    avio_closep(&c.ofmt->pb);
    avformat_free_context(c.ofmt);
    avformat_free_context(c.ifmt);
    av_free(c.smap);
    return 0;
}
'''


class EncoderRifeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for tool in ('cc', 'pkg-config', 'ffmpeg', 'ffprobe'):
            if not shutil.which(tool):
                raise unittest.SkipTest(f'{tool} is required')
        roots = [os.environ.get('CUDA_PATH', ''),
                 os.environ.get('CUDAToolkit_ROOT', ''),
                 '/usr/local/cuda', '/opt/cuda', '/usr']
        cuda_include = next((Path(p) / 'include' for p in roots
                             if p and (Path(p) / 'include/cuda_runtime.h').is_file()), None)
        if cuda_include is None:
            raise unittest.SkipTest('CUDA headers are required; no GPU is needed')
        flags = subprocess.run(['pkg-config', '--cflags', '--libs',
                                'libavformat', 'libavcodec', 'libavutil',
                                'libavfilter', 'libswscale'], capture_output=True, text=True)
        if flags.returncode:
            raise unittest.SkipTest('FFmpeg development libraries are required')
        cls.temporary = tempfile.TemporaryDirectory(prefix='aji-rife-test-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        source = cls.directory / 'rife_test.c'
        source.write_text(HARNESS.replace('ENCODE_SOURCE', str(ROOT / 'src/encode.c')))
        cls.binary = cls.directory / 'rife_test'
        # Link only the used encode paths; mocked inference needs no GPU runtime.
        result = subprocess.run([
            'cc', '-O1', '-ffunction-sections', '-fdata-sections',
            '-Wl,--gc-sections', '-I', str(ROOT / 'include'),
            '-I', str(cuda_include), str(source), '-o', str(cls.binary),
            *shlex.split(flags.stdout)], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def test_scene_reuse_and_pair_reset(self):
        for order, mode, passthrough in product(
                ('before', 'after'), ('scene', 'normal', 'mixed'), (0, 1)):
            with self.subTest(order=order, mode=mode, passthrough=passthrough):
                target = self.directory / f'{order}-{mode}-{passthrough}.mkv'
                result = subprocess.run([
                    str(self.binary), order, mode, str(target), str(passthrough)],
                    capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                expected_rife = {'scene': 1, 'normal': 3, 'mixed': 4}[mode]
                expected_up = ({'scene': 2, 'normal': 5, 'mixed': 6}[mode]
                               if order == 'before' else (3 if mode == 'mixed' else 2))
                if passthrough:
                    expected_up = 0
                frames = 9 if mode == 'mixed' else 5
                self.assertEqual(list(map(int, result.stdout.split())),
                                 [expected_up, expected_rife,
                                  0 if mode == 'normal' else 3, frames])
                probe = subprocess.check_output([
                    'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                    '-show_entries', 'frame=pts_time', '-of', 'json', str(target)],
                    text=True, timeout=15)
                actual_pts = [float(f['pts_time']) for f in json.loads(probe)['frames']]
                expected_pts = [5.0, 5.02, 5.04, 5.06, 5.08]
                if mode == 'mixed':
                    expected_pts += [5.13, 5.18, 5.23, 5.28]
                self.assertEqual(actual_pts, expected_pts)
                raw = subprocess.check_output([
                    'ffmpeg', '-v', 'error', '-i', str(target), '-fps_mode', 'passthrough',
                    '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-'], timeout=15)
                stride = 64 * 64 * 3 // 2
                self.assertEqual(len(raw), frames * stride)
                levels = [48, 84, 120, 156, 192] if mode == 'normal' else [48] * 4 + [192]
                if mode == 'mixed':
                    levels += [196, 200, 204, 208]
                self.assertEqual([raw[i * stride] for i in range(frames)], levels)


if __name__ == '__main__':
    unittest.main()
