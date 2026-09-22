#!/usr/bin/env python3
"""Exercise real encoder mux, SAR and timestamp paths without a CUDA GPU.

Requires a C compiler, CUDA headers, FFmpeg development libraries, ffmpeg and
ffprobe. Run with: python3 sanity/test_encode_metadata.py
"""
import json
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

/* These CPU paths must never submit GPU work. Abort if a regression does. */
int aji_infer(aji_ctx *c, const aji_frame *in, const aji_frame *out, void *s)
{ abort(); }
int aji_infer_rife(aji_ctx *c, const aji_frame *a, const aji_frame *b,
                   double t, const aji_frame *out, void *s)
{ abort(); }
int aji_resize(aji_ctx *c, const aji_frame *in, const aji_frame *out, void *s)
{ abort(); }
/* Passthrough RIFE may wait for an already-complete inference stream. */
uint64_t aji_flush(aji_ctx *c, void *s) { return 1; }
int aji_wait(aji_ctx *c, uint64_t t) { return AJI_OK; }
const char *aji_last_error(aji_ctx *c) { abort(); }

cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t s) { abort(); }

static int synthetic_rife(enc_ctx *c)
{
    c->rife = 1;
    c->rnum = 4;
    c->rden = 1;
    c->out_fps = (AVRational){100, 1};
    const int64_t pts[] = {5000, 5080, 5280};
    const int64_t durations[] = {80, 200, 120};
    AVRational tb = c->ifmt->streams[c->vstream]->time_base;
    for (int i = 0; i < 3; i++) {
        AVFrame *f = av_frame_alloc();
        if (!f) return -1;
        f->format = AV_PIX_FMT_YUV420P;
        f->width = c->src_w;
        f->height = c->src_h;
        if (av_frame_get_buffer(f, 0) < 0) return -1;
        memset(f->data[0], 16, f->linesize[0] * f->height);
        memset(f->data[1], 128, f->linesize[1] * f->height / 2);
        memset(f->data[2], 128, f->linesize[2] * f->height / 2);
        f->best_effort_timestamp = av_rescale_q(pts[i], (AVRational){1, 1000}, tb);
        f->duration = av_rescale_q(durations[i], (AVRational){1, 1000}, tb);
        set_frame_timing(c, f);
        c->src_frames_seen++;
        if (i) {
            c->up_cur = f;
            int64_t previous_pts = c->up_prev->pts;
            for (int phase = 0; phase < 4; phase++)
                if (emit_rife_phase(c, c->up_prev, phase) < 0) return -1;
            if (c->up_prev->pts != previous_pts) return -1;
            av_frame_free(&c->up_prev);
        }
        c->up_prev = f;
        c->up_cur = NULL;
        c->have_prev = 1;
    }
    /* Exercise the real final-frame branch as well as the shared phase code. */
    if (push_rife(c, NULL) < 0 || drain_encoder(c, 1) < 0) return -1;
    av_frame_free(&c->up_prev);
    return av_write_trailer(c->ofmt);
}

int main(int argc, char **argv)
{
    if (argc < 4) return 2;
    enc_ctx c = {0};
    c.o.input = argv[2];
    c.o.output = argv[3];
    c.o.overwrite = 1;
    c.o.no_subs = argc > 4;
    if (open_input(&c) < 0) return 1;
    AVStream *in = c.ifmt->streams[c.vstream];
    c.src_sar = av_guess_sample_aspect_ratio(c.ifmt, in, NULL);
    c.out_tb = in->time_base;

    if (!strcmp(argv[1], "transcode") || !strncmp(argv[1], "rife", 4)) {
        c.o.vcodec = "ffv1";
        c.o.decoder = DEC_CPU;
        c.o.pix_fmt = OUT_420P8;
        c.passthrough = 1;
        c.ring_depth = 4;
        c.up_w = c.out_w = c.src_w;
        c.up_h = c.out_h = c.src_h;
        g_progress = PROG_NONE;
        resolve_pixfmt(&c);
        if (open_output(&c) < 0) return 1;
        if (!strncmp(argv[1], "rife", 4)) {
            c.rife_first = !strcmp(argv[1], "rife-before");
            if (synthetic_rife(&c) < 0) return 1;
        } else if (open_decoder(&c) < 0 || run(&c) < 0) {
            return 1;
        }
        avio_closep(&c.ofmt->pb);
        avcodec_free_context(&c.dec);
        sws_freeContext(c.sws);
        av_free(c.aux_buf);
    } else if (!strcmp(argv[1], "sar")) {
        c.o.vcodec = "ffv1";
        c.o.pix_fmt = OUT_420P8;
        c.out_pixfmt = AV_PIX_FMT_YUV420P;
        /* An intentionally rounded resize must preserve display aspect. */
        AVFrame f = {.width = 130, .height = 96, .format = c.out_pixfmt};
        if (open_encoder(&c, &f) < 0) return 1;
        printf("%d/%d\n", c.enc->sample_aspect_ratio.num,
                           c.enc->sample_aspect_ratio.den);
    } else {
        if (open_output(&c) < 0) return 1;
        c.enc = avcodec_alloc_context3(NULL);
        if (!c.enc || avcodec_parameters_to_context(c.enc, in->codecpar) < 0)
            return 1;
        c.enc->time_base = in->time_base;
        c.enc->sample_aspect_ratio = c.src_sar;
        if (finalize_output(&c) < 0) return 1;
        AVPacket *p = av_packet_alloc();
        if (!p) return 1;
        int r;
        while ((r = av_read_frame(c.ifmt, p)) >= 0) {
            int from = p->stream_index;
            int to = from == c.vstream ? c.vout : c.smap[from];
            if (to >= 0) {
                p->stream_index = to;
                av_packet_rescale_ts(p, c.ifmt->streams[from]->time_base,
                                     c.ofmt->streams[to]->time_base);
                if (av_interleaved_write_frame(c.ofmt, p) < 0) return 1;
            }
            av_packet_unref(p);
        }
        if (r != AVERROR_EOF) return 1;
        long long before = avio_tell(c.ofmt->pb);
        if (av_write_trailer(c.ofmt) < 0) return 1;
        printf("%lld %lld\n", before, (long long)avio_tell(c.ofmt->pb));
        av_packet_free(&p);
        avio_closep(&c.ofmt->pb);
    }
    avcodec_free_context(&c.enc);
    avformat_free_context(c.ofmt);
    avformat_close_input(&c.ifmt);
    av_free(c.smap);
    return 0;
}
'''


class EncoderMetadataTests(unittest.TestCase):
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
        cls.temporary = tempfile.TemporaryDirectory(prefix='aji-metadata-test-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        source = cls.directory / 'mux_test.c'
        source.write_text(HARNESS.replace('ENCODE_SOURCE', str(ROOT / 'src/encode.c')))
        cls.binary = cls.directory / 'mux_test'
        # Dead-function elimination keeps this CPU-only harness independent
        # of libaji and the CUDA runtime while calling the real setup code.
        subprocess.run(['cc', '-O1', '-ffunction-sections', '-fdata-sections',
                        '-Wl,--gc-sections', '-I', str(ROOT / 'include'),
                        '-I', str(cuda_include), str(source), '-o', str(cls.binary),
                        *shlex.split(flags.stdout)], check=True, capture_output=True)
        subtitle = cls.directory / 'sparse.srt'
        subtitle.write_text('1\n00:00:00,000 --> 00:00:00,100\nSparse subtitle\n')
        attachment = cls.directory / 'fixture.ttf'
        attachment.write_bytes(b'Attachment payload preservation fixture')
        cls.source = cls.directory / 'source.mkv'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                        'testsrc2=size=128x96:rate=24:duration=30',
                        '-i', str(subtitle), '-map', '0:v', '-map', '1:s',
                        '-vf', 'setsar=4/3', '-c:v', 'mpeg4', '-q:v', '2',
                        '-c:s', 'srt', '-metadata:s:v:0', 'title=Source video',
                        '-metadata:s:s:0', 'language=jpn', '-disposition:s:0', 'forced',
                        '-attach', str(attachment), '-metadata:s:t:0',
                        'mimetype=application/x-truetype-font', str(cls.source)],
                       check=True, capture_output=True)
        cls.offset_source = cls.directory / 'offset.mkv'
        cls.vfr_source = cls.directory / 'vfr.mkv'
        for output, filters in [(cls.offset_source, []),
                                (cls.vfr_source,
                                 ['-vf', 'select=eq(n\\,0)+eq(n\\,2)+eq(n\\,7)',
                                  '-fps_mode', 'vfr'])]:
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                            'testsrc2=size=64x48:rate=25:duration=0.4',
                            '-f', 'lavfi', '-i', 'sine=duration=0.4',
                            *filters, '-c:v', 'ffv1', '-c:a', 'pcm_s16le',
                            '-output_ts_offset', '5', str(output)],
                           check=True, capture_output=True)

    def mux(self, suffix, no_subs=False):
        output = self.directory / ('output' + suffix)
        result = subprocess.run([str(self.binary), 'mux', str(self.source), str(output),
                                 *(['no-subs'] if no_subs else [])],
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        probe = subprocess.run(['ffprobe', '-v', 'error', '-show_streams',
                                '-show_data', '-of', 'json', str(output)],
                               check=True, text=True, capture_output=True)
        return json.loads(probe.stdout)['streams'], result

    def test_matroska_preserves_fonts_metadata_dispositions_and_sar(self):
        streams, _ = self.mux('.mkv')
        video = next(s for s in streams if s['codec_type'] == 'video')
        self.assertEqual(video['sample_aspect_ratio'], '4:3')
        self.assertEqual(video['tags']['title'], 'Source video')
        subtitle = next(s for s in streams if s['codec_type'] == 'subtitle')
        self.assertEqual(subtitle['tags']['language'], 'jpn')
        self.assertEqual(subtitle['disposition']['forced'], 1)
        attachment = next(s for s in streams if s['codec_type'] == 'attachment')
        self.assertEqual(attachment['tags']['filename'], 'fixture.ttf')
        self.assertEqual(attachment['tags']['mimetype'], 'application/x-truetype-font')
        source = json.loads(subprocess.check_output(
            ['ffprobe', '-v', 'error', '-select_streams', 't', '-show_streams',
             '-show_data', '-of', 'json', str(self.source)], text=True))['streams'][0]
        self.assertEqual(attachment['extradata'], source['extradata'])

    def test_sparse_subtitle_does_not_buffer_entire_movie(self):
        _, result = self.mux('.mkv')
        before, after = map(int, result.stdout.split())
        self.assertGreater(before, after // 2)

    def test_mp4_omits_unsupported_attachments_with_warning(self):
        streams, result = self.mux('.mp4', no_subs=True)
        self.assertFalse(any(s['codec_type'] == 'attachment' for s in streams))
        self.assertIn('skipping attachment stream', result.stderr)

    def test_encoder_sar_preserves_display_aspect_after_rounded_resize(self):
        result = subprocess.run([str(self.binary), 'sar', str(self.source), 'unused'],
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), '256/195')

    def transcode(self, source, mode='transcode'):
        output = self.directory / (source.stem + '-' + mode + '.mkv')
        result = subprocess.run([str(self.binary), mode, str(source), str(output)],
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return output

    def packets(self, source, stream):
        return json.loads(subprocess.check_output(
            ['ffprobe', '-v', 'error', '-select_streams', stream,
             '-show_packets', '-show_entries', 'packet=pts_time,duration_time',
             '-of', 'json', str(source)], text=True))['packets']

    def test_nonzero_video_audio_start_remains_aligned(self):
        output = self.transcode(self.offset_source)
        for stream in ('v:0', 'a:0'):
            before = self.packets(self.offset_source, stream)
            after = self.packets(output, stream)
            self.assertEqual(after[0]['pts_time'], '5.000000')
            self.assertEqual([p['pts_time'] for p in after],
                             [p['pts_time'] for p in before])

    def test_vfr_video_pts_preserved(self):
        output = self.transcode(self.vfr_source)
        before = self.packets(self.vfr_source, 'v:0')
        after = self.packets(output, 'v:0')
        expected = ['5.000000', '5.080000', '5.280000']
        self.assertEqual([p['pts_time'] for p in before], expected)
        self.assertEqual([p['pts_time'] for p in after], expected)

    def test_rife_phases_follow_actual_interval_and_keep_final_duration(self):
        for mode in ('rife', 'rife-before'):
            with self.subTest(mode=mode):
                output = self.transcode(self.offset_source, mode)
                packets = self.packets(output, 'v:0')
                self.assertEqual([p['pts_time'] for p in packets],
                                 ['5.000000', '5.020000', '5.040000', '5.060000',
                                  '5.080000', '5.130000', '5.180000', '5.230000',
                                  '5.280000'])
                self.assertEqual([p['duration_time'] for p in packets],
                                 ['0.020000'] * 4 + ['0.050000'] * 4 + ['0.120000'])


if __name__ == '__main__':
    unittest.main()
