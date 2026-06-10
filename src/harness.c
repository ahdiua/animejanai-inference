/*
 * aji_harness — CLI test harness for the aji shim. No player involved.
 *
 * Reads raw NV12/P010 frames from a file, runs them through aji_infer on
 * the GPU, optionally writes the upscaled raw frames, and reports
 * per-frame device timing (CUDA events around the full pre+infer+post
 * enqueue, measured on the stream).
 *
 * Usage:
 *   aji_harness --engine E.engine --input in.raw --width W --height H
 *               [--format nv12|p010] [--frames N] [--output out.raw]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "aji.h"

#define CK(x) do { \
    cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) { \
        fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #x, \
                cudaGetErrorString(e_)); \
        return 1; \
    } \
} while (0)

static void log_cb(void *opaque, int level, const char *msg)
{
    (void)opaque;
    if (level <= 2) // TRT INTERNAL_ERROR/ERROR/WARNING
        fprintf(stderr, "[trt:%d] %s\n", level, msg);
}

int main(int argc, char **argv)
{
    const char *engine = NULL, *input = NULL, *output = NULL;
    const char *format = "nv12";
    int w = 0, h = 0, max_frames = 1 << 30;

    for (int i = 1; i < argc - 1; i++) {
        if (!strcmp(argv[i], "--engine")) engine = argv[++i];
        else if (!strcmp(argv[i], "--input")) input = argv[++i];
        else if (!strcmp(argv[i], "--output")) output = argv[++i];
        else if (!strcmp(argv[i], "--format")) format = argv[++i];
        else if (!strcmp(argv[i], "--width")) w = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--height")) h = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--frames")) max_frames = atoi(argv[++i]);
    }
    if (!engine || !input || w < 2 || h < 2) {
        fprintf(stderr, "usage: %s --engine E --input raw --width W --height H"
                        " [--format nv12|p010] [--frames N] [--output raw]\n",
                argv[0]);
        return 2;
    }

    const int fmt = strcmp(format, "p010") ? AJI_FMT_NV12 : AJI_FMT_P010;
    const int bpp = fmt == AJI_FMT_P010 ? 2 : 1;
    const size_t y_sz = (size_t)w * h * bpp;
    const size_t uv_sz = y_sz / 2;
    const size_t frame_sz = y_sz + uv_sz;

    aji_create_params params = {
        .api_version = AJI_API_VERSION,
        .cuda_context = NULL, // primary ctx; harness uses the same via runtime
        .engine_path = engine,
        .max_width = w,
        .max_height = h,
        .log = log_cb,
    };
    aji_ctx *aji = aji_create(&params);
    if (!aji) {
        fprintf(stderr, "aji_create failed\n");
        return 1;
    }
    const int scale = aji_scale_factor(aji);
    const int ow = w * scale, oh = h * scale;
    const size_t oy_sz = (size_t)ow * oh * bpp, ouv_sz = oy_sz / 2;
    printf("engine: %s, scale %dx, %dx%d %s -> %dx%d\n", engine, scale, w, h,
           format, ow, oh);

    FILE *fin = fopen(input, "rb");
    if (!fin) { perror(input); return 1; }
    FILE *fout = output ? fopen(output, "wb") : NULL;
    if (output && !fout) { perror(output); return 1; }

    void *h_in, *h_out;
    CK(cudaMallocHost(&h_in, frame_sz));
    CK(cudaMallocHost(&h_out, oy_sz + ouv_sz));

    aji_frame in = {.width = w, .height = h, .format = fmt,
                    .stride = {(ptrdiff_t)(w * bpp), (ptrdiff_t)(w * bpp)}};
    aji_frame out = {.width = ow, .height = oh, .format = fmt,
                     .stride = {(ptrdiff_t)(ow * bpp), (ptrdiff_t)(ow * bpp)}};
    CK(cudaMalloc(&in.plane[0], y_sz));
    CK(cudaMalloc(&in.plane[1], uv_sz));
    CK(cudaMalloc(&out.plane[0], oy_sz));
    CK(cudaMalloc(&out.plane[1], ouv_sz));

    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));
    cudaEvent_t ev0, ev1;
    CK(cudaEventCreate(&ev0));
    CK(cudaEventCreate(&ev1));

    int n = 0, warmup = 3;
    double total_ms = 0;
    int timed = 0;

    while (n < max_frames && fread(h_in, 1, frame_sz, fin) == frame_sz) {
        CK(cudaMemcpyAsync(in.plane[0], h_in, y_sz, cudaMemcpyHostToDevice,
                           stream));
        CK(cudaMemcpyAsync(in.plane[1], (char *)h_in + y_sz, uv_sz,
                           cudaMemcpyHostToDevice, stream));

        CK(cudaEventRecord(ev0, stream));
        int ret = aji_infer(aji, &in, &out, stream);
        if (ret != AJI_OK) {
            fprintf(stderr, "aji_infer: %d: %s\n", ret, aji_last_error(aji));
            return 1;
        }
        CK(cudaEventRecord(ev1, stream));

        if (fout) {
            CK(cudaMemcpyAsync(h_out, out.plane[0], oy_sz,
                               cudaMemcpyDeviceToHost, stream));
            CK(cudaMemcpyAsync((char *)h_out + oy_sz, out.plane[1], ouv_sz,
                               cudaMemcpyDeviceToHost, stream));
        }
        CK(cudaStreamSynchronize(stream));

        float ms = 0;
        CK(cudaEventElapsedTime(&ms, ev0, ev1));
        if (n >= warmup) {
            total_ms += ms;
            timed++;
        }
        if (fout && fwrite(h_out, 1, oy_sz + ouv_sz, fout) != oy_sz + ouv_sz) {
            perror("fwrite");
            return 1;
        }
        n++;
    }

    printf("frames: %d, device pre+infer+post: %.3f ms/frame avg "
           "(%d timed frames)\n", n, timed ? total_ms / timed : 0.0, timed);

    if (fout) fclose(fout);
    fclose(fin);
    aji_destroy(&aji);
    return 0;
}
