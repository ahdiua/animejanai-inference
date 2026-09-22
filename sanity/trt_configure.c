/* TensorRT configuration regressions; requires CUDA but no real models.
 * Build from the project root:
 * cc -std=c11 -O2 -Iinclude -I/usr/local/cuda/include sanity/trt_configure.c \
 *   -Lbuild -Wl,-rpath,"$PWD/build" -laji_trt \
 *   -L/usr/local/cuda/lib64 -lcudart -o /tmp/aji_trt_configure_test
 * Run both with and without AJI_NO_GRAPH=1:
 * compute-sanitizer --tool memcheck --error-exitcode 99 /tmp/aji_trt_configure_test
 * /bin/false substitutes for trtexec, so no ONNX files are parsed or built.
 */
#define _POSIX_C_SOURCE 200809L
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <cuda_runtime_api.h>
#include "aji.h"

static void log_error(void *opaque, int level, const char *msg)
{
    (void)opaque;
    if (level <= 2)
        fprintf(stderr, "%s\n", msg);
}

static int infer_source(aji_ctx *ctx, int ow, int oh)
{
    aji_frame in = {.width = 64, .height = 64, .format = AJI_FMT_NV12,
                    .matrix = AJI_MATRIX_BT709, .range = AJI_RANGE_LIMITED,
                    .siting = AJI_SITING_LEFT, .stride = {64, 64}};
    aji_frame out = in;
    out.width = ow;
    out.height = oh;
    out.stride[0] = out.stride[1] = ow;
    cudaStream_t stream = NULL;
    int ok = 0;
    if (cudaStreamCreate(&stream) != cudaSuccess ||
        cudaMalloc(&in.plane[0], 64 * 64 * 3 / 2) != cudaSuccess ||
        cudaMalloc(&out.plane[0], (size_t)ow * oh * 3 / 2) != cudaSuccess)
        goto done;
    in.plane[1] = (char *)in.plane[0] + 64 * 64;
    out.plane[1] = (char *)out.plane[0] + ow * oh;
    if (cudaMemset(in.plane[0], 128, 64 * 64 * 3 / 2) != cudaSuccess)
        goto done;
    // Exercise graph capture and replay, or the plain path with AJI_NO_GRAPH.
    for (int i = 0; i < 3; i++) {
        if (aji_infer(ctx, &in, &out, stream) != AJI_OK ||
            cudaStreamSynchronize(stream) != cudaSuccess)
            goto done;
    }
    unsigned char *pixels = malloc((size_t)ow * oh * 3 / 2);
    if (!pixels)
        goto done;
    ok = cudaMemcpy(pixels, out.plane[0], (size_t)ow * oh * 3 / 2,
                    cudaMemcpyDeviceToHost) == cudaSuccess;
    for (size_t i = 0; ok && i < (size_t)ow * oh * 3 / 2; i++)
        ok = abs((int)pixels[i] - 128) <= 1;
    free(pixels);
done:
    cudaFree(in.plane[0]);
    cudaFree(out.plane[0]);
    if (stream)
        cudaStreamDestroy(stream);
    if (!ok)
        fprintf(stderr, "source inference failed: %s\n", aji_last_error(ctx));
    return ok;
}

static int check_case(const char *root, int resize, int rife_factor)
{
    char conf[512];
    snprintf(conf, sizeof(conf), "%s/test.conf", root);
    FILE *f = fopen(conf, "w");
    if (!f)
        return 0;
    fprintf(f, "[global]\nconfig_version=2\n"
               "trt_engine_settings=--maxShapes=input:1x3x64x64\n"
               "[slot_1]\nchain_1_model_1_name=\n"
               "chain_1_model_1_resize_factor_before_upscale=%d\n"
               "chain_1_rife=%s\nchain_1_rife_model=414\n"
               "chain_1_rife_factor_numerator=%d\n"
               "chain_1_rife_before_upscale=yes\n",
            resize, rife_factor ? "yes" : "no", rife_factor);
    fclose(f);

    aji_create_params params = {.api_version = AJI_API_VERSION,
                               .conf_path = conf, .model_dir = root,
                               .trtexec = "/bin/false", .slot = 1,
                               .rife_model_dir = root, .async_build = 1,
                               .log = log_error};
    aji_ctx *ctx = aji_create(&params);
    if (!ctx) {
        fprintf(stderr, "aji_create failed\n");
        return 0;
    }
    int ow = 0, oh = 0, ok = 0;
    const int expected = resize == 200 ? 64 : 32;
    if (aji_configure(ctx, 64, 64, 24, &ow, &oh) != 1 ||
        ow != expected || oh != expected ||
        aji_rife_before_upscale(ctx) != 0 ||
        aji_pre_resize(ctx, NULL, NULL) != 0 ||
        aji_rife_factor(ctx, NULL, NULL) != 0 ||
        !infer_source(ctx, ow, oh))
        goto done;
    if (rife_factor == 2) {
        int completed = 0;
        const struct timespec pause = {.tv_nsec = 1000000};
        for (int i = 0; i < 5000 && !completed; i++) {
            completed = aji_poll(ctx);
            if (!completed)
                nanosleep(&pause, NULL);
        }
        if (!completed ||
            aji_configure(ctx, 64, 64, 24, &ow, &oh) != 1 ||
            ow != expected || oh != expected ||
            aji_pre_resize(ctx, NULL, NULL) != 0 ||
            !infer_source(ctx, ow, oh))
            goto done;
    } else if (rife_factor == 1 && aji_poll(ctx)) {
        fprintf(stderr, "factor 1 unexpectedly started an engine build\n");
        goto done;
    }
    ok = 1;
done:
    if (!ok)
        fprintf(stderr, "resize=%d rife=%d failed: %s\n%s\n", resize,
                rife_factor, aji_last_error(ctx), aji_current_log(ctx));
    aji_destroy(&ctx);
    return ok;
}

int main(void)
{
    char root[] = "/tmp/aji-trt-configure-XXXXXX";
    if (!mkdtemp(root))
        return 1;
    char path[512];
    snprintf(path, sizeof(path), "%s/rife_v4.14.onnx", root);
    FILE *f = fopen(path, "w");
    if (!f)
        return 1;
    fputs("not parsed: builder is /bin/false", f);
    fclose(f);
    int ok = check_case(root, 200, 0) && check_case(root, 50, 2) &&
             check_case(root, 50, 1);
    DIR *dir = opendir(root);
    struct dirent *entry;
    while (dir && (entry = readdir(dir))) {
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, ".."))
            continue;
        snprintf(path, sizeof(path), "%s/%s", root, entry->d_name);
        unlink(path);
    }
    if (dir)
        closedir(dir);
    rmdir(root);
    if (ok)
        puts("PASS: intermediate resize, pending/failed RIFE, disabled RIFE");
    return ok ? 0 : 1;
}
