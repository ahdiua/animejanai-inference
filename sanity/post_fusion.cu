/* Regression and optional timing for the fused CUDA postprocessor.
 * Build from the project root (select an architecture for your GPU):
 * /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_89 -Iinclude -Isrc \
 *   sanity/post_fusion.cu -o /tmp/aji_post_fusion_test
 * /tmp/aji_post_fusion_test --benchmark
 * compute-sanitizer --tool memcheck --error-exitcode 1 \
 *   /tmp/aji_post_fusion_test --small
 * compute-sanitizer --tool racecheck --error-exitcode 1 \
 *   /tmp/aji_post_fusion_test --small
 *
 * The reference explicitly runs the original three-pass algorithm, retained
 * in kernels.cu for odd widths, with its own full-resolution UV scratch.
 * It never calls aji_run_post, so even-width tests exercise different paths.
 * Compare all output bytes (including padding/guards), and report pixel
 * maximum error and PSNR in native 8/10-bit units. No model is required.
 */
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../src/kernels.cu"

#define CK(call) do { \
    cudaError_t err = (cudaError_t)(call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, \
                cudaGetErrorString(err)); \
        exit(2); \
    } \
} while (0)

struct device_buffer {
    uint8_t *data = nullptr;
    size_t bytes;
    explicit device_buffer(size_t size) : bytes(size) { CK(cudaMalloc(&data, size)); }
    ~device_buffer() { CK(cudaFree(data)); }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
};

static void reference_post(aji_plan *p, const void *rgb, aji_csp csp,
                           void *y, ptrdiff_t ys, void *uv, ptrdiff_t uvs,
                           float *full_uv, float *half_uv, cudaStream_t stream)
{
    const int w = p->w, h = p->h, cw = w / 2, ch = h / 2;
    if (p->format == AJI_FMT_P010) {
        k_post_matrix<uint16_t><<<GRID(w, h, 1), 0, stream>>>(
            (const __half *)rgb, w, h, csp, 64.0f, 1023.0f,
            (uint8_t *)y, ys, full_uv);
    } else {
        k_post_matrix<uint8_t><<<GRID(w, h, 1), 0, stream>>>(
            (const __half *)rgb, w, h, csp, 1.0f, 255.0f,
            (uint8_t *)y, ys, full_uv);
    }
    k_h_f32<<<GRID(cw, h, 2), 0, stream>>>(full_uv, h, p->ph, half_uv);
    if (p->format == AJI_FMT_P010) {
        k_uv_v_store<uint16_t><<<GRID(cw, ch, 1), 0, stream>>>(
            half_uv, cw, p->pv, csp, 64.0f, 1023.0f, (uint8_t *)uv, uvs);
    } else {
        k_uv_v_store<uint8_t><<<GRID(cw, ch, 1), 0, stream>>>(
            half_uv, cw, p->pv, csp, 1.0f, 255.0f, (uint8_t *)uv, uvs);
    }
    CK(cudaGetLastError());
}

struct metrics {
    size_t cases = 0, different_bytes = 0, guard_errors = 0, pixels = 0;
    unsigned max_error = 0;
    double normalized_sse = 0.0;
};

static constexpr size_t guard = 128;
static constexpr uint8_t sentinel = 0xcd;

static void compare_plane(const device_buffer &reference, const device_buffer &actual,
                           int width, int height, size_t pitch, int bpp, metrics &m)
{
    std::vector<uint8_t> a(reference.bytes), b(actual.bytes);
    CK(cudaMemcpy(a.data(), reference.data, a.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(), actual.data, b.size(), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < a.size(); i++) {
        m.different_bytes += a[i] != b[i];
        const bool active = i >= guard && i < guard + pitch * height &&
                            (i - guard) % pitch < (size_t)width * bpp;
        if (!active)
            m.guard_errors += a[i] != sentinel || b[i] != sentinel;
    }
    const double peak = bpp == 1 ? 255.0 : 1023.0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const size_t idx = guard + pitch * y + x * bpp;
            unsigned av = a[idx], bv = b[idx];
            if (bpp == 2) {
                av = (av | (unsigned)a[idx + 1] << 8) >> 6;
                bv = (bv | (unsigned)b[idx + 1] << 8) >> 6;
            }
            const unsigned error = av > bv ? av - bv : bv - av;
            m.max_error = std::max(m.max_error, error);
            m.normalized_sse += (error / peak) * (error / peak);
            m.pixels++;
        }
    }
}

template <typename Run>
static float time_run(Run run, cudaStream_t stream)
{
    cudaEvent_t start, stop;
    CK(cudaEventCreate(&start));
    CK(cudaEventCreate(&stop));
    CK(cudaEventRecord(start, stream));
    const int iterations = 100;
    for (int i = 0; i < iterations; i++)
        run();
    CK(cudaEventRecord(stop, stream));
    CK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, start, stop));
    CK(cudaEventDestroy(start));
    CK(cudaEventDestroy(stop));
    return ms / iterations;
}

static void check_size(int w, int h, bool exhaustive, bool benchmark,
                       cudaStream_t stream, metrics &total)
{
    const size_t plane = (size_t)w * h;
    device_buffer rgb(3 * plane * sizeof(__half));
    device_buffer full_uv(2 * plane * sizeof(float));
    device_buffer half_uv((size_t)2 * (w / 2) * h * sizeof(float));
    std::vector<__half> values(3 * plane);
    uint32_t state = 12345;
    for (size_t i = 0; i < values.size(); i++) {
        state = 1664525u * state + 1013904223u;
        float v = (state & 65535) / 65535.0f * 1.5f - 0.25f;
        // Include saturated/neutral colors and out-of-gamut ringing values.
        switch (i % 19) {
        case 0: v = 0.0f; break;
        case 1: v = 1.0f; break;
        case 2: v = 0.5f; break;
        case 3: v = -0.5f; break;
        case 4: v = 1.5f; break;
        }
        values[i] = __float2half(v);
    }
    CK(cudaMemcpy(rgb.data, values.data(), rgb.bytes, cudaMemcpyHostToDevice));

    for (int format : {AJI_FMT_NV12, AJI_FMT_P010}) {
        const int bpp = format == AJI_FMT_NV12 ? 1 : 2;
        // Independent non-warp-aligned pitches and device-pointer offsets.
        const size_t ys = (w + 37) * bpp, uvs = (2 * (w / 2) + 19) * bpp;
        device_buffer y_ref(2 * guard + ys * h), y_new(y_ref.bytes);
        device_buffer uv_ref(2 * guard + uvs * (h / 2)), uv_new(uv_ref.bytes);
        for (int filter : {AJI_FILTER_SPLINE36, AJI_FILTER_BILINEAR}) {
            for (int siting : {AJI_SITING_LEFT, AJI_SITING_CENTER, AJI_SITING_TOPLEFT}) {
                if (!exhaustive && siting != AJI_SITING_LEFT)
                    continue;
                aji_plan *p = aji_post_plan_create(format, w, h, siting, filter);
                if (!p) {
                    fprintf(stderr, "post plan creation failed\n");
                    exit(2);
                }
                if (!(w & 1) && p->tmp1) {
                    fprintf(stderr, "even-width post allocated obsolete scratch\n");
                    exit(2);
                }
                for (int matrix : {AJI_MATRIX_BT601, AJI_MATRIX_BT709, AJI_MATRIX_BT2020}) {
                    for (int range : {AJI_RANGE_LIMITED, AJI_RANGE_FULL}) {
                        if (!exhaustive && (matrix != AJI_MATRIX_BT709 ||
                                            range != AJI_RANGE_LIMITED))
                            continue;
                        const aji_csp csp = aji_make_csp(format, matrix, range);
                        auto before = [&]() {
                            reference_post(p, rgb.data, csp,
                                y_ref.data + guard, ys, uv_ref.data + guard, uvs,
                                (float *)full_uv.data, (float *)half_uv.data, stream);
                        };
                        auto after = [&]() {
                            CK(aji_run_post(p, rgb.data, &csp,
                                y_new.data + guard, ys, uv_new.data + guard, uvs, stream));
                        };
                        CK(cudaMemsetAsync(y_ref.data, sentinel, y_ref.bytes, stream));
                        CK(cudaMemsetAsync(y_new.data, sentinel, y_new.bytes, stream));
                        CK(cudaMemsetAsync(uv_ref.data, sentinel, uv_ref.bytes, stream));
                        CK(cudaMemsetAsync(uv_new.data, sentinel, uv_new.bytes, stream));
                        before();
                        after();
                        CK(cudaStreamSynchronize(stream));
                        const size_t old_diff = total.different_bytes;
                        const size_t old_guards = total.guard_errors;
                        compare_plane(y_ref, y_new, w, h, ys, bpp, total);
                        compare_plane(uv_ref, uv_new, 2 * (w / 2), h / 2, uvs, bpp, total);
                        total.cases++;
                        if (old_diff != total.different_bytes || old_guards != total.guard_errors) {
                            fprintf(stderr, "FAIL %dx%d format=%d filter=%d siting=%d "
                                    "matrix=%d range=%d different_bytes=%zu guard_errors=%zu\n",
                                    w, h, format, filter, siting, matrix, range,
                                    total.different_bytes - old_diff,
                                    total.guard_errors - old_guards);
                        }
                        if (benchmark && filter == AJI_FILTER_SPLINE36 &&
                            siting == AJI_SITING_LEFT && matrix == AJI_MATRIX_BT709 &&
                            range == AJI_RANGE_LIMITED) {
                            for (int i = 0; i < 10; i++) { before(); after(); }
                            std::vector<float> old_ms, new_ms;
                            for (int i = 0; i < 7; i++) {
                                // Alternate order to reduce warmup/clock bias.
                                if (i & 1) {
                                    new_ms.push_back(time_run(after, stream));
                                    old_ms.push_back(time_run(before, stream));
                                } else {
                                    old_ms.push_back(time_run(before, stream));
                                    new_ms.push_back(time_run(after, stream));
                                }
                            }
                            std::sort(old_ms.begin(), old_ms.end());
                            std::sort(new_ms.begin(), new_ms.end());
                            printf("TIMING %dx%d %s old_ms=%.6f fused_ms=%.6f "
                                   "scratch_saved_bytes=%zu\n", w, h,
                                   bpp == 1 ? "NV12" : "P010", old_ms[3], new_ms[3],
                                   2 * plane * sizeof(float));
                            fflush(stdout);
                        }
                    }
                }
                aji_plan_destroy(p);
            }
        }
    }
}

int main(int argc, char **argv)
{
    bool benchmark = false, small = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--benchmark")) benchmark = true;
        else if (!strcmp(argv[i], "--small")) small = true;
        else {
            fprintf(stderr, "usage: %s [--benchmark] [--small]\n", argv[0]);
            return 2;
        }
    }
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("device=%s\n", prop.name);
    cudaStream_t stream;
    CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    metrics total;
    const int sizes[][2] = {
        {2, 2}, {2, 18}, {130, 2}, {6, 4}, {62, 6}, {64, 8}, {66, 10},
        {126, 14}, {130, 18}, {258, 34}, {65, 9}, {129, 17},
    };
    for (const auto &size : sizes)
        check_size(size[0], size[1], true, false, stream, total);
    if (!small) {
        check_size(1920, 1080, false, benchmark, stream, total);
        check_size(3840, 2160, false, benchmark, stream, total);
    }
    const double psnr = total.normalized_sse ?
        10.0 * log10(total.pixels / total.normalized_sse) : INFINITY;
    printf("%s cases=%zu different_bytes=%zu guard_errors=%zu max_error=%u PSNR=%g dB\n",
           total.different_bytes || total.guard_errors ? "FAIL" : "PASS",
           total.cases, total.different_bytes, total.guard_errors, total.max_error, psnr);
    CK(cudaStreamDestroy(stream));
    return total.different_bytes || total.guard_errors ? 1 : 0;
}
