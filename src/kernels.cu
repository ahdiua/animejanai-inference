/*
 * Pre/post-processing and resize CUDA kernels for the aji shim.
 *
 * pre:    NV12/P010 -> fp16 NCHW RGB [0,1], parametrized matrix/range/siting
 * post:   fp16 NCHW RGB [0,1] -> NV12/P010
 * resize: Catmull-Rom on fp16 NCHW RGB (chain resize steps)
 *
 * Remaining quality deltas vs the zimg path (quantified by the parity
 * harness): chroma upsampling is bilinear (zimg: spline36 via VS resize),
 * chroma downsampling is a 2x2 box, resize is Catmull-Rom not Spline36.
 */

#include <cuda_fp16.h>
#include <stdint.h>

#include "kernels.h"

__device__ __forceinline__ float3 ycbcr_to_rgb(float y, float u, float v,
                                               float kr, float kb)
{
    const float kg = 1.0f - kr - kb;
    float r = y + 2.0f * (1.0f - kr) * v;
    float b = y + 2.0f * (1.0f - kb) * u;
    float g = y - (2.0f * kb * (1.0f - kb) * u + 2.0f * kr * (1.0f - kr) * v) / kg;
    return make_float3(__saturatef(r), __saturatef(g), __saturatef(b));
}

template <typename T>
__global__ void k_pre(const uint8_t *y_plane, ptrdiff_t y_stride,
                      const uint8_t *uv_plane, ptrdiff_t uv_stride,
                      int w, int h, aji_csp csp, __half *dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;

    const T *yrow = (const T *)(y_plane + (size_t)y * y_stride);
    const float Y = ((float)yrow[x] - csp.yoff) / csp.yscale;

    // bilinear chroma upsample at the configured siting
    const int cw = w >> 1, ch = h >> 1;
    float cxf = x * 0.5f + csp.cox;
    float cyf = y * 0.5f + csp.coy;
    int cx0 = (int)floorf(cxf), cy0 = (int)floorf(cyf);
    const float fx = cxf - cx0, fy = cyf - cy0;
    int cx1 = min(cx0 + 1, cw - 1), cy1 = min(cy0 + 1, ch - 1);
    cx0 = max(cx0, 0); cy0 = max(cy0, 0);

    const T *uv0 = (const T *)(uv_plane + (size_t)cy0 * uv_stride);
    const T *uv1 = (const T *)(uv_plane + (size_t)cy1 * uv_stride);
    const float w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy);
    const float w01 = (1 - fx) * fy,       w11 = fx * fy;

    float U = w00 * uv0[2 * cx0] + w10 * uv0[2 * cx1] +
              w01 * uv1[2 * cx0] + w11 * uv1[2 * cx1];
    float V = w00 * uv0[2 * cx0 + 1] + w10 * uv0[2 * cx1 + 1] +
              w01 * uv1[2 * cx0 + 1] + w11 * uv1[2 * cx1 + 1];
    U = (U - csp.coff) / csp.cscale;
    V = (V - csp.coff) / csp.cscale;

    const float3 rgb = ycbcr_to_rgb(Y, U, V, csp.kr, csp.kb);

    const size_t plane = (size_t)w * h, idx = (size_t)y * w + x;
    dst[idx]             = __float2half(rgb.x);
    dst[plane + idx]     = __float2half(rgb.y);
    dst[2 * plane + idx] = __float2half(rgb.z);
}

template <typename T>
__global__ void k_post_luma(const __half *src, int w, int h, aji_csp csp,
                            uint8_t *y_plane, ptrdiff_t y_stride)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;

    const size_t plane = (size_t)w * h, idx = (size_t)y * w + x;
    const float r = __half2float(src[idx]);
    const float g = __half2float(src[plane + idx]);
    const float b = __half2float(src[2 * plane + idx]);

    const float Y = csp.kr * r + (1.0f - csp.kr - csp.kb) * g + csp.kb * b;
    const float maxv = (float)(T)~(T)0;
    float val = fminf(fmaxf(rintf(__saturatef(Y) * csp.yscale + csp.yoff), 0.0f), maxv);

    T *yrow = (T *)(y_plane + (size_t)y * y_stride);
    yrow[x] = (T)val;
}

template <typename T>
__global__ void k_post_chroma(const __half *src, int w, int h, aji_csp csp,
                              uint8_t *uv_plane, ptrdiff_t uv_stride)
{
    const int cx = blockIdx.x * blockDim.x + threadIdx.x;
    const int cy = blockIdx.y * blockDim.y + threadIdx.y;
    const int cw = w >> 1, ch = h >> 1;
    if (cx >= cw || cy >= ch)
        return;

    // 2x2 box average of RGB, then derive CbCr from the averaged color.
    const size_t plane = (size_t)w * h;
    float r = 0, g = 0, b = 0;
    #pragma unroll
    for (int dy = 0; dy < 2; dy++) {
        #pragma unroll
        for (int dx = 0; dx < 2; dx++) {
            const size_t idx = (size_t)(2 * cy + dy) * w + (2 * cx + dx);
            r += __half2float(src[idx]);
            g += __half2float(src[plane + idx]);
            b += __half2float(src[2 * plane + idx]);
        }
    }
    r *= 0.25f; g *= 0.25f; b *= 0.25f;

    const float Y = csp.kr * r + (1.0f - csp.kr - csp.kb) * g + csp.kb * b;
    const float U = (b - Y) / (2.0f * (1.0f - csp.kb));
    const float V = (r - Y) / (2.0f * (1.0f - csp.kr));

    const float maxv = (float)(T)~(T)0;
    float uval = fminf(fmaxf(rintf(U * csp.cscale + csp.coff), 0.0f), maxv);
    float vval = fminf(fmaxf(rintf(V * csp.cscale + csp.coff), 0.0f), maxv);

    T *uvrow = (T *)(uv_plane + (size_t)cy * uv_stride);
    uvrow[2 * cx]     = (T)uval;
    uvrow[2 * cx + 1] = (T)vval;
}

__device__ __forceinline__ float catmull_rom(float t)
{
    t = fabsf(t);
    if (t <= 1.0f)
        return (1.5f * t - 2.5f) * t * t + 1.0f;
    if (t < 2.0f)
        return ((-0.5f * t + 2.5f) * t - 4.0f) * t + 2.0f;
    return 0.0f;
}

__global__ void k_resize(const __half *src, int sw, int sh,
                         __half *dst, int dw, int dh)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int p = blockIdx.z;  // plane 0..2
    if (x >= dw || y >= dh)
        return;

    const float sx = (x + 0.5f) * sw / dw - 0.5f;
    const float sy = (y + 0.5f) * sh / dh - 0.5f;
    const int ix = (int)floorf(sx), iy = (int)floorf(sy);
    const float fx = sx - ix, fy = sy - iy;

    float wx[4], wy[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        wx[i] = catmull_rom(fx - (i - 1));
        wy[i] = catmull_rom(fy - (i - 1));
    }

    const __half *plane = src + (size_t)p * sw * sh;
    float acc = 0.0f, wsum = 0.0f;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const int yy = min(max(iy + j - 1, 0), sh - 1);
        const __half *row = plane + (size_t)yy * sw;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int xx = min(max(ix + i - 1, 0), sw - 1);
            const float wgt = wx[i] * wy[j];
            acc += wgt * __half2float(row[xx]);
            wsum += wgt;
        }
    }
    dst[(size_t)p * dw * dh + (size_t)y * dw + x] =
        __float2half(__saturatef(acc / wsum));
}

#define BLOCK 16
#define GRID2(w, h) dim3(((w) + BLOCK - 1) / BLOCK, ((h) + BLOCK - 1) / BLOCK)

extern "C" int aji_launch_pre(int format,
                              const void *y_plane, ptrdiff_t y_stride,
                              const void *uv_plane, ptrdiff_t uv_stride,
                              int w, int h, const aji_csp *csp,
                              void *dst_f16, void *stream)
{
    dim3 block(BLOCK, BLOCK);
    cudaStream_t s = (cudaStream_t)stream;
    if (format == 1) {
        k_pre<uint8_t><<<GRID2(w, h), block, 0, s>>>(
            (const uint8_t *)y_plane, y_stride, (const uint8_t *)uv_plane,
            uv_stride, w, h, *csp, (__half *)dst_f16);
    } else {
        k_pre<uint16_t><<<GRID2(w, h), block, 0, s>>>(
            (const uint8_t *)y_plane, y_stride, (const uint8_t *)uv_plane,
            uv_stride, w, h, *csp, (__half *)dst_f16);
    }
    return (int)cudaGetLastError();
}

extern "C" int aji_launch_post(int format, const void *src_f16, int w, int h,
                               const aji_csp *csp,
                               void *y_plane, ptrdiff_t y_stride,
                               void *uv_plane, ptrdiff_t uv_stride,
                               void *stream)
{
    dim3 block(BLOCK, BLOCK);
    cudaStream_t s = (cudaStream_t)stream;
    if (format == 1) {
        k_post_luma<uint8_t><<<GRID2(w, h), block, 0, s>>>(
            (const __half *)src_f16, w, h, *csp, (uint8_t *)y_plane, y_stride);
        k_post_chroma<uint8_t><<<GRID2(w / 2, h / 2), block, 0, s>>>(
            (const __half *)src_f16, w, h, *csp, (uint8_t *)uv_plane, uv_stride);
    } else {
        k_post_luma<uint16_t><<<GRID2(w, h), block, 0, s>>>(
            (const __half *)src_f16, w, h, *csp, (uint8_t *)y_plane, y_stride);
        k_post_chroma<uint16_t><<<GRID2(w / 2, h / 2), block, 0, s>>>(
            (const __half *)src_f16, w, h, *csp, (uint8_t *)uv_plane, uv_stride);
    }
    return (int)cudaGetLastError();
}

extern "C" int aji_launch_resize(const void *src_f16, int sw, int sh,
                                 void *dst_f16, int dw, int dh, void *stream)
{
    dim3 block(BLOCK, BLOCK);
    dim3 grid(((dw) + BLOCK - 1) / BLOCK, ((dh) + BLOCK - 1) / BLOCK, 3);
    k_resize<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __half *)src_f16, sw, sh, (__half *)dst_f16, dw, dh);
    return (int)cudaGetLastError();
}
