/*
 * Pre/post-processing CUDA kernels for the aji shim.
 *
 * pre:  NV12/P010 (BT.709 limited, MPEG chroma siting) -> fp16 NCHW RGB [0,1]
 * post: fp16 NCHW RGB [0,1] -> NV12/P010 (BT.709 limited)
 *
 * Spike notes: BT.709 limited is hardcoded (matrix/range/siting become
 * per-frame parameters in phase 1); chroma upsampling is bilinear with
 * left-sited horizontal / centered vertical taps; chroma downsampling is a
 * 2x2 box filter. The parity harness quantifies the difference vs zimg.
 */

#include <cuda_fp16.h>
#include <stdint.h>

#include "kernels.h"

__device__ __forceinline__ float saturate01(float v)
{
    return __saturatef(v);
}

// BT.709 limited-range YCbCr -> RGB (inputs already normalized: y in [0,1],
// u/v in [-0.5, 0.5]).
__device__ __forceinline__ float3 ycbcr709_to_rgb(float y, float u, float v)
{
    return make_float3(saturate01(y + 1.5748f * v),
                       saturate01(y - 0.187324f * u - 0.468124f * v),
                       saturate01(y + 1.8556f * u));
}

template <typename T, int SHIFT>
__global__ void k_pre(const uint8_t *y_plane, ptrdiff_t y_stride,
                      const uint8_t *uv_plane, ptrdiff_t uv_stride,
                      int w, int h, __half *dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;

    // Limited-range normalization, with 8-bit reference values scaled by
    // 1<<SHIFT to the raw container scale (P010 raw u16 = 10-bit << 6 =
    // 8-bit reference * 256, so SHIFT=8 there).
    const float yoff = 16.0f * (1 << SHIFT);
    const float ymax = 219.0f * (1 << SHIFT);
    const float coff = 128.0f * (1 << SHIFT);
    const float cmax = 224.0f * (1 << SHIFT);

    const T *yrow = (const T *)(y_plane + (size_t)y * y_stride);
    const float Y = ((float)yrow[x] - yoff) / ymax;

    // Chroma plane coords: left-sited horizontally, centered vertically.
    const int cw = w >> 1, ch = h >> 1;
    float cxf = x * 0.5f;
    float cyf = y * 0.5f - 0.25f;
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
    U = (U - coff) / cmax;
    V = (V - coff) / cmax;

    const float3 rgb = ycbcr709_to_rgb(Y, U, V);

    const size_t plane = (size_t)w * h, idx = (size_t)y * w + x;
    dst[idx]             = __float2half(rgb.x);
    dst[plane + idx]     = __float2half(rgb.y);
    dst[2 * plane + idx] = __float2half(rgb.z);
}

template <typename T, int SHIFT>
__global__ void k_post_luma(const __half *src, int w, int h,
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

    const float Y = 0.2126f * r + 0.7152f * g + 0.0722f * b;

    const float yoff = 16.0f * (1 << SHIFT);
    const float ymax = 219.0f * (1 << SHIFT);
    float val = rintf(saturate01(Y) * ymax + yoff);

    T *yrow = (T *)(y_plane + (size_t)y * y_stride);
    // P010 keeps the 10-bit value in the MSBs of 16.
    yrow[x] = (T)((unsigned)val << (SHIFT ? 16 - 10 : 0));
}

template <typename T, int SHIFT>
__global__ void k_post_chroma(const __half *src, int w, int h,
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

    const float Y = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    const float U = (b - Y) / 1.8556f;  // [-0.5, 0.5]
    const float V = (r - Y) / 1.5748f;

    const float coff = 128.0f * (1 << SHIFT);
    const float cmax = 224.0f * (1 << SHIFT);
    const float lo = 0.0f, hi = (256.0f * (1 << SHIFT)) - 1.0f;

    float uval = fminf(fmaxf(rintf(U * cmax + coff), lo), hi);
    float vval = fminf(fmaxf(rintf(V * cmax + coff), lo), hi);

    T *uvrow = (T *)(uv_plane + (size_t)cy * uv_stride);
    const int shift = SHIFT ? 16 - 10 : 0;
    uvrow[2 * cx]     = (T)((unsigned)uval << shift);
    uvrow[2 * cx + 1] = (T)((unsigned)vval << shift);
}

#define BLOCK 16

extern "C" int aji_launch_pre(int format,
                              const void *y_plane, ptrdiff_t y_stride,
                              const void *uv_plane, ptrdiff_t uv_stride,
                              int w, int h, void *dst_f16, void *stream)
{
    dim3 block(BLOCK, BLOCK);
    dim3 grid((w + BLOCK - 1) / BLOCK, (h + BLOCK - 1) / BLOCK);
    cudaStream_t s = (cudaStream_t)stream;
    if (format == 1) { // AJI_FMT_NV12
        k_pre<uint8_t, 0><<<grid, block, 0, s>>>(
            (const uint8_t *)y_plane, y_stride, (const uint8_t *)uv_plane,
            uv_stride, w, h, (__half *)dst_f16);
    } else {           // AJI_FMT_P010: raw u16 = 8-bit reference * (1<<8)
        k_pre<uint16_t, 8><<<grid, block, 0, s>>>(
            (const uint8_t *)y_plane, y_stride, (const uint8_t *)uv_plane,
            uv_stride, w, h, (__half *)dst_f16);
    }
    return (int)cudaGetLastError();
}

extern "C" int aji_launch_post(int format, const void *src_f16, int w, int h,
                               void *y_plane, ptrdiff_t y_stride,
                               void *uv_plane, ptrdiff_t uv_stride,
                               void *stream)
{
    dim3 block(BLOCK, BLOCK);
    dim3 grid_l((w + BLOCK - 1) / BLOCK, (h + BLOCK - 1) / BLOCK);
    dim3 grid_c((w / 2 + BLOCK - 1) / BLOCK, (h / 2 + BLOCK - 1) / BLOCK);
    cudaStream_t s = (cudaStream_t)stream;
    if (format == 1) {
        k_post_luma<uint8_t, 0><<<grid_l, block, 0, s>>>(
            (const __half *)src_f16, w, h, (uint8_t *)y_plane, y_stride);
        k_post_chroma<uint8_t, 0><<<grid_c, block, 0, s>>>(
            (const __half *)src_f16, w, h, (uint8_t *)uv_plane, uv_stride);
    } else {
        k_post_luma<uint16_t, 2><<<grid_l, block, 0, s>>>(
            (const __half *)src_f16, w, h, (uint8_t *)y_plane, y_stride);
        k_post_chroma<uint16_t, 2><<<grid_c, block, 0, s>>>(
            (const __half *)src_f16, w, h, (uint8_t *)uv_plane, uv_stride);
    }
    return (int)cudaGetLastError();
}
