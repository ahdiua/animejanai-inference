#ifndef AJI_KERNELS_H
#define AJI_KERNELS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* format: enum aji_format (1=NV12, 2=P010). Return 0 or cudaError_t. */
int aji_launch_pre(int format,
                   const void *y_plane, ptrdiff_t y_stride,
                   const void *uv_plane, ptrdiff_t uv_stride,
                   int w, int h, void *dst_f16, void *stream);

int aji_launch_post(int format, const void *src_f16, int w, int h,
                    void *y_plane, ptrdiff_t y_stride,
                    void *uv_plane, ptrdiff_t uv_stride,
                    void *stream);

#ifdef __cplusplus
}
#endif

#endif
