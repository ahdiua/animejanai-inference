#ifndef AJI_KERNELS_H
#define AJI_KERNELS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Colorspace constants, computed host-side from (format, matrix, range,
 * siting). Normalization works in raw container units (P010 values are
 * MSB-aligned u16, so its scales include the <<6). */
typedef struct aji_csp {
    float kr, kb;          /* matrix coefficients */
    float yoff, yscale;    /* y = (raw - yoff) / yscale */
    float coff, cscale;    /* c = (raw - coff) / cscale */
    float cox, coy;        /* chroma grid offsets (siting): cx = x*0.5 + cox */
} aji_csp;

/* format: enum aji_format (1=NV12, 2=P010). All return 0 or cudaError_t. */

int aji_launch_pre(int format,
                   const void *y_plane, ptrdiff_t y_stride,
                   const void *uv_plane, ptrdiff_t uv_stride,
                   int w, int h, const aji_csp *csp,
                   void *dst_f16, void *stream);

int aji_launch_post(int format, const void *src_f16, int w, int h,
                    const aji_csp *csp,
                    void *y_plane, ptrdiff_t y_stride,
                    void *uv_plane, ptrdiff_t uv_stride,
                    void *stream);

/* Catmull-Rom resize of fp16 NCHW RGB (3 planes), src WxH -> dst WxH. */
int aji_launch_resize(const void *src_f16, int sw, int sh,
                      void *dst_f16, int dw, int dh, void *stream);

#ifdef __cplusplus
}
#endif

#endif
