/*
 * aji_trt — TensorRT backend for the aji C ABI (see include/aji.h).
 *
 * Context model: the caller supplies a driver CUcontext (or NULL for the
 * device-0 primary context). All CUDA work, including runtime-API kernel
 * launches, happens with that context pushed current — the CUDA runtime
 * binds to the current driver context, so runtime and driver usage agree.
 */

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <NvInfer.h>

#include "aji.h"
#include "kernels.h"

namespace {

class Logger : public nvinfer1::ILogger {
public:
    aji_log_fn fn = nullptr;
    void *opaque = nullptr;
    void log(Severity sev, const char *msg) noexcept override {
        if (fn)
            fn(opaque, (int)sev, msg);
        else if (sev <= Severity::kWARNING)
            fprintf(stderr, "[aji-trt] %s\n", msg);
    }
};

} // namespace

struct aji_ctx {
    Logger logger;
    CUcontext cu_ctx = nullptr;
    bool owns_primary = false;

    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> exec;

    const char *in_name = nullptr;
    const char *out_name = nullptr;

    void *in_buf = nullptr;   // fp16 NCHW 1x3xHxW at max shape
    void *out_buf = nullptr;  // fp16 NCHW 1x3x2Hx2W at max shape
    int max_w = 0, max_h = 0;
    int scale = 2;

    char errbuf[512] = {0};

    void set_error(const char *fmt, ...) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(errbuf, sizeof(errbuf), fmt, ap);
        va_end(ap);
        if (logger.fn)
            logger.fn(logger.opaque, /*kERROR*/ 1, errbuf);
    }
};

namespace {

struct CtxGuard {
    CUcontext ctx;
    bool ok;
    explicit CtxGuard(CUcontext c) : ctx(c), ok(cuCtxPushCurrent(c) == CUDA_SUCCESS) {}
    ~CtxGuard() {
        if (ok) {
            CUcontext dummy;
            cuCtxPopCurrent(&dummy);
        }
    }
};

} // namespace

extern "C" AJI_EXPORT aji_ctx *aji_create(const aji_create_params *params)
{
    if (!params || params->api_version != AJI_API_VERSION ||
        !params->engine_path || params->max_width < 2 || params->max_height < 2)
        return nullptr;

    auto c = std::make_unique<aji_ctx>();
    c->logger.fn = params->log;
    c->logger.opaque = params->log_opaque;
    c->max_w = params->max_width;
    c->max_h = params->max_height;

    if (cuInit(0) != CUDA_SUCCESS) {
        c->set_error("cuInit failed");
        return nullptr;
    }

    if (params->cuda_context) {
        c->cu_ctx = (CUcontext)params->cuda_context;
    } else {
        CUdevice dev;
        if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS ||
            cuDevicePrimaryCtxRetain(&c->cu_ctx, dev) != CUDA_SUCCESS) {
            c->set_error("failed to retain primary CUDA context");
            return nullptr;
        }
        c->owns_primary = true;
    }

    CtxGuard guard(c->cu_ctx);
    if (!guard.ok) {
        c->set_error("cuCtxPushCurrent failed");
        return nullptr;
    }

    std::ifstream f(params->engine_path, std::ios::binary);
    if (!f) {
        c->set_error("cannot open engine: %s", params->engine_path);
        return nullptr;
    }
    std::vector<char> blob((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());

    c->runtime.reset(nvinfer1::createInferRuntime(c->logger));
    if (!c->runtime) {
        c->set_error("createInferRuntime failed");
        return nullptr;
    }
    c->engine.reset(c->runtime->deserializeCudaEngine(blob.data(), blob.size()));
    if (!c->engine) {
        c->set_error("engine deserialization failed");
        return nullptr;
    }

    for (int i = 0; i < c->engine->getNbIOTensors(); i++) {
        const char *name = c->engine->getIOTensorName(i);
        if (c->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
            c->in_name = name;
        else
            c->out_name = name;
    }
    if (!c->in_name || !c->out_name) {
        c->set_error("engine I/O tensors not found");
        return nullptr;
    }
    if (c->engine->getTensorDataType(c->in_name) != nvinfer1::DataType::kHALF ||
        c->engine->getTensorDataType(c->out_name) != nvinfer1::DataType::kHALF) {
        c->set_error("engine I/O is not fp16");
        return nullptr;
    }

    c->exec.reset(c->engine->createExecutionContext());
    if (!c->exec) {
        c->set_error("createExecutionContext failed");
        return nullptr;
    }

    // Probe the scale factor at a small shape.
    if (!c->exec->setInputShape(c->in_name, nvinfer1::Dims4{1, 3, 64, 64})) {
        c->set_error("setInputShape probe failed (profile min > 64?)");
        return nullptr;
    }
    nvinfer1::Dims od = c->exec->getTensorShape(c->out_name);
    if (od.nbDims != 4 || od.d[3] % 64 != 0) {
        c->set_error("unexpected output dims from probe");
        return nullptr;
    }
    c->scale = (int)(od.d[3] / 64);

    const size_t in_bytes = (size_t)3 * c->max_w * c->max_h * 2;
    const size_t out_bytes = in_bytes * c->scale * c->scale;
    if (cudaMalloc(&c->in_buf, in_bytes) != cudaSuccess ||
        cudaMalloc(&c->out_buf, out_bytes) != cudaSuccess) {
        c->set_error("cudaMalloc of tensor buffers failed (%zu + %zu bytes)",
                     in_bytes, out_bytes);
        return nullptr;
    }
    if (!c->exec->setTensorAddress(c->in_name, c->in_buf) ||
        !c->exec->setTensorAddress(c->out_name, c->out_buf)) {
        c->set_error("setTensorAddress failed");
        return nullptr;
    }

    return c.release();
}

extern "C" AJI_EXPORT int aji_infer(aji_ctx *c, const aji_frame *in,
                                    const aji_frame *out, void *cu_stream)
{
    if (!c || !in || !out)
        return AJI_ERR;
    if (in->format != AJI_FMT_NV12 && in->format != AJI_FMT_P010) {
        c->set_error("unsupported input format %d", in->format);
        return AJI_ERR_FORMAT;
    }
    if (out->format != in->format) {
        c->set_error("output format must match input (spike limitation)");
        return AJI_ERR_FORMAT;
    }
    if (in->width < 2 || in->height < 2 || (in->width & 1) || (in->height & 1) ||
        in->width > c->max_w || in->height > c->max_h) {
        c->set_error("bad input dims %dx%d (max %dx%d, must be even)",
                     in->width, in->height, c->max_w, c->max_h);
        return AJI_ERR_SHAPE;
    }
    if (out->width != in->width * c->scale || out->height != in->height * c->scale) {
        c->set_error("output dims %dx%d != %dx scale %d", out->width,
                     out->height, in->width, c->scale);
        return AJI_ERR_SHAPE;
    }

    CtxGuard guard(c->cu_ctx);
    if (!guard.ok) {
        c->set_error("cuCtxPushCurrent failed");
        return AJI_ERR_CUDA;
    }
    cudaStream_t stream = (cudaStream_t)cu_stream;

    int err = aji_launch_pre(in->format, in->plane[0], in->stride[0],
                             in->plane[1], in->stride[1], in->width,
                             in->height, c->in_buf, stream);
    if (err) {
        c->set_error("pre-kernel launch failed: %s",
                     cudaGetErrorString((cudaError_t)err));
        return AJI_ERR_CUDA;
    }

    if (!c->exec->setInputShape(c->in_name,
                                nvinfer1::Dims4{1, 3, in->height, in->width})) {
        c->set_error("setInputShape %dx%d rejected by engine profile",
                     in->width, in->height);
        return AJI_ERR_SHAPE;
    }
    if (!c->exec->enqueueV3(stream)) {
        c->set_error("enqueueV3 failed");
        return AJI_ERR_ENGINE;
    }

    err = aji_launch_post(out->format, c->out_buf, out->width, out->height,
                          out->plane[0], out->stride[0], out->plane[1],
                          out->stride[1], stream);
    if (err) {
        c->set_error("post-kernel launch failed: %s",
                     cudaGetErrorString((cudaError_t)err));
        return AJI_ERR_CUDA;
    }

    return AJI_OK;
}

extern "C" AJI_EXPORT int aji_scale_factor(aji_ctx *c)
{
    return c ? c->scale : 0;
}

extern "C" AJI_EXPORT const char *aji_last_error(aji_ctx *c)
{
    return c ? c->errbuf : "no context";
}

extern "C" AJI_EXPORT void aji_destroy(aji_ctx **pc)
{
    if (!pc || !*pc)
        return;
    aji_ctx *c = *pc;
    {
        CtxGuard guard(c->cu_ctx);
        cudaFree(c->in_buf);
        cudaFree(c->out_buf);
        c->exec.reset();
        c->engine.reset();
        c->runtime.reset();
    }
    if (c->owns_primary) {
        CUdevice dev;
        if (cuDeviceGet(&dev, 0) == CUDA_SUCCESS)
            cuDevicePrimaryCtxRelease(dev);
    }
    delete c;
    *pc = nullptr;
}
