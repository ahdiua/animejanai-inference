#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>

namespace aji_cache {
namespace fs = std::filesystem;

// TRT 11: strongly-typed is the default (--stronglyTyped is a no-op), and sanitize_settings_trt11
// strips --inputIOFormats/--outputIOFormats/--tacticSources anyway (types come from the network;
// the cuDNN/cuBLAS tactic sources are gone). So the only flags worth carrying are the builder
// optimization level, the build shape profile, and skip-inference (we do the inference ourselves).
inline constexpr const char *DEFAULT_TRT_ENGINE_SETTINGS =
    "--builderOptimizationLevel=5 "
    "--minShapes=input:%video_resolution% "
    "--optShapes=input:%video_resolution% "
    "--maxShapes=input:%video_resolution% "
    "--skipInference";

// zlib-compatible CRC-32 (matches Python zlib.crc32 used for engine names).
inline uint32_t crc32_z(const std::string &data)
{
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++)
                c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        init = true;
    }
    uint32_t c = 0xFFFFFFFFu;
    for (unsigned char ch : data)
        c = table[(c ^ ch) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

inline std::string sanitize_token(std::string s)
{
    for (auto &ch : s) {
        if (ch == ' ')
            ch = '-';
    }
    std::string out;
    for (char ch : s) {
        if (isalnum((unsigned char)ch) || ch == '.' || ch == '_' || ch == '-')
            out += ch;
    }
    return out.empty() ? "device0" : out;
}


inline std::string trt_version_token()
{
    int32_t v = getInferLibVersion();
    int major, minor, patch;
    if (v < 10000) {
        major = v / 1000; minor = (v / 100) % 10; patch = v % 100;
    } else {
        major = v / 10000; minor = (v / 100) % 100; patch = v % 100;
    }
    char buf[32];
    snprintf(buf, sizeof(buf), "%d.%d.%d", major, minor, patch);
    return buf;
}

inline std::string gpu_token()
{
    cudaDeviceProp prop = {};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess)
        return "unknown";
    std::string tok = sanitize_token(prop.name);
    return tok + "-sm" + std::to_string(prop.major);
}

inline std::string engine_suffix()
{
    return ".trt-" + trt_version_token() + ".gpu-" + gpu_token() + ".engine";
}

inline std::string engine_path_for(const std::string &model_dir,
                            const std::string &onnx_name,
                            const std::string &settings)
{
    return (fs::path(model_dir) /
            (onnx_name + "." + std::to_string(crc32_z(settings)) + engine_suffix()))
        .string();
}

inline std::string short_engine_path_for(const std::string &model_dir,
                                  const std::string &onnx_name,
                                  const std::string &settings)
{
    char model_hash[9];
    snprintf(model_hash, sizeof(model_hash), "%08x", crc32_z(onnx_name));
    return (fs::path(model_dir) /
            ("aji-" + std::string(model_hash) + "." +
             std::to_string(crc32_z(settings)) + engine_suffix()))
        .string();
}


} // namespace aji_cache
