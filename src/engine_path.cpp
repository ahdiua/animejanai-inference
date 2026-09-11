// Print the same cache path and default build settings used by the backend.
#include "engine_cache.h"
#include <cstdlib>
#include <iostream>

int main(int argc, char **argv)
{
    if (argc != 6) {
        std::cerr << "usage: aji_engine_path ONNX MODEL_DIR WIDTH HEIGHT INPUT_NAME\n";
        return 2;
    }
    for (int i = 3; i <= 4; i++) {
        char *end = nullptr;
        long value = std::strtol(argv[i], &end, 10);
        if (!*argv[i] || *end || value < 2 || value > 65536) {
            std::cerr << "invalid input dimension\n";
            return 2;
        }
    }
    cudaDeviceProp prop = {};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        std::cerr << "cannot query CUDA device 0 for engine cache identity\n";
        return 1;
    }
    std::string settings = aji_cache::DEFAULT_TRT_ENGINE_SETTINGS;
    const std::string dims = "1x3x" + std::to_string(std::atoi(argv[4])) +
                             "x" + std::to_string(std::atoi(argv[3]));
    size_t pos;
    while ((pos = settings.find("%video_resolution%")) != std::string::npos)
        settings.replace(pos, std::string("%video_resolution%").size(), dims);
    size_t offset = 0;
    while ((pos = settings.find("input:", offset)) != std::string::npos) {
        settings.replace(pos, 5, argv[5]);
        offset = pos + std::string(argv[5]).size() + 1;
    }
    std::cout << aji_cache::short_engine_path_for(argv[2],
        std::filesystem::path(argv[1]).stem().string(), settings) << '\n'
        << settings << '\n';
}
