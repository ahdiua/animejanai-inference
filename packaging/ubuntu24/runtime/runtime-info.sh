#!/usr/bin/env bash
set -Eeuo pipefail
RUNTIME_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="${RUNTIME_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${RUNTIME_ROOT}/build:${RUNTIME_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "AnimeJaNai runtime: @PACKAGE_VERSION@"
echo "Target OS: Ubuntu 24.04 x86_64"
echo "Builder architecture: @GPU_ARCH@"
echo "Build CUDA: @CUDA_VERSION@"
echo "TensorRT: @TRT_VERSION@"
echo
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader \
        2>/dev/null || nvidia-smi
else
    echo "nvidia-smi: not found (install a compatible NVIDIA driver on the host)"
fi
echo
echo "Packaged models:"
find "${RUNTIME_ROOT}/onnx" -type f -name '*.onnx' -printf '  %P\n' | sort -V
