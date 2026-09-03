#!/usr/bin/env bash
# Build a self-contained AnimeJaNai runtime archive for Ubuntu 24.04.
#
# The resulting archive contains the full TensorRT runtime and one TensorRT
# builder resource (matching the selected GPU SM), so missing engines can be
# built on the target GPU without installing CUDA Toolkit or TensorRT there.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ARCH="89"
OUTPUT_DIR="${PROJECT_ROOT}/dist"
LOCAL_MODELS_DIR="${PROJECT_ROOT}/onnx"
INSTALL_DEPS=0
KEEP_WORK=0
PACKAGE_VERSION="${PACKAGE_VERSION:-}"

usage() {
    cat <<'EOF'
Usage: scripts/package-ubuntu24-runtime.sh [options]

Options:
  --gpu-arch <75|80|86|89|90|100|120>
                              TensorRT builder/GPU architecture (default: 89)
  --output-dir <path>         Archive output directory (default: ./dist)
  --models-dir <path>         Prefer models from this directory before download
  --version <value>           Package version (default: git describe/commit)
  --install-deps              Install Ubuntu 24.04 build dependencies with apt
  --keep-work                 Keep the temporary work directory for inspection
  -h, --help                  Show this help

The script must run on Ubuntu 24.04 x86_64. GitHub Actions invokes this exact
script with --install-deps. A target GPU is not needed while packaging.
EOF
}

while (($#)); do
    case "$1" in
        --gpu-arch)
            GPU_ARCH="${2#sm}"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --models-dir)
            LOCAL_MODELS_DIR="$2"
            shift 2
            ;;
        --version)
            PACKAGE_VERSION="$2"
            shift 2
            ;;
        --install-deps)
            INSTALL_DEPS=1
            shift
            ;;
        --keep-work)
            KEEP_WORK=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "${GPU_ARCH}" in
    75|80|86|89|90|100|120) ;;
    *)
        echo "Unsupported GPU architecture: sm${GPU_ARCH}" >&2
        exit 2
        ;;
esac

if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "This packager currently supports x86_64 only." >&2
    exit 1
fi

if [[ ! -r /etc/os-release ]]; then
    echo "Cannot identify the host OS." >&2
    exit 1
fi
# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
    echo "Ubuntu 24.04 is required; detected ${PRETTY_NAME:-unknown}." >&2
    exit 1
fi

run_root() {
    if [[ "$(id -u)" -eq 0 ]]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        echo "Root access is required to install packages: $*" >&2
        return 1
    fi
}

download() {
    local url="$1"
    local destination="$2"
    echo "Downloading $(basename "${destination}")"
    curl --fail --location --retry 4 --retry-delay 2 \
        --output "${destination}.part" "${url}"
    mv "${destination}.part" "${destination}"
}

install_dependencies() {
    local dep_tmp
    dep_tmp="$(mktemp -d)"
    run_root apt-get update
    run_root apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg git build-essential cmake ninja-build pkg-config \
        patchelf p7zip-full python3 python3-venv xz-utils zstd

    download \
        "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb" \
        "${dep_tmp}/cuda-keyring.deb"
    run_root dpkg -i "${dep_tmp}/cuda-keyring.deb"
    run_root apt-get update

    local cuda_suffix
    cuda_suffix="$(apt-cache search --names-only '^cuda-nvcc-13-[0-9]+$' \
        | awk '{print $1}' | sed 's/^cuda-nvcc-//' | sort -V | tail -n1)"
    if [[ -z "${cuda_suffix}" ]]; then
        echo "No CUDA 13 nvcc package was found in the NVIDIA repository." >&2
        return 1
    fi
    echo "Installing CUDA ${cuda_suffix//-/.} build components and TensorRT 11"
    run_root apt-get install -y --no-install-recommends \
        "cuda-nvcc-${cuda_suffix}" \
        "cuda-cudart-dev-${cuda_suffix}" \
        "cuda-driver-dev-${cuda_suffix}" \
        tensorrt-dev libnvinfer-bin
    run_root ldconfig
    rm -rf "${dep_tmp}"
}

if [[ "${INSTALL_DEPS}" -eq 1 ]]; then
    install_dependencies
fi

for command_name in cmake ninja curl patchelf 7z zstd pkg-config; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Missing required command: ${command_name} (use --install-deps)" >&2
        exit 1
    fi
done

NVCC="$(command -v nvcc || true)"
if [[ -z "${NVCC}" && -x /usr/local/cuda/bin/nvcc ]]; then
    NVCC=/usr/local/cuda/bin/nvcc
fi
if [[ -z "${NVCC}" ]]; then
    NVCC="$(find /usr/local -maxdepth 3 -type f -path '*/cuda-*/bin/nvcc' \
        -print 2>/dev/null | sort -V | tail -n1)"
fi
if [[ -z "${NVCC}" ]]; then
    echo "nvcc was not found (use --install-deps)." >&2
    exit 1
fi

TRTEXEC="$(command -v trtexec || true)"
if [[ -z "${TRTEXEC}" ]]; then
    for candidate in /usr/src/tensorrt/bin/trtexec /usr/bin/trtexec; do
        if [[ -x "${candidate}" ]]; then
            TRTEXEC="${candidate}"
            break
        fi
    done
fi
if [[ -z "${TRTEXEC}" ]]; then
    echo "trtexec was not found (install libnvinfer-bin)." >&2
    exit 1
fi

if [[ -z "${PACKAGE_VERSION}" ]]; then
    PACKAGE_VERSION="$(git -C "${PROJECT_ROOT}" describe --tags --always --dirty 2>/dev/null || date -u +%Y%m%d)"
fi
PACKAGE_VERSION="$(printf '%s' "${PACKAGE_VERSION}" | tr -cs 'A-Za-z0-9._-' '-')"

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/aji-runtime.XXXXXX")"
PROGRESS_PID=""
cleanup() {
    if [[ -n "${PROGRESS_PID}" ]]; then
        kill "${PROGRESS_PID}" 2>/dev/null || true
        wait "${PROGRESS_PID}" 2>/dev/null || true
    fi
    if [[ "${KEEP_WORK}" -eq 1 ]]; then
        echo "Kept work directory: ${WORK_DIR}"
    else
        rm -rf "${WORK_DIR}"
    fi
}
trap cleanup EXIT

FFMPEG_ROOT="${WORK_DIR}/ffmpeg"
FFMPEG_ARCHIVE="${WORK_DIR}/ffmpeg.tar.xz"
download \
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-shared-8.1.tar.xz" \
    "${FFMPEG_ARCHIVE}"
mkdir -p "${FFMPEG_ROOT}"
tar -xJf "${FFMPEG_ARCHIVE}" -C "${FFMPEG_ROOT}" --strip-components=1
for pc_file in "${FFMPEG_ROOT}"/lib/pkgconfig/*.pc; do
    sed -i "s|^prefix=.*|prefix=${FFMPEG_ROOT}|" "${pc_file}"
done

BUILD_DIR="${WORK_DIR}/build"
PKG_CONFIG_PATH="${FFMPEG_ROOT}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}" \
CUDACXX="${NVCC}" \
cmake -B "${BUILD_DIR}" -S "${PROJECT_ROOT}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DAJI_TRT_ROOT=/usr \
    -DCMAKE_CUDA_ARCHITECTURES="${GPU_ARCH}-real"
cmake --build "${BUILD_DIR}" --parallel "$(nproc)"

PACKAGE_NAME="animejanai-ubuntu24-x86_64-sm${GPU_ARCH}-${PACKAGE_VERSION}"
STAGE_ROOT="${WORK_DIR}/${PACKAGE_NAME}"
mkdir -p "${STAGE_ROOT}/build" "${STAGE_ROOT}/bin" "${STAGE_ROOT}/lib" \
    "${STAGE_ROOT}/onnx/rife" "${STAGE_ROOT}/licenses"

for output in libaji.so libaji_trt.so aji_encode aji_harness aji_kernel_test; do
    if [[ ! -e "${BUILD_DIR}/${output}" ]]; then
        echo "Expected build output is missing: ${output}" >&2
        exit 1
    fi
    cp -a "${BUILD_DIR}/${output}" "${STAGE_ROOT}/build/"
done
cp -a "${TRTEXEC}" "${STAGE_ROOT}/bin/trtexec.real"
cp -a "${FFMPEG_ROOT}/bin/ffmpeg" "${STAGE_ROOT}/bin/ffmpeg.real"
cp -a "${FFMPEG_ROOT}/bin/ffprobe" "${STAGE_ROOT}/bin/ffprobe.real"
cp -a "${FFMPEG_ROOT}"/lib/*.so* "${STAGE_ROOT}/lib/"
ln -s trtexec.real "${STAGE_ROOT}/bin/trtexec"
ln -s ffmpeg.real "${STAGE_ROOT}/bin/ffmpeg"
ln -s ffprobe.real "${STAGE_ROOT}/bin/ffprobe"

find_library_path() {
    local soname="$1"
    local result
    result="$(ldconfig -p 2>/dev/null | awk -v name="${soname}" '$1 == name { print $NF; exit }')"
    if [[ -z "${result}" || ! -e "${result}" ]]; then
        result="$(find /usr /usr/local -type f -name "${soname}*" \
            -print -quit 2>/dev/null)"
    fi
    if [[ -z "${result}" || ! -e "${result}" ]]; then
        echo "Required library was not found: ${soname}" >&2
        return 1
    fi
    readlink -f "${result}"
}

copy_library_family() {
    local soname="$1"
    local real_file source_dir base_prefix candidate
    real_file="$(find_library_path "${soname}")"
    source_dir="$(dirname "${real_file}")"
    base_prefix="${soname%.so*}.so"
    cp -a "${real_file}" "${STAGE_ROOT}/lib/"
    for candidate in "${source_dir}/${base_prefix}" "${source_dir}/${soname}"; do
        if [[ -L "${candidate}" ]]; then
            cp -a "${candidate}" "${STAGE_ROOT}/lib/"
        fi
    done
    if [[ ! -e "${STAGE_ROOT}/lib/${soname}" ]]; then
        ln -s "$(basename "${real_file}")" "${STAGE_ROOT}/lib/${soname}"
    fi
}

copy_library_family libnvinfer.so.11
copy_library_family libnvinfer_plugin.so.11
copy_library_family libnvonnxparser.so.11
copy_library_family "libnvinfer_builder_resource_sm${GPU_ARCH}.so.11"
copy_library_family libcudart.so.13

for elf_file in \
    "${STAGE_ROOT}/build/aji_encode" \
    "${STAGE_ROOT}/build/aji_harness" \
    "${STAGE_ROOT}/build/aji_kernel_test"; do
    patchelf --set-rpath '$ORIGIN:$ORIGIN/../lib' "${elf_file}"
done
patchelf --set-rpath '$ORIGIN:$ORIGIN/../lib' "${STAGE_ROOT}/build/libaji_trt.so"
patchelf --set-rpath '$ORIGIN/../lib' "${STAGE_ROOT}/bin/trtexec.real"
patchelf --set-rpath '$ORIGIN/../lib' "${STAGE_ROOT}/bin/ffmpeg.real"
patchelf --set-rpath '$ORIGIN/../lib' "${STAGE_ROOT}/bin/ffprobe.real"

copy_model() {
    local relative_path="$1"
    local url="$2"
    local expected_sha="$3"
    local local_path="${LOCAL_MODELS_DIR}/${relative_path}"
    local staged_path="${STAGE_ROOT}/onnx/${relative_path}"
    mkdir -p "$(dirname "${staged_path}")"
    if [[ -s "${local_path}" ]]; then
        echo "Using local model: ${relative_path}"
        cp -aL "${local_path}" "${staged_path}"
    else
        download "${url}" "${staged_path}"
    fi
    if [[ -n "${expected_sha}" ]]; then
        printf '%s  %s\n' "${expected_sha}" "${staged_path}" | sha256sum --check --status || {
            echo "Checksum mismatch for ${relative_path}" >&2
            exit 1
        }
    fi
}

copy_model \
    "2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx" \
    "https://r2.ahdiua.com/2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx" \
    "7b804d87320f37b7269d6e27796046fbd975e95e2aa1f65d5557f52a36282716"
copy_model \
    "2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx" \
    "https://r2.ahdiua.com/2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx" \
    "b308e93c3dda3c4f9f968295b2b7b8d1c0583a711bca53a08d51134d9d1cbd5c"
copy_model \
    "2x_AnimeJaNai_HD_V3.1Sharp1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx" \
    "https://r2.ahdiua.com/2x_AnimeJaNai_HD_V3.1Sharp1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx" \
    "72246eded3aab52f8d1a71613c0a1d900734b358d1820b62eb36f1ed376e1cf4"
copy_model \
    "2x_AnimeJaNai_HD_V3.1Sharp1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx" \
    "https://r2.ahdiua.com/2x_AnimeJaNai_HD_V3.1Sharp1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx" \
    "dde763c8202cfebac5ec030ea6644f3ee339ea35bbda8a76e257405399552f0f"
copy_model \
    "2x_AnimeJaNai_SD_V1beta34_Compact_1x3xHxW_dyn-HW_strong_fp16_op21_dynamo.onnx" \
    "https://r2.ahdiua.com/2x_AnimeJaNai_SD_V1beta34_Compact_1x3xHxW_dyn-HW_strong_fp16_op21_dynamo.onnx" \
    "ded46591ddfaeb22b3daaf84ad7b36838335aa3efca66acfa36177ce4ab21eb5"
copy_model \
    "RealESRGANv2-animevideo-xsx2-v0.2.3.0-fp16-dynamic.onnx" \
    "https://r2.ahdiua.com/RealESRGANv2-animevideo-xsx2-v0.2.3.0-fp16-dynamic.onnx" \
    "885b5abb8d7203b2cedea8c678913432ce146dca60109e2532bb5fcb53656fd5"

prepare_rife_model() {
    local version="$1"
    local expected_sha="$2"
    local relative_path="rife/rife_v${version}.onnx"
    local local_path="${LOCAL_MODELS_DIR}/${relative_path}"
    local staged_path="${STAGE_ROOT}/onnx/${relative_path}"
    if [[ -s "${local_path}" ]]; then
        echo "Using local RIFE model: v${version}"
        cp -aL "${local_path}" "${staged_path}"
        printf '%s  %s\n' "${expected_sha}" "${staged_path}" \
            | sha256sum --check --status || {
                echo "Local RIFE v${version} is not the expected fp16 model." >&2
                exit 1
            }
        return
    fi

    local archive="${WORK_DIR}/rife_v${version}.7z"
    local source_dir="${WORK_DIR}/rife-v${version}-source"
    download \
        "https://github.com/AmusementClub/vs-mlrt/releases/download/external-models/rife_v${version}.7z" \
        "${archive}"
    mkdir -p "${source_dir}"
    7z x -y -o"${source_dir}" "${archive}" >/dev/null
    local source_model="${source_dir}/rife/rife_v${version}.onnx"
    if [[ ! -s "${source_model}" ]]; then
        echo "rife_v${version}.onnx was not found in the downloaded archive." >&2
        exit 1
    fi

    local python_env="${WORK_DIR}/model-converter"
    if [[ ! -x "${python_env}/bin/python" ]]; then
        python3 -m venv "${python_env}"
        "${python_env}/bin/pip" install --disable-pip-version-check --no-cache-dir \
            'numpy==2.5.2' 'onnx==1.22.0'
    fi
    local converter_input="${WORK_DIR}/converter-input-v${version}"
    mkdir -p "${converter_input}"
    cp -a "${source_model}" "${converter_input}/rife_v${version}.onnx"
    "${python_env}/bin/python" "${PROJECT_ROOT}/tools/convert_rife_fp16.py" \
        "${converter_input}" "${STAGE_ROOT}/onnx/rife" "rife_v${version}.onnx"
    if [[ ! -s "${staged_path}" ]]; then
        echo "RIFE v${version} fp16 conversion failed." >&2
        exit 1
    fi
    printf '%s  %s\n' "${expected_sha}" "${staged_path}" \
        | sha256sum --check --status || {
            echo "Converted RIFE v${version} checksum mismatch." >&2
            exit 1
        }
}

prepare_rife_model "4.26" \
    "fd6b06538898c9a94d20f0fe55febceadbcde5ad64952240733b7275ef4c9d59"
prepare_rife_model "4.25" \
    "de0fb71ae50fe082eb3903fd5cf752063dea5e9fd61e8bbf3c4577e11b877c9c"

cp -a "${PROJECT_ROOT}/packaging/ubuntu24/runtime/." "${STAGE_ROOT}/"
cp -a "${PROJECT_ROOT}/generate_cmd.sh" "${STAGE_ROOT}/generate_cmd.sh"
chmod +x "${STAGE_ROOT}/aji_encode" "${STAGE_ROOT}/ffmpeg" \
    "${STAGE_ROOT}/ffprobe" "${STAGE_ROOT}/trtexec" \
    "${STAGE_ROOT}/aji_harness" "${STAGE_ROOT}/aji_kernel_test" \
    "${STAGE_ROOT}/runtime-info.sh" "${STAGE_ROOT}/generate_cmd.sh"

TRT_VERSION="$(dpkg-query -W -f='${Version}' libnvinfer11 2>/dev/null || true)"
CUDA_VERSION="$(${NVCC} --version | sed -n 's/.*release \([^,]*\).*/\1/p' | head -n1)"
TRT_VERSION="${TRT_VERSION:-11}"
sed -i \
    -e "s/@GPU_ARCH@/sm${GPU_ARCH}/g" \
    -e "s/@PACKAGE_VERSION@/${PACKAGE_VERSION}/g" \
    -e "s/@TRT_VERSION@/${TRT_VERSION}/g" \
    -e "s/@CUDA_VERSION@/${CUDA_VERSION}/g" \
    "${STAGE_ROOT}/README.md" "${STAGE_ROOT}/runtime-info.sh"

if [[ -d /usr/share/doc/libnvinfer11 ]]; then
    cp -a /usr/share/doc/libnvinfer11/copyright \
        "${STAGE_ROOT}/licenses/TensorRT-copyright" 2>/dev/null || true
fi
find "${FFMPEG_ROOT}" -maxdepth 2 -type f \
    \( -iname 'license*' -o -iname 'copying*' \) \
    -exec cp -a {} "${STAGE_ROOT}/licenses/" \; 2>/dev/null || true

(
    cd "${STAGE_ROOT}"
    find . -type f ! -name SHA256SUMS -print0 \
        | sort -z | xargs -0 sha256sum > SHA256SUMS
)

export LD_LIBRARY_PATH="${STAGE_ROOT}/build:${STAGE_ROOT}/lib"
for elf_file in "${STAGE_ROOT}/build/aji_encode" \
                "${STAGE_ROOT}/build/libaji_trt.so" \
                "${STAGE_ROOT}/bin/trtexec.real"; do
    missing="$(ldd "${elf_file}" 2>&1 | awk '/not found/ && $1 != "libcuda.so.1" {print}')"
    if [[ -n "${missing}" ]]; then
        echo "Unresolved packaged dependency in ${elf_file}:" >&2
        echo "${missing}" >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}"
ARCHIVE_PATH="${OUTPUT_DIR}/${PACKAGE_NAME}.tar.zst"
SOURCE_EPOCH="${SOURCE_DATE_EPOCH:-$(git -C "${PROJECT_ROOT}" log -1 --format=%ct 2>/dev/null || date +%s)}"
STAGE_SIZE="$(du -sh "${STAGE_ROOT}" | cut -f1)"
ARCHIVE_STARTED_AT="$(date +%s)"

echo
echo "Compressing ${STAGE_SIZE} runtime archive with zstd -19 using all CPU threads..."
archive_progress() {
    local now elapsed minutes seconds output_size
    while sleep 30; do
        now="$(date +%s)"
        elapsed=$((now - ARCHIVE_STARTED_AT))
        minutes=$((elapsed / 60))
        seconds=$((elapsed % 60))
        output_size="pending"
        if [[ -f "${ARCHIVE_PATH}" ]]; then
            output_size="$(du -h "${ARCHIVE_PATH}" | cut -f1)"
        fi
        printf '[archive] zstd -19 running: %dm %02ds elapsed, %s written\n' \
            "${minutes}" "${seconds}" "${output_size}"
    done
}

archive_progress &
PROGRESS_PID=$!
archive_status=0
tar --sort=name --mtime="@${SOURCE_EPOCH}" \
    --owner=0 --group=0 --numeric-owner \
    -I 'zstd -T0 -19' \
    -C "${WORK_DIR}" -cf "${ARCHIVE_PATH}" "${PACKAGE_NAME}" \
    || archive_status=$?
kill "${PROGRESS_PID}" 2>/dev/null || true
wait "${PROGRESS_PID}" 2>/dev/null || true
PROGRESS_PID=""
if ((archive_status != 0)); then
    echo "Runtime archive compression failed with exit code ${archive_status}." >&2
    exit "${archive_status}"
fi
ARCHIVE_ELAPSED=$(( $(date +%s) - ARCHIVE_STARTED_AT ))
printf 'Compression finished in %dm %02ds.\n' \
    "$((ARCHIVE_ELAPSED / 60))" "$((ARCHIVE_ELAPSED % 60))"
(
    cd "${OUTPUT_DIR}"
    sha256sum "$(basename "${ARCHIVE_PATH}")" \
        > "$(basename "${ARCHIVE_PATH}").sha256"
)
if [[ -n "${AJI_OUTPUT_UID:-}" && -n "${AJI_OUTPUT_GID:-}" ]]; then
    chown "${AJI_OUTPUT_UID}:${AJI_OUTPUT_GID}" \
        "${ARCHIVE_PATH}" "${ARCHIVE_PATH}.sha256"
fi

echo
echo "Runtime package created:"
du -h "${ARCHIVE_PATH}" "${ARCHIVE_PATH}.sha256"
echo "GPU builder architecture: sm${GPU_ARCH}"
echo "Models: 6 upscale + RIFE v4.26/v4.25"
