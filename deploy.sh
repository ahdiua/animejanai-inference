#!/usr/bin/env bash
# ==============================================================================
# AnimeJaNai-Inference 交互式环境检测、安装与部署脚本
# 适用环境：Ubuntu 24.04 / 22.04 LTS (AutoDL / 云 GPU 容器 / 本地 GPU 服务器)
# 特别集成：AutoDL 旧源清理、动态匹配最高 CUDA Toolkit (支持 13.x / 12.8 / 多版本共存与保留)、
#          NVIDIA 官方源配置、TensorRT 11 自动适配 (Strongly-Typed 无需 --fp16)、
#          BtbN FFmpeg (n8.1/master) Shared、NVENC 容器多卡修复补丁 (libnvenc_fix)、
#          Python venv、3 大常用超分模型与 1 分钟快速验证
# ==============================================================================

set -o pipefail

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 默认路径与 URL 配置
FFMPEG_INSTALL_DIR="/opt/ffmpeg"
FFMPEG_PC_DIR="${FFMPEG_INSTALL_DIR}/lib/pkgconfig"
MODELS_DIR="${HOME}/models"
NVENC_FIX_SO="/opt/libnvenc_fix.so"
NVENC_FIX_SRC_URL="https://raw.githubusercontent.com/flexgrip/nvidia-gpu-enumeration/master/nvenc_fix.c"
NVENC_FIX_SRC_MIRROR="https://ghproxy.net/https://raw.githubusercontent.com/flexgrip/nvidia-gpu-enumeration/master/nvenc_fix.c"

# AnimeJaNai 官方发布资源包 (内含 V3.1 Performance 与 Balanced 等全部高清超分 ONNX 模型)
ANIMEJANAI_OVERLAY_URL="https://github.com/the-database/mpv-AnimeJaNai/releases/download/3.6.0/mpv-upscale-2x_animejanai-overlay-3.6.0-linux-x64.7z"
ANIMEJANAI_OVERLAY_MIRROR="https://ghproxy.net/https://github.com/the-database/mpv-AnimeJaNai/releases/download/3.6.0/mpv-upscale-2x_animejanai-overlay-3.6.0-linux-x64.7z"

# Real-ESRGAN Anime 6B 经典动漫模型 (Hugging Face / HF-Mirror)
DEFAULT_ONNX_REALESRGAN_ANIME6B_URL="https://huggingface.co/deepghs/imgutils-models/resolve/main/real_esrgan/RealESRGAN_x4plus_anime_6B.onnx"
DEFAULT_ONNX_REALESRGAN_ANIME6B_MIRROR="https://hf-mirror.com/deepghs/imgutils-models/resolve/main/real_esrgan/RealESRGAN_x4plus_anime_6B.onnx"

# FFmpeg 预设构建 (n8.1 与当前主流驱动 570~595 的 NVENC API 完美兼容；master 则为最新构建)
FFMPEG_TAR_N81_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-shared-8.1.tar.xz"
FFMPEG_TAR_MASTER_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl-shared.tar.xz"
CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
REPO_GIT_URL="https://github.com/the-database/animejanai-inference.git"

# 权限检测辅助函数
run_as_root() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    else
        if command -v sudo &>/dev/null; then
            sudo "$@"
        else
            echo -e "${RED}[错误] 需要 root 权限执行此命令，但当前用户不是 root 且未安装 sudo。${NC}"
            return 1
        fi
    fi
}

# 智能定位与初始化项目根目录 (PROJECT_ROOT)
init_project_root() {
    if [ -f "CMakeLists.txt" ] && grep -qi "animejanai" "CMakeLists.txt" 2>/dev/null; then
        PROJECT_ROOT="$(pwd)"
    elif [ -f "${SCRIPT_DIR}/CMakeLists.txt" ] && grep -qi "animejanai" "${SCRIPT_DIR}/CMakeLists.txt" 2>/dev/null; then
        PROJECT_ROOT="${SCRIPT_DIR}"
    elif [ -d "${SCRIPT_DIR}/animejanai-inference" ] && [ -f "${SCRIPT_DIR}/animejanai-inference/CMakeLists.txt" ]; then
        PROJECT_ROOT="${SCRIPT_DIR}/animejanai-inference"
    elif [ -d "./animejanai-inference" ] && [ -f "./animejanai-inference/CMakeLists.txt" ]; then
        PROJECT_ROOT="$(pwd)/animejanai-inference"
    elif [ -d "/root/animejanai-inference" ] && [ -f "/root/animejanai-inference/CMakeLists.txt" ]; then
        PROJECT_ROOT="/root/animejanai-inference"
    elif [ -d "$HOME/animejanai-inference" ] && [ -f "$HOME/animejanai-inference/CMakeLists.txt" ]; then
        PROJECT_ROOT="$HOME/animejanai-inference"
    else
        echo -e "${YELLOW}未检测到 animejanai-inference 源码目录。${NC}"
        echo -e "${CYAN}正在从 GitHub 克隆仓库: ${REPO_GIT_URL} ...${NC}"
        git clone "$REPO_GIT_URL" animejanai-inference
        if [ -d "animejanai-inference" ] && [ -f "animejanai-inference/CMakeLists.txt" ]; then
            PROJECT_ROOT="$(pwd)/animejanai-inference"
            echo -e "${GREEN}✔ 仓库克隆成功: ${PROJECT_ROOT}${NC}"
        else
            echo -e "${RED}[错误] 无法找到或克隆 animejanai-inference 源码！请确认网络或手动 git clone。${NC}"
            PROJECT_ROOT="$(pwd)"
        fi
    fi

    cd "$PROJECT_ROOT"
    VENV_DIR="${PROJECT_ROOT}/.venv"
}

# 智能同步与激活最高版本的 CUDA Toolkit
sync_and_activate_highest_cuda() {
    local highest_cuda=$(find /usr/local -maxdepth 1 -type d -name "cuda-[0-9]*" 2>/dev/null | sort -V | tail -n 1)
    if [ -n "$highest_cuda" ]; then
        local current_target="$(readlink -f /usr/local/cuda 2>/dev/null || echo "")"
        if [ "$current_target" != "$highest_cuda" ]; then
            run_as_root ln -sfn "$highest_cuda" /usr/local/cuda
        fi
    fi

    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME="/usr/local/cuda"
        export PATH="/usr/local/cuda/bin:${PATH}"
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
    fi
}

# 获取 APT 官方源中可用的 CUDA Toolkit 版本列表 (如 13.2 13.0 12.8 12.6)
get_available_cuda_packages() {
    apt-cache pkgnames 2>/dev/null | grep -E '^cuda-nvcc-[0-9]+-[0-9]+$' | sed 's/cuda-nvcc-//' | tr '-' '.' | sort -V -r
}

# 卸载并彻底清理旧版 CUDA 软件包（用户主动选择清理时调用）
purge_old_cuda_packages() {
    echo -e "${CYAN}正在扫描并清理系统中残留的旧版本 CUDA 软件包...${NC}"
    local old_pkgs=$(dpkg -l 2>/dev/null | awk '/^ii/ {print $2}' | grep -E '^(cuda|libcu|nsight)' | grep -E -- '-(12-[0-7]|11-[0-9]|10-[0-9])' || echo "")
    
    if [ -n "$old_pkgs" ]; then
        echo -e "${YELLOW}检测到以下旧版本 CUDA 软件包，正在通过 apt 彻底卸载以释放空间：${NC}"
        echo "$old_pkgs" | tr '\n' ' '
        echo -e "\n"
        run_as_root apt-get purge -y $old_pkgs
        run_as_root apt-get autoremove -y --purge
        echo -e "${GREEN}✔ 旧版本 CUDA 软件包卸载完成！${NC}"
    else
        echo -e "${GREEN}✔ 系统中未发现冗余的早期 CUDA 软件包。${NC}"
    fi

    for old_dir in /usr/local/cuda-12.[0-7] /usr/local/cuda-11.*; do
        if [ -d "$old_dir" ]; then
            echo -e "${CYAN}清理旧残留目录: ${old_dir}${NC}"
            run_as_root rm -rf "$old_dir"
        fi
    done

    sync_and_activate_highest_cuda
}

# 打印标题
print_header() {
    echo -e "${CYAN}==============================================================================${NC}"
    echo -e "${BOLD}${MAGENTA}       AnimeJaNai-Inference 交互式环境检测与部署向导${NC}"
    echo -e "${CYAN}==============================================================================${NC}"
    echo -e " 项目根目录: ${BOLD}${PROJECT_ROOT:-$(pwd)}${NC}"
    echo -e " 系统时间:   $(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${CYAN}------------------------------------------------------------------------------${NC}"
}

# 步骤 1: 检查 NVIDIA 驱动（如果未安装则强制退出）
check_nvidia_driver() {
    echo -e "\n${BOLD}[1/9] 检查 NVIDIA GPU 驱动...${NC}"
    if ! command -v nvidia-smi &>/dev/null; then
        echo -e "${RED}✖ 未找到 nvidia-smi 命令！${NC}"
        echo -e "${YELLOW}说明：通常情况下 GPU 容器（如 AutoDL）或宿主机应预装驱动。${NC}"
        echo -e "${YELLOW}如果这是云容器，请在实例控制台选择附带驱动与 CUDA 的镜像模板；如果是实体机，请先安装 nvidia-driver。${NC}"
        echo -e "${RED}[致命错误] 驱动未安装，脚本退出。${NC}"
        exit 1
    fi

    if ! nvidia-smi &>/dev/null; then
        echo -e "${RED}✖ nvidia-smi 执行失败，无法与 NVIDIA 驱动内核模块通信！${NC}"
        echo -e "${RED}[致命错误] 驱动异常，脚本退出。${NC}"
        exit 1
    fi

    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
    CUDA_MAX_VER=$(nvidia-smi | grep -o "CUDA Version: [0-9.]*" | head -n 1 | awk '{print $3}')
    
    echo -e "${GREEN}✔ NVIDIA 驱动正常！${NC}"
    echo -e "  - GPU 型号:       ${BOLD}${GPU_NAME}${NC}"
    echo -e "  - 驱动版本:       ${BOLD}${DRIVER_VER}${NC}"
    echo -e "  - 支持最高 CUDA:  ${BOLD}${CUDA_MAX_VER}${NC}"
}

# 检查系统信息
check_os_info() {
    echo -e "\n${BOLD}[2/9] 检查操作系统与发行版...${NC}"
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo -e "  - 发行版:         ${PRETTY_NAME:-$NAME}"
        echo -e "  - 系统架构:       $(uname -m)"
        if [[ "$ID" != "ubuntu" && "$ID_LIKE" != *"ubuntu"* && "$ID_LIKE" != *"debian"* ]]; then
            echo -e "${YELLOW}  ! 警告: 当前系统非 Ubuntu/Debian，部分 apt 安装步骤可能需要适配。${NC}"
        fi
    else
        echo -e "  - 系统架构:       $(uname -m)"
    fi
}

# 检查 CUDA Toolkit 并自动切换到最新版本
check_cuda_toolkit() {
    echo -e "\n${BOLD}[3/9] 检查 CUDA Toolkit (nvcc)...${NC}"
    
    # 自动同步软链接至最高版本
    sync_and_activate_highest_cuda

    local all_cuda_dirs=($(find /usr/local -maxdepth 1 -type d -name "cuda-[0-9]*" 2>/dev/null | sort -V))
    local active_target="$(readlink -f /usr/local/cuda 2>/dev/null || echo "")"
    
    local nvcc_bin="/usr/local/cuda/bin/nvcc"
    if [ ! -x "$nvcc_bin" ]; then
        nvcc_bin="$(command -v nvcc || echo "")"
    fi

    if [ -n "$nvcc_bin" ] && [ -x "$nvcc_bin" ]; then
        NVCC_VER=$("$nvcc_bin" --version | grep "release" | sed -E 's/.*release ([0-9]+\.[0-9]+).*/\1/')
        echo -e "${GREEN}✔ 当前激活 CUDA Toolkit: ${BOLD}${NVCC_VER}${NC} (${nvcc_bin})"
        if [ -n "$active_target" ]; then
            echo -e "  - 软链接路径:     /usr/local/cuda -> ${active_target}"
        fi
        
        # 显示共存版本
        if [ ${#all_cuda_dirs[@]} -gt 1 ]; then
            local ver_list=()
            for d in "${all_cuda_dirs[@]}"; do
                ver_list+=("$(basename "$d" | sed 's/cuda-//')")
            done
            echo -e "  - 系统共存版本:   ${BOLD}${ver_list[*]}${NC} (已激活最高版本 ${NVCC_VER})"
        fi
        return 0
    else
        echo -e "${YELLOW}✖ 未找到可用的 nvcc (CUDA Toolkit 未安装或未正确链接)。${NC}"
        return 1
    fi
}

# 检查 TensorRT
check_tensorrt() {
    echo -e "\n${BOLD}[4/9] 检查 TensorRT...${NC}"
    local trt_ok=0

    if command -v trtexec &>/dev/null; then
        TRT_VER=$(trtexec --help 2>&1 | head -n 2 | grep -o "TensorRT v[0-9]*" || echo "已安装")
        echo -e "${GREEN}✔ 已安装 trtexec 工具: ${BOLD}${TRT_VER}${NC}"
        trt_ok=1
    elif [ -x "/usr/src/tensorrt/bin/trtexec" ]; then
        echo -e "${GREEN}✔ 找到 trtexec: /usr/src/tensorrt/bin/trtexec${NC}"
        trt_ok=1
    else
        echo -e "${YELLOW}✖ 未在 PATH 找到 trtexec 工具。${NC}"
    fi

    if [ -f "/usr/include/NvInfer.h" ] || [ -f "/usr/include/x86_64-linux-gnu/NvInfer.h" ] || [ -f "$HOME/sdk/tensorrt/usr/include/NvInfer.h" ]; then
        echo -e "${GREEN}✔ 找到 NvInfer.h 头文件${NC}"
    else
        echo -e "${YELLOW}✖ 未检测到 TensorRT C++ 开发头文件 (NvInfer.h)${NC}"
        trt_ok=0
    fi

    if ldconfig -p 2>/dev/null | grep -q "libnvinfer.so" || ls /usr/lib/x86_64-linux-gnu/libnvinfer.so* &>/dev/null || [ -d "$HOME/sdk/tensorrt" ]; then
        echo -e "${GREEN}✔ 找到 TensorRT 动态库 (libnvinfer.so)${NC}"
    else
        echo -e "${YELLOW}✖ 未在动态链接器路径找到 libnvinfer.so${NC}"
        trt_ok=0
    fi

    return $((1 - trt_ok))
}

# 检查 FFmpeg (BtbN Shared Build)
check_ffmpeg() {
    echo -e "\n${BOLD}[5/9] 检查 FFmpeg 及开发库...${NC}"
    local ffmpeg_ok=1

    if command -v ffmpeg &>/dev/null; then
        FFMPEG_VER=$(ffmpeg -version 2>&1 | head -n 1)
        echo -e "${GREEN}✔ FFmpeg 命令可用: ${BOLD}${FFMPEG_VER}${NC}"
    else
        echo -e "${YELLOW}✖ 未在 PATH 中找到 ffmpeg 命令。${NC}"
        ffmpeg_ok=0
    fi

    export PKG_CONFIG_PATH="${FFMPEG_PC_DIR}:${PKG_CONFIG_PATH:-}"
    if pkg-config --exists libavformat libavcodec libavutil libavfilter libswscale 2>/dev/null; then
        AVFORMAT_VER=$(pkg-config --modversion libavformat)
        AVCODEC_VER=$(pkg-config --modversion libavcodec)
        echo -e "${GREEN}✔ pkg-config 成功识别 libav* 开发库 (libavformat: ${AVFORMAT_VER}, libavcodec: ${AVCODEC_VER})${NC}"
    else
        echo -e "${YELLOW}✖ pkg-config 未能找到完整的 libav* 开发库 (aji_encode 编译依赖)。${NC}"
        ffmpeg_ok=0
    fi

    return $((1 - ffmpeg_ok))
}

# 检查 NVENC 硬件编码及容器多卡补丁
check_nvenc_and_patch() {
    echo -e "\n${BOLD}[6/9] 检查 NVENC 编码器与容器补丁 (libnvenc_fix.so)...${NC}"
    
    if [ -f "$NVENC_FIX_SO" ]; then
        echo -e "${GREEN}✔ 已安装 NVENC 容器多卡修复补丁: ${BOLD}${NVENC_FIX_SO}${NC}"
    else
        echo -e "${YELLOW}ℹ 未安装 NVENC 容器修复补丁 (${NVENC_FIX_SO})${NC}"
    fi

    local raw_output=""
    local fix_output=""

    raw_output=$(ffmpeg -f lavfi -i nullsrc=s=256x256:d=0.1 -c:v hevc_nvenc -f null - 2>&1 || true)
    if echo "$raw_output" | grep -q "Output #0, null"; then
        echo -e "${GREEN}✔ NVENC (hevc_nvenc) 硬件编码器原生运行正常！${NC}"
        return 0
    fi

    if [ -f "$NVENC_FIX_SO" ]; then
        fix_output=$(LD_PRELOAD="$NVENC_FIX_SO" ffmpeg -f lavfi -i nullsrc=s=256x256:d=0.1 -c:v hevc_nvenc -f null - 2>&1 || true)
        if echo "$fix_output" | grep -q "Output #0, null"; then
            echo -e "${GREEN}✔ NVENC (hevc_nvenc) 在加载 ${NVENC_FIX_SO} 补丁后工作正常！${NC}"
            return 0
        fi
    fi

    echo -e "${RED}✖ NVENC 硬件编码器测试未通过！详细诊断信息如下：${NC}"
    local sample_err=$(echo "${fix_output:-$raw_output}" | grep -E -i "nvenc|unsupported|device|driver|cannot load|error" | head -n 4)
    if [ -n "$sample_err" ]; then
        echo -e "${YELLOW}------------------------------------------------------------------------------"
        echo "$sample_err"
        echo -e "------------------------------------------------------------------------------${NC}"
    fi

    if echo "${fix_output:-$raw_output}" | grep -qi "required nvenc API version"; then
        echo -e "${CYAN}[诊断建议] FFmpeg 构建需要的 NVENC API 版本高于当前显卡驱动。${NC}"
        echo -e "${CYAN}           解决办法：请运行选项 [5]，切换安装 FFmpeg n8.1 版本构建！${NC}"
    elif echo "${fix_output:-$raw_output}" | grep -qi "cannot load libnvidia-encode"; then
        echo -e "${CYAN}[诊断建议] 容器内缺少 libnvidia-encode.so 库（容器启动时未开放 video capability）。${NC}"
    elif echo "${fix_output:-$raw_output}" | grep -qi -E "unsupported device|no capable devices"; then
        echo -e "${CYAN}[诊断建议] 驱动 570+ 多卡容器枚举问题，请在菜单选择 [6] 重新编译安装修复补丁。${NC}"
    fi

    return 1
}

# 检查基础编译工具链
check_build_tools() {
    echo -e "\n${BOLD}[7/9] 检查编译工具链 (cmake, gcc, g++, pkg-config, git)...${NC}"
    local all_ok=1
    for tool in cmake gcc g++ pkg-config git; do
        if command -v "$tool" &>/dev/null; then
            echo -e "  - $tool: ${GREEN}✔ $($tool --version 2>&1 | head -n 1)${NC}"
        else
            echo -e "  - $tool: ${RED}✖ 未安装${NC}"
            all_ok=0
        fi
    done
    return $((1 - all_ok))
}

# 检查 Python 及虚拟环境 (venv)
check_python_venv() {
    echo -e "\n${BOLD}[8/9] 检查 Python 与虚拟环境 (venv)...${NC}"
    if command -v python3 &>/dev/null; then
        PY_VER=$(python3 --version)
        echo -e "${GREEN}✔ 系统 Python3: ${BOLD}${PY_VER}${NC}"
    else
        echo -e "${RED}✖ 未安装 python3${NC}"
        return 1
    fi

    if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
        echo -e "${GREEN}✔ 检测到已存在的 Python 虚拟环境: ${BOLD}${VENV_DIR}${NC}"
        return 0
    else
        echo -e "${YELLOW}ℹ 虚拟环境尚未创建 (${VENV_DIR})${NC}"
        return 2
    fi
}

# 检查 AnimeJaNai 构建产物及模型
check_build_and_models() {
    echo -e "\n${BOLD}[9/9] 检查 AnimeJaNai 构建产物与模型状态...${NC}"
    
    if [ -f "${PROJECT_ROOT}/build/libaji.so" ] && [ -f "${PROJECT_ROOT}/build/libaji_trt.so" ] && [ -f "${PROJECT_ROOT}/build/aji_encode" ]; then
        echo -e "${GREEN}✔ 项目构建产物齐全 (${PROJECT_ROOT}/build/libaji.so, libaji_trt.so, aji_encode)${NC}"
    else
        echo -e "${YELLOW}ℹ 项目尚未完整编译 (缺少 build/aji_encode 或 libaji*.so)${NC}"
    fi

    if [ -d "$MODELS_DIR" ]; then
        local onnx_count=$(find "$MODELS_DIR" -maxdepth 1 -name "*.onnx" 2>/dev/null | wc -l)
        local engine_count=$(find "$MODELS_DIR" -maxdepth 1 -name "*.engine" 2>/dev/null | wc -l)
        echo -e "  - 模型目录 (${MODELS_DIR}): 找到 ${onnx_count} 个 ONNX 模型，${engine_count} 个 Engine 文件"
        if [ "$onnx_count" -gt 0 ]; then
            for m in "$MODELS_DIR"/*.onnx; do
                [ -f "$m" ] && echo -e "    * ONNX: $(basename "$m") ($(du -h "$m" | cut -f1))"
            done
        fi
        if [ "$engine_count" -gt 0 ]; then
            for e in "$MODELS_DIR"/*.engine; do
                [ -f "$e" ] && echo -e "    * Engine: $(basename "$e") ($(du -h "$e" | cut -f1))"
            done
        fi
    else
        echo -e "  - 模型目录 (${MODELS_DIR}): 尚未创建"
    fi

    if [ -f "${PROJECT_ROOT}/example.mkv" ]; then
        echo -e "${GREEN}✔ 根目录已就绪测试视频: ${PROJECT_ROOT}/example.mkv ($(du -h "${PROJECT_ROOT}/example.mkv" | cut -f1))${NC}"
    else
        echo -e "${YELLOW}ℹ 项目目录下未找到 example.mkv (测试功能需要此文件)${NC}"
    fi
}

# 综合环境诊断
diagnose_all() {
    init_project_root
    print_header
    check_nvidia_driver
    check_os_info
    check_cuda_toolkit || true
    check_tensorrt || true
    check_ffmpeg || true
    check_nvenc_and_patch || true
    check_build_tools || true
    check_python_venv || true
    check_build_and_models || true
    echo -e "\n${CYAN}==============================================================================${NC}"
    echo -e "${BOLD}环境自检完成！${NC}"
    echo -e "${CYAN}==============================================================================${NC}"
}

# 步骤 A: AutoDL 旧 keyring 清理与 NVIDIA 官方网络源配置
setup_nvidia_network_repo() {
    print_header
    echo -e "${BOLD}${MAGENTA}[步骤] 清理 AutoDL 旧 keyring 并配置 NVIDIA 官方 Network 仓库${NC}\n"
    
    check_nvidia_driver

    echo -e "${CYAN}1. 正在清理旧版本 cuda-keyring 及潜在的 APT Pin 锁定配置...${NC}"
    run_as_root dpkg -P cuda-keyring 2>/dev/null || true
    run_as_root apt-get purge -y cuda-keyring 2>/dev/null || true
    run_as_root rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia-cuda*.list
    run_as_root rm -f /etc/apt/preferences.d/*cuda* /etc/apt/preferences.d/*nvidia*

    echo -e "\n${CYAN}2. 正在下载并安装 NVIDIA 官方最新 cuda-keyring (Ubuntu 24.04)...${NC}"
    local tmp_deb="/tmp/cuda-keyring_1.1-1_all.deb"
    wget -q --show-progress -O "$tmp_deb" "$CUDA_KEYRING_URL"
    if [ $? -ne 0 ] || [ ! -s "$tmp_deb" ]; then
        echo -e "${RED}[错误] 下载 cuda-keyring 失败，请检查网络连接！${NC}"
        rm -f "$tmp_deb"
        return 1
    fi

    run_as_root dpkg -i "$tmp_deb"
    rm -f "$tmp_deb"

    echo -e "\n${CYAN}3. 配置 APT Pin 策略（保护系统驱动避免被意外覆盖）...${NC}"
    cat << 'EOF' | run_as_root tee /etc/apt/preferences.d/cuda-repository-pin-600 > /dev/null
Package: nsight-compute* nsight-systems*
Pin: origin developer.download.nvidia.com
Pin-Priority: -1
EOF

    echo -e "\n${CYAN}4. 正在执行 apt-get update 刷新软件包缓存...${NC}"
    run_as_root apt-get update

    echo -e "\n${GREEN}✔ NVIDIA 官方 Network 源配置完成！${NC}"
}

# 步骤 B: 动态匹配显卡驱动支持上限安装最高 CUDA Toolkit 与 TensorRT (支持 12.8 / 13.x 并保留已有版本)
install_cuda_and_tensorrt() {
    print_header
    echo -e "${BOLD}${MAGENTA}[步骤] 自动检测显卡驱动并安装最高匹配 CUDA Toolkit 与 TensorRT${NC}\n"

    # 1. 驱动检查并获取最高支持 CUDA 版本
    check_nvidia_driver

    echo -e "\n${CYAN}1. 正在扫描 NVIDIA 官方源中可安装的 CUDA Toolkit 版本...${NC}"
    local avail_vers=($(get_available_cuda_packages))
    
    # 过滤出不超过驱动支持上限的版本
    local valid_vers=()
    for v in "${avail_vers[@]}"; do
        if [ "$(printf '%s\n%s' "$v" "$CUDA_MAX_VER" | sort -V | tail -n 1)" = "$CUDA_MAX_VER" ]; then
            valid_vers+=("$v")
        fi
    done

    # 备选回退
    if [ ${#valid_vers[@]} -eq 0 ]; then
        valid_vers=("12.8")
    fi

    local target_cuda="${valid_vers[0]}"

    echo -e "当前显卡驱动支持最高 CUDA: ${BOLD}${CUDA_MAX_VER}${NC}"
    echo -e "官方软件源支持的可选版本:  ${BOLD}${valid_vers[*]}${NC}"
    echo -e "\n请选择要安装的 CUDA Toolkit 版本："
    for i in "${!valid_vers[@]}"; do
        local note=""
        [ $i -eq 0 ] && note=" (官方源最高适配版本 - 推荐)"
        echo -e "  ${BOLD}$((i+1)))${NC} CUDA ${valid_vers[$i]}${note}"
    done
    echo -e "  ${BOLD}$(( ${#valid_vers[@]} + 1 )))${NC} 手动输入特定版本 (如 13.2 / 12.8)"
    
    read -rp "请输入选项 [1-$(( ${#valid_vers[@]} + 1 )), 默认 1]: " v_choice
    v_choice=${v_choice:-1}

    if [ "$v_choice" -le "${#valid_vers[@]}" ] && [ "$v_choice" -ge 1 ]; then
        target_cuda="${valid_vers[$((v_choice-1))]}"
    elif [ "$v_choice" -eq "$(( ${#valid_vers[@]} + 1 ))" ]; then
        read -rp "请输入目标 CUDA 版本号 (如 13.2 或 12.8): " manual_ver
        target_cuda="${manual_ver:-$target_cuda}"
    fi

    local cuda_pkg_suffix="$(echo "$target_cuda" | tr '.' '-')"
    echo -e "\n${GREEN}✔ 选择目标版本: CUDA ${target_cuda} (包后缀: ${cuda_pkg_suffix})${NC}"

    # 检查并询问是否保留已有的 CUDA 版本
    local existing_cuda=($(find /usr/local -maxdepth 1 -type d -name "cuda-[0-9]*" 2>/dev/null | sort -V))
    if [ ${#existing_cuda[@]} -gt 0 ]; then
        echo -e "\n检测到当前系统已存在以下 CUDA 版本: ${BOLD}${existing_cuda[*]}${NC}"
        echo -e "是否保留已有版本？(默认保留，将多版本共存并自动激活最高版本)"
        read -rp "是否保留已有 CUDA 版本？[Y/n, 默认 Y]: " keep_old
        keep_old=${keep_old:-Y}
        if [[ ! "$keep_old" =~ ^[Yy]$ ]]; then
            purge_old_cuda_packages
        fi
    fi

    echo -e "\n${CYAN}2. 正在通过 NVIDIA 官方源安装 CUDA ${target_cuda} 开发套件与 TensorRT...${NC}"
    run_as_root apt-get update
    run_as_root apt-get install -y --no-install-recommends \
        tensorrt \
        tensorrt-dev \
        "cuda-nvcc-${cuda_pkg_suffix}" \
        "cuda-cudart-dev-${cuda_pkg_suffix}" \
        "cuda-driver-dev-${cuda_pkg_suffix}"

    # 若指定版本包不存在则自动回退 12.8
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}提示: 源中若暂未上线 cuda-nvcc-${cuda_pkg_suffix}，尝试安装稳定版 cuda-nvcc-12-8 ...${NC}"
        run_as_root apt-get install -y --no-install-recommends \
            tensorrt \
            tensorrt-dev \
            cuda-nvcc-12-8 \
            cuda-cudart-dev-12-8 \
            cuda-driver-dev-12-8
    fi

    # 自动同步并激活最高版本 CUDA
    echo -e "\n${CYAN}3. 自动配置并激活 /usr/local/cuda 环境变量...${NC}"
    sync_and_activate_highest_cuda

    cat << 'EOF' | run_as_root tee /etc/profile.d/cuda.sh > /dev/null
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EOF
    run_as_root chmod +x /etc/profile.d/cuda.sh

    if ! grep -q "CUDA_HOME=/usr/local/cuda" "$HOME/.bashrc" 2>/dev/null; then
        echo 'export CUDA_HOME=/usr/local/cuda' >> "$HOME/.bashrc"
        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> "$HOME/.bashrc"
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> "$HOME/.bashrc"
    fi

    echo -e "\n${GREEN}✔ CUDA Toolkit 与 TensorRT 安装就绪！${NC}"
    check_cuda_toolkit || true
    check_tensorrt || true
}

# 步骤 C: 安装编译工具与最新 BtbN Shared FFmpeg (支持 n8.1 与 master 选择)
install_build_tools_and_ffmpeg() {
    print_header
    echo -e "${BOLD}${MAGENTA}[步骤] 安装编译工具与最新 BtbN FFmpeg Shared 构建${NC}\n"

    echo -e "${CYAN}1. 正在通过 apt 安装基础编译链 (build-essential, cmake, pkg-config, git, p7zip-full, wget, tar, xz-utils)...${NC}"
    run_as_root apt-get update
    run_as_root apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        pkg-config \
        git \
        p7zip-full \
        wget \
        curl \
        tar \
        xz-utils

    echo -e "\n请选择 FFmpeg BtbN 构建版本："
    echo -e "  ${BOLD}1)${NC} ${GREEN}FFmpeg n8.1 Shared (强烈推荐：完美适配当前 570~595 系列显卡驱动与 NVENC API)${NC}"
    echo -e "  ${BOLD}2)${NC} FFmpeg master 最新构建 (实验性/最新特性，要求 610+ 驱动)"
    read -rp "请输入选项 [1-2, 默认 1]: " ffmpeg_choice
    ffmpeg_choice=${ffmpeg_choice:-1}

    local ffmpeg_download_url="$FFMPEG_TAR_N81_URL"
    if [ "$ffmpeg_choice" -eq 2 ]; then
        ffmpeg_download_url="$FFMPEG_TAR_MASTER_URL"
    fi

    echo -e "\n${CYAN}2. 正在下载并安装 BtbN FFmpeg Linux64 GPL Shared 版本...${NC}"
    echo -e "下载地址: ${ffmpeg_download_url}"
    
    local tmp_tar="/tmp/ffmpeg-btbn-shared.tar.xz"
    wget -q --show-progress -O "$tmp_tar" "$ffmpeg_download_url"
    if [ $? -ne 0 ] || [ ! -s "$tmp_tar" ]; then
        echo -e "${RED}[错误] 下载 FFmpeg 压缩包失败，请检查网络！${NC}"
        rm -f "$tmp_tar"
        return 1
    fi

    echo -e "正在解压并部署到 ${FFMPEG_INSTALL_DIR} ..."
    run_as_root rm -rf "$FFMPEG_INSTALL_DIR"
    run_as_root mkdir -p "$FFMPEG_INSTALL_DIR"
    run_as_root tar -xJf "$tmp_tar" -C "$FFMPEG_INSTALL_DIR" --strip-components=1
    rm -f "$tmp_tar"

    echo -e "\n${CYAN}3. 修正 pkg-config 描述文件中的 prefix 路径...${NC}"
    if [ -d "${FFMPEG_PC_DIR}" ]; then
        for pc in "${FFMPEG_PC_DIR}"/*.pc; do
            [ -f "$pc" ] && run_as_root sed -i "s|^prefix=.*|prefix=${FFMPEG_INSTALL_DIR}|" "$pc"
        done
    else
        echo -e "${RED}[警告] 未找到 FFmpeg pkgconfig 目录: ${FFMPEG_PC_DIR}${NC}"
    fi

    echo -e "\n${CYAN}4. 配置系统动态链接库 (ldconfig) 与全局软链接...${NC}"
    echo "${FFMPEG_INSTALL_DIR}/lib" | run_as_root tee /etc/ld.so.conf.d/ffmpeg.conf > /dev/null
    run_as_root ldconfig

    run_as_root ln -sf "${FFMPEG_INSTALL_DIR}/bin/ffmpeg" /usr/local/bin/ffmpeg
    run_as_root ln -sf "${FFMPEG_INSTALL_DIR}/bin/ffprobe" /usr/local/bin/ffprobe

    cat << EOF | run_as_root tee /etc/profile.d/ffmpeg.sh > /dev/null
export PKG_CONFIG_PATH="${FFMPEG_PC_DIR}:\${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="${FFMPEG_INSTALL_DIR}/lib:\${LD_LIBRARY_PATH:-}"
export PATH="${FFMPEG_INSTALL_DIR}/bin:\$PATH"
EOF
    run_as_root chmod +x /etc/profile.d/ffmpeg.sh

    if ! grep -q "PKG_CONFIG_PATH.*${FFMPEG_PC_DIR}" "$HOME/.bashrc" 2>/dev/null; then
        echo "export PKG_CONFIG_PATH=\"${FFMPEG_PC_DIR}:\${PKG_CONFIG_PATH:-}\"" >> "$HOME/.bashrc"
        echo "export LD_LIBRARY_PATH=\"${FFMPEG_INSTALL_DIR}/lib:\${LD_LIBRARY_PATH:-}\"" >> "$HOME/.bashrc"
        echo "export PATH=\"${FFMPEG_INSTALL_DIR}/bin:\$PATH\"" >> "$HOME/.bashrc"
    fi

    export PKG_CONFIG_PATH="${FFMPEG_PC_DIR}:${PKG_CONFIG_PATH:-}"
    export LD_LIBRARY_PATH="${FFMPEG_INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"
    export PATH="${FFMPEG_INSTALL_DIR}/bin:$PATH"

    echo -e "\n${GREEN}✔ BtbN FFmpeg 及编译工具链安装配置完成！${NC}"
    check_ffmpeg || true
    check_nvenc_and_patch || true
}

# 步骤 D: 下载并编译 NVENC 容器多卡枚举修复补丁 (libnvenc_fix.so)
install_nvenc_fix() {
    print_header
    echo -e "${BOLD}${MAGENTA}[步骤] 下载并编译 NVENC 容器多卡修复补丁 (libnvenc_fix.so)${NC}\n"
    echo -e "${CYAN}背景说明：${NC}"
    echo -e "在 NVIDIA 驱动 ≥ 570 的多 GPU 宿主机（如 AutoDL 等云容器）中，"
    echo -e "容器内的 libnvidia-encode 在初始化时会查询宿主机全部 GPU 并尝试 peer-init。"
    echo -e "因容器未挂载其他卡设备节点，会导致 NVENC 报错：${YELLOW}unsupported device / No capable devices found${NC}。"
    echo -e "本补丁通过 LD_PRELOAD 拦截 ioctl，重写 GPU 列表使得 NVENC 在容器内恢复正常。\n"

    if ! command -v gcc &>/dev/null; then
        echo -e "${CYAN}正在安装 gcc 编译器...${NC}"
        run_as_root apt-get update && run_as_root apt-get install -y gcc build-essential
    fi

    local tmp_c="/tmp/nvenc_fix.c"
    echo -e "${CYAN}1. 正在从 GitHub 获取 nvenc_fix.c 补丁源码...${NC}"
    
    if ! curl -fsSL --retry 2 "$NVENC_FIX_SRC_URL" -o "$tmp_c"; then
        echo -e "${YELLOW}主源下载失败，尝试 GitHub 加速镜像...${NC}"
        curl -fsSL --retry 2 "$NVENC_FIX_SRC_MIRROR" -o "$tmp_c" || true
    fi

    if [ ! -s "$tmp_c" ]; then
        echo -e "${RED}[错误] 无法从 GitHub 下载 nvenc_fix.c，请检查网络！${NC}"
        return 1
    fi

    echo -e "\n${CYAN}2. 正在编译 ${NVENC_FIX_SO} ...${NC}"
    run_as_root gcc -shared -fPIC -O2 -o "$NVENC_FIX_SO" "$tmp_c" -ldl
    local compile_res=$?
    rm -f "$tmp_c"

    if [ $compile_res -eq 0 ] && [ -f "$NVENC_FIX_SO" ]; then
        run_as_root chmod 755 "$NVENC_FIX_SO"
        echo -e "\n${GREEN}✔ 补丁编译成功并安装到: ${BOLD}${NVENC_FIX_SO}${NC}"
        echo -e "\n${CYAN}3. 验证 NVENC 编码器状态...${NC}"
        check_nvenc_and_patch || true
    else
        echo -e "\n${RED}[错误] gcc 编译 libnvenc_fix.so 失败！${NC}"
        return 1
    fi
}

# 步骤 E: 创建并配置 Python 虚拟环境 (venv)
setup_python_venv() {
    init_project_root
    print_header
    echo -e "${BOLD}${MAGENTA}[步骤] 创建并配置 Python 虚拟环境 (venv)${NC}\n"

    echo -e "${CYAN}1. 检查并安装 python3, python3-venv, python3-pip, p7zip-full...${NC}"
    run_as_root apt-get update
    run_as_root apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        p7zip-full

    echo -e "\n${CYAN}2. 创建虚拟环境到 ${VENV_DIR} ...${NC}"
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
    fi

    echo -e "\n${CYAN}3. 升级 venv 中的 pip 并安装基础支持包 (numpy, onnx, tqdm, py7zr)...${NC}"
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install --no-cache-dir numpy onnx tqdm py7zr

    echo -e "\n${GREEN}✔ Python 虚拟环境创建成功: ${BOLD}${VENV_DIR}${NC}"
    echo -e "  可通过以下命令进入虚拟环境: ${BOLD}source ${VENV_DIR}/bin/activate${NC}"
}

# 步骤 F: 编译 AnimeJaNai-Inference
build_project() {
    init_project_root
    sync_and_activate_highest_cuda
    print_header
    echo -e "${BOLD}${MAGENTA}[步骤] 编译 AnimeJaNai-Inference${NC}\n"

    if [ ! -f "${PROJECT_ROOT}/CMakeLists.txt" ]; then
        echo -e "${RED}[错误] 在 ${PROJECT_ROOT} 未找到 CMakeLists.txt！${NC}"
        return 1
    fi

    export PKG_CONFIG_PATH="${FFMPEG_PC_DIR}:${PKG_CONFIG_PATH:-}"
    export LD_LIBRARY_PATH="${FFMPEG_INSTALL_DIR}/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
    export PATH="/usr/local/cuda/bin:${FFMPEG_INSTALL_DIR}/bin:$PATH"
    
    local nvcc_path="/usr/local/cuda/bin/nvcc"
    if [ ! -x "$nvcc_path" ]; then
        nvcc_path="$(command -v nvcc || true)"
    fi

    if [ -z "$nvcc_path" ]; then
        echo -e "${RED}[错误] 未找到 nvcc，无法编译 CUDA 内核！请先安装 CUDA Toolkit。${NC}"
        return 1
    fi

    echo -e "${CYAN}1. 配置 CMake 构建工程...${NC}"
    echo -e "  - 源码目录:     ${PROJECT_ROOT}"
    echo -e "  - 构建目录:     ${PROJECT_ROOT}/build"
    echo -e "  - CUDACXX:      ${nvcc_path}"
    echo -e "  - AJI_TRT_ROOT: /usr"
    echo -e "  - CUDA_ARCH:    native (适配当前 GPU)"
    
    cd "$PROJECT_ROOT"
    CUDACXX="$nvcc_path" cmake -B "${PROJECT_ROOT}/build" -S "${PROJECT_ROOT}" \
        -DAJI_TRT_ROOT=/usr \
        -DCMAKE_CUDA_ARCHITECTURES=native \
        -DCMAKE_BUILD_TYPE=Release

    if [ $? -ne 0 ]; then
        echo -e "${RED}[错误] CMake 配置失败，请检查上方报错信息。${NC}"
        return 1
    fi

    echo -e "\n${CYAN}2. 正在执行多核编译 (cmake --build build -j$(nproc))...${NC}"
    cmake --build "${PROJECT_ROOT}/build" -j$(nproc)
    if [ $? -ne 0 ]; then
        echo -e "${RED}[错误] 编译失败！${NC}"
        return 1
    fi

    echo -e "\n${CYAN}3. 验证构建结果产物...${NC}"
    local missing=0
    for bin in "${PROJECT_ROOT}/build/libaji.so" "${PROJECT_ROOT}/build/libaji_trt.so" "${PROJECT_ROOT}/build/aji_harness" "${PROJECT_ROOT}/build/aji_encode"; do
        if [ -f "$bin" ]; then
            echo -e "  - $(basename "$bin"): ${GREEN}✔ $(ls -lh "$bin" | awk '{print $5}')${NC}"
        else
            echo -e "  - $(basename "$bin"): ${RED}✖ 缺失${NC}"
            missing=1
        fi
    done

    if [ $missing -eq 0 ]; then
        echo -e "\n${GREEN}✔ AnimeJaNai-Inference 编译完全成功！${NC}"
        return 0
    else
        echo -e "\n${YELLOW}⚠ 部分目标未生成，如果缺少 aji_encode 请检查 FFmpeg dev 是否正确配置。${NC}"
        return 1
    fi
}

# 下载辅助函数（带镜像回退）
download_model_file() {
    local target_path="$1"
    local primary_url="$2"
    local mirror_url="$3"
    local model_title="$4"

    if [ -s "$target_path" ]; then
        echo -e "${GREEN}✔ 已存在模型: ${model_title} ($(du -h "$target_path" | cut -f1))${NC}"
        return 0
    fi

    echo -e "${CYAN}正在下载: ${model_title} ...${NC}"
    echo -e "目标路径: ${target_path}"
    
    if wget -q --show-progress -O "$target_path" "$primary_url"; then
        if [ -s "$target_path" ]; then
            echo -e "${GREEN}✔ 下载成功: ${model_title}${NC}"
            return 0
        fi
    fi

    if [ -n "$mirror_url" ]; then
        echo -e "${YELLOW}主链接下载失败，正在尝试镜像加速链接...${NC}"
        rm -f "$target_path"
        if wget -q --show-progress -O "$target_path" "$mirror_url"; then
            if [ -s "$target_path" ]; then
                echo -e "${GREEN}✔ 镜像下载成功: ${model_title}${NC}"
                return 0
            fi
        fi
    fi

    echo -e "${RED}[错误] ${model_title} 下载失败！${NC}"
    rm -f "$target_path"
    return 1
}

# 批量下载并提取 AnimeJaNai V3.1 官方高清超分模型
download_animejanai_models() {
    local perf_dest="${MODELS_DIR}/performance.onnx"
    local bal_dest="${MODELS_DIR}/balanced.onnx"

    if [ -s "$perf_dest" ] && [ -s "$bal_dest" ]; then
        echo -e "${GREEN}✔ AnimeJaNai Performance 与 Balanced 模型均已存在！${NC}"
        return 0
    fi

    echo -e "${CYAN}正在下载 AnimeJaNai V3.1 官方模型资源包 (约 37MB)...${NC}"
    local tmp_7z="/tmp/animejanai_overlay.7z"
    local extract_dir="/tmp/animejanai_models_extracted"
    
    if ! wget -q --show-progress -O "$tmp_7z" "$ANIMEJANAI_OVERLAY_URL"; then
        echo -e "${YELLOW}主链接下载失败，正在尝试镜像加速链接...${NC}"
        wget -q --show-progress -O "$tmp_7z" "$ANIMEJANAI_OVERLAY_MIRROR" || true
    fi

    if [ ! -s "$tmp_7z" ]; then
        echo -e "${RED}[错误] 下载 AnimeJaNai 官方资源包失败，请检查网络！${NC}"
        rm -f "$tmp_7z"
        return 1
    fi

    echo -e "${CYAN}正在提取 Performance 与 Balanced ONNX 模型...${NC}"
    rm -rf "$extract_dir"
    mkdir -p "$extract_dir"

    if command -v 7z &>/dev/null; then
        7z e -y "$tmp_7z" -o"$extract_dir" "*SPANF3*" &>/dev/null || 7z x -y "$tmp_7z" -o"$extract_dir" &>/dev/null
    elif command -v 7zz &>/dev/null; then
        7zz e -y "$tmp_7z" -o"$extract_dir" "*SPANF3*" &>/dev/null || 7zz x -y "$tmp_7z" -o"$extract_dir" &>/dev/null
    else
        python3 -m pip install py7zr -q 2>/dev/null || "$VENV_DIR/bin/pip" install py7zr -q 2>/dev/null || true
        python3 -c "
import py7zr
with py7zr.SevenZipFile('$tmp_7z', mode='r') as z:
    z.extractall(path='$extract_dir')
" 2>/dev/null || "$VENV_DIR/bin/python3" -c "
import py7zr
with py7zr.SevenZipFile('$tmp_7z', mode='r') as z:
    z.extractall(path='$extract_dir')
" 2>/dev/null || true
    fi

    local found_perf=$(find "$extract_dir" -name "*Performance_SPANF3*b5f48*.onnx" 2>/dev/null | head -n 1)
    local found_bal=$(find "$extract_dir" -name "*Balanced_SPANF3*b8f64*.onnx" 2>/dev/null | head -n 1)

    if [ -s "$found_perf" ]; then
        cp -f "$found_perf" "$perf_dest"
        echo -e "${GREEN}✔ 成功获取 AnimeJaNai Performance (2x): ${perf_dest} ($(du -h "$perf_dest" | cut -f1))${NC}"
    fi

    if [ -s "$found_bal" ]; then
        cp -f "$found_bal" "$bal_dest"
        echo -e "${GREEN}✔ 成功获取 AnimeJaNai Balanced (2x): ${bal_dest} ($(du -h "$bal_dest" | cut -f1))${NC}"
    fi

    rm -f "$tmp_7z"
    rm -rf "$extract_dir"
}

# 单模型 Engine 构建函数
build_single_engine() {
    local onnx_path="$1"
    local opt_w="$2"
    local opt_h="$3"
    local engine_suffix="$4"

    if [ ! -s "$onnx_path" ]; then
        echo -e "${RED}[错误] ONNX 文件不存在或为空: ${onnx_path}${NC}"
        return 1
    fi

    local model_basename="$(basename "$onnx_path" .onnx)"
    local engine_path="${MODELS_DIR}/${model_basename}_${engine_suffix}.engine"

    # 动态获取 ONNX 模型的第一个输入节点名称 (默认为 input)
    local input_name="input"
    if [ -f "$VENV_DIR/bin/python3" ]; then
        local detected_name=$("$VENV_DIR/bin/python3" -c "
import onnx, sys
try:
    m = onnx.load('$onnx_path')
    print(m.graph.input[0].name)
except:
    print('input')
" 2>/dev/null || echo "input")
        [ -n "$detected_name" ] && input_name="$detected_name"
    fi

    # 检测 trtexec 是否支持 --fp16 参数 (TensorRT 11+ 已移除 --fp16 选项，默认为强类型模式)
    local fp16_arg=()
    if trtexec --help 2>&1 | grep -q -- "--fp16"; then
        fp16_arg=("--fp16")
    fi

    echo -e "\n${CYAN}==============================================================================${NC}"
    echo -e "正在调用 trtexec 为当前 GPU 构建 TensorRT Engine..."
    echo -e "  - 输入 ONNX:   ${onnx_path}"
    echo -e "  - 输入节点:     ${input_name}"
    echo -e "  - 目标优化:     ${opt_w}x${opt_h} (max/opt: 1x3x${opt_h}x${opt_w})"
    echo -e "  - 输出 Engine:  ${engine_path}"
    echo -e "构建大约需要 1-3 分钟，请稍候...\n"

    trtexec \
        --onnx="$onnx_path" \
        --minShapes="${input_name}:1x3x64x64" \
        --optShapes="${input_name}:1x3x${opt_h}x${opt_w}" \
        --maxShapes="${input_name}:1x3x${opt_h}x${opt_w}" \
        "${fp16_arg[@]}" \
        --skipInference \
        --saveEngine="$engine_path"

    if [ $? -eq 0 ] && [ -f "$engine_path" ]; then
        echo -e "\n${GREEN}✔ TensorRT Engine 构建成功！${NC}"
        echo -e "  Engine 路径: ${BOLD}${engine_path}${NC} ($(du -h "$engine_path" | cut -f1))"
        return 0
    else
        echo -e "\n${RED}[错误] trtexec 构建 Engine 失败！如果显存不足请关闭其他占用显存的进程。${NC}"
        return 1
    fi
}

# 步骤 G: 下载超分 ONNX 模型并构建 TensorRT Engine
download_and_build_engine() {
    init_project_root
    print_header
    echo -e "${BOLD}${MAGENTA}[步骤] 下载超分模型 (ONNX) 并构建 TensorRT Engine${NC}\n"

    mkdir -p "$MODELS_DIR"

    local perf_onnx="${MODELS_DIR}/performance.onnx"
    local balanced_onnx="${MODELS_DIR}/balanced.onnx"
    local anime6b_onnx="${MODELS_DIR}/realesrgan_anime6b.onnx"

    echo -e "请选择操作："
    echo -e "  ${BOLD}1)${NC} ${GREEN}一键获取全部 3 个常用默认模型${NC} (AnimeJaNai Perf + Balanced + Real-ESRGAN Anime 6B)"
    echo -e "  ${BOLD}2)${NC} 获取 AnimeJaNai V3.1 Performance (2x, 极速偏画质)"
    echo -e "  ${BOLD}3)${NC} 获取 AnimeJaNai V3.1 Balanced (2x, 均衡推荐)"
    echo -e "  ${BOLD}4)${NC} 下载 Real-ESRGAN Anime 6B (4x, 经典原版动漫模型)"
    echo -e "  ${BOLD}5)${NC} 自定义 ONNX 模型下载链接 / 本地已有路径"
    read -rp "请输入选项 [1-5, 默认 1]: " model_choice
    model_choice=${model_choice:-1}

    local target_onnx_list=()

    case "$model_choice" in
        1)
            echo -e "\n${CYAN}>>> 正在获取全部 3 个常用超分模型到 ${MODELS_DIR} ...${NC}\n"
            download_animejanai_models
            download_model_file "$anime6b_onnx" "$DEFAULT_ONNX_REALESRGAN_ANIME6B_URL" "$DEFAULT_ONNX_REALESRGAN_ANIME6B_MIRROR" "Real-ESRGAN Anime 6B (4x)"
            target_onnx_list=("$perf_onnx" "$balanced_onnx" "$anime6b_onnx")
            ;;
        2)
            download_animejanai_models
            target_onnx_list=("$perf_onnx")
            ;;
        3)
            download_animejanai_models
            target_onnx_list=("$balanced_onnx")
            ;;
        4)
            download_model_file "$anime6b_onnx" "$DEFAULT_ONNX_REALESRGAN_ANIME6B_URL" "$DEFAULT_ONNX_REALESRGAN_ANIME6B_MIRROR" "Real-ESRGAN Anime 6B (4x)"
            target_onnx_list=("$anime6b_onnx")
            ;;
        5)
            read -rp "请输入 ONNX 下载 URL 或本地绝对路径: " custom_input
            if [ -f "$custom_input" ]; then
                target_onnx_list=("$custom_input")
            else
                local custom_dest="${MODELS_DIR}/custom_model.onnx"
                download_model_file "$custom_dest" "$custom_input" "" "Custom Model"
                target_onnx_list=("$custom_dest")
            fi
            ;;
        *)
            echo -e "${RED}无效输入，返回。${NC}"
            return 1
            ;;
    esac

    echo -e "\n请选择构建 Engine 的源视频优化分辨率 (optShapes/maxShapes)："
    echo -e "  ${BOLD}1)${NC} 1080p (1920x1080 -> 2x 超分至 4K / 4x 超分至 8K) [标准推荐]"
    echo -e "  ${BOLD}2)${NC} 720p  (1280x720  -> 2x 超分至 2K / 4x 超分至 4K)"
    echo -e "  ${BOLD}3)${NC} 自定义分辨率 (如 960x540 或非标)"
    echo -e "  ${BOLD}4)${NC} 暂不构建 Engine (仅下载模型)"
    read -rp "请输入选项 [1-4, 默认 1]: " res_choice
    res_choice=${res_choice:-1}

    local opt_h=1080
    local opt_w=1920
    local engine_suffix="1080p"
    case "$res_choice" in
        1)
            opt_h=1080
            opt_w=1920
            engine_suffix="1080p"
            ;;
        2)
            opt_h=720
            opt_w=1280
            engine_suffix="720p"
            ;;
        3)
            read -rp "请输入源视频宽度 (Width, 如 1920): " opt_w
            read -rp "请输入源视频高度 (Height, 如 1080): " opt_h
            engine_suffix="${opt_w}x${opt_h}"
            ;;
        4)
            echo -e "${GREEN}✔ 模型获取完成，已跳过 Engine 构建。${NC}"
            return 0
            ;;
    esac

    if [ ${#target_onnx_list[@]} -gt 1 ]; then
        echo -e "\n检测到已获取多个模型，请选择构建策略："
        echo -e "  ${BOLD}1)${NC} 为全部已获取模型构建 ${engine_suffix} Engine"
        echo -e "  ${BOLD}2)${NC} 仅为 AnimeJaNai Performance 构建 Engine (最快)"
        echo -e "  ${BOLD}3)${NC} 仅为 AnimeJaNai Balanced 构建 Engine"
        echo -e "  ${BOLD}4)${NC} 仅为 Real-ESRGAN Anime 6B 构建 Engine"
        read -rp "请输入选项 [1-4, 默认 1]: " build_strat
        build_strat=${build_strat:-1}

        case "$build_strat" in
            1)
                for onnx_item in "${target_onnx_list[@]}"; do
                    [ -s "$onnx_item" ] && build_single_engine "$onnx_item" "$opt_w" "$opt_h" "$engine_suffix"
                done
                ;;
            2) build_single_engine "$perf_onnx" "$opt_w" "$opt_h" "$engine_suffix" ;;
            3) build_single_engine "$balanced_onnx" "$opt_w" "$opt_h" "$engine_suffix" ;;
            4) build_single_engine "$anime6b_onnx" "$opt_w" "$opt_h" "$engine_suffix" ;;
        esac
    elif [ ${#target_onnx_list[@]} -eq 1 ]; then
        build_single_engine "${target_onnx_list[0]}" "$opt_w" "$opt_h" "$engine_suffix"
    fi
}

# 步骤 H: 截取 1 分钟 example.mkv 进行超分测试
run_test_clip() {
    init_project_root
    print_header
    echo -e "${BOLD}${MAGENTA}[步骤] 截取 1 分钟视频并运行超分测试${NC}\n"

    if [ ! -f "${PROJECT_ROOT}/build/aji_encode" ]; then
        echo -e "${RED}[错误] 未找到 ${PROJECT_ROOT}/build/aji_encode！请先执行编译步骤。${NC}"
        return 1
    fi

    local test_source="${PROJECT_ROOT}/example.mkv"
    if [ ! -f "$test_source" ]; then
        local found_mkv=$(find "$PROJECT_ROOT" -maxdepth 1 -name "*.mkv" ! -name "test_clip*" | head -n 1)
        echo -e "${YELLOW}未在 ${PROJECT_ROOT} 目录下找到默认的 example.mkv 文件！${NC}"
        if [ -n "$found_mkv" ]; then
            echo -e "检测到目录中存在视频文件: ${BOLD}${found_mkv}${NC}"
            read -rp "是否直接使用该视频作为测试源？[Y/n]: " use_found
            use_found=${use_found:-Y}
            if [[ "$use_found" =~ ^[Yy]$ ]]; then
                test_source="$found_mkv"
            fi
        fi

        if [ ! -f "$test_source" ]; then
            read -rp "请输入测试视频文件的相对或绝对路径: " manual_path
            if [ -f "$manual_path" ]; then
                test_source="$manual_path"
            else
                echo -e "${RED}[错误] 文件 ${manual_path} 不存在！${NC}"
                echo -e "${YELLOW}提示：请将待测视频命名为 example.mkv 放置在 ${PROJECT_ROOT}/ 目录下。${NC}"
                return 1
            fi
        fi
    fi

    echo -e "${GREEN}✔ 使用测试视频源: ${BOLD}${test_source}${NC}"

    export PATH="${FFMPEG_INSTALL_DIR}/bin:$PATH"
    if ! command -v ffprobe &>/dev/null; then
        echo -e "${RED}[错误] 未找到 ffprobe 命令！${NC}"
        return 1
    fi

    local src_width=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of default=noprint_wrappers=1:nokey=1 "$test_source" 2>/dev/null | head -n 1)
    local src_height=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of default=noprint_wrappers=1:nokey=1 "$test_source" 2>/dev/null | head -n 1)
    local src_codec=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "$test_source" 2>/dev/null | head -n 1)
    
    src_width=${src_width:-1920}
    src_height=${src_height:-1080}
    echo -e "  - 视频编码: ${src_codec:-未知}"
    echo -e "  - 原始分辨率: ${src_width} x ${src_height}"

    local engine_file=""
    local available_engines=($(find "$MODELS_DIR" -maxdepth 1 -name "*.engine" 2>/dev/null))
    if [ ${#available_engines[@]} -eq 0 ]; then
        echo -e "${YELLOW}未在 ${MODELS_DIR} 找到现成的 .engine 文件！${NC}"
        read -rp "请输入自定义 Engine 文件路径: " engine_file
        if [ ! -f "$engine_file" ]; then
            echo -e "${RED}[错误] Engine 文件不存在，请先构建 Engine！${NC}"
            return 1
        fi
    elif [ ${#available_engines[@]} -eq 1 ]; then
        engine_file="${available_engines[0]}"
        echo -e "${GREEN}✔ 自动匹配到 Engine: ${BOLD}${engine_file}${NC}"
    else
        echo -e "\n找到以下 Engine 文件，请选择："
        for i in "${!available_engines[@]}"; do
            echo -e "  ${BOLD}$((i+1)))${NC} ${available_engines[$i]}"
        done
        read -rp "请输入序号 [1-${#available_engines[@]}]: " engine_idx
        engine_idx=${engine_idx:-1}
        engine_file="${available_engines[$((engine_idx-1))]}"
    fi

    local clip_input="${PROJECT_ROOT}/test_clip_1min.mkv"
    local clip_output="${PROJECT_ROOT}/test_clip_1min_upscaled.mkv"
    local start_time="00:00:30"
    
    echo -e "\n${CYAN}正在从 ${test_source} 截取 1 分钟片段 (从 ${start_time} 开始，流拷贝无损快速提取)...${NC}"
    ffmpeg -y -ss "$start_time" -i "$test_source" -t 60 -c copy "$clip_input"
    if [ $? -ne 0 ] || [ ! -s "$clip_input" ]; then
        echo -e "${RED}[错误] 截取测试片段失败！${NC}"
        return 1
    fi
    echo -e "${GREEN}✔ 截取完成: ${clip_input} ($(du -h "$clip_input" | cut -f1))${NC}"

    local vcodec="hevc_nvenc"
    local preload_fix=""

    if [ -f "$NVENC_FIX_SO" ]; then
        preload_fix="$NVENC_FIX_SO"
    fi

    echo -e "\n正在检测硬件编码器 (NVENC)..."
    if ffmpeg -f lavfi -i nullsrc=s=256x256:d=0.1 -c:v hevc_nvenc -f null - &>/dev/null; then
        echo -e "${GREEN}✔ NVENC (hevc_nvenc) 硬件编码器支持正常！${NC}"
        vcodec="hevc_nvenc"
    elif [ -n "$preload_fix" ] && LD_PRELOAD="$preload_fix" ffmpeg -f lavfi -i nullsrc=s=256x256:d=0.1 -c:v hevc_nvenc -f null - &>/dev/null; then
        echo -e "${GREEN}✔ 检测到启用 ${NVENC_FIX_SO} 后 NVENC 工作正常！${NC}"
        vcodec="hevc_nvenc"
    else
        echo -e "${YELLOW}⚠ 当前 GPU 不支持 NVENC 或环境未配置，切换为 CPU 软件编码器 (libx265)${NC}"
        vcodec="libx265"
    fi

    echo -e "\n${CYAN}==============================================================================${NC}"
    echo -e "${BOLD}${MAGENTA}开始执行超分推理${NC}"
    echo -e "  - 输入文件: ${clip_input}"
    echo -e "  - 输出文件: ${clip_output}"
    echo -e "  - 编码器:   ${vcodec}"
    echo -e "  - Engine:   ${engine_file}"
    if [ -n "$preload_fix" ]; then
        echo -e "  - LD_PRELOAD: ${preload_fix} (NVENC 容器多卡修复生效)"
    fi
    echo -e "${CYAN}------------------------------------------------------------------------------${NC}"

    export LD_LIBRARY_PATH="${PROJECT_ROOT}/build:${FFMPEG_INSTALL_DIR}/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
    if [ -n "$preload_fix" ]; then
        export LD_PRELOAD="${preload_fix}${LD_PRELOAD:+ $LD_PRELOAD}"
    fi

    cd "$PROJECT_ROOT"
    local start_ts=$(date +%s)
    ./build/aji_encode \
        --input "$clip_input" \
        --output "$clip_output" \
        --engine "$engine_file" \
        --max-width "$src_width" \
        --max-height "$src_height" \
        --vcodec "$vcodec" \
        --overwrite

    local encode_status=$?
    local end_ts=$(date +%s)
    local elapsed=$((end_ts - start_ts))

    if [ $encode_status -eq 0 ] && [ -f "$clip_output" ]; then
        echo -e "\n${GREEN}==============================================================================${NC}"
        echo -e "${BOLD}${GREEN}🎉 超分测试成功完成！${NC}"
        echo -e "  - 耗时:       ${elapsed} 秒"
        echo -e "  - 输出大小:   $(du -h "$clip_output" | cut -f1)"
        echo -e "  - 输出分辨率: $(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$clip_output" 2>/dev/null || echo "检测中...")"
        echo -e "${GREEN}==============================================================================${NC}"
    else
        echo -e "\n${RED}✖ 超分测试失败，请检查上方日志输出！${NC}"
    fi
}

# 一键全自动部署流程
one_click_setup() {
    init_project_root
    print_header
    echo -e "${BOLD}${MAGENTA}开始执行一键全自动部署流程...${NC}\n"

    check_nvidia_driver
    
    echo -e "\n>>> 步骤 1/6: 配置 NVIDIA 官方 Network 源并清理旧 Keyring"
    if ! setup_nvidia_network_repo; then
        echo -e "${RED}[错误] 步骤 1 失败！${NC}"
        return 1
    fi

    echo -e "\n>>> 步骤 2/6: 自动匹配最高 CUDA Toolkit 并安装 TensorRT"
    if ! install_cuda_and_tensorrt; then
        echo -e "${RED}[错误] 步骤 2 失败！${NC}"
        return 1
    fi

    echo -e "\n>>> 步骤 3/6: 安装编译链工具与 BtbN FFmpeg Shared 构建"
    if ! install_build_tools_and_ffmpeg; then
        echo -e "${RED}[错误] 步骤 3 失败！${NC}"
        return 1
    fi

    echo -e "\n>>> 步骤 4/6: 安装 NVENC 容器多卡枚举修复补丁 (libnvenc_fix)"
    install_nvenc_fix || echo -e "${YELLOW}补丁安装遇到问题，继续后续步骤...${NC}"

    echo -e "\n>>> 步骤 5/6: 创建 Python 虚拟环境 (venv)"
    if ! setup_python_venv; then
        echo -e "${RED}[错误] 步骤 5 失败！${NC}"
        return 1
    fi

    echo -e "\n>>> 步骤 6/6: 编译 AnimeJaNai-Inference 核心组件"
    if ! build_project; then
        echo -e "${RED}[错误] 步骤 6 编译失败，请检查编译日志！${NC}"
        return 1
    fi

    echo -e "\n${GREEN}==============================================================================${NC}"
    echo -e "${BOLD}${GREEN}🎉 一键环境安装与编译完全成功！${NC}"
    echo -e "接下来您可以选择 [9] 下载模型构建 Engine，然后选择 [10] 运行 1 分钟测试。"
    echo -e "${GREEN}==============================================================================${NC}"
}

# 交互式主菜单
main_menu() {
    init_project_root
    while true; do
        print_header
        echo -e "${BOLD}请选择操作：${NC}"
        echo -e "  ${BOLD}[1]${NC}  ${CYAN}全面环境自检与诊断${NC} (Check All Environment)"
        echo -e "  ${BOLD}[2]${NC}  ${GREEN}一键全自动安装与编译${NC} (One-Click Setup: CUDA Toolkit/TRT -> FFmpeg -> Fix -> Build)"
        echo -e "  ----------------------------------------------------------------------"
        echo -e "  ${BOLD}[3]${NC}  清理旧 Keyring 并配置 NVIDIA 官方源 (Fix AutoDL Keyring & Setup Repo)"
        echo -e "  ${BOLD}[4]${NC}  ${MAGENTA}自动检测驱动支持上限并安装最高 CUDA Toolkit 与 TensorRT${NC}"
        echo -e "  ${BOLD}[5]${NC}  安装编译工具链与 BtbN FFmpeg Shared (Install Build Tools & FFmpeg)"
        echo -e "  ${BOLD}[6]${NC}  ${MAGENTA}下载并编译 NVENC 容器多卡修复补丁${NC} (Build libnvenc_fix.so)"
        echo -e "  ${BOLD}[7]${NC}  创建并配置 Python 虚拟环境 (Setup Python .venv)"
        echo -e "  ${BOLD}[8]${NC}  编译 AnimeJaNai-Inference (CMake Build)"
        echo -e "  ----------------------------------------------------------------------"
        echo -e "  ${BOLD}[9]${NC}  ${YELLOW}下载超分模型 (ONNX) 并构建 TensorRT Engine${NC} (Download Models & Build Engine)"
        echo -e "  ${BOLD}[10]${NC} ${MAGENTA}截取 1 分钟 example.mkv 进行超分测试${NC} (Run 1-Min Encode Test)"
        echo -e "  ----------------------------------------------------------------------"
        echo -e "  ${BOLD}[11]${NC} ${CYAN}一键清理系统中多余旧版 CUDA 软件包 (释放空间)${NC} (Purge Old CUDA Packages)"
        echo -e "  ${BOLD}[0]${NC}  退出 (Exit)"
        echo -e "${CYAN}------------------------------------------------------------------------------${NC}"
        read -rp "请输入选项数字 [0-11]: " choice

        case "$choice" in
            1) diagnose_all ;;
            2) one_click_setup ;;
            3) setup_nvidia_network_repo ;;
            4) install_cuda_and_tensorrt ;;
            5) install_build_tools_and_ffmpeg ;;
            6) install_nvenc_fix ;;
            7) setup_python_venv ;;
            8) build_project ;;
            9) download_and_build_engine ;;
            10) run_test_clip ;;
            11) purge_old_cuda_packages ;;
            0) echo -e "\n${GREEN}感谢使用，再见！${NC}"; exit 0 ;;
            *) echo -e "\n${RED}无效选项，请重新输入！${NC}" ;;
        esac

        echo -e "\n按回车键返回主菜单..."
        read -r
    done
}

# 命令行非交互模式参数支持
init_project_root

if [ $# -gt 0 ]; then
    case "$1" in
        --check|check|-c) diagnose_all ;;
        --all|all|--install-all) one_click_setup ;;
        --repo) setup_nvidia_network_repo ;;
        --cuda) install_cuda_and_tensorrt ;;
        --ffmpeg) install_build_tools_and_ffmpeg ;;
        --nvenc-fix|--fix) install_nvenc_fix ;;
        --venv) setup_python_venv ;;
        --build) build_project ;;
        --engine|--models) download_and_build_engine ;;
        --clean-cuda|--purge-cuda) purge_old_cuda_packages ;;
        --test) run_test_clip ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  (无参数)        启动交互式向导主菜单"
            echo "  --check         全面环境自检与诊断"
            echo "  --all           一键全自动安装与编译"
            echo "  --repo          清理旧 Keyring 并配置 NVIDIA Network 源"
            echo "  --cuda          自动检测并安装匹配显卡驱动最高上限的 CUDA 与 TensorRT"
            echo "  --ffmpeg        安装编译工具链与 BtbN FFmpeg Shared"
            echo "  --nvenc-fix     下载并编译 NVENC 多卡容器修复补丁 (libnvenc_fix.so)"
            echo "  --venv          创建并配置 Python 虚拟环境"
            echo "  --build         编译 AnimeJaNai-Inference"
            echo "  --engine        下载模型并构建 TensorRT Engine"
            echo "  --clean-cuda    清理系统中的多余旧版本 CUDA 软件包"
            echo "  --test          截取 1 分钟 example.mkv 进行超分测试"
            ;;
        *) echo -e "${RED}未知参数: $1${NC}，请使用 --help 查看帮助。"; exit 1 ;;
    esac
else
    main_menu
fi
