#!/usr/bin/env bash
# ==============================================================================
# AnimeJaNai-Inference 交互式环境检测、安装与部署脚本
# 适用环境：Ubuntu 24.04 / 22.04 LTS (AutoDL / 云 GPU 容器 / 本地 GPU 服务器)
# 特别集成：AutoDL 旧源清理、严格要求 CUDA Toolkit 13.x 与 TensorRT 11.x 自动部署、
#          NVIDIA 官方源配置、TensorRT 11 强类型自动适配、
#          BtbN FFmpeg (n8.1/master) Shared、NVENC 容器多卡修复补丁 (libnvenc_fix)、
#          Python venv、按 conf 布局下载模型（含 RIFE）与 1 分钟快速验证
# ==============================================================================

set -o pipefail

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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
NVENC_FIX_SRC_URL="https://raw.githubusercontent.com/flexgrip/nvidia-gpu-enumeration/1475091ae9e8fb23cc22d565adf932a488d00bcd/nvenc_fix.c"
NVENC_FIX_SRC_MIRROR="https://ghproxy.net/https://raw.githubusercontent.com/flexgrip/nvidia-gpu-enumeration/1475091ae9e8fb23cc22d565adf932a488d00bcd/nvenc_fix.c"

# 模型的文件名、来源与 SHA256 见 scripts/runtime-models.tsv（与 Runtime 打包共用）。

# FFmpeg 固定日期构建；更新 URL 时必须同时更新对应 SHA256。
FFMPEG_TAR_N81_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-08-31-13-27/ffmpeg-n8.1.2-50-g1a748fe2cd-linux64-gpl-shared-8.1.tar.xz"
FFMPEG_N81_SHA256="35dc428bf78d3f8a4447a68707338ecf9560ebcc7459967974f9ff5b1f84be24"
FFMPEG_TAR_MASTER_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-08-31-13-27/ffmpeg-N-126342-gf88b741dbf-linux64-gpl-shared.tar.xz"
FFMPEG_MASTER_SHA256="390706494998c2e2e97d64f260b47c21389b7ad08bdcaca292661ae893486875"
NVENC_FIX_SHA256="13d3b4256f447dae4fb36b2f1f6fc2bb3a4bcc48813e96df149bdbaa63d5d7db"
CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
REPO_GIT_URL="https://github.com/ahdiua/animejanai-inference.git"

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
    local root="${PROJECT_ROOT:-$SCRIPT_DIR}"
    if [ ! -f "$root/CMakeLists.txt" ] || ! grep -qi animejanai "$root/CMakeLists.txt"; then
        echo "未找到项目，请使用 --project-root 指定 fork 的源码目录。" >&2
        echo "可手动克隆: git clone $REPO_GIT_URL" >&2
        return 1
    fi
    PROJECT_ROOT=$(cd -- "$root" && pwd -P) || return 1
    cd -- "$PROJECT_ROOT" || return 1
    VENV_DIR="$PROJECT_ROOT/.venv"
}

# 智能同步与激活最高版本的 CUDA Toolkit
# 只在本进程选择 Toolkit；不改写系统 CUDA 软链接。
resolve_cuda() {
    local candidate="${CUDA_ROOT:+$CUDA_ROOT/bin/nvcc}"
    candidate=${candidate:-${CUDACXX:-}}
    if [ -z "$candidate" ]; then
        candidate=$(command -v nvcc) || candidate=/usr/local/cuda/bin/nvcc
    fi
    candidate=$(command -v -- "$candidate") || { echo "未找到 nvcc。" >&2; return 1; }
    NVCC_BIN=$(readlink -f -- "$candidate") || return 1
    NVCC_VER=$("$NVCC_BIN" --version | sed -nE 's/.*release (13\.[0-9]+).*/\1/p') || return 1
    [[ "$NVCC_VER" =~ ^13\.[0-9]+$ ]] || { echo "要求 CUDA Toolkit 13.x: $NVCC_BIN" >&2; return 1; }
    CUDA_ROOT=$(dirname -- "$(dirname -- "$NVCC_BIN")")
}

# 移除空路径项及重复项，避免当前目录意外进入动态库搜索路径。
prepend_path() {
    local name="$1" first="$2" item result="" rest
    rest="$first:${!name}"
    while [ -n "$rest" ]; do
        item=${rest%%:*}
        if [[ "$rest" == *:* ]]; then rest=${rest#*:}; else rest=""; fi
        [ -n "$item" ] || continue
        case ":$result:" in *":$item:"*) continue ;; esac
        result+="${result:+:}$item"
    done
    printf -v "$name" '%s' "$result"
    export "${name?}"
}

activate_build_environment() {
    resolve_cuda || return 1
    export CUDA_HOME="$CUDA_ROOT"
    prepend_path PATH "$CUDA_ROOT/bin"
    prepend_path LD_LIBRARY_PATH "$CUDA_ROOT/lib64"
    prepend_path PATH "$FFMPEG_INSTALL_DIR/bin"
    prepend_path LD_LIBRARY_PATH "$FFMPEG_INSTALL_DIR/lib"
    prepend_path PKG_CONFIG_PATH "$FFMPEG_PC_DIR"
}

valid_index() {
    [[ "$1" =~ ^[0-9]{1,6}$ ]] && (( 10#$1 >= $2 && 10#$1 <= $3 ))
}

prompt_value() {
    local name="$1" prompt="$2" default="${3:-}" answer
    if [ "${ASSUME_DEFAULTS:-0}" = 1 ]; then
        [ -n "$default" ] || { echo "非交互模式需要明确输入: $prompt" >&2; return 1; }
        answer=$default
    else
        read -r -p "$prompt" answer || { echo "输入已关闭，取消当前操作。" >&2; return 1; }
        answer=${answer:-$default}
    fi
    printf -v "$name" '%s' "$answer"
}

check_install_platform() {
    local ID="" VERSION_ID=""
    # shellcheck source=/dev/null
    . /etc/os-release || return 1
    if [ "$ID" != ubuntu ] || [[ "$VERSION_ID" != 22.04 && "$VERSION_ID" != 24.04 ]] || [ "$(uname -m)" != x86_64 ]; then
        echo "自动安装仅支持 Ubuntu 22.04/24.04 x86_64。" >&2
        return 1
    fi
    CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//./}/x86_64/cuda-keyring_1.1-1_all.deb"
    case "$VERSION_ID" in
        22.04) CUDA_KEYRING_SHA256=d93190d50b98ad4699ff40f4f7af50f16a76dac3bb8da1eaaf366d47898ff8df ;;
        24.04) CUDA_KEYRING_SHA256=d2a6b11c096396d868758b86dab1823b25e14d70333f1dfa74da5ddaf6a06dba ;;
    esac
}


# The generated profile expands variables when sourced, not during installation.
# shellcheck disable=SC2016
install_environment_profile() (
    local component="$1" tmp_dir profile directory
    local bashrc="${2:-$HOME/.bashrc}"
    tmp_dir=$(mktemp -d) || return 1
    trap 'rm -rf -- "$tmp_dir"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    case "$component" in
        cuda)
            directory="$CUDA_ROOT"
            printf 'export CUDA_HOME=%q\n' "$directory" > "$tmp_dir/profile" || return 1
            ;;
        ffmpeg)
            directory="$FFMPEG_INSTALL_DIR"
            printf 'export PKG_CONFIG_PATH=%q"${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"\n' "$FFMPEG_PC_DIR" > "$tmp_dir/profile" || return 1
            ;;
        *) return 1 ;;
    esac
    printf 'export PATH=%q"${PATH:+:$PATH}"\n' "$directory/bin" >> "$tmp_dir/profile" || return 1
    local lib="$directory/lib"
    [ "$component" != cuda ] || lib="$directory/lib64"
    printf 'export LD_LIBRARY_PATH=%q"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n' "$lib" >> "$tmp_dir/profile" || return 1
    cat >> "$tmp_dir/profile" <<'PROFILE' || return 1
# Remove empty entries inherited from older deployment profiles.
while :; do
    case "$LD_LIBRARY_PATH" in
        :*) LD_LIBRARY_PATH=${LD_LIBRARY_PATH#:} ;;
        *:) LD_LIBRARY_PATH=${LD_LIBRARY_PATH%:} ;;
        *::*) LD_LIBRARY_PATH=${LD_LIBRARY_PATH%%::*}:${LD_LIBRARY_PATH#*::} ;;
        *) break ;;
    esac
done
export LD_LIBRARY_PATH
PROFILE
    profile="/etc/profile.d/$component.sh"
    run_as_root install -m 644 "$tmp_dir/profile" "$profile" || return 1
    # Replace only exact lines emitted by older deploy.sh; retain user settings.
    if [ -f "$bashrc" ]; then
        awk -v component="$component" '
            component == "cuda" && ($0 == "export CUDA_HOME=/usr/local/cuda" ||
                $0 == "export PATH=/usr/local/cuda/bin:$PATH" ||
                $0 == "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH") {next}
            component == "ffmpeg" && ($0 == "export PKG_CONFIG_PATH=\"/opt/ffmpeg/lib/pkgconfig:${PKG_CONFIG_PATH:-}\"" ||
                $0 == "export LD_LIBRARY_PATH=\"/opt/ffmpeg/lib:${LD_LIBRARY_PATH:-}\"" ||
                $0 == "export PATH=\"/opt/ffmpeg/bin:$PATH\"") {next}
            {print}
        ' "$bashrc" > "$tmp_dir/bashrc" || return 1
        cat "$tmp_dir/bashrc" > "$bashrc" || return 1
    fi
    local line="[ ! -r $profile ] || . $profile"
    if ! grep -Fxq "$line" "$bashrc" 2>/dev/null; then
        printf '\n%s\n' "$line" >> "$bashrc" || return 1
    fi
)

fetch_verified() {
    local url="$1" destination="$2" expected="$3" actual
    [[ "$url" == https://* && "$expected" =~ ^[a-f0-9]{64}$ ]] || return 1
    if command -v curl >/dev/null; then
        curl --fail --location --proto '=https' --proto-redir '=https' \
            --connect-timeout 20 --max-time 1800 --retry 2 --output "$destination" "$url" || return 1
    elif command -v wget >/dev/null; then
        wget --https-only --timeout=30 --tries=3 -O "$destination" -- "$url" || return 1
    else
        echo "需要 curl 或 wget 下载依赖。" >&2
        return 1
    fi
    [ -s "$destination" ] || return 1
    actual=$(sha256sum < "$destination") || return 1
    [ "${actual%% *}" = "$expected" ] || { echo "SHA256 不匹配: $url" >&2; return 1; }
}

# 获取 APT 官方源中可用的 CUDA Toolkit 版本列表 (如 13.2 13.0 12.8 12.6)
get_available_cuda_packages() {
    apt-cache pkgnames 2>/dev/null | grep -E '^cuda-nvcc-[0-9]+-[0-9]+$' | sed 's/cuda-nvcc-//' | tr '-' '.' | sort -V -r
}

# 同时检查包名和版本号；NCCL/TensorRT 的 CUDA 版本只出现在版本号中。
get_old_cuda_packages() {
    dpkg-query -W -f='${binary:Package}\t${Version}\t${db:Status-Abbrev}\n' | awk '
        $3 ~ /^ii/ && $1 ~ /^(cuda($|-)|libcu|libnccl|libnvinfer|libnvonnxparsers|python3-libnvinfer|tensorrt($|-)|nsight-)/ {
            name = $1
            sub(/:[^:]+$/, "", name)
            if (name ~ /-(10|11|12)(-|$)/ || $2 ~ /[+]cuda(10|11|12)([.~-]|$)/)
                print $1
        }
    '
}

# 卸载并清理可识别的旧版 CUDA 软件包（用户主动选择清理时调用）
purge_old_cuda_packages() {
    echo -e "${CYAN}正在扫描并清理系统中残留的旧版本 CUDA 软件包 (CUDA < 13.x)...${NC}"
    local old_pkgs
    old_pkgs=$(get_old_cuda_packages) || return 1
    
    if [ -n "$old_pkgs" ]; then
        echo -e "${YELLOW}检测到以下旧版本 CUDA 软件包，正在通过 apt 彻底卸载以释放空间：${NC}"
        echo "$old_pkgs" | tr '\n' ' '
        echo -e "\n"
        local old_packages=()
        mapfile -t old_packages <<< "$old_pkgs"
        run_as_root apt-get purge -y "${old_packages[@]}" || return 1
        run_as_root apt-get autoremove -y --purge || return 1
        echo -e "${GREEN}✔ 旧版本 CUDA 软件包卸载完成！${NC}"
    else
        echo -e "${GREEN}✔ 未发现包名或版本号明确标记为 CUDA 10/11/12 的已安装软件包。${NC}"
    fi

    for old_dir in /usr/local/cuda-12.* /usr/local/cuda-11.* /usr/local/cuda-10.*; do
        if [ -d "$old_dir" ]; then
            echo -e "${CYAN}清理旧残留目录: ${old_dir}${NC}"
            run_as_root rm -rf "$old_dir" || return 1
        fi
    done

    return 0
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
        echo -e "${RED}[致命错误] 驱动未安装。${NC}"
        return 1
    fi

    if ! nvidia-smi &>/dev/null; then
        echo -e "${RED}✖ nvidia-smi 执行失败，无法与 NVIDIA 驱动内核模块通信！${NC}"
        echo -e "${RED}[致命错误] 驱动异常。${NC}"
        return 1
    fi

    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
    CUDA_MAX_VER=$(nvidia-smi | grep -o "CUDA Version: [0-9.]*" | head -n 1 | awk '{print $3}')
    
    echo -e "${GREEN}✔ NVIDIA 驱动正常！${NC}"
    echo -e "  - GPU 型号:       ${BOLD}${GPU_NAME}${NC}"
    echo -e "  - 驱动版本:       ${BOLD}${DRIVER_VER}${NC}"
    echo -e "  - 支持最高 CUDA:  ${BOLD}${CUDA_MAX_VER}${NC}"

    local drv_cuda_major

    drv_cuda_major=$(echo "$CUDA_MAX_VER" | cut -d. -f1) || return 1
    if [ -n "$drv_cuda_major" ] && [ "$drv_cuda_major" -lt 13 ]; then
        echo -e "${YELLOW}  ! 警告: 显卡驱动支持最高 CUDA 为 ${CUDA_MAX_VER} (< 13.0)。${NC}"
        echo -e "${YELLOW}    项目严格要求 CUDA Toolkit 13.x，建议升级显卡驱动至 580+。${NC}"
    fi
}

# 检查系统信息
check_os_info() {
    echo -e "\n${BOLD}[2/9] 检查操作系统与发行版...${NC}"
    if [ -f /etc/os-release ]; then
        # shellcheck source=/dev/null
        . /etc/os-release || return 1
        echo -e "  - 发行版:         ${PRETTY_NAME:-$NAME}"
        echo -e "  - 系统架构:       $(uname -m)"
        if [[ "$ID" != "ubuntu" && "$ID_LIKE" != *"ubuntu"* && "$ID_LIKE" != *"debian"* ]]; then
            echo -e "${YELLOW}  ! 警告: 当前系统非 Ubuntu/Debian，部分 apt 安装步骤可能需要适配。${NC}"
        fi
    else
        echo -e "  - 系统架构:       $(uname -m)"
    fi
}

# 检查选定 CUDA Toolkit（只读）
check_cuda_toolkit() {
    echo -e "\n${BOLD}[3/9] 检查 CUDA Toolkit (nvcc)...${NC}"
    resolve_cuda || return 1
    printf 'CUDA Toolkit %s: %s\n' "$NVCC_VER" "$NVCC_BIN"
}

# 检查 TensorRT
header_number() {
    awk -v name="$2" '
        $1 == "#define" {value[$2] = $3}
        END {
            for (i = 0; i < 8 && name !~ /^[0-9]+$/; i++) name = value[name]
            if (name ~ /^[0-9]+$/) print name; else exit 1
        }' "$1"
}

resolve_tensorrt() {
    local root header libdir tool output encoded version major minor patch
    local roots=("${TRT_ROOT:-/usr}")
    [ -n "${TRT_ROOT:-}" ] || roots+=("$HOME/sdk/tensorrt/usr")
    for root in "${roots[@]}"; do
        header="$root/include/x86_64-linux-gnu/NvInferVersion.h"
        [ -f "$header" ] || header="$root/include/NvInferVersion.h"
        [ -f "$header" ] || continue
        major=$(header_number "$header" NV_TENSORRT_MAJOR) || continue
        minor=$(header_number "$header" NV_TENSORRT_MINOR) || continue
        patch=$(header_number "$header" NV_TENSORRT_PATCH) || continue
        [[ "$major" = 11 && "$minor" =~ ^[0-9]+$ && "$patch" =~ ^[0-9]+$ ]] || continue
        libdir="$root/lib/x86_64-linux-gnu"
        [ -f "$libdir/libnvinfer.so" ] || libdir="$root/lib"
        [ -f "$libdir/libnvinfer.so" ] || continue
        tool="${TRTEXEC_BIN:-}"
        if [ -z "$tool" ]; then
            for tool in "$root/src/tensorrt/bin/trtexec" "$root/bin/trtexec" "${root%/usr}/bin/trtexec"; do
                [ -x "$tool" ] && break
            done
            [ -x "$tool" ] || tool=$(command -v trtexec) || continue
        fi
        [ -x "$tool" ] || continue
        output=$(LD_LIBRARY_PATH="$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$tool" --help 2>&1) || continue
        encoded=$(sed -nE 's/.*TensorRT v([0-9]+).*/\1/p' <<< "$output" | head -n 1)
        [[ "$encoded" =~ ^[0-9]{5,6}$ ]] || continue
        version=$((10#$major * 10000 + 10#$minor * 100 + 10#$patch))
        [ "$encoded" = "$version" ] || continue
        TRT_ROOT=$(cd -- "$root" && pwd -P) || return 1
        TRT_INCLUDE_DIR=$(dirname -- "$header")
        TRT_LIB_DIR="$libdir"
        TRTEXEC_BIN=$(readlink -f -- "$tool") || return 1
        TRT_VER="$major.$minor.$patch"
        return 0
    done
    echo "未找到一致的 TensorRT 11 开发库和 trtexec；请指定 --trt-root / --trtexec。" >&2
    return 1
}

check_tensorrt() {
    echo -e "\n${BOLD}[4/9] 检查 TensorRT...${NC}"
    resolve_tensorrt || return 1
    printf 'TensorRT %s: %s\ntrtexec: %s\n' "$TRT_VER" "$TRT_ROOT" "$TRTEXEC_BIN"
}

# 检查 FFmpeg (BtbN Shared Build)
check_ffmpeg() {
    echo -e "\n${BOLD}[5/9] 检查 FFmpeg 及开发库...${NC}"
    local ffmpeg_ok=1

    if command -v ffmpeg &>/dev/null; then
        local output
        output=$(ffmpeg -version 2>&1) || { printf '%s\n' "$output" >&2; return 1; }
        FFMPEG_VER=${output%%$'\n'*}
        echo -e "${GREEN}✔ FFmpeg 命令可用: ${BOLD}${FFMPEG_VER}${NC}"
    else
        echo -e "${YELLOW}✖ 未在 PATH 中找到 ffmpeg 命令。${NC}"
        ffmpeg_ok=0
    fi

    prepend_path PKG_CONFIG_PATH "$FFMPEG_PC_DIR"
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
# 成功时设置 NVENC_PRELOAD；只有符合枚举故障特征才尝试补丁。
check_nvenc_and_patch() {
    local output
    NVENC_PRELOAD=""
    NVENC_PATCH_NEEDED=0
    if output=$(ffmpeg -nostdin -v error -f lavfi -i nullsrc=s=256x256:d=0.1 -c:v hevc_nvenc -f null - 2>&1); then
        echo "NVENC 原生编码正常。"
        return 0
    fi
    if grep -qiE 'unsupported device|no capable devices' <<< "$output"; then
        NVENC_PATCH_NEEDED=1
        if [ -f "$NVENC_FIX_SO" ] && output=$(LD_PRELOAD="$NVENC_FIX_SO${LD_PRELOAD:+ $LD_PRELOAD}" \
            ffmpeg -nostdin -v error -f lavfi -i nullsrc=s=256x256:d=0.1 -c:v hevc_nvenc -f null - 2>&1); then
            NVENC_PRELOAD="$NVENC_FIX_SO"
            echo "NVENC 在容器枚举补丁下编码正常。"
            return 0
        fi
    fi
    printf 'NVENC 检测失败:\n%s\n' "$output" >&2
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
    local status=0
    echo -e "\n${BOLD}[9/9] 检查 AnimeJaNai 构建产物与模型状态...${NC}"
    
    if [ -f "${PROJECT_ROOT}/build/libaji.so" ] && [ -f "${PROJECT_ROOT}/build/libaji_trt.so" ] && [ -f "${PROJECT_ROOT}/build/aji_encode" ]; then
        echo -e "${GREEN}✔ 项目构建产物齐全 (${PROJECT_ROOT}/build/libaji.so, libaji_trt.so, aji_encode)${NC}"
    else
        echo -e "${YELLOW}ℹ 项目尚未完整编译 (缺少 build/aji_encode 或 libaji*.so)${NC}"
        status=1
    fi

    if [ -d "$MODELS_DIR" ]; then
        load_model_assets || return 1
        local row=() total=0 present=0 missing=()
        while IFS=$'\t' read -r -u 3 -a row; do
            total=$((total + 1))
            if [ -s "$MODELS_DIR/${row[2]}" ]; then
                present=$((present + 1))
            else
                missing+=("${row[2]}")
            fi
        done 3< <(aji_model_rows)
        echo -e "  - 模型目录 (${MODELS_DIR}): 内置方案模型 ${present}/${total}"
        local file
        for file in "${missing[@]}"; do
            echo -e "    * 缺失: ${file}"
        done
        [ ${#missing[@]} -eq 0 ] || echo -e "${YELLOW}ℹ 可运行 ./deploy.sh --models 下载缺失模型${NC}"
        report_legacy_model_files
    else
        echo -e "  - 模型目录 (${MODELS_DIR}): 尚未创建"
    fi

    if [ -f "${PROJECT_ROOT}/example.mkv" ]; then
        echo -e "${GREEN}✔ 根目录已就绪测试视频: ${PROJECT_ROOT}/example.mkv ($(du -h "${PROJECT_ROOT}/example.mkv" | cut -f1))${NC}"
    else
        echo -e "${YELLOW}ℹ 项目目录下未找到 example.mkv (测试功能需要此文件)${NC}"
    fi
    return "$status"
}

# 综合环境诊断
diagnose_all() (
    init_project_root || return 1
    print_header
    local check failures=0
    for check in check_nvidia_driver check_os_info check_cuda_toolkit check_tensorrt \
        check_ffmpeg check_nvenc_and_patch check_build_tools check_python_venv check_build_and_models; do
        "$check" || failures=$((failures + 1))
    done
    printf '\n环境自检完成，失败项目: %s\n' "$failures"
    [ "$failures" -eq 0 ]
)

# 步骤 A: AutoDL 旧 keyring 清理与 NVIDIA 官方网络源配置
setup_nvidia_network_repo() (
    print_header
    echo -e "${BOLD}${MAGENTA}[步骤] 清理 AutoDL 旧 keyring 并配置 NVIDIA 官方 Network 仓库${NC}\n"
    
    check_install_platform || return 1
    check_nvidia_driver || return 1

    echo -e "${CYAN}1. 正在清理旧版本 cuda-keyring 及潜在的 APT Pin 锁定配置...${NC}"
    run_as_root dpkg -P cuda-keyring 2>/dev/null || true
    run_as_root apt-get purge -y cuda-keyring 2>/dev/null || true
    run_as_root rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia-cuda*.list || return 1
    run_as_root rm -f /etc/apt/preferences.d/*cuda* /etc/apt/preferences.d/*nvidia* || return 1

    echo -e "\n${CYAN}2. 正在下载并安装 NVIDIA 官方最新 cuda-keyring (匹配当前 Ubuntu 版本)...${NC}"
    local tmp_dir tmp_deb
    tmp_dir=$(mktemp -d) || return 1
    trap 'rm -rf -- "$tmp_dir"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    tmp_deb="$tmp_dir/keyring.deb"
    if ! fetch_verified "$CUDA_KEYRING_URL" "$tmp_deb" "$CUDA_KEYRING_SHA256"; then
        echo -e "${RED}[错误] 下载 cuda-keyring 失败，请检查网络连接！${NC}"
        rm -f "$tmp_deb"
        return 1
    fi

    run_as_root dpkg -i "$tmp_deb" || return 1
    rm -f "$tmp_deb"

    echo -e "\n${CYAN}3. 配置 APT Pin 策略（排除 nsight 工具）...${NC}"
    cat << 'EOF' | run_as_root tee /etc/apt/preferences.d/cuda-repository-pin-600 > /dev/null || return 1
Package: nsight-compute* nsight-systems*
Pin: origin developer.download.nvidia.com
Pin-Priority: -1
EOF

    echo -e "\n${CYAN}4. 正在执行 apt-get update 刷新软件包缓存...${NC}"
    run_as_root apt-get update || return 1

    echo -e "\n${GREEN}✔ NVIDIA 官方 Network 源配置完成！${NC}"
)

# 步骤 B: 安装严格要求的 CUDA Toolkit 13.x 与 TensorRT 11.x
install_cuda_and_tensorrt() {
    check_install_platform || return 1
    check_nvidia_driver || return 1
    [[ "$DRIVER_VER" =~ ^[0-9]+\. ]] && [ "${DRIVER_VER%%.*}" -ge 580 ] || {
        echo "CUDA 13.x 自动安装要求 NVIDIA 驱动 580+。" >&2; return 1;
    }
    local target_cuda="${CUDA_VERSION:-}" v_choice manual_ver cuda_pkg_suffix trt_version dev_version
    local valid_vers=()
    mapfile -t valid_vers < <(get_available_cuda_packages | grep -E '^13\.[0-9]+$')
    if [ -z "$target_cuda" ]; then
        [ "${#valid_vers[@]}" -gt 0 ] || { echo "APT 中没有 CUDA 13.x，请先配置官方源。" >&2; return 1; }
        printf '可用 CUDA 版本: %s\n' "${valid_vers[*]}"
        local i
        for i in "${!valid_vers[@]}"; do printf '%s) %s\n' "$((i+1))" "${valid_vers[$i]}"; done
        printf '%s) 手动输入版本\n' "$((${#valid_vers[@]}+1))"
        prompt_value v_choice '选择 CUDA 版本 [默认 1]: ' 1 || return 1
        valid_index "$v_choice" 1 "$((${#valid_vers[@]}+1))" || return 1
        v_choice=$((10#$v_choice))
        if [ "$v_choice" -le "${#valid_vers[@]}" ]; then
            target_cuda=${valid_vers[$((v_choice-1))]}
        else
            prompt_value manual_ver '请输入 CUDA 13.x 版本: ' || return 1
            target_cuda=$manual_ver
        fi
    fi
    [[ "$target_cuda" =~ ^13\.[0-9]{1,2}$ ]] || { echo "CUDA 版本必须为 13.x。" >&2; return 1; }
    cuda_pkg_suffix=${target_cuda//./-}
    local existing_cuda=() clean_old=""
    mapfile -d '' -t existing_cuda < <(find /usr/local -maxdepth 1 -type d -name 'cuda-[0-9]*' -print0 | sort -zV)
    if [ "${#existing_cuda[@]}" -gt 0 ]; then
        printf '已有 CUDA: %s\n' "${existing_cuda[*]}"
        prompt_value clean_old '是否清理旧版 CUDA (12.x/11.x/10.x)？[y/N]: ' N || return 1
        case "$clean_old" in
            y|Y) purge_old_cuda_packages || return 1 ;;
            n|N) ;;
            *) echo "请输入 y 或 n。" >&2; return 1 ;;
        esac
    fi
    run_as_root apt-get update || return 1
    trt_version=$(apt-cache madison tensorrt | awk '$3 ~ /^11\./ {print $3}' | sort -V | tail -n 1)
    [ -n "$trt_version" ] || { echo "APT 中没有 TensorRT 11。" >&2; return 1; }
    dev_version=$(apt-cache madison tensorrt-dev | awk -v v="$trt_version" '$3 == v {print $3; exit}')
    [ "$dev_version" = "$trt_version" ] || { echo "缺少配套 tensorrt-dev。" >&2; return 1; }
    run_as_root apt-get install -y --no-install-recommends \
        "tensorrt=$trt_version" "tensorrt-dev=$trt_version" \
        "cuda-nvcc-$cuda_pkg_suffix" "cuda-cudart-dev-$cuda_pkg_suffix" \
        "cuda-driver-dev-$cuda_pkg_suffix" || return 1
    CUDA_ROOT="/usr/local/cuda-$target_cuda"
    TRT_ROOT=/usr
    TRTEXEC_BIN=""
    activate_build_environment || return 1
    check_tensorrt || return 1
    install_environment_profile cuda || return 1
    printf '安装完成: CUDA %s，TensorRT %s。\n' "$NVCC_VER" "$TRT_VER"
    echo "后续可用 --cuda-root $CUDA_ROOT 明确选择该 Toolkit。"
}

# 步骤 C: 安装编译工具与固定版本 BtbN Shared FFmpeg
install_build_tools_and_ffmpeg() (
    check_install_platform || return 1
    local ffmpeg_choice url digest tmp_dir
    case "${FFMPEG_VARIANT:-}" in
        n8.1) ffmpeg_choice=1 ;;
        master) ffmpeg_choice=2 ;;
        '') prompt_value ffmpeg_choice 'FFmpeg: 1) n8.1  2) master [默认 1]: ' 1 || return 1 ;;
        *) echo "FFmpeg 版本必须为 n8.1 或 master。" >&2; return 1 ;;
    esac
    valid_index "$ffmpeg_choice" 1 2 || return 1
    url=$FFMPEG_TAR_N81_URL; digest=$FFMPEG_N81_SHA256
    if (( 10#$ffmpeg_choice == 2 )); then url=$FFMPEG_TAR_MASTER_URL; digest=$FFMPEG_MASTER_SHA256; fi
    run_as_root apt-get update || return 1
    run_as_root apt-get install -y --no-install-recommends \
        build-essential cmake pkg-config git p7zip-full aria2 wget curl ca-certificates tar xz-utils || return 1
    tmp_dir=$(mktemp -d) || return 1
    trap 'rm -rf -- "$tmp_dir"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    fetch_verified "$url" "$tmp_dir/ffmpeg.tar.xz" "$digest" || return 1
    # 校验整个压缩流和必要成员后再移除旧安装；不保留回滚副本。
    tar -tJf "$tmp_dir/ffmpeg.tar.xz" > "$tmp_dir/members" || return 1
    local member
    for member in bin/ffmpeg bin/ffprobe lib/pkgconfig/libavcodec.pc lib/pkgconfig/libavformat.pc; do
        grep -qE "^[^/]+/$member$" "$tmp_dir/members" || { echo "FFmpeg 压缩包缺少 $member" >&2; return 1; }
    done
    run_as_root rm -rf -- "$FFMPEG_INSTALL_DIR" || return 1
    run_as_root mkdir -p -- "$FFMPEG_INSTALL_DIR" || return 1
    run_as_root tar -xJf "$tmp_dir/ffmpeg.tar.xz" -C "$FFMPEG_INSTALL_DIR" --strip-components=1 --no-same-owner || return 1
    local pc
    for pc in "$FFMPEG_PC_DIR"/*.pc; do
        [ -f "$pc" ] || return 1
        run_as_root sed -i "s|^prefix=.*|prefix=$FFMPEG_INSTALL_DIR|" "$pc" || return 1
    done
    printf '%s\n' "$FFMPEG_INSTALL_DIR/lib" | run_as_root tee /etc/ld.so.conf.d/ffmpeg.conf >/dev/null || return 1
    run_as_root ldconfig || return 1
    run_as_root ln -sf "$FFMPEG_INSTALL_DIR/bin/ffmpeg" /usr/local/bin/ffmpeg || return 1
    run_as_root ln -sf "$FFMPEG_INSTALL_DIR/bin/ffprobe" /usr/local/bin/ffprobe || return 1
    prepend_path PATH "$FFMPEG_INSTALL_DIR/bin"
    prepend_path LD_LIBRARY_PATH "$FFMPEG_INSTALL_DIR/lib"
    prepend_path PKG_CONFIG_PATH "$FFMPEG_PC_DIR"
    check_ffmpeg || return 1
    install_environment_profile ffmpeg || return 1
    echo "FFmpeg 安装完成。"
)

# 步骤 D: 下载并编译 NVENC 容器多卡枚举修复补丁 (libnvenc_fix.so)
install_nvenc_fix() (
    check_install_platform || return 1
    local tmp_dir
    if ! command -v gcc >/dev/null; then
        run_as_root apt-get update || return 1
        run_as_root apt-get install -y gcc build-essential || return 1
    fi
    tmp_dir=$(mktemp -d) || return 1
    trap 'rm -rf -- "$tmp_dir"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    if ! fetch_verified "$NVENC_FIX_SRC_URL" "$tmp_dir/nvenc_fix.c" "$NVENC_FIX_SHA256"; then
        fetch_verified "$NVENC_FIX_SRC_MIRROR" "$tmp_dir/nvenc_fix.c" "$NVENC_FIX_SHA256" || return 1
    fi
    gcc -shared -fPIC -O2 -o "$tmp_dir/libnvenc_fix.so" "$tmp_dir/nvenc_fix.c" -ldl || return 1
    run_as_root install -m 755 "$tmp_dir/libnvenc_fix.so" "$NVENC_FIX_SO" || return 1
    echo "固定版本 NVENC 补丁已安装。"
    check_nvenc_and_patch
)

# 步骤 E: 创建并配置 Python 虚拟环境 (venv)
setup_python_venv() {
    init_project_root || return 1
    check_install_platform || return 1
    run_as_root apt-get update || return 1
    run_as_root apt-get install -y --no-install-recommends python3 python3-venv python3-pip p7zip-full aria2 || return 1
    if [ ! -x "$VENV_DIR/bin/python3" ] || [ ! -f "$VENV_DIR/pyvenv.cfg" ]; then
        python3 -m venv "$VENV_DIR" || return 1
    fi
    "$VENV_DIR/bin/python3" -m pip install --upgrade pip || return 1
    # Same pins as the runtime packager, so tools/prepare_apisr.py writes the same model.
    "$VENV_DIR/bin/python3" -m pip install --no-cache-dir 'numpy==2.5.2' 'onnx==1.22.0' tqdm py7zr || return 1
    printf 'Python 虚拟环境配置完成: %s\n' "$VENV_DIR"
}

# 步骤 F: 编译 AnimeJaNai-Inference
build_project() (
    init_project_root || return 1
    activate_build_environment || return 1
    resolve_tensorrt || return 1
    prepend_path LD_LIBRARY_PATH "$TRT_LIB_DIR"
    local nvcc_path="$NVCC_BIN"
    echo -e "${CYAN}1. 配置 CMake 构建工程...${NC}"
    echo -e "  - 源码目录:     ${PROJECT_ROOT}"
    echo -e "  - 构建目录:     ${PROJECT_ROOT}/build"
    echo -e "  - CUDACXX:      ${nvcc_path}"
    echo -e "  - AJI_TRT_ROOT: $TRT_ROOT"
    echo -e "  - CUDA_ARCH:    native (适配当前 GPU)"
    
    cd -- "$PROJECT_ROOT" || return 1
    if ! CUDACXX="$nvcc_path" cmake -B "${PROJECT_ROOT}/build" -S "${PROJECT_ROOT}" \
        -DAJI_TRT_ROOT="$TRT_ROOT" \
        -DAJI_TRT_INCLUDE="$TRT_INCLUDE_DIR" -DAJI_TRT_LIB="$TRT_LIB_DIR" \
        -DCMAKE_CUDA_COMPILER="$NVCC_BIN" \
        -DCMAKE_CUDA_ARCHITECTURES=native \
        -DCMAKE_BUILD_TYPE=Release; then
        echo -e "${RED}[错误] CMake 配置失败，请检查上方报错信息。${NC}"
        return 1
    fi

    echo -e "\n${CYAN}2. 正在执行多核编译 (cmake --build build -j$(nproc))...${NC}"
    if ! cmake --build "${PROJECT_ROOT}/build" -j"$(nproc)"; then
        echo -e "${RED}[错误] 编译失败！${NC}"
        return 1
    fi

    echo -e "\n${CYAN}3. 验证构建结果产物...${NC}"
    local missing=0
    for bin in "${PROJECT_ROOT}/build/libaji.so" "${PROJECT_ROOT}/build/libaji_trt.so" "${PROJECT_ROOT}/build/aji_harness" "${PROJECT_ROOT}/build/aji_encode"; do
        if [ -f "$bin" ]; then
            echo -e "  - $(basename "$bin"): ${GREEN}✔ $(du -h "$bin" | cut -f1)${NC}"
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
)

# ---- 模型（与打包 Runtime 共用 scripts/runtime-models.tsv）---------------
# 源码安装与 Runtime 包使用同一布局：$MODELS_DIR/<conf 中的模型名>.onnx 与
# $MODELS_DIR/rife/rife_v<版本>.onnx。Engine 不再预构建：native kernel 直接运行
# 支持的模型，其余模型由 backend 首次使用时构建并缓存。

load_model_assets() {
    # shellcheck source=scripts/model-assets.sh
    source "$PROJECT_ROOT/scripts/model-assets.sh"
}

ensure_model_python() {
    [ -x "$VENV_DIR/bin/python3" ] || { echo "请先运行 --venv 配置模型校验环境。" >&2; return 1; }
    "$VENV_DIR/bin/python3" -c 'import onnx, numpy' || return 1
}

download_model_file() {
    ensure_model_python || return 1
    printf '获取并校验模型: %s\n' "$4"
    local sha_args=()
    [ -z "${5:-}" ] || [ "$5" = "-" ] || sha_args=(--sha256 "$5")
    "$VENV_DIR/bin/python3" "$PROJECT_ROOT/scripts/download_model.py" "$2" "$1" \
        --mirror "$3" "${sha_args[@]}"
}

fetch_https() {
    local url="$1" destination="$2"
    [[ "$url" == https://* ]] || return 1
    if command -v curl >/dev/null; then
        curl --fail --location --proto '=https' --proto-redir '=https' \
            --connect-timeout 20 --max-time 1800 --retry 2 --output "$destination" "$url"
    elif command -v wget >/dev/null; then
        wget --https-only --timeout=30 --tries=3 -O "$destination" -- "$url"
    else
        echo "需要 curl 或 wget 下载依赖。" >&2
        return 1
    fi
}

prepare_apisr_model() (
    ensure_model_python || return 1
    local target_path="$1" tmp_dir
    if [ -s "$target_path" ]; then
        "$VENV_DIR/bin/python3" -c 'import onnx,sys; onnx.checker.check_model(sys.argv[1])' "$target_path" || return 1
        echo "已校验 APISR: $target_path"
        return 0
    fi
    tmp_dir=$(mktemp -d "$MODELS_DIR/.apisr.XXXXXX") || return 1
    trap 'rm -rf -- "$tmp_dir"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    local local_model="$PROJECT_ROOT/models/2x_APISR_RRDB_GAN_fp16.onnx"
    if [ -s "$local_model" ]; then
        cp -- "$local_model" "$tmp_dir/model.onnx" || return 1
        "$VENV_DIR/bin/python3" -c 'import onnx,sys; onnx.checker.check_model(sys.argv[1])' "$tmp_dir/model.onnx" || return 1
    else
        "$VENV_DIR/bin/python3" "$PROJECT_ROOT/tools/prepare_apisr.py" --output "$tmp_dir/model.onnx" || return 1
    fi
    mv -f -- "$tmp_dir/model.onnx" "$target_path" || return 1
    echo "APISR 模型已准备好。"
)

# RIFE: one archive (upstream's fp16 RIFE release) holds every version;
# install_model_groups downloads it at most once into $RIFE_ARCHIVE_DIR.
prepare_rife_model() (
    local target="$1" file="$2" url="$3" sha="$4" archive tmp_dir
    if aji_sha_ok "$target" "$sha"; then
        echo "已校验 RIFE: $target"
        return 0
    fi
    archive="$RIFE_ARCHIVE_DIR/$(basename -- "$url")"
    if [ ! -s "$archive" ]; then
        echo "获取上游 fp16 RIFE 模型包（约 320 MB，含全部版本）..."
        fetch_https "$url" "$archive.part" && mv -f -- "$archive.part" "$archive" || return 1
    fi
    command -v 7z >/dev/null 2>&1 || ensure_model_python || return 1
    tmp_dir=$(mktemp -d "$MODELS_DIR/.rife.XXXXXX") || return 1
    trap 'rm -rf -- "$tmp_dir"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    aji_rife_extract "$archive" "$file" "$target" "$VENV_DIR/bin/python3" "$tmp_dir" || return 1
    echo "RIFE 已准备好: $target"
)

# One manifest row: group kind file sha url mirror.
install_model_row() {
    local _group="$1" kind="$2" file="$3" sha="$4" url="$5" mirror="$6"
    local target="$MODELS_DIR/$file"
    [ "$mirror" != "-" ] || mirror=""
    case "$kind" in
        onnx)
            if aji_sha_ok "$target" "$sha"; then
                printf '已存在并校验通过: %s\n' "$file"
                return 0
            fi
            download_model_file "$target" "$url" "$mirror" "$file" "$sha" ;;
        apisr) prepare_apisr_model "$target" ;;
        rife) prepare_rife_model "$target" "$file" "$url" "$sha" ;;
        *) echo "未知模型类型: $kind" >&2; return 1 ;;
    esac
}

install_model_groups() (
    local row=() failed=0
    RIFE_ARCHIVE_DIR=$(mktemp -d "$MODELS_DIR/.rife-archive.XXXXXX") || return 1
    trap 'rm -rf -- "$RIFE_ARCHIVE_DIR"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    while IFS=$'\t' read -r -u 3 -a row; do
        install_model_row "${row[@]}" || failed=1
    done 3< <(aji_model_rows "$@")
    return "$failed"
)

# 自定义 ONNX：URL 或本地路径，放入 $MODELS_DIR，在生成器中选“自定义模型”。
import_custom_model() {
    local source="" name dest url_hash
    load_model_assets || return 1
    prompt_value source "请输入 ONNX 下载 URL 或本地路径: " || return 1
    source=$(printf '%s\n' "$source" | sed -e "s/^['\"]//" -e "s/['\"]$//")
    if [ -f "$source" ]; then
        name=$(basename -- "$source")
    else
        name=$(basename -- "${source%%[?#]*}")
    fi
    if [[ ! "$name" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*\.onnx$ ]]; then
        url_hash=$(printf '%s' "$source" | sha256sum) || return 1
        name="custom_${url_hash:0:16}.onnx"
    fi
    if aji_model_rows | cut -f3 | grep -Fxq -- "$name"; then
        echo "文件名 ${name} 与内置方案的模型相同，请改名后再导入。" >&2
        return 1
    fi
    dest="$MODELS_DIR/$name"
    if [ -e "$dest" ]; then
        echo "已存在同名模型: $dest" >&2
        return 1
    fi
    if [ -f "$source" ]; then
        ensure_model_python || return 1
        "$VENV_DIR/bin/python3" -c 'import onnx,sys; onnx.checker.check_model(sys.argv[1])' "$source" || return 1
        cp -- "$source" "$dest.part" && mv -f -- "$dest.part" "$dest" || return 1
    else
        download_model_file "$dest" "$source" "" "$name" || return 1
    fi
    echo -e "${GREEN}✔ 已导入: ${dest}${NC}"
    echo "在 generate_cmd.sh 中选择“自定义模型”即可使用；Engine 会在首次运行时按需构建。"
}

# 步骤 G: 下载模型（Engine 首次使用时自动构建）
download_models() {
    init_project_root || return 1
    load_model_assets || return 1
    print_header
    echo -e "${BOLD}${MAGENTA}[步骤] 下载超分 / 插帧模型${NC}"
    echo -e "模型目录: ${MODELS_DIR}（与 Runtime 包的 onnx/ 布局相同）"
    echo -e "Engine 无需预构建：native kernel 直接运行支持的模型，其余模型首次运行时自动构建并缓存。\n"
    mkdir -p "$MODELS_DIR" || return 1

    echo -e "请选择要获取的模型："
    echo -e "  ${BOLD}1)${NC} ${GREEN}全部（所有内置方案所需模型，推荐）${NC}"
    echo -e "  ${BOLD}2)${NC} AnimeJaNai HD V3.1 Performance / Balanced（含 Sharp1）(Slot 1001-1003, 2001, 2002)"
    echo -e "  ${BOLD}3)${NC} AnimeJaNai SD Compact (Slot 1001-1003 的 SD 链, 2003)"
    echo -e "  ${BOLD}4)${NC} RealESRGAN AnimeVideo-v3 (4x, Slot 2004)"
    echo -e "  ${BOLD}5)${NC} APISR 2x RRDB GAN (Slot 2005)"
    echo -e "  ${BOLD}6)${NC} Real-ESRGAN x4plus Anime 6B (4x, Slot 2006)"
    echo -e "  ${BOLD}7)${NC} RIFE v4.25 / v4.26 fp16 (Slot 2025, 2026, 3025, 3026 与自定义插帧)"
    echo -e "  ${BOLD}8)${NC} 自定义 ONNX 模型（下载链接或本地路径）"
    local model_choice=""
    prompt_value model_choice "请输入选项 [1-8, 默认 1]: " 1 || return 1
    valid_index "$model_choice" 1 8 || return 1
    case "$((10#$model_choice))" in
        1) install_model_groups ;;
        2) install_model_groups hd ;;
        3) install_model_groups sd ;;
        4) install_model_groups animevideo ;;
        5) install_model_groups apisr ;;
        6) install_model_groups anime6b ;;
        7) install_model_groups rife ;;
        8) import_custom_model ;;
    esac || return 1
    report_legacy_model_files
}

# ---- 旧安装迁移 -------------------------------------------------------------
# 旧版 deploy.sh 用短文件名保存模型，并在 $MODELS_DIR 中手工构建可读名称的
# Engine。新的 conf 布局按完整模型名查找；libaji 首次运行时会删除所有不带
# 当前 ".trt-<版本>.gpu-<GPU>" 后缀的 *.engine。

LEGACY_MODEL_NAMES=(performance.onnx balanced.onnx performance_sharp1.onnx
                    balanced_sharp1.onnx sd_compact.onnx realesrgan_anime6b.onnx)

# Old engine artifacts: hand-built engines (not backend cache names) and
# their .label/.build.log, timing caches and leftover build directories.
legacy_engine_files() {
    local dir file base
    for dir in "$MODELS_DIR" "$MODELS_DIR/rife"; do
        [ -d "$dir" ] || continue
        while IFS= read -r -d '' file; do
            base=$(basename -- "$file")
            case "$base" in
                *.trt-*.gpu-*.engine|*.trt-*.gpu-*.engine.onnxinfo) continue ;;
            esac
            printf '%s\0' "$file"
        done < <(find "$dir" -maxdepth 1 \( -type f \( -name '*.engine' -o -name '*.engine.label' \
                     -o -name '*.engine.build.log' -o -name '*.timing.cache' \) \
                     -o -type d -name '.aji-engine.*' \) -print0 2>/dev/null | sort -z)
    done
}

report_legacy_model_files() {
    local legacy=() name files=()
    for name in "${LEGACY_MODEL_NAMES[@]}"; do
        [ -f "$MODELS_DIR/$name" ] && legacy+=("$name")
    done
    mapfile -d '' -t files < <(legacy_engine_files)
    if [ ${#legacy[@]} -gt 0 ] || [ ${#files[@]} -gt 0 ]; then
        echo -e "\n${YELLOW}ℹ 检测到旧版安装留下的文件（${#legacy[@]} 个短名称模型，${#files[@]} 个手工 Engine 相关文件）。"
        echo -e "  运行 ./deploy.sh --migrate 可按校验和改名旧模型并清理旧 Engine。${NC}"
    fi
}

migrate_legacy_models() {
    init_project_root || return 1
    load_model_assets || return 1
    print_header
    echo -e "${BOLD}${MAGENTA}[步骤] 迁移旧版模型与 Engine${NC}"
    echo -e "模型目录: ${MODELS_DIR}\n"
    [ -d "$MODELS_DIR" ] || { echo "模型目录不存在，无需迁移。"; return 0; }

    local name path actual target row reply renamed=0 duplicates=() unknown=()
    local -A by_sha=()
    while IFS=$'\t' read -r -u 3 -a row; do
        [ "${row[1]}" = onnx ] && [ "${row[3]}" != "-" ] && by_sha[${row[3]}]=${row[2]}
    done 3< <(aji_model_rows)

    for name in "${LEGACY_MODEL_NAMES[@]}"; do
        path="$MODELS_DIR/$name"
        [ -f "$path" ] || continue
        actual=$(sha256sum < "$path") || return 1
        actual=${actual%% *}
        target=${by_sha[$actual]:-}
        if [ -z "$target" ]; then
            printf '  ? %s：内容与任何内置模型都不一致，需要重新下载对应模型\n' "$name"
            unknown+=("$path")
        elif [ -e "$MODELS_DIR/$target" ]; then
            printf '  = %s：与已存在的 %s 相同（重复文件）\n' "$name" "$target"
            duplicates+=("$path")
        else
            printf '  → %s 可改名为 %s\n' "$name" "$target"
            prompt_value reply "    改名？[Y/n]: " Y || return 1
            if [[ "${reply:-Y}" =~ ^[Yy]$ ]]; then
                mv -n -- "$path" "$MODELS_DIR/$target" || return 1
                renamed=$((renamed + 1))
            fi
        fi
    done
    [ "$renamed" -eq 0 ] || echo -e "${GREEN}✔ 已改名 ${renamed} 个模型。${NC}"

    local old=("${duplicates[@]}" "${unknown[@]}")
    if [ ${#old[@]} -gt 0 ]; then
        echo -e "\n以下旧模型文件不再被任何方案使用："
        du -ch -- "${old[@]}" | sed 's/^/    /'
        prompt_value reply "删除这些文件？[y/N]: " N || return 1
        if [[ "${reply:-N}" =~ ^[Yy]$ ]]; then
            rm -f -- "${old[@]}" || return 1
            echo -e "${GREEN}✔ 已删除。${NC}"
        fi
    fi

    local engines=()
    mapfile -d '' -t engines < <(legacy_engine_files)
    if [ ${#engines[@]} -gt 0 ]; then
        echo -e "\n以下是旧版手工构建的 Engine 及其附属文件（Engine 现在首次使用时自动构建）："
        du -csh -- "${engines[@]}" | sed 's/^/    /'
        echo -e "${YELLOW}即使不删除，下次用 ${MODELS_DIR} 运行 conf 方案时 libaji 也会删除其中的 *.engine。${NC}"
        prompt_value reply "删除这些文件？[y/N]: " N || return 1
        if [[ "${reply:-N}" =~ ^[Yy]$ ]]; then
            rm -rf -- "${engines[@]}" || return 1
            echo -e "${GREEN}✔ 已删除。${NC}"
        fi
    fi

    local missing=0
    while IFS=$'\t' read -r -u 3 -a row; do
        [ -s "$MODELS_DIR/${row[2]}" ] || missing=$((missing + 1))
    done 3< <(aji_model_rows)
    if [ "$missing" -gt 0 ]; then
        echo -e "\n${YELLOW}仍有 ${missing} 个内置方案模型缺失，可运行 ./deploy.sh --models 下载。${NC}"
    else
        echo -e "\n${GREEN}✔ 内置方案所需模型齐全。${NC}"
    fi
}

# 步骤 H: 截取 1 分钟 example.mkv 进行超分测试
run_test_clip() (
    init_project_root || return 1
    print_header
    echo -e "${BOLD}${MAGENTA}[步骤] 截取 1 分钟视频并运行超分测试${NC}\n"

    if [ ! -f "${PROJECT_ROOT}/build/aji_encode" ]; then
        echo -e "${RED}[错误] 未找到 ${PROJECT_ROOT}/build/aji_encode！请先执行编译步骤。${NC}"
        return 1
    fi

    local manual_path="" use_found="" engine_idx=""
    local test_source="${PROJECT_ROOT}/example.mkv"
    if [ ! -f "$test_source" ]; then
        local found_mkv="" videos=()
        mapfile -d '' -t videos < <(find "$PROJECT_ROOT" -maxdepth 1 -type f -name '*.mkv' ! -name 'test_clip*' -print0 | sort -z)
        [ "${#videos[@]}" -eq 0 ] || found_mkv=${videos[0]}
        echo -e "${YELLOW}未在 ${PROJECT_ROOT} 目录下找到默认的 example.mkv 文件！${NC}"
        if [ -n "$found_mkv" ]; then
            echo -e "检测到目录中存在视频文件: ${BOLD}${found_mkv}${NC}"
            prompt_value use_found "是否直接使用该视频作为测试源？[Y/n]: " Y || return 1
            use_found=${use_found:-Y}
            if [[ "$use_found" =~ ^[Yy]$ ]]; then
                test_source="$found_mkv"
            fi
        fi

        if [ ! -f "$test_source" ]; then
            prompt_value manual_path "请输入测试视频文件的相对或绝对路径: " || return 1
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

    local src_width

    src_width=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of default=noprint_wrappers=1:nokey=1 "$test_source" 2>/dev/null | head -n 1) || return 1
    local src_height
    src_height=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of default=noprint_wrappers=1:nokey=1 "$test_source" 2>/dev/null | head -n 1) || return 1
    local src_codec
    src_codec=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "$test_source" 2>/dev/null | head -n 1) || return 1
    
    if ! valid_index "$src_width" 2 65536 || ! valid_index "$src_height" 2 65536; then
        echo "无法读取有效的视频尺寸。" >&2; return 1
    fi
    echo -e "  - 视频编码: ${src_codec:-未知}"
    echo -e "  - 原始分辨率: ${src_width} x ${src_height}"

    # 内置 Performance 方案（Slot 1003），与生成器使用同一 conf 与模型布局
    load_model_assets || return 1
    local test_conf="${PROJECT_ROOT}/packaging/ubuntu24/runtime/animejanai.conf"
    local test_slot=1003 trtexec_bin model
    resolve_tensorrt || return 1
    trtexec_bin=$TRTEXEC_BIN
    for model in 2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx \
                 2x_AnimeJaNai_SD_V1beta34_Compact_1x3xHxW_dyn-HW_strong_fp16_op21_dynamo.onnx; do
        if [ ! -s "$MODELS_DIR/$model" ]; then
            echo -e "${RED}[错误] 缺少模型 ${MODELS_DIR}/${model}，请先运行 ./deploy.sh --models。${NC}"
            return 1
        fi
    done

    local test_dir
    test_dir=$(mktemp -d "$PROJECT_ROOT/build/deploy-test.XXXXXX") || return 1
    local clip_input="$test_dir/input.mkv"
    local clip_output="$test_dir/upscaled.mkv"
    local start_time="00:00:30"
    
    echo -e "\n${CYAN}正在从 ${test_source} 截取 1 分钟片段 (从 ${start_time} 开始，流拷贝无损快速提取)...${NC}"
    if ! ffmpeg -nostdin -y -ss "$start_time" -i "$test_source" -t 60 -c copy "$clip_input" || [ ! -s "$clip_input" ]; then
        echo -e "${RED}[错误] 截取测试片段失败！${NC}"
        return 1
    fi
    echo -e "${GREEN}✔ 截取完成: ${clip_input} ($(du -h "$clip_input" | cut -f1))${NC}"

    local vcodec="hevc_nvenc" preload_fix=""
    if check_nvenc_and_patch; then
        preload_fix=$NVENC_PRELOAD
    else
        echo "NVENC 不可用，使用 libx265。"
        vcodec=libx265
    fi

    echo -e "\n${CYAN}==============================================================================${NC}"
    echo -e "${BOLD}${MAGENTA}开始执行超分推理${NC}"
    echo -e "  - 输入文件: ${clip_input}"
    echo -e "  - 输出文件: ${clip_output}"
    echo -e "  - 编码器:   ${vcodec}"
    echo -e "  - 方案:     Performance (Slot ${test_slot}，native kernel 优先，必要时自动构建 Engine)"
    if [ -n "$preload_fix" ]; then
        echo -e "  - LD_PRELOAD: ${preload_fix} (NVENC 容器多卡修复生效)"
    fi
    echo -e "${CYAN}------------------------------------------------------------------------------${NC}"

    prepend_path LD_LIBRARY_PATH "$FFMPEG_INSTALL_DIR/lib"
    prepend_path LD_LIBRARY_PATH "$PROJECT_ROOT/build"
    local encode_env=(env)
    [ -z "$preload_fix" ] || encode_env+=("LD_PRELOAD=$preload_fix${LD_PRELOAD:+ $LD_PRELOAD}")
    cd -- "$PROJECT_ROOT" || return 1
    local start_ts
    start_ts=$(date +%s) || return 1
    "${encode_env[@]}" ./build/aji_encode \
        --input "$clip_input" \
        --output "$clip_output" \
        --conf "$test_conf" \
        --slot "$test_slot" \
        --model-dir "$MODELS_DIR" \
        --rife-model-dir "$MODELS_DIR/rife" \
        --trtexec "$trtexec_bin" \
        --vcodec "$vcodec" \
        --overwrite

    local encode_status=$?
    local end_ts
    end_ts=$(date +%s) || return 1
    local elapsed
    elapsed=$((end_ts - start_ts)) || return 1

    if [ $encode_status -eq 0 ] && [ -f "$clip_output" ]; then
        echo -e "\n${GREEN}==============================================================================${NC}"
        echo -e "${BOLD}${GREEN}🎉 超分测试成功完成！${NC}"
        echo -e "  - 耗时:       ${elapsed} 秒"
        echo -e "  - 输出大小:   $(du -h "$clip_output" | cut -f1)"
        echo -e "  - 输出分辨率: $(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$clip_output" 2>/dev/null || echo "检测中...")"
        echo -e "${GREEN}==============================================================================${NC}"
    else
        echo -e "\n${RED}✖ 超分测试失败，请检查上方日志输出！${NC}"
        return 1
    fi
)

# 一键全自动部署流程
one_click_setup() {
    if [ ! -t 0 ] && [ "${ASSUME_DEFAULTS:-0}" != 1 ]; then
        echo "--all 非交互执行需要 --yes；可用 --cuda-version 和 --ffmpeg-variant 指定版本。" >&2
        return 1
    fi
    init_project_root || return 1
    print_header
    echo -e "${BOLD}${MAGENTA}开始执行一键全自动部署流程...${NC}\n"

    check_nvidia_driver || return 1
    
    echo -e "\n>>> 步骤 1/6: 配置 NVIDIA 官方 Network 源并清理旧 Keyring"
    if ! setup_nvidia_network_repo; then
        echo -e "${RED}[错误] 步骤 1 失败！${NC}"
        return 1
    fi

    echo -e "\n>>> 步骤 2/6: 安装严格要求的 CUDA Toolkit 13.x 与 TensorRT 11.x"
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
    prepend_path PATH "$FFMPEG_INSTALL_DIR/bin"
    prepend_path LD_LIBRARY_PATH "$FFMPEG_INSTALL_DIR/lib"
    if ! check_nvenc_and_patch; then
        if [ "$NVENC_PATCH_NEEDED" = 1 ]; then
            install_nvenc_fix || return 1
        else
            echo "未检测到容器枚举故障，跳过补丁；编码时可使用 CPU 编码器。"
        fi
    fi

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
    echo -e "接下来您可以选择 [9] 下载模型，然后选择 [10] 运行 1 分钟测试；Engine 无需预构建。"
    echo -e "${GREEN}==============================================================================${NC}"
}

# 交互式主菜单
main_menu() {
    local choice=""
    [ "${ASSUME_DEFAULTS:-0}" != 1 ] || { echo "--yes 需要指定操作。" >&2; return 2; }
    init_project_root || return 1
    while true; do
        print_header
        echo -e "${BOLD}请选择操作：${NC}"
        echo -e "  ${BOLD}[1]${NC}  ${CYAN}全面环境自检与诊断${NC} (Check All Environment)"
        echo -e "  ${BOLD}[2]${NC}  ${GREEN}一键全自动安装与编译${NC} (One-Click Setup: CUDA 13.x/TRT 11.x -> FFmpeg -> Fix -> Build)"
        echo -e "  ----------------------------------------------------------------------"
        echo -e "  ${BOLD}[3]${NC}  清理旧 Keyring 并配置 NVIDIA 官方源 (Fix AutoDL Keyring & Setup Repo)"
        echo -e "  ${BOLD}[4]${NC}  ${MAGENTA}安装严格要求的 CUDA Toolkit 13.x 与 TensorRT 11.x${NC} (Install CUDA 13.x & TRT 11.x)"
        echo -e "  ${BOLD}[5]${NC}  安装编译工具链与 BtbN FFmpeg Shared (Install Build Tools & FFmpeg)"
        echo -e "  ${BOLD}[6]${NC}  ${MAGENTA}下载并编译 NVENC 容器多卡修复补丁${NC} (Build libnvenc_fix.so)"
        echo -e "  ${BOLD}[7]${NC}  创建并配置 Python 虚拟环境 (Setup Python .venv)"
        echo -e "  ${BOLD}[8]${NC}  编译 AnimeJaNai-Inference (CMake Build)"
        echo -e "  ----------------------------------------------------------------------"
        echo -e "  ${BOLD}[9]${NC}  ${YELLOW}下载超分 / 插帧模型${NC} (Download Models; engines build on first use)"
        echo -e "  ${BOLD}[10]${NC} ${MAGENTA}截取 1 分钟 example.mkv 进行超分测试${NC} (Run 1-Min Encode Test)"
        echo -e "  ----------------------------------------------------------------------"
        echo -e "  ${BOLD}[11]${NC} ${CYAN}一键清理系统中旧版 CUDA 软件包 (如 12.x/11.x 释放空间)${NC} (Purge Old CUDA Packages)"
        echo -e "  ${BOLD}[12]${NC} ${GREEN}交互式超分任务配置向导与命令生成器 (不自动执行)${NC} (Task Wizard & Generator)"
        echo -e "  ${BOLD}[13]${NC} 迁移旧版模型文件名并清理旧 Engine (Migrate Legacy Models & Engines)"
        echo -e "  ${BOLD}[0]${NC}  退出 (Exit)"
        echo -e "${CYAN}------------------------------------------------------------------------------${NC}"
        prompt_value choice "请输入选项数字 [0-13]: " || return 1

        case "$choice" in
            1) diagnose_all ;;
            2) one_click_setup ;;
            3) setup_nvidia_network_repo ;;
            4) install_cuda_and_tensorrt ;;
            5) install_build_tools_and_ffmpeg ;;
            6) install_nvenc_fix ;;
            7) setup_python_venv ;;
            8) build_project ;;
            9) download_models ;;
            10) run_test_clip ;;
            11) purge_old_cuda_packages ;;
            12) 
                if [ -f "${PROJECT_ROOT}/generate_cmd.sh" ]; then
                    bash "${PROJECT_ROOT}/generate_cmd.sh"
                else
                    echo -e "${RED}[错误] 未找到 generate_cmd.sh 脚本！${NC}"
                fi
                ;;
            13) migrate_legacy_models ;;
            0) echo -e "\n${GREEN}感谢使用，再见！${NC}"; exit 0 ;;
            *) echo -e "\n${RED}无效选项，请重新输入！${NC}" ;;
        esac

        echo -e "\n按回车键返回主菜单..."
        read -r || return 0
    done
}


show_help() {
    cat <<'HELP'
用法: ./deploy.sh [操作] [参数]
无操作时打开交互菜单。
操作: --check --all --repo --cuda --ffmpeg --nvenc-fix --venv --build
      --models --migrate --clean-cuda --test --gen --help
  --models   按 conf 布局下载模型到 ~/models（含 RIFE fp16）；不构建 Engine
  --migrate  旧版短名称模型按校验和改名，列出并（确认后）删除旧 Engine
  --engine   已废弃，等同 --models（Engine 现在首次使用时自动构建）
参数:
  --project-root PATH     源码目录（默认脚本目录）
  --cuda-root PATH        CUDA Toolkit 13.x 根目录
  --trt-root PATH         TensorRT 11 根目录
  --trtexec PATH          配套 trtexec 绝对路径
  --cuda-version 13.x     安装指定 CUDA 版本
  --ffmpeg-variant NAME   n8.1 或 master（固定日期构建）
  --yes                   明确接受已有默认选项，用于非交互部署
--check 不安装、不克隆、不修改 CUDA 系统链接；依赖缺失返回非零。
HELP
}

main() {
    local action="" argument
    while [ "$#" -gt 0 ]; do
        argument=$1; shift
        case "$argument" in
            --help|-h) show_help; return 0 ;;
            --yes) ASSUME_DEFAULTS=1 ;;
            --project-root|--cuda-root|--trt-root|--trtexec|--cuda-version|--ffmpeg-variant)
                [ "$#" -gt 0 ] && [ -n "$1" ] || { echo "参数缺少值: $argument" >&2; return 2; }
                case "$argument" in
                    --project-root) PROJECT_ROOT=$1 ;;
                    --cuda-root) CUDA_ROOT=$1 ;;
                    --trt-root) TRT_ROOT=$1 ;;
                    --trtexec) TRTEXEC_BIN=$1 ;;
                    --cuda-version) CUDA_VERSION=$1 ;;
                    --ffmpeg-variant) FFMPEG_VARIANT=$1 ;;
                esac
                shift ;;
            --check|check|-c|--all|all|--install-all|--repo|--cuda|--ffmpeg|--nvenc-fix|--fix|--venv|--build|--engine|--models|--migrate|--clean-cuda|--purge-cuda|--test|--run|--gen|--generate|--wizard)
                [ -z "$action" ] || { echo "每次只能指定一个操作。" >&2; return 2; }
                action=$argument ;;
            *) echo "未知参数: $argument" >&2; return 2 ;;
        esac
    done
    # 删除继承自旧版脚本的空库路径项；不更改用户的其他条目。
    prepend_path LD_LIBRARY_PATH ""
    local path_name
    for path_name in CUDA_ROOT TRT_ROOT TRTEXEC_BIN; do
        if [ -n "${!path_name}" ]; then
            printf -v "$path_name" '%s' "$(realpath -m -- "${!path_name}")" || return 1
        fi
    done
    init_project_root || return 1
    case "$action" in
        --check|check|-c) diagnose_all ;;
        --all|all|--install-all) one_click_setup ;;
        --repo) setup_nvidia_network_repo ;;
        --cuda) install_cuda_and_tensorrt ;;
        --ffmpeg) install_build_tools_and_ffmpeg ;;
        --nvenc-fix|--fix) install_nvenc_fix ;;
        --venv) setup_python_venv ;;
        --build) build_project ;;
        --engine)
            echo "--engine 已废弃：Engine 现在首次使用时自动构建，改为下载模型。" >&2
            download_models ;;
        --models) download_models ;;
        --migrate) migrate_legacy_models ;;
        --clean-cuda|--purge-cuda) purge_old_cuda_packages ;;
        --test) run_test_clip ;;
        --run|--gen|--generate|--wizard) bash "$PROJECT_ROOT/generate_cmd.sh" ;;
        '') main_menu ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" = "$0" ]]; then
    main "$@"
fi
