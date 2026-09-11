#!/usr/bin/env bash
# ==============================================================================
# AnimeJaNai-Inference 交互式任务配置与命令生成器 (Command Generator)
# 功能说明：
#   通过向导式问答收集输入视频、处理方案、编码器、画质、位深等参数，
#   生成标准高效的 aji_encode 运行命令与独立执行脚本 (.sh)，
#   默认不自动执行，支持直接复制命令或手动运行生成的脚本。
# ==============================================================================

set -o pipefail

# 终端颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 定位项目根目录。Runtime 包必须优先于 $HOME 中可能存在的源码树，
# 否则从解压目录启动时会误用源码树的 build/aji_encode。
if [ -x "${SCRIPT_DIR}/aji_encode" ] && [ -x "${SCRIPT_DIR}/bin/ffmpeg.real" ] && \
   [ -d "${SCRIPT_DIR}/lib" ]; then
    PROJECT_ROOT="${SCRIPT_DIR}"
elif [ -f "${SCRIPT_DIR}/CMakeLists.txt" ] || [ -f "${SCRIPT_DIR}/build/aji_encode" ]; then
    PROJECT_ROOT="${SCRIPT_DIR}"
elif [ -f "CMakeLists.txt" ] || [ -f "build/aji_encode" ]; then
    PROJECT_ROOT="$(pwd)"
elif [ -d "$HOME/animejanai-inference" ]; then
    PROJECT_ROOT="$HOME/animejanai-inference"
elif [ -d "/root/animejanai-inference" ]; then
    PROJECT_ROOT="/root/animejanai-inference"
else
    PROJECT_ROOT="${SCRIPT_DIR}"
fi

MODELS_DIR="${HOME}/models"
FFMPEG_INSTALL_DIR="/opt/ffmpeg"
CUDA_BIN_DIR="/usr/local/cuda/bin"
CUDA_LIB_DIR="/usr/local/cuda/lib64"
AJI_ENCODE_BIN="${PROJECT_ROOT}/build/aji_encode"
RUNTIME_MODE=0
if [ -x "${PROJECT_ROOT}/bin/ffmpeg.real" ] && [ -d "${PROJECT_ROOT}/lib" ]; then
    # Self-contained Ubuntu runtime package: all tools and shared libraries
    # live beside this script instead of under /opt and /usr/local/cuda.
    RUNTIME_MODE=1
    FFMPEG_INSTALL_DIR="${PROJECT_ROOT}"
    CUDA_BIN_DIR="${PROJECT_ROOT}/bin"
    CUDA_LIB_DIR="${PROJECT_ROOT}/lib"
    AJI_ENCODE_BIN="${PROJECT_ROOT}/aji_encode"
fi
NVENC_FIX_SO="/opt/libnvenc_fix.so"

# 全局配置变量初始化
INPUT_VIDEO=""
INPUT_VIDEOS=()
OUTPUT_VIDEOS=()
SOURCE_WIDTHS=()
SOURCE_HEIGHTS=()
ENGINE_FILE=""
OUTPUT_VIDEO=""
USE_RUNTIME_CONFIG=${RUNTIME_MODE}
RUNTIME_CONFIG="${PROJECT_ROOT}/animejanai.conf"
CONFIG_MODEL_DIR="${PROJECT_ROOT}/onnx"
CONFIG_TRTEXEC="${PROJECT_ROOT}/bin/trtexec.real"
RUNTIME_SLOT=1003
RUNTIME_PROFILE="Performance"
UPSCALE_ENABLED=1
SRC_WIDTH=1920
SRC_HEIGHT=1080
SRC_CODEC="未知"
SRC_FPS="24"
SRC_DURATION="未知"
IS_CLIP=0
CLIP_START="00:01:00"
CLIP_DURATION="60"
VCODEC="hevc_nvenc"
VQUALITY="-cq 18 -preset p7 -tune hq -split_encode_mode 2"
DECODER="nvdec"
PIX_FMT="yuv420p10"
OVERWRITE_FLAG=""
EXTRA_FLAGS=()
RIFE_ENABLED=0
RIFE_MODEL=""
RIFE_MODEL_DIR="${PROJECT_ROOT}/onnx/rife"
RIFE_FACTOR=2
RIFE_ORDER="before"
RIFE_SCD_THRESHOLD="0.150"
GPU_NAME="未检测"
GPU_COMPUTE_CAP=""
# Preserve the previous dual-split default when GPU detection is unavailable.
NVENC_SPLIT_COUNT=2

# Blackwell supports up to a three-way single-session split. Keep the existing
# two-way request elsewhere; NVENC clamps either request to the number of
# encoder engines that physically exist on the selected GPU.
detect_nvenc_split_count() {
    local detected_name=""
    local detected_cap=""

    if command -v nvidia-smi >/dev/null 2>&1; then
        if detected_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null); then
            detected_name=$(printf '%s\n' "$detected_name" \
                | sed -n '1{s/^[[:space:]]*//;s/[[:space:]]*$//;p;}')
        else
            detected_name=""
        fi
        if detected_cap=$(nvidia-smi --query-gpu=compute_cap \
            --format=csv,noheader,nounits 2>/dev/null); then
            detected_cap=$(printf '%s\n' "$detected_cap" \
                | sed -n '1{s/[[:space:]]//g;p;}')
        else
            detected_cap=""
        fi
    fi

    # If NVML is temporarily unavailable in a container, the Runtime's
    # single-architecture TensorRT builder resource is a reliable fallback.
    if [ -z "$detected_cap" ] && [ "$RUNTIME_MODE" -eq 1 ]; then
        if compgen -G "${PROJECT_ROOT}/lib/libnvinfer_builder_resource_sm120.so*" >/dev/null; then
            detected_name=${detected_name:-"Runtime target sm120"}
            detected_cap="12.0"
        elif compgen -G "${PROJECT_ROOT}/lib/libnvinfer_builder_resource_sm100.so*" >/dev/null; then
            detected_name=${detected_name:-"Runtime target sm100"}
            detected_cap="10.0"
        fi
    fi

    [ -n "$detected_name" ] && GPU_NAME="$detected_name"
    GPU_COMPUTE_CAP="$detected_cap"

    case "$GPU_COMPUTE_CAP" in
        10.*|12.*) NVENC_SPLIT_COUNT=3 ;; # Blackwell data-center/consumer
        *)
            case "$GPU_NAME" in
                *Blackwell*|*GeForce*RTX*50[0-9][0-9]*)
                    NVENC_SPLIT_COUNT=3 ;;
                *)  NVENC_SPLIT_COUNT=2 ;; # Retain legacy behaviour
            esac
            ;;
    esac

    VQUALITY="-cq 18 -preset p7 -tune hq -split_encode_mode ${NVENC_SPLIT_COUNT}"
}

print_header() {
    clear 2>/dev/null || true
    echo -e "${CYAN}==============================================================================${NC}"
    echo -e "${BOLD}${MAGENTA}      AnimeJaNai-Inference 交互式 AI 视频任务配置与命令生成器${NC}"
    echo -e "${CYAN}==============================================================================${NC}"
    echo -e " 核心工具: ${BOLD}${AJI_ENCODE_BIN}${NC}"
    echo -e " 系统时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${CYAN}------------------------------------------------------------------------------${NC}\n"
}

# 1. 检查 aji_encode 二进制是否存在
check_binaries() {
    if [ ! -f "$AJI_ENCODE_BIN" ]; then
        echo -e "${RED}[错误] 未找到 aji_encode 编译产物: ${AJI_ENCODE_BIN}${NC}"
        echo -e "${YELLOW}请先运行 ./deploy.sh 选项 [8] 或 [2] 编译项目后再使用此生成器。${NC}"
        exit 1
    fi

    if [ "$RUNTIME_MODE" -eq 1 ]; then
        if [ ! -f "$RUNTIME_CONFIG" ] || [ ! -d "${PROJECT_ROOT}/onnx" ] || \
           [ ! -x "${PROJECT_ROOT}/bin/trtexec.real" ]; then
            echo -e "${RED}[错误] Runtime 包不完整，缺少 animejanai.conf、onnx/ 或 bin/trtexec.real。${NC}"
            echo -e "${YELLOW}请重新解压完整的 Runtime 包后再运行生成器。${NC}"
            exit 1
        fi
    fi
}

# 2. 视频输入选择 (自动真实路径去重)
select_input_video() {
    echo -e "${BOLD}${CYAN}[步骤 1/8] 选择输入视频文件 (Input Video)${NC}"
    
    # 自动搜索常见目录下的视频文件（自动规范化为真实绝对路径并严格去重）
    local search_dirs=("$PWD" "$PROJECT_ROOT" "$HOME" "/root" "$HOME/videos" "$HOME/autodl-tmp" "/root/autodl-tmp" "/root/autodl-fs")
    local raw_found=()

    for d in "${search_dirs[@]}"; do
        if [ -d "$d" ]; then
            local abs_dir="$(readlink -f "$d" 2>/dev/null || echo "$d")"
            while IFS= read -r f; do
                if [ -f "$f" ]; then
                    local abs_file="$(readlink -f "$f")"
                    raw_found+=("$abs_file")
                fi
            done < <(find -L "$abs_dir" -maxdepth 2 -type f \( -iname "*.mkv" -o -iname "*.mp4" -o -iname "*.mov" -o -iname "*.flv" -o -iname "*.ts" -o -iname "*.webm" \) ! -iname "*_upscaled*" ! -iname "*_aji*" ! -iname "test_clip*" ! -iname "temp_clip*" 2>/dev/null | head -n 30)
        fi
    done

    # 严格去重
    local found_videos=()
    if [ ${#raw_found[@]} -gt 0 ]; then
        mapfile -t found_videos < <(printf "%s\n" "${raw_found[@]}" | sort -u)
    fi

    INPUT_VIDEOS=()
    local v_idx token manual_input abs_file existing duplicate
    local selections=()
    if [ ${#found_videos[@]} -gt 0 ]; then
        echo -e "在系统中自动检索到以下视频文件："
        for i in "${!found_videos[@]}"; do
            printf '  %d) %s\n' "$((i+1))" "${found_videos[$i]}"
        done
        echo "  a) 全选"
        echo "  $(( ${#found_videos[@]} + 1 ))) 手动输入路径（可添加多个）"
        while true; do
            read -rp "请选择编号（空格或逗号分隔，如 1 3 或 1,3；默认 1）: " v_idx || return 1
            v_idx=${v_idx:-1}
            if [[ "$v_idx" == [Aa] ]]; then
                INPUT_VIDEOS=("${found_videos[@]}")
                break
            fi
            if [ "$v_idx" = "$(( ${#found_videos[@]} + 1 ))" ]; then
                break
            fi
            read -ra selections <<< "${v_idx//,/ }"
            INPUT_VIDEOS=()
            for token in "${selections[@]}"; do
                if [[ ! "$token" =~ ^[0-9]{1,9}$ ]] ||
                   (( 10#$token < 1 || 10#$token > ${#found_videos[@]} )); then
                    INPUT_VIDEOS=()
                    break
                fi
                abs_file=${found_videos[$((10#$token-1))]}
                duplicate=0
                for existing in "${INPUT_VIDEOS[@]}"; do
                    [ "$existing" = "$abs_file" ] && duplicate=1
                done
                [ "$duplicate" -eq 1 ] || INPUT_VIDEOS+=("$abs_file")
            done
            [ ${#INPUT_VIDEOS[@]} -gt 0 ] && break
            echo -e "${RED}[错误] 请输入有效编号；手动输入选项需单独选择。${NC}"
        done
    fi

    if [ ${#INPUT_VIDEOS[@]} -eq 0 ]; then
        while true; do
            read -rp "请输入视频路径（每行一个，添加完毕后回车）: " manual_input || return 1
            if [ -z "$manual_input" ] && [ ${#INPUT_VIDEOS[@]} -gt 0 ]; then
                break
            fi
            manual_input=$(printf '%s\n' "$manual_input" | sed -e "s/^['\"]//" -e "s/['\"]$//")
            if [ ! -f "$manual_input" ]; then
                echo -e "${RED}[错误] 文件不存在，请重新输入！${NC}"
                continue
            fi
            abs_file=$(readlink -f -- "$manual_input")
            duplicate=0
            for existing in "${INPUT_VIDEOS[@]}"; do
                [ "$existing" = "$abs_file" ] && duplicate=1
            done
            [ "$duplicate" -eq 1 ] || INPUT_VIDEOS+=("$abs_file")
        done
    fi

    echo -e "${GREEN}✔ 已选择 ${#INPUT_VIDEOS[@]} 个视频，将按以下顺序串行处理，共用后续处理参数。${NC}"
    SOURCE_WIDTHS=()
    SOURCE_HEIGHTS=()
    for INPUT_VIDEO in "${INPUT_VIDEOS[@]}"; do
        printf '\n输入视频: %s\n' "$INPUT_VIDEO"
        probe_input_video
        SOURCE_WIDTHS+=("$SRC_WIDTH")
        SOURCE_HEIGHTS+=("$SRC_HEIGHT")
    done
    INPUT_VIDEO=${INPUT_VIDEOS[0]}
    SRC_WIDTH=${SOURCE_WIDTHS[0]}
    SRC_HEIGHT=${SOURCE_HEIGHTS[0]}
}

probe_input_video() {
    SRC_WIDTH=1920
    SRC_HEIGHT=1080
    SRC_CODEC="未知"
    SRC_FPS="24"
    SRC_DURATION="未知"
    # 使用 ffprobe 获取视频详细元数据
    export PATH="${FFMPEG_INSTALL_DIR}/bin:$PATH"
    if command -v ffprobe &>/dev/null; then
        SRC_WIDTH=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$INPUT_VIDEO" 2>/dev/null || echo 1920)
        SRC_HEIGHT=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$INPUT_VIDEO" 2>/dev/null || echo 1080)
        SRC_CODEC=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$INPUT_VIDEO" 2>/dev/null || echo "未知")
        SRC_FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$INPUT_VIDEO" 2>/dev/null || echo "24")
        SRC_DURATION=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$INPUT_VIDEO" 2>/dev/null | awk '{printf "%.1f 分钟", $1/60}' || echo "未知")
    fi

    SRC_WIDTH=${SRC_WIDTH:-1920}
    SRC_HEIGHT=${SRC_HEIGHT:-1080}

    echo -e "${CYAN}视频元信息解析：${NC}"
    echo -e "  - 原始分辨率:   ${BOLD}${SRC_WIDTH} x ${SRC_HEIGHT}${NC}"
    echo -e "  - 编码格式:     ${SRC_CODEC}"
    echo -e "  - 原始帧率:     ${SRC_FPS}"
    echo -e "  - 视频时长:     ${SRC_DURATION}\n"
}

# 3. 选择/匹配 TensorRT Engine 模型 (自动真实路径去重)
select_engine() {
    echo -e "${BOLD}${CYAN}[步骤 2/8] 选择超分 TensorRT Engine 模型${NC}"

    local search_dirs=("$MODELS_DIR" "$PROJECT_ROOT" "$HOME" "/root/models" "/root")
    local raw_engines=()

    for d in "${search_dirs[@]}"; do
        if [ -d "$d" ]; then
            local abs_dir="$(readlink -f "$d" 2>/dev/null || echo "$d")"
            while IFS= read -r f; do
                if [ -f "$f" ]; then
                    local abs_file="$(readlink -f "$f")"
                    raw_engines+=("$abs_file")
                fi
            done < <(find -L "$abs_dir" -maxdepth 2 -type f -name "*.engine" 2>/dev/null | head -n 30)
        fi
    done

    local found_engines=()
    if [ ${#raw_engines[@]} -gt 0 ]; then
        mapfile -t found_engines < <(printf "%s\n" "${raw_engines[@]}" | sort -u)
    fi

    ENGINE_FILE=""

    if [ ${#found_engines[@]} -gt 0 ]; then
        echo -e "检测到以下已构建好的 TensorRT Engine 文件："
        for i in "${!found_engines[@]}"; do
            local esize=$(ls -lh "${found_engines[$i]}" 2>/dev/null | awk '{print $5}')
            local engine_label=""
            if [ -f "${found_engines[$i]}.label" ]; then
                IFS= read -r engine_label < "${found_engines[$i]}.label"
            fi
            [ -z "$engine_label" ] || printf '     %s\n' "$engine_label"
            echo -e "  ${BOLD}$((i+1)))${NC} ${found_engines[$i]} (${GREEN}${esize}${NC})"
        done
        echo -e "  ${BOLD}$(( ${#found_engines[@]} + 1 )))${NC} 手动输入自定义 Engine 路径"

        read -rp "请选择 Engine 编号 [1-$(( ${#found_engines[@]} + 1 )), 默认 1]: " e_idx
        e_idx=${e_idx:-1}

        if [ "$e_idx" -le "${#found_engines[@]}" ] && [ "$e_idx" -ge 1 ]; then
            ENGINE_FILE="${found_engines[$((e_idx-1))]}"
        fi
    fi

    while [ -z "$ENGINE_FILE" ] || [ ! -f "$ENGINE_FILE" ]; do
        echo -e "${YELLOW}未检测到现成 Engine，请手动输入路径或先运行 deploy.sh 选项 [9] 生成：${NC}"
        read -rp "请输入 .engine 文件的绝对路径: " manual_engine
        manual_engine=$(echo "$manual_engine" | sed -e "s/^['\"]//" -e "s/['\"]$//")
        if [ -f "$manual_engine" ]; then
            ENGINE_FILE="$manual_engine"
        else
            echo -e "${RED}[错误] Engine 文件不存在: ${manual_engine}${NC}"
        fi
    done

    ENGINE_FILE=$(readlink -f "$ENGINE_FILE")
    echo -e "${GREEN}✔ 已选择 Engine: ${BOLD}${ENGINE_FILE}${NC}"

    # 针对 4x / 2x 视频模型的适配与优化提示
    local model_identity="$ENGINE_FILE"
    if [ -f "${ENGINE_FILE}.label" ]; then
        IFS= read -r model_identity < "${ENGINE_FILE}.label"
    fi
    if [[ "$model_identity" == *"anime6b"* ]] || [[ "$model_identity" == *"4x"* ]] || [[ "$model_identity" == *"animevideov3"* ]]; then
        echo -e "\n${CYAN}ℹ️  [提示] 所选模型为 4x 模型（直推输出 4x 超高分辨率）。${NC}"
        echo -e "${GREEN}   - 经过底层 CUDA 帧池瘦身与流水线优化，已自动为您加入 --pipeline-depth 2 保证显存平稳运行。${NC}"
        EXTRA_FLAGS+=("--pipeline-depth" "2")
    fi
    echo ""
}

# Runtime 包不依赖预先生成的 .engine。选中 Slot 后，aji_encode
# 会按输入分辨率在首次执行时构建 Engine，后续自动复用缓存。
select_runtime_profile() {
    echo -e "${BOLD}${CYAN}[步骤 2/8] 选择 Runtime 处理方案 (Slot)${NC}"
    echo -e "${GREEN}已识别到免安装 Runtime 环境；无需手动准备 .engine。${NC}"
    echo -e "  ${BOLD}1)${NC} Quality                  - 质量优先超分 (Slot 1001)"
    echo -e "  ${BOLD}2)${NC} Balanced                 - 平衡超分 (Slot 1002)"
    echo -e "  ${BOLD}3)${NC} ${GREEN}Performance              - 性能优先超分，默认推荐 (Slot 1003)${NC}"
    echo -e "  ${BOLD}4)${NC} Sharp Balanced           - 锐化平衡超分 (Slot 2001)"
    echo -e "  ${BOLD}5)${NC} Sharp Performance        - 锐化性能超分 (Slot 2002)"
    echo -e "  ${BOLD}6)${NC} SD Compact               - SD 素材超分 (Slot 2003)"
    echo -e "  ${BOLD}7)${NC} RIFE v4.25 2x           - 仅插帧 (Slot 2025)"
    echo -e "  ${BOLD}8)${NC} RIFE v4.26 2x           - 仅插帧 (Slot 2026)"
    echo -e "  ${BOLD}9)${NC} Performance + RIFE 4.25 - 超分 + 2x 插帧 (Slot 3025)"
    echo -e "  ${BOLD}10)${NC} Performance + RIFE 4.26 - 超分 + 2x 插帧 (Slot 3026)"
    echo -e "  ${BOLD}11)${NC} RealESRGAN AnimeVideo-v3 - 原生 4x 动漫视频超分 (Slot 2004)"
    echo -e "  ${BOLD}12)${NC} APISR 2x RRDB GAN       - 2x 动漫超分 (Slot 2005)"

    local profile_choice
    read -rp "请选择处理方案 [1-12, 默认 3]: " profile_choice
    profile_choice=${profile_choice:-3}

    USE_RUNTIME_CONFIG=1
    UPSCALE_ENABLED=1
    RIFE_ENABLED=0
    RIFE_MODEL=""
    RIFE_FACTOR=2
    RIFE_ORDER="before"
    RIFE_SCD_THRESHOLD="0.150"

    case "$profile_choice" in
        1) RUNTIME_SLOT=1001; RUNTIME_PROFILE="Quality" ;;
        2) RUNTIME_SLOT=1002; RUNTIME_PROFILE="Balanced" ;;
        3) RUNTIME_SLOT=1003; RUNTIME_PROFILE="Performance" ;;
        4) RUNTIME_SLOT=2001; RUNTIME_PROFILE="Sharp Balanced" ;;
        5) RUNTIME_SLOT=2002; RUNTIME_PROFILE="Sharp Performance" ;;
        6) RUNTIME_SLOT=2003; RUNTIME_PROFILE="SD Compact" ;;
        7)
            RUNTIME_SLOT=2025
            RUNTIME_PROFILE="RIFE v4.25 2x only"
            UPSCALE_ENABLED=0
            RIFE_ENABLED=1
            RIFE_MODEL="rife_v4.25"
            ;;
        8)
            RUNTIME_SLOT=2026
            RUNTIME_PROFILE="RIFE v4.26 2x only"
            UPSCALE_ENABLED=0
            RIFE_ENABLED=1
            RIFE_MODEL="rife_v4.26"
            ;;
        9)
            RUNTIME_SLOT=3025
            RUNTIME_PROFILE="Performance + RIFE v4.25 2x"
            RIFE_ENABLED=1
            RIFE_MODEL="rife_v4.25"
            ;;
        10)
            RUNTIME_SLOT=3026
            RUNTIME_PROFILE="Performance + RIFE v4.26 2x"
            RIFE_ENABLED=1
            RIFE_MODEL="rife_v4.26"
            ;;
        11)
            RUNTIME_SLOT=2004
            RUNTIME_PROFILE="RealESRGAN AnimeVideo-v3 (4x)"
            EXTRA_FLAGS+=("--pipeline-depth" "2")
            echo -e "${CYAN}原生 4x 输出：1080p → 8K，可在后续输出设置中缩小至 4K。${NC}"
            ;;
        12)
            select_apisr_profile || return 1
            ;;
        *)
            echo -e "${YELLOW}无效选项，已使用默认 Performance (Slot 1003)。${NC}"
            RUNTIME_SLOT=1003
            RUNTIME_PROFILE="Performance"
            ;;
    esac

    echo -e "${GREEN}✔ 已选择 Runtime Slot: ${BOLD}${RUNTIME_SLOT} (${RUNTIME_PROFILE})${NC}"
    echo -e "${CYAN}首次处理新分辨率时会自动构建 Engine，之后直接复用 onnx/ 中的缓存。${NC}\n"
}

select_apisr_profile() {
    local model_name="2x_APISR_RRDB_GAN_fp16.onnx"
    if [ "$RUNTIME_MODE" -eq 0 ]; then
        RUNTIME_CONFIG="${PROJECT_ROOT}/apisr-animejanai.conf"
        CONFIG_MODEL_DIR="$MODELS_DIR"
        if [ ! -s "${CONFIG_MODEL_DIR}/${model_name}" ]; then
            CONFIG_MODEL_DIR="${PROJECT_ROOT}/models"
        fi
        CONFIG_TRTEXEC=$(command -v trtexec) || {
            echo -e "${RED}未找到 trtexec，请先运行 deploy.sh 安装 TensorRT。${NC}"
            return 1
        }
    fi
    if [ ! -f "$RUNTIME_CONFIG" ] || ! grep -Eq '^\[slot_2005\][[:space:]]*$' "$RUNTIME_CONFIG"; then
        echo -e "${RED}配置缺少 APISR Slot 2005: ${RUNTIME_CONFIG}，请更新配置。${NC}"
        return 1
    fi
    if [ ! -s "${CONFIG_MODEL_DIR}/${model_name}" ]; then
        echo -e "${RED}未找到 APISR 模型: ${CONFIG_MODEL_DIR}/${model_name}${NC}"
        echo -e "${YELLOW}源码环境请运行 ./deploy.sh --models 选择 9；Runtime 请使用包含 APISR 的新版包。${NC}"
        return 1
    fi
    local i
    for i in "${!INPUT_VIDEOS[@]}"; do
        if (( SOURCE_WIDTHS[i] % 2 != 0 || SOURCE_HEIGHTS[i] % 2 != 0 )); then
            printf 'APISR 要求源视频宽高均为偶数: %s\n' "${INPUT_VIDEOS[$i]}"
            return 1
        fi
    done
    USE_RUNTIME_CONFIG=1
    RUNTIME_SLOT=2005
    RUNTIME_PROFILE="APISR 2x RRDB GAN"
    UPSCALE_ENABLED=1
    RIFE_ENABLED=0
    echo -e "${GREEN}✔ APISR 2x RRDB GAN：首次编码按视频尺寸自动构建引擎。${NC}"
}

select_processing_profile() {
    if [ "$RUNTIME_MODE" -eq 1 ]; then
        select_runtime_profile
    else
        local processing_choice
        echo -e "  ${BOLD}1)${NC} 选择已有 TensorRT Engine"
        echo -e "  ${BOLD}2)${NC} APISR 2x RRDB GAN（自动构建 Engine）"
        read -rp "请选择 [1-2, 默认 1]: " processing_choice
        if [ "${processing_choice:-1}" = 2 ]; then
            select_apisr_profile
        else
            select_engine
        fi
    fi
}

# 4. 可选 RIFE 插帧（与超分在同一 aji_encode 管道内执行）
select_rife() {
    echo -e "${BOLD}${CYAN}[步骤 3/8] RIFE AI 插帧设置 (单次解码/编码管道)${NC}"

    local search_dirs=("${PROJECT_ROOT}/onnx/rife" "${MODELS_DIR}/rife" "${HOME}/onnx/rife")
    local raw_models=()
    for d in "${search_dirs[@]}"; do
        [ -d "$d" ] || continue
        while IFS= read -r f; do
            raw_models+=("$(readlink -f "$f")")
        done < <(find -L "$d" -maxdepth 1 -type f -name 'rife_v*.onnx' 2>/dev/null)
    done

    local found_models=()
    if [ ${#raw_models[@]} -gt 0 ]; then
        mapfile -t found_models < <(printf "%s\n" "${raw_models[@]}" | sort -uV)
    fi
    if [ ${#found_models[@]} -eq 0 ]; then
        echo -e "${YELLOW}未找到 RIFE ONNX（预期目录：${PROJECT_ROOT}/onnx/rife），本次不启用插帧。${NC}\n"
        RIFE_ENABLED=0
        return
    fi

    read -rp "是否同时启用 RIFE AI 插帧？[y/N, 默认 N]: " enable_rife
    enable_rife=${enable_rife:-N}
    if [[ ! "$enable_rife" =~ ^[Yy]$ ]]; then
        RIFE_ENABLED=0
        echo -e "${GREEN}✔ 本次仅执行超分，不插帧。${NC}\n"
        return
    fi

    RIFE_ENABLED=1
    echo -e "检测到以下 RIFE 模型："
    local default_idx=1
    for i in "${!found_models[@]}"; do
        local name="$(basename "${found_models[$i]}")"
        if [ "$name" = "rife_v4.26.onnx" ]; then
            default_idx=$((i + 1))
        fi
        echo -e "  ${BOLD}$((i+1)))${NC} ${name}"
    done
    read -rp "请选择 RIFE 模型 [1-${#found_models[@]}, 默认 ${default_idx}]: " rife_idx
    rife_idx=${rife_idx:-$default_idx}
    if [[ ! "$rife_idx" =~ ^[0-9]+$ ]] ||
       [ "$rife_idx" -lt 1 ] || [ "$rife_idx" -gt "${#found_models[@]}" ]; then
        rife_idx=$default_idx
    fi
    local rife_file="${found_models[$((rife_idx-1))]}"
    RIFE_MODEL_DIR="$(dirname "$rife_file")"
    RIFE_MODEL="$(basename "$rife_file" .onnx)"

    echo -e "\n插帧倍率："
    echo -e "  ${BOLD}1)${NC} ${GREEN}2x（推荐，如 24→48 / 30→60 fps）${NC}"
    echo -e "  ${BOLD}2)${NC} 4x（如 24→96 fps，耗时和输出体积显著增加）"
    read -rp "请选择 [1-2, 默认 1]: " factor_choice
    [ "${factor_choice:-1}" = "2" ] && RIFE_FACTOR=4 || RIFE_FACTOR=2

    echo -e "\n处理顺序："
    echo -e "  ${BOLD}1)${NC} ${GREEN}先插帧、再超分（推荐；RIFE 在源分辨率运行）${NC}"
    echo -e "  ${BOLD}2)${NC} 先超分、再插帧（细节优先；RIFE 在输出分辨率运行，通常很慢）"
    read -rp "请选择 [1-2, 默认 1]: " order_choice
    [ "${order_choice:-1}" = "2" ] && RIFE_ORDER="after" || RIFE_ORDER="before"

    read -rp "转场检测阈值 [默认 0.150]: " scd_input
    RIFE_SCD_THRESHOLD=${scd_input:-0.150}
    echo -e "${GREEN}✔ RIFE: ${BOLD}${RIFE_MODEL}${NC} | ${RIFE_FACTOR}x | ${RIFE_ORDER} upscale | SCD ${RIFE_SCD_THRESHOLD}${NC}\n"
}

select_rife_settings() {
    if [ "$USE_RUNTIME_CONFIG" -eq 0 ]; then
        select_rife
        return
    fi

    echo -e "${BOLD}${CYAN}[步骤 3/8] RIFE AI 插帧设置${NC}"
    if [ "$RIFE_ENABLED" -eq 1 ]; then
        echo -e "${GREEN}✔ 已由 Slot ${RUNTIME_SLOT} 启用 ${BOLD}${RIFE_MODEL} 2x${NC}${GREEN} 插帧。${NC}"
    else
        echo -e "${GREEN}✔ Slot ${RUNTIME_SLOT} 仅执行超分，不插帧。${NC}"
    fi
    echo -e "${CYAN}Runtime 的 RIFE 模型、倍率与处理顺序由所选 Slot 统一配置。${NC}\n"
}

# 5. 选择压制范围（全片或截取片段测试）
select_clip_mode() {
    echo -e "${BOLD}${CYAN}[步骤 4/8] 压制范围选择 (Full Video or Clip Test)${NC}"
    echo -e "  ${BOLD}1)${NC} ${GREEN}整片完整处理${NC} (Full Encode - 标准输出整部影片)"
    echo -e "  ${BOLD}2)${NC} ${YELLOW}截取指定时间段进行测试${NC} (Clip Slice Test - 快速验证画质与压制速度)"
    read -rp "请选择 [1-2, 默认 1]: " clip_choice
    clip_choice=${clip_choice:-1}

    IS_CLIP=0
    CLIP_START="00:01:00"
    CLIP_DURATION="60"

    if [ "$clip_choice" -eq 2 ]; then
        IS_CLIP=1
        read -rp "请输入测试片段开始时间 (格式如 00:00:30 或 00:01:00, 默认 00:01:00): " input_start
        CLIP_START=${input_start:-00:01:00}
        read -rp "请输入测试片段持续时长 (单位: 秒, 默认 60): " input_dur
        CLIP_DURATION=${input_dur:-60}
        echo -e "${GREEN}✔ 已设置为截取测试模式: 从 ${CLIP_START} 开始截取 ${CLIP_DURATION} 秒${NC}\n"
    else
        echo -e "${GREEN}✔ 已设置为整片完整处理模式${NC}\n"
    fi
}

# 6. 输出文件路径设置
select_output_path() {
    echo -e "${BOLD}${CYAN}[步骤 5/8] 设置输出文件路径 (Output Destination)${NC}"

    OUTPUT_VIDEOS=()
    local existing canonical_output
    for INPUT_VIDEO in "${INPUT_VIDEOS[@]}"; do
        printf '\n输入视频: %s\n' "$INPUT_VIDEO"
        local dir_name="$(dirname "$INPUT_VIDEO")"
        local base_name="$(basename "$INPUT_VIDEO")"
        local raw_name="${base_name%.*}"

        local default_out=""
        if [ "$IS_CLIP" -eq 1 ]; then
            if [ "$RIFE_ENABLED" -eq 1 ] && [ "$UPSCALE_ENABLED" -eq 1 ]; then
                default_out="${dir_name}/${raw_name}_clip_${CLIP_DURATION}s_rife${RIFE_FACTOR}x_upscaled.mkv"
            elif [ "$RIFE_ENABLED" -eq 1 ]; then
                default_out="${dir_name}/${raw_name}_clip_${CLIP_DURATION}s_rife${RIFE_FACTOR}x.mkv"
            else
                default_out="${dir_name}/${raw_name}_clip_${CLIP_DURATION}s_upscaled.mkv"
            fi
        else
            if [ "$RIFE_ENABLED" -eq 1 ] && [ "$UPSCALE_ENABLED" -eq 1 ]; then
                default_out="${dir_name}/${raw_name}_rife${RIFE_FACTOR}x_upscaled.mkv"
            elif [ "$RIFE_ENABLED" -eq 1 ]; then
                default_out="${dir_name}/${raw_name}_rife${RIFE_FACTOR}x.mkv"
            else
                default_out="${dir_name}/${raw_name}_upscaled.mkv"
            fi
        fi

        echo -e "默认推荐输出路径: ${BOLD}${default_out}${NC}"
        read -rp "是否使用此输出路径？回车默认确认，或直接输入新路径: " user_out || return 1
        if [ -n "$user_out" ]; then
            OUTPUT_VIDEO=$(printf '%s\n' "$user_out" | sed -e "s/^['\"]//" -e "s/['\"]$//")
        else
            OUTPUT_VIDEO="$default_out"
        fi

        canonical_output=$(readlink -m -- "$OUTPUT_VIDEO") || return 1
        for existing in "${INPUT_VIDEOS[@]}" "${OUTPUT_VIDEOS[@]}"; do
            if [ "$canonical_output" = "$existing" ] || [ "$OUTPUT_VIDEO" -ef "$existing" ]; then
                echo -e "${RED}[错误] 输出路径与输入视频或其他任务的输出冲突，请重新配置。${NC}"
                return 1
            fi
        done
        OUTPUT_VIDEOS+=("$canonical_output")
    done
    INPUT_VIDEO=${INPUT_VIDEOS[0]}
    OUTPUT_VIDEO=${OUTPUT_VIDEOS[0]}

    read -rp "若输出文件已存在，是否默认自动覆盖？[Y/n, 默认 Y]: " overwrite_choice
    overwrite_choice=${overwrite_choice:-Y}
    if [[ "$overwrite_choice" =~ ^[Yy]$ ]]; then
        OVERWRITE_FLAG="--overwrite"
    else
        OVERWRITE_FLAG=""
    fi

    printf "✔ 输出路径: %s\n" "${OUTPUT_VIDEOS[@]}"
}

# 7. 选择编码器与画质配置
select_encoder_and_quality() {
    echo -e "${BOLD}${CYAN}[步骤 6/8] 选择视频编码器与画质参数 (Video Encoder & Quality)${NC}"
    echo -e "  ${BOLD}1)${NC} ${GREEN}hevc_nvenc (NVIDIA NVENC H.265 硬件编码 - 强烈推荐，极速高画质)${NC}"
    echo -e "  ${BOLD}2)${NC} ${GREEN}av1_nvenc  (NVIDIA NVENC AV1 硬件编码 - Ada/RTX 40 系及更新 GPU)${NC}"
    echo -e "  ${BOLD}3)${NC} h264_nvenc (NVIDIA NVENC H.264 硬件编码 - 兼容性好)"
    echo -e "  ${BOLD}4)${NC} libx265   (CPU H.265 软件编码 - 极致压缩率，但速度慢)"
    echo -e "  ${BOLD}5)${NC} libx264   (CPU H.264 软件编码)"
    echo -e "  ${BOLD}6)${NC} ffv1      (无损归档编码)"
    read -rp "请选择编码器 [1-6, 默认 1]: " enc_choice
    enc_choice=${enc_choice:-1}

    case "$enc_choice" in
        1) VCODEC="hevc_nvenc" ;;
        2) VCODEC="av1_nvenc" ;;
        3) VCODEC="h264_nvenc" ;;
        4) VCODEC="libx265" ;;
        5) VCODEC="libx264" ;;
        6) VCODEC="ffv1" ;;
        *) VCODEC="hevc_nvenc" ;;
    esac

    echo -e "\n请选择画质与预设参数 (VQuality Preset)："
    if [[ "$VCODEC" == *"nvenc"* ]]; then
        local nvenc_extra="-tune hq"
        if [[ "$VCODEC" == "hevc_nvenc" || "$VCODEC" == "av1_nvenc" ]]; then
            # HEVC/AV1 can split one frame across multiple NVENC engines.
            # H.264 does not support split-frame encoding.
            if [ "$NVENC_SPLIT_COUNT" -gt 1 ]; then
                nvenc_extra+=" -split_encode_mode ${NVENC_SPLIT_COUNT}"
                echo -e "  ${CYAN}自动 NVENC 分割: ${NVENC_SPLIT_COUNT} 路 (${GPU_NAME}, SM ${GPU_COMPUTE_CAP:-未知})${NC}"
            else
                echo -e "  ${CYAN}自动 NVENC 分割: 单路 (${GPU_NAME}, SM ${GPU_COMPUTE_CAP:-未知})${NC}"
            fi
        fi

        echo -e "  ${BOLD}1)${NC} ${GREEN}高质量动漫推荐: -cq 18 -preset p7 ${nvenc_extra}${NC} (兼顾绝佳画质与合理体积)"
        echo -e "  ${BOLD}2)${NC} 标准平衡预设:   -cq 20 -preset p6 ${nvenc_extra} (默认标准)"
        echo -e "  ${BOLD}3)${NC} 高速压制预设:   -cq 23 -preset p4 ${nvenc_extra} (速度优先)"
        echo -e "  ${BOLD}4)${NC} 自定义输入参数"
        read -rp "请选择画质档位 [1-4, 默认 1]: " q_choice
        q_choice=${q_choice:-1}

        case "$q_choice" in
            1) VQUALITY="-cq 18 -preset p7 ${nvenc_extra}" ;;
            2) VQUALITY="-cq 20 -preset p6 ${nvenc_extra}" ;;
            3) VQUALITY="-cq 23 -preset p4 ${nvenc_extra}" ;;
            4)
                read -rp "请输入自定义 FFmpeg 编码参数 (如 -cq 16 -preset p7): " custom_q
                VQUALITY="${custom_q:--cq 18 -preset p7 ${nvenc_extra}}"
                ;;
            *) VQUALITY="-cq 18 -preset p7 ${nvenc_extra}" ;;
        esac
    else
        echo -e "  ${BOLD}1)${NC} CRF 18 (高质量) -preset slow"
        echo -e "  ${BOLD}2)${NC} CRF 20 (标准)   -preset medium"
        echo -e "  ${BOLD}3)${NC} 自定义输入参数"
        read -rp "请选择画质档位 [1-3, 默认 1]: " q_choice
        q_choice=${q_choice:-1}

        case "$q_choice" in
            1) VQUALITY="-crf 18 -preset slow" ;;
            2) VQUALITY="-crf 20 -preset medium" ;;
            3)
                read -rp "请输入自定义 FFmpeg 编码参数: " custom_q
                VQUALITY="${custom_q:--crf 18 -preset slow}"
                ;;
            *) VQUALITY="-crf 18 -preset slow" ;;
        esac
    fi

    echo -e "${GREEN}✔ 编码器: ${BOLD}${VCODEC}${NC} | 参数: ${BOLD}${VQUALITY}${NC}\n"
}

# 8. 选择解码器与像素格式
select_decoder_and_pixfmt() {
    echo -e "${BOLD}${CYAN}[步骤 7/8] 硬件解码器与色彩像素格式 (Decoder & Pixel Format)${NC}"

    echo -e "解码器选择 (--decoder)："
    echo -e "  ${BOLD}1)${NC} ${GREEN}nvdec (NVIDIA 硬件加速解码 - 推荐，显存零拷贝极速)${NC}"
    echo -e "  ${BOLD}2)${NC} auto  (自动根据格式探测)"
    echo -e "  ${BOLD}3)${NC} cpu   (CPU 软件解码，兼容奇特封装格式)"
    read -rp "请选择解码器 [1-3, 默认 1]: " dec_choice
    dec_choice=${dec_choice:-1}
    case "$dec_choice" in
        1) DECODER="nvdec" ;;
        2) DECODER="auto" ;;
        3) DECODER="cpu" ;;
        *) DECODER="nvdec" ;;
    esac

    echo -e "\n色彩像素位深格式 (--pix-fmt)："
    echo -e "  ${BOLD}1)${NC} ${GREEN}yuv420p10 (10-bit 色深 - 强烈推荐，彻底杜绝动漫暗部色带断层)${NC}"
    echo -e "  ${BOLD}2)${NC} yuv420p   (8-bit 色深 - 通用兼容)"
    echo -e "  ${BOLD}3)${NC} yuv444p10 (4:4:4 10-bit 无抽样，需软编支持)"
    read -rp "请选择色彩格式 [1-3, 默认 1]: " pix_choice
    pix_choice=${pix_choice:-1}
    case "$pix_choice" in
        1) PIX_FMT="yuv420p10" ;;
        2) PIX_FMT="yuv420p" ;;
        3) PIX_FMT="yuv444p10" ;;
        *) PIX_FMT="yuv420p10" ;;
    esac

    echo -e "${GREEN}✔ 解码器: ${BOLD}${DECODER}${NC} | 像素格式: ${BOLD}${PIX_FMT}${NC}\n"
}

# 9. 音轨/字幕与降采样选项
select_optional_flags() {
    echo -e "${BOLD}${CYAN}[步骤 8/8] 音频、字幕与输出缩放附加选项${NC}"

    read -rp "是否保留原视频中的全部音频流？[Y/n, 默认 Y]: " keep_audio
    keep_audio=${keep_audio:-Y}
    if [[ ! "$keep_audio" =~ ^[Yy]$ ]]; then
        EXTRA_FLAGS+=("--no-audio")
    fi

    read -rp "是否保留原视频中的全部字幕流？[Y/n, 默认 Y]: " keep_subs
    keep_subs=${keep_subs:-Y}
    if [[ ! "$keep_subs" =~ ^[Yy]$ ]]; then
        EXTRA_FLAGS+=("--no-subs")
    fi

    read -rp "是否在 AI 处理后缩放到特定高度？(如输入 1080 实现 2x 超分再降采样抗锯齿，留空表示不缩放): " resize_h
    if [ -n "$resize_h" ] && [ "$resize_h" -gt 0 ] 2>/dev/null; then
        EXTRA_FLAGS+=("--final-resize-height" "$resize_h")
        echo -e "${GREEN}✔ 已开启后处理下采样至高度: ${resize_h}p${NC}"
    fi

    echo ""
}

# 输出可直接由 Bash 执行的命令，保留空格、引号及 $ 等文件名字符。
write_shell_command() {
    printf '%q ' "$@"
    printf '\n'
}

# 9. 生成最终命令与独立脚本；set -e 保证任务失败时停止队列。
generate_final_command_and_script() {
    local gen_script_path="${PROJECT_ROOT}/run_encode.sh"
    local i effective_input
    local cmd_args=()

    {
        printf '#!/usr/bin/env bash\nset -euo pipefail\n\n'
        printf 'export PATH=%q:"$PATH"\n' "${CUDA_BIN_DIR}:${FFMPEG_INSTALL_DIR}/bin"
        printf 'export LD_LIBRARY_PATH=%q:"${LD_LIBRARY_PATH:-}"\n' "${PROJECT_ROOT}/build:${FFMPEG_INSTALL_DIR}/lib:${CUDA_LIB_DIR}"
        if [ -f "$NVENC_FIX_SO" ]; then
            printf 'export LD_PRELOAD=%q"${LD_PRELOAD:+ $LD_PRELOAD}"\n' "$NVENC_FIX_SO"
        fi
        if [ "$IS_CLIP" -eq 1 ]; then
            # 每次执行使用独立临时目录；失败退出时也清理片段。
            cat << 'EOF'
clip_dir=$(mktemp -d)
trap 'rm -rf -- "$clip_dir"' EXIT
EOF
        fi
        for i in "${!INPUT_VIDEOS[@]}"; do
            printf '\n'
            write_shell_command printf '%s\n' "[$((i+1))/${#INPUT_VIDEOS[@]}] 输入: ${INPUT_VIDEOS[$i]}" "输出: ${OUTPUT_VIDEOS[$i]}"
            effective_input=${INPUT_VIDEOS[$i]}
            if [ "$IS_CLIP" -eq 1 ]; then
                write_shell_command printf '%s\n' "正在截取测试片段（${CLIP_START}，${CLIP_DURATION} 秒）..."
                printf '%q ' ffmpeg -y -ss "$CLIP_START" -i "$effective_input" -t "$CLIP_DURATION" -c copy
                printf '"$clip_dir/clip.mkv"\n'
            fi
            cmd_args=("--output" "${OUTPUT_VIDEOS[$i]}")
            if [ "$USE_RUNTIME_CONFIG" -eq 1 ]; then
                cmd_args+=("--conf" "${RUNTIME_CONFIG}")
                cmd_args+=("--slot" "${RUNTIME_SLOT}")
                cmd_args+=("--model-dir" "${CONFIG_MODEL_DIR}")
                cmd_args+=("--rife-model-dir" "${CONFIG_MODEL_DIR}/rife")
                cmd_args+=("--trtexec" "${CONFIG_TRTEXEC}")
            else
                cmd_args+=("--engine" "${ENGINE_FILE}")
                cmd_args+=("--max-width" "${SOURCE_WIDTHS[$i]}")
                cmd_args+=("--max-height" "${SOURCE_HEIGHTS[$i]}")
            fi
            cmd_args+=("--decoder" "${DECODER}")
            cmd_args+=("--vcodec" "${VCODEC}")
            cmd_args+=("--vquality" "${VQUALITY}")
            cmd_args+=("--pix-fmt" "${PIX_FMT}")
            if [ "$RIFE_ENABLED" -eq 1 ] && [ "$USE_RUNTIME_CONFIG" -eq 0 ]; then
                cmd_args+=("--rife-model-dir" "${RIFE_MODEL_DIR}")
                cmd_args+=("--rife-model" "${RIFE_MODEL}")
                cmd_args+=("--rife-factor" "${RIFE_FACTOR}")
                cmd_args+=("--rife-order" "${RIFE_ORDER}")
                cmd_args+=("--rife-scene-threshold" "${RIFE_SCD_THRESHOLD}")
            fi
            [ -n "$OVERWRITE_FLAG" ] && cmd_args+=("${OVERWRITE_FLAG}")
            [ ${#EXTRA_FLAGS[@]} -gt 0 ] && cmd_args+=("${EXTRA_FLAGS[@]}")

            printf '%q ' "$AJI_ENCODE_BIN" --input
            if [ "$IS_CLIP" -eq 1 ]; then
                printf '"$clip_dir/clip.mkv" '
            else
                printf '%q ' "$effective_input"
            fi
            write_shell_command "${cmd_args[@]}"
        done
        printf '\n'
        write_shell_command printf '%s\n' "✔ 全部 ${#INPUT_VIDEOS[@]} 个视频处理完成！"
    } > "$gen_script_path" || return 1
    chmod +x "$gen_script_path" || return 1

    print_header
    echo -e "${BOLD}${GREEN}✔ 已生成 ${#INPUT_VIDEOS[@]} 个视频的串行任务脚本，未自动执行。${NC}"
    echo "全部视频共用处理参数；任一任务失败时停止后续任务。"
    echo -e "\n${BOLD}${YELLOW}方式一：运行独立脚本${NC}"
    write_shell_command bash "$gen_script_path"
    printf 'nohup bash %q > encode.log 2>&1 &\n' "$gen_script_path"
    echo "查看进度: tail -f encode.log"
    echo -e "\n${BOLD}${YELLOW}方式二：复制以下完整命令执行（包含串行队列）${NC}"
    # 子 Shell 限定 set/trap 的作用域，手动粘贴执行不会退出用户的终端。
    printf '(\n'
    cat "$gen_script_path"
    printf ')\n'
}

# 主执行流
main() {
    check_binaries
    detect_nvenc_split_count
    print_header
    select_input_video || return 1
    select_processing_profile || return 1
    select_rife_settings
    select_clip_mode
    select_output_path || return 1
    select_encoder_and_quality
    select_decoder_and_pixfmt
    select_optional_flags
    generate_final_command_and_script
}

main
