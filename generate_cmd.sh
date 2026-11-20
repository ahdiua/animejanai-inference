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
OUTPUT_VIDEO=""
# The packaged runtime and a source install share one conf and model layout:
# <model dir>/<model name>.onnx and <model dir>/rife/rife_v<ver>.onnx.
if [ "$RUNTIME_MODE" -eq 1 ]; then
    BASE_CONF="${PROJECT_ROOT}/animejanai.conf"
    CONFIG_MODEL_DIR="${PROJECT_ROOT}/onnx"
    CONFIG_TRTEXEC="${PROJECT_ROOT}/bin/trtexec.real"
else
    BASE_CONF="${PROJECT_ROOT}/packaging/ubuntu24/runtime/animejanai.conf"
    CONFIG_MODEL_DIR="${MODELS_DIR}"
    CONFIG_TRTEXEC="$(command -v trtexec 2>/dev/null || true)"
fi
RIFE_MODEL_DIR="${CONFIG_MODEL_DIR}/rife"
GEN_SCRIPT_PATH="${PROJECT_ROOT}/run_encode.sh"
TASK_CONF_PATH="${PROJECT_ROOT}/run_encode.conf"
RUNTIME_SLOT=1003
RUNTIME_PROFILE="Performance"
SLOT_CHAIN_LINES=""
CUSTOM_MODEL=""
IS_4X_MODEL=0
REQUIRES_EVEN_SIZE=0
NATIVE_OFF=0
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
RIFE_FACTOR=2
RIFE_FACTOR_NUM=2
RIFE_FACTOR_DEN=1
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
        echo -e "${YELLOW}请先运行 ./deploy.sh 编译项目后再使用此生成器。${NC}"
        exit 1
    fi

    if [ "$RUNTIME_MODE" -eq 1 ]; then
        if [ ! -f "$BASE_CONF" ] || [ ! -d "$CONFIG_MODEL_DIR" ] || \
           [ ! -x "$CONFIG_TRTEXEC" ]; then
            echo -e "${RED}[错误] Runtime 包不完整，缺少 animejanai.conf、onnx/ 或 bin/trtexec.real。${NC}"
            echo -e "${YELLOW}请重新解压完整的 Runtime 包后再运行生成器。${NC}"
            exit 1
        fi
    else
        if [ ! -f "$BASE_CONF" ]; then
            echo -e "${RED}[错误] 未找到配置文件: ${BASE_CONF}${NC}"
            exit 1
        fi
        if [ -z "$CONFIG_TRTEXEC" ]; then
            echo -e "${RED}[错误] 未找到 trtexec；native kernel 不支持的模型需要它构建 Engine。${NC}"
            echo -e "${YELLOW}请先运行 ./deploy.sh 安装 TensorRT。${NC}"
            exit 1
        fi
        if [ ! -d "$CONFIG_MODEL_DIR" ]; then
            echo -e "${RED}[错误] 模型目录不存在: ${CONFIG_MODEL_DIR}${NC}"
            echo -e "${YELLOW}请先运行 ./deploy.sh --models 下载模型。${NC}"
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
# ---- conf slots -------------------------------------------------------------
# Every profile runs through an animejanai.conf slot. Models the native CUDA
# kernels support need no engine; the rest fall back to TensorRT, which builds
# and caches an engine in the model directory on first use.

# Built-in slots 1001-1003 live in libaji (src/aji_conf.cpp add_builtin_slots),
# not in the conf file. Mirror them here as conf lines so a task conf can add
# RIFE to them and so their models can be checked before a run.
HD_BAL_MODEL="2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16"
HD_PERF_MODEL="2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16"
SD_MODEL="2x_AnimeJaNai_SD_V1beta34_Compact_1x3xHxW_dyn-HW_strong_fp16_op21_dynamo"
CUSTOM_SLOT=9001

builtin_chain_lines() {
    local hd="$HD_BAL_MODEL"
    [ "$1" = 1003 ] && hd="$HD_PERF_MODEL"
    printf '%s\n' \
        "chain_1_min_resolution=1280x720" "chain_1_max_resolution=1920x1080" \
        "chain_1_max_fps=31" "chain_1_model_1_name=${hd}" \
        "chain_2_min_resolution=0x0" "chain_2_max_resolution=1280x720" \
        "chain_2_max_fps=31" "chain_2_model_1_name=${SD_MODEL}"
    if [ "$1" = 1001 ]; then
        printf '%s\n' \
            "chain_3_min_resolution=0x0" "chain_3_max_resolution=1920x1080" \
            "chain_3_max_fps=61" "chain_3_model_1_name=${HD_BAL_MODEL}"
    fi
}

# key=value lines of one INI section, without comments or blank lines.
conf_section_lines() {
    local file="$1" section="$2"
    awk -v want="[${section}]" '
        { line = $0; sub(/\r$/, "", line); sub(/^[ \t]+/, "", line); sub(/[ \t]+$/, "", line) }
        line ~ /^\[/ { inside = (tolower(line) == tolower(want)); next }
        inside && line != "" && line !~ /^[#;]/ { print line }
    ' "$file"
}

# The chain lines of a slot: the conf file's section, else a built-in.
slot_chain_lines() {
    local slot="$1"
    if grep -Eqi "^[[:space:]]*\[slot_${slot}\][[:space:]]*$" "$BASE_CONF" 2>/dev/null; then
        conf_section_lines "$BASE_CONF" "slot_${slot}" | grep -Ei '^chain_'
    else
        case "$slot" in
            1001|1002|1003) builtin_chain_lines "$slot" ;;
            *) return 1 ;;
        esac
    fi
}

conf_value() {
    printf '%s\n' "$1" | sed -n "s/^[^=]*=[[:space:]]*//p"
}

# libaji's rife_model_name: 425 -> rife_v4.25, 4251 -> rife_v4.25_lite.
rife_code_to_name() {
    local code="$1" ensemble="${2:-no}" name
    [[ "$code" =~ ^[0-9]{2,4}$ ]] || return 1
    if [ "${#code}" -eq 2 ]; then
        name="rife_v${code:0:1}.${code:1:1}"
    else
        name="rife_v${code:0:1}.${code:1:2}"
    fi
    [ "${#code}" -eq 4 ] && [ "${code: -1}" = 1 ] && name+="_lite"
    [[ "$ensemble" =~ ^([Yy][Ee][Ss]|[Yy]|1|true|on)$ ]] && name+="_ensemble"
    printf '%s\n' "$name"
}

# The conf code (and ensemble flag) for a rife_v*.onnx basename, checked by
# mapping it back; prints "<code> <yes|no>".
rife_name_to_code() {
    local name="$1" ensemble=no lite=0 major minor code
    if [[ "$name" == *_ensemble ]]; then ensemble=yes; name=${name%_ensemble}; fi
    if [[ "$name" == *_lite ]]; then lite=1; name=${name%_lite}; fi
    [[ "$name" =~ ^rife_v([0-9])\.([0-9]{1,2})$ ]] || return 1
    major=${BASH_REMATCH[1]}
    minor=${BASH_REMATCH[2]}
    code="${major}${minor}"
    [ "$lite" -eq 1 ] && code+="1"
    [ "$(rife_code_to_name "$code" "$ensemble")" = "$1" ] || return 1
    printf '%s %s\n' "$code" "$ensemble"
}

# ONNX files (relative to their directory) that a set of chain lines needs.
required_models() {
    local line key value chain ensemble
    local -A rife_on=() rife_code=() rife_ens=()
    while IFS= read -r line; do
        key=$(printf '%s' "${line%%=*}" | tr '[:upper:]' '[:lower:]')
        value=$(conf_value "$line")
        case "$key" in
            chain_*_model_*_name|chain_*_temporal_fix_model)
                [ -n "$value" ] && printf 'model %s.onnx\n' "$value" ;;
            chain_*_rife|chain_*_rife_model|chain_*_rife_ensemble)
                chain=${key#chain_}; chain=${chain%%_*}
                case "$key" in
                    *_rife) [[ "$value" =~ ^([Yy][Ee][Ss]|[Yy]|1|true|on)$ ]] && rife_on[$chain]=1 ;;
                    *_rife_model) rife_code[$chain]=$value ;;
                    *_rife_ensemble) rife_ens[$chain]=$value ;;
                esac ;;
        esac
    done
    for chain in "${!rife_on[@]}"; do
        ensemble=${rife_ens[$chain]:-no}
        value=$(rife_code_to_name "${rife_code[$chain]:-414}" "$ensemble") || continue
        printf 'rife %s.onnx\n' "$value"
    done
}

# 1 when the chain lines enable RIFE with a factor above 1.
chain_lines_use_rife() {
    local lines="$1"
    printf '%s\n' "$lines" | grep -Eqi '^chain_[0-9]+_rife=[[:space:]]*(yes|y|1|true|on)[[:space:]]*$'
}

missing_models_for() {
    local kind file missing=0
    while read -r kind file; do
        [ -n "$file" ] || continue
        if [ "$kind" = rife ]; then
            [ -s "${RIFE_MODEL_DIR}/${file}" ] && continue
            printf '  - %s/%s\n' "$RIFE_MODEL_DIR" "$file"
        else
            [ -s "${CONFIG_MODEL_DIR}/${file}" ] && continue
            printf '  - %s/%s\n' "$CONFIG_MODEL_DIR" "$file"
        fi
        missing=1
    done < <(printf '%s\n' "$1" | required_models | sort -u)
    return "$missing"
}

is_4x_model_name() {
    local name
    name=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
    [[ "$name" == *4x* || "$name" == *anime6b* || "$name" == *animevideov3* ]]
}

# Engines in the model directories that are not backend cache files. libaji
# removes every *.engine there without the current ".trt-<ver>.gpu-<gpu>"
# suffix on its first run, hand-built engines included.
warn_unmanaged_engines() {
    local dir file found=()
    for dir in "$CONFIG_MODEL_DIR" "$RIFE_MODEL_DIR"; do
        [ -d "$dir" ] || continue
        while IFS= read -r file; do
            [[ "$(basename "$file")" == *.trt-*.gpu-*.engine ]] || found+=("$file")
        done < <(find "$dir" -maxdepth 1 -type f -name '*.engine' 2>/dev/null | sort)
    done
    [ ${#found[@]} -gt 0 ] || return 0
    echo -e "${YELLOW}⚠ 模型目录中有不是由 backend 缓存生成的 .engine 文件：${NC}"
    printf '    %s\n' "${found[@]}"
    echo -e "${YELLOW}  首次运行时 libaji 会删除这些文件（engine 现在按需自动构建）。"
    echo -e "  如需先处理，请运行 ./deploy.sh --migrate。${NC}\n"
}

reset_profile_state() {
    RUNTIME_SLOT=1003
    RUNTIME_PROFILE="Performance"
    SLOT_CHAIN_LINES=""
    UPSCALE_ENABLED=1
    RIFE_ENABLED=0
    RIFE_MODEL=""
    RIFE_FACTOR_NUM=2
    RIFE_FACTOR_DEN=1
    RIFE_FACTOR=2
    RIFE_ORDER="before"
    RIFE_SCD_THRESHOLD="0.150"
    CUSTOM_MODEL=""
    IS_4X_MODEL=0
    REQUIRES_EVEN_SIZE=0
}

# Models in the model directory that no packaged or built-in slot uses.
list_custom_models() {
    local known file name
    known=$( {
        builtin_chain_lines 1001
        builtin_chain_lines 1003
        grep -Ei '^[[:space:]]*chain_[0-9]+_(model_[0-9]+_name|temporal_fix_model)=' "$BASE_CONF" 2>/dev/null
    } | sed 's/^[^=]*=[[:space:]]*//' | sort -u)
    while IFS= read -r file; do
        name=$(basename "$file" .onnx)
        printf '%s\n' "$known" | grep -Fxq -- "$name" || printf '%s\n' "$name"
    done < <(find "$CONFIG_MODEL_DIR" -maxdepth 1 -type f -name '*.onnx' 2>/dev/null | sort)
}

select_custom_model() {
    local models=() choice
    mapfile -t models < <(list_custom_models)
    if [ ${#models[@]} -eq 0 ]; then
        echo -e "${YELLOW}${CONFIG_MODEL_DIR} 中没有方案以外的 ONNX 模型。"
        echo -e "可运行 ./deploy.sh --models 选择“自定义 ONNX”导入。${NC}"
        return 1
    fi
    echo -e "\n${CONFIG_MODEL_DIR} 中的自定义模型："
    for i in "${!models[@]}"; do
        printf '  %d) %s\n' "$((i+1))" "${models[$i]}"
    done
    read -rp "请选择模型 [1-${#models[@]}, 默认 1]: " choice || return 1
    choice=${choice:-1}
    if [[ ! "$choice" =~ ^[0-9]{1,4}$ ]] || (( 10#$choice < 1 || 10#$choice > ${#models[@]} )); then
        echo -e "${RED}[错误] 无效编号。${NC}"
        return 1
    fi
    CUSTOM_MODEL=${models[$((10#$choice-1))]}
    RUNTIME_SLOT=$CUSTOM_SLOT
    RUNTIME_PROFILE="Custom: ${CUSTOM_MODEL}"
    SLOT_CHAIN_LINES=$(printf '%s\n' "chain_1_max_fps=61" "chain_1_model_1_name=${CUSTOM_MODEL}")
    is_4x_model_name "$CUSTOM_MODEL" && IS_4X_MODEL=1
    return 0
}

select_processing_profile() {
    echo -e "${BOLD}${CYAN}[步骤 2/8] 选择处理方案 (Slot)${NC}"
    echo -e "${GREEN}Engine 无需手动准备：native kernel 直接运行支持的模型，其余模型首次运行时自动构建 TensorRT Engine 并缓存。${NC}"
    echo -e "  ${BOLD}1)${NC} Quality                       - 质量优先超分 (Slot 1001)"
    echo -e "  ${BOLD}2)${NC} Balanced                      - 平衡超分 (Slot 1002)"
    echo -e "  ${BOLD}3)${NC} ${GREEN}Performance                   - 性能优先超分，默认推荐 (Slot 1003)${NC}"
    echo -e "  ${BOLD}4)${NC} Sharp Balanced                - 锐化平衡超分 (Slot 2001)"
    echo -e "  ${BOLD}5)${NC} Sharp Performance             - 锐化性能超分 (Slot 2002)"
    echo -e "  ${BOLD}6)${NC} SD Compact                    - SD 素材超分 (Slot 2003)"
    echo -e "  ${BOLD}7)${NC} RealESRGAN AnimeVideo-v3 (4x) - 原生 4x 动漫视频超分 (Slot 2004)"
    echo -e "  ${BOLD}8)${NC} APISR 2x RRDB GAN             - 2x 动漫超分，需偶数宽高 (Slot 2005)"
    echo -e "  ${BOLD}9)${NC} Real-ESRGAN x4plus Anime 6B   - 原生 4x 动漫超分 (Slot 2006)"
    echo -e "  ${BOLD}10)${NC} RIFE v4.25 2x                - 仅插帧 (Slot 2025)"
    echo -e "  ${BOLD}11)${NC} RIFE v4.26 2x                - 仅插帧 (Slot 2026)"
    echo -e "  ${BOLD}12)${NC} Performance + RIFE 4.25 2x   - 超分 + 插帧 (Slot 3025)"
    echo -e "  ${BOLD}13)${NC} Performance + RIFE 4.26 2x   - 超分 + 插帧 (Slot 3026)"
    echo -e "  ${BOLD}14)${NC} 自定义模型                   - ${CONFIG_MODEL_DIR} 中的其他 ONNX"
    echo -e "  ${BOLD}15)${NC} 仅自定义插帧                 - 不超分，自选 RIFE 模型与倍率"

    local choice
    while true; do
        reset_profile_state
        read -rp "请选择处理方案 [1-15, 默认 3]: " choice || return 1
        choice=${choice:-3}
        case "$choice" in
            1) RUNTIME_SLOT=1001; RUNTIME_PROFILE="Quality" ;;
            2) RUNTIME_SLOT=1002; RUNTIME_PROFILE="Balanced" ;;
            3) RUNTIME_SLOT=1003; RUNTIME_PROFILE="Performance" ;;
            4) RUNTIME_SLOT=2001; RUNTIME_PROFILE="Sharp Balanced" ;;
            5) RUNTIME_SLOT=2002; RUNTIME_PROFILE="Sharp Performance" ;;
            6) RUNTIME_SLOT=2003; RUNTIME_PROFILE="SD Compact" ;;
            7) RUNTIME_SLOT=2004; RUNTIME_PROFILE="RealESRGAN AnimeVideo-v3 (4x)"; IS_4X_MODEL=1 ;;
            8) RUNTIME_SLOT=2005; RUNTIME_PROFILE="APISR 2x RRDB GAN"; REQUIRES_EVEN_SIZE=1 ;;
            9) RUNTIME_SLOT=2006; RUNTIME_PROFILE="Real-ESRGAN x4plus Anime 6B (4x)"; IS_4X_MODEL=1 ;;
            10) RUNTIME_SLOT=2025; RUNTIME_PROFILE="RIFE v4.25 2x only" ;;
            11) RUNTIME_SLOT=2026; RUNTIME_PROFILE="RIFE v4.26 2x only" ;;
            12) RUNTIME_SLOT=3025; RUNTIME_PROFILE="Performance + RIFE v4.25 2x" ;;
            13) RUNTIME_SLOT=3026; RUNTIME_PROFILE="Performance + RIFE v4.26 2x" ;;
            14) select_custom_model || continue ;;
            15)
                RUNTIME_SLOT=$CUSTOM_SLOT
                RUNTIME_PROFILE="Custom RIFE only"
                UPSCALE_ENABLED=0
                SLOT_CHAIN_LINES="chain_1_max_fps=61"
                select_rife_options || continue
                ;;
            *) echo -e "${RED}[错误] 无效选项。${NC}"; continue ;;
        esac

        if [ -z "$SLOT_CHAIN_LINES" ]; then
            if ! SLOT_CHAIN_LINES=$(slot_chain_lines "$RUNTIME_SLOT") ||
               [ -z "$SLOT_CHAIN_LINES" ]; then
                echo -e "${RED}[错误] ${BASE_CONF} 中没有 Slot ${RUNTIME_SLOT}，请更新配置文件。${NC}"
                continue
            fi
        fi
        if chain_lines_use_rife "$SLOT_CHAIN_LINES"; then
            RIFE_ENABLED=1
            printf '%s\n' "$SLOT_CHAIN_LINES" | grep -Eqi '^chain_[0-9]+_model_[0-9]+_name=.' || UPSCALE_ENABLED=0
            describe_slot_rife
        fi
        if [ "$REQUIRES_EVEN_SIZE" -eq 1 ] && ! check_even_sizes; then
            continue
        fi
        break
    done

    echo -e "${GREEN}✔ 已选择方案: ${BOLD}${RUNTIME_PROFILE}${NC}${GREEN} (Slot ${RUNTIME_SLOT})${NC}"
    if [ "$IS_4X_MODEL" -eq 1 ]; then
        echo -e "${CYAN}ℹ️  4x 模型：1080p 输入会输出 8K，可在输出设置中缩小（如缩放到 2160）。"
        echo -e "   已自动加入 --pipeline-depth 2 以降低显存占用。${NC}"
        EXTRA_FLAGS+=("--pipeline-depth" "2")
    fi
    echo ""
}

check_even_sizes() {
    local i
    for i in "${!INPUT_VIDEOS[@]}"; do
        if (( SOURCE_WIDTHS[i] % 2 != 0 || SOURCE_HEIGHTS[i] % 2 != 0 )); then
            printf '%b该方案要求源视频宽高均为偶数: %s (%sx%s)%b\n' "$RED" \
                "${INPUT_VIDEOS[$i]}" "${SOURCE_WIDTHS[$i]}" "${SOURCE_HEIGHTS[$i]}" "$NC"
            return 1
        fi
    done
}

# RIFE settings of the selected slot's chains, for the summary and file name.
describe_slot_rife() {
    local line key value code="" num="" den=""
    while IFS= read -r line; do
        key=$(printf '%s' "${line%%=*}" | tr '[:upper:]' '[:lower:]')
        value=$(conf_value "$line")
        case "$key" in
            chain_1_rife_model) code=$value ;;
            chain_1_rife_factor_numerator) num=$value ;;
            chain_1_rife_factor_denominator) den=$value ;;
        esac
    done <<< "$SLOT_CHAIN_LINES"
    RIFE_MODEL=$(rife_code_to_name "${code:-414}") || RIFE_MODEL="rife (${code})"
    RIFE_FACTOR_NUM=${num:-1}
    RIFE_FACTOR_DEN=${den:-1}
    set_rife_factor_label
}

set_rife_factor_label() {
    if [ "$RIFE_FACTOR_DEN" = 1 ]; then
        RIFE_FACTOR=$RIFE_FACTOR_NUM
    else
        RIFE_FACTOR="${RIFE_FACTOR_NUM}-${RIFE_FACTOR_DEN}"
    fi
}

# Ask for a RIFE model, factor, order and scene threshold from the RIFE
# model directory; the chain lines get the matching chain_1_rife_* keys.
select_rife_options() {
    local files=() names=() codes=() ensembles=() file name parsed i
    local choice default_idx=1 factor_choice order_choice scd_input
    while IFS= read -r file; do
        name=$(basename "$file" .onnx)
        parsed=$(rife_name_to_code "$name") || continue
        names+=("$name")
        codes+=("${parsed% *}")
        ensembles+=("${parsed#* }")
    done < <(find -L "$RIFE_MODEL_DIR" -maxdepth 1 -type f -name 'rife_v*.onnx' 2>/dev/null | sort -V)
    if [ ${#names[@]} -eq 0 ]; then
        echo -e "${YELLOW}未在 ${RIFE_MODEL_DIR} 找到 rife_v*.onnx，请先运行 ./deploy.sh --models。${NC}"
        return 1
    fi
    echo -e "\n检测到以下 RIFE 模型："
    for i in "${!names[@]}"; do
        [ "${names[$i]}" = rife_v4.26 ] && default_idx=$((i + 1))
        printf '  %d) %s\n' "$((i+1))" "${names[$i]}"
    done
    read -rp "请选择 RIFE 模型 [1-${#names[@]}, 默认 ${default_idx}]: " choice || return 1
    choice=${choice:-$default_idx}
    if [[ ! "$choice" =~ ^[0-9]{1,4}$ ]] || (( 10#$choice < 1 || 10#$choice > ${#names[@]} )); then
        choice=$default_idx
    fi
    i=$((10#$choice - 1))
    RIFE_MODEL=${names[$i]}

    echo -e "\n插帧倍率："
    echo -e "  ${BOLD}1)${NC} ${GREEN}2x（推荐，如 24→48 fps）${NC}"
    echo -e "  ${BOLD}2)${NC} 5/2x（如 24→60 fps）"
    echo -e "  ${BOLD}3)${NC} 4x（如 24→96 fps，耗时和输出体积显著增加）"
    read -rp "请选择 [1-3, 默认 1]: " factor_choice || return 1
    case "${factor_choice:-1}" in
        2) RIFE_FACTOR_NUM=5; RIFE_FACTOR_DEN=2 ;;
        3) RIFE_FACTOR_NUM=4; RIFE_FACTOR_DEN=1 ;;
        *) RIFE_FACTOR_NUM=2; RIFE_FACTOR_DEN=1 ;;
    esac
    set_rife_factor_label

    RIFE_ORDER="before"
    if [ "$UPSCALE_ENABLED" -eq 1 ]; then
        echo -e "\n处理顺序："
        echo -e "  ${BOLD}1)${NC} ${GREEN}先插帧、再超分（推荐；RIFE 在源分辨率运行）${NC}"
        echo -e "  ${BOLD}2)${NC} 先超分、再插帧（RIFE 在输出分辨率运行，通常很慢）"
        read -rp "请选择 [1-2, 默认 1]: " order_choice || return 1
        [ "${order_choice:-1}" = 2 ] && RIFE_ORDER="after"
    fi

    while true; do
        read -rp "转场检测阈值 [默认 0.150]: " scd_input || return 1
        RIFE_SCD_THRESHOLD=${scd_input:-0.150}
        [[ "$RIFE_SCD_THRESHOLD" =~ ^[0-9]*\.?[0-9]+$ ]] && break
        echo -e "${RED}[错误] 请输入数字，如 0.150。${NC}"
    done

    local before=yes chain chains
    [ "$RIFE_ORDER" = after ] && before=no
    # Every chain the slot defines gets the same RIFE settings, so a built-in
    # slot keeps its per-resolution model choice.
    chains=$(printf '%s\n' "$SLOT_CHAIN_LINES" | sed -n 's/^[Cc][Hh][Aa][Ii][Nn]_\([0-9]\+\)_.*/\1/p' | sort -un)
    [ -n "$chains" ] || chains=1
    SLOT_CHAIN_LINES=$(printf '%s\n' "$SLOT_CHAIN_LINES" | grep -Eiv '^chain_[0-9]+_rife')
    for chain in $chains; do
        SLOT_CHAIN_LINES+=$'\n'"chain_${chain}_rife=yes"
        SLOT_CHAIN_LINES+=$'\n'"chain_${chain}_rife_model=${codes[$i]}"
        [ "${ensembles[$i]}" = yes ] && SLOT_CHAIN_LINES+=$'\n'"chain_${chain}_rife_ensemble=yes"
        SLOT_CHAIN_LINES+=$'\n'"chain_${chain}_rife_factor_numerator=${RIFE_FACTOR_NUM}"
        SLOT_CHAIN_LINES+=$'\n'"chain_${chain}_rife_factor_denominator=${RIFE_FACTOR_DEN}"
        SLOT_CHAIN_LINES+=$'\n'"chain_${chain}_rife_before_upscale=${before}"
        SLOT_CHAIN_LINES+=$'\n'"chain_${chain}_rife_scene_detect_threshold=${RIFE_SCD_THRESHOLD}"
    done
    SLOT_CHAIN_LINES=$(printf '%s\n' "$SLOT_CHAIN_LINES" | sed '/^$/d')
    RIFE_ENABLED=1
    echo -e "${GREEN}✔ RIFE: ${BOLD}${RIFE_MODEL}${NC}${GREEN} | ${RIFE_FACTOR_NUM}/${RIFE_FACTOR_DEN}x | ${RIFE_ORDER} upscale | SCD ${RIFE_SCD_THRESHOLD}${NC}"
}

# Step 3: optional RIFE on top of an upscale-only slot (task conf slot 9001).
select_rife_settings() {
    echo -e "${BOLD}${CYAN}[步骤 3/8] RIFE AI 插帧设置${NC}"
    if [ "$RIFE_ENABLED" -eq 1 ]; then
        echo -e "${GREEN}✔ 方案已包含插帧: ${BOLD}${RIFE_MODEL}${NC}${GREEN} ${RIFE_FACTOR_NUM}/${RIFE_FACTOR_DEN}x。${NC}\n"
        return 0
    fi
    local enable_rife
    read -rp "是否叠加 RIFE 插帧（自选模型、倍率与顺序）？[y/N, 默认 N]: " enable_rife || return 1
    if [[ ! "${enable_rife:-N}" =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}✔ 本次仅执行超分，不插帧。${NC}\n"
        return 0
    fi
    local base_profile=$RUNTIME_PROFILE
    if select_rife_options; then
        RUNTIME_PROFILE="${base_profile} + ${RIFE_MODEL} ${RIFE_FACTOR_NUM}/${RIFE_FACTOR_DEN}x"
        RUNTIME_SLOT=$CUSTOM_SLOT
    else
        echo -e "${YELLOW}本次不启用插帧。${NC}"
    fi
    echo ""
}

# Fail before writing a task when a model the slot needs is missing.
check_profile_models() {
    local missing
    if missing=$(missing_models_for "$SLOT_CHAIN_LINES"); then
        return 0
    fi
    echo -e "${RED}[错误] 方案 ${RUNTIME_PROFILE} 需要的模型不存在：${NC}"
    printf '%s\n' "$missing"
    if [ "$RUNTIME_MODE" -eq 1 ]; then
        echo -e "${YELLOW}请重新解压完整的 Runtime 包。${NC}"
    else
        echo -e "${YELLOW}请先运行 ./deploy.sh --models 下载模型。${NC}"
    fi
    return 1
}

# The task conf: the base conf's [global] (plus native=off when asked) and,
# when the slot is customized or must be copied, that slot's lines.
write_task_conf() {
    local path="$1"
    {
        printf '# Generated by generate_cmd.sh for %s; overwritten on the next generation.\n' \
            "$(basename "$GEN_SCRIPT_PATH")"
        printf '# Base configuration: %s\n\n[global]\n' "$BASE_CONF"
        conf_section_lines "$BASE_CONF" global | grep -Eiv '^native[[:space:]]*='
        [ "$NATIVE_OFF" -eq 1 ] && printf 'native=off\n'
        if [ "$RUNTIME_SLOT" = "$CUSTOM_SLOT" ] ||
           grep -Eqi "^[[:space:]]*\[slot_${RUNTIME_SLOT}\][[:space:]]*$" "$BASE_CONF"; then
            printf '\n[slot_%s]\nprofile_name=%s\n' "$RUNTIME_SLOT" "$RUNTIME_PROFILE"
            printf '%s\n' "$SLOT_CHAIN_LINES"
        fi
    } > "$path"
}

task_conf_needed() {
    [ "$RUNTIME_SLOT" = "$CUSTOM_SLOT" ] || [ "$NATIVE_OFF" -eq 1 ]
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
    local existing canonical_output user_out overwrite_choice
    local batch_dir='' suffix='' dir_name base_name raw_name default_out
    if [ "$IS_CLIP" -eq 1 ]; then
        suffix="_clip_${CLIP_DURATION}s"
    fi
    if [ "$RIFE_ENABLED" -eq 1 ]; then
        suffix+="_rife${RIFE_FACTOR}x"
    fi
    if [ "$UPSCALE_ENABLED" -eq 1 ] || [ "$RIFE_ENABLED" -eq 0 ]; then
        suffix+="_upscaled"
    fi
    suffix+=".mkv"

    if [ "${#INPUT_VIDEOS[@]}" -gt 1 ]; then
        printf '已选择 %s 个视频，统一设置输出位置。\n' "${#INPUT_VIDEOS[@]}"
        printf '默认文件夹：各输入视频所在文件夹。\n默认命名：原文件名（去掉扩展名）+ %s\n' "$suffix"
        read -rp "回车使用默认文件夹和后缀，或输入统一输出文件夹（应用到所有视频，保留默认后缀）: " user_out || return 1
        if [ -n "$user_out" ]; then
            batch_dir=$(printf '%s\n' "$user_out" | sed -e "s/^['\"]//" -e "s/['\"]$//")
            [ -n "$batch_dir" ] || { echo "[错误] 输出文件夹不能为空。" >&2; return 1; }
            batch_dir=$(readlink -m -- "$batch_dir") || return 1
            if [ -e "$batch_dir" ] && [ ! -d "$batch_dir" ]; then
                printf '[错误] 输出位置不是文件夹: %s\n' "$batch_dir" >&2
                return 1
            fi
        fi
    fi

    for INPUT_VIDEO in "${INPUT_VIDEOS[@]}"; do
        dir_name=$(dirname -- "$INPUT_VIDEO")
        base_name=$(basename -- "$INPUT_VIDEO")
        raw_name=${base_name%.*}
        default_out="${batch_dir:-$dir_name}/${raw_name}${suffix}"
        OUTPUT_VIDEO="$default_out"

        if [ "${#INPUT_VIDEOS[@]}" -eq 1 ]; then
            printf '\n输入视频: %s\n' "$INPUT_VIDEO"
            printf '默认推荐输出路径: %b%s%b\n' "$BOLD" "$default_out" "$NC"
            read -rp "是否使用此输出路径？回车默认确认，或直接输入新路径: " user_out || return 1
            if [ -n "$user_out" ]; then
                OUTPUT_VIDEO=$(printf '%s\n' "$user_out" | sed -e "s/^['\"]//" -e "s/['\"]$//")
            fi
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
    if [ -n "$batch_dir" ]; then
        mkdir -p -- "$batch_dir" || return 1
    fi
    INPUT_VIDEO=${INPUT_VIDEOS[0]}
    OUTPUT_VIDEO=${OUTPUT_VIDEOS[0]}

    read -rp "若输出文件已存在，是否默认自动覆盖？[Y/n, 默认 Y]: " overwrite_choice || return 1
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

    # Held-frame reuse is on by default (RIFE chains never reuse).
    read -rp "是否关闭静止帧复用（逐帧推理，速度更慢）？[y/N, 默认 N]: " no_reuse
    if [[ "${no_reuse:-N}" =~ ^[Yy]$ ]]; then
        EXTRA_FLAGS+=("--no-reuse-held-frames")
    fi

    read -rp "是否改用 TensorRT 而不是 native kernel（对比测试用）？[y/N, 默认 N]: " use_trt
    if [[ "${use_trt:-N}" =~ ^[Yy]$ ]]; then
        NATIVE_OFF=1
        echo -e "${YELLOW}将在任务配置中写入 native=off；首次运行会构建 TensorRT Engine。${NC}"
    else
        NATIVE_OFF=0
    fi

    echo ""
}

# 用单引号保留特殊字符，避免 %q 将路径和提示文字显示成大量反斜杠。
write_shell_word() {
    if [[ "$1" =~ ^[a-zA-Z0-9_./:=,+@%-]+$ ]]; then
        printf '%s' "$1"
    else
        printf "'%s'" "${1//\'/\'\\\'\'}"
    fi
}

# 输出可直接由 Bash 执行的命令，保留空格、引号及 $ 等文件名字符。
write_shell_command() {
    local word separator=''
    for word in "$@"; do
        printf '%s' "$separator"
        write_shell_word "$word"
        separator=' '
    done
    printf '\n'
}

# 9. 生成最终命令与独立脚本；set -e 保证任务失败时停止队列。
generate_final_command_and_script() {
    local gen_script_path="$GEN_SCRIPT_PATH"
    local i effective_input arg conf_path="$BASE_CONF"
    local cmd_args=()

    if task_conf_needed; then
        conf_path="$TASK_CONF_PATH"
        write_task_conf "$conf_path" || return 1
    fi

    {
        printf '#!/usr/bin/env bash\nset -euo pipefail\n\n'
        printf 'export PATH='
        write_shell_word "${CUDA_BIN_DIR}:${FFMPEG_INSTALL_DIR}/bin"
        printf ':"$PATH"\n'
        printf 'export LD_LIBRARY_PATH='
        write_shell_word "${PROJECT_ROOT}/build:${FFMPEG_INSTALL_DIR}/lib:${CUDA_LIB_DIR}"
        printf ':"${LD_LIBRARY_PATH:-}"\n'
        if [ -f "$NVENC_FIX_SO" ]; then
            printf 'export LD_PRELOAD='
            write_shell_word "$NVENC_FIX_SO"
            printf '"${LD_PRELOAD:+ $LD_PRELOAD}"\n'
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
                printf 'ffmpeg -y -ss '
                write_shell_word "$CLIP_START"
                printf ' \\\n    -i '
                write_shell_word "$effective_input"
                printf ' \\\n    -t '
                write_shell_word "$CLIP_DURATION"
                printf ' -c copy "$clip_dir/clip.mkv"\n'
            fi
            cmd_args=("--output" "${OUTPUT_VIDEOS[$i]}")
            cmd_args+=("--conf" "${conf_path}")
            cmd_args+=("--slot" "${RUNTIME_SLOT}")
            cmd_args+=("--model-dir" "${CONFIG_MODEL_DIR}")
            cmd_args+=("--rife-model-dir" "${RIFE_MODEL_DIR}")
            cmd_args+=("--trtexec" "${CONFIG_TRTEXEC}")
            cmd_args+=("--decoder" "${DECODER}")
            cmd_args+=("--vcodec" "${VCODEC}")
            cmd_args+=("--vquality" "${VQUALITY}")
            cmd_args+=("--pix-fmt" "${PIX_FMT}")
            [ -n "$OVERWRITE_FLAG" ] && cmd_args+=("${OVERWRITE_FLAG}")
            [ ${#EXTRA_FLAGS[@]} -gt 0 ] && cmd_args+=("${EXTRA_FLAGS[@]}")

            write_shell_word "$AJI_ENCODE_BIN"
            printf ' \\\n    --input '
            if [ "$IS_CLIP" -eq 1 ]; then
                printf '"$clip_dir/clip.mkv"'
            else
                write_shell_word "$effective_input"
            fi
            for arg in "${cmd_args[@]}"; do
                if [[ "$arg" == --* ]]; then
                    printf ' \\\n    '
                else
                    printf ' '
                fi
                write_shell_word "$arg"
            done
            printf '\n'
        done
        printf '\n'
        write_shell_command printf '%s\n' "✔ 全部 ${#INPUT_VIDEOS[@]} 个视频处理完成！"
    } > "$gen_script_path" || return 1
    chmod +x "$gen_script_path" || return 1

    print_header
    echo -e "${BOLD}${GREEN}✔ 已生成 ${#INPUT_VIDEOS[@]} 个视频的串行任务脚本，未自动执行。${NC}"
    echo "全部视频共用处理参数；任一任务失败时停止后续任务。"
    if task_conf_needed; then
        printf '任务配置: %s（Slot %s；重新生成时会覆盖）\n' "$conf_path" "$RUNTIME_SLOT"
    fi
    printf '%b\n' "${YELLOW}以下两种方式任选一种；青色为可复制命令，黄色为操作指引。${NC}"
    echo -e "\n${BOLD}${YELLOW}方式一：运行独立脚本${NC}"
    printf '%b\n' "${YELLOW}前台运行：只复制下面一行。${NC}"
    printf '%b' "$CYAN"
    write_shell_command bash "$gen_script_path"
    printf '%b\n' "$NC"
    printf '%b\n' "${YELLOW}后台运行：只复制下面一行，与前台运行二选一。${NC}"
    printf '%b' "$CYAN"
    printf 'nohup bash '
    write_shell_word "$gen_script_path"
    printf ' > encode.log 2>&1 &\n'
    printf '%b\n' "$NC"
    printf '%b\n' "${YELLOW}查看后台进度：只复制下面一行。${NC}"
    printf '%btail -f encode.log\n%b' "$CYAN" "$NC"
    echo -e "\n${BOLD}${YELLOW}方式二：复制以下完整命令执行（包含串行队列）${NC}"
    printf '%b\n' "${YELLOW}从单独的左括号 ( 开始，到最后单独的右括号 ) 结束，整段一次复制。"
    printf '%b\n' "必须包含这两行括号；不要复制黄色标记行，也不要逐行执行。${NC}"
    printf '\n%b\n' "${BOLD}${YELLOW}# ==================== 开始复制（下一行起） ====================${NC}"
    printf '%b' "$CYAN"
    # 子 Shell 限定 set/trap 的作用域，手动粘贴执行不会退出用户的终端。
    printf '(\n'
    # shebang 只用于独立脚本，不需要放进复制执行的子 Shell。
    tail -n +2 "$gen_script_path"
    printf ')\n'
    printf '%b\n\n' "${NC}${BOLD}${YELLOW}# ==================== 结束复制（上一行止） ====================${NC}"
}

# 主执行流
main() {
    check_binaries
    detect_nvenc_split_count
    print_header
    select_input_video || return 1
    select_processing_profile || return 1
    select_rife_settings || return 1
    check_profile_models || return 1
    warn_unmanaged_engines
    select_clip_mode
    select_output_path || return 1
    select_encoder_and_quality
    select_decoder_and_pixfmt
    select_optional_flags
    generate_final_command_and_script
}

main
