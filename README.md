# animejanai-inference

Inference shim for the native mpv AnimeJaNai pipeline. `libaji` exposes a
strict C ABI (`include/aji.h`) that `vf_animejanai` in the
[the-database/mpv](https://github.com/the-database/mpv) fork loads at
runtime — mpv never links TensorRT, so the backend and TRT version can be
swapped without rebuilding the player, and on Windows the MSVC-built shim
coexists with a mingw-built mpv.

Current backend: TensorRT (`src/aji_trt.cpp`) with CUDA pre/post kernels
(`src/kernels.cu`): NV12/P010 → fp16 NCHW RGB (BT.709 limited) → engine →
NV12/P010, all device-side. Phase 0 spike scope; chain selection,
engine building, colorspace metadata and CUDA graphs land here in phase 1.

---

## ✨ Fork 特色

本 Fork 在上游基础上新增了两大实用工具，让从零到出片的整个流程从 **数小时手动折腾** 缩短到 **几分钟交互式操作**：

### 1. 🚀 一键自动部署 — `deploy.sh`

交互式环境检测、安装与部署脚本。只需运行 `./deploy.sh`，即可进入向导式菜单，
自动完成从裸机/云容器到可运行超分编码的全部环境搭建：

| 功能 | 说明 |
|------|------|
| 全面环境自检 | 自动检测 GPU 驱动、CUDA 版本、TensorRT、FFmpeg、编译工具链 |
| 一键全自动安装 | CUDA 13.x + TensorRT 11.x + BtbN FFmpeg + NVENC 修复 + 编译，一条命令搞定 |
| NVIDIA 官方源管理 | 自动清理 AutoDL 等平台旧源、配置 NVIDIA Network Repo |
| NVENC 容器多卡修复 | 自动编译 `libnvenc_fix.so`，解决容器环境 NVENC 报错 |
| 模型下载 & Engine 构建 | 内置 AnimeJaNai V3.1 / Real-ESRGAN 等模型直链，自动 trtexec 构建 |
| 1 分钟快速验证 | 截取 1 分钟片段自动超分测试，确认环境无误 |

支持交互式菜单和命令行非交互模式：

```sh
./deploy.sh            # 交互式向导菜单
./deploy.sh --all      # 一键全自动安装与编译
./deploy.sh --check    # 仅环境诊断
./deploy.sh --help     # 查看全部命令行选项
```

### 2. 🎬 交互式命令生成器 — `generate_cmd.sh`

8 步向导式问答，自动生成 `aji_encode` 运行命令：

1. **自动扫描并列出** 系统中的视频文件，选择即用
2. 源码环境自动发现 **Engine**；Runtime 环境自动改为选择内置 **Slot**
3. 可选 **RIFE AI 插帧**：源码模式可选 2x/4x 与处理顺序，Runtime 模式直接选择 RIFE Slot
4. 支持 **整片压制** 或 **截取片段测试**
5. 智能推荐 **输出路径**（自动后缀 `_upscaled` / `_rife2x_upscaled`）
6. 预设多档 **编码器 × 画质** 组合（NVENC / x265 / FFV1 等）
7. **硬件解码器与色彩格式** 选择（nvdec / cpu / yuv420p10 等）
8. 汇总确认后 **生成可复制的完整命令** 和独立 `.sh` 脚本，不自动执行

启用 RIFE 时，插帧和超分在同一个 `aji_encode` 进程中完成：视频只解码、
编码各一次，中间帧始终保留在 CUDA 显存。默认顺序为先插帧再超分，避免在
4K 输出分辨率上执行 RIFE。

```sh
./generate_cmd.sh              # 直接运行
./deploy.sh --gen              # 或从 deploy.sh 菜单进入
```

> [!TIP]
> 详细的手动部署步骤请参阅 [DEPLOY.md](DEPLOY.md)。

---

## Build (Linux)

Needs CUDA toolkit 13.x and TensorRT 11.x (default path:
`~/sdk/tensorrt/usr`, override with `-DAJI_TRT_ROOT=`; on a system TensorRT
install from NVIDIA's apt repo, pass `-DAJI_TRT_ROOT=/usr`).

```sh
CUDACXX=/usr/local/cuda/bin/nvcc cmake -B build -S .
cmake --build build -j
```

## Ubuntu 24.04 免安装 Runtime 包

Runtime 包适合直接部署到 AutoDL、云 GPU 容器或 Ubuntu 24.04 x86_64
主机。包内已经包含 AnimeJaNai/RIFE ONNX 模型、FFmpeg、CUDA Runtime、
TensorRT 运行库、`trtexec` 和对应架构的 TensorRT Builder Resource；宿主机
只需安装兼容的 NVIDIA 驱动，不需要另外安装 CUDA Toolkit、TensorRT、
FFmpeg 或 Python。

### 1. 选择并下载包

每次 Runtime Action 成功运行后，两个压缩包会一起发布到本仓库的
[Releases](../../releases) 页面，并标记为 prerelease：

| 文件名中的架构 | 适用 GPU |
|---|---|
| `sm89` | Ada Lovelace，RTX 40 系 |
| `sm120` | Blackwell，RTX 50 系（包括 RTX 5090） |

下载对应的 `.tar.zst` 及同名 `.sha256` 文件。不要在 RTX 50 系上使用
`sm89` 包，也不要跨 GPU 架构复制已经生成的 `.engine`。

### 2. 校验并解压

以下以 RTX 5090 的 `sm120` 包为例：

```sh
sha256sum -c animejanai-ubuntu24-x86_64-sm120-*.tar.zst.sha256
tar --zstd -xf animejanai-ubuntu24-x86_64-sm120-*.tar.zst
cd animejanai-ubuntu24-x86_64-sm120-*/

# 检查包内文件和当前 GPU/驱动
sha256sum -c SHA256SUMS
./runtime-info.sh
```

如果系统的 `tar` 不支持 `--zstd`，先安装 `zstd`：

```sh
sudo apt-get update
sudo apt-get install -y zstd
```

请把 Runtime 解压到可写目录。程序会在首次运行时把生成的 TensorRT Engine
缓存在 `onnx/` 模型旁边。

### 3. 使用交互式生成器（推荐）

Runtime 包内的 `generate_cmd.sh` 会自动识别当前是免安装 Runtime
环境，因此不会要求选择已有 `.engine`。它会交互式选择输入视频、
Runtime Slot、压制范围、输出路径、编码器和像素格式，最后生成
`run_encode.sh`：

```sh
./generate_cmd.sh
bash ./run_encode.sh
```

生成器会自动为 RTX 40/Ada 使用 `-split_encode_mode 2`，为
RTX 50/Blackwell 使用 `-split_encode_mode 3`。选择 RIFE-only Slot 或
“超分 + RIFE”Slot 时，输出文件名也会自动加上对应后缀。

首次运行某个分辨率会自动构建 TensorRT Engine 并缓存到 `onnx/`；
不需要先手工运行一次 `aji_encode`。同一分辨率后续任务会直接复用缓存。

### 4. 手动运行 `aji_encode`

如需自己组合命令，请使用包根目录的 `./aji_encode` 启动器。它会自动设置动态库路径，
并补齐包内的配置文件、模型目录、RIFE 模型目录和 `trtexec` 路径：

```sh
./aji_encode \
  --input "/path/to/input.mkv" \
  --output "/path/to/output_upscaled.mkv" \
  --slot 1003 \
  --decoder nvdec \
  --vcodec hevc_nvenc \
  --vquality "-cq 18 -preset p7 -tune hq -split_encode_mode 3" \
  --pix-fmt yuv420p10 \
  --overwrite
```

上例适用于 RTX 5090/Blackwell。RTX 40 系把 `-split_encode_mode 3` 改为
`-split_encode_mode 2`。首次运行会根据输入分辨率构建 TensorRT Engine，
因此启动阶段会比后续运行慢；同一分辨率再次运行时会直接复用缓存。

`--slot` 用来选择内置处理方案：

| Slot | 处理方案 |
|---:|---|
| `1001` | Quality 超分 |
| `1002` | Balanced 超分 |
| `1003` | Performance 超分（默认） |
| `2001` | Sharp Balanced 超分 |
| `2002` | Sharp Performance 超分 |
| `2003` | SD Compact 超分 |
| `2025` / `2026` | 仅 RIFE v4.25 / v4.26 2x 插帧 |
| `3025` / `3026` | Performance 超分 + RIFE 2x 插帧 |

省略 `--slot` 时默认使用 `1003`，也可以临时设置默认值：

```sh
AJI_SLOT=3026 ./aji_encode \
  --input "/path/to/input.mkv" \
  --output "/path/to/output_rife2x_upscaled.mkv" \
  --decoder nvdec --vcodec hevc_nvenc \
  --vquality "-cq 18 -preset p7 -tune hq -split_encode_mode 3" \
  --pix-fmt yuv420p10 --overwrite
```

包内的 FFmpeg、FFprobe 和 TensorRT 工具也应通过根目录启动器调用，以确保
自动加载包内动态库：

```sh
./ffprobe -hide_banner "/path/to/input.mkv"
./ffmpeg -hide_banner -encoders
./trtexec --version
```

如需使用自己的 `.engine`，`aji_encode` 仍支持直连模式：传入
`--engine`、`--max-width` 和 `--max-height`。

### 5. 本地构建 Runtime 包

如需自行打包或构建其他 GPU 架构：

```sh
./scripts/package-ubuntu24-runtime-local.sh --gpu-arch 89
./scripts/package-ubuntu24-runtime-local.sh --gpu-arch 120
```

产物位于 `dist/`。详细的架构列表和打包说明见
[`packaging/ubuntu24/README.md`](packaging/ubuntu24/README.md)。

## Engine

```sh
trtexec --onnx=models/<model>.onnx --fp16 \
        --minShapes=input:1x3x1080x1920 --optShapes=input:1x3x1080x1920 \
        --maxShapes=input:1x3x1080x1920 --builderOptimizationLevel=5 \
        --saveEngine=models/<model>.engine
```

## Harness (no player needed)

```sh
ffmpeg -i clip.mkv -frames:v 12 -pix_fmt p010le -f rawvideo in.raw
./build/aji_harness --engine models/<model>.engine --input in.raw \
    --width 1400 --height 1080 --format p010 --output out.raw
```

Reports device-side ms/frame for the full pre+infer+post chain
(RTX 5090 reference: ~3.8 ms at 1920x1080→4K fp16).

## mpv usage (spike)

```sh
mpv --hwdec=nvdec \
    --vf=animejanai=engine=models/<model>.engine:lib=build/libaji.so video.mkv
```
