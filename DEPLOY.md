# AnimeJaNai-Inference 从零部署指南

> 适用系统：Ubuntu 24.04 LTS（AutoDL / 云 GPU 容器）  
> 适用 GPU：NVIDIA Turing 及以上（RTX 20/30/40/50 系列，计算能力 ≥ 7.5）  
> 项目地址：[the-database/animejanai-inference](https://github.com/the-database/animejanai-inference)

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [前提条件](#2-前提条件)
3. [Step 1：安装 NVIDIA 驱动（如未预装）](#3-step-1安装-nvidia-驱动如未预装)
4. [Step 2：安装 CUDA Toolkit](#4-step-2安装-cuda-toolkit)
5. [Step 3：安装 TensorRT](#5-step-3安装-tensorrt)
6. [Step 4：安装编译工具与 ffmpeg 开发库](#6-step-4安装编译工具与-ffmpeg-开发库)
7. [Step 5：克隆并编译项目](#7-step-5克隆并编译项目)
8. [Step 6：下载超分模型（ONNX）](#8-step-6下载超分模型onnx)
9. [Step 7：构建 TensorRT Engine](#9-step-7构建-tensorrt-engine)
10. [Step 8：运行超分（aji_encode）](#10-step-8运行超分aji_encode)
11. [aji_encode 完整参数参考](#11-aji_encode-完整参数参考)
12. [常见问题排错](#12-常见问题排错)
13. [附录：GPU 架构对照表](#13-附录gpu-架构对照表)

---

## 1. 整体架构概览

```
输入视频 (.mkv/.mp4)
  │
  ▼
ffmpeg/NVDEC 硬件解码
  │
  ▼
libaji (CUDA 色彩空间转换: NV12/P010 → FP16 RGB)
  │
  ▼
TensorRT Engine 推理 (2× 超分辨率)
  │
  ▼
libaji (CUDA 色彩空间转换: FP16 RGB → NV12/P010)
  │
  ▼
ffmpeg/NVENC 硬件编码
  │
  ▼
输出 4K 视频 (.mkv/.mp4)
```

全链路在 GPU 显存内完成（零拷贝），不产生未压缩的中间文件。

---

## 2. 前提条件

| 组件 | 最低版本 | 推荐版本 | 说明 |
|------|---------|---------|------|
| NVIDIA Driver | 570+ | 570+ / 580+ | 需支持 CUDA 13.x+ |
| CUDA Toolkit | 13.x | 13.x (严格要求) | 项目编译 sm_100 / sm_120 .cu 内核 |
| TensorRT | 11.x | 11.x (严格要求) | 超分模型推理引擎 (强类型) |
| CMake | 3.24+ | 3.28+ | 构建系统 |
| GCC/G++ | 11+ | 13+ | C17 / C++17 编译器 |
| ffmpeg (libav*) | 6.x | 7.x+ / 8.x (BtbN) | aji_encode 需要开发库 |
| GPU 显存 | 4 GB | 8 GB+ | 4K 超分推理约需 3-4 GB |

> [!IMPORTANT]
> AutoDL 等平台的 GPU 容器通常已预装 NVIDIA 驱动和 CUDA Toolkit，可跳过 Step 1-2，直接从 Step 3 开始。
> 启动容器时请选择 **CUDA 13.x 版本的镜像**。

---

## 3. Step 1：安装 NVIDIA 驱动（如未预装）

> AutoDL / 云容器一般已内置驱动，可先运行 `nvidia-smi` 验证。如已有输出则跳过本步。

```bash
# 验证驱动
nvidia-smi

# 如未安装，Ubuntu 24.04 可用：
sudo apt update
sudo apt install -y nvidia-driver-570
sudo reboot
```

---

## 4. Step 2：安装 CUDA Toolkit 13.x

> [!NOTE]
> AutoDL 镜像通常已有 CUDA。检查方法：`nvcc --version`。如已安装 13.x 可跳过。

### 方式 A：通过 NVIDIA 官方 .run 安装器（推荐，版本可控）

```bash
# 下载 CUDA 13.3（示例，请去 https://developer.nvidia.com/cuda-downloads 获取最新链接）
wget https://developer.download.nvidia.com/compute/cuda/13.3.0/local_installers/cuda_13.3.0_570.86.15_linux.run

# 安装（仅 toolkit，不覆盖驱动）
sudo sh cuda_13.3.0_570.86.15_linux.run --toolkit --silent

# 添加环境变量
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证
nvcc --version
```

### 方式 B：通过 deploy.sh 自动化脚本安装（推荐）

```bash
./deploy.sh --cuda
```

---

## 5. Step 3：安装 TensorRT 11.x

### 方式 A：通过 NVIDIA apt 仓库安装到系统路径（推荐，最简单）

```bash
# 1. 添加 NVIDIA apt 仓库（如还没有）
#    参考：https://docs.nvidia.com/tensorrt/install-guide/index.html
#    以 CUDA 13.x 为例：

# 安装 NVIDIA keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# 2. 安装 TensorRT 11（开发库 + 运行时 + trtexec 工具）
sudo apt install -y tensorrt tensorrt-dev

# 3. 验证
trtexec --help | head -1
# 应输出类似：&&&& RUNNING TensorRT.trtexec [TensorRT v110100]
dpkg -l | grep nvinfer
```

> [!TIP]
> 通过 apt 安装的 TensorRT，头文件在 `/usr/include/`，库文件在 `/usr/lib/x86_64-linux-gnu/`。
> 编译时需要传递参数：`-DAJI_TRT_ROOT=/usr`

### 方式 B：手动下载 deb/tar 包解压

```bash
# 从 https://developer.nvidia.com/tensorrt 下载对应 CUDA 13.x 版本的 tar 包
# 解压到任意位置，例如 ~/sdk/tensorrt
mkdir -p ~/sdk/tensorrt
tar -xzf TensorRT-11.x.x.x.Linux.x86_64-gnu.cuda-13.x.tar.gz -C ~/sdk/tensorrt --strip-components=1

# 此时默认的 CMake 路径 ~/sdk/tensorrt/usr 即可用，无需额外参数
# 或者将 trtexec 加入 PATH：
export PATH=~/sdk/tensorrt/usr/bin:$PATH
```

---

## 6. Step 4：安装编译工具与 BtbN FFmpeg Shared 开发库

由于 Ubuntu 官方 apt 源的 FFmpeg 版本较旧且缺少部分硬件加速支持，推荐安装最新的 **BtbN FFmpeg Shared** 构建：

```bash
# 1. 安装基础编译工具
sudo apt update
sudo apt install -y build-essential cmake pkg-config git wget tar xz-utils

# 2. 下载并安装 BtbN FFmpeg Shared 构建
# 强烈推荐 n8.1 版本（与驱动 570~595 的 NVENC API 完美适配）：
wget -O /tmp/ffmpeg.tar.xz https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-shared-8.1.tar.xz
sudo mkdir -p /opt/ffmpeg
sudo tar -xJf /tmp/ffmpeg.tar.xz -C /opt/ffmpeg --strip-components=1

# 3. 修正 pkg-config 中的 prefix 路径
for pc in /opt/ffmpeg/lib/pkgconfig/*.pc; do
    sudo sed -i 's|^prefix=.*|prefix=/opt/ffmpeg|' "$pc"
done

# 4. 配置系统动态链接器与环境变量
echo "/opt/ffmpeg/lib" | sudo tee /etc/ld.so.conf.d/ffmpeg.conf
sudo ldconfig
sudo ln -sf /opt/ffmpeg/bin/ffmpeg /usr/local/bin/ffmpeg
sudo ln -sf /opt/ffmpeg/bin/ffprobe /usr/local/bin/ffprobe

export PKG_CONFIG_PATH="/opt/ffmpeg/lib/pkgconfig:$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="/opt/ffmpeg/lib:$LD_LIBRARY_PATH"
export PATH="/opt/ffmpeg/bin:$PATH"
```

> [!NOTE]
> `libav*-dev` 是 `aji_encode` 编码工具的编译依赖。配置好 `PKG_CONFIG_PATH=/opt/ffmpeg/lib/pkgconfig` 后，CMake 会自动识别 BtbN 的完整 FFmpeg 共享库。

---

## 7. Step 5：克隆并编译项目

```bash
# 克隆代码
git clone https://github.com/the-database/animejanai-inference.git
cd animejanai-inference

# 配置（根据 TensorRT 安装方式选择参数）

# ---- 如果 TensorRT 通过 apt 安装在系统路径 (/usr) ----
CUDACXX=/usr/local/cuda/bin/nvcc cmake -B build -S . -DAJI_TRT_ROOT=/usr

# ---- 如果 TensorRT 手动解压在 ~/sdk/tensorrt/usr（默认值，无需额外参数）----
CUDACXX=/usr/local/cuda/bin/nvcc cmake -B build -S .

# ---- 如果只有当前机器的 GPU（加速编译，只编译当前架构）----
CUDACXX=/usr/local/cuda/bin/nvcc cmake -B build -S . -DAJI_TRT_ROOT=/usr -DCMAKE_CUDA_ARCHITECTURES=native

# 编译
cmake --build build -j$(nproc)
```

### 验证编译成功

```bash
ls -lh build/libaji.so build/libaji_trt.so build/aji_harness build/aji_encode
```

应看到 4 个文件：
| 文件 | 说明 |
|------|------|
| `libaji.so` | C ABI 调度器（mpv 加载用） |
| `libaji_trt.so` | TensorRT + CUDA 推理后端 |
| `aji_harness` | 底层 Raw 帧推理测试工具 |
| `aji_encode` | **端到端视频超分编码工具（主要使用这个）** |

---

## 8. Step 6：获取超分模型（ONNX）

推荐常用超分模型：

| 模型名称 | 用途 | 速度 / 倍率 | 输入节点名 |
|----------|------|-------------|------------|
| `2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx` (`performance.onnx`) | 2× 超分（极速偏画质 - 推荐） | ⚡⚡⚡ 最快 (2x) | `input` |
| `2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx` (`balanced.onnx`) | 2× 超分（画质与速度均衡） | ⚡⚡ 均衡 (2x) | `input` |
| `2x_AnimeJaNai_HD_V3.1Sharp1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx` (`balanced_sharp1.onnx`) | 2× 超分（清晰锐化版） | ⚡⚡ 锐化 (2x) | `input` |
| `2x_AnimeJaNai_SD_V1beta34_Compact_1x3xHxW_dyn-HW_strong_fp16_op21_dynamo.onnx` (`sd_compact.onnx`) | 2× 标清修复（适合 480p/老番） | ⚡⚡ 标清 (2x) | `input` |
| `realesr-animevideov3-v0.2.5.0-fp16-dynamic.onnx` | AnimeVideo-v3 动漫视频轻量超分 | 原生 4x | `input` |
| `RealESRGAN_x4plus_anime_6B.onnx` (`realesrgan_anime6b.onnx`) | 4× 经典原版动漫超分（线条强化与降噪） | ⚡ 经典 (4x) | `image.1` |

AnimeVideo-v3 使用官方 v0.2.5.0 的 `realesr-animevideov3.pth` 转换，
输入/输出为 FP16 RGB NCHW，batch 和宽高动态，ONNX opset 17。
`deploy.sh` 模型选项 6 自动下载并构建 Engine；Runtime 使用 Slot 2004，
首次按输入尺寸构建 Engine 并缓存。原生 4× 表示 1080p → 8K；
如需最终 4K，可在 `generate_cmd.sh` 的输出缩放设置中指定高度 2160。
SHA256：`d584ed9c21a2c2448d96d5e3e0a2d719a0e573d36eb4244cb07e68be96e3a0ab`。

```bash
# 创建模型目录
mkdir -p ~/models

# AnimeVideo-v3 (native 4x)
wget -O ~/models/realesr-animevideov3-v0.2.5.0-fp16-dynamic.onnx "https://r2.ahdiua.com/realesr-animevideov3-v0.2.5.0-fp16-dynamic.onnx"

# 1 & 2: 获取 AnimeJaNai 官方 V3.1 模型 (R2 CDN 高速直连秒级下载)
wget -O ~/models/performance.onnx "https://r2.ahdiua.com/2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx"
wget -O ~/models/balanced.onnx "https://r2.ahdiua.com/2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx"

# 3: 下载原版 Real-ESRGAN Anime 6B (4x)
wget -O ~/models/realesrgan_anime6b.onnx \
  "https://huggingface.co/deepghs/imgutils-models/resolve/main/real_esrgan/RealESRGAN_x4plus_anime_6B.onnx"
```

---

## 9. Step 7：构建 TensorRT Engine

> [!IMPORTANT]
> **Engine 文件与 GPU 型号 + TensorRT 版本 + CUDA 版本绑定**，不能跨平台复用。
> 每台新机器必须用 `trtexec` 从 ONNX 重新构建。构建约需 1-2 分钟。

```bash
# 1. 为 AnimeJaNai 构建 Engine（输入节点为 input，以 1080p 为例）
trtexec \
  --onnx=~/models/performance.onnx \
  --minShapes=input:1x3x1080x1920 \
  --optShapes=input:1x3x1080x1920 \
  --maxShapes=input:1x3x1080x1920 \
  --builderOptimizationLevel=5 \
  --skipInference \
  --saveEngine=~/models/performance_1080p.engine

# 2. 为 Real-ESRGAN 构建 Engine（注意：其输入节点名称为 image.1）
trtexec \
  --onnx=~/models/realesrgan_anime6b.onnx \
  --minShapes=image.1:1x3x1080x1920 \
  --optShapes=image.1:1x3x1080x1920 \
  --maxShapes=image.1:1x3x1080x1920 \
  --builderOptimizationLevel=5 \
  --skipInference \
  --saveEngine=~/models/realesrgan_anime6b_1080p.engine
```

### 构建参数说明

| 参数 | 说明 |
|------|------|
| `--minShapes` / `--optShapes` / `--maxShapes` | 三者设为相同的源视频分辨率，生成固定 shape Engine |
| `--builderOptimizationLevel=5` | 使用 TensorRT 最高构建优化等级（构建会更慢） |
| `--skipInference` | 仅构建 Engine，跳过推理基准 |
| `--fp16` | （仅适用于 TensorRT 10 及以下版本）在 TensorRT 11+ 中已废弃 |

### 不同源分辨率的 Engine 示例

```bash
# 720p 输入
trtexec --onnx=~/models/performance.onnx \
  --minShapes=input:1x3x720x1280 \
  --optShapes=input:1x3x720x1280 \
  --maxShapes=input:1x3x720x1280 \
  --builderOptimizationLevel=5 \
  --skipInference \
  --saveEngine=~/models/performance_720p.engine

# 1080p 输入
trtexec --onnx=~/models/performance.onnx \
  --minShapes=input:1x3x1080x1920 \
  --optShapes=input:1x3x1080x1920 \
  --maxShapes=input:1x3x1080x1920 \
  --builderOptimizationLevel=5 \
  --skipInference \
  --saveEngine=~/models/performance_1080p.engine
```

> [!TIP]
> 每个不同的源分辨率建议单独生成固定 shape Engine。如果改为不同的
> `minShapes` / `optShapes` / `maxShapes`，将会生成通用性更高但速度较低的动态 Engine。

---

## 10. Step 8：运行超分（aji_encode）

### 基本用法

```bash
cd animejanai-inference

# 1080p → 4K 超分（NVENC 硬件编码，10-bit HEVC）
./build/aji_encode \
  --input /path/to/input_video.mkv \
  --output /path/to/output_4k.mkv \
  --engine ~/models/performance_1080p.engine \
  --max-width 1920 --max-height 1080 \
  --vcodec hevc_nvenc \
  --overwrite
```

### 截取片段测试（推荐先用短片段验证）

```bash
# 截取 1 分钟测试片段（零消耗，只是流拷贝）
ffmpeg -y -ss 00:01:00 -i input_video.mkv -t 60 -c copy test_clip.mkv

# 对测试片段超分
./build/aji_encode \
  --input test_clip.mkv \
  --output test_clip_4k.mkv \
  --engine ~/models/performance_1080p.engine \
  --max-width 1920 --max-height 1080 \
  --vcodec hevc_nvenc \
  --overwrite

# 验证输出
ffprobe -hide_banner test_clip_4k.mkv
# 应显示：Video: hevc, 3840x2160 [SAR 1:1 DAR 16:9]
```

### 使用 AV1 NVENC（Ada / RTX 40 系及更新 GPU）

使用 `generate_cmd.sh` 时会自动按 GPU 架构选择 split-frame 数量：Blackwell
请求 3 路，其他架构保留 2 路请求。若具体 GPU 的 NVENC 数量少于请求值，
驱动会自动收缩到实际可用数量。手写命令时也可以按需指定：

```bash
./build/aji_encode \
  --input test_clip.mkv \
  --output test_clip_4k_av1.mkv \
  --engine ~/models/performance_1080p.engine \
  --max-width 1920 --max-height 1080 \
  --vcodec av1_nvenc \
  --vquality "-cq 20 -preset p6 -tune hq -split_encode_mode 2" \
  --pix-fmt yuv420p10 \
  --overwrite
```

### 使用软件编码器（无 NVENC 的 GPU，如 Tesla T4 / A10）

```bash
./build/aji_encode \
  --input test_clip.mkv \
  --output test_clip_4k.mkv \
  --engine ~/models/performance_1080p.engine \
  --max-width 1920 --max-height 1080 \
  --vcodec libx265 \
  --overwrite
```

---

## 11. aji_encode 完整参数参考

```
aji_encode --input <f> --output <f> [选项...]

必选参数（直连模式）:
  --input <file>          输入视频文件路径
  --output <file>         输出视频文件路径
  --engine <file>         TensorRT Engine 文件路径
  --max-width <W>         输入视频的最大宽度
  --max-height <H>        输入视频的最大高度

可选参数:
  --vcodec <codec>        视频编码器（默认 hevc_nvenc）
                          可选：hevc_nvenc, av1_nvenc, h264_nvenc,
                                libx265, libx264, ffv1
  --vquality "<args>"     编码器质量参数
  --pix-fmt <fmt>         输出像素格式（默认 yuv420p10）
                          可选：yuv420p, yuv420p10, yuv444p, yuv444p10
  --final-resize-height H 最终输出高度（缩小到指定高度）
  --final-resize-factor % 最终输出缩放百分比
  --decoder <mode>        解码模式：auto（默认）, nvdec, cpu
  --overwrite             覆盖已有输出文件
  --no-audio              不拷贝音轨
  --no-subs               不拷贝字幕
  --no-chapters           不拷贝章节信息
  --progress <mode>       进度格式：line（默认，含 ETA）, json, none
  --log <file>            日志输出到文件
  --build-only            仅构建 Engine，不执行编码

直连 Engine 模式的可选 RIFE 参数（与超分同一管道）:
  --rife-model <name>     RIFE ONNX 文件名（不含 .onnx，如 rife_v4.26）
  --rife-model-dir <dir>  RIFE ONNX 与 Engine 缓存目录
  --rife-factor <N>       整数插帧倍率 2-8（默认 2）
  --rife-order <order>    before（先插帧再超分，默认）或 after
  --rife-scene-threshold  转场检测阈值（默认 0.150）

配置模式（替代 --engine 直连模式）:
  --conf <file>           animejanai.conf 配置文件路径
  --slot <N>              配置 Slot 编号
  --model-dir <dir>       ONNX 模型目录
  --trtexec <path>        trtexec 可执行文件路径
  --rife-model-dir <dir>  RIFE 插帧模型目录
  --backend tensorrt      推理后端（默认 tensorrt）
```

例如，使用同一次解码/编码完成 RIFE 2x 插帧与 2x 超分：

```sh
aji_encode --input input.mkv --output output.mkv \
  --engine models/balanced_1080p.engine --max-width 1920 --max-height 1080 \
  --rife-model-dir onnx/rife --rife-model rife_v4.26 \
  --rife-factor 2 --rife-order before \
  --decoder nvdec --vcodec hevc_nvenc --pix-fmt yuv420p10
```

首次运行会按 RIFE 工作分辨率构建并缓存 TensorRT Engine；后续任务直接复用。

---

## 12. 常见问题排错

### Q1: CMake 报 `Could NOT find CUDAToolkit`
```
确保 nvcc 在 PATH 中，或通过 CUDACXX 环境变量显式指定：
  CUDACXX=/usr/local/cuda/bin/nvcc cmake -B build -S .
```

### Q2: CMake 报 `Package 'libavformat' not found`
```
安装 ffmpeg 开发库：
  sudo apt install libavformat-dev libavcodec-dev libavutil-dev libavfilter-dev libswscale-dev
CMake 会跳过 aji_encode 的编译，但 libaji.so 和 aji_harness 仍能正常编译。
```

### Q3: 编译报 TensorRT 头文件找不到（`NvInfer.h: No such file`）
```
检查 TensorRT 安装路径并在 cmake 中指定：
  # apt 安装到系统路径
  cmake -B build -S . -DAJI_TRT_ROOT=/usr
  # 手动解压到 ~/sdk/tensorrt
  cmake -B build -S . -DAJI_TRT_ROOT=$HOME/sdk/tensorrt/usr
```

### Q4: trtexec 构建 Engine 时报 `CUDA out of memory`
```
减小 --maxShapes 的分辨率，或关闭其他占用 GPU 显存的进程。
4K Engine 构建约需 3-4 GB 显存。
```

### Q5: aji_encode 运行时报 `engine rejects input WxH`
```
源视频分辨率超出了 Engine 的 --maxShapes。
需要用更大的 --maxShapes 重新构建 Engine。
```

### Q6: aji_encode 运行时报 `libaji_trt.so: cannot open shared object file`
```
确保运行 aji_encode 时在 animejanai-inference 项目目录下执行，
或将 build/ 目录加入 LD_LIBRARY_PATH：
  export LD_LIBRARY_PATH=$(pwd)/build:$LD_LIBRARY_PATH
```

### Q7: NVENC 编码报错（Tesla / 数据中心 GPU）
```
Tesla T4、A10 等数据中心 GPU 支持 NVENC，但某些无显示输出的型号可能不支持。
改用软件编码器：
  --vcodec libx265   或   --vcodec libx264
```

---

## 13. 附录：GPU 架构对照表

| 架构名称 | SM 版本 | 代表 GPU | 说明 |
|----------|--------|----------|------|
| Turing | sm_75 | RTX 2060/2070/2080, T4 | 最低支持 |
| Ampere | sm_80 | A100 | 数据中心 |
| Ampere | sm_86 | RTX 3060/3070/3080/3090 | 消费级 |
| Ada Lovelace | sm_89 | RTX 4060/4070/4080/4090 | 消费级 |
| Hopper | sm_90 | H100 | 数据中心 |
| Blackwell | sm_100 | B100/B200 | 数据中心 |
| Blackwell | sm_120 | RTX 5070/5080/5090 | 消费级 |

本项目默认编译所有架构的 CUDA 内核（sm75 ~ sm120），可在任意支持的 GPU 上运行。
如只在当前机器运行，可添加 `-DCMAKE_CUDA_ARCHITECTURES=native` 加速编译。

---

## 快速一键脚本（AutoDL Ubuntu 24.04 + CUDA 已预装）

```bash
#!/bin/bash
set -e

# 1. 安装依赖
sudo apt update && sudo apt install -y \
    build-essential cmake pkg-config git \
    libavformat-dev libavcodec-dev libavutil-dev libavfilter-dev libswscale-dev \
    ffmpeg

# 2. 安装 TensorRT（如未预装）
# sudo apt install -y tensorrt tensorrt-dev

# 3. 克隆并编译
git clone https://github.com/the-database/animejanai-inference.git
cd animejanai-inference
CUDACXX=/usr/local/cuda/bin/nvcc cmake -B build -S . \
    -DAJI_TRT_ROOT=/usr \
    -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j$(nproc)

# 4. 下载模型
mkdir -p ~/models
# wget -O ~/models/performance.onnx "<ONNX_DOWNLOAD_URL>"

# 5. 构建 Engine（替换为你的模型路径）
trtexec \
    --onnx=~/models/performance.onnx \
    --minShapes=input:1x3x1080x1920 \
    --optShapes=input:1x3x1080x1920 \
    --maxShapes=input:1x3x1080x1920 \
    --builderOptimizationLevel=5 \
    --skipInference \
    --saveEngine=~/models/performance_1080p.engine

# 6. 超分测试
# ./build/aji_encode \
#     --input input.mkv --output output_4k.mkv \
#     --engine ~/models/performance_1080p.engine \
#     --max-width 1920 --max-height 1080 \
#     --vcodec hevc_nvenc --overwrite

echo "✅ 部署完成！请参考上方注释的第 4-6 步下载模型并运行。"
```
## deploy 与自动缓存的引擎命名

`deploy.sh` 使用可读的 `<模型名>_<分辨率>.engine` 命名，例如
`performance_1080p.engine`、`balanced_720p.engine`、`performance_960x540.engine`。
构建不再依赖 `build/aji_engine_path`，每次按当前 GPU 和 TensorRT 环境重新构建，
成功后才替换同名引擎；构建失败保留原文件。
这些引擎通过 `--engine` 直连模式使用；配置模式仍使用推理后端自己的哈希缓存名称。

构建输出同时保存在 `.engine.build.log`；`.engine.label` 记录模型名和输入尺寸，
供命令生成器展示。已有哈希名称引擎不会被自动改名或删除，仍可通过直连模式使用。
引擎目录不进行基于文件名后缀的自动清理。

模型下载优先使用 `aria2c`（最多 4 连接，失败重试），未安装或下载失败时回退到 Python。
`--ffmpeg` / `--venv` 安装步骤会安装 `aria2`，已有环境也可运行 `sudo apt-get install -y aria2`。
下载显式设置兼容 CDN 的 User-Agent，避免默认 `Python-urllib` 请求被返回 HTTP 403；
模型地址要求 HTTPS，保留证书验证、ONNX 校验、下载收据和镜像回退，失败时打印对应 URL。

## deploy.sh 的检测、版本选择与下载校验

`--check` 仅诊断，不自动克隆源码或切换 `/usr/local/cuda`，依赖检查失败返回非零。
脚本默认操作自身所在的 fork 目录；从其他目录调用时，可用 `--project-root` 明确指定项目。
自动安装支持 Ubuntu 22.04/24.04 x86_64，并根据发行版选择对应 NVIDIA 源。

```sh
./deploy.sh --check
./deploy.sh --build --cuda-root /usr/local/cuda-13.2 --trt-root /usr
./deploy.sh --all --yes --cuda-version 13.2 --ffmpeg-variant n8.1
```

`--yes` 明确接受交互问题已有的默认值；没有默认值的问题仍然需要输入。
未指定 `--yes` 时，EOF 会取消操作；`--all` 在非交互环境中要求 `--yes`。
CUDA 安装版本必须为 13.x，TensorRT 的安装包限定 11.x。编译时 CUDA 以
`--cuda-root`、`CUDACXX`、PATH 的顺序选择；TensorRT 以 `--trt-root` 为准，
未指定时依次检查 `/usr` 和 `~/sdk/tensorrt/usr`。可用 `--trtexec` 指定工具路径，
其报告版本必须与所选开发头文件一致。CMake 和引擎构建使用同一组解析结果。

FFmpeg 固定到 BtbN `autobuild-2026-08-31-13-27` 的月底构建，支持 n8.1 与 master
两个变体；NVENC 补丁固定到 `1475091ae9e8fb23cc22d565adf932a488d00bcd`。
这两类下载和 NVIDIA keyring 均核对脚本中记录的 SHA256；更新时须同时更新 URL 与摘要。
BtbN 的月底构建按其发布策略保留两年，固定资产失效时应更新版本，不会静默回退到 latest。
FFmpeg 在移除旧安装前验证压缩包及必要成员，安装失败返回非零，不保留回滚副本。
NVENC 补丁仅在检测到设备枚举故障且补丁测试通过时用于单次编码进程。

模型下载需要先执行 `--venv`。`scripts/download_model.py` 检查 HTTPS、下载长度及
ONNX 结构后才替换目标文件，并记录 `.download.json` 中的来源与 SHA256。
该记录用于检查本地文件完整性和 URL 是否一致，不是发布者签名；辅助工具支持
`--sha256` 传入可信摘要。无记录的旧下载会重新获取，自定义 URL 使用不同文件名。
模型必须是包含全部权重的单文件 ONNX；外置权重模型请自行准备并通过本地路径使用。
这些下载记录不改变引擎缓存的命名或复用规则。

安装会更新对应的 CUDA/FFmpeg profile，并迁移旧脚本写入 `.bashrc` 的已知环境变量行，
保留其他用户配置。库搜索路径不包含空目录项。一分钟测试的输入和输出放在独立的
`build/deploy-test.*` 目录，避免覆盖已有测试结果；失败返回非零。

部署逻辑的隔离回归测试可运行 `python3 sanity/test_deploy.py`，不需要 root、GPU 或网络。
安装 `onnx` 后会额外运行真实 ONNX 校验测试。静态检查使用 `shellcheck deploy.sh`。
