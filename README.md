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

7 步向导式问答，自动生成 `aji_encode` 运行命令：

1. **自动扫描并列出** 系统中的视频文件，选择即用
2. **自动发现 Engine** 模型文件，无需手动拼路径
3. 支持 **整片压制** 或 **截取片段测试**
4. 智能推荐 **输出路径**（自动后缀 `_upscaled`）
5. 预设多档 **编码器 × 画质** 组合（NVENC / x265 / FFV1 等）
6. **硬件解码器与色彩格式** 选择（nvdec / cpu / yuv420p10 等）
7. 汇总确认后 **生成可复制的完整命令** 和独立 `.sh` 脚本，不自动执行

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

## Engine

```sh
trtexec --onnx=models/<model>.onnx --fp16 \
        --minShapes=input:1x3x64x64 --optShapes=input:1x3x1080x1920 \
        --maxShapes=input:1x3x1088x1920 --saveEngine=models/<model>.engine
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
