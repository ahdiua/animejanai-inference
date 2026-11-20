# animejanai-inference

Optional addon interfaces are documented in [docs/ADDON-API.md](docs/ADDON-API.md), including build dependencies, inactive behavior and qualification limits.

A GPU inference engine for upscaling. `libaji` runs ONNX upscaling models, plus RIFE
interpolation and temporal models, entirely on the GPU, with three backends: TensorRT and
TensorRT-RTX (NVIDIA, through CUDA) and DirectML (Windows, any DirectX 12 GPU). Callers load it at
runtime through one of two C ABIs, so they never link TensorRT:

- **Video** (`include/aji.h`): GPU-resident YUV frames in, upscaled frames out, with chains,
  RIFE and temporal models configured by `animejanai.conf`. `vf_animejanai` in the
  [the-database/mpv](https://github.com/the-database/mpv) fork uses it for real-time playback, and
  `aji_encode` uses it to upscale video files.
- **Still images** (`include/aji_image.h`): 8-bit images of any size, tiled on the GPU.

Because the player loads it at runtime, the backend and TensorRT version can change without
rebuilding mpv, and on Windows the MSVC-built library coexists with a mingw-built mpv.

Used by [mpv-AnimeJaNai](https://github.com/the-database/mpv-AnimeJaNai) (real-time playback),
[VideoJaNai](https://github.com/the-database/VideoJaNai) (video files, through `aji_encode`),
[MangaJaNai GUI](https://github.com/the-database/MangaJaNaiGUI) and
[MangaJaNaiCLI](https://github.com/the-database/MangaJaNaiCLI) (manga and images), and
[yosh](https://github.com/the-database/yosh)'s AI upscaling add-on.

---

## ✨ Fork 特色

本 Fork 在上游基础上新增了离线压制与 Ubuntu 免安装 Runtime 的完整流程，让从零到出片
从 **数小时手动折腾** 缩短到 **几分钟交互式操作**。所有处理方案都通过
`animejanai.conf` 的 Slot 运行，**不再需要手动构建或选择 TensorRT Engine**：

- **native kernel（默认开启）**：上游手写的 CUDA kernel 直接解析 ONNX，覆盖本 Fork
  打包的全部模型（SPANF3、Compact、ESRGAN/RRDB、RIFE），不需要 Engine。
- **其余模型** 回退 TensorRT：首次处理某个分辨率时由 backend 自动构建 Engine 并缓存在
  模型目录，之后直接复用。日志会打印 `[engine-build] Starting trtexec ...` / `Completed`。
- **静止帧复用（默认开启）**：画面不变的帧复用上一帧的超分结果（RIFE 方案不复用），
  可用 `--no-reuse-held-frames` 关闭。

### 1. 🚀 一键自动部署 — `deploy.sh`

| 功能 | 说明 |
|------|------|
| 全面环境自检 | 检测 GPU 驱动、CUDA、TensorRT、FFmpeg、编译工具链与模型是否齐全 |
| 一键全自动安装 | CUDA 13.x + TensorRT 11.x + BtbN FFmpeg + NVENC 修复 + 编译 |
| NVIDIA 官方源管理 | 清理 AutoDL 等平台旧源、配置 NVIDIA Network Repo |
| NVENC 容器多卡修复 | 编译 `libnvenc_fix.so`，解决容器环境 NVENC 报错 |
| 模型下载 | 按 conf 布局下载到 `~/models`（含 APISR、Anime 6B 与 fp16 RIFE），与 Runtime 包使用同一份清单和 SHA256 |
| 旧安装迁移 | `--migrate`：按校验和把旧版短文件名模型改名为 conf 名称，列出旧的手工 Engine，确认后才删除 |
| 1 分钟快速验证 | 截取 1 分钟片段用 Slot 1003 超分测试 |

```sh
./deploy.sh            # 交互式向导菜单
./deploy.sh --all      # 一键全自动安装与编译
./deploy.sh --models   # 下载模型（Engine 首次使用时自动构建）
./deploy.sh --migrate  # 迁移旧版模型文件名、清理旧 Engine
./deploy.sh --check    # 仅环境诊断
```

源码安装的模型布局与 Runtime 包相同：`~/models/<conf 中的模型名>.onnx` 与
`~/models/rife/rife_v<版本>.onnx`。

> [!IMPORTANT]
> 从旧版升级：旧版 `deploy.sh` 把模型存成 `performance.onnx` 等短名称，并在 `~/models`
> 中手工构建 `*_1080p.engine`。新版 libaji 首次用 `~/models` 运行时，会删除其中所有
> 不带当前 `.trt-<版本>.gpu-<GPU>` 后缀的 `*.engine`。请先运行 `./deploy.sh --migrate`。

### 2. 🎬 交互式命令生成器 — `generate_cmd.sh`

8 步向导，源码环境与 Runtime 包使用同一菜单，生成 `run_encode.sh`，不自动执行：

1. 自动扫描并选择一个或多个输入视频（多个视频串行处理，共用后续参数）
2. 选择处理方案：内置 Slot 1001–1003、2001–2006、2025/2026、3025/3026，
   **自定义模型**（模型目录中的其他 ONNX），或 **仅自定义插帧**
3. 可在任意超分方案上 **叠加自定义 RIFE**：模型、2x / 5/2x / 4x、先插帧或先超分、转场阈值
4. 整片压制或截取片段测试
5. 输出路径（自动后缀 `_upscaled` / `_rife2x_upscaled`）
6. 编码器与画质（NVENC 自动选择 split：Ada 2 路、Blackwell 3 路；AV1 / HEVC / H.264 / x265 / FFV1）
7. 解码器与像素格式
8. 音轨 / 字幕、输出缩放、关闭静止帧复用、改用 TensorRT 对比（`native=off`）

自定义 RIFE、自定义模型和 `native=off` 会写入与脚本同目录的任务配置
`run_encode.conf`（Slot 9001，重新生成时覆盖）。4x 模型（Slot 2004、2006 及名称含
`4x` 的自定义模型）自动加入 `--pipeline-depth 2`；1080p 输入会输出 8K，可在输出设置中缩小。

```sh
./generate_cmd.sh              # 直接运行
./deploy.sh --gen              # 或从 deploy.sh 菜单进入
```

> [!TIP]
> 详细的手动部署步骤请参阅 [DEPLOY.md](DEPLOY.md)。

## Ubuntu 24.04 免安装 Runtime 包

Runtime 包适合直接部署到 AutoDL、云 GPU 容器或 Ubuntu 24.04 x86_64 主机。包内已经
包含全部 ONNX / RIFE 模型、FFmpeg、CUDA Runtime、TensorRT 运行库、`trtexec` 和对应架构的
TensorRT Builder Resource；宿主机只需安装兼容的 NVIDIA 驱动。

### 1. 选择并下载包

每次 Runtime Action 成功运行后，两个压缩包会一起发布到本仓库的
[Releases](../../releases) 页面，并标记为 prerelease：

| 文件名中的架构 | 适用 GPU |
|---|---|
| `sm89` | Ada Lovelace，RTX 40 系 |
| `sm120` | Blackwell，RTX 50 系（包括 RTX 5090） |

下载对应的 `.tar.zst` 及同名 `.sha256` 文件。不要在 RTX 50 系上使用 `sm89` 包。

### 2. 校验并解压

```sh
sha256sum -c animejanai-ubuntu24-x86_64-sm120-*.tar.zst.sha256
tar --zstd -xf animejanai-ubuntu24-x86_64-sm120-*.tar.zst
cd animejanai-ubuntu24-x86_64-sm120-*/
sha256sum -c SHA256SUMS
./runtime-info.sh
```

请把 Runtime 解压到可写目录：回退到 TensorRT 的模型会把 Engine 缓存在 `onnx/` 中。

### 3. 使用交互式生成器（推荐）

```sh
./generate_cmd.sh
bash ./run_encode.sh
```

### 4. 手动运行 `aji_encode`

包根目录的 `./aji_encode` 启动器会设置动态库路径，并补齐包内的配置文件、模型目录、
RIFE 模型目录和 `trtexec`：

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

上例适用于 RTX 5090/Blackwell；RTX 40 系改为 `-split_encode_mode 2`。

| Slot | 处理方案 |
|---:|---|
| `1001` | Quality 超分 |
| `1002` | Balanced 超分 |
| `1003` | Performance 超分（默认） |
| `2001` | Sharp Balanced 超分 |
| `2002` | Sharp Performance 超分 |
| `2003` | SD Compact 超分 |
| `2004` | RealESRGAN AnimeVideo-v3 4× 超分 |
| `2005` | APISR RRDB GAN 2× 超分 |
| `2006` | Real-ESRGAN x4plus Anime 6B 4× 超分 |
| `2025` / `2026` | 仅 RIFE v4.25 / v4.26 2x 插帧 |
| `3025` / `3026` | RIFE 2x 插帧 + Performance 超分 |

省略 `--slot` 时默认使用 `1003`，也可设置 `AJI_SLOT=3026`。自定义插帧可直接写一个
conf（`[global]` 复制自 `animejanai.conf`，再加一个 Slot）后用 `--conf my.conf --slot N`
运行，例如：

```ini
[slot_9001]
profile_name=Performance + RIFE v4.26 4x
chain_1_max_fps=61
chain_1_model_1_name=2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16
chain_1_rife=yes
chain_1_rife_model=426
chain_1_rife_factor_numerator=4
chain_1_rife_factor_denominator=1
chain_1_rife_before_upscale=yes
```

需要纯 TensorRT 结果做对比时，在 conf 的 `[global]` 中写 `native=off`。

包内的 FFmpeg、FFprobe 和 TensorRT 工具也应通过根目录启动器调用：

```sh
./ffprobe -hide_banner "/path/to/input.mkv"
./ffmpeg -hide_banner -encoders
./trtexec --version
```

### 5. 本地构建 Runtime 包

```sh
./scripts/package-ubuntu24-runtime-local.sh --gpu-arch 89
./scripts/package-ubuntu24-runtime-local.sh --gpu-arch 120
```

产物位于 `dist/`。详细说明见 [`packaging/ubuntu24/README.md`](packaging/ubuntu24/README.md)。

## 模型来源

模型清单（文件名、SHA256、下载地址）见 [`scripts/runtime-models.tsv`](scripts/runtime-models.tsv)，
`deploy.sh --models` 与 Runtime 打包共用。RIFE v4.25 / v4.26 取自上游发布的 fp16 模型包
（`models-rife-fp16-1`）。APISR 2× RRDB GAN 由 `tools/prepare_apisr.py` 从
[Xenova 的 FP16 ONNX 导出](https://huggingface.co/Xenova/2x_APISR_RRDB_GAN_generator-onnx)
固定版本生成（适配为 `input` / `output`、FP16、NCHW、batch=1）。Real-ESRGAN x4plus Anime 6B
使用 [deepghs 的 ONNX 导出](https://huggingface.co/deepghs/imgutils-models)（fp32，动态 H/W）。
各模型许可见 `packaging/ubuntu24/runtime/THIRD_PARTY_NOTICES.txt`。

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

## Still images (`include/aji_image.h`)

A separate C ABI for upscaling single images with any ONNX super-resolution
model — no conf, slots or YUV. 8-bit gray or RGB pixels go in from host
memory, the upscaled image comes back; tiling, overlap blending, padding and
dtype conversion happen inside.

```c
#include "aji_image.h"

aji_img_create_params cp = {0};
cp.api_version = AJI_IMAGE_API_VERSION;
cp.backend = AJI_IMG_DIRECTML;          /* or AJI_IMG_TENSORRT */
char err[256];
aji_img_ctx *c = aji_img_create(&cp, err, sizeof err);

aji_img_model_params mp = {0};
mp.onnx_path = "4x_model.onnx";
aji_img_model *m = aji_img_load(c, &mp);  /* NULL: aji_img_last_error(c) */

aji_img_model_info info;
aji_img_model_info_get(m, &info);         /* info.scale sizes the output */

aji_img_buffer in  = {w, h, 3, w * 3, pixels};
aji_img_buffer out = {w * info.scale, h * info.scale, 3,
                      w * info.scale * 3, out_pixels};
aji_img_upscale(m, &in, &out, NULL, NULL);

aji_img_release(&m);
aji_img_destroy(&c);
```

Load once and reuse: creating the context and loading a model are the slow
parts (DirectML compiles the model; TensorRT deserializes or builds an engine),
upscaling is per image. `src/aji_image_cli.cpp` (the `aji_image` tool, PNM in
and out) is the complete working example.

Things the header can't tell you:

- **Deployment.** Load `aji.dll` (`libaji.so`); it loads `aji_dml` / `aji_trt`
  from its own directory. Their runtimes must sit beside them: `onnxruntime.dll`
  + `DirectML.dll` for DirectML; `nvinfer_11`, `nvinfer_plugin_11`,
  `nvonnxparser_11`, `cudart64_13`, the builder resource for the GPU's SM and
  `trtexec` for TensorRT (`trtexec` can be passed explicitly instead).
- **First TensorRT load builds an engine** — minutes, once per model, tile
  size, GPU and TensorRT version; later loads hit the cache (`cache_dir`, or
  the model's directory). `aji_img_load` blocks through the build by default;
  interactive apps pass `AJI_IMG_LOAD_ASYNC_BUILD` and poll
  `aji_img_model_state` instead.
- **Models**: one NCHW input and one NCHW output, 1 or 3 channels, an integer
  scale, fp32/fp16 (TensorRT also bf16), dynamic or fixed spatial shape (a fixed
  shape becomes the tile). For DirectML, export at **opset 19 or lower** — the
  DirectML EP lacks newer ops such as opset-20 `Gelu`, which then fall back to
  the CPU and run tens of times slower.
- **Channels**: a gray image fed to a 3-channel model is replicated in and
  channel 0 is kept. Alpha is not handled: composite onto a background, or
  upscale the alpha yourself, before and after calling.
- **Threading**: a context works from any thread; calls on one model are
  serialized, different models may run concurrently.

## mpv usage (single engine, for testing)

```sh
mpv --hwdec=nvdec \
    --vf=animejanai=engine=models/<model>.engine:lib=build/libaji.so video.mkv
```
