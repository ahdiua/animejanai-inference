# AnimeJaNai Ubuntu 24.04 Runtime

Version: `@PACKAGE_VERSION@`
Target: Ubuntu 24.04 x86_64, NVIDIA `@GPU_ARCH@`
Build stack: CUDA `@CUDA_VERSION@`, TensorRT `@TRT_VERSION@`

This is a self-contained **full TensorRT runtime**, not a Lean runtime. It
includes `trtexec`, the ONNX parser, TensorRT plugin library and the
`@GPU_ARCH@` builder resource so engines can be built on first use. CUDA
Toolkit, TensorRT, FFmpeg and Python do not need to be installed separately.
A compatible NVIDIA driver is still required on the host.

## Quick start

```bash
tar --zstd -xf animejanai-ubuntu24-x86_64-@GPU_ARCH@-*.tar.zst
cd animejanai-ubuntu24-x86_64-@GPU_ARCH@-*
sha256sum -c SHA256SUMS
./runtime-info.sh

./generate_cmd.sh
bash ./run_encode.sh
```

`generate_cmd.sh` detects the extracted Runtime environment and presents an
interactive menu for the input video, Slot, full/clip processing, output path,
encoder, quality, decoder and pixel format. Every packaged model runs on
libaji's native CUDA kernels, so no engine is needed; a model that falls back to
TensorRT gets one built on first use and cached in `onnx/`.

Besides the Slots, the generator offers a custom model (another ONNX file in
`onnx/`) and custom RIFE (model, 2x / 5/2x / 4x, order, scene threshold) on any
upscale Slot or alone; those, and the TensorRT comparison option (`native=off`),
are written to a task conf `run_encode.conf` beside `run_encode.sh`. It
requests two-way HEVC/AV1 split-frame encoding on Ada/RTX 40 and three-way
encoding on Blackwell/RTX 50, and adds `--pipeline-depth 2` for 4x models.

## Manual command

To bypass the interactive generator, invoke the root launcher directly:

```bash
./aji_encode \
  --input input.mkv \
  --output output.mkv \
  --slot 1003 \
  --decoder nvdec \
  --vcodec hevc_nvenc \
  --vquality "-cq 18 -preset p7 -tune hq" \
  --pix-fmt yuv420p10 \
  --overwrite
```

Run the launchers from the extracted package root. They configure the bundled
shared-library paths automatically; do not invoke the `.real` executables in
`bin/` directly. The package directory must be writable: engines for models
that fall back to TensorRT are cached beside the models in `onnx/`, and libaji
removes any other `*.engine` there.

The default slot is `1003` (Performance). Held-frame reuse is on by default
(`--no-reuse-held-frames` turns it off; RIFE Slots never reuse).

Useful slots:

| Slot | Profile |
|---:|---|
| 1001 | Quality |
| 1002 | Balanced |
| 1003 | Performance (default) |
| 2001 | Sharp Balanced |
| 2002 | Sharp Performance |
| 2003 | SD Compact |
| 2004 | RealESRGAN AnimeVideo-v3 (native 4x) |
| 2005 | APISR RRDB GAN 2x (no RIFE) |
| 2006 | Real-ESRGAN x4plus Anime 6B (native 4x) |
| 2025 | RIFE v4.25 2x only |
| 2026 | RIFE v4.26 2x only |
| 3025 | Performance + RIFE v4.25 2x |
| 3026 | Performance + RIFE v4.26 2x |

Select another profile with `--slot`, for example:

```bash
./aji_encode --input input.mkv --output output.mkv --slot 3026 \
  --decoder nvdec --vcodec hevc_nvenc --pix-fmt yuv420p10 --overwrite
```

For HEVC/AV1 split-frame encoding, use `-split_encode_mode 2` on Ada/RTX 40
and `-split_encode_mode 3` on Blackwell/RTX 50. For example:

```bash
./aji_encode --input input.mkv --output output.mkv --slot 1003 \
  --decoder nvdec --vcodec hevc_nvenc \
  --vquality "-cq 18 -preset p7 -tune hq -split_encode_mode 3" \
  --pix-fmt yuv420p10 --overwrite
```

Interpolation is configured in conf Slots only. For other RIFE settings, copy
`animejanai.conf`, add a Slot with `chain_1_rife=yes`, `chain_1_rife_model`
(`426` = v4.26), `chain_1_rife_factor_numerator/denominator` and
`chain_1_rife_before_upscale`, and pass `--conf my.conf --slot <N>`. For a
TensorRT comparison, add `native=off` to its `[global]` section. The root
launchers set all relative runtime paths automatically.
Use `./ffmpeg`, `./ffprobe`, `./trtexec`, `./aji_harness` and
`./aji_kernel_test` when invoking the bundled tools.

## Compatibility

This package contains only the `@GPU_ARCH@` TensorRT builder resource to keep
the archive small. It is intended for the matching GPU compute capability. Use
a package built for another `smXX` when targeting a different GPU generation.
TensorRT engines are generated locally and should not be copied between
different GPU/TensorRT environments.

Run `sha256sum -c SHA256SUMS` to verify all extracted files.

APISR is available in `generate_cmd.sh` or with `--slot 2005`; the generator
requires even input width and height for it.
