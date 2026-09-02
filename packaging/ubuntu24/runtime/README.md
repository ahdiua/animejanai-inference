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
interactive menu for the input video, built-in Slot, full/clip processing,
output path, encoder, quality, decoder and pixel format. It does not require an
existing `.engine`: the generated `run_encode.sh` uses configuration mode, so
the first run builds the required fixed-shape engines and later runs reuse the
cache in `onnx/`.

The generator automatically requests two-way HEVC/AV1 split-frame encoding on
Ada/RTX 40 and three-way encoding on Blackwell/RTX 50. RIFE-only and combined
upscale + RIFE 2x profiles can be selected from the same Slot menu.

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
`bin/` directly. The package directory must be writable because generated
TensorRT engines are cached beside the models in `onnx/`.

The default slot is `1003` (Performance). The first run at a new resolution
builds and caches a fixed-shape TensorRT engine beside the ONNX model; later
runs reuse it. Upscale and RIFE engines use `minShapes=optShapes=maxShapes` and
TensorRT's maximum `builderOptimizationLevel=5`.

Useful slots:

| Slot | Profile |
|---:|---|
| 1001 | Quality |
| 1002 | Balanced |
| 1003 | Performance (default) |
| 2001 | Sharp Balanced |
| 2002 | Sharp Performance |
| 2003 | SD Compact |
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

Direct Engine mode remains available by passing `--engine`, `--max-width` and
`--max-height`. The root launchers set all relative runtime paths automatically.
Use `./ffmpeg`, `./ffprobe`, `./trtexec`, `./aji_harness` and
`./aji_kernel_test` when invoking the bundled tools.

## Compatibility

This package contains only the `@GPU_ARCH@` TensorRT builder resource to keep
the archive small. It is intended for the matching GPU compute capability. Use
a package built for another `smXX` when targeting a different GPU generation.
TensorRT engines are generated locally and should not be copied between
different GPU/TensorRT environments.

Run `sha256sum -c SHA256SUMS` to verify all extracted files.
