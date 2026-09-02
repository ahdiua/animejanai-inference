# Ubuntu 24.04 runtime packaging

From Ubuntu 24.04, run the packager directly:

```bash
./scripts/package-ubuntu24-runtime.sh --install-deps --gpu-arch 89
```

From Arch, WSL, or another Linux distribution, use the local wrapper. It runs
the same packager in an Ubuntu 24.04 Docker/Podman container:

```bash
./scripts/package-ubuntu24-runtime-local.sh --gpu-arch 89
```

The process installs build-only packages, compiles a single-SM release,
downloads a BtbN FFmpeg shared build, stages the minimal full TensorRT runtime,
and creates `dist/*.tar.zst` plus its SHA-256 file. A GPU and GPU passthrough
are not required for packaging. In container mode, custom output/model paths
should remain inside the repository mount.

By default, models already present below `onnx/` are reused. Missing models are
downloaded. Override the source directory with `--models-dir`.

Supported `--gpu-arch` values:

| Value | Typical generation |
|---:|---|
| 75 | Turing / RTX 20 |
| 80 | Ampere data center |
| 86 | Ampere / RTX 30 |
| 89 | Ada / RTX 40 (default) |
| 90 | Hopper |
| 100 | Blackwell data center |
| 120 | Blackwell / RTX 50 |

Only the selected TensorRT builder resource is included. Build one archive per
target architecture rather than putting every multi-hundred-MB builder resource
in a universal archive.

Engines built on first use are fixed to the video's working resolution
(`minShapes=optShapes=maxShapes`) and use
`--builderOptimizationLevel=5`. A different resolution gets its own cached
engine.
