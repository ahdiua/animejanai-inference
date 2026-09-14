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

Archive downloads retry up to five times with 5/10/20/40-second backoff.
GitHub release retries add a fresh query parameter to avoid reusing cached
gateway errors or expired redirects. Logs identify the failing URL, and only
successful, nonempty downloads replace the destination file.

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

The `Package Ubuntu 24.04 runtime` GitHub Actions workflow builds the `sm89`
(Ada / RTX 40) and `sm120` (Blackwell / RTX 50) archives in parallel. After
both packages pass checksum verification, every successful workflow run
publishes them together as a GitHub release. Branch runs publish prereleases;
`v*` tag pushes publish stable releases. Workflow dispatches may supply a custom
prerelease tag; otherwise the workflow generates a unique tag from the run
number, attempt, and commit.

After a prerelease publishes successfully, only the newest published prerelease
is kept. Older prereleases, their attached archives, and their Git tags are
deleted; stable releases and drafts are preserved. The `Prune old prereleases`
workflow also supports manual cleanup and applies the policy when its workflow
or script changes on `main`. Cleanup and publishing are serialized. For a local
preview, set `GITHUB_REPOSITORY=ahdiua/animejanai-inference` and run
`python3 scripts/prune-prereleases.py`; add `--apply` with `GH_TOKEN` configured
to perform the deletions.

Engines built on first use are fixed to the video's working resolution
(`minShapes=optShapes=maxShapes`) and use
`--builderOptimizationLevel=5`. A different resolution gets its own cached
engine.

APISR 2x RRDB GAN is included as Slot 2005. Packaging first looks for
`2x_APISR_RRDB_GAN_fp16.onnx` in `--models-dir`, then in the repository
`models/` directory. Otherwise it uses `tools/prepare_apisr.py` to download,
verify and adapt the pinned upstream export in an isolated Python environment.
