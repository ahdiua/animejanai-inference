#!/usr/bin/env python3
"""Check that configuring a model preserves unrelated engines (requires CUDA).

Usage: python3 sanity/check_engine_preservation.py [build/aji_harness]
The requested build deliberately fails before parsing ONNX; no real models or
user caches are touched. Run with GPU access and the built TensorRT backend.
"""
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    harness = Path(sys.argv[1] if len(sys.argv) > 1 else
                   "build/aji_harness").resolve()
    with tempfile.TemporaryDirectory(prefix="aji-cache-preservation-") as tmp:
        root = Path(tmp)
        originals = {
            "performance_1080p.engine": b"deploy performance engine",
            "balanced_1080p.engine": b"deploy balanced engine",
            "performance_sharp1_1080p.engine": b"deploy sharp performance engine",
            "balanced_sharp1_1080p.engine": b"deploy sharp balanced engine",
            "aji-12345678.123.trt-10.0.0.gpu-other-sm8.engine": b"other GPU cache",
        }
        for name, data in originals.items():
            (root / name).write_bytes(data)
        (root / "probe.onnx").write_bytes(b"not parsed: builder is /bin/false")
        conf = root / "test.conf"
        conf.write_text("[global]\nconfig_version=2\nbackend=TensorRT\n"
                        "[slot_1]\nchain_1_model_1_name=probe\n")
        frame = root / "input.nv12"
        frame.write_bytes(bytes(32 * 32 * 3 // 2))
        result = subprocess.run([
            str(harness), "--conf", str(conf), "--slot", "1",
            "--model-dir", str(root), "--trtexec", "/bin/false",
            "--input", str(frame), "--width", "32", "--height", "32",
            "--frames", "1",
        ], capture_output=True, text=True, timeout=60)
        log = result.stdout + result.stderr
        if result.returncode == 0 or "trtexec failed building engine" not in log:
            raise RuntimeError(f"did not reach the intended build failure:\n{log}")
        path_tool = harness.with_name("aji_engine_path")
        cache_info = subprocess.check_output([
            str(path_tool), str(root / "probe.onnx"), str(root), "32", "32", "input",
        ], text=True, timeout=30).splitlines()
        if not Path(cache_info[0] + ".build.log").is_file():
            raise AssertionError("deploy path helper disagrees with backend cache path")
        for name, data in originals.items():
            if not (root / name).exists() or (root / name).read_bytes() != data:
                raise AssertionError(f"unrelated engine was removed or changed: {name}")
        print("PASS: deploy/backend cache paths match; all five unrelated engines preserved")


if __name__ == "__main__":
    main()
