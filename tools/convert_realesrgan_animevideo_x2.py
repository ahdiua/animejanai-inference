#!/usr/bin/env python3
"""Export the official RealESRGANv2 AnimeVideo XS x2 checkpoint to ONNX.

The output uses dynamic NCHW spatial dimensions and float16 model I/O so it can
be consumed directly by AnimeJaNai-Inference's TensorRT/DirectML pipelines.
"""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto
from onnxconverter_common import float16
from torch import nn
from torch.nn import functional as F


class SRVGGNetCompact(nn.Module):
    """Official XS Real-ESRGAN generator architecture."""

    def __init__(self) -> None:
        super().__init__()
        features = 64
        scale = 2
        self.scale = scale
        self.body = nn.ModuleList(
            [nn.Conv2d(3, features, 3, 1, 1), nn.PReLU(features)]
        )
        for _ in range(16):
            self.body.extend(
                [nn.Conv2d(features, features, 3, 1, 1), nn.PReLU(features)]
            )
        self.body.append(nn.Conv2d(features, 3 * scale * scale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(scale)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        output = tensor
        for layer in self.body:
            output = layer(output)
        output = self.upsampler(output)
        return output + F.interpolate(
            tensor, scale_factor=self.scale, mode="nearest"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Official .pth checkpoint")
    parser.add_argument("output", type=Path, help="Destination FP16 .onnx file")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    if "params_ema" in checkpoint:
        state_dict = checkpoint["params_ema"]
    elif "params" in checkpoint:
        state_dict = checkpoint["params"]
    else:
        state_dict = checkpoint

    model = SRVGGNetCompact().eval()
    model.load_state_dict(state_dict, strict=True)
    sample = torch.rand(1, 3, 64, 64, dtype=torch.float32)

    with tempfile.TemporaryDirectory(prefix="realesrgan-onnx-") as temp_dir:
        fp32_path = Path(temp_dir) / "model-fp32.onnx"
        torch.onnx.export(
            model,
            (sample,),
            fp32_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            dynamo=False,
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "output": {
                    0: "batch",
                    2: "height_x2",
                    3: "width_x2",
                },
            },
        )

        fp32_model = onnx.load(fp32_path)
        onnx.checker.check_model(fp32_model)

        with torch.inference_mode():
            expected = model(sample).numpy()
        session = ort.InferenceSession(
            str(fp32_path), providers=["CPUExecutionProvider"]
        )
        actual = session.run(["output"], {"input": sample.numpy()})[0]
        if actual.shape != (1, 3, 128, 128):
            raise RuntimeError(f"unexpected output shape: {actual.shape}")
        np.testing.assert_allclose(expected, actual, rtol=1e-4, atol=1e-5)

        fp16_model = float16.convert_float_to_float16(
            fp32_model, keep_io_types=False
        )
        onnx.checker.check_model(fp16_model)
        input_type = fp16_model.graph.input[0].type.tensor_type.elem_type
        output_type = fp16_model.graph.output[0].type.tensor_type.elem_type
        if input_type != TensorProto.FLOAT16 or output_type != TensorProto.FLOAT16:
            raise RuntimeError(
                f"expected FP16 I/O, got input={input_type}, output={output_type}"
            )
        onnx.save(fp16_model, args.output)

    print(f"Created: {args.output}")
    print(f"Size:    {args.output.stat().st_size} bytes")
    print(f"SHA256:  {sha256(args.output)}")


if __name__ == "__main__":
    main()
