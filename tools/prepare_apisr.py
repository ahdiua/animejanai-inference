#!/usr/bin/env python3
"""Prepare the pinned APISR 2x RRDB FP16 ONNX for aji (requires onnx).

Downloads Xenova's export, or accepts the same file through --source.
Only the model interface is changed; internal weights/operators are preserved.
"""
import argparse
import hashlib
from pathlib import Path
import tempfile
import urllib.request

import onnx
from onnx import TensorProto


REVISION = "6361f81701564a71fe9aed63b1f1a150e0340e8f"
URL = ("https://huggingface.co/Xenova/2x_APISR_RRDB_GAN_generator-onnx/"
       f"resolve/{REVISION}/onnx/model_fp16.onnx")
SHA256 = "658eb8808a80e4904f1e3c6845b1917cf1e88b6a0594e2013b583741a6f8aa79"
MODEL_NAME = "2x_APISR_RRDB_GAN_fp16"


def prepare(source: Path, destination: Path):
    if hashlib.sha256(source.read_bytes()).hexdigest() != SHA256:
        raise ValueError("source SHA256 mismatch: expected the pinned FP16 export")
    model = onnx.load(source)
    graph = model.graph
    # Remove the exporter's FP32 interface casts. aji supplies planar FP16 RGB.
    aliases = {"pixel_values": "input", "graph_input_cast_0": "input",
               "graph_output_cast_0": "output", "reconstruction": "output"}
    nodes = [n for n in graph.node
             if n.name not in {"graph_input_cast0", "graph_output_cast0"}]
    del graph.node[:]
    graph.node.extend(nodes)
    for node in graph.node:
        for names in (node.input, node.output):
            for i, name in enumerate(names):
                names[i] = aliases.get(name, name)
    for value, name, height, width in (
            (graph.input[0], "input", "height", "width"),
            (graph.output[0], "output", "output_height", "output_width")):
        value.name = name
        tensor = value.type.tensor_type
        tensor.elem_type = TensorProto.FLOAT16
        tensor.shape.dim[0].dim_value = 1
        tensor.shape.dim[2].dim_param = height
        tensor.shape.dim[3].dim_param = width
    # Discard stale type/shape annotations from the original interface.
    del graph.value_info[:]
    onnx.helper.set_model_props(model, {
        "aji.source": URL,
        "aji.source_sha256": SHA256,
        "aji.input": "FP16 RGB NCHW [0,1]; batch=1; even height and width",
        "aji.scale": "2",
    })
    onnx.checker.check_model(model, full_check=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete, checked model.
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".onnx",
                                     delete=False) as temp:
        temporary = Path(temp.name)
    try:
        onnx.save(model, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Prepared {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="local pinned model_fp16.onnx")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "models" /
                        f"{MODEL_NAME}.onnx")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}; choose another --output")
    if args.source:
        prepare(args.source, args.output)
    else:
        with tempfile.TemporaryDirectory(prefix="aji-apisr-") as directory:
            source = Path(directory) / "model_fp16.onnx"
            print(f"Downloading {URL}")
            with urllib.request.urlopen(URL, timeout=60) as response:
                source.write_bytes(response.read())
            prepare(source, args.output)


if __name__ == "__main__":
    main()
