"""Export the decoded detector to ONNX.

The graph returns one tensor of shape (batch, anchors, 4 + nc): xyxy in the
network input frame, followed by sigmoid class scores. NMS stays outside the
graph. Batch size is fixed to 1.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch
import torch.nn as nn

from yolov28.model import load_checkpoint
from yolov28.ops import decode_predictions


class _OnnxDetector(nn.Module):
    def __init__(self, model: nn.Module, end2end: bool):
        super().__init__()
        self.model = model
        self.end2end = end2end

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        preds = self.model(images)
        branch = "o2e" if self.end2end else "o2m"
        boxes, logits, _ = decode_predictions(preds[branch], self.model.stride)
        return torch.cat((boxes, logits.sigmoid()), dim=-1)


def export_onnx(weights: str, output: str, imgsz: int = 640, end2end: bool = False, opset: int = 17) -> Path:
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("ONNX export needs the onnx package. Install it with: pip install onnx") from exc
    if imgsz % 32 != 0:
        raise ValueError("imgsz must be a multiple of 32")
    model, _, _ = load_checkpoint(weights, device="cpu")
    model.eval()
    wrapper = _OnnxDetector(model, end2end=end2end).eval()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros(1, 3, imgsz, imgsz)
    kwargs = {
        "opset_version": opset,
        "input_names": ["images"],
        "output_names": ["detections"],
        "dynamo": False,
    }
    supported = inspect.signature(torch.onnx.export).parameters
    kwargs = {key: value for key, value in kwargs.items() if key in supported}
    torch.onnx.export(wrapper, sample, str(destination), **kwargs)
    print(f"wrote {destination} ({destination.stat().st_size / 1e6:.2f} MB)")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Export unofficial YOLOv28 weights to ONNX.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default="yolov28.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--end2end", action="store_true")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    export_onnx(args.weights, args.output, imgsz=args.imgsz, end2end=args.end2end, opset=args.opset)


if __name__ == "__main__":
    main()
