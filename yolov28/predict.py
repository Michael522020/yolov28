"""Run a trained or randomly initialized YOLOv28 checkpoint on images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image, ImageDraw

from yolov28.data import IMAGE_SUFFIXES, image_to_tensor, letterbox
from yolov28.model import YOLOv28, load_checkpoint
from yolov28.ops import decode_detections
from yolov28.train import next_run_dir, select_device

_PALETTE = (
    (230, 57, 70),
    (42, 157, 143),
    (38, 70, 83),
    (244, 162, 97),
    (233, 196, 106),
    (88, 129, 87),
    (69, 123, 157),
    (181, 101, 118),
)


def scale_boxes_to_original(
    boxes: torch.Tensor, scale: float, left: int, top: int, width: int, height: int
) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    out = boxes.clone()
    out[:, [0, 2]] = (out[:, [0, 2]] - left) / scale
    out[:, [1, 3]] = (out[:, [1, 3]] - top) / scale
    out[:, 0].clamp_(0, width)
    out[:, 2].clamp_(0, width)
    out[:, 1].clamp_(0, height)
    out[:, 3].clamp_(0, height)
    return out


def list_images(source: str) -> List[Path]:
    path = Path(source)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise FileNotFoundError(f"no images in {path}")
        return files
    raise FileNotFoundError(f"source not found: {source}")


def draw_detections(image: Image.Image, detections: torch.Tensor, names: Dict[int, str]) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    width = max(1, image.size[0] // 200)
    for row in detections.tolist():
        x1, y1, x2, y2, score, cls = row
        cls_id = int(cls)
        color = _PALETTE[cls_id % len(_PALETTE)]
        label = names.get(cls_id, str(cls_id))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        draw.text((x1 + 2, max(0, y1 - 12)), f"{label} {score:.2f}", fill=color)
    return canvas


@torch.no_grad()
def predict_model(
    model: YOLOv28,
    names: Dict[int, str],
    source: str,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 300,
    end2end: bool = False,
    device: str = "",
    save_dir: str | None = None,
) -> List[dict]:
    if imgsz % 32 != 0:
        raise ValueError("imgsz must be a multiple of 32")
    torch_device = select_device(device)
    model.to(torch_device)
    training = model.training
    model.eval()
    output_dir = Path(save_dir) if save_dir else next_run_dir("runs/predict", "exp")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for path in list_images(source):
            with Image.open(path) as handle:
                image = handle.convert("RGB")
                image.load()
            width, height = image.size
            canvas, scale, left, top = letterbox(image, imgsz)
            tensor = image_to_tensor(canvas).unsqueeze(0).to(torch_device)
            detections = decode_detections(
                model, model(tensor), conf, iou, max_det, end2end=end2end
            )[0]
            detections = detections.cpu()
            detections[:, :4] = scale_boxes_to_original(
                detections[:, :4], scale, left, top, width, height
            )
            drawn = draw_detections(image, detections, names)
            saved = output_dir / f"{path.stem}.jpg"
            drawn.save(saved, quality=95)
            classes = detections[:, 5].to(torch.long) if detections.numel() else torch.zeros(0, dtype=torch.long)
            scores = detections[:, 4] if detections.numel() else torch.zeros(0)
            boxes = detections[:, :4] if detections.numel() else torch.zeros((0, 4))
            results.append(
                {
                    "path": str(path),
                    "saved": str(saved),
                    "boxes": boxes,
                    "scores": scores,
                    "classes": classes,
                }
            )
            print(f"{path} -> {saved}  detections={boxes.shape[0]}")
    finally:
        model.train(training)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with an unofficial YOLOv28 checkpoint.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="image file or directory")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--end2end", action="store_true", help="use the one-to-one head and skip NMS")
    parser.add_argument("--device", default="")
    parser.add_argument("--project", default="runs/predict")
    parser.add_argument("--name", default="exp")
    args = parser.parse_args()
    model, names, _ = load_checkpoint(args.weights, device="cpu")
    save_dir = next_run_dir(args.project, args.name)
    predict_model(
        model,
        names,
        args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        end2end=args.end2end,
        device=args.device,
        save_dir=str(save_dir),
    )


if __name__ == "__main__":
    main()
