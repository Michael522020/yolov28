"""YOLO-format detection data and a synthetic demo set."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def letterbox(
    image: Image.Image, size: int, color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[Image.Image, float, int, int]:
    """Resize ``image`` to fit in a square canvas. Returns canvas, scale, left, top."""
    width, height = image.size
    scale = min(size / width, size / height)
    resized_w = int(round(width * scale))
    resized_h = int(round(height * scale))
    resized = image.resize((resized_w, resized_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), color)
    left = (size - resized_w) // 2
    top = (size - resized_h) // 2
    canvas.paste(resized, (left, top))
    return canvas, scale, left, top


def xywhn_to_xyxy(
    labels: torch.Tensor,
    width: int,
    height: int,
    scale: float,
    left: int,
    top: int,
) -> torch.Tensor:
    """Convert ``(n, 5)`` cls/xc/yc/w/h normalized labels to cls/xyxy pixels."""
    if labels.numel() == 0:
        return labels.new_zeros((0, 5))
    cls = labels[:, 0:1]
    xc = labels[:, 1] * width
    yc = labels[:, 2] * height
    bw = labels[:, 3] * width
    bh = labels[:, 4] * height
    x1 = (xc - bw / 2) * scale + left
    y1 = (yc - bh / 2) * scale + top
    x2 = (xc + bw / 2) * scale + left
    y2 = (yc + bh / 2) * scale + top
    return torch.stack((cls.squeeze(1), x1, y1, x2, y2), dim=1)


def flip_boxes(boxes: torch.Tensor, size: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    flipped = boxes.clone()
    flipped[:, 1] = size - boxes[:, 3]
    flipped[:, 3] = size - boxes[:, 1]
    return flipped


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def clamp_boxes(boxes: torch.Tensor, size: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    xy = boxes[:, 1:5].clamp(0, size)
    keep = (xy[:, 2] - xy[:, 0] >= 1) & (xy[:, 3] - xy[:, 1] >= 1)
    out = boxes.clone()
    out[:, 1:5] = xy
    return out[keep]


def read_label_file(path: Path) -> torch.Tensor:
    if not path.is_file():
        return torch.zeros((0, 5), dtype=torch.float32)
    rows: List[List[float]] = []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return torch.zeros((0, 5), dtype=torch.float32)
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            values = [float(part) for part in parts[:5]]
        except ValueError:
            continue
        rows.append(values)
    if not rows:
        return torch.zeros((0, 5), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def load_data_config(path: str | Path) -> Tuple[Path, str, str | None, Dict[int, str], int]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    root = Path(cfg["path"])
    if not root.is_absolute():
        beside_yaml = (path.parent / root).resolve()
        beside_cwd = (Path.cwd() / root).resolve()
        root = beside_yaml if beside_yaml.exists() else beside_cwd
    names_raw = cfg["names"]
    if isinstance(names_raw, dict):
        names = {int(key): str(value) for key, value in names_raw.items()}
    elif isinstance(names_raw, list):
        names = {i: str(value) for i, value in enumerate(names_raw)}
    else:
        raise ValueError("names must be a list or a mapping of class index to name")
    nc = int(cfg.get("nc", len(names)))
    val = cfg.get("val")
    return root, str(cfg["train"]), None if val is None else str(val), names, nc


def label_path_for(root: Path, image_path: Path) -> Path:
    relative = image_path.relative_to(root)
    parts = list(relative.parts)
    if not parts or parts[0] != "images":
        raise ValueError(
            f"{image_path} is not under an images/ directory inside {root}. "
            "Expected layout: images/train/a.jpg and labels/train/a.txt."
        )
    parts[0] = "labels"
    return root.joinpath(*parts).with_suffix(".txt")


class DetectionDataset(Dataset):
    def __init__(self, root: Path, split: str, imgsz: int, augment: bool, nc: int):
        self.root = Path(root)
        self.imgsz = imgsz
        self.augment = augment
        self.nc = nc
        split_root = self.root / split
        if not split_root.is_dir():
            raise FileNotFoundError(
                f"missing image directory: {split_root}\n"
                "Expected:\n"
                "  <root>/images/train/*.jpg\n"
                "  <root>/labels/train/*.txt\n"
                "Label rows are: class xc yc w h, normalized to the original image."
            )
        self.files = sorted(
            path for path in split_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.files:
            raise FileNotFoundError(f"no images found in {split_root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        with Image.open(path) as handle:
            image = handle.convert("RGB")
            image.load()
        width, height = image.size
        labels = read_label_file(label_path_for(self.root, path))
        image, scale, left, top = letterbox(image, self.imgsz)
        boxes = xywhn_to_xyxy(labels, width, height, scale, left, top)
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            boxes = flip_boxes(boxes, self.imgsz)
        boxes = clamp_boxes(boxes, self.imgsz)
        if boxes.numel():
            keep = (boxes[:, 0] >= 0) & (boxes[:, 0] < self.nc)
            boxes = boxes[keep]
        return image_to_tensor(image), boxes


class SyntheticDetection(Dataset):
    """Colored rectangles on noise. Used by ``--demo`` and the smoke test."""

    def __init__(self, length: int = 16, imgsz: int = 320, nc: int = 4, boxes: int = 3, seed: int = 0):
        self.length = length
        self.imgsz = imgsz
        self.nc = nc
        self.boxes = boxes
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        generator = torch.Generator().manual_seed(self.seed + index)
        image = torch.rand(3, self.imgsz, self.imgsz, generator=generator) * 0.3
        rows = []
        for _ in range(self.boxes):
            cls = int(torch.randint(0, self.nc, (1,), generator=generator))
            bw = int(torch.randint(self.imgsz // 8, self.imgsz // 3, (1,), generator=generator))
            bh = int(torch.randint(self.imgsz // 8, self.imgsz // 3, (1,), generator=generator))
            x1 = int(torch.randint(0, self.imgsz - bw, (1,), generator=generator))
            y1 = int(torch.randint(0, self.imgsz - bh, (1,), generator=generator))
            color = torch.zeros(3, 1, 1)
            color[cls % 3] = 0.85
            if cls >= 3:
                color[(cls + 1) % 3] = 0.55
            image[:, y1 : y1 + bh, x1 : x1 + bw] = color
            rows.append([cls, x1, y1, x1 + bw, y1 + bh])
        target = torch.tensor(rows, dtype=torch.float32)
        return image, target


def collate(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets


def demo_names(nc: int) -> Dict[int, str]:
    palette = ["red", "green", "blue", "yellow", "cyan", "magenta", "white", "orange"]
    return {i: palette[i] if i < len(palette) else f"class{i}" for i in range(nc)}
