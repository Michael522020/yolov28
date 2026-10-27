"""Smoke tests for geometry, assignment, training, prediction, and ONNX export."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolov28 import YOLO
from yolov28.data import (
    DetectionDataset,
    flip_boxes,
    letterbox,
    xywhn_to_xyxy,
)
from yolov28.export import export_onnx
from yolov28.loss import DetectionLoss
from yolov28.metrics import average_precision
from yolov28.model import YOLOv28, load_checkpoint
from yolov28.ops import ciou, task_aligned_assign
from yolov28.train import TrainConfig, train_model


def test_geometry() -> None:
    labels = torch.tensor([[0.0, 0.5, 0.5, 0.4, 0.2]])
    boxes = xywhn_to_xyxy(labels, width=100, height=80, scale=0.64, left=0, top=6)
    expected = torch.tensor([[0.0, 19.2, 26.48, 44.8, 36.72]])
    if not torch.allclose(boxes, expected, atol=1e-4):
        raise AssertionError(f"label transform mismatch: {boxes}")
    flipped = flip_boxes(boxes, size=64)
    if not torch.allclose(flipped[0, 1], torch.tensor(64 - 44.8), atol=1e-4):
        raise AssertionError("horizontal flip did not mirror x1")
    image = Image.new("RGB", (100, 80), (10, 20, 30))
    canvas, scale, left, top = letterbox(image, 64)
    if canvas.size != (64, 64) or abs(scale - 0.64) > 1e-6 or (left, top) != (0, 6):
        raise AssertionError(f"letterbox mismatch: {canvas.size} {scale} {left} {top}")
    same = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    if not torch.allclose(ciou(same, same), torch.ones(1), atol=1e-4):
        raise AssertionError(f"identical boxes should have CIoU 1, got {ciou(same, same)}")
    perfect = average_precision(torch.tensor([1.0]), torch.tensor([0.9]), npos=1)
    if abs(perfect - 1.0) > 1e-5:
        raise AssertionError(f"perfect AP should be 1, got {perfect}")


def test_assignment_fallback() -> None:
    points = torch.tensor([[32.0, 32.0], [48.0, 48.0]])
    boxes = torch.tensor([[0.0, 0.0, 4.0, 4.0], [10.0, 10.0, 20.0, 20.0]])
    scores = torch.tensor([[0.4, 0.1], [0.2, 0.3]])
    gt_boxes = torch.tensor([[1.0, 1.0, 3.0, 3.0]])
    gt_labels = torch.tensor([0])
    fg, _, target_cls = task_aligned_assign(
        boxes, scores, points, gt_boxes, gt_labels, topk=1
    )
    if int(fg.sum()) != 1 or int(target_cls[int(fg.nonzero()[0]), 0]) != 1:
        raise AssertionError("tiny ground truth did not receive a nearest anchor")


def test_forward_and_backward() -> None:
    model = YOLOv28(nc=4, scale="n")
    if model.stride.tolist() != [8.0, 16.0, 32.0]:
        raise AssertionError(f"unexpected strides {model.stride.tolist()}")
    images = torch.rand(2, 3, 128, 128)
    preds = model(images)
    for branch in ("o2m", "o2e"):
        if len(preds[branch]) != 3:
            raise AssertionError("expected three detection scales")
        if preds[branch][0].shape[1] != 4 + 4:
            raise AssertionError("channel count should be 4 box values plus nc")
    targets = [
        torch.tensor([[0.0, 16.0, 16.0, 96.0, 96.0]]),
        torch.zeros((0, 5)),
    ]
    criterion = DetectionLoss(model)
    loss, parts = criterion(preds, targets)
    if not torch.isfinite(loss) or float(parts["box"]) <= 0:
        raise AssertionError(f"loss did not use the foreground box: {float(loss)} {parts}")
    loss.backward()
    grad = model.stem[0].conv.weight.grad
    if grad is None or not torch.isfinite(grad).all() or float(grad.abs().sum()) == 0:
        raise AssertionError("backbone did not receive a finite gradient")


def test_dataset_and_checkpoint(tmp: Path) -> None:
    root = tmp / "data"
    image_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    val_images = root / "images" / "val"
    val_labels = root / "labels" / "val"
    for directory in (image_dir, label_dir, val_images, val_labels):
        directory.mkdir(parents=True)
    Image.new("RGB", (100, 80), (255, 0, 0)).save(image_dir / "a.jpg")
    (label_dir / "a.txt").write_text("0 0.5 0.5 0.4 0.2\n", encoding="utf-8")
    Image.new("RGB", (100, 80), (0, 255, 0)).save(val_images / "b.jpg")
    (val_labels / "b.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    yaml_path = tmp / "data.yaml"
    yaml_path.write_text(
        f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: red\n  1: green\n",
        encoding="utf-8",
    )
    dataset = DetectionDataset(root, "images/train", imgsz=64, augment=False, nc=2)
    image, boxes = dataset[0]
    if image.shape != (3, 64, 64) or boxes.shape != (1, 5):
        raise AssertionError(f"dataset sample shape mismatch {image.shape} {boxes.shape}")
    expected = xywhn_to_xyxy(
        torch.tensor([[0.0, 0.5, 0.5, 0.4, 0.2]]), 100, 80, 0.64, 0, 6
    )
    if not torch.allclose(boxes, expected, atol=1e-3):
        raise AssertionError(f"dataset boxes {boxes} != {expected}")

    result = train_model(
        TrainConfig(
            data=str(yaml_path),
            scale="n",
            epochs=1,
            batch=1,
            imgsz=64,
            project=str(tmp / "runs"),
            name="real",
            workers=0,
        )
    )
    loaded, names, _ = load_checkpoint(str(result["last"]))
    if loaded.nc != 2 or names[0] != "red":
        raise AssertionError("checkpoint did not keep nc and class names")
    probe = Image.new("RGB", (80, 60), (20, 20, 20))
    probe_path = tmp / "probe.jpg"
    probe.save(probe_path)
    yolo = YOLO(str(result["last"]))
    predictions = yolo.predict(str(probe_path), imgsz=64, conf=0.01, save_dir=str(tmp / "predict"))
    if not Path(predictions[0]["saved"]).is_file():
        raise AssertionError("prediction image was not written")
    try:
        import onnx  # noqa: F401
    except ImportError:
        print("onnx is not installed; export file check skipped")
        return
    onnx_path = tmp / "yolov28n.onnx"
    export_onnx(str(result["last"]), str(onnx_path), imgsz=64)
    if not onnx_path.is_file() or onnx_path.stat().st_size < 1000:
        raise AssertionError("onnx export did not write a model file")


def test_overfit_one_batch() -> None:
    model = YOLOv28(nc=2, scale="n")
    images = torch.rand(2, 3, 128, 128)
    targets = [
        torch.tensor([[0.0, 20.0, 20.0, 100.0, 100.0], [1.0, 40.0, 50.0, 120.0, 110.0]]),
        torch.tensor([[1.0, 10.0, 30.0, 70.0, 90.0]]),
    ]
    criterion = DetectionLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
    losses = []
    model.train()
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    print("overfit losses:", " ".join(f"{value:.4f}" for value in losses))
    if not all(map(math_isfinite, losses)):
        raise AssertionError("overfit produced a non-finite loss")
    if losses[-1] >= losses[0]:
        raise AssertionError(f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}")


def math_isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def main() -> None:
    test_geometry()
    test_assignment_fallback()
    test_forward_and_backward()
    test_overfit_one_batch()
    with tempfile.TemporaryDirectory() as directory:
        test_dataset_and_checkpoint(Path(directory))
    print("smoke tests passed")


if __name__ == "__main__":
    main()
