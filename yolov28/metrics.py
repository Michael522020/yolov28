"""Validation loss, mAP50, and mAP50-95."""

from __future__ import annotations

from typing import Dict, List

import torch

from yolov28.loss import DetectionLoss
from yolov28.model import YOLOv28
from yolov28.ops import box_iou, decode_detections


def average_precision(tp: torch.Tensor, scores: torch.Tensor, npos: int) -> float:
    """101-point interpolated average precision. ``tp`` is 1 for a true positive."""
    if npos <= 0:
        return 0.0
    if scores.numel() == 0:
        return 0.0
    order = scores.argsort(descending=True)
    hits = tp[order].to(dtype=torch.float32)
    false = 1.0 - hits
    tp_cum = hits.cumsum(0)
    fp_cum = false.cumsum(0)
    recall = tp_cum / npos
    precision = tp_cum / (tp_cum + fp_cum).clamp(min=1e-9)
    total = 0.0
    for threshold in torch.linspace(0, 1, 101):
        eligible = recall >= threshold
        total += float(precision[eligible].max()) if bool(eligible.any()) else 0.0
    return total / 101


class MapAccumulator:
    def __init__(self, nc: int):
        self.nc = nc
        self.scores: List[List[float]] = [[] for _ in range(nc)]
        self.tps: List[List[float]] = [[] for _ in range(nc)]
        self.npos = [0 for _ in range(nc)]

    def update(self, detections: torch.Tensor, gt: torch.Tensor, iou_thres: float = 0.5) -> None:
        if gt.numel():
            for label in gt[:, 0].long().tolist():
                if 0 <= label < self.nc:
                    self.npos[label] += 1
        if detections.numel() == 0:
            return
        used = set()
        order = detections[:, 4].argsort(descending=True)
        for det_index in order.tolist():
            cls = int(detections[det_index, 5].item())
            if cls < 0 or cls >= self.nc:
                continue
            score = float(detections[det_index, 4].item())
            best_iou = iou_thres
            best_gt = -1
            if gt.numel():
                same = [j for j in range(gt.shape[0]) if int(gt[j, 0].item()) == cls and j not in used]
                if same:
                    gt_boxes = gt[same, 1:5]
                    ious = box_iou(detections[det_index, :4][None], gt_boxes).squeeze(0)
                    value, local = ious.max(dim=0)
                    if float(value) >= best_iou:
                        best_iou = float(value)
                        best_gt = same[int(local)]
            is_tp = best_gt >= 0
            if is_tp:
                used.add(best_gt)
            self.scores[cls].append(score)
            self.tps[cls].append(1.0 if is_tp else 0.0)

    def compute(self) -> float:
        values = []
        for cls in range(self.nc):
            if self.npos[cls] == 0:
                continue
            values.append(
                average_precision(
                    torch.tensor(self.tps[cls], dtype=torch.float32),
                    torch.tensor(self.scores[cls], dtype=torch.float32),
                    self.npos[cls],
                )
            )
        if not values:
            return 0.0
        return float(sum(values) / len(values))


class PredictionStore:
    """Keep detections and ground truth, then score them at one or more IoU thresholds."""

    def __init__(self, nc: int):
        self.nc = nc
        self.rows: List[tuple] = []

    def update(self, detections: torch.Tensor, gt: torch.Tensor) -> None:
        self.rows.append((detections.detach().cpu(), gt.detach().cpu()))

    def compute(self, iou_thres: float = 0.5) -> float:
        accumulator = MapAccumulator(self.nc)
        for detections, gt in self.rows:
            accumulator.update(detections, gt, iou_thres)
        return accumulator.compute()

    def compute_range(self) -> float:
        thresholds = [round(0.50 + 0.05 * index, 2) for index in range(10)]
        scores = [self.compute(threshold) for threshold in thresholds]
        return float(sum(scores) / len(scores))


@torch.no_grad()
def evaluate(
    model: YOLOv28,
    loader,
    criterion: DetectionLoss,
    device: torch.device,
    conf: float = 0.001,
    iou_thres: float = 0.45,
    match_iou: float = 0.5,
    max_det: int = 300,
) -> Dict[str, float]:
    training = model.training
    model.eval()
    loss_sum = 0.0
    seen = 0
    many = PredictionStore(model.nc)
    one = PredictionStore(model.nc)
    for images, targets in loader:
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        preds = model(images)
        loss, _ = criterion(preds, targets)
        batch = images.shape[0]
        loss_sum += float(loss.detach()) * batch
        seen += batch
        det_many = decode_detections(model, preds, conf, iou_thres, max_det, end2end=False)
        det_one = decode_detections(model, preds, conf, iou_thres, max_det, end2end=True)
        for index in range(batch):
            gt = targets[index]
            many.update(det_many[index], gt)
            one.update(det_one[index], gt)
    model.train(training)
    return {
        "loss": loss_sum / max(seen, 1),
        "map50": many.compute(match_iou),
        "map50_95": many.compute_range(),
        "map50_e2e": one.compute(match_iou),
    }
