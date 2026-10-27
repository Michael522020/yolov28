"""Box geometry, anchor assignment, and detection post-processing."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torchvision

from yolov28.model import YOLOv28


def box_iou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Pairwise IoU. ``a`` is (N, 4), ``b`` is (M, 4), both xyxy."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + eps)


def ciou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Complete IoU for paired boxes. Both arguments are (N, 4) xyxy."""
    b1_x1, b1_y1, b1_x2, b1_y2 = a.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = b.unbind(-1)
    w1 = (b1_x2 - b1_x1).clamp(min=0)
    h1 = (b1_y2 - b1_y1).clamp(min=0)
    w2 = (b2_x2 - b2_x1).clamp(min=0)
    h2 = (b2_y2 - b2_y1).clamp(min=0)
    inter_w = (torch.minimum(b1_x2, b2_x2) - torch.maximum(b1_x1, b2_x1)).clamp(min=0)
    inter_h = (torch.minimum(b1_y2, b2_y2) - torch.maximum(b1_y1, b2_y1)).clamp(min=0)
    inter = inter_w * inter_h
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    cw = torch.maximum(b1_x2, b2_x2) - torch.minimum(b1_x1, b2_x1)
    ch = torch.maximum(b1_y2, b2_y2) - torch.minimum(b1_y1, b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps
    rho2 = ((b1_x1 + b1_x2 - b2_x1 - b2_x2).pow(2) + (b1_y1 + b1_y2 - b2_y1 - b2_y2).pow(2)) / 4
    v = (4 / math.pi**2) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - rho2 / c2 - alpha * v


def decode_predictions(
    maps: Sequence[torch.Tensor], strides: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode raw head maps into xyxy boxes, class logits, and anchor points.

    Returns
    -------
    boxes:
        (batch, anchors, 4) xyxy in input-pixel coordinates.
    logits:
        (batch, anchors, nc)
    points:
        (anchors, 2) grid-center coordinates shared by the batch.
    """
    boxes, logits, points = [], [], []
    for feat, stride in zip(maps, strides):
        batch, _, height, width = feat.shape
        feat = feat.permute(0, 2, 3, 1).contiguous()
        step = stride.to(device=feat.device, dtype=feat.dtype)
        ltrb = F_softplus(feat[..., :4])
        gy, gx = torch.meshgrid(
            torch.arange(height, device=feat.device, dtype=feat.dtype),
            torch.arange(width, device=feat.device, dtype=feat.dtype),
            indexing="ij",
        )
        cx = (gx + 0.5) * step
        cy = (gy + 0.5) * step
        decoded = torch.stack(
            (
                cx - ltrb[..., 0] * step,
                cy - ltrb[..., 1] * step,
                cx + ltrb[..., 2] * step,
                cy + ltrb[..., 3] * step,
            ),
            dim=-1,
        )
        boxes.append(decoded.reshape(batch, -1, 4))
        logits.append(feat[..., 4:].reshape(batch, -1, feat.shape[-1] - 4))
        points.append(torch.stack((cx, cy), dim=-1).reshape(-1, 2))
    return torch.cat(boxes, dim=1), torch.cat(logits, dim=1), torch.cat(points, dim=0)


def F_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(x)


def task_aligned_assign(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    points: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    topk: int = 13,
    alpha: float = 1.0,
    beta: float = 6.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pick positive anchors for one image.

    Alignment score is ``class_score ** alpha * IoU ** beta``. Anchors whose
    center lies inside a ground-truth box are eligible. Each ground-truth box
    keeps ``topk`` anchors. If a box contains no anchor center, the nearest
    anchor is used so a tiny object still receives a gradient.

    Returns a foreground mask (N,), target boxes (N, 4), and hard class
    targets (N, nc).
    """
    anchors = boxes.shape[0]
    classes = scores.shape[-1]
    device = boxes.device
    target_box = torch.zeros(anchors, 4, device=device, dtype=boxes.dtype)
    target_cls = torch.zeros(anchors, classes, device=device, dtype=boxes.dtype)
    fg = torch.zeros(anchors, dtype=torch.bool, device=device)
    gt_count = gt_boxes.shape[0]
    if gt_count == 0 or anchors == 0:
        return fg, target_box, target_cls

    ious = box_iou(boxes, gt_boxes)
    class_score = scores[:, gt_labels].clamp(0, 1)
    px = points[:, 0:1]
    py = points[:, 1:2]
    x1, y1, x2, y2 = gt_boxes.unbind(-1)
    inside = (px > x1) & (py > y1) & (px < x2) & (py < y2)
    missing = inside.sum(dim=0) == 0
    if bool(missing.any()):
        centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) * 0.5
        distance = (points[:, None, :] - centers[None, :, :]).pow(2).sum(dim=-1)
        nearest = distance.argmin(dim=0)
        for index in torch.where(missing)[0].tolist():
            inside[nearest[index], index] = True

    alignment = class_score.pow(alpha) * ious.clamp(min=0).pow(beta)
    alignment = torch.where(inside, alignment.clamp(min=1e-6), torch.zeros_like(alignment))
    k = min(int(topk), anchors)
    _, chosen = alignment.topk(k, dim=0)

    best_iou = torch.zeros(anchors, device=device, dtype=boxes.dtype)
    best_gt = torch.full((anchors,), -1, device=device, dtype=torch.long)
    for gt_index in range(gt_count):
        anchor_index = chosen[:, gt_index]
        selected_iou = ious[anchor_index, gt_index]
        better = selected_iou > best_iou[anchor_index]
        winners = anchor_index[better]
        best_iou[winners] = selected_iou[better]
        best_gt[winners] = gt_index

    fg = best_gt >= 0
    if bool(fg.any()):
        matched = best_gt[fg]
        target_box[fg] = gt_boxes[matched]
        target_cls[fg, gt_labels[matched]] = 1.0
    return fg, target_box, target_cls


def non_max_suppression(
    boxes: torch.Tensor,
    logits: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
    use_nms: bool = True,
) -> List[torch.Tensor]:
    """Turn a decoded batch into a list of ``(M, 6)`` xyxy/score/class tensors."""
    scores = logits.sigmoid()
    outputs: List[torch.Tensor] = []
    for image_boxes, image_scores in zip(boxes, scores):
        class_score, class_id = image_scores.max(dim=1)
        keep = class_score > conf_thres
        image_boxes = image_boxes[keep]
        class_score = class_score[keep]
        class_id = class_id[keep]
        if image_boxes.numel() == 0:
            outputs.append(image_boxes.new_zeros((0, 6)))
            continue
        if use_nms:
            order = torchvision.ops.batched_nms(image_boxes, class_score, class_id, iou_thres)
        else:
            order = class_score.argsort(descending=True)
        order = order[:max_det]
        outputs.append(
            torch.cat(
                (
                    image_boxes[order],
                    class_score[order, None],
                    class_id[order, None].to(image_boxes.dtype),
                ),
                dim=1,
            )
        )
    return outputs


def decode_detections(
    model: YOLOv28,
    preds: dict,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    end2end: bool,
) -> List[torch.Tensor]:
    branch = "o2e" if end2end else "o2m"
    boxes, logits, _ = decode_predictions(preds[branch], model.stride)
    return non_max_suppression(
        boxes,
        logits,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
        use_nms=not end2end,
    )
