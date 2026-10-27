"""Detection loss for the one-to-many and one-to-one heads."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from yolov28.model import YOLOv28
from yolov28.ops import ciou, decode_predictions, task_aligned_assign


class DetectionLoss:
    """Plain callable so the detector is not re-registered as a submodule."""

    def __init__(
        self,
        model: YOLOv28,
        box_gain: float = 2.0,
        cls_gain: float = 0.5,
        o2e_gain: float = 1.0,
        topk: int = 13,
    ):
        self.model = model
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.o2e_gain = o2e_gain
        self.topk = topk

    def __call__(
        self, preds: Dict[str, List[torch.Tensor]], targets: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.forward(preds, targets)

    def forward(
        self, preds: Dict[str, List[torch.Tensor]], targets: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = preds["o2m"][0].device
        strides = self.model.stride.to(device)
        many, many_parts = self._branch(preds["o2m"], strides, targets, topk=self.topk)
        one, one_parts = self._branch(preds["o2e"], strides, targets, topk=1)
        total = many + self.o2e_gain * one
        parts = {
            "box": many_parts["box"] + one_parts["box"],
            "cls": many_parts["cls"] + one_parts["cls"],
            "o2m": many.detach(),
            "o2e": one.detach(),
        }
        return total, parts

    def _branch(self, maps, strides, targets: List[torch.Tensor], topk: int):
        boxes, logits, points = decode_predictions(maps, strides)
        batch = boxes.shape[0]
        cls_acc = boxes.new_zeros(())
        box_acc = boxes.new_zeros(())
        for index in range(batch):
            gt = targets[index]
            if gt.numel() == 0:
                gt_boxes = boxes.new_zeros((0, 4))
                gt_labels = torch.zeros(0, dtype=torch.long, device=boxes.device)
            else:
                gt_labels = gt[:, 0].long().clamp(min=0, max=self.model.nc - 1)
                gt_boxes = gt[:, 1:5]
            fg, target_box, target_cls = task_aligned_assign(
                boxes[index].detach(),
                logits[index].detach().sigmoid(),
                points,
                gt_boxes,
                gt_labels,
                topk=topk,
            )
            bce = F.binary_cross_entropy_with_logits(logits[index], target_cls, reduction="none")
            if bool(fg.any()):
                cls_acc = cls_acc + bce[fg].sum(dim=1).mean()
                box_acc = box_acc + (1.0 - ciou(boxes[index][fg], target_box[fg])).mean()
            else:
                cls_acc = cls_acc + logits[index].sum() * 0
                box_acc = box_acc + boxes[index].sum() * 0
            background = ~fg
            if bool(background.any()):
                cls_acc = cls_acc + bce[background].sum(dim=1).mean()
            else:
                cls_acc = cls_acc + logits[index].sum() * 0
        cls_loss = cls_acc / batch
        box_loss = box_acc / batch
        total = self.box_gain * box_loss + self.cls_gain * cls_loss
        return total, {"box": box_loss.detach(), "cls": cls_loss.detach()}
