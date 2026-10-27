"""Exponential moving average of a detector's weights."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn


class ModelEMA:
    """Track a smoothed copy of ``model`` for validation and saved checkpoints."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.updates = 0
        self.ema = copy.deepcopy(model).eval()
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = self.decay * (1.0 - math.exp(-self.updates / 2000.0))
        source_state = model.state_dict()
        with torch.no_grad():
            for name, tensor in self.ema.state_dict().items():
                source = source_state[name].detach()
                keep = (
                    "running_mean" in name
                    or "running_var" in name
                    or name.endswith("stride")
                    or not tensor.dtype.is_floating_point
                )
                if keep:
                    tensor.copy_(source)
                else:
                    tensor.mul_(decay).add_(source, alpha=1.0 - decay)
