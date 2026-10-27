"""YOLOv28 network.

The building blocks below are an original layout: a gated cross-stage mixer,
a serial pyramid pool, a PAN neck, and a dual decoupled head. Box regression
is a positive left/top/right/bottom distance, not a distribution-focal bin head.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# depth multiple, width multiple
SCALES: Dict[str, Tuple[float, float]] = {
    "n": (0.33, 0.25),
    "s": (0.33, 0.50),
    "m": (0.67, 0.75),
    "l": (1.00, 1.00),
    "x": (1.33, 1.25),
}


def make_divisible(value: float, divisor: int = 8) -> int:
    return max(divisor, int(math.ceil(value / divisor) * divisor))


class Conv(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, g: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class MixUnit(nn.Module):
    """Residual mixer.

    A softmax blends depthwise 3x3 and 5x5 responses per channel. That local
    mix is compared with a pooled context path. The residual gate bias starts
    at -4, so the block is close to an identity residual at step 0.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.local3 = Conv(channels, channels, 3, g=channels)
        self.local5 = Conv(channels, channels, 5, g=channels)
        self.project = Conv(channels, channels, 1)
        self.context = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv(channels, channels, 1),
        )
        self.mix = nn.Conv2d(channels, channels * 2, 1, bias=True)
        self.gate = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        mix = self.mix(x).view(batch, 2, channels, height, width).softmax(dim=1)
        local = self.project(mix[:, 0] * self.local3(x) + mix[:, 1] * self.local5(x))
        delta = local - self.context(x)
        return x + torch.sigmoid(self.gate(x)) * delta


class GateCSP(nn.Module):
    """Cross-stage block. One split is mixed; the other is forwarded."""

    def __init__(self, c1: int, c2: int, n: int = 1):
        super().__init__()
        hidden = max(c2 // 2, 8)
        self.cv1 = Conv(c1, hidden * 2, 1)
        self.cv2 = Conv(hidden * 2, c2, 1)
        self.m = nn.Sequential(*(MixUnit(hidden) for _ in range(max(int(n), 1))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).chunk(2, dim=1)
        return self.cv2(torch.cat((self.m(a), b), dim=1))


class StripGate(nn.Module):
    """Spatial gate from horizontal and vertical averages.

    The bias starts at 4, so the gate is nearly open and the block starts
    close to an identity.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        horizontal = x.mean(dim=3, keepdim=True)
        vertical = x.mean(dim=2, keepdim=True)
        return x * torch.sigmoid(self.conv(horizontal + vertical))


class PyramidPool(nn.Module):
    """Three serial max-pools, concatenated with the unpooled features."""

    def __init__(self, channels: int, kernel: int = 5):
        super().__init__()
        hidden = max(channels // 2, 8)
        self.cv1 = Conv(channels, hidden, 1)
        self.cv2 = Conv(hidden * 4, channels, 1)
        self.pool = nn.MaxPool2d(kernel, stride=1, padding=kernel // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), dim=1))


class Detect(nn.Module):
    """Decoupled one-to-many and one-to-one heads.

    Each map is ``(batch, 4 + nc, h, w)``. The first four channels are raw
    left/top/right/bottom distances; the rest are class logits.
    """

    def __init__(self, nc: int, channels: Sequence[int]):
        super().__init__()
        self.nc = nc
        self.nl = len(channels)
        box = max(16, channels[0] // 4)
        cls_hid = max(channels[0], min(nc, 128))
        self.stem = nn.ModuleList(Conv(c, c, 1) for c in channels)
        self.reg = nn.ModuleList(self._branch(c, box, 4) for c in channels)
        self.cls = nn.ModuleList(self._branch(c, cls_hid, nc) for c in channels)
        self.reg_e2e = nn.ModuleList(self._branch(c, box, 4) for c in channels)
        self.cls_e2e = nn.ModuleList(self._branch(c, cls_hid, nc) for c in channels)
        self.register_buffer("stride", torch.zeros(self.nl))

    @staticmethod
    def _branch(c_in: int, c_hid: int, c_out: int) -> nn.Sequential:
        return nn.Sequential(
            Conv(c_in, c_hid, 3),
            Conv(c_hid, c_hid, 3),
            nn.Conv2d(c_hid, c_out, 1),
        )

    def bias_init(self) -> None:
        # softplus(bias) ~= 2 grid cells. Class prior ~= 0.01.
        box_bias = math.log(math.exp(2.0) - 1.0)
        cls_bias = math.log(0.01 / 0.99)
        for tower in (self.reg, self.reg_e2e):
            for conv in tower:
                last = conv[-1]
                nn.init.normal_(last.weight, std=0.01)
                nn.init.constant_(last.bias, box_bias)
        for tower in (self.cls, self.cls_e2e):
            for conv in tower:
                last = conv[-1]
                nn.init.normal_(last.weight, std=0.01)
                nn.init.constant_(last.bias, cls_bias)

    def forward(self, feats: List[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        many: List[torch.Tensor] = []
        one: List[torch.Tensor] = []
        for i, feat in enumerate(feats):
            x = self.stem[i](feat)
            many.append(torch.cat((self.reg[i](x), self.cls[i](x)), dim=1))
            one.append(torch.cat((self.reg_e2e[i](x), self.cls_e2e[i](x)), dim=1))
        return {"o2m": many, "o2e": one}


def _same_size_cat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape[-2:] != b.shape[-2:]:
        raise RuntimeError(
            f"feature map sizes differ: {tuple(a.shape)} vs {tuple(b.shape)}. "
            "Input height and width must both be multiples of 32."
        )
    return torch.cat((a, b), dim=1)


class YOLOv28(nn.Module):
    """Three-scale anchor-free detector with a dual assignment head."""

    def __init__(self, nc: int = 80, scale: str = "n"):
        super().__init__()
        if scale not in SCALES:
            known = ", ".join(SCALES)
            raise ValueError(f"scale must be one of: {known}")
        if nc < 1:
            raise ValueError("nc must be positive")
        depth, width = SCALES[scale]
        self.nc = nc
        self.scale = scale

        def c(channels: int) -> int:
            return make_divisible(channels * width)

        def repeats(count: int) -> int:
            return max(1, int(round(count * depth)))

        c0, c1, c2, c3, c4 = c(64), c(128), c(256), c(512), c(512)
        self.stem = nn.Sequential(
            Conv(3, c0, 3, 2),
            Conv(c0, c1, 3, 2),
            GateCSP(c1, c1, repeats(3)),
        )
        self.stage3 = nn.Sequential(Conv(c1, c2, 3, 2), GateCSP(c2, c2, repeats(6)))
        self.stage4 = nn.Sequential(Conv(c2, c3, 3, 2), GateCSP(c3, c3, repeats(6)))
        self.stage5 = nn.Sequential(
            Conv(c3, c4, 3, 2),
            GateCSP(c4, c4, repeats(3)),
            PyramidPool(c4),
        )
        self.lat5 = GateCSP(c4 + c3, c3, repeats(3))
        self.lat4 = GateCSP(c3 + c2, c2, repeats(3))
        self.down4 = Conv(c2, c2, 3, 2)
        self.pan4 = GateCSP(c2 + c3, c3, repeats(3))
        self.down5 = Conv(c3, c3, 3, 2)
        self.pan5 = GateCSP(c3 + c4, c4, repeats(3))
        self.p3_gate = StripGate(c2)
        self.p4_gate = StripGate(c3)
        self.p5_gate = StripGate(c4)
        self.detect = Detect(nc, (c2, c3, c4))
        self._init_weights()
        self.detect.bias_init()
        self._init_strides()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.modules():
            if isinstance(module, MixUnit):
                nn.init.zeros_(module.mix.weight)
                nn.init.zeros_(module.mix.bias)
                nn.init.zeros_(module.gate.weight)
                nn.init.constant_(module.gate.bias, -4.0)
            elif isinstance(module, StripGate):
                nn.init.zeros_(module.conv.weight)
                nn.init.constant_(module.conv.bias, 4.0)

    def _init_strides(self) -> None:
        training = self.training
        self.eval()
        with torch.no_grad():
            maps = self.forward(torch.zeros(1, 3, 256, 256))["o2m"]
        strides = [256 / float(feat.shape[-2]) for feat in maps]
        self.detect.stride.copy_(torch.tensor(strides, dtype=self.detect.stride.dtype))
        self.train(training)

    @property
    def stride(self) -> torch.Tensor:
        return self.detect.stride

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        if x.shape[-1] % 32 or x.shape[-2] % 32:
            raise ValueError(
                f"input size must be a multiple of 32, got {tuple(x.shape[-2:])}"
            )
        p3_in = self.stem(x)
        p3 = self.stage3(p3_in)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        n4 = self.lat5(_same_size_cat(F.interpolate(p5, scale_factor=2, mode="nearest"), p4))
        n3 = self.lat4(_same_size_cat(F.interpolate(n4, scale_factor=2, mode="nearest"), p3))
        d4 = self.pan4(_same_size_cat(self.down4(n3), n4))
        d5 = self.pan5(_same_size_cat(self.down5(d4), p5))
        return self.detect([self.p3_gate(n3), self.p4_gate(d4), self.p5_gate(d5)])

    def extra_repr(self) -> str:
        return f"nc={self.nc}, scale={self.scale}"


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_checkpoint(path: str, device: str | torch.device = "cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = YOLOv28(nc=int(ckpt["nc"]), scale=str(ckpt["scale"]))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    names = {int(k): str(v) for k, v in dict(ckpt.get("names", {})).items()}
    if not names:
        names = {i: str(i) for i in range(model.nc)}
    return model, names, ckpt
