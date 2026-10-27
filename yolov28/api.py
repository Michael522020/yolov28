"""Small Python entry points shared by the CLI and library users."""

from __future__ import annotations

from typing import Dict

from yolov28.model import YOLOv28, load_checkpoint
from yolov28.predict import predict_model
from yolov28.train import TrainConfig, train_model


class YOLO:
    """Thin wrapper around :class:`yolov28.model.YOLOv28`.

    ``model`` is a scale letter (``n``, ``s``, ``m``, ``l``, ``x``) or a path
    to a checkpoint produced by this repository.
    """

    def __init__(self, model: str = "n", nc: int = 80):
        self.names: Dict[int, str]
        if str(model).endswith(".pt"):
            self.model, self.names, _ = load_checkpoint(model)
            self.scale = self.model.scale
            self.nc = self.model.nc
        else:
            self.scale = str(model)
            self.nc = nc
            self.names = {i: str(i) for i in range(nc)}
            self.model = YOLOv28(nc=nc, scale=self.scale)

    def train(self, **kwargs) -> dict:
        cfg = TrainConfig(scale=self.scale, nc=self.nc, **kwargs)
        result = train_model(cfg, model=self.model)
        self.names = result["names"]
        self.model = result["model"]
        return result

    def predict(self, source: str, **kwargs) -> list:
        return predict_model(self.model, self.names, source, **kwargs)
