"""Unofficial independent anchor-free detector.

This project is not affiliated with Ultralytics and is not an official YOLO release.
"""

from yolov28.api import YOLO
from yolov28.model import SCALES, YOLOv28

__version__ = "0.1.0"
__all__ = ["YOLO", "YOLOv28", "SCALES", "__version__"]
