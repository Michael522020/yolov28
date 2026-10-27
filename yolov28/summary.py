"""Print parameter counts for each YOLOv28 scale."""

from __future__ import annotations

from yolov28.model import SCALES, YOLOv28, count_parameters


def main() -> None:
    print(f"{'scale':<8}{'params':>12}{'strides':>20}")
    for scale in SCALES:
        model = YOLOv28(nc=80, scale=scale)
        strides = ",".join(str(int(value)) for value in model.stride.tolist())
        print(f"{scale:<8}{count_parameters(model):>12,}{strides:>20}")


if __name__ == "__main__":
    main()
