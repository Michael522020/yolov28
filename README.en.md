<div align="center" dir="ltr">

[简体中文](README.md) · [繁體中文](README.zh-TW.md) · [English](README.en.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [العربية](README.ar.md)

</div>

# YOLOv28

The same guide is available in seven languages. The links are at the top of the page.

This repository is an independent anchor-free object detector: model, loss, training, prediction, and ONNX export. Weights start from random initialization. The docs were checked in October 2026. At that time the latest family published by Ultralytics was [YOLO26](https://docs.ultralytics.com/models/yolo26) (January 2026). [YOLO27](https://docs.ultralytics.com/models/yolo27) was still a preview, with no public weights or implementation. If a newer official release appears, the Ultralytics docs are the source to follow.

The code in this repository is MIT licensed. Ultralytics source is not included. Cite this project as an independent implementation.

The current code version is 0.2.0. This version changes the network, so a 0.1.0 checkpoint cannot be loaded and training has to start again.

## Where it sits among public YOLO families

| Version | Status | Public detection ideas associated with it |
| --- | --- | --- |
| YOLOv1–v3 | Published | One-stage grid prediction; later work added multi-scale features and anchors |
| YOLOv4 | Published | CSP, PAN, Mosaic, CIoU |
| YOLOv5 | Published | A widely used PyTorch training and deployment flow |
| YOLOv6 | Published | Re-parameterized backbone, decoupled head |
| YOLOv7 | Published | E-ELAN |
| YOLOv8 | Published | Anchor-free prediction, task-aligned assignment, DFL |
| YOLOv9 | Published | GELAN, programmable gradient information |
| YOLOv10 | Published | One-to-many and one-to-one branches |
| YOLO11 | Published | C3k2, C2PSA |
| YOLO12 | Community model | Attention-centric backbone |
| YOLO26 | Released January 2026 | DFL removed; optional end-to-end head that skips NMS |
| YOLO27 | Preview, weights unreleased | Compact models use two detection scales; larger models use query decoding |
| This repo | Independent code 0.2.0 | 3×3/5×5 MixUnit, StripGate, softplus box distances, dual-assignment head |

Ultralytics docs mark YOLO12 as a community model and present YOLO26 as the current release for new work.

## Network

Input height and width must both be multiples of 32. The three detection strides are 8, 16, and 32.

- The stem is two stride-2 convolutions, followed by GateCSP.
- MixUnit uses a softmax to blend depthwise 3×3 and 5×5 responses per channel, then subtracts a 3×3 average-pooled context. The residual gate bias starts at -4, so the block starts close to an identity.
- Each of the three detection scales passes through a StripGate before the head. The gate is built from a horizontal average and a vertical average. Its bias starts at 4, so it starts nearly open.
- GateCSP splits channels into two parts: one part stacks MixUnit blocks, the other passes through, and the parts are merged.
- The lowest-resolution map goes through PyramidPool: three serial 5×5 max-pools concatenated with the unpooled features.
- The neck is a P3 / P4 / P5 PAN.
- The head separates classification and boxes. A box is a softplus left/top/right/bottom distance in grid units, multiplied by the stride to reach pixels on the network input. Class outputs are logits. This head does not use DFL bins.
- The one-to-many head keeps up to 13 positives per ground-truth box, ranked by class score times IoU. Prediction applies class-aware NMS.
- The one-to-one head keeps one positive per ground-truth box. `--end2end` keeps the top boxes by score and skips NMS.
- If a ground-truth box contains no grid center, the nearest grid point is used as a positive.

The training loss is CIoU on positives plus BCE-with-logits at every location. The optimizer is AdamW. The YOLO26 training recipe documented by Ultralytics is MuSGD, STAL, and Progressive Loss. This repository uses its own defaults below.

## Install

```bash
pip install -r requirements.txt
```

Training and prediction use a GPU when CUDA is available. Pass `--device cpu` or `--device cuda` to choose. On Windows, DataLoader `--workers` defaults to 0.

ONNX export needs an extra package:

```bash
pip install onnx
```

## Synthetic smoke run

The synthetic set is noise with colored rectangles, 8 training images. The command below only checks that training, validation, and checkpoint writing finish. After one epoch, validation mAP50 can be 0.

```bash
python -m yolov28.train --demo --epochs 1 --batch 4 --imgsz 320
```

Weights are written to `runs/train/exp/weights/last.pt` and `best.pt`. If that directory already exists, the next run uses `exp2`, then `exp3`.

## Train on your data

The folder layout matches the usual YOLO text format:

```text
datasets/custom/
  images/train/a.jpg
  labels/train/a.txt
  images/val/b.jpg
  labels/val/b.txt
```

Each label row is `class xc yc w h`, with the four box values normalized by the original image width and height. See `configs/dataset.example.yaml`.

```bash
python -m yolov28.train --data configs/dataset.example.yaml --scale n --epochs 100 --batch 8 --imgsz 640
```

Augmentation is mosaic on about 75% of samples, a random horizontal flip, HSV, and letterbox. When training runs for at least 10 epochs, mosaic turns off for the last 10% of epochs. `--demo` skips mosaic, and `--no-mosaic` turns it off for a real dataset. Validation and the saved `last.pt` / `best.pt` files use an exponential moving average of the weights with decay 0.999. The default learning rate is `0.001`, weight decay is `0.0005`, and gradients are clipped at 10. With more than one epoch, the first 3 epochs warm up linearly and the rest follow a cosine decay. A single-epoch run uses the base learning rate directly. The box loss gain is 2.0, the classification gain is 0.5, and the one-to-one branch is multiplied by 1.0. When a validation set exists, `best.pt` is chosen by validation mAP50-95.

`--scale` accepts `n`, `s`, `m`, `l`, or `x`.

## Predict

```bash
python -m yolov28.predict --weights runs/train/exp/weights/last.pt --source image.jpg --conf 0.25
```

`--source` can be one image or a directory. Drawings are saved under `runs/predict/exp/` on the original image. Returned boxes are in original-image pixels.

One-to-one head:

```bash
python -m yolov28.predict --weights runs/train/exp/weights/last.pt --source image.jpg --end2end
```

A randomly initialized checkpoint, or one trained for only a few steps, still emits many low-quality boxes. Raise `--conf` to hide more of them.

## Export ONNX

```bash
python -m yolov28.export --weights runs/train/exp/weights/last.pt --imgsz 640 --output yolov28n.onnx
```

The exported input is `1×3×imgsz×imgsz`. The output is `(1, anchors, 4+classes)`: the first four numbers are xyxy on the network canvas, followed by sigmoid class scores. NMS stays outside the graph. The batch size is fixed at 1. `--end2end` selects the one-to-one head; the graph still has no NMS.

## Python API

```python
from yolov28 import YOLO

model = YOLO("n", nc=4)
model.train(demo=True, epochs=1, batch=4, imgsz=320, name="demo")
model.predict("image.jpg", imgsz=320, conf=0.25)
```

`YOLO("runs/train/demo/weights/last.pt")` loads a checkpoint written by this repository.

## Scales

Parameter counts below are for 80 classes, measured with `python -m yolov28.summary`. Changing the class count changes the classification head, so the count changes too.

| Scale | Depth | Width | Parameters at 80 classes |
| --- | ---: | ---: | ---: |
| n | 0.33 | 0.25 | 1,896,568 |
| s | 0.33 | 0.50 | 6,563,320 |
| m | 0.67 | 0.75 | 15,997,320 |
| l | 1.00 | 1.00 | 30,649,208 |
| x | 1.33 | 1.25 | 51,356,104 |

## Validation metric

mAP50 in the log is average precision at IoU 0.5. mAP50-95 averages the ten thresholds from 0.50 to 0.95 in steps of 0.05. Classes with no ground truth in the validation set are left out of the mean. `best.pt` is selected by mAP50-95.

## Tests

```bash
python tests/test_smoke.py
```

The test covers label geometry, positive assignment, backpropagation, a short overfit, and training plus prediction on a real image folder. If `onnx` is installed, it also checks that an export file is written.

## License

[MIT](LICENSE)
