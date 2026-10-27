<div align="center" dir="ltr">

[简体中文](README.md) · [繁體中文](README.zh-TW.md) · [English](README.en.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [العربية](README.ar.md)

</div>

# YOLOv28

同一份说明有七个语言版本，切换链接在页面顶部。

本仓库是一套独立编写的 anchor-free 目标检测代码，包含网络、损失、训练、预测和 ONNX 导出。权重从随机初始化开始。文档核对日期是 2026 年 10 月：Ultralytics 当时已经发布的最新系列是 [YOLO26](https://docs.ultralytics.com/models/yolo26)（2026 年 1 月），[YOLO27](https://docs.ultralytics.com/models/yolo27) 仍是预览，权重和实现都还没有公开。之后若官方发布新版本，以 Ultralytics 文档为准。

这里的代码使用 MIT 许可，仓库里没有 Ultralytics 的源码。请把本仓库当作独立实现来引用，不要把它写成官方 YOLO 发行版。

## 和公开 YOLO 系列的关系

| 版本 | 状态 | 公开资料里和检测结构有关的要点 |
| --- | --- | --- |
| YOLOv1–v3 | 已发表 | 单阶段网格预测；后续加入多尺度和 anchor |
| YOLOv4 | 已发表 | CSP、PAN、Mosaic、CIoU |
| YOLOv5 | 已发表 | 被广泛使用的 PyTorch 训练和部署流程 |
| YOLOv6 | 已发表 | 重参数化骨干、解耦检测头 |
| YOLOv7 | 已发表 | E-ELAN |
| YOLOv8 | 已发表 | anchor-free、任务对齐分配、DFL |
| YOLOv9 | 已发表 | GELAN、可编程梯度信息 |
| YOLOv10 | 已发表 | 一对多和一对一双分支 |
| YOLO11 | 已发表 | C3k2、C2PSA |
| YOLO12 | 社区模型 | 以注意力为主的骨干 |
| YOLO26 | 2026-01 已发布 | 去掉 DFL，可选端到端、推理时不做 NMS 的头 |
| YOLO27 | 预览，权重未公开 | 小模型用两个检测尺度，大模型用查询解码 |
| 本仓库 | 独立代码 | GateCSP、softplus 框距离、双分配检测头 |

Ultralytics 文档把 YOLO12 标为社区模型，并把 YOLO26 作为新项目的正式系列。

## 网络

输入高和宽都要是 32 的倍数。三个检测尺度的步长是 8、16、32。

- 茎部做两次步长为 2 的卷积，然后进入 GateCSP。
- MixUnit 把深度可分离 3×3 和 3×3 平均池化上下文做差。门控偏置初始为 -4，sigmoid 接近 0，残差一开始接近恒等映射。
- GateCSP 把通道拆成两支：一支堆 MixUnit，另一支直通，再合并。
- 最低分辨率特征后面接 PyramidPool：三次串联的 5×5 最大池化，与池化前的特征拼在一起。
- 颈部是 P3、P4、P5 的 PAN。
- 检测头把分类和框分开。框是 softplus 之后的左、上、右、下距离，单位是网格，乘步长后变成输入图像上的像素。分类头输出的是 logits，还没有做 sigmoid。这里不用 DFL 分桶。
- 一对多头按「类别分数 × IoU」给每个目标保留最多 13 个正样本，预测时按类别做 NMS。
- 一对一头给每个目标保留 1 个正样本。加上 `--end2end` 时按分数取框，不做 NMS。
- 标注框里如果没有网格中心，就用离框中心最近的网格点作为正样本。

训练损失是正样本上的 CIoU，加上全部位置的 BCE-with-logits。优化器是 AdamW。YOLO26 文档中的训练配方是 MuSGD、STAL 和 Progressive Loss；本仓库使用自己的这一套默认值。

## 安装

```bash
pip install -r requirements.txt
```

有 CUDA 时，训练和预测会使用 GPU。也可以显式传入 `--device cpu` 或 `--device cuda`。Windows 上 DataLoader 的 `--workers` 默认为 0。

导出 ONNX 需要另装：

```bash
pip install onnx
```

## 用合成数据跑通

合成集是带颜色矩形的噪声图，训练集 8 张。下面这条命令只检查训练、验证和存档能不能走完。跑完一轮后，验证 mAP50 可以是 0。

```bash
python -m yolov28.train --demo --epochs 1 --batch 4 --imgsz 320
```

权重写在 `runs/train/exp/weights/last.pt` 和 `best.pt`。同名目录已存在时，新目录会变成 `exp2`、`exp3`。

## 训练自己的数据

目录和标注与常见的 YOLO 文本格式一致：

```text
datasets/custom/
  images/train/a.jpg
  labels/train/a.txt
  images/val/b.jpg
  labels/val/b.txt
```

标注每一行是 `类别 xc yc w h`，四个坐标相对原图宽高归一化到 0–1。示例配置见 `configs/dataset.example.yaml`。

```bash
python -m yolov28.train --data configs/dataset.example.yaml --scale n --epochs 100 --batch 8 --imgsz 640
```

当前增强只有随机水平翻转和 letterbox。Mosaic 不在这个训练循环里。默认学习率 `0.001`，权重衰减 `0.0005`，梯度裁剪阈值 10。多于 1 个 epoch 时，前 3 个 epoch 线性预热，之后余弦下降；只训练 1 个 epoch 时直接用基础学习率。框损失系数 2.0，分类损失系数 0.5，一对一分支再乘 1.0。有验证集时，`best.pt` 按验证 mAP50 保存。

`--scale` 可以是 `n`、`s`、`m`、`l`、`x`。

## 预测

```bash
python -m yolov28.predict --weights runs/train/exp/weights/last.pt --source image.jpg --conf 0.25
```

`--source` 可以是一张图或一个目录。结果画在原图上，保存到 `runs/predict/exp/`。返回的框是原图像素坐标。

一对一头：

```bash
python -m yolov28.predict --weights runs/train/exp/weights/last.pt --source image.jpg --end2end
```

随机初始化或只训练了几步的权重，仍会输出许多低质量框。把 `--conf` 调高可以少看一些。

## 导出 ONNX

```bash
python -m yolov28.export --weights runs/train/exp/weights/last.pt --imgsz 640 --output yolov28n.onnx
```

导出文件的输入是 `1×3×imgsz×imgsz`。输出是 `(1, anchors, 4+类别数)`：前 4 个数是网络输入画布上的 xyxy，后面是 sigmoid 后的类别分数。NMS 留在图外面。批次固定为 1。`--end2end` 改走一对一头，图里仍然没有 NMS。

## Python 接口

```python
from yolov28 import YOLO

model = YOLO("n", nc=4)
model.train(demo=True, epochs=1, batch=4, imgsz=320, name="demo")
model.predict("image.jpg", imgsz=320, conf=0.25)
```

`YOLO("runs/train/demo/weights/last.pt")` 会载入本仓库保存的检查点。

## 规模

下面的参数量是 80 类时由 `python -m yolov28.summary` 统计的。类别数一变，分类头的输出通道会变，参数量也会变。

| 规模 | 深度系数 | 宽度系数 | 80 类参数量 |
| --- | ---: | ---: | ---: |
| n | 0.33 | 0.25 | 1,789,192 |
| s | 0.33 | 0.50 | 6,163,224 |
| m | 0.67 | 0.75 | 14,573,736 |
| l | 1.00 | 1.00 | 27,206,712 |
| x | 1.33 | 1.25 | 44,564,424 |

## 验证指标

日志里的 mAP50 是 IoU 阈值为 0.5 的平均精度。验证集中没有真值的类别不进入平均。COCO 上常见的 mAP 会在 IoU 0.50 到 0.95 之间再取平均；本仓库打印的是前一种单阈值结果。

## 测试

```bash
python tests/test_smoke.py
```

它检查标注几何、正样本分配、反向传播、短程过拟合、真实图片目录的训练和预测。机器上装了 `onnx` 时，还会检查导出文件。

## 许可

[MIT](LICENSE)
