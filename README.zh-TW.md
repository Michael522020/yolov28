<div align="center" dir="ltr">

[简体中文](README.md) · [繁體中文](README.zh-TW.md) · [English](README.en.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [العربية](README.ar.md)

</div>

# YOLOv28

同一份說明有七個語言版本，切換連結在頁面頂部。

本儲存庫是一套獨立編寫的 anchor-free 物件偵測程式，包含網路、損失、訓練、推論和 ONNX 匯出。權重從隨機初始化開始。文件核對日期是 2026 年 10 月：Ultralytics 當時已經發布的最新系列是 [YOLO26](https://docs.ultralytics.com/models/yolo26)（2026 年 1 月），[YOLO27](https://docs.ultralytics.com/models/yolo27) 仍是預覽，權重和實作都還沒有公開。之後若官方發布新版本，以 Ultralytics 文件為準。

這裡的程式碼使用 MIT 授權，儲存庫裡沒有 Ultralytics 的原始碼。引用時請把它視為獨立實作，不要寫成官方 YOLO 發行版。

目前程式版本是 0.2.0。這一版改了網路，0.1.0 的檢查點不能繼續載入，需要重新訓練。

## 和公開 YOLO 系列的關係

| 版本 | 狀態 | 公開資料裡和偵測結構有關的要點 |
| --- | --- | --- |
| YOLOv1–v3 | 已發表 | 單階段網格預測；後續加入多尺度和 anchor |
| YOLOv4 | 已發表 | CSP、PAN、Mosaic、CIoU |
| YOLOv5 | 已發表 | 被廣泛使用的 PyTorch 訓練和部署流程 |
| YOLOv6 | 已發表 | 重參數化骨幹、解耦偵測頭 |
| YOLOv7 | 已發表 | E-ELAN |
| YOLOv8 | 已發表 | anchor-free、任務對齊分配、DFL |
| YOLOv9 | 已發表 | GELAN、可程式化梯度資訊 |
| YOLOv10 | 已發表 | 一對多和一對一雙分支 |
| YOLO11 | 已發表 | C3k2、C2PSA |
| YOLO12 | 社群模型 | 以注意力為主的骨幹 |
| YOLO26 | 2026-01 已發布 | 去掉 DFL，可選端對端、推論時不做 NMS 的頭 |
| YOLO27 | 預覽，權重未公開 | 小模型用兩個偵測尺度，大模型用查詢解碼 |
| 本儲存庫 | 獨立程式碼 0.2.0 | 3×3/5×5 MixUnit、StripGate、softplus 框距離、雙分配偵測頭 |

Ultralytics 文件把 YOLO12 標為社群模型，並把 YOLO26 作為新專案的正式系列。

## 網路

輸入高和寬都要是 32 的倍數。三個偵測尺度的步長是 8、16、32。

- 莖部做兩次步長為 2 的卷積，然後進入 GateCSP。
- MixUnit 用 softmax 在深度可分離 3×3 和 5×5 之間做逐通道混合，再減去 3×3 平均池化上下文。殘差閘控偏置初始為 -4，一開始接近恆等映射。
- 三個偵測尺度在進入偵測頭之前各經過一個 StripGate：沿高和寬分別取平均，合成一個空間閘。閘的偏置初始為 4，一開始接近全開。
- GateCSP 把通道拆成兩支：一支堆 MixUnit，另一支直通，再合併。
- 最低解析度特徵後面接 PyramidPool：三次串聯的 5×5 最大池化，與池化前的特徵拼在一起。
- 頸部是 P3、P4、P5 的 PAN。
- 偵測頭把分類和框分開。框是 softplus 之後的左、上、右、下距離，單位是網格，乘步長後變成輸入圖像上的像素。分類頭輸出的是 logits，還沒有做 sigmoid。這裡不用 DFL 分桶。
- 一對多頭按「類別分數 × IoU」給每個目標保留最多 13 個正樣本，預測時按類別做 NMS。
- 一對一頭給每個目標保留 1 個正樣本。加上 `--end2end` 時按分數取框，不做 NMS。
- 標註框裡如果沒有網格中心，就用離框中心最近的網格點作為正樣本。

訓練損失是正樣本上的 CIoU，加上全部位置的 BCE-with-logits。最佳化器是 AdamW。YOLO26 文件中的訓練配方是 MuSGD、STAL 和 Progressive Loss；本儲存庫使用自己的這一套預設值。

## 安裝

```bash
pip install -r requirements.txt
```

有 CUDA 時，訓練和推論會使用 GPU。也可以明確傳入 `--device cpu` 或 `--device cuda`。Windows 上 DataLoader 的 `--workers` 預設為 0。

匯出 ONNX 需要另裝：

```bash
pip install onnx
```

## 用合成資料跑通

合成集是帶顏色矩形的雜訊圖，訓練集 8 張。下面這條指令只檢查訓練、驗證和存檔能不能走完。跑完一輪後，驗證 mAP50 可以是 0。

```bash
python -m yolov28.train --demo --epochs 1 --batch 4 --imgsz 320
```

權重寫在 `runs/train/exp/weights/last.pt` 和 `best.pt`。同名目錄已存在時，新目錄會變成 `exp2`、`exp3`。

## 訓練自己的資料

目錄和標註與常見的 YOLO 文字格式一致：

```text
datasets/custom/
  images/train/a.jpg
  labels/train/a.txt
  images/val/b.jpg
  labels/val/b.txt
```

標註每一行是 `類別 xc yc w h`，四個座標相對原圖寬高正規化到 0–1。範例設定見 `configs/dataset.example.yaml`。

```bash
python -m yolov28.train --data configs/dataset.example.yaml --scale n --epochs 100 --batch 8 --imgsz 640
```

訓練增強包括 Mosaic（約 75% 的樣本）、隨機水平翻轉、HSV 和 letterbox。訓練不少於 10 個 epoch 時，最後 10% 的輪次關閉 Mosaic。`--demo` 的合成圖不做 Mosaic，也可以用 `--no-mosaic` 關掉。驗證以及寫入的 `last.pt` 和 `best.pt` 使用權重的指數滑動平均（衰減 0.999）。預設學習率 `0.001`，權重衰減 `0.0005`，梯度裁剪閾值 10。多於 1 個 epoch 時，前 3 個 epoch 線性預熱，之後餘弦下降；只訓練 1 個 epoch 時直接用基礎學習率。框損失係數 2.0，分類損失係數 0.5，一對一分支再乘 1.0。有驗證集時，`best.pt` 按驗證 mAP50-95 儲存。

`--scale` 可以是 `n`、`s`、`m`、`l`、`x`。

## 推論

```bash
python -m yolov28.predict --weights runs/train/exp/weights/last.pt --source image.jpg --conf 0.25
```

`--source` 可以是一張圖或一個目錄。結果畫在原圖上，儲存到 `runs/predict/exp/`。回傳的框是原圖像素座標。

一對一頭：

```bash
python -m yolov28.predict --weights runs/train/exp/weights/last.pt --source image.jpg --end2end
```

隨機初始化或只訓練了幾步的權重，仍會輸出許多低品質框。把 `--conf` 調高可以少看一些。

## 匯出 ONNX

```bash
python -m yolov28.export --weights runs/train/exp/weights/last.pt --imgsz 640 --output yolov28n.onnx
```

匯出檔的輸入是 `1×3×imgsz×imgsz`。輸出是 `(1, anchors, 4+類別數)`：前 4 個數是網路輸入畫布上的 xyxy，後面是 sigmoid 後的類別分數。NMS 留在圖外面。批次固定為 1。`--end2end` 改走一對一頭，圖裡仍然沒有 NMS。

## Python 介面

```python
from yolov28 import YOLO

model = YOLO("n", nc=4)
model.train(demo=True, epochs=1, batch=4, imgsz=320, name="demo")
model.predict("image.jpg", imgsz=320, conf=0.25)
```

`YOLO("runs/train/demo/weights/last.pt")` 會載入本儲存庫儲存的檢查點。

## 規模

下面的參數量是 80 類時由 `python -m yolov28.summary` 統計的。類別數一變，分類頭的輸出通道會變，參數量也會變。

| 規模 | 深度係數 | 寬度係數 | 80 類參數量 |
| --- | ---: | ---: | ---: |
| n | 0.33 | 0.25 | 1,896,568 |
| s | 0.33 | 0.50 | 6,563,320 |
| m | 0.67 | 0.75 | 15,997,320 |
| l | 1.00 | 1.00 | 30,649,208 |
| x | 1.33 | 1.25 | 51,356,104 |

## 驗證指標

日誌裡的 mAP50 是 IoU 閾值為 0.5 的平均精度。mAP50-95 把 IoU 從 0.50 到 0.95、步長 0.05 的 10 個閾值再取平均。驗證集中沒有真值的類別不進入平均。`best.pt` 按 mAP50-95 挑選。

## 測試

```bash
python tests/test_smoke.py
```

它檢查標註幾何、正樣本分配、反向傳播、短程過擬合、真實圖片目錄的訓練和推論。機器上裝了 `onnx` 時，還會檢查匯出檔。

## 授權

[MIT](LICENSE)
