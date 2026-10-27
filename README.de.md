<div align="center" dir="ltr">

[简体中文](README.md) · [繁體中文](README.zh-TW.md) · [English](README.en.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [العربية](README.ar.md)

</div>

# YOLOv28

Dieselbe Anleitung gibt es in sieben Sprachen. Die Links stehen oben auf der Seite.

Dieses Repository ist ein unabhängiger ankerfreier Objektdetektor: Netz, Verlust, Training, Vorhersage und ONNX-Export. Die Gewichte starten bei einer Zufallsinitialisierung. Der Dokumentationsstand wurde im Oktober 2026 geprüft. Die damals neueste von Ultralytics veröffentlichte Familie war [YOLO26](https://docs.ultralytics.com/models/yolo26) (Januar 2026). [YOLO27](https://docs.ultralytics.com/models/yolo27) war noch eine Vorschau, ohne öffentliche Gewichte oder Implementierung. Erscheint später eine neuere offizielle Fassung, gilt die Ultralytics-Dokumentation.

Der Code in diesem Repository steht unter der MIT-Lizenz. Ultralytics-Quelltext ist nicht enthalten. Zitiere das Projekt als unabhängige Implementierung.

## Einordnung zu den öffentlichen YOLO-Familien

| Version | Status | Öffentlich beschriebene Ideen zur Detektion |
| --- | --- | --- |
| YOLOv1–v3 | Veröffentlicht | Einstufige Gittervorhersage; später kamen Mehrskalenmerkmale und Anker dazu |
| YOLOv4 | Veröffentlicht | CSP, PAN, Mosaic, CIoU |
| YOLOv5 | Veröffentlicht | Ein weit verbreiteter PyTorch-Ablauf für Training und Deployment |
| YOLOv6 | Veröffentlicht | Reparametrisiertes Backbone, entkoppelter Kopf |
| YOLOv7 | Veröffentlicht | E-ELAN |
| YOLOv8 | Veröffentlicht | Ankerfreie Vorhersage, aufgabenausgerichtete Zuordnung, DFL |
| YOLOv9 | Veröffentlicht | GELAN, programmierbare Gradienteninformation |
| YOLOv10 | Veröffentlicht | Eins-zu-viele- und Eins-zu-eins-Zweig |
| YOLO11 | Veröffentlicht | C3k2, C2PSA |
| YOLO12 | Community-Modell | Aufmerksamkeit im Backbone |
| YOLO26 | Veröffentlicht im Januar 2026 | DFL entfernt; optionaler End-to-End-Kopf ohne NMS |
| YOLO27 | Vorschau, Gewichte nicht veröffentlicht | Kompakte Modelle nutzen zwei Detektionsskalen; größere Modelle eine Query-Decodierung |
| Dieses Repo | Unabhängiger Code | GateCSP, Softplus-Boxabstände, Kopf mit doppelter Zuordnung |

Die Ultralytics-Dokumentation führt YOLO12 als Community-Modell und YOLO26 als aktuelle Fassung für neue Arbeiten.

## Netz

Höhe und Breite der Eingabe müssen Vielfache von 32 sein. Die drei Detektionsschritte sind 8, 16 und 32.

- Der Stamm besteht aus zwei Faltungen mit Schritt 2, danach folgt GateCSP.
- MixUnit zieht einen 3×3-Mittelwertkontext von einem tiefenseparierten 3×3-Pfad ab. Der Gate-Bias startet bei -4, der Sigmoid liegt nahe 0, und das Residuum beginnt nahe an der Identität.
- GateCSP teilt die Kanäle: ein Teil stapelt MixUnit-Blöcke, der andere läuft durch, danach werden beide zusammengeführt.
- Die gröbste Karte geht durch PyramidPool: drei serielle 5×5-Max-Pools, verkettet mit den ungepoolten Merkmalen.
- Der Hals ist ein PAN über P3, P4 und P5.
- Der Kopf trennt Klasse und Box. Eine Box ist ein Softplus-Abstand links/oben/rechts/unten in Gittereinheiten, multipliziert mit dem Schritt, also Pixel auf der Netzeingabe. Klassenausgänge sind Logits. Dieser Kopf benutzt keine DFL-Bins.
- Der Eins-zu-viele-Kopf behält bis zu 13 Positive je Ground-Truth-Box, sortiert nach Klassenscore mal IoU. Die Vorhersage nutzt klassenweises NMS.
- Der Eins-zu-eins-Kopf behält ein Positiv je Ground-Truth-Box. `--end2end` nimmt die besten Boxen nach Score und lässt NMS weg.
- Liegt in einer Ground-Truth-Box kein Gitterzentrum, wird der nächste Gitterpunkt als Positiv verwendet.

Der Trainingsverlust ist CIoU auf den Positiven plus BCE-with-logits an jeder Stelle. Der Optimierer ist AdamW. Das in der YOLO26-Dokumentation beschriebene Rezept ist MuSGD, STAL und Progressive Loss. Dieses Repository verwendet die eigenen Vorgaben unten.

## Installation

```bash
pip install -r requirements.txt
```

Training und Vorhersage nutzen eine GPU, sobald CUDA vorhanden ist. Mit `--device cpu` oder `--device cuda` legst du das Gerät fest. Unter Windows ist `--workers` für den DataLoader standardmäßig 0.

Der ONNX-Export braucht ein zusätzliches Paket:

```bash
pip install onnx
```

## Synthetischer Probelauf

Die synthetische Menge besteht aus Rauschen mit farbigen Rechtecken, 8 Trainingsbilder. Der Befehl prüft nur, dass Training, Validierung und das Schreiben des Checkpoints durchlaufen. Nach einer Epoche kann die Validierungs-mAP50 bei 0 liegen.

```bash
python -m yolov28.train --demo --epochs 1 --batch 4 --imgsz 320
```

Die Gewichte liegen in `runs/train/exp/weights/last.pt` und `best.pt`. Existiert das Verzeichnis schon, heißt der nächste Lauf `exp2`, danach `exp3`.

## Eigene Daten trainieren

Die Ordner folgen dem üblichen YOLO-Textformat:

```text
datasets/custom/
  images/train/a.jpg
  labels/train/a.txt
  images/val/b.jpg
  labels/val/b.txt
```

Jede Beschriftungszeile ist `Klasse xc yc w h`. Die vier Boxwerte sind auf Breite und Höhe des Originalbildes normiert. Siehe `configs/dataset.example.yaml`.

```bash
python -m yolov28.train --data configs/dataset.example.yaml --scale n --epochs 100 --batch 8 --imgsz 640
```

Die Augmentierung in dieser Schleife ist eine zufällige horizontale Spiegelung plus Letterbox. Mosaic gehört nicht dazu. Die Lernrate ist `0.001`, der Weight Decay `0.0005`, Gradienten werden bei 10 beschnitten. Bei mehr als einer Epoche wärmen die ersten 3 Epochen linear auf, danach folgt ein Kosinusabfall. Ein Lauf mit einer Epoche nutzt direkt die Basislernrate. Der Box-Gewinn ist 2.0, der Klassengewinn 0.5, der Eins-zu-eins-Zweig wird mit 1.0 multipliziert. Gibt es eine Validierungsmenge, wird `best.pt` nach der Validierungs-mAP50 gewählt.

`--scale` akzeptiert `n`, `s`, `m`, `l` oder `x`.

## Vorhersage

```bash
python -m yolov28.predict --weights runs/train/exp/weights/last.pt --source image.jpg --conf 0.25
```

`--source` kann ein Bild oder ein Verzeichnis sein. Die Zeichnungen liegen unter `runs/predict/exp/` auf dem Originalbild. Zurückgegebene Boxen sind Pixel des Originalbildes.

Eins-zu-eins-Kopf:

```bash
python -m yolov28.predict --weights runs/train/exp/weights/last.pt --source image.jpg --end2end
```

Ein zufällig initialisierter Checkpoint oder einer nach nur wenigen Schritten liefert weiterhin viele schwache Boxen. Ein höheres `--conf` blendet mehr davon aus.

## ONNX-Export

```bash
python -m yolov28.export --weights runs/train/exp/weights/last.pt --imgsz 640 --output yolov28n.onnx
```

Die exportierte Eingabe ist `1×3×imgsz×imgsz`. Die Ausgabe ist `(1, anchors, 4+Klassen)`: die ersten vier Zahlen sind xyxy auf der Netzeingabe, danach folgen Sigmoid-Klassenscores. NMS bleibt außerhalb des Graphen. Die Batchgröße ist auf 1 festgelegt. `--end2end` wählt den Eins-zu-eins-Kopf; auch dieser Graph enthält kein NMS.

## Python-API

```python
from yolov28 import YOLO

model = YOLO("n", nc=4)
model.train(demo=True, epochs=1, batch=4, imgsz=320, name="demo")
model.predict("image.jpg", imgsz=320, conf=0.25)
```

`YOLO("runs/train/demo/weights/last.pt")` lädt einen von diesem Repository geschriebenen Checkpoint.

## Größen

Die Parameterzahlen gelten für 80 Klassen und stammen von `python -m yolov28.summary`. Ändert sich die Klassenzahl, ändert sich der Klassifikationskopf und damit die Parameterzahl.

| Größe | Tiefe | Breite | Parameter bei 80 Klassen |
| --- | ---: | ---: | ---: |
| n | 0.33 | 0.25 | 1,789,192 |
| s | 0.33 | 0.50 | 6,163,224 |
| m | 0.67 | 0.75 | 14,573,736 |
| l | 1.00 | 1.00 | 27,206,712 |
| x | 1.33 | 1.25 | 44,564,424 |

## Validierungsmetrik

mAP50 im Log ist die durchschnittliche Präzision bei IoU 0.5. Klassen ohne Ground Truth in der Validierungsmenge gehen nicht in den Mittelwert ein. Das COCO-übliche mAP mittelt IoU-Schwellen von 0.50 bis 0.95. Die hier ausgegebene Zahl ist das Ergebnis bei einer einzigen Schwelle.

## Tests

```bash
python tests/test_smoke.py
```

Der Test prüft Beschriftungsgeometrie, positive Zuordnung, Backpropagation, ein kurzes Overfitting sowie Training und Vorhersage auf einem echten Bildordner. Ist `onnx` installiert, prüft er zusätzlich, dass eine Exportdatei geschrieben wird.

## Lizenz

[MIT](LICENSE)
