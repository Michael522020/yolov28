<div align="center" dir="ltr">

[简体中文](README.md) · [繁體中文](README.zh-TW.md) · [English](README.en.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [العربية](README.ar.md)

</div>

# YOLOv28

La misma guía está en siete idiomas. Los enlaces están en la parte superior de la página.

Este repositorio es un detector de objetos sin anclas, escrito de forma independiente: red, pérdida, entrenamiento, predicción y exportación a ONNX. Los pesos empiezan en una inicialización aleatoria. La documentación se revisó en octubre de 2026. La familia más reciente publicada entonces por Ultralytics era [YOLO26](https://docs.ultralytics.com/models/yolo26) (enero de 2026). [YOLO27](https://docs.ultralytics.com/models/yolo27) seguía en vista previa, sin pesos ni implementación públicos. Si después aparece una versión oficial más nueva, manda la documentación de Ultralytics.

El código de este repositorio usa la licencia MIT. El código de Ultralytics no está incluido. Cita este proyecto como una implementación independiente.

## Lugar entre las familias YOLO públicas

| Versión | Estado | Ideas de detección descritas en público |
| --- | --- | --- |
| YOLOv1–v3 | Publicada | Predicción en una etapa sobre una cuadrícula; después se añadieron varias escalas y anclas |
| YOLOv4 | Publicada | CSP, PAN, Mosaic, CIoU |
| YOLOv5 | Publicada | Un flujo de entrenamiento y despliegue en PyTorch muy usado |
| YOLOv6 | Publicada | Backbone reparametrizado, cabeza desacoplada |
| YOLOv7 | Publicada | E-ELAN |
| YOLOv8 | Publicada | Predicción sin anclas, asignación alineada con la tarea, DFL |
| YOLOv9 | Publicada | GELAN, información de gradiente programable |
| YOLOv10 | Publicada | Ramas de uno a muchos y de uno a uno |
| YOLO11 | Publicada | C3k2, C2PSA |
| YOLO12 | Modelo comunitario | Backbone centrado en la atención |
| YOLO26 | Publicada en enero de 2026 | Se quitó DFL; cabeza de extremo a extremo opcional, sin NMS |
| YOLO27 | Vista previa, pesos sin publicar | Los modelos pequeños usan dos escalas; los grandes, decodificación por consultas |
| Este repo | Código independiente | GateCSP, distancias de caja con softplus, cabeza de doble asignación |

La documentación de Ultralytics marca YOLO12 como modelo comunitario y presenta YOLO26 como la familia actual para un trabajo nuevo.

## Red

El alto y el ancho de la entrada tienen que ser múltiplos de 32. Los tres pasos de detección son 8, 16 y 32.

- El tallo hace dos convoluciones de paso 2 y después entra en GateCSP.
- MixUnit resta un contexto de pooling medio 3×3 a un camino local depthwise 3×3. El sesgo de la compuerta empieza en -4, así que la sigmoide queda cerca de 0 y el residual empieza cerca de la identidad.
- GateCSP parte los canales: una rama apila bloques MixUnit, la otra pasa directa, y luego se unen.
- El mapa de menor resolución pasa por PyramidPool: tres max-pools 5×5 en serie, concatenados con las características sin pooling.
- El cuello es un PAN de P3, P4 y P5.
- La cabeza separa la clasificación y las cajas. Una caja es una distancia izquierda/arriba/derecha/abajo tras softplus, en unidades de celda, multiplicada por el paso para llegar a píxeles de la entrada de la red. Las clases son logits. Esta cabeza no usa intervalos DFL.
- La cabeza de uno a muchos conserva hasta 13 positivos por caja real, ordenados por puntuación de clase por IoU. La predicción aplica NMS por clase.
- La cabeza de uno a uno conserva un positivo por caja real. `--end2end` se queda con las mejores cajas según la puntuación y no aplica NMS.
- Si una caja real no contiene ningún centro de celda, se usa el punto de la cuadrícula más cercano como positivo.

La pérdida de entrenamiento es CIoU en los positivos más BCE-with-logits en todas las posiciones. El optimizador es AdamW. La receta documentada de YOLO26 es MuSGD, STAL y Progressive Loss. Este repositorio usa los valores propios de abajo.

## Instalación

```bash
pip install -r requirements.txt
```

El entrenamiento y la predicción usan la GPU cuando hay CUDA. `--device cpu` o `--device cuda` fija el dispositivo. En Windows, `--workers` del DataLoader vale 0 por defecto.

La exportación a ONNX necesita otro paquete:

```bash
pip install onnx
```

## Prueba con datos sintéticos

El conjunto sintético es ruido con rectángulos de color, 8 imágenes de entrenamiento. El comando solo comprueba que el entrenamiento, la validación y la escritura del punto de control terminan. Tras una época, la mAP50 de validación puede ser 0.

```bash
python -m yolov28.train --demo --epochs 1 --batch 4 --imgsz 320
```

Los pesos quedan en `runs/train/exp/weights/last.pt` y `best.pt`. Si esa carpeta ya existe, la siguiente ejecución usa `exp2` y luego `exp3`.

## Entrenar tus datos

Las carpetas siguen el formato de texto YOLO habitual:

```text
datasets/custom/
  images/train/a.jpg
  labels/train/a.txt
  images/val/b.jpg
  labels/val/b.txt
```

Cada fila de etiqueta es `clase xc yc w h`. Los cuatro valores de la caja están normalizados por el ancho y el alto de la imagen original. Mira `configs/dataset.example.yaml`.

```bash
python -m yolov28.train --data configs/dataset.example.yaml --scale n --epochs 100 --batch 8 --imgsz 640
```

El aumento de este bucle es un volteo horizontal aleatorio más letterbox. Mosaic no entra en el bucle. La tasa de aprendizaje es `0.001`, el weight decay es `0.0005` y los gradientes se recortan en 10. Con más de una época, las 3 primeras suben de forma lineal y el resto sigue un descenso en coseno. Una sola época usa directamente la tasa base. La ganancia de caja es 2.0, la de clasificación es 0.5 y la rama de uno a uno se multiplica por 1.0. Si hay conjunto de validación, `best.pt` se elige por la mAP50 de validación.

`--scale` acepta `n`, `s`, `m`, `l` o `x`.

## Predicción

```bash
python -m yolov28.predict --weights runs/train/exp/weights/last.pt --source image.jpg --conf 0.25
```

`--source` puede ser una imagen o un directorio. Los dibujos se guardan en `runs/predict/exp/` sobre la imagen original. Las cajas devueltas están en píxeles de esa imagen.

Cabeza de uno a uno:

```bash
python -m yolov28.predict --weights runs/train/exp/weights/last.pt --source image.jpg --end2end
```

Un punto de control con inicialización aleatoria, o entrenado solo unos pasos, sigue emitiendo muchas cajas de baja calidad. Sube `--conf` para ocultar más.

## Exportar a ONNX

```bash
python -m yolov28.export --weights runs/train/exp/weights/last.pt --imgsz 640 --output yolov28n.onnx
```

La entrada exportada es `1×3×imgsz×imgsz`. La salida es `(1, anchors, 4+clases)`: los cuatro primeros números son xyxy en el lienzo de la red, seguidos de puntuaciones de clase tras la sigmoide. El NMS queda fuera del grafo. El tamaño de lote es fijo en 1. `--end2end` elige la cabeza de uno a uno; el grafo sigue sin NMS.

## API de Python

```python
from yolov28 import YOLO

model = YOLO("n", nc=4)
model.train(demo=True, epochs=1, batch=4, imgsz=320, name="demo")
model.predict("image.jpg", imgsz=320, conf=0.25)
```

`YOLO("runs/train/demo/weights/last.pt")` carga un punto de control escrito por este repositorio.

## Escalas

Las cifras de parámetros son para 80 clases y salen de `python -m yolov28.summary`. Si cambia el número de clases, cambia la cabeza de clasificación y también el número de parámetros.

| Escala | Profundidad | Anchura | Parámetros con 80 clases |
| --- | ---: | ---: | ---: |
| n | 0.33 | 0.25 | 1,789,192 |
| s | 0.33 | 0.50 | 6,163,224 |
| m | 0.67 | 0.75 | 14,573,736 |
| l | 1.00 | 1.00 | 27,206,712 |
| x | 1.33 | 1.25 | 44,564,424 |

## Métrica de validación

La mAP50 del registro es la precisión media con IoU 0.5. Las clases sin verdad de referencia en la validación quedan fuera de la media. La mAP al estilo COCO promedia umbrales de IoU desde 0.50 hasta 0.95. El número que se imprime aquí es el resultado de un solo umbral.

## Pruebas

```bash
python tests/test_smoke.py
```

La prueba cubre la geometría de las etiquetas, la asignación de positivos, la retropropagación, un sobreajuste corto y el entrenamiento más la predicción sobre una carpeta de imágenes reales. Si `onnx` está instalado, también comprueba que se escribe un archivo de exportación.

## Licencia

[MIT](LICENSE)
