<div align="center" dir="ltr">

[简体中文](README.md) · [繁體中文](README.zh-TW.md) · [English](README.en.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [العربية](README.ar.md)

</div>

# YOLOv28

Le même guide existe en sept langues. Les liens sont en haut de la page.

Ce dépôt est un détecteur d'objets sans ancres, écrit de façon indépendante : réseau, perte, entraînement, prédiction et export ONNX. Les poids partent d'une initialisation aléatoire. La documentation a été vérifiée en octobre 2026. La famille alors publiée la plus récente chez Ultralytics était [YOLO26](https://docs.ultralytics.com/models/yolo26) (janvier 2026). [YOLO27](https://docs.ultralytics.com/models/yolo27) restait un aperçu, sans poids ni implémentation publics. Si une version officielle plus récente paraît, la documentation Ultralytics fait foi.

Le code de ce dépôt est sous licence MIT. Le code source d'Ultralytics n'y figure pas. Citez ce projet comme une implémentation indépendante.

## Place parmi les familles YOLO publiques

| Version | État | Idées de détection décrites publiquement |
| --- | --- | --- |
| YOLOv1–v3 | Publiée | Prédiction en une étape sur une grille ; les suites ont ajouté le multi-échelle et les ancres |
| YOLOv4 | Publiée | CSP, PAN, Mosaic, CIoU |
| YOLOv5 | Publiée | Un flux PyTorch d'entraînement et de déploiement très répandu |
| YOLOv6 | Publiée | Backbone reparamétré, tête découplée |
| YOLOv7 | Publiée | E-ELAN |
| YOLOv8 | Publiée | Prédiction sans ancres, assignation alignée sur la tâche, DFL |
| YOLOv9 | Publiée | GELAN, information de gradient programmable |
| YOLOv10 | Publiée | Branches un-vers-plusieurs et un-vers-un |
| YOLO11 | Publiée | C3k2, C2PSA |
| YOLO12 | Modèle communautaire | Backbone centré sur l'attention |
| YOLO26 | Publiée en janvier 2026 | DFL retiré ; tête de bout en bout optionnelle, sans NMS |
| YOLO27 | Aperçu, poids non publiés | Les petits modèles utilisent deux échelles ; les plus grands, un décodage par requêtes |
| Ce dépôt | Code indépendant | GateCSP, distances de boîte en softplus, tête à double assignation |

La documentation Ultralytics présente YOLO12 comme un modèle communautaire et YOLO26 comme la famille actuelle pour un nouveau travail.

## Réseau

La hauteur et la largeur d'entrée doivent être des multiples de 32. Les trois pas de détection sont 8, 16 et 32.

- La tige fait deux convolutions de pas 2, puis entre dans GateCSP.
- MixUnit soustrait un contexte de pooling moyen 3×3 à un chemin local en convolution depthwise 3×3. Le biais de la porte vaut -4 au départ, donc le sigmoïde est proche de 0 et le résidu commence près de l'identité.
- GateCSP sépare les canaux : une branche empile des MixUnit, l'autre passe telle quelle, puis les deux sont fusionnées.
- La carte de plus basse résolution passe par PyramidPool : trois max-pools 5×5 en série, concaténés aux caractéristiques non poolées.
- Le cou est un PAN sur P3, P4 et P5.
- La tête sépare la classification et les boîtes. Une boîte est une distance gauche/haut/droite/bas après softplus, en unités de grille, multipliée par le pas pour obtenir des pixels sur l'entrée du réseau. Les classes sont des logits. Cette tête n'utilise pas de casiers DFL.
- La tête un-vers-plusieurs garde jusqu'à 13 positifs par vérité terrain, classés par score de classe fois IoU. La prédiction applique un NMS par classe.
- La tête un-vers-un garde un positif par vérité terrain. `--end2end` garde les meilleures boîtes selon le score et n'applique pas de NMS.
- Si une vérité terrain ne contient aucun centre de grille, le point de grille le plus proche sert de positif.

La perte d'entraînement est le CIoU sur les positifs, plus une BCE-with-logits partout. L'optimiseur est AdamW. La recette documentée pour YOLO26 est MuSGD, STAL et Progressive Loss. Ce dépôt utilise ses propres valeurs par défaut ci-dessous.

## Installation

```bash
pip install -r requirements.txt
```

L'entraînement et la prédiction utilisent un GPU quand CUDA est disponible. `--device cpu` ou `--device cuda` impose le choix. Sous Windows, `--workers` du DataLoader vaut 0 par défaut.

L'export ONNX demande un paquet supplémentaire :

```bash
pip install onnx
```

## Essai sur données synthétiques

Le jeu synthétique est du bruit avec des rectangles colorés, 8 images d'entraînement. La commande vérifie seulement que l'entraînement, la validation et l'écriture du point de contrôle se terminent. Après une époque, la mAP50 de validation peut valoir 0.

```bash
python -m yolov28.train --demo --epochs 1 --batch 4 --imgsz 320
```

Les poids sont écrits dans `runs/train/exp/weights/last.pt` et `best.pt`. Si le dossier existe déjà, l'exécution suivante prend `exp2`, puis `exp3`.

## Entraîner vos données

L'arborescence suit le format texte YOLO habituel :

```text
datasets/custom/
  images/train/a.jpg
  labels/train/a.txt
  images/val/b.jpg
  labels/val/b.txt
```

Chaque ligne d'étiquette est `classe xc yc w h`. Les quatre valeurs de boîte sont normalisées par la largeur et la hauteur de l'image d'origine. Voir `configs/dataset.example.yaml`.

```bash
python -m yolov28.train --data configs/dataset.example.yaml --scale n --epochs 100 --batch 8 --imgsz 640
```

L'augmentation de cette boucle est un retournement horizontal aléatoire plus un letterbox. Mosaic n'en fait pas partie. Le taux d'apprentissage vaut `0.001`, le weight decay `0.0005`, et les gradients sont coupés à 10. Au-delà d'une époque, les 3 premières montent linéairement, puis une décroissance en cosinus prend le relais. Une seule époque utilise directement le taux de base. Le gain de boîte est 2.0, le gain de classification 0.5, et la branche un-vers-un est multipliée par 1.0. S'il existe un jeu de validation, `best.pt` est choisi selon la mAP50 de validation.

`--scale` accepte `n`, `s`, `m`, `l` ou `x`.

## Prédiction

```bash
python -m yolov28.predict --weights runs/train/exp/weights/last.pt --source image.jpg --conf 0.25
```

`--source` peut être une image ou un dossier. Les dessins sont enregistrés sous `runs/predict/exp/` sur l'image d'origine. Les boîtes renvoyées sont en pixels de cette image.

Tête un-vers-un :

```bash
python -m yolov28.predict --weights runs/train/exp/weights/last.pt --source image.jpg --end2end
```

Un point de contrôle initialisé au hasard, ou entraîné pendant quelques pas seulement, produit encore beaucoup de boîtes de faible qualité. Montez `--conf` pour en masquer davantage.

## Export ONNX

```bash
python -m yolov28.export --weights runs/train/exp/weights/last.pt --imgsz 640 --output yolov28n.onnx
```

L'entrée exportée est `1×3×imgsz×imgsz`. La sortie est `(1, anchors, 4+classes)` : les quatre premiers nombres sont les xyxy sur le canevas du réseau, suivis des scores de classe après sigmoïde. Le NMS reste hors du graphe. La taille de lot est fixe à 1. `--end2end` sélectionne la tête un-vers-un ; le graphe ne contient toujours pas de NMS.

## API Python

```python
from yolov28 import YOLO

model = YOLO("n", nc=4)
model.train(demo=True, epochs=1, batch=4, imgsz=320, name="demo")
model.predict("image.jpg", imgsz=320, conf=0.25)
```

`YOLO("runs/train/demo/weights/last.pt")` charge un point de contrôle écrit par ce dépôt.

## Échelles

Les nombres de paramètres ci-dessous valent pour 80 classes et viennent de `python -m yolov28.summary`. Changer le nombre de classes change la tête de classification, donc le nombre de paramètres.

| Échelle | Profondeur | Largeur | Paramètres à 80 classes |
| --- | ---: | ---: | ---: |
| n | 0.33 | 0.25 | 1,789,192 |
| s | 0.33 | 0.50 | 6,163,224 |
| m | 0.67 | 0.75 | 14,573,736 |
| l | 1.00 | 1.00 | 27,206,712 |
| x | 1.33 | 1.25 | 44,564,424 |

## Métrique de validation

La mAP50 du journal est la précision moyenne à un IoU de 0.5. Les classes sans vérité terrain dans le jeu de validation sont exclues de la moyenne. La mAP de style COCO moyenne les seuils d'IoU de 0.50 à 0.95. Le nombre affiché ici est le résultat à un seul seuil.

## Tests

```bash
python tests/test_smoke.py
```

Le test couvre la géométrie des étiquettes, l'assignation des positifs, la rétropropagation, un court surapprentissage, puis l'entraînement et la prédiction sur un dossier d'images réelles. Si `onnx` est installé, il vérifie aussi qu'un fichier d'export est écrit.

## Licence

[MIT](LICENSE)
