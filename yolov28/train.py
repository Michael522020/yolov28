"""Training loop for the unofficial YOLOv28 detector."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from yolov28.data import (
    DetectionDataset,
    SyntheticDetection,
    collate,
    demo_names,
    load_data_config,
)
from yolov28.ema import ModelEMA
from yolov28.loss import DetectionLoss
from yolov28.metrics import evaluate
from yolov28.model import YOLOv28, count_parameters, load_checkpoint

VERSION = "0.2.0"


@dataclass
class TrainConfig:
    data: str = ""
    scale: str = "n"
    epochs: int = 100
    batch: int = 8
    imgsz: int = 640
    lr: float = 1e-3
    weight_decay: float = 5e-4
    device: str = ""
    workers: int = 0
    project: str = "runs/train"
    name: str = "exp"
    demo: bool = False
    seed: int = 0
    log_every: int = 20
    nc: int | None = None
    weights: str = ""
    mosaic: bool = True


def select_device(name: str) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def learning_rate(epoch: int, epochs: int, base: float, warmup: int = 3) -> float:
    if epochs <= 1:
        return base
    warmup = min(warmup, epochs)
    if epoch < warmup:
        return base * float(epoch + 1) / float(warmup)
    progress = (epoch - warmup) / max(epochs - warmup, 1)
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


def next_run_dir(project: str, name: str) -> Path:
    root = Path(project)
    candidate = root / name
    if not candidate.exists():
        return candidate
    index = 2
    while (root / f"{name}{index}").exists():
        index += 1
    return root / f"{name}{index}"


def mosaic_this_epoch(epoch: int, epochs: int, enabled: bool) -> bool:
    """Keep mosaic on for short runs. On longer runs, drop it for the last 10%."""
    if not enabled:
        return False
    if epochs < 10:
        return True
    return epoch < epochs - max(epochs // 10, 1)


def save_checkpoint(path: Path, model: YOLOv28, names: dict, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "nc": model.nc,
            "scale": model.scale,
            "names": names,
            "epoch": epoch,
            "version": VERSION,
        },
        path,
    )


def train_model(cfg: TrainConfig, model: YOLOv28 | None = None) -> dict:
    if cfg.demo and cfg.data:
        raise ValueError("choose either demo=True or a data yaml, not both")
    if cfg.imgsz % 32 != 0:
        raise ValueError("imgsz must be a multiple of 32")
    if cfg.epochs < 1 or cfg.batch < 1:
        raise ValueError("epochs and batch must be positive")

    seed_everything(cfg.seed)
    device = select_device(cfg.device)

    if cfg.demo:
        nc = cfg.nc if cfg.nc is not None else (model.nc if model is not None else 4)
        names = demo_names(nc)
        train_set = SyntheticDetection(
            length=max(8, cfg.batch), imgsz=cfg.imgsz, nc=nc, boxes=3, seed=cfg.seed
        )
        val_set = SyntheticDetection(
            length=max(4, cfg.batch // 2),
            imgsz=cfg.imgsz,
            nc=nc,
            boxes=3,
            seed=cfg.seed + 1000,
        )
    else:
        if not cfg.data:
            raise ValueError("pass --data path/to/dataset.yaml, or pass --demo")
        root, train_split, val_split, names, nc = load_data_config(cfg.data)
        if cfg.nc is not None and cfg.nc != nc:
            raise ValueError(f"--nc {cfg.nc} does not match the dataset nc {nc}")
        train_set = DetectionDataset(
            root, train_split, cfg.imgsz, augment=True, nc=nc, mosaic=cfg.mosaic
        )
        val_set = (
            DetectionDataset(root, val_split, cfg.imgsz, augment=False, nc=nc, mosaic=False)
            if val_split
            else None
        )

    if model is None:
        model = YOLOv28(nc=nc, scale=cfg.scale)
        if cfg.weights:
            loaded, _, _ = load_checkpoint(cfg.weights)
            if loaded.nc != model.nc or loaded.scale != model.scale:
                raise ValueError(
                    f"checkpoint is scale={loaded.scale} nc={loaded.nc}, "
                    f"training is scale={model.scale} nc={model.nc}"
                )
            model.load_state_dict(loaded.state_dict())
    elif model.nc != nc:
        raise ValueError(f"model nc={model.nc} does not match dataset nc={nc}")

    model.to(device)
    criterion = DetectionLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ema = ModelEMA(model)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    def make_train_loader() -> DataLoader:
        return DataLoader(
            train_set,
            batch_size=cfg.batch,
            shuffle=True,
            num_workers=cfg.workers,
            collate_fn=collate,
            generator=generator,
        )

    val_loader = (
        None
        if val_set is None
        else DataLoader(
            val_set,
            batch_size=cfg.batch,
            shuffle=False,
            num_workers=cfg.workers,
            collate_fn=collate,
        )
    )

    save_dir = next_run_dir(cfg.project, cfg.name)
    weight_dir = save_dir / "weights"
    weight_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "results.csv"
    log_path.write_text(
        "epoch,train_loss,val_loss,map50,map50_95,map50_e2e\n", encoding="utf-8"
    )

    print(
        f"YOLOv28-{model.scale} {VERSION}  nc={model.nc}  params={count_parameters(model):,}  "
        f"device={device}  imgsz={cfg.imgsz}  mosaic={cfg.mosaic and not cfg.demo}"
    )
    print("Unofficial random initialization. This checkpoint is not a COCO pretrained model.")
    print(f"run directory: {save_dir}")

    best_score = float("-inf")
    best_path = weight_dir / "best.pt"
    last_path = weight_dir / "last.pt"

    for epoch in range(cfg.epochs):
        if not cfg.demo:
            train_set.mosaic = mosaic_this_epoch(epoch, cfg.epochs, cfg.mosaic)
        lr = learning_rate(epoch, cfg.epochs, cfg.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        train_loader = make_train_loader()
        running = 0.0
        seen = 0
        for step, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device)
            targets = [target.to(device) for target in targets]
            optimizer.zero_grad(set_to_none=True)
            loss, parts = criterion(model(images), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            ema.update(model)
            batch = images.shape[0]
            running += float(loss.detach()) * batch
            seen += batch
            if step == 1 or step % cfg.log_every == 0:
                print(
                    f"epoch {epoch + 1}/{cfg.epochs} step {step} "
                    f"loss {float(loss.detach()):.4f} box {float(parts['box']):.4f} "
                    f"cls {float(parts['cls']):.4f} lr {lr:.6f}"
                )
        train_loss = running / max(seen, 1)
        val_loss = ""
        map50 = ""
        map50_95 = ""
        map50_e2e = ""
        score = -train_loss
        if val_loader is not None:
            stats = evaluate(ema.ema, val_loader, criterion, device)
            val_loss = f"{stats['loss']:.6f}"
            map50 = f"{stats['map50']:.6f}"
            map50_95 = f"{stats['map50_95']:.6f}"
            map50_e2e = f"{stats['map50_e2e']:.6f}"
            score = stats["map50_95"]
            print(
                f"epoch {epoch + 1}/{cfg.epochs} train {train_loss:.4f} "
                f"val {stats['loss']:.4f} mAP50 {stats['map50']:.4f} "
                f"mAP50-95 {stats['map50_95']:.4f} mAP50-e2e {stats['map50_e2e']:.4f}"
            )
        else:
            print(f"epoch {epoch + 1}/{cfg.epochs} train {train_loss:.4f}")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{epoch + 1},{train_loss:.6f},{val_loss},{map50},{map50_95},{map50_e2e}\n"
            )
        save_checkpoint(last_path, ema.ema, names, epoch + 1)
        if score > best_score:
            best_score = score
            save_checkpoint(best_path, ema.ema, names, epoch + 1)

    return {
        "last": last_path,
        "best": best_path,
        "names": names,
        "save_dir": save_dir,
        "model": model,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the unofficial YOLOv28 detector. Not an Ultralytics release."
    )
    parser.add_argument("--data", default="", help="dataset yaml, see configs/dataset.example.yaml")
    parser.add_argument("--scale", default="n", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", default="", help="cpu, cuda, or cuda:0. Default picks cuda when present.")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="exp")
    parser.add_argument("--demo", action="store_true", help="train on synthetic rectangles instead of a dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--nc", type=int, default=None, help="class count for --demo; dataset yaml sets this otherwise")
    parser.add_argument("--weights", default="", help="optional checkpoint to continue from")
    parser.add_argument("--no-mosaic", action="store_true", help="disable mosaic on a real dataset")
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        data=args.data,
        scale=args.scale,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        demo=args.demo,
        seed=args.seed,
        log_every=args.log_every,
        nc=args.nc,
        weights=args.weights,
        mosaic=not args.no_mosaic,
    )


def main() -> None:
    train_model(config_from_args(build_parser().parse_args()))


if __name__ == "__main__":
    main()
