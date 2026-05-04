#!/usr/bin/env python
"""Fine-tune an existing best model from normal/abnormal image folders."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import _load_model_from_best_info

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
TQDM_DISABLE = not sys.stderr.isatty()


@dataclass(frozen=True)
class ImageItem:
    path: Path
    label: int
    class_name: str


class FolderFineTuneDataset(Dataset):
    def __init__(self, items: list[ImageItem], transform) -> None:
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        image = Image.open(item.path).convert("RGB")
        return self.transform(image), item.label, str(item.path)


def collect_items(image_root: Path, classes: list[str], limit_per_class: int, seed: int) -> list[ImageItem]:
    if "normal" not in classes or "abnormal" not in classes:
        raise SystemExit(f"add-training requires binary classes with normal/abnormal, got: {classes}")

    rng = random.Random(seed)
    items: list[ImageItem] = []
    for class_name in ["normal", "abnormal"]:
        class_dir = image_root / class_name
        if not class_dir.exists():
            raise SystemExit(f"missing class folder: {class_dir}")
        paths = sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        rng.shuffle(paths)
        if limit_per_class > 0:
            paths = paths[:limit_per_class]
        if not paths:
            raise SystemExit(f"no images found under: {class_dir}")
        label = classes.index(class_name)
        items.extend(ImageItem(path=p, label=label, class_name=class_name) for p in paths)
    return items


def stratified_split(items: list[ImageItem], val_ratio: float, seed: int) -> tuple[list[ImageItem], list[ImageItem]]:
    rng = random.Random(seed)
    by_class: dict[str, list[ImageItem]] = {}
    for item in items:
        by_class.setdefault(item.class_name, []).append(item)

    train_items: list[ImageItem] = []
    val_items: list[ImageItem] = []
    for class_items in by_class.values():
        class_items = list(class_items)
        rng.shuffle(class_items)
        if len(class_items) < 2 or val_ratio <= 0:
            train_items.extend(class_items)
            continue
        val_count = max(1, int(round(len(class_items) * val_ratio)))
        val_count = min(val_count, len(class_items) - 1)
        val_items.extend(class_items[:val_count])
        train_items.extend(class_items[val_count:])

    if not train_items:
        raise SystemExit("empty train split")
    if not val_items:
        val_items = list(train_items)
    rng.shuffle(train_items)
    rng.shuffle(val_items)
    return train_items, val_items


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def maybe_freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        is_head = any(part in name.lower() for part in ["head", "classifier", "fc"])
        param.requires_grad = is_head


def build_optimizer(model: nn.Module, lr: float, head_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    if head_lr <= 0 or abs(head_lr - lr) < 1e-15:
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    head_params = []
    body_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(part in name.lower() for part in ["head", "classifier", "fc"]):
            head_params.append(param)
        else:
            body_params.append(param)
    groups = []
    if body_params:
        groups.append({"params": body_params, "lr": lr})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    epochs: int,
    warmup_epochs: int,
    step_size: int,
    step_gamma: float,
):
    if scheduler_name == "none":
        return None, None

    warmup = None
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_epochs,
        )

    if scheduler_name == "cosine":
        main_epochs = max(1, epochs - warmup_epochs)
        main = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=main_epochs, eta_min=1e-7)
    elif scheduler_name == "step":
        main = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=step_gamma)
    else:
        raise SystemExit(f"unknown scheduler: {scheduler_name}")

    if warmup is None:
        return main, None
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, main],
        milestones=[warmup_epochs],
    ), None


def confusion_matrix_np(labels: np.ndarray, preds: np.ndarray, n_classes: int) -> np.ndarray:
    idx = labels.astype(np.int64) * n_classes + preds.astype(np.int64)
    counts = np.bincount(idx, minlength=n_classes * n_classes)
    return counts.reshape(n_classes, n_classes)


def class_metrics(labels: np.ndarray, preds: np.ndarray, classes: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for i, class_name in enumerate(classes):
        mask = labels == i
        tp = int(((preds == i) & (labels == i)).sum())
        fp = int(((preds == i) & (labels != i)).sum())
        fn = int(((preds != i) & (labels == i)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[class_name] = {
            "count": int(mask.sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return out


def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device, classes: list[str]) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_labels: list[int] = []
    all_preds: list[int] = []
    with torch.no_grad():
        for images, labels, _paths in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * int(labels.numel())
            total += int(labels.numel())
            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    labels_np = np.asarray(all_labels, dtype=np.int64)
    preds_np = np.asarray(all_preds, dtype=np.int64)
    metrics = class_metrics(labels_np, preds_np, classes)
    avg_f1 = float(np.mean([m["f1"] for m in metrics.values()])) if metrics else 0.0
    acc = float((labels_np == preds_np).mean()) if len(labels_np) else 0.0
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": acc,
        "f1": avg_f1,
        "metrics": metrics,
        "labels": labels_np,
        "preds": preds_np,
    }


def train_one_epoch(model, loader, criterion, optimizer, device, amp: bool, desc: str) -> float:
    model.train()
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    total_loss = 0.0
    total = 0
    for images, labels, _paths in tqdm(loader, desc=desc, ncols=100, disable=TQDM_DISABLE):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * int(labels.numel())
        total += int(labels.numel())
    return total_loss / max(total, 1)


def save_confusion_matrix(labels: np.ndarray, preds: np.ndarray, classes: list[str], out_path: Path) -> None:
    mat = confusion_matrix_np(labels, preds, len(classes))
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Add-Training Validation Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_manifest(items: list[ImageItem], split: str, writer: csv.DictWriter) -> None:
    for item in items:
        writer.writerow({"split": split, "class": item.class_name, "label": item.label, "path": str(item.path)})


def load_parent_hparams(model_run: Path) -> dict[str, Any]:
    info_path = model_run / "best_info.json"
    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    return dict(info.get("hparams", {}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run", required=True, type=Path, help="logs/<run> containing best_model.pth/best_info.json")
    parser.add_argument("--image-root", required=True, type=Path, help="folder containing normal/ and abnormal/")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: logs/addtrain_<timestamp>")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5, help="fine-tune LR for loaded weights")
    parser.add_argument("--head-lr", type=float, default=0.0, help="optional separate head LR")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--scheduler", choices=["cosine", "step", "none"], default="cosine")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--step-size", type=int, default=3)
    parser.add_argument("--step-gamma", type=float, default=0.5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-per-class", type=int, default=0)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    args.model_run = args.model_run.resolve()
    if not (args.model_run / "best_model.pth").exists():
        raise SystemExit(f"best_model.pth not found under model run: {args.model_run}")

    out_dir = args.out_dir or (Path("logs") / f"addtrain_{datetime.now().strftime('%y%m%d_%H%M%S')}")
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"output already exists, pass --overwrite: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, classes = _load_model_from_best_info(args.model_run, device)
    if args.freeze_backbone:
        maybe_freeze_backbone(model)

    items = collect_items(args.image_root, classes, args.limit_per_class, args.seed)
    train_items, val_items = stratified_split(items, args.val_ratio, args.seed)
    transform = build_transform()
    train_ds = FolderFineTuneDataset(train_items, transform)
    val_ds = FolderFineTuneDataset(val_items, transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, args.lr, args.head_lr, args.weight_decay)
    scheduler, _ = build_scheduler(
        optimizer,
        args.scheduler,
        args.epochs,
        args.warmup_epochs,
        args.step_size,
        args.step_gamma,
    )
    amp = device.type == "cuda"

    print(f"Device: {device}")
    print(f"Images: train={len(train_items)}, val={len(val_items)}, classes={classes}")
    print(f"Fine-tune: lr={args.lr:g}, head_lr={args.head_lr:g}, scheduler={args.scheduler}, epochs={args.epochs}")

    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_payload: dict[str, Any] | None = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            amp,
            desc=f"addtrain {epoch}/{args.epochs}",
        )
        val = evaluate(model, val_loader, criterion, device, classes)
        lrs = [group["lr"] for group in optimizer.param_groups]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_acc": val["accuracy"],
            "val_f1": val["f1"],
            "lr": lrs,
            "metrics": val["metrics"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.4f} "
            f"val_loss={val['loss']:.4f} val_acc={val['accuracy']:.4f} val_f1={val['f1']:.4f}"
        )

        if val["f1"] > best_f1:
            best_f1 = float(val["f1"])
            torch.save(model.state_dict(), out_dir / "best_model.pth")
            save_confusion_matrix(val["labels"], val["preds"], classes, out_dir / "confusion_matrix.png")
            best_payload = {
                "epoch": epoch,
                "val_loss": val["loss"],
                "val_acc": val["accuracy"],
                "val_f1": val["f1"],
                "metrics": val["metrics"],
            }

        if scheduler is not None:
            scheduler.step()

        with (out_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

    parent_hparams = load_parent_hparams(args.model_run)
    hparams = dict(parent_hparams)
    hparams.update(
        {
            "classes": classes,
            "num_classes": len(classes),
            "add_training": True,
            "parent_model_run": str(args.model_run),
            "image_root": str(args.image_root),
            "epochs": args.epochs,
            "lr": args.lr,
            "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "warmup_epochs": args.warmup_epochs,
            "freeze_backbone": args.freeze_backbone,
        }
    )
    best_info = dict(best_payload or {})
    best_info["hparams"] = hparams
    with (out_dir / "best_info.json").open("w", encoding="utf-8") as handle:
        json.dump(best_info, handle, indent=2)

    with (out_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "class", "label", "path"])
        writer.writeheader()
        write_manifest(train_items, "train", writer)
        write_manifest(val_items, "val", writer)

    print(f"Saved add-training run: {out_dir}")
    print(f"  best_f1={best_f1:.4f}")
    print(f"  model={out_dir / 'best_model.pth'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
