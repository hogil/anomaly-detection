"""Benchmark train/val epoch time for different DataLoader num_workers values.

Uses the same dataset, transforms, model, and batch size as the main training
loop, but only runs a few train+val epochs and reports wall-clock timings.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import yaml
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train  # noqa: E402
from src.models.focal_loss import FocalLoss  # noqa: E402


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_frames(data_dir: Path, normal_ratio: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    sc_df = pd.read_csv(data_dir / "scenarios.csv")
    train_df = sc_df[sc_df["split"] == "train"].copy()
    val_df = sc_df[sc_df["split"] == "val"].copy()

    normal_df = train_df[train_df["class"] == "normal"]
    abnormal_df = train_df[train_df["class"] != "normal"]
    if normal_ratio < len(normal_df):
        normal_df = normal_df.sample(n=normal_ratio, random_state=42)
    elif normal_ratio > len(normal_df):
        extra = normal_df.sample(n=normal_ratio - len(normal_df), random_state=42, replace=True)
        normal_df = pd.concat([normal_df, extra])
    train_df = pd.concat([normal_df, abnormal_df]).reset_index(drop=True)
    return train_df, val_df


def build_transforms(model_name: str):
    tmp_model = timm.create_model(model_name, pretrained=False)
    data_cfg = timm.data.resolve_model_data_config(tmp_model)
    del tmp_model

    input_mean = list(data_cfg.get("mean", (0.485, 0.456, 0.406)))
    input_std = list(data_cfg.get("std", (0.229, 0.224, 0.225)))
    input_size = data_cfg.get("input_size", (3, 224, 224))
    img_hw = (input_size[1], input_size[2])

    train_transform = transforms.Compose([
        transforms.Resize(img_hw),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        transforms.ToTensor(),
        transforms.Normalize(input_mean, input_std),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(img_hw),
        transforms.ToTensor(),
        transforms.Normalize(input_mean, input_std),
    ])
    return train_transform, val_transform


def run_single_benchmark(
    *,
    config: dict,
    model_name: str,
    batch_size: int,
    normal_ratio: int,
    seed: int,
    num_workers: int,
    prefetch_factor: int,
    epochs: int,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float,
    dropout: float,
    stochastic_depth_rate: float,
    focal_gamma: float,
) -> dict:
    seed_all(seed)
    train._GLOBAL_WORKER_SEED = seed

    data_dir = ROOT / config["output"]["data_dir"]
    image_dir = ROOT / config["output"]["image_dir"]
    train_df, val_df = build_frames(data_dir, normal_ratio)
    train_transform, val_transform = build_transforms(model_name)

    train_ds = train.ChartImageDataset(image_dir, train_df, config["dataset"]["classes"], train_transform, mode="binary")
    val_ds = train.ChartImageDataset(image_dir, val_df, config["dataset"]["classes"], val_transform, mode="binary")

    common = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=train._worker_init_fn if num_workers > 0 else None,
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.float16 if device.type == "cuda" else None
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    model = train.create_model(
        2,
        model_name,
        device,
        dropout=dropout,
        stochastic_depth_rate=stochastic_depth_rate,
    )
    backbone_params = [p for n, p in model.named_parameters() if "head" not in n]
    head_params = [p for n, p in model.named_parameters() if "head" in n]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ], weight_decay=weight_decay)

    n_normal = int((train_df["class"] == "normal").sum())
    n_abnormal = int((train_df["class"] != "normal").sum())
    total = n_normal + n_abnormal
    alpha = [n_abnormal / total * 2, n_normal / total * 2]
    criterion = FocalLoss(alpha=alpha, gamma=focal_gamma).to(device)

    epoch_rows = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc, _, _ = train.train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            epochs,
            scaler=scaler,
            use_mixup=False,
            ohem_ratio=0.0,
            amp_dtype=amp_dtype,
            ema=None,
            grad_clip=1.0,
        )
        _, val_acc, val_recall, val_f1, _, _, _ = train.evaluate(
            model,
            val_loader,
            criterion,
            device,
            ["normal", "abnormal"],
            desc=f"Bench nw={num_workers} ep={epoch}",
            amp_dtype=amp_dtype,
        )
        elapsed = time.time() - t0
        row = {
            "epoch": epoch,
            "elapsed_sec": round(elapsed, 2),
            "train_steps": len(train_loader),
            "val_steps": len(val_loader),
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
            "val_recall": round(val_recall, 4),
            "val_f1": round(val_f1, 4),
            "train_it_per_sec": round(len(train_loader) / elapsed, 4),
        }
        epoch_rows.append(row)
        print(
            f"BENCH_EPOCH nw={num_workers} epoch={epoch} elapsed={elapsed:.2f}s "
            f"train_it/s={len(train_loader) / elapsed:.4f} val_f1={val_f1:.4f}"
        )

    result = {
        "num_workers": num_workers,
        "device": str(device),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "epochs": epoch_rows,
        "epoch2_elapsed_sec": epoch_rows[-1]["elapsed_sec"],
        "epoch2_train_it_per_sec": epoch_rows[-1]["train_it_per_sec"],
    }

    del model, optimizer, criterion, train_loader, val_loader, train_ds, val_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dataset.yaml")
    parser.add_argument("--model_name", default="convnextv2_tiny.fcmae_ft_in22k_in1k")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--normal_ratio", type=int, default=1400)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--num_workers", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--lr_backbone", type=float, default=2e-5)
    parser.add_argument("--lr_head", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--stochastic_depth_rate", type=float, default=0.0)
    parser.add_argument("--focal_gamma", type=float, default=0.0)
    parser.add_argument(
        "--output",
        default="logs/bench_num_workers_n1400_seed2.json",
        help="Path to write JSON results.",
    )
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    results = []
    for num_workers in args.num_workers:
        print(f"BENCH_START nw={num_workers}")
        result = run_single_benchmark(
            config=config,
            model_name=args.model_name,
            batch_size=args.batch_size,
            normal_ratio=args.normal_ratio,
            seed=args.seed,
            num_workers=num_workers,
            prefetch_factor=args.prefetch_factor,
            epochs=args.epochs,
            lr_backbone=args.lr_backbone,
            lr_head=args.lr_head,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            stochastic_depth_rate=args.stochastic_depth_rate,
            focal_gamma=args.focal_gamma,
        )
        results.append(result)

    out_path = ROOT / args.output
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SAVED {out_path}")
    for result in results:
        print(
            f"BENCH_SUMMARY nw={result['num_workers']} "
            f"epoch2_elapsed={result['epoch2_elapsed_sec']:.2f}s "
            f"epoch2_train_it/s={result['epoch2_train_it_per_sec']:.4f}"
        )


if __name__ == "__main__":
    main()
