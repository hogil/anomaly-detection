"""
Generate a small mean-shift-only preview dataset and render images.

Usage:
    python scripts/generate_mean_shift_preview.py --config dataset.yaml --count 50 --workers 0
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_data import generate  # noqa: E402
from generate_images import render_all  # noqa: E402


def build_preview_config(base_cfg: dict, count: int, tag: str) -> dict:
    cfg = copy.deepcopy(base_cfg)
    preview_root = ROOT / "preview" / tag

    cfg["dataset"]["version"] = f"{base_cfg['dataset'].get('version', 'dataset')}_mean_shift_preview"
    cfg["dataset"]["classes"] = ["mean_shift"]
    cfg["dataset"]["samples_per_class"] = {"mean_shift": int(count)}
    cfg["dataset"]["split"] = {
        "train": 0.0,
        "val": 0.0,
        "test": 1.0,
    }

    cfg["output"]["data_dir"] = str(preview_root / "data")
    cfg["output"]["image_dir"] = str(preview_root / "images")
    cfg["output"]["display_dir"] = str(preview_root / "display")
    return cfg


def build_preview_config_with_split(
    base_cfg: dict,
    train_count: int,
    test_count: int,
    test_difficulty: float,
    tag: str,
    include_normal: bool = False,
) -> dict:
    total = int(train_count) + int(test_count)
    cfg = copy.deepcopy(base_cfg)
    preview_root = ROOT / "preview" / tag

    cfg["dataset"]["version"] = f"{base_cfg['dataset'].get('version', 'dataset')}_mean_shift_preview"
    preview_classes = ["mean_shift", "normal"] if include_normal else ["mean_shift"]
    cfg["dataset"]["classes"] = preview_classes
    cfg["dataset"]["samples_per_class"] = {cls: total for cls in preview_classes}
    cfg["dataset"]["split"] = {
        "train": float(train_count) / total if total else 0.0,
        "val": 0.0,
        "test": float(test_count) / total if total else 0.0,
    }
    cfg["dataset"]["test_difficulty_scale"] = float(test_difficulty)

    cfg["output"]["data_dir"] = str(preview_root / "data")
    cfg["output"]["image_dir"] = str(preview_root / "images")
    cfg["output"]["display_dir"] = str(preview_root / "display")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dataset.yaml")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--test-count", type=int, default=None)
    parser.add_argument("--test-difficulty", type=float, default=None)
    parser.add_argument("--include-normal", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    with open(ROOT / args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    tag = args.tag or f"mean_shift_preview_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.train_count is not None or args.test_count is not None:
        train_count = int(args.train_count or 0)
        test_count = int(args.test_count or 0)
        if train_count <= 0 or test_count <= 0:
            raise ValueError("--train-count and --test-count must both be > 0 when used")
        test_difficulty = (
            float(args.test_difficulty)
            if args.test_difficulty is not None
            else float(base_cfg["dataset"].get("test_difficulty_scale", 1.0))
        )
        cfg = build_preview_config_with_split(
            base_cfg,
            train_count=train_count,
            test_count=test_count,
            test_difficulty=test_difficulty,
            tag=tag,
            include_normal=args.include_normal,
        )
    else:
        cfg = build_preview_config(base_cfg, count=args.count, tag=tag)

    preview_root = ROOT / "preview" / tag
    preview_root.mkdir(parents=True, exist_ok=True)

    preview_cfg_path = preview_root / "preview_config.yaml"
    with open(preview_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print(f"[preview] config: {preview_cfg_path}")
    generate(cfg, num_workers=args.workers)
    render_all(cfg, num_workers=args.workers)

    print(f"[preview] done: {preview_root}")
    print(f"[preview] data: {preview_root / 'data'}")
    print(f"[preview] display: {preview_root / 'display'}")


if __name__ == "__main__":
    main()
