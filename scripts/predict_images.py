#!/usr/bin/env python
"""Predict already-rendered images with a saved best_model.pth.

The model argument may be either a run directory or a direct best_model.pth
path. The parent directory must contain best_info.json.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import _load_model_from_best_info, load_prob_limits, resolve_prob_limit  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        "--model-run",
        dest="model_run",
        required=True,
        help="Run directory containing best_model.pth/best_info.json, or direct best_model.pth path.",
    )
    parser.add_argument("--image-root", type=Path, default=None, help="Directory of images to predict recursively.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional manifest CSV from image generation.")
    parser.add_argument("--image-col", default="model_input", help="Manifest column containing image paths.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for predictions.csv and copied images.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16",
                        help="예측 정밀도 (CUDA만 적용, CPU는 fp32). train.py eval 기본과 같은 fp16 권장.")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile 활성화 (대량 추론 + Linux GPU 서버에서만 이득, warmup 수십 초).")
    parser.add_argument(
        "--normal-threshold",
        type=float,
        default=None,
        help="Binary gate override. normal iff p_normal > threshold; otherwise abnormal.",
    )
    parser.add_argument(
        "--prob-limit-csv",
        default=None,
        help="차트 종류별 판정 한계 CSV (컬럼: device/step/item 일부 + prob_limit, 빈칸/*=wildcard). "
             "manifest 행에 매칭되면 p_abnormal >= prob_limit → abnormal 로 판정 (normal-threshold보다 우선).",
    )
    parser.add_argument("--no-copy", action="store_true", help="Do not copy images into predictions/<class>/ folders.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output-dir when it already exists.")
    return parser.parse_args()


def resolve_run(path_text: str) -> Path:
    path = Path(path_text)
    run_dir = path.parent if path.is_file() else path
    if not (run_dir / "best_model.pth").exists():
        raise FileNotFoundError(f"best_model.pth not found under run dir: {run_dir}")
    if not (run_dir / "best_info.json").exists():
        raise FileNotFoundError(f"best_info.json not found under run dir: {run_dir}")
    return run_dir


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def read_default_threshold(run_dir: Path, classes: list[str]) -> float | None:
    if "normal" not in classes or "abnormal" not in classes:
        return None
    with (run_dir / "best_info.json").open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    selected = info.get("selected_normal_threshold")
    if selected is not None:
        return float(selected)
    hparams = info.get("hparams") or {}
    if hparams.get("normal_threshold") is not None:
        return float(hparams["normal_threshold"])
    return 0.5


def sanitize_part(value: object) -> str:
    text = str(value).strip()
    safe = [ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text]
    cleaned = "".join(safe).strip("._-")
    return cleaned or "image"


def resolve_manifest_path(raw: object, manifest_path: Path, image_root: Path | None) -> Path:
    text = str(raw).strip()
    path = Path(text)
    if path.is_absolute():
        return path
    candidates = [manifest_path.parent / path, ROOT / path]
    if image_root is not None:
        candidates.append(image_root / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def collect_from_manifest(manifest_path: Path, image_col: str, image_root: Path | None) -> list[dict[str, Any]]:
    df = pd.read_csv(manifest_path)
    if image_col not in df.columns:
        raise ValueError(f"manifest column not found: {image_col}")
    rows = []
    for idx, row in df.iterrows():
        image_path = resolve_manifest_path(row[image_col], manifest_path, image_root)
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        if not image_path.exists():
            raise FileNotFoundError(f"image from manifest not found: {image_path}")
        meta = row.to_dict()
        meta["_manifest_row"] = int(idx)
        meta["_image_path"] = image_path
        rows.append(meta)
    return rows


def collect_from_root(image_root: Path) -> list[dict[str, Any]]:
    rows = []
    for image_path in sorted(image_root.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
            rows.append({"_image_path": image_path})
    return rows


def decide_prediction(probs: torch.Tensor, classes: list[str], normal_threshold: float | None) -> str:
    if normal_threshold is not None and "normal" in classes and "abnormal" in classes:
        p_normal = float(probs[classes.index("normal")].item())
        return "normal" if p_normal > normal_threshold else "abnormal"
    pred_idx = int(torch.argmax(probs).item())
    return classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)


def unique_dest(dest_dir: Path, source: Path, index: int) -> Path:
    name = sanitize_part(source.name)
    candidate = dest_dir / name
    if not candidate.exists():
        return candidate
    return dest_dir / f"{index:06d}_{name}"


def main() -> int:
    args = parse_args()
    if args.manifest is None and args.image_root is None:
        raise SystemExit("pass --image-root or --manifest")

    run_dir = resolve_run(args.model_run)
    device = resolve_device(args.device)
    model, classes = _load_model_from_best_info(run_dir, device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        if hasattr(torch, "compile") and device.type == "cuda":
            try:
                model = torch.compile(model)
            except Exception as exc:
                print(f"[predict-images] torch.compile failed, falling back: {exc}")
        else:
            print("[predict-images] torch.compile requested but unsupported (cpu or old torch)")
    if device.type == "cuda" and args.precision == "fp16":
        amp_dtype = torch.float16
    elif device.type == "cuda" and args.precision == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = None
    normal_threshold = args.normal_threshold
    if normal_threshold is None:
        normal_threshold = read_default_threshold(run_dir, classes)

    prob_limit_rules = None
    if args.prob_limit_csv:
        prob_limit_rules = load_prob_limits(args.prob_limit_csv, ["device", "step", "item"])
        if "normal" not in classes or "abnormal" not in classes:
            print("[predict-images] prob_limit CSV는 binary(normal/abnormal) 모델에서만 적용됩니다 — 무시함")
            prob_limit_rules = None

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output-dir already exists: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    if args.manifest is not None:
        rows = collect_from_manifest(args.manifest, args.image_col, args.image_root)
    else:
        rows = collect_from_root(args.image_root)
    if not rows:
        raise SystemExit("no images found")

    pred_root = args.output_dir / "predictions"
    if not args.no_copy:
        for cls in classes:
            (pred_root / sanitize_part(cls)).mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    results: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(rows), args.batch_size), desc="predict-images"):
        batch_rows = rows[start:start + args.batch_size]
        tensors = []
        for row in batch_rows:
            img = Image.open(row["_image_path"]).convert("RGB")
            tensors.append(transform(img))
        x = torch.stack(tensors).to(device)
        if device.type == "cuda":
            x = x.to(memory_format=torch.channels_last)
        amp_ctx = (torch.amp.autocast("cuda", dtype=amp_dtype)
                   if amp_dtype is not None else contextlib.nullcontext())
        with torch.no_grad(), amp_ctx:
            probs_batch = F.softmax(model(x), dim=1).float().detach().cpu()

        for offset, (row, probs) in enumerate(zip(batch_rows, probs_batch)):
            image_path: Path = row["_image_path"]
            prob_limit = resolve_prob_limit(row, prob_limit_rules) if prob_limit_rules else None
            if prob_limit is not None:
                p_abn = float(probs[classes.index("abnormal")].item())
                pred = "abnormal" if p_abn >= prob_limit else "normal"
            else:
                pred = decide_prediction(probs, classes, normal_threshold)
            result = {
                k: v
                for k, v in row.items()
                if k not in {"_image_path"} and not str(k).startswith("_")
            }
            result.update(
                {
                    "image_path": str(image_path),
                    "predicted": pred,
                    "prob_limit": "" if prob_limit is None else prob_limit,
                    "normal_threshold": "" if normal_threshold is None else normal_threshold,
                }
            )
            for idx, cls in enumerate(classes):
                result[f"p_{sanitize_part(cls)}"] = round(float(probs[idx].item()), 6)

            if not args.no_copy:
                dest_dir = pred_root / sanitize_part(pred)
                dest = unique_dest(dest_dir, image_path, start + offset)
                shutil.copy2(image_path, dest)
                result["copied_image"] = str(dest.relative_to(args.output_dir)).replace("\\", "/")
            results.append(result)

    fieldnames = sorted({key for row in results for key in row.keys()})
    preferred = ["image_path", "predicted", "p_normal", "p_abnormal", "prob_limit", "normal_threshold", "copied_image"]
    ordered = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(results)

    counts = Counter(row["predicted"] for row in results)
    summary = {
        "model_run": str(run_dir),
        "processed": len(results),
        "classes": classes,
        "normal_threshold": normal_threshold,
        "predicted_counts": dict(counts),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"[predict-images] processed={len(results)} output={args.output_dir}")
    print(f"[predict-images] counts={dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
