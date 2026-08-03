#!/usr/bin/env python
"""Evaluate a saved best_model.pth against labeled image folders.

라벨이 폴더 이름인 이미지 디렉터리 (generate_images.py 출력 구조) 를 그대로 채점한다.

  <images>/normal/*.png
  <images>/abnormal/*.png

  python scripts/eval_model.py --model logs/<run>/best_model.pth --images images/test
  python scripts/eval_model.py --model logs/<run> --images images --split test

출력: confusion matrix + 클래스별 recall/precision/F1 + FN/FP 개수.
--save-csv 를 주면 이미지별 확률·판정을 CSV 로 남긴다.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import _load_model_from_best_info  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
TQDM_DISABLE = not sys.stderr.isatty()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="best_model.pth 파일 경로 또는 run 폴더 (best_info.json 필요)")
    parser.add_argument("--images", required=True, type=Path,
                        help="클래스 폴더를 담은 디렉터리 (예: images/test)")
    parser.add_argument("--split", default=None,
                        help="--images 아래 split 하위폴더 (예: test). 생략 시 --images 를 그대로 사용")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16",
                        help="CUDA 에서만 적용 (기본 fp16 — train.py eval 과 동일)")
    parser.add_argument("--normal-threshold", type=float, default=None,
                        help="normal iff p_normal > threshold. 생략 시 best_info.json 값, 없으면 argmax")
    parser.add_argument("--save-csv", type=Path, default=None,
                        help="이미지별 예측 CSV 저장 경로")
    return parser.parse_args()


def resolve_run(path_text: str) -> Path:
    path = Path(path_text)
    run_dir = path.parent if path.is_file() else path
    for name in ("best_model.pth", "best_info.json"):
        if not (run_dir / name).exists():
            raise SystemExit(f"{name} 이 없습니다: {run_dir}")
    return run_dir


def default_threshold(run_dir: Path, classes: list[str]) -> float | None:
    if "normal" not in classes or "abnormal" not in classes:
        return None
    info = json.loads((run_dir / "best_info.json").read_text(encoding="utf-8"))
    selected = info.get("selected_normal_threshold")
    if selected is not None:
        return float(selected)
    hparams = info.get("hparams") or {}
    if hparams.get("normal_threshold") is not None:
        return float(hparams["normal_threshold"])
    return 0.5


def collect_images(root: Path) -> list[tuple[Path, str]]:
    items = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                items.append((path, class_dir.name))
    return items


def main() -> int:
    args = parse_args()
    root = args.images / args.split if args.split else args.images
    if not root.is_dir():
        raise SystemExit(f"이미지 폴더가 없습니다: {root}")
    items = collect_images(root)
    if not items:
        raise SystemExit(f"클래스 하위폴더에서 이미지를 찾지 못했습니다: {root}")

    run_dir = resolve_run(args.model)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA 를 요청했지만 사용할 수 없습니다")

    model, classes = _load_model_from_best_info(run_dir, device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    amp_dtype = None
    if device.type == "cuda" and args.precision != "fp32":
        amp_dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16

    threshold = args.normal_threshold
    if threshold is None:
        threshold = default_threshold(run_dir, classes)

    labels = sorted({label for _p, label in items})
    print(f"[eval] model: {run_dir}")
    print(f"[eval] images: {root}  ({len(items)}장, 폴더 라벨: {', '.join(labels)})")
    print(f"[eval] classes: {classes}  threshold: "
          f"{'argmax' if threshold is None else f'p_normal > {threshold}'}  device: {device.type}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    rows = []
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(items), args.batch_size), desc="eval",
                          disable=TQDM_DISABLE):
            batch = items[start:start + args.batch_size]
            tensors = [transform(Image.open(p).convert("RGB")) for p, _ in batch]
            x = torch.stack(tensors).to(device)
            if device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            ctx = (torch.amp.autocast("cuda", dtype=amp_dtype) if amp_dtype
                   else contextlib.nullcontext())
            with ctx:
                logits = model(x)
            probs = F.softmax(logits.float(), dim=1)
            for (path, truth), prob in zip(batch, probs):
                if threshold is not None and "normal" in classes and "abnormal" in classes:
                    p_normal = float(prob[classes.index("normal")])
                    pred = "normal" if p_normal > threshold else "abnormal"
                else:
                    pred = classes[int(torch.argmax(prob))]
                rows.append({
                    "image": str(path.relative_to(root)).replace("\\", "/"),
                    "true": truth, "pred": pred,
                    "p_abnormal": round(float(prob[classes.index("abnormal")]), 6)
                    if "abnormal" in classes else "",
                })

    # 폴더 라벨이 defect 종류여도 binary 로 환산 (normal 이 아니면 abnormal)
    def to_binary(name: str) -> str:
        return "normal" if name == "normal" else "abnormal"

    tp = sum(1 for r in rows if to_binary(r["true"]) == "abnormal" and to_binary(r["pred"]) == "abnormal")
    fn = sum(1 for r in rows if to_binary(r["true"]) == "abnormal" and to_binary(r["pred"]) == "normal")
    fp = sum(1 for r in rows if to_binary(r["true"]) == "normal" and to_binary(r["pred"]) == "abnormal")
    tn = sum(1 for r in rows if to_binary(r["true"]) == "normal" and to_binary(r["pred"]) == "normal")

    recall = tp / (tp + fn) if tp + fn else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    nor_recall = tn / (tn + fp) if tn + fp else float("nan")
    acc = (tp + tn) / len(rows) if rows else float("nan")

    print()
    print("  Confusion (행=실제, 열=예측)")
    print(f"    {'':>10} {'normal':>10} {'abnormal':>10}")
    print(f"    {'normal':>10} {tn:>10} {fp:>10}")
    print(f"    {'abnormal':>10} {fn:>10} {tp:>10}")
    print()
    print(f"    {'class':>10} {'recall':>9} {'precision':>10} {'F1':>9} {'N':>7}")
    print(f"    {'normal':>10} {nor_recall:>9.4f} "
          f"{(tn / (tn + fn) if tn + fn else float('nan')):>10.4f} {'':>9} {tn + fp:>7}")
    print(f"    {'abnormal':>10} {recall:>9.4f} {precision:>10.4f} {f1:>9.4f} {tp + fn:>7}")
    print()
    print(f"  binary F1={f1:.4f}  abn_recall={recall:.4f}  accuracy={acc:.4f}  "
          f"FN={fn} FP={fp}  (N={len(rows)})")

    if args.save_csv:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.save_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image", "true", "pred", "p_abnormal"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  CSV: {args.save_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
