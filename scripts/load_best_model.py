#!/usr/bin/env python
"""Load a saved best_model.pth — 다른 코드에서 import 하거나 단독 점검용으로 실행.

import 해서 쓰기:

    from scripts.load_best_model import load_best_model, preprocess

    model, classes, device = load_best_model("logs/<run>")   # 또는 .../best_model.pth
    x = preprocess("chart.png").unsqueeze(0).to(device)      # 1장
    with torch.no_grad():
        p_abnormal = torch.softmax(model(x), dim=1)[0, classes.index("abnormal")].item()

단독 실행 — checkpoint 가 멀쩡한지 확인 (모델명/클래스/파라미터 수/threshold 출력,
더미 입력 1회 forward):

    python scripts/load_best_model.py logs/<run>
    python scripts/load_best_model.py logs/<run>/best_model.pth --image chart.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import _load_model_from_best_info  # noqa: E402

# train.py / predict_images.py eval 과 동일한 전처리
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def preprocess(image_path) -> torch.Tensor:
    """이미지 1장 -> (3, 224, 224) 텐서. batch 로 쓰려면 unsqueeze(0)."""
    return _TRANSFORM(Image.open(image_path).convert("RGB"))


def load_best_model(model_path, device: str = "auto"):
    """run 폴더 또는 best_model.pth 경로 -> (model, classes, device).

    model 은 eval() 상태이고, best_info.json 의 hparams 로 구조를 복원한다.
    """
    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available())
                       else ("cpu" if device == "auto" else device))
    model, classes = _load_model_from_best_info(Path(model_path), dev)
    return model, classes, dev


def normal_threshold_of(model_path) -> float | None:
    """학습 때 선택된 normal threshold (없으면 None)."""
    run_dir = Path(model_path)
    if run_dir.is_file():
        run_dir = run_dir.parent
    info = json.loads((run_dir / "best_info.json").read_text(encoding="utf-8"))
    selected = info.get("selected_normal_threshold")
    if selected is not None:
        return float(selected)
    return (info.get("hparams") or {}).get("normal_threshold")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="run 폴더 또는 best_model.pth 경로")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--image", default=None,
                        help="이 이미지 1장을 실제로 예측해 본다 (없으면 더미 입력)")
    args = parser.parse_args()

    model, classes, device = load_best_model(args.model, args.device)
    params = sum(p.numel() for p in model.parameters())
    print(f"    params: {params / 1e6:.1f}M   device: {device.type}")
    print(f"    normal_threshold: {normal_threshold_of(args.model)}")

    if args.image:
        x = preprocess(args.image).unsqueeze(0).to(device)
    else:
        x = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    pairs = ", ".join(f"{c}={float(p):.4f}" for c, p in zip(classes, probs))
    print(f"    forward OK {'(' + args.image + ')' if args.image else '(더미 입력)'}: {pairs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
