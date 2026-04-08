"""실험에 필요한 모든 backbone pretrained weights 일괄 다운로드.

폐쇄망 H200 서버에는 인터넷이 없어 timm/HuggingFace 다운이 불가능하다.
이 스크립트를 인터넷이 되는 머신에서 한 번 실행하여 weights/ 폴더에 모든 모델을
다운받은 뒤, 폴더 전체를 sneakernet (USB / scp / jump host) 으로 폐쇄망 서버에 옮긴다.

사용법:
    pip install timm torch

    # 1) 필수 모델만 (default: convnextv2_tiny, ~110MB)
    python scripts/download_weights.py

    # 2) 6 backbone 비교 실험용 전부 (~1.2GB)
    python scripts/download_weights.py --preset all

    # 3) 특정 모델만
    python scripts/download_weights.py --models convnextv2_tiny convnextv2_base

    # 4) 다운 후 fp16 변환까지 (각 절반 사이즈)
    python scripts/download_weights.py --preset all --fp16

    # 5) 목록만 보기
    python scripts/download_weights.py --list

저장 경로: weights/<short_name>.pth (예: weights/convnextv2_tiny.pth)
train.py 자동 인식 기준 파일: weights/convnextv2_tiny_pretrained.pth
  → 이 스크립트는 default 모델만 그 이름으로 추가 저장 (그래야 train.py가 바로 사용)
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

# Windows cp949 콘솔에서도 한글/유니코드 출력
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# ============================================================================
# Backbone catalog — 실험에서 쓰는 모든 timm 모델
# ============================================================================
# (short_key, timm_model_name, approx_params_M, description)

BACKBONES = {
    "convnextv2_tiny": (
        "convnextv2_tiny.fcmae_ft_in22k_in1k",
        28.6,
        "ConvNeXtV2-Tiny — current default, winning config 기준 모델",
    ),
    "convnextv2_base": (
        "convnextv2_base.fcmae_ft_in22k_in1k",
        88.7,
        "ConvNeXtV2-Base — 더 큰 backbone (3x), 비교 실험용",
    ),
    "efficientnetv2_s": (
        "tf_efficientnetv2_s.in21k_ft_in1k",
        21.5,
        "EfficientNetV2-S — 가벼운 CNN, 속도 비교용",
    ),
    "swin_tiny": (
        "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
        28.3,
        "Swin-Tiny — Vision Transformer, ConvNeXt 와 동급 사이즈",
    ),
    "maxvit_tiny": (
        "maxvit_tiny_tf_224.in1k",
        30.9,
        "MaxViT-Tiny — Hybrid CNN+ViT",
    ),
    "clip_vit_b16": (
        "vit_base_patch16_clip_224.laion2b_ft_in12k_in1k",
        86.6,
        "CLIP-ViT-B/16 — pretrained on LAION-2B (가장 큰 데이터)",
    ),
}

# Preset 정의
PRESETS = {
    "default": ["convnextv2_tiny"],
    "all":     list(BACKBONES.keys()),
    "small":   ["convnextv2_tiny", "efficientnetv2_s", "swin_tiny"],
    "large":   ["convnextv2_base", "clip_vit_b16"],
}

# train.py 가 자동 인식하는 default 파일 이름 (convnextv2_tiny 전용)
DEFAULT_PATH_FOR_TINY = "weights/convnextv2_tiny_pretrained.pth"


def list_backbones():
    print()
    print(f"{'key':<22} {'params':>8}  timm model name")
    print("-" * 100)
    for key, (name, params, desc) in BACKBONES.items():
        print(f"{key:<22} {params:>6.1f}M  {name}")
        print(f"{'':<22} {'':>8}  └ {desc}")
    print()
    print(f"presets: {', '.join(PRESETS.keys())}")
    print(f"  default → {PRESETS['default']}")
    print(f"  small   → {PRESETS['small']}")
    print(f"  large   → {PRESETS['large']}")
    print(f"  all     → {len(PRESETS['all'])} models")


def download_one(key: str, model_name: str, params_M: float,
                 out_dir: Path, fp16: bool, force: bool) -> dict:
    import timm
    import torch

    # 표준 파일명: weights/<key>.pth (또는 .fp16.pth)
    fname = f"{key}.fp16.pth" if fp16 else f"{key}.pth"
    out_path = out_dir / fname

    status = {"key": key, "name": model_name, "path": str(out_path),
              "skipped": False, "size_mb": 0.0, "elapsed_sec": 0.0}

    if out_path.exists() and not force:
        size_mb = out_path.stat().st_size / 1024 / 1024
        status["skipped"] = True
        status["size_mb"] = size_mb
        print(f"  [SKIP] {key:<22} {size_mb:>6.1f}MB  (이미 존재: {out_path.name})")
        return status

    print(f"  [DL]   {key:<22} (~{params_M:.0f}M params, {model_name})")
    t0 = time.time()
    try:
        model = timm.create_model(model_name, pretrained=True)
    except Exception as e:
        print(f"         FAILED: {e}")
        status["error"] = str(e)
        return status

    state_dict = model.state_dict()

    if fp16:
        state_dict = {
            k: (v.half() if hasattr(v, "dtype") and v.dtype in (torch.float32, torch.float64) else v)
            for k, v in state_dict.items()
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, str(out_path))

    size_mb = out_path.stat().st_size / 1024 / 1024
    elapsed = time.time() - t0
    status["size_mb"] = size_mb
    status["elapsed_sec"] = round(elapsed, 1)
    print(f"         OK     {size_mb:>6.1f}MB  ({elapsed:.1f}s)  → {out_path}")

    # convnextv2_tiny는 train.py default 경로로도 추가 저장 (no convert, just copy)
    if key == "convnextv2_tiny" and not fp16:
        import shutil
        default_path = Path(DEFAULT_PATH_FOR_TINY)
        if not default_path.exists():
            shutil.copy2(str(out_path), str(default_path))
            print(f"         + copied → {default_path} (train.py default)")

    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", type=str, default="default",
                        choices=list(PRESETS.keys()),
                        help="다운받을 모델 묶음 (default: convnextv2_tiny만)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="개별 모델 (preset 무시). 키 목록은 --list")
    parser.add_argument("--out", type=str, default="weights",
                        help="저장 폴더 (default: weights/)")
    parser.add_argument("--fp16", action="store_true",
                        help="fp16 변환하여 저장 (절반 사이즈)")
    parser.add_argument("--force", action="store_true",
                        help="이미 존재하는 파일도 재다운")
    parser.add_argument("--list", action="store_true",
                        help="사용 가능한 backbone 목록 출력 후 종료")
    args = parser.parse_args()

    if args.list:
        list_backbones()
        return 0

    try:
        import timm  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        print(f"[ERROR] timm/torch 미설치: {e}")
        print("        pip install timm torch")
        return 1

    # 다운받을 키 결정
    if args.models:
        keys = []
        for m in args.models:
            if m not in BACKBONES:
                print(f"[ERROR] unknown model key: {m}")
                print(f"        valid keys: {list(BACKBONES.keys())}")
                return 2
            keys.append(m)
    else:
        keys = PRESETS[args.preset]

    out_dir = Path(args.out)

    print()
    print("=" * 80)
    print(f" Backbone weights 다운로드")
    print("=" * 80)
    print(f"  preset:    {args.preset if not args.models else 'custom'}")
    print(f"  models:    {len(keys)}개 → {keys}")
    print(f"  out dir:   {out_dir.resolve()}")
    print(f"  fp16:      {args.fp16}")
    print(f"  force:     {args.force}")
    print()

    statuses = []
    for key in keys:
        model_name, params_M, _ = BACKBONES[key]
        s = download_one(key, model_name, params_M, out_dir, args.fp16, args.force)
        statuses.append(s)

    # Summary
    print()
    print("=" * 80)
    print(" SUMMARY")
    print("=" * 80)
    n_dl = sum(1 for s in statuses if not s["skipped"] and "error" not in s)
    n_skip = sum(1 for s in statuses if s["skipped"])
    n_err = sum(1 for s in statuses if "error" in s)
    total_mb = sum(s["size_mb"] for s in statuses)

    print(f"  downloaded: {n_dl}")
    print(f"  skipped:    {n_skip}")
    print(f"  failed:     {n_err}")
    print(f"  total size: {total_mb:.1f} MB ({total_mb/1024:.2f} GB)")
    print()
    print(f"  files in {out_dir}/:")
    for s in statuses:
        if "error" in s:
            print(f"    [ERR ] {s['key']:<22} {s.get('error', '')}")
        else:
            mark = "SKIP" if s["skipped"] else "NEW "
            print(f"    [{mark}] {Path(s['path']).name:<35} {s['size_mb']:>7.1f}MB")
    print()

    if n_err > 0:
        return 3

    # 다음 단계 안내
    print("다음 단계:")
    print(f"  1. weights/ 폴더 전체를 폐쇄망 서버로 복사")
    print(f"     scp -r weights/ user@server:/path/to/anomaly-detection/")
    print(f"     또는 USB로 옮긴 뒤 동일 경로에 배치")
    print(f"  2. 서버에서 train.py 실행 시 자동 사용 (네트워크 호출 없음)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
