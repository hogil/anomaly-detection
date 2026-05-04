"""
Pretrained backbone weights downloader.

HuggingFace 에서 timm pretrained 가중치를 받아 weights/{HF model id}.pth 로 저장한다.
파일명은 timm/HF 정식 model id 그대로 (e.g. convnextv2_tiny.fcmae_ft_in22k_in1k.pth).

- 항상 MODELS 목록 전부 시도. 1개 실패해도 나머지는 계속 진행.
- 이미 있는 파일은 자동 skip (재다운 안 함).
- 학습/추론 코드는 항상 pretrained=False + weights/{model_name}.pth 로드.
- 가중치 파일은 절대 git 에 올리지 않는다 (.gitignore 처리됨).
- 폐쇄망 서버: 인터넷 머신에서 받아 weights/ 폴더 통째로 복사할 것.

Usage:
    python download.py             # MODELS 전부, 이미 있는 건 skip
    python download.py --force      # 이미 있어도 덮어쓰기
"""
import argparse
import os
import sys
import timm
import torch

# HF / timm model id 목록 (그대로 파일명으로 사용)
MODELS = [
    "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "convnextv2_base.fcmae_ft_in22k_in1k",
    "tf_efficientnetv2_s.in21k_ft_in1k",
    "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
    "maxvit_tiny_tf_224.in1k",
    "vit_base_patch16_clip_224.laion2b_ft_in12k_in1k",
]


def download_one(model_name: str, force: bool = False) -> str:
    """단일 모델 다운로드. 반환: 'ok' / 'skip' / 'fail'."""
    out_path = f"weights/{model_name}.pth"
    if os.path.exists(out_path) and not force:
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  skip   {out_path} ({size_mb:.0f} MB, already exists)")
        return "skip"
    print(f"  download {model_name} ...")
    try:
        m = timm.create_model(model_name, pretrained=True)
        torch.save(m.state_dict(), out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  saved  {out_path} ({size_mb:.0f} MB)")
        return "ok"
    except Exception as e:
        print(f"  FAIL   {model_name}: {type(e).__name__}: {e}", file=sys.stderr)
        return "fail"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--force", action="store_true", help="이미 있어도 덮어쓰기")
    args = ap.parse_args()

    os.makedirs("weights", exist_ok=True)
    print(f"Targets: {len(MODELS)} models")
    print(f"Output:  weights/{{model_name}}.pth")
    print()

    counts = {"ok": 0, "skip": 0, "fail": 0}
    failures = []
    for i, name in enumerate(MODELS, 1):
        print(f"[{i}/{len(MODELS)}] {name}")
        result = download_one(name, force=args.force)
        counts[result] += 1
        if result == "fail":
            failures.append(name)
        print()

    print("=" * 60)
    print(f"Done: {counts['ok']} downloaded, {counts['skip']} skipped, {counts['fail']} failed")
    if failures:
        print("Failures:")
        for n in failures:
            print(f"  - {n}")
        sys.exit(1)


if __name__ == "__main__":
    main()
