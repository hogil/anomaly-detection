"""
Pretrained backbone weights downloader.

HuggingFace 에서 timm pretrained 가중치를 받아 weights/{HF model id}.pth 로 저장한다.
파일명은 timm/HF 정식 model id 그대로 (e.g. convnextv2_tiny.fcmae_ft_in22k_in1k.pth).

- 학습/추론 코드는 항상 pretrained=False 로 모델을 만들고 weights/{model_name}.pth 로 로드.
- 가중치 파일은 절대 git 에 올리지 않는다 (.gitignore 처리됨).
- 폐쇄망 서버: 인터넷 머신에서 받아 weights/ 폴더 통째로 복사할 것.

Usage:
    python download.py                  # 기본: convnextv2_tiny.fcmae_ft_in22k_in1k 만
    python download.py --all             # 6 backbone 전부
"""
import argparse
import os
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


def download(model_name: str, force: bool = False):
    out_path = f"weights/{model_name}.pth"
    if os.path.exists(out_path) and not force:
        print(f"  skip {out_path} (already exists)")
        return
    print(f"  downloading {model_name}")
    m = timm.create_model(model_name, pretrained=True)
    torch.save(m.state_dict(), out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  saved  {out_path} ({size_mb:.0f} MB)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--all", action="store_true", help="6 backbone 전부 받기")
    ap.add_argument("--force", action="store_true", help="이미 있어도 덮어쓰기")
    args = ap.parse_args()

    os.makedirs("weights", exist_ok=True)
    targets = MODELS if args.all else [MODELS[0]]
    for m in targets:
        download(m, force=args.force)
    print("\ndone.")


if __name__ == "__main__":
    main()
