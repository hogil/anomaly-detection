"""
Pretrained backbone weights downloader.

HuggingFace에서 timm pretrained 가중치를 받아 weights/<short>.pth 로 저장한다.
- 학습/추론 코드는 항상 pretrained=False 로 모델을 만들고 weights/ 폴더에서만 로드한다.
- 가중치 파일은 절대 git에 올리지 않는다 (.gitignore 처리됨).
- 폐쇄망 서버: 인터넷 머신에서 받아 weights/ 폴더 통째로 복사할 것.

Usage:
    python download.py                  # 기본: convnextv2_tiny 만
    python download.py --all             # 6개 backbone 전부
"""
import argparse
import os
import timm
import torch

# (short_name, hf_full_name) — short_name 은 weights/<short>.pth 파일명
MODELS = {
    "convnextv2_tiny":  "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "convnextv2_base":  "convnextv2_base.fcmae_ft_in22k_in1k",
    "efficientnetv2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
    "swin_tiny":        "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
    "maxvit_tiny":      "maxvit_tiny_tf_224.in1k",
    "clip_vit_b16":     "vit_base_patch16_clip_224.laion2b_ft_in12k_in1k",
}


def download(short: str, hf_name: str, force: bool = False):
    out_path = f"weights/{short}.pth"
    if os.path.exists(out_path) and not force:
        print(f"  skip {out_path} (already exists)")
        return
    print(f"  downloading {short} ← {hf_name}")
    m = timm.create_model(hf_name, pretrained=True)
    torch.save(m.state_dict(), out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  saved {out_path} ({size_mb:.0f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="6개 backbone 전부 받기")
    ap.add_argument("--force", action="store_true", help="이미 있어도 덮어쓰기")
    args = ap.parse_args()

    os.makedirs("weights", exist_ok=True)
    targets = list(MODELS.items()) if args.all else [("convnextv2_tiny", MODELS["convnextv2_tiny"])]
    for short, hf_name in targets:
        download(short, hf_name, force=args.force)
    print("\ndone.")


if __name__ == "__main__":
    main()
