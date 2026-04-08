"""ConvNeXtV2-Tiny pretrained weights 사전 다운로드 스크립트.

폐쇄망 H200 서버에는 인터넷이 없어 timm/HuggingFace 다운이 불가능하다.
이 스크립트를 인터넷이 되는 머신에서 한 번 실행하여 weights/ 폴더에 저장한 뒤,
sneakernet (USB / scp via jump host) 으로 폐쇄망 서버로 옮긴다.

사용법:
    # 인터넷 가능한 dev 머신에서
    pip install timm torch
    python scripts/download_pretrained.py

    # → weights/convnextv2_tiny_pretrained.pth 생성
    # → 이 파일을 폐쇄망 서버의 동일 경로로 복사
    # → train.py 가 자동으로 이 파일을 사용 (다운 없이)

다른 모델 (예: convnext_base):
    python scripts/download_pretrained.py --model convnext_base.fb_in22k_ft_in1k --out weights/convnext_base.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str,
                        default="convnextv2_tiny.fcmae_ft_in22k_in1k",
                        help="timm model name")
    parser.add_argument("--out", type=str,
                        default="weights/convnextv2_tiny_pretrained.pth",
                        help="저장 경로 (default: weights/convnextv2_tiny_pretrained.pth)")
    args = parser.parse_args()

    try:
        import timm
        import torch
    except ImportError as e:
        print(f"[ERROR] timm/torch 미설치: {e}")
        print("        pip install timm torch")
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] downloading: {args.model}")
    print(f"[INFO] dest:        {out_path.resolve()}")

    try:
        model = timm.create_model(args.model, pretrained=True)
    except Exception as e:
        print(f"[ERROR] 다운로드 실패 (인터넷 또는 모델명 확인): {e}")
        sys.exit(2)

    # state_dict만 저장 (architecture는 timm이 알고 있음)
    state_dict = model.state_dict()
    torch.save(state_dict, str(out_path))

    n_params = sum(p.numel() for p in model.parameters())
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[OK]    saved {size_mb:.1f} MB ({n_params/1e6:.1f}M params)")
    print()
    print(f"폐쇄망 서버로 복사:")
    print(f"  scp {out_path} user@server:/path/to/anomaly-detection/{out_path}")
    print(f"  또는 USB로 옮긴 뒤 동일 경로에 배치")
    print()
    print(f"train.py 가 자동 인식 (코드 변경 불필요):")
    print(f"  if Path('weights/convnextv2_tiny_pretrained.pth').exists():")
    print(f"      → 파일 로드 (네트워크 호출 없음)")


if __name__ == "__main__":
    main()
