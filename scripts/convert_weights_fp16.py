"""ConvNeXtV2-Tiny pretrained weights를 fp16으로 변환.

이유: GitHub 100MB 파일 제한을 우회하기 위해 (fp32 110MB → fp16 55MB).
폐쇄망 H200 서버는 GitHub repo만 받을 수 있고 별도 sneakernet 불가능.
fp16 → fp32 cast는 train.py 가 로드 시 자동 처리 (정밀도 손실 무시 가능).

사용법:
    python scripts/convert_weights_fp16.py
    # → weights/convnextv2_tiny_pretrained.pth (110MB fp32, 로컬)
    #    → weights/convnextv2_tiny.fp16.pth     (55MB fp16, 커밋용)

train.py는 다음 순서로 weights 검색:
    1. weights/convnextv2_tiny_pretrained.pth  (로컬 fp32 우선)
    2. weights/convnextv2_tiny.fp16.pth        (커밋된 fp16, 폐쇄망 서버)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=str,
                        default="weights/convnextv2_tiny_pretrained.pth",
                        help="입력 fp32 weights 경로")
    parser.add_argument("--dst", type=str,
                        default="weights/convnextv2_tiny.fp16.pth",
                        help="출력 fp16 weights 경로")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        print(f"[ERROR] {src} 없음")
        print(f"        먼저 'python scripts/download_pretrained.py' 실행")
        return 1

    print(f"[INFO] loading: {src} ({src.stat().st_size / 1024 / 1024:.1f} MB)")
    state_dict = torch.load(str(src), map_location="cpu", weights_only=True)

    # 모든 floating point tensor를 fp16으로 변환 (int/long은 그대로)
    fp16_state = {}
    n_converted = 0
    for k, v in state_dict.items():
        if v.dtype in (torch.float32, torch.float64):
            fp16_state[k] = v.half()
            n_converted += 1
        else:
            fp16_state[k] = v

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fp16_state, str(dst))

    src_mb = src.stat().st_size / 1024 / 1024
    dst_mb = dst.stat().st_size / 1024 / 1024
    print(f"[OK]   {dst}")
    print(f"       {src_mb:.1f} MB -> {dst_mb:.1f} MB ({100 * dst_mb / src_mb:.0f}% of original)")
    print(f"       {n_converted} tensors converted to fp16")
    print()
    print("이제 git add weights/convnextv2_tiny.fp16.pth && git commit && git push")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
