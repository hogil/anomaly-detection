#!/usr/bin/env python
"""Fail fast when the torch/torchvision binary pair is unusable."""

from __future__ import annotations

import sys


FIX = """\
Fix the environment, then rerun:

  python -m pip uninstall -y torch torchvision torchaudio
  python -m pip cache purge
  rm -rf ~/.cache/pip
  python -m pip install --no-cache-dir --force-reinstall \\
    torch==2.3.1+cu121 \\
    torchvision==0.18.1+cu121 \\
    torchaudio==2.3.1+cu121
  python -m pip install -r requirements.txt

Use the company PyPI/mirror configured on the server. Do not use external
PyTorch index URLs on the server. Do not mix CPU torchvision with CUDA torch.
"""


def fail(message: str, exc: BaseException | None = None) -> int:
    print(f"[runtime-check] FATAL: {message}", file=sys.stderr)
    if exc is not None:
        print(f"[runtime-check] {type(exc).__name__}: {exc}", file=sys.stderr)
    print(FIX, file=sys.stderr)
    return 1


def main() -> int:
    print(f"[runtime-check] python={sys.executable}")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment guard
        return fail("cannot import torch", exc)

    print(
        "[runtime-check] "
        f"torch={getattr(torch, '__version__', '?')} "
        f"torch_cuda={getattr(torch.version, 'cuda', None)}"
    )

    try:
        import torchvision
    except Exception as exc:  # pragma: no cover - environment guard
        return fail("cannot import torchvision; torch/torchvision wheels are likely mismatched", exc)

    print(f"[runtime-check] torchvision={getattr(torchvision, '__version__', '?')}")

    try:
        from torchvision.ops import nms

        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 1.1, 1.1]])
        scores = torch.tensor([0.9, 0.8])
        keep = nms(boxes, scores, 0.5)
        if keep.numel() == 0:
            return fail("torchvision.ops.nms returned no indices in smoke test")
    except Exception as exc:  # pragma: no cover - environment guard
        return fail("torchvision.ops.nms is unavailable", exc)

    if torch.cuda.is_available():
        try:
            x = torch.ones(1, device="cuda")
            torch.cuda.synchronize()
            del x
        except Exception as exc:  # pragma: no cover - environment guard
            return fail("CUDA tensor smoke test failed", exc)
        print(f"[runtime-check] cuda=ok devices={torch.cuda.device_count()}")
    else:
        print("[runtime-check] cuda=not available")

    print("[runtime-check] ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
