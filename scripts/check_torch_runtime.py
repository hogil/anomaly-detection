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
    torch==2.3.1 \\
    torchvision==0.18.1 \\
    torchaudio==2.3.1
  python -m pip install -r requirements.txt

Use the company PyPI/mirror configured on the server. Do not use external
PyTorch index URLs on the server. The company mirror must resolve these
versions to CUDA-enabled builds. Do not mix CPU torchvision with CUDA torch.
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

    torch_version = getattr(torch, "__version__", "?")
    print(
        "[runtime-check] "
        f"torch={torch_version} "
        f"torch_cuda={getattr(torch.version, 'cuda', None)} "
        f"torch_file={getattr(torch, '__file__', '?')}"
    )
    if not str(torch_version).startswith("2.3.1"):
        return fail(f"unexpected torch version: {torch_version}; expected 2.3.1")

    try:
        import torchvision
    except Exception as exc:  # pragma: no cover - environment guard
        return fail("cannot import torchvision; torch/torchvision wheels are likely mismatched", exc)

    vision_version = getattr(torchvision, "__version__", "?")
    print(
        "[runtime-check] "
        f"torchvision={vision_version} "
        f"torchvision_file={getattr(torchvision, '__file__', '?')}"
    )
    if not str(vision_version).startswith("0.18.1"):
        return fail(f"unexpected torchvision version: {vision_version}; expected 0.18.1")

    try:
        import torchaudio
    except Exception as exc:  # pragma: no cover - environment guard
        return fail("cannot import torchaudio; torch/torchaudio wheels are likely mismatched", exc)

    audio_version = getattr(torchaudio, "__version__", "?")
    print(
        "[runtime-check] "
        f"torchaudio={audio_version} "
        f"torchaudio_file={getattr(torchaudio, '__file__', '?')}"
    )
    if not str(audio_version).startswith("2.3.1"):
        return fail(f"unexpected torchaudio version: {audio_version}; expected 2.3.1")

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
