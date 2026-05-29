#!/usr/bin/env python
"""Fail fast when the torch/torchvision binary pair is unusable."""

from __future__ import annotations

import sys
import os


DEFAULT_EXPECTED = ("2.3.1", "0.18.1", "2.3.1")
H200_EXPECTED = ("2.7.0", "0.22.0", "2.7.0")


def infer_profile(torch_module) -> str:
    raw = os.environ.get("AD_TORCH_PROFILE", "").strip().lower()
    if raw:
        return raw
    try:
        if torch_module.cuda.is_available():
            for idx in range(torch_module.cuda.device_count()):
                if "h200" in torch_module.cuda.get_device_name(idx).lower():
                    return "h200"
    except Exception:
        pass
    return "default"


def expected_versions(profile: str) -> tuple[str, str, str]:
    if profile == "h200":
        return H200_EXPECTED
    return DEFAULT_EXPECTED


def fix_text(expected_torch: str, expected_vision: str, expected_audio: str, profile: str) -> str:
    req_file = "requirements-h200.txt" if profile == "h200" else "requirements.txt"
    return f"""\
Fix the environment, then rerun:

  python -m pip uninstall -y torch torchvision torchaudio
  python -m pip cache purge
  rm -rf ~/.cache/pip
  python -m pip install --no-cache-dir --force-reinstall \\
    torch=={expected_torch} \\
    torchvision=={expected_vision} \\
    torchaudio=={expected_audio}
  python -m pip install -r {req_file}

Default/non-H200 runtime is torch 2.3.1 with CUDA 12.1 wheels. H200 runtime is
torch 2.7.0. Official PyTorch 2.7.0 wheels are cu126/cu128; use the company
PyPI/mirror if it provides a custom cu130 build. Do not mix CPU torchvision
with CUDA torch.
"""


def fail(message: str, exc: BaseException | None = None, fix: str = "") -> int:
    print(f"[runtime-check] FATAL: {message}", file=sys.stderr)
    if exc is not None:
        print(f"[runtime-check] {type(exc).__name__}: {exc}", file=sys.stderr)
    if fix:
        print(fix, file=sys.stderr)
    return 1


def main() -> int:
    print(f"[runtime-check] python={sys.executable}")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment guard
        expected_torch, expected_vision, expected_audio = DEFAULT_EXPECTED
        return fail(
            "cannot import torch",
            exc,
            fix_text(expected_torch, expected_vision, expected_audio, "default"),
        )

    torch_version = getattr(torch, "__version__", "?")
    profile = infer_profile(torch)
    expected_torch, expected_vision, expected_audio = expected_versions(profile)
    fix = fix_text(expected_torch, expected_vision, expected_audio, profile)
    print(
        "[runtime-check] "
        f"profile={profile} "
        f"torch={torch_version} "
        f"torch_cuda={getattr(torch.version, 'cuda', None)} "
        f"torch_file={getattr(torch, '__file__', '?')}"
    )
    if not str(torch_version).startswith(expected_torch):
        return fail(f"unexpected torch version: {torch_version}; expected {expected_torch}", fix=fix)

    try:
        import torchvision
    except Exception as exc:  # pragma: no cover - environment guard
        return fail("cannot import torchvision; torch/torchvision wheels are likely mismatched", exc, fix)

    vision_version = getattr(torchvision, "__version__", "?")
    print(
        "[runtime-check] "
        f"torchvision={vision_version} "
        f"torchvision_file={getattr(torchvision, '__file__', '?')}"
    )
    if not str(vision_version).startswith(expected_vision):
        return fail(f"unexpected torchvision version: {vision_version}; expected {expected_vision}", fix=fix)

    try:
        import torchaudio
    except Exception as exc:  # pragma: no cover - environment guard
        return fail("cannot import torchaudio; torch/torchaudio wheels are likely mismatched", exc, fix)

    audio_version = getattr(torchaudio, "__version__", "?")
    print(
        "[runtime-check] "
        f"torchaudio={audio_version} "
        f"torchaudio_file={getattr(torchaudio, '__file__', '?')}"
    )
    if not str(audio_version).startswith(expected_audio):
        return fail(f"unexpected torchaudio version: {audio_version}; expected {expected_audio}", fix=fix)

    try:
        from torchvision.ops import nms

        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 1.1, 1.1]])
        scores = torch.tensor([0.9, 0.8])
        keep = nms(boxes, scores, 0.5)
        if keep.numel() == 0:
            return fail("torchvision.ops.nms returned no indices in smoke test", fix=fix)
    except Exception as exc:  # pragma: no cover - environment guard
        return fail("torchvision.ops.nms is unavailable", exc, fix)

    if torch.cuda.is_available():
        try:
            x = torch.ones(1, device="cuda")
            torch.cuda.synchronize()
            del x
        except Exception as exc:  # pragma: no cover - environment guard
            return fail("CUDA tensor smoke test failed", exc, fix)
        print(f"[runtime-check] cuda=ok devices={torch.cuda.device_count()}")
    else:
        print("[runtime-check] cuda=not available")

    print("[runtime-check] ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
