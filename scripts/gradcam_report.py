#!/usr/bin/env python
"""Generate Grad-CAM overlays and left/right heat summaries for saved models."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import _load_model_from_best_info

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
RIGHT_START = 0.70


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def resolve_layer(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    modules = dict(model.named_modules())
    if layer_name not in modules:
        nearby = [name for name in modules if name.endswith("conv_dw") or "stages.3" in name]
        hint = "\n".join(nearby[-20:])
        raise SystemExit(f"Grad-CAM layer not found: {layer_name}\nAvailable late layers:\n{hint}")
    return modules[layer_name]


def collect_images(image_root: Path, include_classes: set[str], limit_per_class: int) -> list[tuple[Path, str]]:
    if image_root.is_file():
        return [(image_root, image_root.parent.name)]
    if not image_root.exists():
        raise SystemExit(f"image root not found: {image_root}")

    out: list[tuple[Path, str]] = []
    for class_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
        class_name = class_dir.name
        if include_classes and class_name not in include_classes:
            continue
        paths = sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if limit_per_class > 0:
            paths = paths[:limit_per_class]
        out.extend((path, class_name) for path in paths)
    return out


class GradCam:
    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module) -> None:
        self.model = model
        self.layer = layer
        self.activation: torch.Tensor | None = None
        self.gradient: torch.Tensor | None = None
        self.handles = [
            layer.register_forward_hook(self._forward_hook),
            layer.register_full_backward_hook(self._backward_hook),
        ]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def _forward_hook(self, _module, _inputs, output) -> None:
        self.activation = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output) -> None:
        self.gradient = grad_output[0].detach()

    @staticmethod
    def _as_nchw(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 4:
            raise ValueError(f"expected 4D activation, got shape={tuple(tensor.shape)}")
        # ConvNeXt blocks may expose either NCHW or NHWC depending on hook point.
        if tensor.shape[1] <= 4096 and tensor.shape[1] >= tensor.shape[-1]:
            return tensor
        if tensor.shape[-1] <= 4096 and tensor.shape[-1] > tensor.shape[1]:
            return tensor.permute(0, 3, 1, 2)
        return tensor

    def __call__(self, image: torch.Tensor, target_idx: int) -> tuple[np.ndarray, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        score = logits[:, target_idx].sum()
        score.backward()

        if self.activation is None or self.gradient is None:
            raise RuntimeError("Grad-CAM hooks did not capture activation/gradient")

        activation = self._as_nchw(self.activation)
        gradient = self._as_nchw(self.gradient)
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        denom = cam.max().clamp_min(1e-12)
        cam = cam / denom
        return cam.detach().cpu().numpy(), logits.detach()


def heat_stats(cam: np.ndarray) -> dict[str, float]:
    h, w = cam.shape
    total = float(cam.sum()) + 1e-12
    right_idx = int(round(w * RIGHT_START))
    left_idx = int(round(w * 0.35))
    mid_idx = int(round(w * RIGHT_START))
    xs = np.arange(w, dtype=np.float64)
    col_mass = cam.sum(axis=0)
    peak_x = float((col_mass * xs).sum() / total / max(w - 1, 1))
    return {
        "left_mass_35": float(cam[:, :left_idx].sum() / total),
        "mid_mass_35": float(cam[:, left_idx:mid_idx].sum() / total),
        "right_mass_30": float(cam[:, right_idx:].sum() / total),
        "peak_x_norm": peak_x,
    }


def overlay_cam(image_path: Path, cam: np.ndarray, out_path: Path, title: str) -> None:
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    base = np.asarray(image).astype(np.float32) / 255.0
    heat = plt.get_cmap("jet")(cam)[..., :3]
    overlay = np.clip(0.55 * base + 0.45 * heat, 0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=130)
    axes[0].imshow(base)
    axes[0].set_title("input", fontsize=8)
    axes[1].imshow(overlay)
    axes[1].axvline(224 * RIGHT_START, color="white", linewidth=1, linestyle="--")
    axes[1].set_title("Grad-CAM", fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def summarize_rows(rows: list[dict[str, Any]]) -> str:
    by_class: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_class.setdefault(str(row["true_class"]), []).append(row)

    lines = [
        "# Grad-CAM Summary",
        "",
        "| class | n | pred_abnormal | mean p_abnormal | mean right_mass_30 | mean peak_x_norm |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for class_name, class_rows in sorted(by_class.items()):
        n = len(class_rows)
        pred_abn = sum(1 for r in class_rows if r["pred_class"] == "abnormal")
        mean_p = np.mean([float(r["p_abnormal"]) for r in class_rows])
        mean_right = np.mean([float(r["right_mass_30"]) for r in class_rows])
        mean_peak = np.mean([float(r["peak_x_norm"]) for r in class_rows])
        lines.append(f"| {class_name} | {n} | {pred_abn} | {mean_p:.4f} | {mean_right:.4f} | {mean_peak:.4f} |")
    lines.extend(
        [
            "",
            "Notes:",
            "- `right_mass_30` is the fraction of Grad-CAM heat in the rightmost 30% of the image.",
            "- For binary models, CAM is computed against the `abnormal` logit by default.",
            "- Non-context defect classes are expected to concentrate more to the right if the model learned the generated right-side defect cue.",
            "- `context` can be valid outside the right-side rule because it is a fleet-relative/global condition.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run", required=True, type=Path, help="logs/<run> containing best model")
    parser.add_argument("--image-root", default="images/test", type=Path)
    parser.add_argument("--out-dir", default="validations/gradcam", type=Path)
    parser.add_argument("--layer", default="stages.3.blocks.2.conv_dw")
    parser.add_argument("--target-class", default="abnormal", help="class logit to explain, or 'predicted'")
    parser.add_argument("--include-classes", default="", help="comma-separated image folder classes")
    parser.add_argument("--limit-per-class", type=int, default=5)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, classes = _load_model_from_best_info(args.model_run, device)
    layer = resolve_layer(model, args.layer)
    gradcam = GradCam(model, layer)
    transform = build_transform()

    include_classes = {x.strip() for x in args.include_classes.split(",") if x.strip()}
    images = collect_images(args.image_root, include_classes, args.limit_per_class)
    if not images:
        raise SystemExit(f"no images found: {args.image_root}")

    rows: list[dict[str, Any]] = []
    overlay_dir = args.out_dir / "overlays"
    try:
        for image_path, true_class in images:
            image = Image.open(image_path).convert("RGB")
            x = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_logits = model(x)
                probs = F.softmax(pred_logits, dim=1)[0]
                pred_idx = int(torch.argmax(probs).item())
                pred_class = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)
                abn_idx = classes.index("abnormal") if "abnormal" in classes else min(1, len(classes) - 1)
                p_abnormal = float(probs[abn_idx].item())

            if args.target_class == "predicted":
                target_idx = pred_idx
                explained_class = pred_class
            else:
                if args.target_class not in classes:
                    raise SystemExit(f"target class not in model classes: {args.target_class}; classes={classes}")
                target_idx = classes.index(args.target_class)
                explained_class = args.target_class

            cam, _ = gradcam(x, target_idx)
            stats = heat_stats(cam)
            rel = image_path.relative_to(args.image_root) if image_path.is_relative_to(args.image_root) else image_path.name
            out_name = str(rel).replace("\\", "__").replace("/", "__")
            overlay_path = overlay_dir / out_name
            title = (
                f"true={true_class} pred={pred_class} p_abn={p_abnormal:.3f} "
                f"right={stats['right_mass_30']:.2f}"
            )
            overlay_cam(image_path, cam, overlay_path, title)
            rows.append(
                {
                    "image": str(image_path),
                    "true_class": true_class,
                    "pred_class": pred_class,
                    "explained_class": explained_class,
                    "p_abnormal": p_abnormal,
                    "overlay": str(overlay_path),
                    **stats,
                }
            )
    finally:
        gradcam.close()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "gradcam.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "image",
            "true_class",
            "pred_class",
            "explained_class",
            "p_abnormal",
            "left_mass_35",
            "mid_mass_35",
            "right_mass_30",
            "peak_x_norm",
            "overlay",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.out_dir / "summary.md"
    summary_path.write_text(summarize_rows(rows), encoding="utf-8")
    print(f"[gradcam] wrote {csv_path}")
    print(f"[gradcam] wrote {summary_path}")
    print(f"[gradcam] overlays={overlay_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
