#!/usr/bin/env python
"""Generate Grad-CAM overlays and heat summaries for saved models."""

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
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import _load_model_from_best_info

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
RIGHT_START = 0.70


def load_font(size: int) -> ImageFont.ImageFont:
    for font_name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit_multiline_font(text: str, max_width: int, max_height: int, start_size: int, min_size: int) -> ImageFont.ImageFont:
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    for size in range(start_size, min_size - 1, -1):
        font = load_font(size)
        bbox = probe.multiline_textbbox((0, 0), text, font=font, spacing=4, align="center")
        if bbox[2] - bbox[0] <= max_width and bbox[3] - bbox[1] <= max_height:
            return font
    return load_font(min_size)


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


def save_heat_only_cam(cam: np.ndarray, out_path: Path, threshold: float) -> None:
    """Save only the CAM heat as RGBA; non-heat pixels are transparent."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cam_to_rgba(cam, threshold, min_alpha=0.0)).save(out_path)


def cam_to_rgba(cam: np.ndarray, threshold: float, min_alpha: float = 0.0) -> np.ndarray:
    """Convert normalized CAM to RGBA with transparent low-heat pixels."""
    cam = np.asarray(cam, dtype=np.float32)
    threshold = float(np.clip(threshold, 0.0, 0.95))
    min_alpha = float(np.clip(min_alpha, 0.0, 0.95))
    alpha = np.clip((cam - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    alpha = np.where(alpha > 0.0, min_alpha + alpha * (1.0 - min_alpha), 0.0)
    heat = plt.get_cmap("jet")(cam)
    rgba = np.zeros((*cam.shape, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(heat[..., :3] * 255.0, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return rgba


def save_cam_on_image(image_path: Path, cam: np.ndarray, out_path: Path, threshold: float, min_alpha: float) -> None:
    """Overlay thresholded CAM heat on top of the original trend image."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = Image.open(image_path).convert("RGBA").resize((224, 224))
    heat = Image.fromarray(cam_to_rgba(cam, threshold, min_alpha=min_alpha)).resize((224, 224))
    base.alpha_composite(heat)
    base.save(out_path)


def build_cam_overlay_gallery(rows: list[dict[str, Any]], out_path: Path) -> None:
    """Build a class-row gallery from trend images with CAM overlay."""
    preferred = ["normal", "mean_shift", "standard_deviation", "spike", "drift", "context"]
    by_class: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        overlay_path = str(row.get("cam_overlay", ""))
        if overlay_path and Path(overlay_path).exists():
            by_class.setdefault(str(row["true_class"]), []).append(row)
    if not by_class:
        return

    class_order = [name for name in preferred if name in by_class]
    class_order.extend(sorted(name for name in by_class if name not in class_order))
    cols = max(len(by_class[name]) for name in class_order)
    label_w, cell_w, cell_h = 360, 224, 224
    pad = 16
    canvas_w = label_w + cols * cell_w + (cols + 1) * pad
    canvas_h = len(class_order) * cell_h + (len(class_order) + 1) * pad
    canvas = Image.new("RGBA", (canvas_w, canvas_h), "WHITE")
    draw = ImageDraw.Draw(canvas)

    for row_idx, class_name in enumerate(class_order):
        y = pad + row_idx * (cell_h + pad)
        label_text = class_name.replace("_", "\n")
        label_font = fit_multiline_font(label_text, label_w - 2 * pad, cell_h - 2 * pad, 54, 28)
        label_bbox = draw.multiline_textbbox((0, 0), label_text, font=label_font, spacing=6, align="center")
        label_text_w = label_bbox[2] - label_bbox[0]
        label_text_h = label_bbox[3] - label_bbox[1]
        label_x = pad + (label_w - 2 * pad - label_text_w) // 2
        label_y = y + (cell_h - label_text_h) // 2
        draw.multiline_text(
            (label_x, label_y),
            label_text,
            fill=(0, 0, 0, 255),
            font=label_font,
            spacing=6,
            align="center",
        )
        for col_idx, row in enumerate(by_class[class_name]):
            x = label_w + pad + col_idx * (cell_w + pad)
            overlay = Image.open(str(row["cam_overlay"])).convert("RGBA").resize((224, 224))
            canvas.alpha_composite(overlay, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


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
            "- CAM is model evidence location, not guaranteed anomaly location.",
            "- Broad/generated defects can produce broad heat; local defects such as spikes often produce tighter heat.",
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
    parser.add_argument("--save-heat-only", action="store_true", help="also save transparent heat-only CAM PNGs")
    parser.add_argument("--heat-threshold", type=float, default=0.20, help="alpha threshold for heat-only CAM PNGs")
    parser.add_argument("--heat-min-alpha", type=float, default=0.0, help="minimum alpha for visible CAM pixels")
    parser.add_argument("--gallery-out", default=None, type=Path, help="optional class-level trend+CAM gallery PNG")
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
    heat_only_dir = args.out_dir / "heat_only"
    cam_overlay_dir = args.out_dir / "cam_on_image"
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
            heat_only_path = ""
            cam_overlay_path = ""
            if args.save_heat_only or args.gallery_out is not None:
                heat_path = heat_only_dir / out_name
                save_heat_only_cam(cam, heat_path, args.heat_threshold)
                heat_only_path = str(heat_path)
                sparse_overlay_path = cam_overlay_dir / out_name
                save_cam_on_image(image_path, cam, sparse_overlay_path, args.heat_threshold, args.heat_min_alpha)
                cam_overlay_path = str(sparse_overlay_path)
            rows.append(
                {
                    "image": str(image_path),
                    "true_class": true_class,
                    "pred_class": pred_class,
                    "explained_class": explained_class,
                    "p_abnormal": p_abnormal,
                    "overlay": str(overlay_path),
                    "heat_only": heat_only_path,
                    "cam_overlay": cam_overlay_path,
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
            "heat_only",
            "cam_overlay",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.out_dir / "summary.md"
    summary_path.write_text(summarize_rows(rows), encoding="utf-8")
    if args.gallery_out is not None:
        build_cam_overlay_gallery(rows, args.gallery_out)
        print(f"[gradcam] gallery={args.gallery_out}")
    print(f"[gradcam] wrote {csv_path}")
    print(f"[gradcam] wrote {summary_path}")
    print(f"[gradcam] overlays={overlay_dir}")
    if args.save_heat_only or args.gallery_out is not None:
        print(f"[gradcam] heat_only={heat_only_dir}")
        print(f"[gradcam] cam_on_image={cam_overlay_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
