"""Publish strict one-factor report from validations/ to docs/reports/.

This keeps auto-generated local artifacts under validations/ while exporting a
GitHub-friendly snapshot under docs/reports/strict_one_factor_latest/.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_REPORT = ROOT / "validations" / "paper_strict_single_factor_report.md"
DEFAULT_SOURCE_SUMMARY = ROOT / "validations" / "paper_strict_single_factor_summary.md"
DEFAULT_SOURCE_ROUND2 = ROOT / "validations" / "paper_strict_single_factor_round2_summary.md"
DEFAULT_SOURCE_PLOTS = ROOT / "validations" / "paper_strict_single_factor_plots"
DEFAULT_TARGET_DIR = ROOT / "docs" / "reports" / "strict_one_factor_latest"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Publish strict report snapshot into docs/reports.")
    ap.add_argument("--source-report", default=str(DEFAULT_SOURCE_REPORT))
    ap.add_argument("--source-summary", default=str(DEFAULT_SOURCE_SUMMARY))
    ap.add_argument("--source-round2", default=str(DEFAULT_SOURCE_ROUND2))
    ap.add_argument("--source-plots", default=str(DEFAULT_SOURCE_PLOTS))
    ap.add_argument("--target-dir", default=str(DEFAULT_TARGET_DIR))
    return ap.parse_args()


def rewrite_report_text(text: str) -> str:
    return text.replace("paper_strict_single_factor_plots/", "plots/")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_plots(source_dir: Path, target_dir: Path) -> list[str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for src in sorted(source_dir.glob("*.png")):
        dst = target_dir / src.name
        shutil.copy2(src, dst)
        copied.append(src.name)
    return copied


def main() -> int:
    args = parse_args()
    source_report = Path(args.source_report)
    source_summary = Path(args.source_summary)
    source_round2 = Path(args.source_round2)
    source_plots = Path(args.source_plots)
    target_dir = Path(args.target_dir)

    if not source_report.exists():
        raise FileNotFoundError(f"source report not found: {source_report}")
    if not source_summary.exists():
        raise FileNotFoundError(f"source summary not found: {source_summary}")
    if not source_plots.exists():
        raise FileNotFoundError(f"source plots dir not found: {source_plots}")

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    report_text = rewrite_report_text(source_report.read_text(encoding="utf-8"))
    summary_text = rewrite_report_text(source_summary.read_text(encoding="utf-8"))
    round2_text = rewrite_report_text(source_round2.read_text(encoding="utf-8")) if source_round2.exists() else ""

    write_text(target_dir / "report.md", report_text)
    write_text(target_dir / "summary.md", summary_text)
    if round2_text:
        write_text(target_dir / "round2_summary.md", round2_text)

    copied = copy_plots(source_plots, target_dir / "plots")

    index = [
        "# Strict One-Factor Latest",
        "",
        "GitHub-published snapshot of the current strict one-factor experiment report.",
        "",
        "## Files",
        "",
        "- [report.md](report.md)",
        "- [summary.md](summary.md)",
    ]
    if round2_text:
        index.append("- [round2_summary.md](round2_summary.md)")
    index.extend(
        [
            "",
            "## Plots",
            "",
        ]
    )
    for name in copied:
        index.append(f"- [{name}](plots/{name})")
    write_text(target_dir / "README.md", "\n".join(index) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
