"""Publish strict one-factor report from validations/ to docs/.

Source of truth stays under validations/.
GitHub-facing snapshot is exported directly to:
  - docs/summary.md
  - docs/round2_summary.md
  - docs/plots/*.png
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_REPORT = ROOT / "validations" / "paper_strict_single_factor_report.md"
DEFAULT_SOURCE_ROUND2 = ROOT / "validations" / "paper_strict_single_factor_round2_summary.md"
DEFAULT_SOURCE_PLOTS = ROOT / "validations" / "paper_strict_single_factor_plots"
DEFAULT_DOCS_DIR = ROOT / "docs"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Publish strict report snapshot into docs/.")
    ap.add_argument("--source-report", default=str(DEFAULT_SOURCE_REPORT))
    ap.add_argument("--source-round2", default=str(DEFAULT_SOURCE_ROUND2))
    ap.add_argument("--source-plots", default=str(DEFAULT_SOURCE_PLOTS))
    ap.add_argument("--docs-dir", default=str(DEFAULT_DOCS_DIR))
    return ap.parse_args()


def rewrite_report_text(text: str) -> str:
    replacements = {
        "# Strict Single-Factor Report": "# 실험 요약",
        "_Auto-updated at `": "_자동 갱신 시각: `",
        "## Experiment Protocol": "## 실험 방식",
        "## Performance Summary": "## 성능 요약",
        "## Summary": "## 요약",
        "## Interpretation": "## 결과 해석",
        "## Evidence Limits And Next Fixes": "## 한계와 수정 필요 사항",
        "## Best Known Method": "## Best Known Method",
        "_This is the current best-known method table from one-factor evidence. Joint combo validation still has to be run after round-2 closes._": "",
        "## Pending Round-2 Checks": "## 남은 Round-2 확인 항목",
        "## Plot Index": "## 플롯 목록",
        "Frozen ref:": "고정 기준 ref:",
        "Main strict queue:": "메인 strict queue:",
        "Round-2 refinement:": "Round-2 refinement:",
        "currently best completed point is": "현재 완료된 조건 중 최선은",
        "broad good band remains active; current lowest completed total error is around": "넓은 양호 구간이 유지되고 있으며 완료 조건 중 현재 총 오류가 가장 낮은 쪽은 대략",
        "completed runs": "완료 run",
        "completed run": "완료 run",
        "completed point": "완료된 조건",
        "broad good band remains active; current lowest total error is around": "넓은 양호 구간이 유지되고 있으며 현재 총 오류가 가장 낮은 쪽은 대략",
        "Incomplete values are kept out of the main table.": "미완료 값은 본 표에서 제외했습니다.",
        "helps recall but faint fleet hurts FP": "trend 빨강은 recall에 도움되고 fleet를 너무 연하게 하면 FP가 악화됩니다",
        "No queued round-2 conditions.": "대기 중인 round-2 조건이 없습니다.",
        "complete": "완료",
        "partial": "부분완료",
        "strict": "strict",
        "legacy": "legacy",
        "round2": "round2",
        "paper_strict_single_factor_plots/": "plots/",
    }
    for before, after in replacements.items():
        text = text.replace(before, after)
    return text


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_plots(source_dir: Path, target_dir: Path) -> list[str]:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for src in sorted(source_dir.glob("*.png")):
        dst = target_dir / src.name
        shutil.copy2(src, dst)
        copied.append(src.name)
    return copied


def write_docs_index(docs_dir: Path, has_round2: bool, plots: list[str]) -> None:
    lines = [
        "# 문서 개요",
        "",
        "- [summary.md](summary.md): 최신 strict one-factor 결과 요약, 해석, 축별 전체 표",
    ]
    if has_round2:
        lines.append("- [round2_summary.md](round2_summary.md): round-2 진행 현황")
    lines.extend([
        "- [repo_file_audit.md](repo_file_audit.md): GitHub tracked file 유지/정리 판단",
        "- [plots/](plots): 축별 성능 플롯",
        "",
        "## 플롯 파일",
        "",
    ])
    for name in plots:
        lines.append(f"- [{name}](plots/{name})")
    write_text(docs_dir / "README.md", "\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    source_report = Path(args.source_report)
    source_round2 = Path(args.source_round2)
    source_plots = Path(args.source_plots)
    docs_dir = Path(args.docs_dir)

    if not source_report.exists():
        raise FileNotFoundError(f"source report not found: {source_report}")
    if not source_plots.exists():
        raise FileNotFoundError(f"source plots dir not found: {source_plots}")

    report_text = rewrite_report_text(source_report.read_text(encoding="utf-8"))
    round2_text = rewrite_report_text(source_round2.read_text(encoding="utf-8")) if source_round2.exists() else ""

    write_text(docs_dir / "summary.md", report_text)
    if round2_text:
        write_text(docs_dir / "round2_summary.md", round2_text)
    elif (docs_dir / "round2_summary.md").exists():
        (docs_dir / "round2_summary.md").unlink()

    copied = copy_plots(source_plots, docs_dir / "plots")
    write_docs_index(docs_dir, bool(round2_text), copied)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
