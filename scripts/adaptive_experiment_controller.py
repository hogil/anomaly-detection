"""
Minimal adaptive experiment controller for queued train.py runs.

It launches runs sequentially, parses best_info.json/test_history.json for
F1/FN/FP, and stops once the configured target condition is satisfied.

Queue format (JSON):
{
  "defaults": {
    "seeds": [42, 1, 2, 3, 4],
    "args": {"--mode": "binary"}
  },
  "candidates": [
    {
      "name": "adaptive_gc1",
      "tag_template": "adaptive_gc1_s{seed}",
      "args": {"--grad_clip": 1.0}
    }
  ],
  "runs": [
    {
      "tag": "adaptive_gc1_s42",
      "seed": 42,
      "args": {
        "--train_config": "configs/train/winning.yaml",
        "--config": "configs/datasets/v11.yaml",
        "--grad_clip": 1.0,
        "--smooth_window": 3
      }
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
DEFAULT_SUMMARY = ROOT / "validations" / "adaptive_controller_summary.json"
DEFAULT_MARKDOWN = ROOT / "validations" / "adaptive_controller_summary.md"


def load_queue(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        runs = payload
    else:
        runs = expand_candidates(payload) + payload.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"queue has no runs: {path}")
    for idx, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"queue run #{idx} is not an object")
        if not run.get("tag"):
            raise ValueError(f"queue run #{idx} is missing tag")
        if "seed" not in run:
            raise ValueError(f"queue run #{idx} is missing seed")
    return runs


def merge_args(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        merged.update(override)
        return merged
    if isinstance(base, list) and isinstance(override, list):
        return [*base, *override]
    if base in (None, {}, []):
        return override
    if override in (None, {}, []):
        return base
    raise ValueError("cannot merge args with different non-empty types")


def expand_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = payload.get("defaults", {})
    default_seeds = defaults.get("seeds", [42, 1, 2, 3, 4])
    default_args = defaults.get("args", {})
    runs: list[dict[str, Any]] = []
    for candidate in payload.get("candidates", []):
        if not isinstance(candidate, dict):
            raise ValueError("candidate entry is not an object")
        name = candidate.get("name")
        if not name:
            raise ValueError("candidate is missing name")
        seeds = candidate.get("seeds", default_seeds)
        tag_template = candidate.get("tag_template", f"{name}_s{{seed}}")
        args = merge_args(default_args, candidate.get("args", {}))
        for seed in seeds:
            runs.append({
                "tag": str(tag_template).format(seed=seed, name=name),
                "candidate": str(name),
                "seed": int(seed),
                "args": args,
            })
    return runs


def candidate_name(run: dict[str, Any]) -> str:
    if run.get("candidate"):
        return str(run["candidate"])
    tag = str(run["tag"])
    return re.sub(r"_s\d+$", "", tag)


def load_summary(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            text = path.read_text(encoding="utf-8")
            if text.strip():
                return json.loads(text)
        except json.JSONDecodeError:
            backup = path.with_suffix(f"{path.suffix}.corrupt-{now_iso().replace(':', '')}.bak")
            path.replace(backup)
    return {
        "created_at": now_iso(),
        "updated_at": None,
        "target": {"min": None, "max": None},
        "stop_mode": None,
        "runs": {},
        "skipped_candidates": {},
        "decision": "not_started",
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run_dir_matches_tag(path: Path, tag: str) -> bool:
    name = path.name
    prefix = re.match(r"^\d{6}_\d{6}_(.+)$", name)
    condition = prefix.group(1) if prefix else name
    return condition == tag or condition.startswith(f"{tag}_F")


def find_completed_run_dir(tag: str) -> Path | None:
    matches = [
        path
        for path in LOGS_DIR.glob(f"*_{tag}*")
        if path.is_dir() and (path / "best_info.json").exists() and run_dir_matches_tag(path, tag)
    ]
    if not matches:
        direct = LOGS_DIR / tag
        if direct.is_dir() and (direct / "best_info.json").exists():
            return direct
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def last_metric_event(run_dir: Path, best_info: dict[str, Any]) -> dict[str, Any]:
    history_path = run_dir / "test_history.json"
    history = read_json(history_path)
    if not isinstance(history, list):
        history = best_info.get("test_history", [])
    events = [event for event in history if isinstance(event, dict)]
    metric_events = [
        event
        for event in events
        if event.get("test_f1") is not None and (event.get("fn") is not None or event.get("fp") is not None)
    ]
    if metric_events:
        new_best = [event for event in metric_events if event.get("event") in ("NEW_BEST", "TIE")]
        return (new_best or metric_events)[-1]
    return {}


def fallback_counts(best_info: dict[str, Any], event: dict[str, Any]) -> tuple[int | None, int | None]:
    fn = event.get("fn")
    fp = event.get("fp")
    if fn is not None and fp is not None:
        return int(fn), int(fp)

    metrics = best_info.get("test_metrics", {})
    abnormal = metrics.get("abnormal", {})
    normal = metrics.get("normal", {})

    abn_recall = event.get("test_abn_R", abnormal.get("recall", best_info.get("test_abn_recall")))
    nor_recall = event.get("test_nor_R", normal.get("recall", best_info.get("test_nor_recall")))
    abn_count = abnormal.get("count", abnormal.get("support"))
    nor_count = normal.get("count", normal.get("support"))

    if abn_recall is not None and abn_count is not None:
        fn = round((1.0 - float(abn_recall)) * int(abn_count))
    if nor_recall is not None and nor_count is not None:
        fp = round((1.0 - float(nor_recall)) * int(nor_count))

    return (int(fn) if fn is not None else None, int(fp) if fp is not None else None)


def parse_metrics(run_dir: Path, run: dict[str, Any], target_min: int, target_max: int) -> dict[str, Any]:
    best_info = read_json(run_dir / "best_info.json")
    if not isinstance(best_info, dict):
        raise RuntimeError(f"cannot read best_info.json: {run_dir}")

    event = last_metric_event(run_dir, best_info)
    fn, fp = fallback_counts(best_info, event)
    f1 = event.get("test_f1", best_info.get("test_f1"))
    epoch = event.get("epoch", best_info.get("epoch", best_info.get("best_epoch")))
    in_target = fn is not None and fp is not None and target_min <= fn <= target_max and target_min <= fp <= target_max

    return {
        "tag": run["tag"],
        "candidate": candidate_name(run),
        "seed": int(run["seed"]),
        "run_dir": str(run_dir.relative_to(ROOT)),
        "status": "complete",
        "epoch": epoch,
        "event": event.get("event"),
        "test_f1": float(f1) if f1 is not None else None,
        "fn": fn,
        "fp": fp,
        "target_hit": in_target,
        "completed_at": now_iso(),
    }


def flatten_train_args(run: dict[str, Any]) -> list[str]:
    args = run.get("args", {})
    if isinstance(args, list):
        return [str(item) for item in args]
    if not isinstance(args, dict):
        raise ValueError(f"run {run['tag']} args must be an object or list")

    out: list[str] = []
    for key, value in args.items():
        flag = str(key)
        if not flag.startswith("--"):
            flag = f"--{flag}"
        if isinstance(value, bool):
            if value:
                out.append(flag)
            continue
        if value is None:
            continue
        out.extend([flag, str(value)])
    return out


def build_command(run: dict[str, Any]) -> list[str]:
    cmd = [sys.executable, "train.py"]
    cmd.extend(flatten_train_args(run))
    cmd.extend(["--seed", str(run["seed"]), "--log_dir", str(run["tag"])])
    return cmd


def launch_run(run: dict[str, Any], dry_run: bool) -> int:
    cmd = build_command(run)
    print(f"\n=== RUN {run['tag']} seed={run['seed']} ===")
    print(" ".join(cmd))
    if dry_run:
        return 0
    started = time.time()
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    elapsed = round(time.time() - started, 1)
    print(f"=== EXIT {result.returncode} elapsed={elapsed}s tag={run['tag']} ===")
    return int(result.returncode)


def target_seeds(summary: dict[str, Any]) -> set[int]:
    seeds: set[int] = set()
    for record in summary.get("runs", {}).values():
        if record.get("target_hit"):
            seeds.add(int(record["seed"]))
    return seeds


def target_seeds_for_candidate(summary: dict[str, Any], candidate: str) -> set[int]:
    seeds: set[int] = set()
    for record in summary.get("runs", {}).values():
        if record.get("candidate") == candidate and record.get("target_hit"):
            seeds.add(int(record["seed"]))
    return seeds


def decide(summary: dict[str, Any], stop_mode: str, required_by_candidate: dict[str, set[int]]) -> str:
    records = list(summary.get("runs", {}).values())
    hits = [record for record in records if record.get("target_hit")]
    if stop_mode == "first-hit" and hits:
        return "stop:first_hit"
    if stop_mode == "all-seeds":
        for candidate, required_seeds in required_by_candidate.items():
            if required_seeds and required_seeds.issubset(target_seeds_for_candidate(summary, candidate)):
                return f"stop:all_seeds_hit:{candidate}"
    return "continue"


def should_skip_candidate(
    summary: dict[str, Any],
    candidate: str,
    min_runs: int,
    target_min: int,
    target_max: int,
) -> str | None:
    if min_runs <= 0:
        return None
    records = [
        r for r in summary.get("runs", {}).values()
        if r.get("candidate") == candidate and r.get("status") == "complete"
    ]
    if len(records) < min_runs or any(r.get("target_hit") for r in records):
        return None
    fns = [r["fn"] for r in records if r.get("fn") is not None]
    fps = [r["fp"] for r in records if r.get("fp") is not None]
    if len(fns) != len(records) or len(fps) != len(records):
        return None
    fn_mu = mean(fns)
    fp_mu = mean(fps)
    if fn_mu < target_min and fp_mu < target_min:
        return f"too_easy_after_{len(records)}_runs:fn_mean={fn_mu:.2f},fp_mean={fp_mu:.2f}"
    if fn_mu > target_max and fp_mu > target_max:
        return f"too_hard_after_{len(records)}_runs:fn_mean={fn_mu:.2f},fp_mean={fp_mu:.2f}"
    return None


def update_aggregates(summary: dict[str, Any]) -> None:
    records = [r for r in summary.get("runs", {}).values() if r.get("status") == "complete"]
    f1s = [r["test_f1"] for r in records if r.get("test_f1") is not None]
    fns = [r["fn"] for r in records if r.get("fn") is not None]
    fps = [r["fp"] for r in records if r.get("fp") is not None]
    summary["aggregates"] = {
        "complete": len(records),
        "target_hits": sum(1 for r in records if r.get("target_hit")),
        "f1_mean": round(mean(f1s), 6) if f1s else None,
        "f1_std": round(pstdev(f1s), 6) if len(f1s) > 1 else 0.0,
        "fn_mean": round(mean(fns), 3) if fns else None,
        "fp_mean": round(mean(fps), 3) if fps else None,
        "target_hit_seeds": sorted(target_seeds(summary)),
    }
    by_candidate: dict[str, dict[str, Any]] = {}
    for candidate in sorted({r.get("candidate", "") for r in records}):
        candidate_records = [r for r in records if r.get("candidate") == candidate]
        if not candidate:
            continue
        cf1s = [r["test_f1"] for r in candidate_records if r.get("test_f1") is not None]
        cfns = [r["fn"] for r in candidate_records if r.get("fn") is not None]
        cfps = [r["fp"] for r in candidate_records if r.get("fp") is not None]
        by_candidate[candidate] = {
            "complete": len(candidate_records),
            "target_hits": sum(1 for r in candidate_records if r.get("target_hit")),
            "target_hit_seeds": sorted(target_seeds_for_candidate(summary, candidate)),
            "f1_mean": round(mean(cf1s), 6) if cf1s else None,
            "f1_std": round(pstdev(cf1s), 6) if len(cf1s) > 1 else 0.0,
            "fn_mean": round(mean(cfns), 3) if cfns else None,
            "fp_mean": round(mean(cfps), 3) if cfps else None,
        }
    summary["aggregates"]["by_candidate"] = by_candidate


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    rows = sorted(summary.get("runs", {}).values(), key=lambda r: (r.get("seed", 0), r.get("tag", "")))
    lines = [
        "# Adaptive Controller Summary",
        "",
        f"- Decision: `{summary.get('decision')}`",
        f"- Target FN/FP: `{summary.get('target', {}).get('min')}` to `{summary.get('target', {}).get('max')}` per seed",
        f"- Stop mode: `{summary.get('stop_mode')}`",
        "",
        "## Candidates",
        "",
        "| Candidate | Complete | Hits | Hit Seeds | F1 Mean | FN Mean | FP Mean |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    by_candidate = summary.get("aggregates", {}).get("by_candidate", {})
    for candidate, row in sorted(by_candidate.items()):
        f1 = row.get("f1_mean")
        lines.append(
            "| {candidate} | {complete} | {hits} | {seeds} | {f1} | {fn} | {fp} |".format(
                candidate=candidate,
                complete=row.get("complete", 0),
                hits=row.get("target_hits", 0),
                seeds=",".join(str(x) for x in row.get("target_hit_seeds", [])),
                f1=f"{f1:.4f}" if isinstance(f1, float) else "",
                fn=row.get("fn_mean", ""),
                fp=row.get("fp_mean", ""),
            )
        )
    lines.extend([
        "",
        "## Runs",
        "",
        "| Tag | Seed | F1 | FN | FP | Hit | Epoch | Run Dir |",
        "| --- | ---: | ---: | ---: | ---: | :--: | ---: | --- |",
    ])
    for row in rows:
        f1 = row.get("test_f1")
        lines.append(
            "| {tag} | {seed} | {f1} | {fn} | {fp} | {hit} | {epoch} | {run_dir} |".format(
                tag=row.get("tag"),
                seed=row.get("seed"),
                f1=f"{f1:.4f}" if isinstance(f1, float) else "",
                fn=row.get("fn", ""),
                fp=row.get("fp", ""),
                hit="yes" if row.get("target_hit") else "no",
                epoch=row.get("epoch", ""),
                run_dir=row.get("run_dir", ""),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sequential adaptive train.py controller")
    parser.add_argument("--queue", required=True, type=Path, help="JSON queue file")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--target-min", type=int, default=5)
    parser.add_argument("--target-max", type=int, default=15)
    parser.add_argument("--stop-mode", choices=["first-hit", "all-seeds", "never"], default="all-seeds")
    parser.add_argument(
        "--candidate-min-runs-before-skip",
        type=int,
        default=0,
        help="If >0, skip remaining runs for a candidate after this many complete runs when it is clearly too easy/hard.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching train.py")
    parser.add_argument("--force", action="store_true", help="Run even if a completed tag already exists")
    parser.add_argument(
        "--max-launched",
        type=int,
        default=0,
        help="If >0, return after launching this many new train.py runs so a master loop can inspect results.",
    )
    args = parser.parse_args()

    runs = load_queue(args.queue)
    required_by_candidate: dict[str, set[int]] = {}
    for run in runs:
        required_by_candidate.setdefault(candidate_name(run), set()).add(int(run["seed"]))
    summary = load_summary(args.summary)
    summary["updated_at"] = now_iso()
    summary["target"] = {"min": args.target_min, "max": args.target_max}
    summary["stop_mode"] = args.stop_mode
    launched = 0

    for run in runs:
        decision = "continue" if args.stop_mode == "never" else decide(summary, args.stop_mode, required_by_candidate)
        if decision != "continue":
            summary["decision"] = decision
            break

        candidate = candidate_name(run)
        skip_reason = summary.get("skipped_candidates", {}).get(candidate)
        if not skip_reason:
            skip_reason = should_skip_candidate(
                summary,
                candidate,
                args.candidate_min_runs_before_skip,
                args.target_min,
                args.target_max,
            )
            if skip_reason:
                summary.setdefault("skipped_candidates", {})[candidate] = skip_reason
        if skip_reason:
            summary["runs"][run["tag"]] = {
                "tag": run["tag"],
                "candidate": candidate,
                "seed": int(run["seed"]),
                "status": "skipped",
                "skip_reason": skip_reason,
                "target_hit": False,
            }
            update_aggregates(summary)
            summary["decision"] = "continue"
            save_json(args.summary, summary)
            write_markdown(args.markdown, summary)
            continue

        existing = None if args.force else find_completed_run_dir(run["tag"])
        launched_this_run = False
        if existing is None:
            code = launch_run(run, args.dry_run)
            launched_this_run = not args.dry_run
            if launched_this_run:
                launched += 1
            if args.dry_run:
                summary["runs"][run["tag"]] = {
                    "tag": run["tag"],
                    "candidate": candidate,
                    "seed": int(run["seed"]),
                    "status": "dry_run",
                    "command": build_command(run),
                    "target_hit": False,
                }
                continue
            if code != 0:
                summary["runs"][run["tag"]] = {
                    "tag": run["tag"],
                    "candidate": candidate,
                    "seed": int(run["seed"]),
                    "status": "failed",
                    "exit_code": code,
                    "completed_at": now_iso(),
                    "target_hit": False,
                }
                summary["decision"] = f"stop:run_failed:{run['tag']}"
                save_json(args.summary, summary)
                write_markdown(args.markdown, summary)
                return code
            existing = find_completed_run_dir(run["tag"])
            if existing is None:
                raise RuntimeError(f"train.py finished but no best_info.json was found for tag {run['tag']}")

        summary["runs"][run["tag"]] = parse_metrics(existing, run, args.target_min, args.target_max)
        update_aggregates(summary)
        summary["decision"] = "continue"
        save_json(args.summary, summary)
        write_markdown(args.markdown, summary)
        if args.max_launched > 0 and launched >= args.max_launched:
            break

    if summary.get("decision") in ("not_started", "continue"):
        summary["decision"] = "queue_exhausted"
    update_aggregates(summary)
    save_json(args.summary, summary)
    write_markdown(args.markdown, summary)
    print(json.dumps({"decision": summary["decision"], "aggregates": summary.get("aggregates", {})}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
