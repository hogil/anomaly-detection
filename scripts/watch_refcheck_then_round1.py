#!/usr/bin/env python
"""Wait for raw refcheck completion, then launch rawbase strict round1."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO


ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log_line(handle: TextIO, message: str) -> None:
    handle.write(f"[{now()}] {message}\n")
    handle.flush()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def completed_run_count(summary: dict[str, Any]) -> int:
    runs = summary.get("runs", {})
    if not isinstance(runs, dict):
        return 0
    return sum(1 for record in runs.values() if record.get("status") == "complete")


def wait_for_refcheck(args: argparse.Namespace, handle: TextIO) -> None:
    deadline = time.time() + args.timeout_seconds if args.timeout_seconds > 0 else None
    last_status = ""
    while True:
        summary = read_json(args.ref_summary)
        decision = summary.get("decision") if summary else "missing"
        complete = completed_run_count(summary) if summary else 0
        status = f"decision={decision} complete={complete}/{args.min_ref_runs}"
        if status != last_status:
            log_line(handle, f"refcheck status: {status}")
            last_status = status
        if decision == "queue_exhausted" and complete >= args.min_ref_runs:
            return
        if summary and isinstance(decision, str) and decision.startswith("stop:"):
            raise RuntimeError(f"raw refcheck stopped before completion: {decision}")
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError(f"timed out waiting for {args.ref_summary}")
        time.sleep(args.poll_seconds)


def run_logged(cmd: list[str], handle: TextIO, *, output_path: Path | None = None) -> int:
    log_line(handle, "+ " + " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")

    if output_path is None:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if proc.stdout:
            handle.write(proc.stdout)
            if not proc.stdout.endswith("\n"):
                handle.write("\n")
        handle.flush()
        return int(proc.returncode)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", errors="replace") as out:
        out.write(f"\n[{now()}] + {' '.join(cmd)}\n")
        out.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return int(proc.wait())


def launch_round1(args: argparse.Namespace, handle: TextIO) -> int:
    prepare_cmd = [
        sys.executable,
        "scripts/prepare_server_queue.py",
        "--src",
        str(args.src_queue),
        "--dst",
        str(args.prepared_queue),
        "--config",
        args.config,
        "--num-workers",
        str(args.num_workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
    ]
    if args.start_after_axis:
        prepare_cmd.extend(["--start-after-axis", args.start_after_axis])
    if args.start_after_candidate:
        prepare_cmd.extend(["--start-after-candidate", args.start_after_candidate])
    if (
        args.skip_completed_summary
        and not args.keep_completed_in_queue
        and not args.force
    ):
        prepare_cmd.extend(["--skip-completed-summary", str(args.skip_completed_summary)])
    if args.include_axes:
        prepare_cmd.extend(["--include-axes", args.include_axes])

    code = run_logged(prepare_cmd, handle)
    if code != 0:
        return code

    controller_cmd = [
        sys.executable,
        "scripts/adaptive_experiment_controller.py",
        "--queue",
        str(args.prepared_queue),
        "--summary",
        str(args.round1_summary),
        "--markdown",
        str(args.round1_markdown),
        "--target-min",
        str(args.target_min),
        "--target-max",
        str(args.target_max),
        "--stop-mode",
        "never",
        "--candidate-min-runs-before-skip",
        "0",
        "--completion-exit-grace",
        str(args.completion_exit_grace),
        "--update-live-summary",
    ]
    if args.force:
        controller_cmd.append("--force")
    if args.max_launched > 0:
        controller_cmd.extend(["--max-launched", str(args.max_launched)])

    return run_logged(controller_cmd, handle, output_path=args.round1_log)


def launch_optional_pre_round1(args: argparse.Namespace, handle: TextIO) -> int:
    if args.pre_round1_queue is None:
        return 0
    existing = read_json(args.pre_round1_summary)
    if existing and existing.get("decision") == "queue_exhausted":
        log_line(handle, f"pre-round1 queue already complete: {args.pre_round1_summary}")
        return 0

    controller_cmd = [
        sys.executable,
        "scripts/adaptive_experiment_controller.py",
        "--queue",
        str(args.pre_round1_queue),
        "--summary",
        str(args.pre_round1_summary),
        "--markdown",
        str(args.pre_round1_markdown),
        "--target-min",
        str(args.target_min),
        "--target-max",
        str(args.target_max),
        "--stop-mode",
        "never",
        "--candidate-min-runs-before-skip",
        "0",
        "--update-live-summary",
        "--completion-exit-grace",
        str(args.completion_exit_grace),
    ]
    log_line(handle, f"launching pre-round1 queue: {args.pre_round1_queue}")
    return run_logged(controller_cmd, handle, output_path=args.pre_round1_log)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-summary", type=Path, default=ROOT / "validations/server_paper_refcheck_raw_summary.json")
    parser.add_argument("--src-queue", type=Path, default=ROOT / "validations/paper_strict_single_factor_queue.json")
    parser.add_argument("--prepared-queue", type=Path, default=ROOT / "validations/server_paper_rawbase_strict_single_factor_queue.json")
    parser.add_argument("--round1-summary", type=Path, default=ROOT / "validations/server_paper_rawbase_strict_single_factor_summary.json")
    parser.add_argument("--round1-markdown", type=Path, default=ROOT / "validations/server_paper_rawbase_strict_single_factor_summary.md")
    parser.add_argument("--skip-completed-summary", type=Path, default=ROOT / "validations/server_paper_rawbase_strict_single_factor_summary.json")
    parser.add_argument("--watch-log", type=Path, default=ROOT / "validations/paper_rawbase_round1_watcher.log")
    parser.add_argument("--round1-log", type=Path, default=ROOT / "validations/paper_rawbase_round1_live.log")
    parser.add_argument("--pre-round1-queue", type=Path, default=None)
    parser.add_argument("--pre-round1-summary", type=Path, default=ROOT / "validations/server_paper_pre_round1_summary.json")
    parser.add_argument("--pre-round1-markdown", type=Path, default=ROOT / "validations/server_paper_pre_round1_summary.md")
    parser.add_argument("--pre-round1-log", type=Path, default=ROOT / "validations/paper_pre_round1_live.log")
    parser.add_argument("--config", default="dataset.yaml")
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--target-min", type=int, default=5)
    parser.add_argument("--target-max", type=int, default=15)
    parser.add_argument("--min-ref-runs", type=int, default=5)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    parser.add_argument("--completion-exit-grace", type=float, default=15.0)
    parser.add_argument("--start-after-axis", default="")
    parser.add_argument("--start-after-candidate", default="")
    parser.add_argument("--include-axes", default="")
    parser.add_argument("--keep-completed-in-queue", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-launched", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.watch_log.parent.mkdir(parents=True, exist_ok=True)
    with args.watch_log.open("a", encoding="utf-8", errors="replace") as handle:
        log_line(handle, "watcher started")
        wait_for_refcheck(args, handle)
        existing_round1 = read_json(args.round1_summary)
        if existing_round1 and existing_round1.get("decision") == "queue_exhausted":
            log_line(handle, f"round1 already complete: {args.round1_summary}")
            return 0
        code = launch_optional_pre_round1(args, handle)
        if code != 0:
            log_line(handle, f"pre-round1 controller exited with code {code}; round1 not launched")
            return code
        log_line(handle, "raw refcheck complete; launching rawbase round1")
        code = launch_round1(args, handle)
        log_line(handle, f"round1 controller exited with code {code}")
        return code


if __name__ == "__main__":
    raise SystemExit(main())
