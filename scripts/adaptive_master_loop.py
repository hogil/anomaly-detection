"""
Adaptive master loop for paper-reference search.

This script sits above adaptive_experiment_controller.py. It waits for any
currently running controller, reads the controller summary, appends new
candidate runs when the reference target is not met, and relaunches the
controller. The goal is an observable-error reference where every planned seed
has both FN and FP inside the configured band.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = ROOT / "validations" / "adaptive_ref_queue.json"
DEFAULT_SUMMARY = ROOT / "validations" / "adaptive_ref_summary.json"
DEFAULT_MARKDOWN = ROOT / "validations" / "adaptive_ref_summary.md"
DEFAULT_MASTER_STATE = ROOT / "validations" / "adaptive_master_state.json"
DEFAULT_MASTER_MARKDOWN = ROOT / "validations" / "adaptive_master_summary.md"
CONTROLLER = ROOT / "scripts" / "adaptive_experiment_controller.py"

DEFAULT_ARGS: dict[str, Any] = {
    "--mode": "binary",
    "--config": "configs/datasets/dataset_v11.yaml",
    "--epochs": 6,
    "--patience": 5,
    "--smooth_window": 3,
    "--smooth_method": "median",
    "--lr_backbone": "2e-5",
    "--lr_head": "2e-4",
    "--warmup_epochs": 0,
    "--grad_clip": 1.0,
    "--weight_decay": 0.01,
    "--ema_decay": 0.0,
    "--normal_ratio": 700,
    "--batch_size": 32,
    "--dropout": 0.0,
    "--min_epochs": 5,
    "--precision": "fp16",
    "--num_workers": 0,
}

DEFAULT_SEEDS = [42, 1, 2, 3, 4]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def queue_runs(queue: dict[str, Any]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    defaults = queue.get("defaults", {})
    default_seeds = defaults.get("seeds", DEFAULT_SEEDS)
    default_args = defaults.get("args", {})
    for candidate in queue.get("candidates", []):
        args = dict(default_args)
        args.update(candidate.get("args", {}))
        name = str(candidate["name"])
        tag_template = candidate.get("tag_template", f"{name}_s{{seed}}")
        for seed in candidate.get("seeds", default_seeds):
            runs.append({
                "tag": str(tag_template).format(seed=int(seed), name=name),
                "candidate": name,
                "seed": int(seed),
                "args": args,
            })
    runs.extend(queue.get("runs", []))
    return runs


def candidate_args_from_queue(queue: dict[str, Any], candidate: str) -> dict[str, Any]:
    for run in queue_runs(queue):
        if run.get("candidate") == candidate and run.get("args"):
            return dict(run["args"])
    return dict(DEFAULT_ARGS)


def existing_names(queue: dict[str, Any]) -> set[str]:
    names = {str(run.get("candidate", "")) for run in queue_runs(queue)}
    names.update(str(c.get("name", "")) for c in queue.get("candidates", []))
    return {name for name in names if name}


def candidate_records(summary: dict[str, Any], candidate: str) -> list[dict[str, Any]]:
    return [
        record
        for record in summary.get("runs", {}).values()
        if record.get("candidate") == candidate and record.get("status") == "complete"
    ]


def all_seed_hit_candidate(summary: dict[str, Any], seeds: list[int]) -> str | None:
    by_candidate = summary.get("aggregates", {}).get("by_candidate", {})
    required = set(int(seed) for seed in seeds)
    for candidate, row in by_candidate.items():
        hit_seeds = set(int(seed) for seed in row.get("target_hit_seeds", []))
        if required and required.issubset(hit_seeds):
            return str(candidate)
    return None


def distance_to_band(value: float | None, low: int, high: int) -> float:
    if value is None:
        return 999.0
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def candidate_score(row: dict[str, Any], target_min: int, target_max: int, seeds: list[int]) -> float:
    complete = int(row.get("complete") or 0)
    hit_count = int(row.get("target_hits") or 0)
    fn_gap = distance_to_band(row.get("fn_mean"), target_min, target_max)
    fp_gap = distance_to_band(row.get("fp_mean"), target_min, target_max)
    missing_penalty = max(len(seeds) - complete, 0) * 0.25
    hit_bonus = hit_count * 0.2
    return fn_gap + fp_gap + missing_penalty - hit_bonus


def choose_source_candidate(summary: dict[str, Any], target_min: int, target_max: int, seeds: list[int]) -> str | None:
    rows = summary.get("aggregates", {}).get("by_candidate", {})
    if not rows:
        return None
    ranked = sorted(
        rows.items(),
        key=lambda item: (
            candidate_score(item[1], target_min, target_max, seeds),
            -int(item[1].get("complete") or 0),
            str(item[0]),
        ),
    )
    return str(ranked[0][0]) if ranked else None


def numeric_arg(args: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(args.get(key, default))
    except (TypeError, ValueError):
        return default


def int_arg(args: dict[str, Any], key: str, default: int) -> int:
    return int(round(numeric_arg(args, key, default)))


def clamp_int(value: float, low: int, high: int) -> int:
    return max(low, min(high, int(round(value))))


def clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def format_token(value: float | int | str) -> str:
    text = str(value).replace(".", "p").replace("-", "m")
    return text.replace("+", "")


def make_candidate_runs(name: str, args: dict[str, Any], seeds: list[int], reason: str) -> list[dict[str, Any]]:
    runs = []
    for seed in seeds:
        runs.append({
            "tag": f"{name}_s{seed}",
            "candidate": name,
            "seed": int(seed),
            "args": args,
            "reason": reason,
        })
    return runs


def propose_candidates(
    queue: dict[str, Any],
    summary: dict[str, Any],
    state: dict[str, Any],
    *,
    target_min: int,
    target_max: int,
    seeds: list[int],
    max_new_candidates: int,
) -> list[dict[str, Any]]:
    source = choose_source_candidate(summary, target_min, target_max, seeds)
    rows = summary.get("aggregates", {}).get("by_candidate", {})
    source_row = rows.get(source or "", {})
    source_args = candidate_args_from_queue(queue, source or "")
    if not source_args:
        source_args = dict(DEFAULT_ARGS)

    fn_mean = source_row.get("fn_mean")
    fp_mean = source_row.get("fp_mean")
    if fn_mean is None:
        fn_mean = target_min - 1
    if fp_mean is None:
        fp_mean = target_min - 1

    round_no = int(state.get("round", 0)) + 1
    prefix = f"auto_ref_r{round_no:02d}"
    used = existing_names(queue)
    proposals: list[tuple[str, dict[str, Any], str]] = []

    base_epochs = int_arg(source_args, "--epochs", 6)
    base_nr = int_arg(source_args, "--normal_ratio", 700)
    base_aw = numeric_arg(source_args, "--abnormal_weight", 1.0)
    base_gc = numeric_arg(source_args, "--grad_clip", 1.0)
    base_wd = numeric_arg(source_args, "--weight_decay", 0.01)

    def add(suffix: str, patch: dict[str, Any], reason: str) -> None:
        args = dict(source_args)
        args.update(patch)
        args.setdefault("--mode", "binary")
        args.setdefault("--config", "configs/datasets/dataset_v11.yaml")
        args.setdefault("--precision", "fp16")
        args.setdefault("--num_workers", 0)
        name = f"{prefix}_{suffix}"
        if name not in used:
            proposals.append((name, args, reason))
            used.add(name)

    fp_low = float(fp_mean) < target_min
    fp_high = float(fp_mean) > target_max
    fn_low = float(fn_mean) < target_min
    fn_high = float(fn_mean) > target_max

    if fp_low:
        nr = clamp_int(base_nr * 0.75, 250, 1400)
        add(
            f"raisefp_nr{nr}_ep{base_epochs}",
            {"--normal_ratio": nr},
            f"FP mean {fp_mean} is below target; reduce normal_ratio to raise false positives.",
        )
        nr2 = clamp_int(base_nr * 0.55, 200, 1400)
        add(
            f"raisefp2_nr{nr2}_ep{base_epochs}",
            {"--normal_ratio": nr2},
            f"FP mean {fp_mean} is below target; stronger normal_ratio reduction.",
        )

    if fp_high:
        nr = clamp_int(base_nr * 1.35, 300, 2800)
        add(
            f"lowerfp_nr{nr}_ep{base_epochs}",
            {"--normal_ratio": nr},
            f"FP mean {fp_mean} is above target; increase normal_ratio to suppress false positives.",
        )

    if fn_low:
        epochs = clamp_int(base_epochs - 1, 4, 20)
        aw = round(clamp_float(base_aw * 0.85, 0.4, 2.5), 3)
        add(
            f"raisefn_ep{epochs}_aw{format_token(aw)}",
            {"--epochs": epochs, "--abnormal_weight": aw},
            f"FN mean {fn_mean} is below target; shorten training and lower abnormal weight.",
        )

    if fn_high:
        epochs = clamp_int(base_epochs + 2, 4, 24)
        aw = round(clamp_float(base_aw * 1.2, 0.4, 3.0), 3)
        add(
            f"lowerfn_ep{epochs}_aw{format_token(aw)}",
            {"--epochs": epochs, "--abnormal_weight": aw},
            f"FN mean {fn_mean} is above target; train longer and raise abnormal weight.",
        )

    if not proposals:
        # Local search around a close-but-not-perfect reference. This is common
        # when the mean is inside the target band but one or two seeds miss FP.
        nr_down = clamp_int(base_nr * 0.85, 250, 2800)
        nr_up = clamp_int(base_nr * 1.15, 250, 2800)
        aw_down = round(clamp_float(base_aw * 0.9, 0.4, 3.0), 3)
        aw_up = round(clamp_float(base_aw * 1.1, 0.4, 3.0), 3)
        add(
            f"local_nr{nr_down}_aw{format_token(base_aw)}",
            {"--normal_ratio": nr_down},
            "Reference is close but not all seeds hit; slightly reduce normal_ratio to lift low-FP seeds.",
        )
        add(
            f"local_nr{nr_up}_aw{format_token(base_aw)}",
            {"--normal_ratio": nr_up},
            "Reference is close but not all seeds hit; slightly increase normal_ratio to check FP/FN balance.",
        )
        add(
            f"local_aw{format_token(aw_down)}_gc{format_token(base_gc)}",
            {"--abnormal_weight": aw_down, "--grad_clip": base_gc},
            "Reference is close but not all seeds hit; lower abnormal weight to expose FN without changing protocol much.",
        )
        add(
            f"local_aw{format_token(aw_up)}_wd{format_token(base_wd)}",
            {"--abnormal_weight": aw_up, "--weight_decay": base_wd},
            "Reference is close but not all seeds hit; raise abnormal weight to test FN/FP boundary.",
        )

    runs: list[dict[str, Any]] = []
    for name, args, reason in proposals[:max_new_candidates]:
        runs.extend(make_candidate_runs(name, args, seeds, reason))

    state["round"] = round_no
    state["last_source_candidate"] = source
    state["last_source_metrics"] = {
        "fn_mean": fn_mean,
        "fp_mean": fp_mean,
        "complete": source_row.get("complete"),
        "target_hits": source_row.get("target_hits"),
    }
    state["last_generated_candidates"] = [
        {"candidate": name, "reason": reason, "args": args}
        for name, args, reason in proposals[:max_new_candidates]
    ]
    return runs


def append_runs(queue_path: Path, runs: list[dict[str, Any]]) -> None:
    queue = read_json(queue_path, {"runs": []})
    queue.setdefault("runs", [])
    existing_tags = {str(run.get("tag", "")) for run in queue_runs(queue)}
    added = []
    for run in runs:
        if run["tag"] not in existing_tags:
            added.append(run)
            existing_tags.add(run["tag"])
    queue["runs"].extend(added)
    write_json(queue_path, queue)


def run_controller(
    *,
    queue: Path,
    summary: Path,
    markdown: Path,
    target_min: int,
    target_max: int,
    stop_mode: str,
    candidate_min_runs_before_skip: int,
    max_launched: int,
) -> int:
    cmd = [
        sys.executable,
        str(CONTROLLER),
        "--queue",
        str(queue),
        "--summary",
        str(summary),
        "--markdown",
        str(markdown),
        "--target-min",
        str(target_min),
        "--target-max",
        str(target_max),
        "--stop-mode",
        stop_mode,
        "--candidate-min-runs-before-skip",
        str(candidate_min_runs_before_skip),
    ]
    if max_launched > 0:
        cmd.extend(["--max-launched", str(max_launched)])
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    print(f"[{now_iso()}] launching controller: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=ROOT, env=env, check=False)
    return int(result.returncode)


def active_controller_processes() -> list[dict[str, Any]]:
    if os.name != "nt":
        return []
    script = r"""
$selfPid = $PID
Get-CimInstance Win32_Process |
  Where-Object {
    $_.ProcessId -ne $selfPid -and
    $_.CommandLine -like '*adaptive_experiment_controller.py*' -and
    $_.CommandLine -notlike '*Get-CimInstance*'
  } |
  Select-Object ProcessId,CommandLine |
  ConvertTo-Json -Compress
"""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError:
        return []
    text = result.stdout.strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    return []


def wait_for_existing_controller(poll_seconds: int) -> None:
    while True:
        active = active_controller_processes()
        if not active:
            return
        pids = ", ".join(str(proc.get("ProcessId")) for proc in active)
        print(f"[{now_iso()}] waiting for active controller pid(s): {pids}", flush=True)
        time.sleep(poll_seconds)


def evidence_gate(summary: dict[str, Any], seeds: list[int]) -> str:
    hit = all_seed_hit_candidate(summary, seeds)
    if hit:
        return f"E3 paper-grade reference candidate: {hit}"
    rows = summary.get("aggregates", {}).get("by_candidate", {})
    if any((row.get("complete") or 0) >= 3 for row in rows.values()):
        return "E2 candidate reference: 3+ seeds exist but not all seeds meet target."
    if any((row.get("complete") or 0) >= 1 for row in rows.values()):
        return "E1 exploratory: only partial seed evidence."
    return "E0 unusable: no completed metrics."


def write_master_markdown(path: Path, state: dict[str, Any], summary: dict[str, Any], queue: dict[str, Any], seeds: list[int]) -> None:
    lines = [
        "# Adaptive Master Summary",
        "",
        f"- Updated: `{now_iso()}`",
        f"- Master decision: `{state.get('decision')}`",
        f"- Round: `{state.get('round', 0)}`",
        f"- Evidence gate: `{evidence_gate(summary, seeds)}`",
        f"- Source candidate: `{state.get('last_source_candidate')}`",
        "",
        "## Last Source Metrics",
        "",
    ]
    metrics = state.get("last_source_metrics", {})
    if metrics:
        for key, value in metrics.items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Generated Candidates", ""])
    generated = state.get("last_generated_candidates", [])
    if generated:
        lines.extend(["| Candidate | Reason | Key Args |", "| --- | --- | --- |"])
        for item in generated:
            args = item.get("args", {})
            key_args = {
                key: args.get(key)
                for key in ("--epochs", "--normal_ratio", "--abnormal_weight", "--grad_clip", "--weight_decay")
                if key in args
            }
            lines.append(f"| {item.get('candidate')} | {item.get('reason')} | `{json.dumps(key_args, ensure_ascii=False)}` |")
    else:
        lines.append("- none")
    lines.extend(["", "## Queue Size", "", f"- Runs in queue: `{len(queue_runs(queue))}`"])
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)


def has_completed_metrics(summary: dict[str, Any]) -> bool:
    return any(
        record.get("status") == "complete"
        for record in summary.get("runs", {}).values()
        if isinstance(record, dict)
    )


def queued_tags(queue: dict[str, Any]) -> set[str]:
    return {str(run.get("tag", "")) for run in queue_runs(queue) if run.get("tag")}


def missing_seed_expansions(
    queue: dict[str, Any],
    summary: dict[str, Any],
    *,
    target_min: int,
    target_max: int,
    seeds: list[int],
) -> list[dict[str, Any]]:
    tags = queued_tags(queue)
    rows = summary.get("aggregates", {}).get("by_candidate", {})
    for candidate, row in sorted(rows.items(), key=lambda item: (-int(item[1].get("target_hits") or 0), str(item[0]))):
        complete = int(row.get("complete") or 0)
        if complete < 2 or complete >= len(seeds):
            continue
        records = candidate_records(summary, str(candidate))
        if not records:
            continue
        values = [(record.get("fn"), record.get("fp")) for record in records]
        if any(fn is None or fp is None for fn, fp in values):
            continue
        severe = any(fn < target_min - 3 or fn > target_max + 5 or fp < target_min - 3 or fp > target_max + 8 for fn, fp in values)
        near = all(target_min - 1 <= fn <= target_max + 2 and target_min - 1 <= fp <= target_max + 3 for fn, fp in values)
        hit_count = int(row.get("target_hits") or 0)
        if severe or (hit_count == 0 and not near):
            continue
        done = {int(record["seed"]) for record in records}
        args = candidate_args_from_queue(queue, str(candidate))
        reason = (
            f"Pilot for {candidate} is close enough to expand: "
            f"hits={hit_count}/{complete}, fn_mean={row.get('fn_mean')}, fp_mean={row.get('fp_mean')}."
        )
        expansions = []
        for seed in seeds:
            tag = f"{candidate}_s{seed}"
            if seed not in done and tag not in tags:
                expansions.append({
                    "tag": tag,
                    "candidate": str(candidate),
                    "seed": int(seed),
                    "args": args,
                    "reason": reason,
                })
        if expansions:
            state_note = summary.setdefault("master_notes", {})
            state_note["last_expansion_reason"] = reason
            return expansions
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Adaptive master loop for anomaly reference search")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--state", type=Path, default=DEFAULT_MASTER_STATE)
    parser.add_argument("--master-markdown", type=Path, default=DEFAULT_MASTER_MARKDOWN)
    parser.add_argument("--target-min", type=int, default=5)
    parser.add_argument("--target-max", type=int, default=15)
    parser.add_argument("--seeds", type=str, default="42,1,2,3,4")
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--max-new-candidates", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--stop-mode", choices=["first-hit", "all-seeds", "never"], default="all-seeds")
    parser.add_argument("--candidate-min-runs-before-skip", type=int, default=2)
    parser.add_argument("--controller-max-launched", type=int, default=1)
    parser.add_argument("--once", action="store_true", help="Plan one batch and exit after launching/running controller once")
    args = parser.parse_args()

    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    state = read_json(args.state, {"round": 0, "created_at": now_iso()})
    state["updated_at"] = now_iso()
    state["target"] = {"min": args.target_min, "max": args.target_max, "seeds": seeds}

    for _ in range(args.max_rounds):
        wait_for_existing_controller(args.poll_seconds)
        summary = read_json(args.summary, {})
        queue = read_json(args.queue, {"runs": []})

        hit = all_seed_hit_candidate(summary, seeds)
        if hit:
            state["decision"] = f"stop:target_reference_found:{hit}"
            write_json(args.state, state)
            write_master_markdown(args.master_markdown, state, summary, queue, seeds)
            print(state["decision"], flush=True)
            return 0

        decision = summary.get("decision")
        if decision not in (None, "", "queue_exhausted", "continue", "not_started") and not str(decision).startswith("stop:run_failed"):
            state["decision"] = f"stop:controller_decision:{decision}"
            write_json(args.state, state)
            write_master_markdown(args.master_markdown, state, summary, queue, seeds)
            print(state["decision"], flush=True)
            return 0

        if decision == "continue" or (not has_completed_metrics(summary) and queue_runs(queue)):
            if not has_completed_metrics(summary):
                state["decision"] = "run_controller:no_completed_metrics"
                write_json(args.state, state)
                write_master_markdown(args.master_markdown, state, summary, queue, seeds)
            code = run_controller(
                queue=args.queue,
                summary=args.summary,
                markdown=args.markdown,
                target_min=args.target_min,
                target_max=args.target_max,
                stop_mode=args.stop_mode,
                candidate_min_runs_before_skip=args.candidate_min_runs_before_skip,
                max_launched=args.controller_max_launched,
            )
            if code != 0:
                state["decision"] = f"stop:controller_exit:{code}"
                write_json(args.state, state)
                return code
            if args.once:
                break
            continue

        new_runs = missing_seed_expansions(
            queue,
            summary,
            target_min=args.target_min,
            target_max=args.target_max,
            seeds=seeds,
        )
        if not new_runs:
            new_runs = propose_candidates(
                queue,
                summary,
                state,
                target_min=args.target_min,
                target_max=args.target_max,
                seeds=seeds,
                max_new_candidates=args.max_new_candidates,
            )
        else:
            state["round"] = int(state.get("round", 0)) + 1
            state["last_source_candidate"] = str(new_runs[0].get("candidate"))
            state["last_source_metrics"] = summary.get("aggregates", {}).get("by_candidate", {}).get(str(new_runs[0].get("candidate")), {})
            state["last_generated_candidates"] = [{
                "candidate": str(new_runs[0].get("candidate")),
                "reason": str(new_runs[0].get("reason")),
                "args": new_runs[0].get("args", {}),
            }]

        if not new_runs:
            state["decision"] = "stop:no_new_candidates"
            write_json(args.state, state)
            write_master_markdown(args.master_markdown, state, summary, queue, seeds)
            return 2

        append_runs(args.queue, new_runs)
        queue = read_json(args.queue, {"runs": []})
        state["decision"] = f"appended:{len(new_runs)}_runs"
        write_json(args.state, state)
        write_master_markdown(args.master_markdown, state, summary, queue, seeds)

        code = run_controller(
            queue=args.queue,
            summary=args.summary,
            markdown=args.markdown,
            target_min=args.target_min,
            target_max=args.target_max,
            stop_mode=args.stop_mode,
            candidate_min_runs_before_skip=args.candidate_min_runs_before_skip,
            max_launched=args.controller_max_launched,
        )
        if code != 0:
            state["decision"] = f"stop:controller_exit:{code}"
            write_json(args.state, state)
            return code
        if args.once:
            break

    summary = read_json(args.summary, {})
    queue = read_json(args.queue, {"runs": []})
    state["decision"] = "stop:max_rounds_reached"
    state["updated_at"] = now_iso()
    write_json(args.state, state)
    write_master_markdown(args.master_markdown, state, summary, queue, seeds)
    print(state["decision"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
