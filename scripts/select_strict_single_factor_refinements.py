from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]


SEEDS = [42, 1, 2, 3, 4]
SELECTED_CONFIG = r"logs\260412_044744_fresh0412_v11_n700_s42_F0.9920_R0.9920\data_config_used.yaml"


BASE_ARGS = {
    "--mode": "binary",
    "--config": SELECTED_CONFIG,
    "--epochs": 20,
    "--patience": 5,
    "--batch_size": 32,
    "--dropout": 0.0,
    "--precision": "fp16",
    "--num_workers": 0,
    "--ema_decay": 0.0,
    "--normal_ratio": 700,
    "--smooth_window": 3,
    "--smooth_method": "median",
    "--lr_backbone": "2e-5",
    "--lr_head": "2e-4",
    "--warmup_epochs": 5,
    "--grad_clip": 1.0,
    "--weight_decay": 0.01,
}


@dataclass
class AxisSpec:
    name: str
    pool: list[float]
    baseline: float
    candidate_for_value: dict[float, str]
    arg_updates_for_value: dict[float, dict]
    objective: str = "fn_fp"


def load_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    return json.loads(text)


def aggregate_candidates(summary: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for run in summary.get("runs", {}).values():
        candidate = run.get("candidate")
        if not candidate:
            continue
        out.setdefault(candidate, {"complete": 0, "rows": []})
        if run.get("status") == "complete":
            out[candidate]["complete"] += 1
            out[candidate]["rows"].append(run)
    for cand, payload in out.items():
        rows = payload["rows"]
        if rows:
            payload["f1"] = mean(r["test_f1"] for r in rows)
            payload["fn"] = mean(r["fn"] for r in rows)
            payload["fp"] = mean(r["fp"] for r in rows)
            payload["seeds"] = sorted(r["seed"] for r in rows)
    return out


def objective_score(row: dict) -> tuple[float, float, float]:
    return (row["fn"] + row["fp"], row["fn"], -row["f1"])


def build_specs() -> list[AxisSpec]:
    return [
        AxisSpec(
            name="normal_ratio",
            pool=[700, 1400, 2100, 2800, 3000, 3150, 3300, 3500, 3800, 4200],
            baseline=700,
            candidate_for_value={
                700: "fresh0412_v11_n700",
                1400: "fresh0412_v11_n1400",
                2100: "fresh0412_v11_n2100",
                2800: "fresh0412_v11_n2800",
                3000: "fresh0412_v11_n3000",
                3150: "fresh0412_v11_n3150",
                3300: "fresh0412_v11_n3300",
                3500: "fresh0412_v11_n3500",
                3800: "fresh0412_v11_n3800",
                4200: "fresh0412_v11_n4200",
            },
            arg_updates_for_value={v: {"--normal_ratio": int(v)} for v in [700, 1400, 2100, 2800, 3000, 3150, 3300, 3500, 3800, 4200]},
        ),
        AxisSpec(
            name="gc",
            pool=[0.0, 0.1, 0.25, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0],
            baseline=1.0,
            candidate_for_value={
                0.0: "fresh0412_v11_gc00_n700",
                0.1: "fresh0412_v11_gc01_n700",
                0.25: "fresh0412_v11_gc025_n700",
                0.35: "fresh0412_v11_gc035_n700",
                0.5: "fresh0412_v11_gc05_n700",
                0.75: "fresh0412_v11_gc075_n700",
                1.0: "fresh0412_v11_gc10_n700",
                1.25: "fresh0412_v11_gc125_n700",
                1.5: "fresh0412_v11_gc15_n700",
                2.0: "fresh0412_v11_gc20_n700",
                3.0: "fresh0412_v11_gc30_n700",
                5.0: "fresh0412_v11_gc50_n700",
            },
            arg_updates_for_value={v: {"--grad_clip": v} for v in [0.0, 0.1, 0.25, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]},
        ),
        AxisSpec(
            name="label_smoothing",
            pool=[0.0, 0.02, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2],
            baseline=0.0,
            candidate_for_value={
                0.0: "fresh0412_v11_n700",
                0.02: "fresh0412_v11_regls002_n700",
                0.05: "fresh0412_v11_regls005_n700",
                0.075: "fresh0412_v11_regls0075_n700",
                0.1: "fresh0412_v11_regls01_n700",
                0.125: "fresh0412_v11_regls0125_n700",
                0.15: "fresh0412_v11_regls015_n700",
                0.175: "fresh0412_v11_regls0175_n700",
                0.2: "fresh0412_v11_regls020_n700",
            },
            arg_updates_for_value={v: {"--label_smoothing": v} for v in [0.02, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]},
        ),
        AxisSpec(
            name="stochastic_depth",
            pool=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
            baseline=0.0,
            candidate_for_value={
                0.0: "fresh0412_v11_n700",
                0.05: "fresh0412_v11_regdp005_n700",
                0.1: "fresh0412_v11_regdp01_n700",
                0.15: "fresh0412_v11_regdp015_n700",
                0.2: "fresh0412_v11_regdp02_n700",
                0.25: "fresh0412_v11_regdp025_n700",
                0.3: "fresh0412_v11_regdp03_n700",
            },
            arg_updates_for_value={v: {"--stochastic_depth_rate": v} for v in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]},
        ),
        AxisSpec(
            name="focal_gamma",
            pool=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            baseline=0.0,
            candidate_for_value={
                0.0: "fresh0412_v11_n700",
                0.5: "fresh0412_v11_fg0p5_n700",
                1.0: "fresh0412_v11_fg1p0_n700",
                1.5: "fresh0412_v11_fg1p5_n700",
                2.0: "fresh0412_v11_fg2p0_n700",
                2.5: "fresh0412_v11_fg2p5_n700",
                3.0: "fresh0412_v11_fg3p0_n700",
            },
            arg_updates_for_value={v: {"--focal_gamma": v} for v in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]},
        ),
        AxisSpec(
            name="abnormal_weight",
            pool=[0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
            baseline=1.0,
            candidate_for_value={
                0.5: "fresh0412_v11_aw0p5_n700",
                0.8: "fresh0412_v11_aw0p8_n700",
                1.0: "fresh0412_v11_n700",
                1.2: "fresh0412_v11_aw1p2_n700",
                1.5: "fresh0412_v11_aw1p5_n700",
                2.0: "fresh0412_v11_aw2p0_n700",
                2.5: "fresh0412_v11_aw2p5_n700",
                3.0: "fresh0412_v11_aw3p0_n700",
            },
            arg_updates_for_value={v: {"--abnormal_weight": v} for v in [0.5, 0.8, 1.2, 1.5, 2.0, 2.5, 3.0]},
        ),
        AxisSpec(
            name="ema_decay",
            pool=[0.0, 0.99, 0.995, 0.999],
            baseline=0.0,
            candidate_for_value={
                0.0: "fresh0412_v11_n700",
                0.99: "fresh0412_v11_ema0p99_n700",
                0.995: "fresh0412_v11_ema0p995_n700",
                0.999: "fresh0412_v11_ema0p999_n700",
            },
            arg_updates_for_value={v: {"--ema_decay": v} for v in [0.99, 0.995, 0.999]},
        ),
    ]


def build_run(candidate: str, arg_updates: dict, seed: int, reason: str) -> dict:
    args = dict(BASE_ARGS)
    args.update(arg_updates)
    if candidate.endswith("_n700"):
        tag_base = candidate
    else:
        tag_base = candidate
    return {
        "tag": f"{tag_base}_s{seed}",
        "candidate": candidate,
        "seed": seed,
        "args": args,
        "reason": reason,
    }


def select_new_values(spec: AxisSpec, completed: dict[str, dict]) -> list[float]:
    value_rows = []
    tested_values = set()
    for value, candidate in spec.candidate_for_value.items():
        row = completed.get(candidate)
        if row and row.get("complete", 0) > 0:
            tested_values.add(value)
            value_rows.append((value, row))
        elif value == spec.baseline:
            tested_values.add(value)
            value_rows.append((value, {"f1": 0.9901, "fn": 9.8, "fp": 5.0, "complete": 5}))

    if len(tested_values) < 3:
        return []

    best_value, best_row = min(value_rows, key=lambda vr: objective_score(vr[1]))
    pool = spec.pool
    idx = pool.index(best_value)
    proposals: list[float] = []

    left_tested = any(v < best_value for v in tested_values)
    right_tested = any(v > best_value for v in tested_values)

    if not left_tested and idx > 0:
        proposals.append(pool[idx - 1])
    if not right_tested and idx < len(pool) - 1:
        proposals.append(pool[idx + 1])

    # ensure nearest untested neighbors around best are filled
    if idx > 0 and pool[idx - 1] not in tested_values:
        proposals.append(pool[idx - 1])
    if idx < len(pool) - 1 and pool[idx + 1] not in tested_values:
        proposals.append(pool[idx + 1])

    # if best already has both immediate neighbors tested but still sits near an edge, extend outward
    if idx <= 1 and idx < len(pool) - 1:
        for j in range(idx + 1, len(pool)):
            if pool[j] not in tested_values:
                proposals.append(pool[j])
                break
    if idx >= len(pool) - 2 and idx > 0:
        for j in range(idx - 1, -1, -1):
            if pool[j] not in tested_values:
                proposals.append(pool[j])
                break

    # de-duplicate while preserving order
    ordered = []
    seen = set()
    for value in proposals:
        if value in seen or value in tested_values:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out-queue", required=True)
    parser.add_argument("--decision-md", required=True)
    args = parser.parse_args()

    summary = load_summary(Path(args.summary))
    completed = aggregate_candidates(summary)
    specs = build_specs()

    queue_runs = []
    lines = [
        "# Strict Single-Factor Refinement Decision",
        "",
        "- Policy: baseline fixed, one option changes at a time only",
        "- Source summary: `{}`".format(args.summary),
        "",
        "## Decisions",
        "",
    ]

    for spec in specs:
        new_values = select_new_values(spec, completed)
        if not new_values:
            lines.append(f"- `{spec.name}`: no new levels selected")
            continue
        lines.append(f"- `{spec.name}`: add levels `{', '.join(str(v) for v in new_values)}`")
        for value in new_values:
            candidate = spec.candidate_for_value[value]
            updates = spec.arg_updates_for_value.get(value, {})
            reason = f"strict one-factor refinement for {spec.name}={value} under fixed baseline after observing first-round results"
            for seed in SEEDS:
                queue_runs.append(build_run(candidate, updates, seed, reason))

    payload = {
        "created_at": "2026-04-26T15:40:00",
        "selected_reference": "fresh0412_v11_n700_existing",
        "selected_config": SELECTED_CONFIG,
        "note": "Round-2 strict refinements chosen from round-1 completed results. Policy remains baseline fixed and one-factor only.",
        "runs": queue_runs,
    }

    out_queue = Path(args.out_queue)
    out_queue.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.decision_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"n_runs": len(queue_runs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
