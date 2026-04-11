import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
SUMMARY_PATH = ROOT / "experiment_summary" / "v11d_matrix_summary.json"
TRAIN_CONFIG = ROOT / "configs" / "train" / "winning.yaml"
DATA_CONFIG = ROOT / "configs" / "datasets" / "v11.yaml"

SEEDS = [42, 1, 2, 3, 4]
REF_NORMAL_RATIO = 700
NORMAL_SWEEP = [350, 1400, 2800]


def build_runs():
    runs = []
    for normal_ratio in NORMAL_SWEEP:
        for seed in SEEDS:
            runs.append({
                "group": "normal_ratio",
                "tag": f"v11d_n{normal_ratio}_s{seed}",
                "seed": seed,
                "normal_ratio": normal_ratio,
            })
    for seed in SEEDS:
        runs.append({
            "group": "ref",
            "tag": f"v11d_ref_n{REF_NORMAL_RATIO}_s{seed}",
            "seed": seed,
            "normal_ratio": REF_NORMAL_RATIO,
        })
    return runs


def load_summary():
    if SUMMARY_PATH.exists():
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    return {"runs": {}, "aggregates": {}}


def save_summary(summary):
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def find_completed_run_dir(tag: str):
    matches = []
    for path in LOGS_DIR.glob(f"*_{tag}*"):
        if path.is_dir() and (path / "best_info.json").exists():
            matches.append(path)
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def parse_run_metrics(run_dir: Path):
    info = json.loads((run_dir / "best_info.json").read_text(encoding="utf-8"))
    last_event = info.get("test_history", [{}])[-1]
    abnormal = info.get("test_metrics", {}).get("abnormal", {})
    normal = info.get("test_metrics", {}).get("normal", {})
    return {
        "run_dir": str(run_dir),
        "epoch": info.get("epoch"),
        "test_f1": info.get("test_f1"),
        "test_recall": info.get("test_recall"),
        "test_abn_R": abnormal.get("recall"),
        "test_nor_R": normal.get("recall"),
        "fn": last_event.get("fn"),
        "fp": last_event.get("fp"),
        "tn": last_event.get("tn"),
        "tp": last_event.get("tp"),
        "total_time_min": info.get("timing", {}).get("total_time_min"),
    }


def update_aggregates(summary):
    runs = summary["runs"]
    aggregates = {}

    def add_aggregate(name, selected):
        if not selected:
            return
        f1s = [r["test_f1"] for r in selected if r.get("test_f1") is not None]
        fns = [r["fn"] for r in selected if r.get("fn") is not None]
        fps = [r["fp"] for r in selected if r.get("fp") is not None]
        if not f1s:
            return
        aggregates[name] = {
            "count": len(f1s),
            "f1_mean": round(mean(f1s), 6),
            "f1_std": round(pstdev(f1s), 6) if len(f1s) > 1 else 0.0,
            "fn_mean": round(mean(fns), 3) if fns else None,
            "fp_mean": round(mean(fps), 3) if fps else None,
        }

    ref_runs = [runs[k] for k in runs if runs[k]["group"] == "ref"]
    add_aggregate("ref_n700", ref_runs)
    for normal_ratio in NORMAL_SWEEP:
        selected = [
            runs[k] for k in runs
            if runs[k]["group"] == "normal_ratio" and runs[k]["normal_ratio"] == normal_ratio
        ]
        add_aggregate(f"normal_ratio_{normal_ratio}", selected)

    summary["aggregates"] = aggregates


def run_experiment(run):
    cmd = [
        sys.executable,
        "train.py",
        "--train_config",
        str(TRAIN_CONFIG),
        "--config",
        str(DATA_CONFIG),
        "--seed",
        str(run["seed"]),
        "--normal_ratio",
        str(run["normal_ratio"]),
        "--log_dir",
        run["tag"],
    ]
    print(f"\n=== RUN {run['tag']} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    summary = load_summary()

    for run in build_runs():
        existing = find_completed_run_dir(run["tag"])
        if existing is None:
            run_experiment(run)
            existing = find_completed_run_dir(run["tag"])
            if existing is None:
                raise RuntimeError(f"Completed run directory not found for tag: {run['tag']}")

        metrics = parse_run_metrics(existing)
        summary["runs"][run["tag"]] = {
            **run,
            **metrics,
        }
        update_aggregates(summary)
        save_summary(summary)
        print(json.dumps(summary["aggregates"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
