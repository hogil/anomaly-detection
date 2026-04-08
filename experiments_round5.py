"""
5차 실험 - 수정 데이터 + anomaly_type 집중 + 다양한 조합
"""
import subprocess, json, time
from pathlib import Path

EXPERIMENTS = [
    # anomaly_type baseline (수정 데이터)
    {"name": "r5_at_steplr15",
     "args": "--epochs 50 --batch_size 32 --lr_backbone 1e-4 --lr_head 1e-3 --warmup_epochs 5 --scheduler step --step_size 15 --step_gamma 0.5 --use_amp --mode anomaly_type"},
    # anomaly_type + Mixup
    {"name": "r5_at_mixup",
     "args": "--epochs 50 --batch_size 32 --lr_backbone 1e-4 --lr_head 1e-3 --warmup_epochs 5 --scheduler step --step_size 15 --step_gamma 0.5 --use_amp --mode anomaly_type --use_mixup"},
    # anomaly_type + LabelSmoothing
    {"name": "r5_at_ls",
     "args": "--epochs 50 --batch_size 32 --lr_backbone 1e-4 --lr_head 1e-3 --warmup_epochs 5 --scheduler step --step_size 15 --step_gamma 0.5 --use_amp --mode anomaly_type --label_smoothing 0.1"},
    # anomaly_type + lower LR
    {"name": "r5_at_lowlr",
     "args": "--epochs 50 --batch_size 32 --lr_backbone 5e-5 --lr_head 5e-4 --warmup_epochs 5 --scheduler step --step_size 15 --step_gamma 0.5 --use_amp --mode anomaly_type"},
    # anomaly_type + StepLR 10/0.5
    {"name": "r5_at_step10",
     "args": "--epochs 50 --batch_size 32 --lr_backbone 1e-4 --lr_head 1e-3 --warmup_epochs 5 --scheduler step --step_size 10 --step_gamma 0.5 --use_amp --mode anomaly_type"},
    # anomaly_type + Cosine
    {"name": "r5_at_cosine",
     "args": "--epochs 50 --batch_size 32 --lr_backbone 5e-5 --lr_head 5e-4 --warmup_epochs 5 --scheduler cosine --use_amp --mode anomaly_type"},
    # anomaly_type + dropout 0.3 (less dropout)
    {"name": "r5_at_drop03",
     "args": "--epochs 50 --batch_size 32 --lr_backbone 1e-4 --lr_head 1e-3 --warmup_epochs 5 --scheduler step --step_size 15 --step_gamma 0.5 --use_amp --mode anomaly_type --dropout 0.3"},
    # anomaly_type + StepLR 7/0.7
    {"name": "r5_at_step7",
     "args": "--epochs 50 --batch_size 32 --lr_backbone 1e-4 --lr_head 1e-3 --warmup_epochs 5 --scheduler step --step_size 7 --step_gamma 0.7 --use_amp --mode anomaly_type"},
    # binary (수정 데이터)
    {"name": "r5_binary",
     "args": "--epochs 50 --batch_size 32 --lr_backbone 1e-4 --lr_head 1e-3 --warmup_epochs 5 --scheduler step --step_size 15 --step_gamma 0.5 --use_amp --mode binary"},
    # multiclass (수정 데이터, 비교용)
    {"name": "r5_multiclass",
     "args": "--epochs 50 --batch_size 32 --lr_backbone 1e-4 --lr_head 1e-3 --warmup_epochs 5 --scheduler step --step_size 15 --step_gamma 0.5 --use_amp --mode multiclass"},
]

def run_experiment(exp):
    name = exp["name"]
    log_dir = Path("logs") / name
    if (log_dir / "best_info.json").exists():
        info = json.load(open(log_dir / "best_info.json"))
        print(f"  [SKIP] {name}: test_R={info['test_recall']:.3f}")
        return {"name": name, "val_recall": info["val_recall"], "val_f1": info["val_f1"],
                "test_recall": info["test_recall"], "test_f1": info["test_f1"], "best_epoch": info["epoch"]}
    print(f"\n  [RUN] {name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = f"python train.py {exp['args']} --log_dir {log_dir}"
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0
    if (log_dir / "best_info.json").exists():
        info = json.load(open(log_dir / "best_info.json"))
        r = {"name": name, "val_recall": info["val_recall"], "val_f1": info["val_f1"],
             "test_recall": info["test_recall"], "test_f1": info["test_f1"],
             "best_epoch": info["epoch"], "elapsed": round(elapsed, 1)}
        print(f"  [OK] {name}: test_R={r['test_recall']:.3f} F1={r['test_f1']:.3f} ({elapsed:.0f}s)")
        return r
    else:
        err = result.stderr[-500:] if result.stderr else "?"
        print(f"  [FAIL] {name}: {err}")
        return {"name": name, "error": True}

def main():
    results = []
    for exp in EXPERIMENTS:
        try:
            results.append(run_experiment(exp))
        except Exception as e:
            print(f"  [ERR] {exp['name']}: {e}")
            results.append({"name": exp["name"], "error": True})

    print(f"\n{'='*80}")
    print(f"  Round 5 Results")
    print(f"{'='*80}")
    print(f"  {'Name':>20s} | {'Val_R':>6s} | {'Test_R':>6s} | {'Test_F1':>7s} | {'Ep':>3s}")
    print(f"  {'-'*20}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*3}")
    for r in sorted(results, key=lambda x: x.get("test_recall", 0), reverse=True):
        if "error" in r:
            print(f"  {r['name']:>20s} | FAILED")
        else:
            ep = r.get("best_epoch", r.get("epoch", 0))
            print(f"  {r['name']:>20s} | {r['val_recall']:6.3f} | {r['test_recall']:6.3f} | {r['test_f1']:7.3f} | {ep:3d}")
    with open("logs/round5_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
