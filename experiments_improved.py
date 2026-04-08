"""
개선 실험 대규모 - abnormal recall 극대화

축:
1. abnormal_weight: 1.5, 2, 3, 5, 8, 10
2. focal_gamma: 1.0, 1.5, 2.0, 2.5, 3.0
3. min_epochs: 8, 15
4. scheduler: step(15/0.5, 10/0.5, 7/0.7), cosine
5. LR: (1e-4/1e-3), (5e-5/5e-4), (2e-4/2e-3)
6. dropout: 0.3, 0.5, 0.7
7. mixup: off, 0.2, 0.4
8. NT는 모든 실험에서 0.5/0.6/0.7/0.8/0.9 자동 평가
"""
import subprocess, json, time
from pathlib import Path

BASE = "--epochs 50 --batch_size 32 --warmup_epochs 5 --use_amp --mode binary"

EXPERIMENTS = [
    # === 1. Abnormal Weight 세분화 ===
    {"name": "imp_aw1.5",  "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --scheduler step --step_size 15 --step_gamma 0.5 --min_epochs 8 --abnormal_weight 1.5"},
    {"name": "imp_aw2",    "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --scheduler step --step_size 15 --step_gamma 0.5 --min_epochs 8 --abnormal_weight 2.0"},
    {"name": "imp_aw3",    "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --scheduler step --step_size 15 --step_gamma 0.5 --min_epochs 8 --abnormal_weight 3.0"},
    {"name": "imp_aw5",    "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --scheduler step --step_size 15 --step_gamma 0.5 --min_epochs 8 --abnormal_weight 5.0"},
    {"name": "imp_aw8",    "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --scheduler step --step_size 15 --step_gamma 0.5 --min_epochs 8 --abnormal_weight 8.0"},
    {"name": "imp_aw10",   "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --scheduler step --step_size 15 --step_gamma 0.5 --min_epochs 8 --abnormal_weight 10.0"},

    # === 2. Focal Gamma 세분화 (aw=3 고정) ===
    {"name": "imp_g1.0",   "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --scheduler step --step_size 15 --step_gamma 0.5 --min_epochs 8 --abnormal_weight 3.0 --focal_gamma 1.0"},
    {"name": "imp_g1.5",   "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --scheduler step --step_size 15 --step_gamma 0.5 --min_epochs 8 --abnormal_weight 3.0 --focal_gamma 1.5"},
    {"name": "imp_g2.5",   "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --scheduler step --step_size 15 --step_gamma 0.5 --min_epochs 8 --abnormal_weight 3.0 --focal_gamma 2.5"},
    {"name": "imp_g3.0",   "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --scheduler step --step_size 15 --step_gamma 0.5 --min_epochs 8 --abnormal_weight 3.0 --focal_gamma 3.0"},

    # === 3. Scheduler 세분화 (aw=3 고정) ===
    {"name": "imp_s10_05", "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --scheduler step --step_size 10 --step_gamma 0.5 --min_epochs 8 --abnormal_weight 3.0"},
    {"name": "imp_s7_07",  "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --scheduler step --step_size 7 --step_gamma 0.7 --min_epochs 8 --abnormal_weight 3.0"},
    {"name": "imp_cosine", "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --scheduler cosine --min_epochs 8 --abnormal_weight 3.0"},

    # === 4. LR 세분화 (aw=3 고정) ===
    {"name": "imp_lr_low", "args": f"{BASE} --lr_backbone 5e-5 --lr_head 5e-4 --scheduler step --step_size 15 --step_gamma 0.5 --min_epochs 8 --abnormal_weight 3.0"},
    {"name": "imp_lr_hi",  "args": f"{BASE} --lr_backbone 2e-4 --lr_head 2e-3 --scheduler step --step_size 15 --step_gamma 0.5 --min_epochs 8 --abnormal_weight 3.0"},

    # === 5. Dropout 세분화 (aw=3 고정) ===
    {"name": "imp_d03",    "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --scheduler step --step_size 15 --step_gamma 0.5 --min_epochs 8 --abnormal_weight 3.0 --dropout 0.3"},
    {"name": "imp_d07",    "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --scheduler step --step_size 15 --step_gamma 0.5 --min_epochs 8 --abnormal_weight 3.0 --dropout 0.7"},

    # === 6. Mixup (aw=3 고정) ===
    {"name": "imp_mix02",  "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --scheduler step --step_size 15 --step_gamma 0.5 --min_epochs 8 --abnormal_weight 3.0 --use_mixup --mixup_alpha 0.2"},
    {"name": "imp_mix04",  "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --scheduler step --step_size 15 --step_gamma 0.5 --min_epochs 8 --abnormal_weight 3.0 --use_mixup --mixup_alpha 0.4"},

    # === 7. min_epochs 변동 (aw=3 고정) ===
    {"name": "imp_min15",  "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --scheduler step --step_size 15 --step_gamma 0.5 --min_epochs 15 --abnormal_weight 3.0"},

    # === 8. 최적 조합 후보 ===
    {"name": "imp_best1",  "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --scheduler step --step_size 15 --step_gamma 0.5 --min_epochs 8 --abnormal_weight 5.0 --focal_gamma 2.5"},
    {"name": "imp_best2",  "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --scheduler step --step_size 10 --step_gamma 0.5 --min_epochs 8 --abnormal_weight 3.0 --dropout 0.3"},
    {"name": "imp_best3",  "args": f"{BASE} --lr_backbone 5e-5 --lr_head 5e-4 --scheduler cosine --min_epochs 15 --abnormal_weight 5.0"},
    {"name": "imp_best4",  "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --scheduler step --step_size 15 --step_gamma 0.5 --min_epochs 8 --abnormal_weight 3.0 --focal_gamma 1.5 --dropout 0.3"},

    # === 9. 기준선 (개선 없이 min_epochs만) ===
    {"name": "imp_baseline", "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --scheduler step --step_size 15 --step_gamma 0.5 --min_epochs 8"},
]


def run():
    results = []
    total = len(EXPERIMENTS)

    for i, exp in enumerate(EXPERIMENTS):
        name = exp["name"]
        log_dir = Path("logs") / name

        if (log_dir / "best_info.json").exists():
            print(f"[SKIP] {name}")
            info = json.load(open(log_dir / "best_info.json"))
            bm = info["test_metrics"]
            results.append({"name": name, "abn_r": bm["abnormal"]["recall"],
                          "nor_r": bm["normal"]["recall"], "ep": info["epoch"]})
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] {name}")
        print(f"  {exp['args']}")
        print(f"{'='*60}")

        cmd = f'python train.py {exp["args"]} --log_dir logs/{name}'
        t0 = time.time()
        subprocess.run(cmd, shell=True, cwd="D:/project/anomaly-detection")
        elapsed = time.time() - t0

        if (log_dir / "best_info.json").exists():
            info = json.load(open(log_dir / "best_info.json"))
            bm = info["test_metrics"]
            r = {"name": name, "abn_r": bm["abnormal"]["recall"],
                 "nor_r": bm["normal"]["recall"], "ep": info["epoch"],
                 "elapsed": round(elapsed, 1)}
            results.append(r)
            print(f"  -> abn_R={r['abn_r']:.4f}, nor_R={r['nor_r']:.4f}, ep={r['ep']} ({elapsed:.0f}s)")
        else:
            print(f"  -> FAILED")
            results.append({"name": name, "abn_r": 0, "nor_r": 0, "ep": 0})

    # NT 최적값 분석 (마지막 5개만)
    print(f"\n{'='*60}")
    print(f"  NT 최적값 분석 (상위 5개)")
    print(f"{'='*60}")
    top5 = sorted(results, key=lambda x: x.get("abn_r", 0), reverse=True)[:5]
    for r in top5:
        p = f'logs/{r["name"]}/best_info.json'
        if os.path.exists(p):
            info = json.load(open(p))
            nt = info.get("normal_threshold_results", {})
            if nt:
                print(f"\n  {r['name']} (abn_R={r['abn_r']:.4f}):")
                for t, v in sorted(nt.items()):
                    print(f"    NT={t}: recall={v['recall']:.4f} f1={v['f1']:.4f}")

    # 최종 결과
    print(f"\n{'='*60}")
    print(f"  개선 실험 결과 ({total}개, abnormal recall 순)")
    print(f"{'='*60}")
    print(f"  {'Name':>20s} | {'Abn_R':>6s} | {'Nor_R':>6s} | {'Ep':>3s}")
    print(f"  {'-'*20}-+-{'-'*6}-+-{'-'*6}-+-{'-'*3}")
    for r in sorted(results, key=lambda x: x.get("abn_r", 0), reverse=True):
        print(f"  {r['name']:>20s} | {r.get('abn_r',0):.4f} | {r.get('nor_r',0):.4f} | {r.get('ep',0):>3}")

    with open("logs/improved_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run()
