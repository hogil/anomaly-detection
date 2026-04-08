"""
v2 데이터 실험 - Phase 1 + Phase 2

여태 학습 결과 기반:
- Cosine scheduler 압도적
- Binary 필수
- Gamma 1.5~4.0 sweet spot
- AW 1.0~1.5
- min_epochs 8~15
- seed 고정 (재현성)

목표: test abn_R >= 99% + F1 >= 90%
"""
import subprocess, json, time
from pathlib import Path

BASE = "--epochs 50 --batch_size 32 --warmup_epochs 5 --use_amp --mode binary --scheduler cosine --seed 42"

EXPERIMENTS = [
    # === Phase 1: Baseline 확인 (cosine + lr 1e-4 + 다양 변수) ===
    {"name": "v2_base",       "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --min_epochs 10"},
    {"name": "v2_min15",      "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --min_epochs 15"},
    {"name": "v2_min20",      "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --min_epochs 20"},

    # === Phase 2-A: Gamma 변동 ===
    {"name": "v2_g15",        "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --min_epochs 10 --focal_gamma 1.5"},
    {"name": "v2_g20",        "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --min_epochs 10 --focal_gamma 2.0"},
    {"name": "v2_g25",        "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --min_epochs 10 --focal_gamma 2.5"},
    {"name": "v2_g30",        "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --min_epochs 10 --focal_gamma 3.0"},
    {"name": "v2_g40",        "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --min_epochs 10 --focal_gamma 4.0"},

    # === Phase 2-B: AW 미세 ===
    {"name": "v2_aw11",       "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --min_epochs 10 --abnormal_weight 1.1"},
    {"name": "v2_aw12",       "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --min_epochs 10 --abnormal_weight 1.2"},
    {"name": "v2_aw13",       "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --min_epochs 10 --abnormal_weight 1.3"},
    {"name": "v2_aw15",       "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --min_epochs 10 --abnormal_weight 1.5"},

    # === Phase 2-C: LR 미세 ===
    {"name": "v2_lr5e5",      "args": f"{BASE} --lr_backbone 5e-5 --lr_head 5e-4 --min_epochs 10"},
    {"name": "v2_lr8e5",      "args": f"{BASE} --lr_backbone 8e-5 --lr_head 8e-4 --min_epochs 10"},
    {"name": "v2_lr2e4",      "args": f"{BASE} --lr_backbone 2e-4 --lr_head 2e-3 --min_epochs 10"},

    # === Phase 2-D: 최적 조합 후보 ===
    {"name": "v2_g25_aw12",   "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --min_epochs 10 --focal_gamma 2.5 --abnormal_weight 1.2"},
    {"name": "v2_g30_aw12",   "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --min_epochs 10 --focal_gamma 3.0 --abnormal_weight 1.2"},
    {"name": "v2_g40_aw12",   "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --min_epochs 10 --focal_gamma 4.0 --abnormal_weight 1.2"},
    {"name": "v2_g25_lr8",    "args": f"{BASE} --lr_backbone 8e-5 --lr_head 8e-4 --min_epochs 15 --focal_gamma 2.5 --abnormal_weight 1.2"},
    {"name": "v2_long",       "args": f"--epochs 70 --batch_size 32 --warmup_epochs 5 --use_amp --mode binary --scheduler cosine --seed 42 --lr_backbone 1e-4 --lr_head 1e-3 --min_epochs 15 --patience 15 --focal_gamma 2.5"},
]


def run():
    results = []
    total = len(EXPERIMENTS)

    for i, exp in enumerate(EXPERIMENTS):
        name = exp["name"]
        log_dir = Path("logs") / name

        if (log_dir / "best_info.json").exists():
            info = json.load(open(log_dir / "best_info.json"))
            bm = info["test_metrics"]
            results.append({"name": name, "abn": bm["abnormal"]["recall"],
                          "nor": bm["normal"]["recall"], "f1": info["test_f1"], "ep": info["epoch"]})
            print(f"[SKIP] {name}")
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] {name}")
        print(f"{'='*60}")

        cmd = f'python train.py {exp["args"]} --log_dir logs/{name}'
        t0 = time.time()
        subprocess.run(cmd, shell=True, cwd="D:/project/anomaly-detection")
        elapsed = time.time() - t0

        if (log_dir / "best_info.json").exists():
            info = json.load(open(log_dir / "best_info.json"))
            bm = info["test_metrics"]
            r = {"name": name, "abn": bm["abnormal"]["recall"],
                 "nor": bm["normal"]["recall"], "f1": info["test_f1"], "ep": info["epoch"]}
            results.append(r)
            ok = "** OK **" if r["abn"] >= 0.99 and r["f1"] >= 0.90 else ""
            print(f"  -> abn={r['abn']:.4f} nor={r['nor']:.4f} F1={r['f1']:.4f} ep={r['ep']} {ok}")
        else:
            results.append({"name": name, "abn": 0, "nor": 0, "f1": 0, "ep": 0})

    # 결과 정렬
    results.sort(key=lambda x: x.get("f1", 0), reverse=True)

    print(f"\n{'='*60}")
    print(f"  v2 실험 결과 (test 성능, F1 순)")
    print(f"{'='*60}")
    print(f"  {'#':>2s} {'Name':>20s} | {'Abn_R':>6s} | {'Nor_R':>6s} | {'F1':>6s} | Ep | Target")
    print(f"  -- --------------------+--------+--------+--------+----+-------")
    for i, r in enumerate(results):
        ok = '** OK **' if r['abn'] >= 0.99 and r['f1'] >= 0.90 else ''
        print(f"  {i+1:>2d} {r['name']:>20s} | {r['abn']:6.4f} | {r['nor']:6.4f} | {r['f1']:6.4f} | {r['ep']:>2} | {ok}")

    with open("logs/v2_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run()
