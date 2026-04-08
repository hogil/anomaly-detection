"""
개선 실험 2차 - imp_cosine(99.6%) 기반 세분화 + 새로운 축

imp_cosine: cosine + aw3 + min8 → abn=99.6%
이것을 기반으로 더 다양한 조합 탐색
"""
import subprocess, json, time, os
from pathlib import Path

BASE_COS = "--epochs 50 --batch_size 32 --warmup_epochs 5 --use_amp --mode binary --scheduler cosine --min_epochs 8"
BASE_STEP = "--epochs 50 --batch_size 32 --warmup_epochs 5 --use_amp --mode binary --scheduler step --min_epochs 8"

EXPERIMENTS = [
    # === A. Cosine + AW 세분화 (best scheduler 기반) ===
    {"name": "imp2_cos_aw1",   "args": f"{BASE_COS} --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 1.0"},
    {"name": "imp2_cos_aw2",   "args": f"{BASE_COS} --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 2.0"},
    {"name": "imp2_cos_aw4",   "args": f"{BASE_COS} --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 4.0"},
    {"name": "imp2_cos_aw5",   "args": f"{BASE_COS} --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 5.0"},
    {"name": "imp2_cos_aw7",   "args": f"{BASE_COS} --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 7.0"},
    {"name": "imp2_cos_aw10",  "args": f"{BASE_COS} --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 10.0"},

    # === B. Cosine + Gamma 세분화 (aw=3 고정) ===
    {"name": "imp2_cos_g1",    "args": f"{BASE_COS} --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 3.0 --focal_gamma 1.0"},
    {"name": "imp2_cos_g15",   "args": f"{BASE_COS} --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 3.0 --focal_gamma 1.5"},
    {"name": "imp2_cos_g25",   "args": f"{BASE_COS} --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 3.0 --focal_gamma 2.5"},
    {"name": "imp2_cos_g3",    "args": f"{BASE_COS} --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 3.0 --focal_gamma 3.0"},

    # === C. Cosine + LR 미세 조정 ===
    {"name": "imp2_cos_lr1",   "args": f"{BASE_COS} --lr_backbone 3e-5 --lr_head 3e-4 --abnormal_weight 3.0"},
    {"name": "imp2_cos_lr2",   "args": f"{BASE_COS} --lr_backbone 7e-5 --lr_head 7e-4 --abnormal_weight 3.0"},
    {"name": "imp2_cos_lr3",   "args": f"{BASE_COS} --lr_backbone 1.5e-4 --lr_head 1.5e-3 --abnormal_weight 3.0"},
    {"name": "imp2_cos_lr4",   "args": f"{BASE_COS} --lr_backbone 2e-4 --lr_head 2e-3 --abnormal_weight 3.0"},

    # === D. Cosine + Dropout 세분화 ===
    {"name": "imp2_cos_d02",   "args": f"{BASE_COS} --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 3.0 --dropout 0.2"},
    {"name": "imp2_cos_d03",   "args": f"{BASE_COS} --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 3.0 --dropout 0.3"},
    {"name": "imp2_cos_d04",   "args": f"{BASE_COS} --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 3.0 --dropout 0.4"},

    # === E. Warmup 변동 ===
    {"name": "imp2_cos_w3",    "args": f"--epochs 50 --batch_size 32 --warmup_epochs 3 --use_amp --mode binary --scheduler cosine --min_epochs 8 --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 3.0"},
    {"name": "imp2_cos_w10",   "args": f"--epochs 50 --batch_size 32 --warmup_epochs 10 --use_amp --mode binary --scheduler cosine --min_epochs 15 --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 3.0"},

    # === F. Batch size 변동 ===
    {"name": "imp2_cos_b16",   "args": f"--epochs 50 --batch_size 16 --warmup_epochs 5 --use_amp --mode binary --scheduler cosine --min_epochs 8 --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 3.0"},
    {"name": "imp2_cos_b64",   "args": f"--epochs 50 --batch_size 64 --warmup_epochs 5 --use_amp --mode binary --scheduler cosine --min_epochs 8 --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 3.0"},

    # === G. Weight Decay 변동 ===
    {"name": "imp2_cos_wd001", "args": f"{BASE_COS} --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 3.0 --weight_decay 0.001"},
    {"name": "imp2_cos_wd05",  "args": f"{BASE_COS} --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 3.0 --weight_decay 0.05"},

    # === H. StepLR 10/0.5 (2nd best) + AW 세분화 ===
    {"name": "imp2_s10_aw3",   "args": f"{BASE_STEP} --step_size 10 --step_gamma 0.5 --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 3.0"},
    {"name": "imp2_s10_aw5",   "args": f"{BASE_STEP} --step_size 10 --step_gamma 0.5 --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 5.0"},
    {"name": "imp2_s10_aw8",   "args": f"{BASE_STEP} --step_size 10 --step_gamma 0.5 --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 8.0"},

    # === I. 최적 조합 후보 ===
    {"name": "imp2_opt1",      "args": f"{BASE_COS} --lr_backbone 7e-5 --lr_head 7e-4 --abnormal_weight 5.0 --dropout 0.3"},
    {"name": "imp2_opt2",      "args": f"{BASE_COS} --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 5.0 --focal_gamma 2.5 --dropout 0.3"},
    {"name": "imp2_opt3",      "args": f"{BASE_COS} --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 3.0 --focal_gamma 1.5 --dropout 0.3"},
    {"name": "imp2_opt4",      "args": f"{BASE_COS} --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 7.0 --focal_gamma 2.0 --dropout 0.4"},
    {"name": "imp2_opt5",      "args": f"--epochs 70 --batch_size 32 --warmup_epochs 5 --use_amp --mode binary --scheduler cosine --min_epochs 15 --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 5.0 --patience 15"},
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
            results.append({"name": name, "abn": bm["abnormal"]["recall"], "nor": bm["normal"]["recall"], "ep": info["epoch"]})
            print(f"[SKIP] {name} abn={bm['abnormal']['recall']:.4f}")
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
            r = {"name": name, "abn": bm["abnormal"]["recall"], "nor": bm["normal"]["recall"], "ep": info["epoch"]}
            results.append(r)
            print(f"  -> abn={r['abn']:.4f} nor={r['nor']:.4f} ep={r['ep']} ({elapsed:.0f}s)")
        else:
            results.append({"name": name, "abn": 0, "nor": 0, "ep": 0})

    # 결과 정렬
    results.sort(key=lambda x: x.get("abn", 0), reverse=True)

    print(f"\n{'='*60}")
    print(f"  2차 개선 결과 ({len(results)}개, abnormal recall 순)")
    print(f"{'='*60}")
    print(f"  {'#':>2s} {'Name':>20s} | {'Abn_R':>6s} | {'Nor_R':>6s} | Ep")
    print(f"  -- --------------------+--------+--------+---")
    for i, r in enumerate(results):
        print(f"  {i+1:>2d} {r['name']:>20s} | {r.get('abn',0):6.4f} | {r.get('nor',0):6.4f} | {r.get('ep',0):>2}")

    with open("logs/improved2_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run()
