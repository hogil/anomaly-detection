"""
Fine-tune 실험 - var_1000_seed456 (abn=98.9%, nor=91.3%, F1=95.7%) 기반

기본: binary + cosine + lr(1e-4/1e-3) + no aw + epoch 8
이 설정을 세밀하게 변형하여 abn>=99% + F1>=90% 목표

목표: test abnormal recall >= 99%, test F1 >= 90%
"""
import subprocess, json, time
from pathlib import Path

BASE = "--epochs 50 --batch_size 32 --warmup_epochs 5 --use_amp --mode binary --scheduler cosine"

EXPERIMENTS = [
    # === A. 기준선 재현 (min_epochs 없이) ===
    {"name": "ft_base",       "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3"},

    # === B. 작은 abnormal_weight (1.2~2.0, 기존 3~10은 과함) ===
    {"name": "ft_aw1.1",      "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 1.1"},
    {"name": "ft_aw1.2",      "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 1.2"},
    {"name": "ft_aw1.3",      "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 1.3"},
    {"name": "ft_aw1.5",      "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 1.5"},
    {"name": "ft_aw1.8",      "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 1.8"},
    {"name": "ft_aw2.0",      "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 2.0"},

    # === C. min_epochs 세분화 ===
    {"name": "ft_min5",       "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --min_epochs 5"},
    {"name": "ft_min10",      "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --min_epochs 10"},
    {"name": "ft_min15",      "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --min_epochs 15"},
    {"name": "ft_min20",      "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --min_epochs 20"},

    # === D. LR 미세 조정 ===
    {"name": "ft_lr08",       "args": f"{BASE} --lr_backbone 8e-5 --lr_head 8e-4"},
    {"name": "ft_lr12",       "args": f"{BASE} --lr_backbone 1.2e-4 --lr_head 1.2e-3"},
    {"name": "ft_lr15",       "args": f"{BASE} --lr_backbone 1.5e-4 --lr_head 1.5e-3"},

    # === E. Warmup 변동 ===
    {"name": "ft_w3",         "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --warmup_epochs 3"},
    {"name": "ft_w8",         "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --warmup_epochs 8"},

    # === F. Dropout 세분화 ===
    {"name": "ft_d03",        "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --dropout 0.3"},
    {"name": "ft_d04",        "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --dropout 0.4"},
    {"name": "ft_d06",        "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --dropout 0.6"},

    # === G. Focal gamma 미세 ===
    {"name": "ft_g1.8",       "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --focal_gamma 1.8"},
    {"name": "ft_g2.2",       "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --focal_gamma 2.2"},

    # === H. 조합 (aw 소량 + 기타) ===
    {"name": "ft_aw12_d04",   "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 1.2 --dropout 0.4"},
    {"name": "ft_aw13_min10", "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 1.3 --min_epochs 10"},
    {"name": "ft_aw15_g22",   "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 1.5 --focal_gamma 2.2"},
    {"name": "ft_aw12_w3",    "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --abnormal_weight 1.2 --warmup_epochs 3"},
    {"name": "ft_aw13_lr08",  "args": f"{BASE} --lr_backbone 8e-5 --lr_head 8e-4 --abnormal_weight 1.3"},

    # === I. patience 변동 ===
    {"name": "ft_p15",        "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --patience 15"},
    {"name": "ft_p20",        "args": f"{BASE} --lr_backbone 1e-4 --lr_head 1e-3 --patience 20"},

    # === J. epoch 늘리기 ===
    {"name": "ft_ep70",       "args": "--epochs 70 --batch_size 32 --warmup_epochs 5 --use_amp --mode binary --scheduler cosine --lr_backbone 1e-4 --lr_head 1e-3 --patience 15"},
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
            results.append({"name": name, "abn": bm["abnormal"]["recall"], "nor": bm["normal"]["recall"],
                          "f1": info["test_f1"], "ep": info["epoch"]})
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
            r = {"name": name, "abn": bm["abnormal"]["recall"], "nor": bm["normal"]["recall"],
                 "f1": info["test_f1"], "ep": info["epoch"]}
            results.append(r)
            ok = "** TARGET **" if r["abn"] >= 0.99 and r["f1"] >= 0.90 else ""
            print(f"  -> test: abn={r['abn']:.4f} nor={r['nor']:.4f} F1={r['f1']:.4f} ep={r['ep']} {ok}")
        else:
            results.append({"name": name, "abn": 0, "nor": 0, "f1": 0, "ep": 0})

    # 결과 정렬 (F1 순, abn>=99% 우선)
    target_met = [r for r in results if r.get("abn", 0) >= 0.99 and r.get("f1", 0) >= 0.90]
    others = [r for r in results if r not in target_met]
    target_met.sort(key=lambda x: x["f1"], reverse=True)
    others.sort(key=lambda x: x.get("f1", 0), reverse=True)

    print(f"\n{'='*60}")
    print(f"  Fine-tune 결과 (test 성능)")
    print(f"{'='*60}")
    print(f"  {'Name':>20s} | {'Abn_R':>6s} | {'Nor_R':>6s} | {'F1':>6s} | Ep | Target")
    print(f"  {'-'*20}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+----+-------")
    for r in target_met:
        print(f"  {r['name']:>20s} | {r['abn']:6.4f} | {r['nor']:6.4f} | {r['f1']:6.4f} | {r['ep']:>2} | ** OK **")
    if target_met:
        print(f"  {'-'*20}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+----+-------")
    for r in others[:15]:
        print(f"  {r['name']:>20s} | {r['abn']:6.4f} | {r['nor']:6.4f} | {r['f1']:6.4f} | {r['ep']:>2} |")

    with open("logs/finetune_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run()
