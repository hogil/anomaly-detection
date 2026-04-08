"""
2가지 축 실험:
1. Gamma 세분화 (0.5 ~ 5.0)
2. Normal 갯수 변동 (abnormal 고정, normal만 늘리기)

기본: binary + cosine + lr(1e-4/1e-3)
목표: test abn>=99% + test F1>=90%
"""
import subprocess, json, time, yaml
import pandas as pd
from pathlib import Path

BASE = "--epochs 50 --batch_size 32 --lr_backbone 1e-4 --lr_head 1e-3 --warmup_epochs 5 --use_amp --mode binary --scheduler cosine"

EXPERIMENTS = [
    # === A. Gamma 세분화 ===
    {"name": "gm_g05",  "args": f"{BASE} --focal_gamma 0.5"},
    {"name": "gm_g10",  "args": f"{BASE} --focal_gamma 1.0"},
    {"name": "gm_g15",  "args": f"{BASE} --focal_gamma 1.5"},
    {"name": "gm_g20",  "args": f"{BASE} --focal_gamma 2.0"},
    {"name": "gm_g25",  "args": f"{BASE} --focal_gamma 2.5"},
    {"name": "gm_g30",  "args": f"{BASE} --focal_gamma 3.0"},
    {"name": "gm_g35",  "args": f"{BASE} --focal_gamma 3.5"},
    {"name": "gm_g40",  "args": f"{BASE} --focal_gamma 4.0"},
    {"name": "gm_g50",  "args": f"{BASE} --focal_gamma 5.0"},

    # === B. Normal 갯수 변동 (abnormal ~700 고정, normal만 변동) ===
    # max_per_class는 전체에 적용되므로, normal만 늘리려면 별도 처리 필요
    # → scenarios.csv에서 normal 샘플을 복제하는 방식으로 구현
    {"name": "gm_nor100",  "args": f"{BASE} --focal_gamma 2.0 --normal_ratio 100"},
    {"name": "gm_nor200",  "args": f"{BASE} --focal_gamma 2.0 --normal_ratio 200"},
    {"name": "gm_nor350",  "args": f"{BASE} --focal_gamma 2.0 --normal_ratio 350"},
    {"name": "gm_nor500",  "args": f"{BASE} --focal_gamma 2.0 --normal_ratio 500"},
    {"name": "gm_nor700",  "args": f"{BASE} --focal_gamma 2.0 --normal_ratio 700"},
    {"name": "gm_nor1000", "args": f"{BASE} --focal_gamma 2.0 --normal_ratio 1000"},
    {"name": "gm_nor1500", "args": f"{BASE} --focal_gamma 2.0 --normal_ratio 1500"},
    {"name": "gm_nor2000", "args": f"{BASE} --focal_gamma 2.0 --normal_ratio 2000"},

    # === C. Best gamma + normal 조합 ===
    {"name": "gm_g30_n1000", "args": f"{BASE} --focal_gamma 3.0 --normal_ratio 1000"},
    {"name": "gm_g35_n1000", "args": f"{BASE} --focal_gamma 3.5 --normal_ratio 1000"},
    {"name": "gm_g30_n1500", "args": f"{BASE} --focal_gamma 3.0 --normal_ratio 1500"},
    {"name": "gm_g25_n1500", "args": f"{BASE} --focal_gamma 2.5 --normal_ratio 1500"},
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
            ok = "** TARGET **" if r["abn"] >= 0.99 and r["f1"] >= 0.90 else ""
            print(f"  test: abn={r['abn']:.4f} nor={r['nor']:.4f} F1={r['f1']:.4f} ep={r['ep']} {ok}")
        else:
            results.append({"name": name, "abn": 0, "nor": 0, "f1": 0, "ep": 0})

    # 결과
    print(f"\n{'='*60}")
    print(f"  Gamma + Normal 실험 결과 (test 성능)")
    print(f"{'='*60}")

    # Gamma 결과
    gamma_r = [r for r in results if r["name"].startswith("gm_g") and "n" not in r["name"].split("_")[-1]]
    if gamma_r:
        print(f"\n  [Gamma 세분화]")
        print(f"  {'Name':>15s} | {'Abn_R':>6s} | {'Nor_R':>6s} | {'F1':>6s} | Ep")
        print(f"  {'-'*15}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+---")
        for r in sorted(gamma_r, key=lambda x: x.get("abn",0), reverse=True):
            print(f"  {r['name']:>15s} | {r['abn']:6.4f} | {r['nor']:6.4f} | {r['f1']:6.4f} | {r['ep']:>2}")

    # Normal 갯수 결과
    nor_r = [r for r in results if r["name"].startswith("gm_nor")]
    if nor_r:
        print(f"\n  [Normal 갯수 변동]")
        print(f"  {'Name':>15s} | {'Abn_R':>6s} | {'Nor_R':>6s} | {'F1':>6s} | Ep")
        print(f"  {'-'*15}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+---")
        for r in sorted(nor_r, key=lambda x: int(''.join(filter(str.isdigit, x['name'])))):
            print(f"  {r['name']:>15s} | {r['abn']:6.4f} | {r['nor']:6.4f} | {r['f1']:6.4f} | {r['ep']:>2}")

    with open("logs/gamma_normal_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run()
