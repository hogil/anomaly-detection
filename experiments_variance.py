"""
성능 산포 분석 - 동일 조건, 다른 seed × 다른 sample 수

동일 모델/하이퍼파라미터, 데이터만 다르게 생성하여
각 sample 수별 성능의 평균 ± 표준편차를 측정.
"""
import subprocess, json, time, yaml
import numpy as np
from pathlib import Path

N_RUNS = 5
SEEDS = [42, 123, 456, 789, 2024]
SIZES = [200, 400, 600, 800, 1000]
MODE = "binary"

TRAIN_ARGS = "--epochs 50 --batch_size 32 --lr_backbone 1e-4 --lr_head 1e-3 --warmup_epochs 5 --scheduler step --step_size 15 --step_gamma 0.5 --use_amp"


def run():
    all_results = {}

    total_exp = len(SIZES) * len(SEEDS)
    exp_num = 0

    for size in SIZES:
        size_results = []

        for seed in SEEDS:
            exp_num += 1
            name = f"var_{size}_seed{seed}"
            log_dir = Path("logs") / name

            if (log_dir / "best_info.json").exists():
                print(f"[SKIP] {name}")
                info = json.load(open(log_dir / "best_info.json"))
                bm = info["test_metrics"]
                size_results.append({
                    "seed": seed, "size": size,
                    "abnormal_recall": bm["abnormal"]["recall"],
                    "normal_recall": bm["normal"]["recall"],
                    "test_recall": info["test_recall"],
                    "epoch": info["epoch"],
                })
                continue

            print(f"\n{'='*60}")
            print(f"[{exp_num}/{total_exp}] size={size}, seed={seed}")
            print(f"{'='*60}")

            # Config
            with open("config.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            config["seed"] = seed
            config["dataset"]["samples_per_class"] = size

            var_config = f"config_var.yaml"
            with open(var_config, "w", encoding="utf-8") as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

            # 데이터 + 이미지 생성
            print(f"  데이터+이미지 생성...")
            t0 = time.time()
            subprocess.run(f"python generate_data.py --config {var_config}", shell=True)
            subprocess.run(f"python generate_images.py --config {var_config}", shell=True)
            gen_time = time.time() - t0
            print(f"  생성 완료 ({gen_time:.0f}s)")

            # 학습
            print(f"  학습 시작...")
            t0 = time.time()
            subprocess.run(
                f"python train.py {TRAIN_ARGS} --mode {MODE} --log_dir logs/{name}",
                shell=True
            )
            train_time = time.time() - t0

            if (log_dir / "best_info.json").exists():
                info = json.load(open(log_dir / "best_info.json"))
                bm = info["test_metrics"]
                r = {
                    "seed": seed, "size": size,
                    "abnormal_recall": bm["abnormal"]["recall"],
                    "normal_recall": bm["normal"]["recall"],
                    "test_recall": info["test_recall"],
                    "epoch": info["epoch"],
                }
                size_results.append(r)
                print(f"  → abnormal_R={r['abnormal_recall']:.4f}")
            else:
                print(f"  → FAILED")

        all_results[size] = size_results

    # 임시 config 삭제
    Path("config_var.yaml").unlink(missing_ok=True)

    # 결과 요약
    print(f"\n{'='*60}")
    print(f"  성능 산포 분석 ({MODE}, {N_RUNS}회 반복)")
    print(f"{'='*60}")
    print(f"  {'Size':>6s} | {'Abnormal Recall':>20s} | {'Min':>6s} | {'Max':>6s} | {'Normal Recall':>20s}")
    print(f"  {'-'*6}-+-{'-'*20}-+-{'-'*6}-+-{'-'*6}-+-{'-'*20}")

    summary = {}
    for size in SIZES:
        rs = all_results.get(size, [])
        if not rs:
            continue
        abn = [r["abnormal_recall"] for r in rs]
        nor = [r["normal_recall"] for r in rs]
        summary[size] = {
            "abnormal_mean": round(float(np.mean(abn)), 4),
            "abnormal_std": round(float(np.std(abn)), 4),
            "abnormal_min": round(float(min(abn)), 4),
            "abnormal_max": round(float(max(abn)), 4),
            "normal_mean": round(float(np.mean(nor)), 4),
            "normal_std": round(float(np.std(nor)), 4),
        }
        print(f"  {size:>6d} | {np.mean(abn):.4f} ± {np.std(abn):.4f}      | {min(abn):.4f} | {max(abn):.4f} | {np.mean(nor):.4f} ± {np.std(nor):.4f}")

    with open("logs/variance_results.json", "w") as f:
        json.dump({"runs": {str(k): v for k, v in all_results.items()}, "summary": {str(k): v for k, v in summary.items()}}, f, indent=2)

    print(f"\n  저장: logs/variance_results.json")


if __name__ == "__main__":
    run()
