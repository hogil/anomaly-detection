"""
Label Weight 실험 - multiclass 6클래스에서 개별 label별 weight 조정

놓침이 많은 label에 높은 weight → 해당 label recall 향상
normal weight 낮추기 → abnormal 전체 recall 향상

현재 6클래스 recall:
  normal=100%, spike=69.3%, context=89.3%, mean_shift=90.7%, std=92.0%, drift=94.7%
"""
import subprocess, json, time
from pathlib import Path

BASE = "--epochs 50 --batch_size 32 --lr_backbone 1e-4 --lr_head 1e-3 --warmup_epochs 5 --use_amp --mode multiclass --scheduler cosine --min_epochs 8"

EXPERIMENTS = [
    # === 1. Normal weight 낮추기 ===
    {"name": "lw_nor05",     "args": f'{BASE} --label_weights "normal=0.5"'},
    {"name": "lw_nor03",     "args": f'{BASE} --label_weights "normal=0.3"'},
    {"name": "lw_nor01",     "args": f'{BASE} --label_weights "normal=0.1"'},

    # === 2. Spike 집중 (최약 클래스) ===
    {"name": "lw_spk3",      "args": f'{BASE} --label_weights "normal=0.5,spike=3.0"'},
    {"name": "lw_spk5",      "args": f'{BASE} --label_weights "normal=0.5,spike=5.0"'},
    {"name": "lw_spk8",      "args": f'{BASE} --label_weights "normal=0.3,spike=8.0"'},

    # === 3. 약한 클래스 전부 올리기 ===
    {"name": "lw_anom2",     "args": f'{BASE} --label_weights "normal=0.5,spike=3.0,context=2.0,mean_shift=2.0"'},
    {"name": "lw_anom3",     "args": f'{BASE} --label_weights "normal=0.3,spike=5.0,context=3.0,mean_shift=3.0,standard_deviation=2.0"'},
    {"name": "lw_anom5",     "args": f'{BASE} --label_weights "normal=0.3,spike=5.0,context=3.0,mean_shift=3.0,standard_deviation=2.0,drift=1.5"'},

    # === 4. 놓침 건수 비례 weight ===
    # spike=46, context=16, mean_shift=14, std=11, drift=8 → 비례
    {"name": "lw_prop",      "args": f'{BASE} --label_weights "normal=0.3,spike=4.6,context=1.6,mean_shift=1.4,standard_deviation=1.1,drift=0.8"'},

    # === 5. Gamma 조합 ===
    {"name": "lw_g15",       "args": f'{BASE} --label_weights "normal=0.3,spike=5.0,context=3.0,mean_shift=3.0" --focal_gamma 1.5'},
    {"name": "lw_g25",       "args": f'{BASE} --label_weights "normal=0.3,spike=5.0,context=3.0,mean_shift=3.0" --focal_gamma 2.5'},

    # === 6. Dropout 조합 ===
    {"name": "lw_d03",       "args": f'{BASE} --label_weights "normal=0.3,spike=5.0,context=3.0,mean_shift=3.0" --dropout 0.3'},

    # === 7. 비교: binary baseline (동일 설정) ===
    {"name": "lw_binary",    "args": "--epochs 50 --batch_size 32 --lr_backbone 1e-4 --lr_head 1e-3 --warmup_epochs 5 --use_amp --mode binary --scheduler cosine --min_epochs 8 --abnormal_weight 3.0"},
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
            mode = info.get("hparams", {}).get("mode", "?")
            if mode == "binary":
                results.append({"name": name, "mode": mode, "abn_r": bm["abnormal"]["recall"], "nor_r": bm["normal"]["recall"], "ep": info["epoch"]})
            else:
                anomaly_r = {c: bm[c]["recall"] for c in bm if c != "normal"}
                avg_abn = sum(anomaly_r.values()) / len(anomaly_r)
                results.append({"name": name, "mode": mode, "abn_r": avg_abn, "nor_r": bm["normal"]["recall"],
                               "spike_r": bm.get("spike", {}).get("recall", 0), "ep": info["epoch"], "detail": anomaly_r})
            print(f"[SKIP] {name}")
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] {name}")
        print(f"{'='*60}")

        cmd = f'python train.py {exp["args"]} --log_dir logs/{name}'
        subprocess.run(cmd, shell=True, cwd="D:/project/anomaly-detection")

        if (log_dir / "best_info.json").exists():
            info = json.load(open(log_dir / "best_info.json"))
            bm = info["test_metrics"]
            mode = info.get("hparams", {}).get("mode", "?")
            if mode == "binary":
                r = {"name": name, "mode": mode, "abn_r": bm["abnormal"]["recall"], "nor_r": bm["normal"]["recall"], "ep": info["epoch"]}
            else:
                anomaly_r = {c: bm[c]["recall"] for c in bm if c != "normal"}
                avg_abn = sum(anomaly_r.values()) / len(anomaly_r)
                r = {"name": name, "mode": mode, "abn_r": avg_abn, "nor_r": bm["normal"]["recall"],
                     "spike_r": bm.get("spike", {}).get("recall", 0), "ep": info["epoch"], "detail": anomaly_r}
            results.append(r)
            print(f"  -> abn_avg={r['abn_r']:.4f}, nor={r['nor_r']:.4f}, spike={r.get('spike_r','N/A')}")
        else:
            results.append({"name": name, "abn_r": 0, "nor_r": 0})

    # 결과
    results.sort(key=lambda x: x.get("abn_r", 0), reverse=True)
    print(f"\n{'='*60}")
    print(f"  Label Weight 실험 결과 (anomaly avg recall 순)")
    print(f"{'='*60}")
    print(f"  {'Name':>15s} | {'Mode':>10s} | {'Abn_R':>6s} | {'Nor_R':>6s} | {'Spike':>6s} | Ep")
    print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+---")
    for r in results:
        spk = f"{r.get('spike_r',0):.4f}" if 'spike_r' in r else "  N/A "
        print(f"  {r['name']:>15s} | {r.get('mode','?'):>10s} | {r['abn_r']:6.4f} | {r['nor_r']:6.4f} | {spk} | {r.get('ep','?'):>2}")

    with open("logs/labelweight_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    run()
