"""
오분류 이미지 자동 분석
- best 모델의 errors/ 디렉토리 스캔
- 클래스별 오분류 패턴 분석
- defect_params 분포 (어떤 강도가 놓침)
- 이미지 sample 추출 (분석용)
"""
import sys, json, os
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict


def analyze(exp_name: str):
    log_dir = Path(f"logs/{exp_name}")
    info_file = log_dir / "best_info.json"
    err_dir = log_dir / "errors"

    if not info_file.exists():
        print(f"[!] {exp_name}: best_info.json 없음")
        return

    info = json.load(open(info_file))
    bm = info["test_metrics"]
    mode = info.get("hparams", {}).get("mode", "?")

    print(f"\n{'='*60}")
    print(f"  {exp_name} 분석")
    print(f"{'='*60}")
    print(f"  test_recall: {info['test_recall']:.4f}, F1: {info['test_f1']:.4f}, ep: {info['epoch']}")
    if mode == "binary" and "abnormal" in bm:
        print(f"  Binary: abn={bm['abnormal']['recall']:.4f}, nor={bm['normal']['recall']:.4f}")

    if not err_dir.exists():
        print("  errors/ 디렉토리 없음")
        return

    # 오분류 집계
    error_files = {}
    for sub in os.listdir(err_dir):
        if sub.startswith("true_"):
            true_cls = sub.replace("true_", "")
            files = os.listdir(err_dir / sub)
            error_files[true_cls] = files

    print(f"\n  === 클래스별 오분류 ===")
    for true_cls, files in error_files.items():
        # pred별 분류
        pred_counter = Counter()
        for f in files:
            pred = f.split("_ch_")[0].replace("pred_", "")
            pred_counter[pred] += 1
        print(f"    {true_cls:>15s}: {len(files)}건 → {dict(pred_counter)}")

    # defect_params 분포 (놓친 불량의 sigma_factor)
    sc_path = Path("data/scenarios.csv")
    if sc_path.exists() and "true_abnormal" in error_files:
        sc = pd.read_csv(sc_path)
        sigma_by_class = defaultdict(list)
        for f in error_files["true_abnormal"]:
            cid = "ch_" + f.split("ch_")[1].replace(".png", "")
            row = sc[sc["chart_id"] == cid]
            if not row.empty:
                r = row.iloc[0]
                cls = r["class"]
                try:
                    p = json.loads(r["defect_params"]) if r["defect_params"] != "{}" else {}
                    if "sigma_factor" in p:
                        sigma_by_class[cls].append(p["sigma_factor"])
                    elif "scale" in p:
                        sigma_by_class[cls].append(p["scale"])
                except:
                    pass

        print(f"\n  === 놓친 불량 강도 ===")
        for cls, sigmas in sigma_by_class.items():
            if sigmas:
                import numpy as np
                print(f"    {cls:>20s}: n={len(sigmas)}, mean={np.mean(sigmas):.2f}, min={min(sigmas):.2f}, max={max(sigmas):.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze(sys.argv[1])
    else:
        # 가장 최근 v2 실험들 분석
        v2_done = [d for d in sorted(os.listdir("logs"))
                   if d.startswith("v2_") and (Path(f"logs/{d}") / "best_info.json").exists()]
        for d in v2_done[-3:]:
            analyze(d)
