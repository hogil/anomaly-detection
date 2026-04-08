"""Backbone 비교 실험

6 models × 3 seeds = 18 trials
- 각 모델 논문/레시피 기반 per-model hparams (lr, warmup)
- 공통: AdamW, cosine, epochs 20, batch 32, AMP, grad_clip 1.0,
         focal_gamma 0.0, min_epochs 10, val_f1 기준, normal_ratio 700
- Normalization은 train.py가 timm data_config에서 자동 추출
- 시간 측정 best_info.json에 저장
"""
import subprocess, json, os, statistics

# (key, model_name, lr_backbone, lr_head, warmup)
# 가벼운 모델부터 진행, ConvNeXtV2-Base는 크니까 맨 뒤
MODELS = [
    ("convnextv2_tiny",  "convnextv2_tiny.fcmae_ft_in22k_in1k",              5e-5, 5e-4, 5),
    ("efficientnetv2_s", "tf_efficientnetv2_s.in21k_ft_in1k",                5e-5, 5e-4, 5),
    ("swin_tiny",        "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",    5e-5, 5e-4, 10),
    ("maxvit_tiny",      "maxvit_tiny_tf_224.in1k",                          3e-5, 3e-4, 10),
    ("clip_vit_b16",     "vit_base_patch16_clip_224.laion2b_ft_in12k_in1k",  1e-5, 1e-4, 10),
    ("convnextv2_base",  "convnextv2_base.fcmae_ft_in22k_in1k",              3e-5, 3e-4, 5),
]
SEEDS = [42, 1, 2]

BASE = (
    "python train.py --epochs 20 --mode binary "
    "--scheduler cosine --use_amp --num_workers 4 "
    "--batch_size 32 "
    "--dropout 0.0 "
    "--focal_gamma 0.0 "
    "--abnormal_weight 1.0 "
    "--min_epochs 10 "
    "--normal_ratio 700 "
)

total = len(MODELS) * len(SEEDS)
done = 0
for key, model_name, lr_bb, lr_head, warmup in MODELS:
    for seed in SEEDS:
        done += 1
        name = f"bb_{key}_s{seed}"
        log_dir = f"logs/{name}"
        if os.path.exists(f"{log_dir}/best_info.json"):
            print(f"[SKIP {done}/{total}] {name}")
            continue
        print(f"\n{'='*70}\n[START {done}/{total}] {name}")
        print(f"  model={model_name}")
        print(f"  lr_bb={lr_bb}, lr_head={lr_head}, warmup={warmup}")
        print('='*70)
        cmd = (
            f"{BASE}"
            f"--model_name {model_name} "
            f"--lr_backbone {lr_bb} --lr_head {lr_head} "
            f"--warmup_epochs {warmup} "
            f"--seed {seed} "
            f"--log_dir {log_dir}"
        )
        print(f"  CMD: {cmd}")
        subprocess.run(cmd, shell=True)

# ======================================================================
# Results summary
# ======================================================================
print("\n\n" + "="*90)
print("BACKBONE COMPARISON RESULTS")
print("="*90)

def _load(name):
    p = f"logs/{name}/best_info.json"
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)

# Per-trial table
print(f"\n{'Model':<18} {'Seed':>5} {'Params':>8} {'EpTime':>8} {'Total':>8} {'best_ep':>8} "
      f"{'abn_R':>8} {'nor_R':>8} {'F1':>8}")
print("-"*90)
for key, _, _, _, _ in MODELS:
    for seed in SEEDS:
        name = f"bb_{key}_s{seed}"
        bi = _load(name)
        if bi is None:
            print(f"{key:<18} {seed:>5} (no result)")
            continue
        pM = bi.get("params_M", 0)
        timing = bi.get("timing", {})
        ep_t = timing.get("avg_epoch_sec", 0)
        total_t = timing.get("total_time_min", 0)
        best_ep = bi.get("epoch", 0)
        abn = bi["test_metrics"]["abnormal"]["recall"]
        nor = bi["test_metrics"]["normal"]["recall"]
        f1 = bi.get("test_f1", 0)
        print(f"{key:<18} {seed:>5} {pM:>7.1f}M {ep_t:>7.1f}s {total_t:>6.1f}min "
              f"{best_ep:>8} {abn:>8.4f} {nor:>8.4f} {f1:>8.4f}")

# Aggregated (mean ± std per model)
print(f"\n{'='*90}")
print("AGGREGATED (mean ± std over 3 seeds)")
print(f"{'='*90}")
print(f"{'Model':<18} {'Params':>8} {'avg_ep_s':>10} {'total_min':>11} "
      f"{'abn_R':>15} {'nor_R':>15} {'F1':>15}")
print("-"*95)

for key, _, _, _, _ in MODELS:
    results = []
    for seed in SEEDS:
        bi = _load(f"bb_{key}_s{seed}")
        if bi is not None:
            results.append(bi)
    if not results:
        print(f"{key:<18} (no results)")
        continue
    pM = results[0].get("params_M", 0)
    ep_times = [r.get("timing", {}).get("avg_epoch_sec", 0) for r in results]
    total_times = [r.get("timing", {}).get("total_time_min", 0) for r in results]
    abns = [r["test_metrics"]["abnormal"]["recall"] for r in results]
    nors = [r["test_metrics"]["normal"]["recall"] for r in results]
    f1s = [r["test_f1"] for r in results]

    def ms(xs):
        m = statistics.mean(xs)
        s = statistics.stdev(xs) if len(xs) > 1 else 0
        return f"{m:.4f}±{s:.4f}"

    print(f"{key:<18} {pM:>7.1f}M "
          f"{statistics.mean(ep_times):>9.1f}s "
          f"{statistics.mean(total_times):>10.1f}min "
          f"{ms(abns):>15} {ms(nors):>15} {ms(f1s):>15}")

print("\nDONE")
