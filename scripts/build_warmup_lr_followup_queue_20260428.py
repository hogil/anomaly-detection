from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validations" / "paper_strict_warmup_lr_followup_queue.json"
SELECTED_CONFIG = "logs\\260412_044744_fresh0412_v11_n700_s42_F0.9920_R0.9920\\data_config_used.yaml"
SEEDS = [42, 1, 2, 3, 4]
WARMUP_LEVELS = [2, 4, 6, 10]


BASE_ARGS = {
    "--mode": "binary",
    "--config": SELECTED_CONFIG,
    "--epochs": 20,
    "--patience": 5,
    "--batch_size": 32,
    "--dropout": 0.0,
    "--precision": "fp16",
    "--num_workers": 0,
    "--ema_decay": 0.0,
    "--normal_ratio": 700,
    "--smooth_window": 3,
    "--smooth_method": "median",
    "--lr_backbone": "2e-5",
    "--lr_head": "2e-4",
    "--grad_clip": 1.0,
    "--weight_decay": 0.01,
}


def build() -> dict:
    runs = []
    for warmup in WARMUP_LEVELS:
        candidate = f"fresh0412_v11_lrwarm{warmup}_n700"
        for seed in SEEDS:
            args = dict(BASE_ARGS)
            args["--warmup_epochs"] = warmup
            runs.append(
                {
                    "tag": f"{candidate}_s{seed}",
                    "candidate": candidate,
                    "seed": seed,
                    "args": args,
                    "reason": (
                        "strict one-factor warmup follow-up under fixed baseline; "
                        "lr_backbone/lr_head held at 2e-5/2e-4"
                    ),
                }
            )
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selected_reference": "fresh0412_v11_n700_existing",
        "selected_config": SELECTED_CONFIG,
        "note": "Warmup-only follow-up. Baseline fixed; only --warmup_epochs changes. Learning rate remains 2e-5/2e-4.",
        "runs": runs,
    }


def main() -> int:
    payload = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"queue": str(OUT), "runs": len(payload["runs"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
