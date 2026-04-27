from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validations" / "paper_urgent_reference_ls007_queue.json"
BASE_CONFIG = "logs\\260412_044744_fresh0412_v11_n700_s42_F0.9920_R0.9920\\data_config_used.yaml"
SEEDS = [42, 1, 2, 3, 4]


BASE_ARGS = {
    "--mode": "binary",
    "--config": BASE_CONFIG,
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
    "--warmup_epochs": 5,
    "--grad_clip": 1.0,
    "--weight_decay": 0.01,
}


def run(tag: str, candidate: str, seed: int, args: dict[str, object], reason: str) -> dict[str, object]:
    payload = {
        "tag": tag,
        "candidate": candidate,
        "seed": seed,
        "args": dict(args),
        "reason": reason,
    }
    payload["args"]["--seed"] = seed
    return payload


def main() -> int:
    runs: list[dict[str, object]] = []

    # Parse the in-flight gc=1.25 seed after it finishes, before urgent follow-ups.
    gc_args = dict(BASE_ARGS)
    gc_args["--grad_clip"] = 1.25
    runs.append(
        run(
            "fresh0412_v11_gc125_n700_s4",
            "fresh0412_v11_gc125_n700",
            4,
            gc_args,
            "finish parsing in-flight gc=1.25 seed before urgent reference checks",
        )
    )

    for seed in SEEDS:
        runs.append(
            run(
                f"fresh0412_v11_refcheck_n700_s{seed}",
                "fresh0412_v11_refcheck_n700",
                seed,
                BASE_ARGS,
                "urgent same-condition reference rerun to verify baseline reproducibility",
            )
        )

    ls_args = dict(BASE_ARGS)
    ls_args["--label_smoothing"] = 0.07
    for seed in SEEDS:
        runs.append(
            run(
                f"fresh0412_v11_regls007_n700_s{seed}",
                "fresh0412_v11_regls007_n700",
                seed,
                ls_args,
                "label_smoothing=0.07 neighbor added between no-smoothing and 0.1/0.15",
            )
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "created_at": "2026-04-28T08:20:00",
                "selected_reference": "fresh0412_v11_n700_existing",
                "selected_config": BASE_CONFIG,
                "note": "Urgent queue: parse in-flight gc=1.25, rerun reference, then add label_smoothing=0.07.",
                "runs": runs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
