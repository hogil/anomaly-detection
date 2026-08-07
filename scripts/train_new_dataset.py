#!/usr/bin/env python
"""서버 sweep 결과(validations/)의 BKM 조건으로 **새 데이터셋**을 학습한다.

이것만 치면 끝난다 — 데이터 생성 · 이미지 렌더 · 학습까지 알아서 한다:

  python scripts/train_new_dataset.py --validation validations/<TS>_all_dataset_backbone

기본값: dataset = configs/datasets/dataset_v25.yaml
        backbone = convnext_tiny.dinov3_lvd1689m
        전체 데이터 (--max_per_class 0 --normal_ratio 0), seed 42, BKM scope global

데이터/이미지 폴더가 이미 있으면 그대로 쓴다 (덮어쓰지 않는다). 없을 때만 만든다.

기존 스크립트로는 안 되는 조합이라 따로 뒀다:
  final_train_all_datasets.py : 옛 matrix 에 있던 데이터셋만 (새 데이터셋 불가)
  train_combined_datasets.py  : 여러 데이터셋을 합침 (하드링크 사본이 생김)
  이 파일                     : 새 데이터셋 1개를 사본 없이 그대로, backbone 고정

train.py 옵션 override 는 `--` 뒤에:
  python scripts/train_new_dataset.py --validation validations/<TS> -- --batch_size 96
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.final_train_all_datasets import (  # noqa: E402
    FULL_DATA_ARGS, find_run_dir, load_bkm_queues, resolve_roots, warn_short_epochs,
)
from scripts.retrain_from import (  # noqa: E402
    args_dict_to_cli, collect_records, short_backbone,
)
from scripts.train_combined_datasets import pick_bkm_for_backbone  # noqa: E402

DEFAULT_BACKBONE = "convnext_tiny.dinov3_lvd1689m"


def snapshot_for_backbone(logs_root: Path, short: str) -> Path | None:
    """BKM 이 없을 때 대체 — 그 backbone 최고 성적 run 의 train_config_used.yaml."""
    if not logs_root.exists():
        return None
    scored = [r for r in collect_records(logs_root)
              if short_backbone(r.get("model_name")) == short and r.get("test_f1") is not None]
    if not scored:
        return None
    best = max(scored, key=lambda r: r["test_f1"])
    snapshot = best["run_dir"] / "train_config_used.yaml"
    return snapshot if snapshot.exists() else None


def ensure_dataset(cfg_path: Path, python: str, workers: int,
                   generate: bool, dry_run: bool) -> tuple[Path, Path]:
    """config 의 data_dir/image_dir 을 준비한다. 이미 있으면 그대로 쓴다."""
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out = cfg.get("output") or {}
    data_dir = ROOT / str(out.get("data_dir", ""))
    image_dir = ROOT / str(out.get("image_dir", ""))

    for label, path, script in (("데이터", data_dir, "generate_data.py"),
                                ("이미지", image_dir, "generate_images.py")):
        if path.is_dir():
            print(f"[new] {label:<4} : {path.name} 이미 있음 — 건너뜀")
            continue
        cmd = [python, script, "--config", cfg_path.relative_to(ROOT).as_posix(),
               "--workers", str(workers)]
        if dry_run:
            print(f"[new] {label:<4} : (dry-run) 생성 예정 — {' '.join(cmd)}")
            continue
        if not generate:
            raise SystemExit(f"{path} 가 없습니다 (--no-generate 로 생성을 껐습니다)")
        print(f"[new] {label:<4} : 생성 — {' '.join(cmd)}")
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        if rc != 0 or not path.is_dir():
            raise SystemExit(f"{script} 실패 (rc={rc})")
    return data_dir, image_dir


def main() -> int:
    argv = sys.argv[1:]
    extras: list[str] = []
    if "--" in argv:
        idx = argv.index("--")
        argv, extras = argv[:idx], argv[idx + 1:]

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validation", "--validations", "--source", dest="validation",
                        type=Path, default=None,
                        help="예전 sweep 폴더 (validations/<TS>_all_dataset_backbone 또는 logs/…)")
    parser.add_argument("source", type=Path, nargs="?", default=None,
                        help="--validation 대신 위치 인자로 줘도 된다")
    parser.add_argument("--config", default="configs/datasets/dataset_v25.yaml",
                        help="학습할 새 데이터셋 yaml (기본 v25)")
    parser.add_argument("--workers", type=int, default=0,
                        help="데이터/이미지 생성 병렬 worker (0=auto)")
    parser.add_argument("--no-generate", action="store_true",
                        help="데이터/이미지가 없어도 만들지 않고 종료")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE,
                        help=f"timm 모델명 (기본 {DEFAULT_BACKBONE}). "
                             f"weights/<이름>.pth 가 있어야 한다")
    parser.add_argument("--scope", choices=["global", "dataset", "cell"], default="global",
                        help="어떤 BKM 조합을 쓸지 (기본 global = 전 cell 평균). "
                             "새 데이터셋에는 dataset-scope BKM 이 없으므로 global 이 맞다")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-sampling", action="store_true",
                        help="전체 데이터 대신 BKM 의 샘플링 값 유지")
    parser.add_argument("--log-dir", default=None,
                        help="run 이름 (기본: new_<config stem>_<backbone short>)")
    parser.add_argument("--log-dir-group", default=None,
                        help="logs/<group>/ 묶음 이름 (기본: new_<config stem>)")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true",
                        help="학습 없이 선택 결과와 명령만 출력")
    args = parser.parse_args(argv)

    source = args.validation or args.source
    if source is None:
        parser.error("--validation <sweep 폴더> 를 주세요")
    if not source.exists():
        raise SystemExit(f"sweep 폴더가 없습니다: {source}")
    logs_root, val_root = resolve_roots(source)
    if not val_root.exists():
        raise SystemExit(f"validations 쪽 폴더를 찾지 못했습니다: {val_root}")

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    if not cfg_path.exists():
        raise SystemExit(f"--config 를 찾지 못했습니다: {cfg_path}")

    weights = ROOT / "weights" / f"{args.backbone}.pth"
    if not weights.exists():
        raise SystemExit(f"가중치가 없습니다: {weights}\n  python download.py 를 먼저 실행하세요")

    cfg_rel = cfg_path.relative_to(ROOT).as_posix()
    print(f"[new] sweep     : {val_root}")
    print(f"[new] dataset   : {cfg_rel}")
    print(f"[new] backbone  : {args.backbone}")
    data_dir, image_dir = ensure_dataset(cfg_path, args.python, args.workers,
                                         not args.no_generate, args.dry_run)

    short = short_backbone(args.backbone)
    bkm = load_bkm_queues(val_root)
    bkm_args, bkm_src = pick_bkm_for_backbone(bkm, args.scope, short)
    print(f"[new] BKM scopes: {', '.join(sorted(bkm)) or '없음'}")

    if bkm_args:
        train_args = dict(bkm_args)
        train_args["--model_name"] = args.backbone
        train_args["--config"] = cfg_rel
        if not args.keep_sampling:
            train_args.update(FULL_DATA_ARGS)
        cmd = [args.python, "train.py"] + args_dict_to_cli(train_args)
        print(f"[new] BKM       : {bkm_src}")
    else:
        snapshot = snapshot_for_backbone(logs_root, short)
        if snapshot is None:
            raise SystemExit(
                f"BKM 조건도 train_config_used.yaml 스냅샷도 없습니다.\n"
                f"  validations: {val_root}\n  logs: {logs_root}")
        print(f"[new] BKM 없음  : 스냅샷 사용 {snapshot}")
        cmd = [args.python, "train.py", "--train_config", str(snapshot),
               "--model_name", args.backbone, "--config", cfg_rel]
        if not args.keep_sampling:
            cmd += args_dict_to_cli(FULL_DATA_ARGS)

    group = args.log_dir_group or f"new_{cfg_path.stem}"
    log_dir = args.log_dir or f"new_{cfg_path.stem}_{short}"
    cmd += ["--log_dir", log_dir, "--log_dir_group", group, "--seed", str(args.seed)]
    cmd += extras

    print(f"[new] 학습 1회 → logs/{group}/")
    print("  + " + " ".join(cmd))
    warn_short_epochs(cmd)
    if args.dry_run:
        print("[new] dry-run — 실행하지 않았습니다")
        return 0

    rc = subprocess.run(cmd, cwd=ROOT).returncode
    run_dir = find_run_dir(group, log_dir)
    print(f"\n{'=' * 60}")
    if run_dir is None:
        print(f"[new] rc={rc}  run 폴더를 찾지 못했습니다")
        return rc or 1
    info_path = run_dir / "best_info.json"
    f1 = None
    if info_path.exists():
        try:
            f1 = json.loads(info_path.read_text(encoding="utf-8")).get("test_f1")
        except json.JSONDecodeError:
            pass
    model_ok = (run_dir / "best_model.pth").exists()
    print(f"[new] rc={rc}  test_f1={f1 if f1 is not None else 'n/a'}  "
          f"best_model.pth={'OK' if model_ok else '없음'}")
    print(f"[new] {run_dir.relative_to(ROOT).as_posix()}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
