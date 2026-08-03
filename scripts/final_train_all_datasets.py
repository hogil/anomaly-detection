#!/usr/bin/env python
"""Final training: best backbone + BKM conditions, full data, every dataset.

예전에 돌려둔 matrix run 폴더 하나만 주면 (logs/ 든 validations/ 든 상관없다)

  1) 데이터셋마다 성적이 가장 좋았던 backbone(best model) 을 고르고
  2) 그 cell 의 BKM(Best Known Method) 조합 조건을 찾아
  3) sample cap 없이 **전체 데이터**로, seed sweep 없이 **1회씩** 학습·평가하고
  4) best_model.pth 를 데이터셋별로 남긴다

  python scripts/final_train_all_datasets.py logs/20260803_070000_all_dataset_backbone
  python scripts/final_train_all_datasets.py validations/20260803_070000_all_dataset_backbone

BKM 조건 우선순위 (--scope 로 변경):
  dataset : consensus_bkm 의 데이터셋 평균 BKM (기본)
  global  : consensus_bkm 의 전체 평균 BKM — 모든 데이터셋에 같은 조건
  cell    : 각 cell 의 05_bkm_combined 조건 (그 데이터셋·백본 전용)
셋 다 없으면 그 cell 최고 성적 run 의 train_config_used.yaml 로 대체한다.

전체 데이터 = --max_per_class 0 --normal_ratio 0 (train.py 에서 0 = cap 없음).
--keep-sampling 을 주면 BKM 에 기록된 샘플링 값을 그대로 쓴다.

train.py 옵션 override 는 `--` 뒤에 붙인다:
  python scripts/final_train_all_datasets.py logs/<matrix> -- --batch_size 256
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.adaptive_experiment_controller import retention_rank  # noqa: E402
from scripts.retrain_from import (  # noqa: E402
    args_dict_to_cli, collect_records, short_backbone, short_dataset,
)

# 전체 데이터 학습을 위해 덮어쓰는 샘플링 cap (train.py 에서 0 = 제한 없음)
FULL_DATA_ARGS = {"--max_per_class": 0, "--normal_ratio": 0}


def resolve_roots(source: Path) -> tuple[Path, Path]:
    """logs/<X> 또는 validations/<X> 어느 쪽을 줘도 (logs_root, val_root) 반환."""
    parts = source.parts
    if "logs" in parts:
        idx = len(parts) - 1 - parts[::-1].index("logs")
        val_root = Path(*parts[:idx], "validations", *parts[idx + 1:])
        return source, val_root
    if "validations" in parts:
        idx = len(parts) - 1 - parts[::-1].index("validations")
        logs_root = Path(*parts[:idx], "logs", *parts[idx + 1:])
        return logs_root, source
    return source, source


def load_bkm_queues(val_root: Path) -> dict[str, dict[tuple[str, str], dict]]:
    """scope -> {(dataset, backbone): args}. scope 는 dataset/global/cell."""
    found: dict[str, dict[tuple[str, str], dict]] = {}
    if not val_root.exists():
        return found

    patterns = (("consensus", "**/bkm_queue.json"),
                ("cell", "**/05_bkm_combined_queue.json"))
    for kind, pattern in patterns:
        for path in sorted(val_root.glob(pattern)):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            scope = str(payload.get("scope") or "cell") if kind == "consensus" else "cell"
            for run in payload.get("runs", []):
                args = run.get("args") if isinstance(run, dict) else None
                if not args:
                    continue
                key = (short_dataset(args.get("--config")),
                       short_backbone(args.get("--model_name")))
                found.setdefault(scope, {}).setdefault(key, dict(args))
    return found


def pick_bkm_args(bkm: dict, scope: str, dataset: str, backbone: str) -> tuple[dict, str]:
    """scope 우선 -> 같은 데이터셋 아무 백본 -> global -> cell 순으로 완화."""
    order = [scope] + [s for s in ("dataset", "global", "cell") if s != scope]
    for name in order:
        table = bkm.get(name)
        if not table:
            continue
        if (dataset, backbone) in table:
            return dict(table[(dataset, backbone)]), name
        same_ds = [v for (d, _b), v in table.items() if d == dataset]
        if same_ds:
            return dict(same_ds[0]), f"{name}(다른 백본 조건 차용)"
        if name == "global" and table:
            return dict(next(iter(table.values()))), "global"
    return {}, ""


def best_backbone_per_dataset(records: list[dict]) -> dict[str, list[dict]]:
    """dataset -> [{short, full, mean_f1, best_run}] mean F1 내림차순.

    full 은 timm 모델명 그대로 (train.py 가 weights/<full>.pth 를 찾으므로 필수),
    short 는 표시·필터용.
    """
    cells: dict[tuple[str, str], dict[str, list[dict]]] = {}
    for record in records:
        key = (short_dataset(record.get("dataset_config")), str(record.get("model_name")))
        cells.setdefault(key, {}).setdefault(str(record["candidate"]), []).append(record)

    per_dataset: dict[str, list[dict]] = {}
    for (dataset, full_model), by_cand in cells.items():
        best_f1, best_run = float("-inf"), None
        for items in by_cand.values():
            f1s = [r["test_f1"] for r in items if r["test_f1"] is not None]
            if not f1s:
                continue
            m = mean(f1s)
            if m > best_f1:
                best_f1, best_run = m, max(items, key=retention_rank)
        if best_run is not None:
            per_dataset.setdefault(dataset, []).append({
                "short": short_backbone(full_model), "full": full_model,
                "mean_f1": best_f1, "best_run": best_run,
            })
    for rows in per_dataset.values():
        rows.sort(key=lambda r: -r["mean_f1"])
    return per_dataset


def dataset_yaml_for(dataset: str, snapshot_cfg: str) -> str | None:
    """데이터셋 stem 으로 이 repo 의 config yaml 을 찾는다."""
    for cand in (f"{dataset}.yaml", f"{dataset}.yml",
                 f"configs/datasets/{dataset}.yaml", snapshot_cfg):
        if cand and (ROOT / cand).exists():
            return cand
    return None


def warn_short_epochs(cmd: list[str]) -> None:
    """best 저장은 ep10 부터 — 그 전에 끝나면 best_model.pth 가 안 생긴다."""
    if "--epochs" not in cmd:
        return
    try:
        last = len(cmd) - 1 - cmd[::-1].index("--epochs")
        epochs = int(cmd[last + 1])
    except (ValueError, IndexError):
        return
    if epochs < 10:
        print(f"  [경고] --epochs {epochs} 는 best 저장 시작(ep10)보다 짧아 "
              f"best_model.pth 가 생성되지 않습니다")


def find_run_dir(group: str, log_dir: str) -> Path | None:
    base = ROOT / "logs" / group
    if not base.exists():
        return None
    hits = [p for p in base.glob(f"*_{log_dir}*") if p.is_dir()]
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


def main() -> int:
    argv = sys.argv[1:]
    extras: list[str] = []
    if "--" in argv:
        idx = argv.index("--")
        argv, extras = argv[:idx], argv[idx + 1:]

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path,
                        help="예전 matrix run 폴더 (logs/<TS>_all_dataset_backbone 또는 "
                             "validations/<TS>_all_dataset_backbone)")
    parser.add_argument("--scope", choices=["dataset", "global", "cell"], default="dataset",
                        help="어떤 BKM 조합을 쓸지 (기본 dataset)")
    parser.add_argument("--datasets", default=None,
                        help="이 데이터셋만 (콤마 구분, 부분일치). 기본: matrix 의 전부")
    parser.add_argument("--backbone", default=None,
                        help="모든 데이터셋에 이 backbone 강제. 기본: 데이터셋별 최고 성적")
    parser.add_argument("--seed", type=int, default=42,
                        help="단일 학습 seed (기본 42). seed sweep 은 하지 않는다")
    parser.add_argument("--keep-sampling", action="store_true",
                        help="전체 데이터로 덮어쓰지 않고 BKM 의 샘플링 값 유지")
    parser.add_argument("--log-dir-group", default=None,
                        help="logs/<group>/ 묶음 이름 (기본: final_<matrix 폴더명>)")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true",
                        help="학습 없이 선택 결과와 명령만 출력")
    args = parser.parse_args(argv)

    if not args.source.exists():
        raise SystemExit(f"source 가 없습니다: {args.source}")
    logs_root, val_root = resolve_roots(args.source)
    if not logs_root.exists():
        raise SystemExit(f"logs 쪽 폴더를 찾지 못했습니다: {logs_root}")

    records = collect_records(logs_root)
    if not records:
        raise SystemExit(f"성적이 기록된 run 이 없습니다: {logs_root}")
    per_dataset = best_backbone_per_dataset(records)
    bkm = load_bkm_queues(val_root)

    print(f"[final] logs: {logs_root}")
    print(f"[final] validations: {val_root}{'' if val_root.exists() else '  (없음 — BKM 은 스냅샷 대체)'}")
    print(f"[final] {len(records)} runs / {len(per_dataset)} datasets / "
          f"BKM scopes: {', '.join(sorted(bkm)) or '없음'}")

    wanted = [d.strip() for d in args.datasets.split(",")] if args.datasets else None
    targets = []
    for dataset, rows in sorted(per_dataset.items()):
        if wanted and not any(w in dataset for w in wanted):
            continue
        if args.backbone:
            row = next((r for r in rows
                        if args.backbone in r["full"] or args.backbone in r["short"]), None)
            if row is None:
                print(f"  [skip] {dataset}: --backbone {args.backbone} 성적 없음")
                continue
        else:
            row = rows[0]
        targets.append((dataset, row))

    if not targets:
        raise SystemExit("학습할 데이터셋이 없습니다 (--datasets/--backbone 확인)")

    group = args.log_dir_group or f"final_{logs_root.name}"
    print(f"[final] 학습 대상 {len(targets)}개 → logs/{group}/")
    print(f"    {'dataset':<24} {'backbone(best)':<26} {'meanF1':>8}  BKM source")

    plan = []
    for dataset, row in targets:
        bkm_args, bkm_src = pick_bkm_args(bkm, args.scope, dataset, row["short"])
        snapshot = row["best_run"]["run_dir"] / "train_config_used.yaml"
        cfg_yaml = dataset_yaml_for(dataset, str(bkm_args.get("--config") or ""))
        print(f"    {dataset:<24} {row['short']:<26} {row['mean_f1']:>8.4f}  "
              f"{bkm_src or 'snapshot(' + row['best_run']['run_dir'].name[:28] + ')'}")
        plan.append((dataset, row, bkm_args, bkm_src, snapshot, cfg_yaml))

    results = []
    for i, (dataset, row, bkm_args, bkm_src, snapshot, cfg_yaml) in enumerate(plan, 1):
        backbone, full_model = row["short"], row["full"]
        log_dir = f"final_{dataset}_{backbone}"
        print(f"\n{'=' * 60}\n[final {i}/{len(plan)}] {dataset} x {full_model}\n{'=' * 60}")

        if bkm_args:
            train_args = dict(bkm_args)
            train_args["--model_name"] = full_model
            if cfg_yaml:
                train_args["--config"] = cfg_yaml
            if not args.keep_sampling:
                train_args.update(FULL_DATA_ARGS)
            cmd = [args.python, "train.py"] + args_dict_to_cli(train_args)
        elif snapshot.exists():
            cmd = [args.python, "train.py", "--train_config", str(snapshot),
                   "--model_name", full_model]
            if cfg_yaml:
                cmd += ["--config", cfg_yaml]
            if not args.keep_sampling:
                cmd += args_dict_to_cli(FULL_DATA_ARGS)
        else:
            print(f"  [skip] {dataset}: BKM 조건도 스냅샷도 없습니다")
            continue

        cmd += ["--log_dir", log_dir, "--log_dir_group", group, "--seed", str(args.seed)]
        cmd += extras
        print("  + " + " ".join(cmd))
        warn_short_epochs(cmd)
        if args.dry_run:
            continue

        rc = subprocess.run(cmd, cwd=ROOT).returncode
        run_dir = find_run_dir(group, log_dir)
        row = {"dataset": dataset, "backbone": backbone, "rc": rc,
               "run_dir": run_dir, "bkm": bkm_src or "snapshot"}
        if run_dir is not None:
            info_path = run_dir / "best_info.json"
            if info_path.exists():
                try:
                    info = json.loads(info_path.read_text(encoding="utf-8"))
                    row.update({"test_f1": info.get("test_f1"),
                                "model": (run_dir / "best_model.pth").exists()})
                except json.JSONDecodeError:
                    pass
        results.append(row)

    if args.dry_run:
        print("\n[final] dry-run — 실행하지 않았습니다")
        return 0

    print(f"\n{'=' * 60}\n[final] 결과 요약\n{'=' * 60}")
    print(f"    {'dataset':<24} {'backbone':<26} {'testF1':>8} {'model':>6}  run_dir")
    for row in results:
        f1 = row.get("test_f1")
        f1_txt = f"{float(f1):.4f}" if f1 is not None else "  n/a "
        model_txt = "OK" if row.get("model") else "없음"
        rel = row["run_dir"].relative_to(ROOT).as_posix() if row.get("run_dir") else "-"
        print(f"    {row['dataset']:<24} {row['backbone']:<26} {f1_txt:>8} "
              f"{model_txt:>6}  {rel}")
    failed = [r for r in results if r["rc"] != 0]
    if failed:
        print(f"[final] 실패 {len(failed)}개: "
              f"{', '.join(r['dataset'] for r in failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
