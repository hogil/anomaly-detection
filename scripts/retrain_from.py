#!/usr/bin/env python
"""Find the best recipe under a folder and retrain it once, keeping best_model.pth.

서버에서 sweep 한 번 돌리면 logs/<group>/ 이나 validations/<group>/ 아래에 run 폴더가
수십~수백 개 생긴다. 이 스크립트는 **그 상위 폴더 하나만** 주면 안을 전부 뒤져서
가장 성적이 좋은 recipe 를 찾아내고, 그 조건 그대로 train.py 를 1회 실행한다.

  python scripts/retrain_from.py logs/20260504_193121_run_paper_dataset
  python scripts/retrain_from.py validations/20260504_193121_run_paper

recipe 선택 (기본 --select candidate):
  candidate : seed 들의 평균 F1 이 가장 높은 조건 (운 좋은 단일 seed 배제 — 권장)
  run       : 단일 run 중 F1 최고 (동점이면 FN, FP, 최신 순)

조건은 우승 run 폴더의 train_config_used.yaml (CLI override 까지 반영된 최종
스냅샷) 을 그대로 사용한다. 없으면 validations queue/active JSON 의 args 를 쓴다.

best_model.pth 는 train.py 가 새 run 폴더에 항상 저장한다. checkpoint 삭제
(retention) 는 sweep 컨트롤러 전용이라 이 경로에서는 절대 실행되지 않는다.

train.py 옵션 override 는 `--` 뒤에 붙인다 (마지막 값이 이긴다):
  python scripts/retrain_from.py logs/<group> -- --batch_size 96 --num_workers 2
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.adaptive_experiment_controller import (  # noqa: E402
    retention_rank, retention_record_from_dir,
)

# store_true 플래그 — args dict 에서 True 면 플래그만, False 면 생략
BOOL_FLAGS = {
    "--use_amp", "--use_mixup", "--save_strict_only", "--allow_tie_save",
    "--no_fast_exit", "--no_progress", "--filter_nonfinite_loss",
    "--eval_test_every_epoch", "--allow_data_parallel", "--compile",
    "--channels_last",
}

# logs run 폴더명: <yymmdd>_<hhmmss>_<candidate>_s<seed>[_F..._R...]
RUN_NAME_RE = re.compile(
    r"^\d{6}_\d{6}_(?P<cand>.+?)_s(?P<seed>\d+)(?:_F[\d.]+_R[\d.]+)?$")


def candidate_of(run_dir: Path) -> str:
    m = RUN_NAME_RE.match(run_dir.name)
    return m.group("cand") if m else run_dir.name


def collect_records(source: Path) -> list[dict]:
    """source 아래 모든 run 폴더 (best_info.json 보유) 의 성적 레코드."""
    records = []
    seen: set[str] = set()
    for info in source.rglob("best_info.json"):
        run_dir = info.parent
        key = str(run_dir.resolve())
        if key in seen:
            continue
        seen.add(key)
        record = retention_record_from_dir(run_dir)
        if record is None or record.get("test_f1") is None:
            continue
        record["candidate"] = candidate_of(run_dir)
        records.append(record)
    return records


def link_validations_runs(source: Path, records: list[dict]) -> list[dict]:
    """validations 폴더를 준 경우 results JSON 의 run_dir 로 logs run 을 따라간다."""
    if records:
        return records
    linked = []
    seen: set[str] = set()
    for path in sorted(source.rglob("*results*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows = payload.get("runs", {})
        rows = rows.values() if isinstance(rows, dict) else rows
        for row in rows:
            if not isinstance(row, dict) or row.get("status") != "complete":
                continue
            rel = str(row.get("run_dir") or "")
            if not rel:
                continue
            run_dir = ROOT / rel
            key = str(run_dir)
            if key in seen or not run_dir.exists():
                continue
            seen.add(key)
            record = retention_record_from_dir(run_dir)
            if record is None or record.get("test_f1") is None:
                continue
            record["candidate"] = row.get("candidate") or candidate_of(run_dir)
            linked.append(record)
    return linked


def short_dataset(raw: object) -> str:
    return Path(str(raw or "dataset.yaml")).stem


def short_backbone(raw: object) -> str:
    return str(raw or "").split(".")[0]


def rank_candidates(records: list[dict]) -> list[dict]:
    """(dataset x backbone) cell 별 candidate 평균 성적 (F1 desc, FN asc, FP asc).

    5개 dataset matrix 처럼 cell 이 섞인 폴더를 그대로 주면 dataset 이 다른 조건끼리
    비교되므로, cell 을 키에 포함해 같은 데이터셋/백본 안에서만 묶는다.
    """
    groups: dict[tuple, list[dict]] = {}
    for record in records:
        key = (short_dataset(record.get("dataset_config")),
               short_backbone(record.get("model_name")),
               str(record["candidate"]))
        groups.setdefault(key, []).append(record)

    rows = []
    for (dataset, backbone, name), items in groups.items():
        f1s = [r["test_f1"] for r in items if r["test_f1"] is not None]
        fns = [r["fn"] for r in items if r["fn"] is not None]
        fps = [r["fp"] for r in items if r["fp"] is not None]
        rows.append({
            "candidate": name,
            "dataset": dataset,
            "backbone": backbone,
            "n": len(items),
            "mean_f1": mean(f1s) if f1s else float("-inf"),
            "mean_fn": mean(fns) if fns else float("inf"),
            "mean_fp": mean(fps) if fps else float("inf"),
            "best_run": max(items, key=retention_rank),
            "runs": items,
        })
    rows.sort(key=lambda r: (-r["mean_f1"], r["mean_fn"], r["mean_fp"]))
    return rows


def args_dict_to_cli(args_map: dict) -> list[str]:
    cli: list[str] = []
    for flag, value in args_map.items():
        if not str(flag).startswith("--"):
            flag = f"--{flag}"
        if flag in BOOL_FLAGS or isinstance(value, bool):
            if bool(value):
                cli.append(flag)
            continue
        if value is None:
            continue
        cli.extend([flag, str(value)])
    return cli


def queue_args_for(source: Path, candidate: str) -> tuple[dict, int | None]:
    """train_config_used.yaml 이 없을 때 fallback: queue/active JSON 의 args."""
    for pattern in ("*_queue.json", "*_active.json"):
        for path in sorted(source.rglob(pattern)):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for run in payload.get("runs", []):
                if not isinstance(run, dict) or not run.get("args"):
                    continue
                if run.get("candidate") == candidate or run.get("tag") == candidate:
                    return run["args"], run.get("seed")
    return {}, None


def main() -> int:
    argv = sys.argv[1:]
    extras: list[str] = []
    if "--" in argv:
        idx = argv.index("--")
        argv, extras = argv[:idx], argv[idx + 1:]

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path,
                        help="sweep 상위 폴더 (logs/<group> 또는 validations/<group>). "
                             "run 폴더 하나를 직접 줘도 된다")
    parser.add_argument("--select", choices=["candidate", "run"], default="candidate",
                        help="candidate=seed 평균 F1 최고 조건 (기본), run=단일 run F1 최고")
    parser.add_argument("--candidate", default=None,
                        help="자동 선택 대신 이 조건으로 강제 지정")
    parser.add_argument("--dataset", default=None,
                        help="dataset yaml 이름으로 cell 필터 (부분일치, 예: noise_15)")
    parser.add_argument("--backbone", default=None,
                        help="backbone 이름으로 cell 필터 (부분일치, 예: convnextv2_tiny)")
    parser.add_argument("--top", type=int, default=5,
                        help="순위표에 보여줄 개수 (기본 5)")
    parser.add_argument("--min-runs", type=int, default=1,
                        help="이 seed 수 미만인 조건은 후보에서 제외 (기본 1=제외 없음). "
                             "seed 적은 조건이 운으로 1등 하는 것을 막는다")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed override (기본: 우승 run 의 seed)")
    parser.add_argument("--log-dir", default=None,
                        help="새 run 조건명 (기본: retrain_<우승 조건명>)")
    parser.add_argument("--python", default=sys.executable,
                        help="train.py 실행에 쓸 python")
    parser.add_argument("--dry-run", action="store_true",
                        help="train.py 실행 없이 선택 결과와 명령만 출력")
    args = parser.parse_args(argv)

    source = args.source
    if not source.exists():
        raise SystemExit(f"source 가 없습니다: {source}")
    if not source.is_dir():
        raise SystemExit(f"폴더를 지정하세요 (파일이 아니라): {source}")

    records = collect_records(source)
    records = link_validations_runs(source, records)
    if not records:
        raise SystemExit(
            f"성적이 기록된 run 을 찾지 못했습니다: {source}\n"
            f"  best_info.json 을 가진 run 폴더나, run_dir 을 가리키는 "
            f"*results*.json 이 있는 폴더를 지정하세요.")

    ranking = rank_candidates(records)
    if args.dataset:
        ranking = [r for r in ranking if args.dataset in r["dataset"]]
    if args.backbone:
        ranking = [r for r in ranking if args.backbone in r["backbone"]]
    if not ranking:
        raise SystemExit(f"--dataset/--backbone 필터에 맞는 cell 이 없습니다 "
                         f"(dataset={args.dataset} backbone={args.backbone})")

    if args.min_runs > 1:
        kept = [r for r in ranking if r["n"] >= args.min_runs]
        dropped = len(ranking) - len(kept)
        if not kept:
            raise SystemExit(f"--min-runs {args.min_runs} 를 만족하는 조건이 없습니다 "
                             f"(최대 seed 수: {max(r['n'] for r in ranking)})")
        if dropped:
            print(f"[retrain] --min-runs {args.min_runs}: seed 부족한 조건 {dropped}개 제외")
        ranking = kept

    if args.candidate:
        chosen = next((r for r in ranking if r["candidate"] == args.candidate), None)
        if chosen is None:
            raise SystemExit(f"--candidate 를 찾지 못했습니다: {args.candidate}")
    elif args.select == "run":
        # 필터를 통과한 cell 들의 run 중 단일 최고 (F1, -FN, -FP, 최신)
        chosen = max(ranking, key=lambda r: retention_rank(r["best_run"]))
    else:
        chosen = ranking[0]

    cells = {(r["dataset"], r["backbone"]) for r in ranking}
    print(f"[retrain] scanned {len(records)} runs / {len(ranking)} candidates "
          f"/ {len(cells)} cells (dataset x backbone) under {source}")
    if len(cells) > 1:
        print(f"[retrain] cell 이 {len(cells)}개입니다 — dataset 이 다르면 F1 을 직접 비교할 수 없으니 "
              f"--dataset / --backbone 으로 좁혀서 고르는 것을 권장합니다")
    print(f"[retrain] top {min(args.top, len(ranking))} by mean F1:")
    print(f"    {'dataset':<22} {'backbone':<22} {'candidate':<44} "
          f"{'n':>3} {'meanF1':>8} {'meanFN':>7} {'meanFP':>7}")
    for row in ranking[:args.top]:
        mark = " *" if row is chosen or (
            row["candidate"] == chosen["candidate"]
            and row["dataset"] == chosen["dataset"]
            and row["backbone"] == chosen["backbone"]) else "  "
        print(f"  {mark}{row['dataset']:<22} {row['backbone']:<22} {row['candidate']:<44} "
              f"{row['n']:>3} {row['mean_f1']:>8.4f} "
              f"{row['mean_fn']:>7.1f} {row['mean_fp']:>7.1f}")

    max_n = max(r["n"] for r in ranking)
    if chosen["n"] < max_n:
        print(f"[retrain] 주의: 선택된 조건의 seed 수가 {chosen['n']}개로 "
              f"최다({max_n}개)보다 적습니다 — 평균이 흔들릴 수 있으니 "
              f"--min-runs {max_n} 로 다시 확인해 보세요")

    best_run = chosen["best_run"]
    run_dir = best_run["run_dir"]
    print(f"[retrain] picked: {chosen['candidate']}  (select={args.select})")
    print(f"[retrain] recipe from: {run_dir}  "
          f"F1={best_run['test_f1']:.4f} FN={best_run['fn']} FP={best_run['fp']}")

    log_dir = args.log_dir or f"retrain_{chosen['candidate']}"
    tc_yaml = run_dir / "train_config_used.yaml"
    seed = args.seed

    if tc_yaml.exists():
        import yaml
        snapshot = yaml.safe_load(tc_yaml.read_text(encoding="utf-8")) or {}
        cmd = [args.python, "train.py",
               "--train_config", str(tc_yaml), "--log_dir", log_dir]
        if seed is None:
            m = RUN_NAME_RE.match(run_dir.name)
            seed = int(m.group("seed")) if m else snapshot.get("seed")
        data_cfg = str(snapshot.get("config") or "")
        if data_cfg and not (ROOT / data_cfg).exists() and not Path(data_cfg).exists():
            if "--config" not in extras:
                raise SystemExit(
                    f"스냅샷의 data config 가 이 머신에 없습니다: {data_cfg}\n"
                    f"  -- --config dataset.yaml 처럼 override 를 붙여 주세요.")
    else:
        qargs, qseed = queue_args_for(source, str(chosen["candidate"]))
        if not qargs:
            raise SystemExit(
                f"조건을 복원할 수 없습니다 — {run_dir}/train_config_used.yaml 도 없고 "
                f"queue/active JSON 에서 {chosen['candidate']} 도 못 찾았습니다.")
        print(f"[retrain] train_config_used.yaml 없음 → queue args 사용")
        cmd = [args.python, "train.py"] + args_dict_to_cli(qargs) + ["--log_dir", log_dir]
        if seed is None:
            seed = qseed

    if seed is not None:
        cmd += ["--seed", str(seed)]
    cmd += extras  # argparse 는 뒤 값을 쓰므로 extras 가 최종 override

    # best 저장은 ep10 부터 (train.py best_update_start_single) — 그 전에 끝나면 모델이 안 남는다
    if "--epochs" in cmd:
        try:
            last = len(cmd) - 1 - cmd[::-1].index("--epochs")
            epochs = int(cmd[last + 1])
        except (ValueError, IndexError):
            epochs = None
        if epochs is not None and epochs < 10:
            print(f"[retrain] 경고: --epochs {epochs} 는 best 저장 시작(ep10)보다 짧아 "
                  f"best_model.pth 가 생성되지 않습니다")

    print("[retrain] command:")
    print("  " + " ".join(cmd))
    print(f"[retrain] best model 저장 위치: logs/<타임스탬프>_{log_dir}_F<F1>_R<Recall>/best_model.pth")
    if args.dry_run:
        return 0
    return subprocess.run(cmd, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
