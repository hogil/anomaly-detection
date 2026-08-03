#!/usr/bin/env python
"""Merge every dataset into one training set and train a single model.

데이터셋별로 따로 학습하는 final_train_all_datasets.py 와 짝이 되는 스크립트.
이쪽은 여러 데이터셋의 scenarios + 렌더된 이미지를 하나로 합쳐서 **모델 하나**를
학습한다 (데이터셋마다 조건이 조금씩 다른 현장 데이터를 한 모델로 커버할 때).

  python scripts/train_combined_datasets.py logs/<TS>_all_dataset_backbone
  python scripts/train_combined_datasets.py logs/<TS>_all_dataset_backbone \
      --datasets dataset.yaml,dataset1_noise_15.yaml

backbone(best model) 과 BKM 조건은 matrix run 폴더에서 가져온다 (기본 --scope global
= 전 cell 평균 BKM). 합친 학습은 seed sweep 없이 1회, sample cap 없이 전체 데이터.

합치는 방식:
  - 각 데이터셋의 scenarios.csv 를 읽어 chart_id 를 `<dataset>__<chart_id>` 로 prefix
    (데이터셋 간 chart_id 충돌 방지). split/class 는 그대로 유지.
  - 이미지는 hardlink 로 images_combined_<tag>/<split>/<class>/<new_id>.png 에 연결
    (링크 실패 시 복사). 원본은 건드리지 않는다.
  - configs/datasets/combined_<tag>.yaml 을 만들어 train.py 에 넘긴다.
  - 이미 만들어져 있으면 재사용하고, --rebuild 를 줘야 다시 링크한다.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.final_train_all_datasets import (  # noqa: E402
    FULL_DATA_ARGS, best_backbone_per_dataset, dataset_yaml_for, find_run_dir,
    load_bkm_queues, resolve_roots, warn_short_epochs,
)
from scripts.retrain_from import args_dict_to_cli, collect_records  # noqa: E402


def pick_bkm_for_backbone(bkm: dict, scope: str, short: str) -> tuple[dict, str]:
    """합친 학습은 데이터셋 키가 없으므로 backbone 기준으로 BKM 을 고른다.

    1) scope 우선순위대로 backbone 이 일치하는 항목
    2) 그래도 없으면 아무 항목 (BKM 축은 대체로 backbone 과 무관) — 차용했다고 표시
    """
    order = [scope] + [s for s in ("global", "dataset", "cell") if s != scope]
    for name in order:
        table = bkm.get(name) or {}
        hits = [v for (_ds, bb), v in table.items() if bb == short]
        if hits:
            return dict(hits[0]), name
    for name in order:
        table = bkm.get(name) or {}
        if table:
            return dict(next(iter(table.values()))), f"{name}(다른 backbone 조건 차용)"
    return {}, ""


def link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(src, dest)          # 같은 볼륨이면 디스크 추가 사용 0
    except OSError:
        shutil.copy2(src, dest)


def source_image(image_dir: Path, row: dict) -> Path | None:
    """train.py ChartImageDataset 과 같은 후보 순서로 원본 이미지를 찾는다."""
    split, cls, chart_id = row["split"], row["class"], row["chart_id"]
    candidates = []
    name = row.get("image_name")
    if name and not pd.isna(name):
        candidates.append(str(name))
    member = row.get("highlighted_member")
    if member and not pd.isna(member):
        candidates.append(f"{chart_id}_{member}.png")
    candidates.append(f"{chart_id}.png")
    for cand in candidates:
        path = image_dir / str(split) / str(cls) / cand
        if path.exists():
            return path
    return None


def build_combined(cfg_paths: list[Path], tag: str, rebuild: bool) -> tuple[Path, int]:
    """합친 config yaml 경로와 총 이미지 수를 반환."""
    data_dir = ROOT / f"data_combined_{tag}"
    image_dir = ROOT / f"images_combined_{tag}"
    out_cfg = ROOT / "configs" / "datasets" / f"combined_{tag}.yaml"
    scen_path = data_dir / "scenarios.csv"

    if scen_path.exists() and out_cfg.exists() and not rebuild:
        n = len(pd.read_csv(scen_path))
        print(f"[combine] 기존 결과 재사용: {scen_path} ({n} rows) — 다시 만들려면 --rebuild")
        return out_cfg, n

    rows: list[dict] = []
    base_cfg = None
    missing_total = 0
    for cfg_path in cfg_paths:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if base_cfg is None:
            base_cfg = cfg
        prefix = cfg_path.stem
        src_data = ROOT / cfg["output"]["data_dir"]
        src_img = ROOT / cfg["output"]["image_dir"]
        src_scen = src_data / "scenarios.csv"
        if not src_scen.exists():
            raise SystemExit(f"scenarios.csv 가 없습니다: {src_scen} (데이터 생성 먼저)")
        if not src_img.exists():
            raise SystemExit(f"이미지 폴더가 없습니다: {src_img} (generate_images.py 먼저)")

        sdf = pd.read_csv(src_scen)
        linked = missing = 0
        for row in sdf.to_dict(orient="records"):
            src = source_image(src_img, row)
            if src is None:
                missing += 1
                continue
            new_id = f"{prefix}__{row['chart_id']}"
            dest = image_dir / str(row["split"]) / str(row["class"]) / f"{new_id}.png"
            link_or_copy(src, dest)
            merged = dict(row)
            merged["chart_id"] = new_id
            merged["image_name"] = f"{new_id}.png"
            merged["source_dataset"] = prefix
            rows.append(merged)
            linked += 1
        missing_total += missing
        print(f"[combine] {prefix:<28} {linked:>6} images"
              + (f"  (원본 없음 {missing}건 건너뜀)" if missing else ""))

    if not rows:
        raise SystemExit("합칠 이미지가 없습니다")
    if missing_total:
        print(f"[combine] 경고: 렌더 이미지가 없어 제외된 scenario {missing_total}건 "
              f"— generate_images.py 로 먼저 렌더하세요")

    data_dir.mkdir(parents=True, exist_ok=True)
    merged_df = pd.DataFrame(rows)
    merged_df.to_csv(scen_path, index=False)

    combined = dict(base_cfg)
    combined["output"] = dict(base_cfg.get("output", {}))
    combined["output"]["data_dir"] = data_dir.name
    combined["output"]["image_dir"] = image_dir.name
    combined["output"]["display_dir"] = f"display_combined_{tag}"
    combined["combined_from"] = [p.name for p in cfg_paths]
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    out_cfg.write_text(yaml.safe_dump(combined, sort_keys=False, allow_unicode=True),
                       encoding="utf-8")

    counts = merged_df.groupby(["split", "class"]).size().unstack(fill_value=0)
    print(f"[combine] scenarios: {scen_path} ({len(merged_df)} rows)")
    print(f"[combine] images   : {image_dir}")
    print(f"[combine] config   : {out_cfg}")
    print(counts.to_string())
    return out_cfg, len(merged_df)


def main() -> int:
    argv = sys.argv[1:]
    extras: list[str] = []
    if "--" in argv:
        idx = argv.index("--")
        argv, extras = argv[:idx], argv[idx + 1:]

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path,
                        help="matrix run 폴더 (logs/<TS>_all_dataset_backbone 또는 validations/…)")
    parser.add_argument("--datasets", default=None,
                        help="합칠 dataset yaml (콤마 구분). 기본: matrix 에 있는 전부")
    parser.add_argument("--scope", choices=["global", "dataset", "cell"], default="global",
                        help="어떤 BKM 조합을 쓸지 (기본 global = 전 cell 평균)")
    parser.add_argument("--backbone", default=None,
                        help="backbone 강제. 기본: 전체 평균 F1 최고")
    parser.add_argument("--tag", default=None,
                        help="합친 산출물 이름 꼬리표 (기본: 데이터셋 수)")
    parser.add_argument("--rebuild", action="store_true",
                        help="이미 합쳐져 있어도 다시 링크")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-sampling", action="store_true",
                        help="전체 데이터 대신 BKM 의 샘플링 값 유지")
    parser.add_argument("--log-dir-group", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--build-only", action="store_true",
                        help="합치기만 하고 학습은 하지 않는다")
    parser.add_argument("--dry-run", action="store_true")
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

    # 합칠 dataset yaml 결정
    cfg_paths: list[Path] = []
    if args.datasets:
        for raw in args.datasets.split(","):
            raw = raw.strip()
            if not raw:
                continue
            path = ROOT / raw if (ROOT / raw).exists() else None
            if path is None:
                found = dataset_yaml_for(Path(raw).stem, "")
                path = ROOT / found if found else None
            if path is None:
                raise SystemExit(f"dataset yaml 을 찾지 못했습니다: {raw}")
            cfg_paths.append(path)
    else:
        for dataset in sorted(per_dataset):
            found = dataset_yaml_for(dataset, "")
            if found:
                cfg_paths.append(ROOT / found)
            else:
                print(f"  [skip] {dataset}: 이 repo 에 yaml 이 없습니다")
    if not cfg_paths:
        raise SystemExit("합칠 dataset yaml 이 없습니다 (--datasets 로 지정하세요)")

    # backbone: 전체 평균 F1 최고 (--backbone 으로 고정 가능)
    totals: dict[str, list[float]] = {}
    fulls: dict[str, str] = {}
    for rows in per_dataset.values():
        for row in rows:
            totals.setdefault(row["short"], []).append(row["mean_f1"])
            fulls[row["short"]] = row["full"]
    if args.backbone:
        short = next((s for s, f in fulls.items()
                      if args.backbone in f or args.backbone in s), None)
        if short is None:
            raise SystemExit(f"--backbone 성적을 찾지 못했습니다: {args.backbone}")
    else:
        short = max(totals, key=lambda s: sum(totals[s]) / len(totals[s]))
    full_model = fulls[short]
    mean_f1 = sum(totals[short]) / len(totals[short])

    tag = args.tag or f"{len(cfg_paths)}ds"
    print(f"[combine] datasets: {', '.join(p.name for p in cfg_paths)}")
    print(f"[combine] backbone: {full_model}  (cell 평균 F1 {mean_f1:.4f})")

    cfg_yaml, n_rows = build_combined(cfg_paths, tag, args.rebuild)
    if args.build_only:
        print("[combine] --build-only — 학습은 하지 않았습니다")
        return 0

    bkm_args, bkm_src = pick_bkm_for_backbone(bkm, args.scope, short)
    if bkm_args:
        train_args = dict(bkm_args)
        train_args["--model_name"] = full_model
        train_args["--config"] = str(cfg_yaml.relative_to(ROOT)).replace("\\", "/")
        if not args.keep_sampling:
            train_args.update(FULL_DATA_ARGS)
        cmd = [args.python, "train.py"] + args_dict_to_cli(train_args)
        print(f"[combine] BKM: {bkm_src}")
    else:
        best = max((r for rows in per_dataset.values() for r in rows
                    if r["short"] == short), key=lambda r: r["mean_f1"])
        snapshot = best["best_run"]["run_dir"] / "train_config_used.yaml"
        if not snapshot.exists():
            raise SystemExit("BKM 조건도 train_config_used.yaml 스냅샷도 없습니다")
        print(f"[combine] BKM 없음 → 스냅샷 사용: {snapshot}")
        cmd = [args.python, "train.py", "--train_config", str(snapshot),
               "--model_name", full_model,
               "--config", str(cfg_yaml.relative_to(ROOT)).replace("\\", "/")]
        if not args.keep_sampling:
            cmd += args_dict_to_cli(FULL_DATA_ARGS)

    group = args.log_dir_group or f"combined_{logs_root.name}"
    log_dir = f"combined_{tag}_{short}"
    cmd += ["--log_dir", log_dir, "--log_dir_group", group, "--seed", str(args.seed)]
    cmd += extras

    print(f"[combine] 학습 1회 ({n_rows} scenarios 합침) → logs/{group}/")
    print("  + " + " ".join(cmd))
    warn_short_epochs(cmd)
    if args.dry_run:
        print("[combine] dry-run — 실행하지 않았습니다")
        return 0

    rc = subprocess.run(cmd, cwd=ROOT).returncode
    run_dir = find_run_dir(group, log_dir)
    if run_dir is not None:
        info = run_dir / "best_info.json"
        f1 = None
        if info.exists():
            try:
                f1 = json.loads(info.read_text(encoding="utf-8")).get("test_f1")
            except json.JSONDecodeError:
                pass
        print(f"[combine] run: {run_dir.relative_to(ROOT).as_posix()}")
        print(f"[combine] test_f1={f1}  model={'OK' if (run_dir / 'best_model.pth').exists() else '없음'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
