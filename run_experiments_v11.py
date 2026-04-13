"""실험 러너 v11 — v11 데이터셋 기준 전체 sweep

Groups (실행 순서 권장):
    sweep  : Group B — Normal ratio × 5 seed (25 runs) ← 먼저 실행, best_n 확정
    perclass: Group B2 — Train max_per_class × 5 seed (50 runs, train split only)
    lr     : Group C — LR sweep × 3 seeds (18 runs, --base_n 필요)
    gc     : Group D — Gradient clipping sweep × 3 seeds (9 runs, --base_n 필요)
    smooth : Group F — Smoothing window ablation × 3 seeds (9 runs, --base_n 필요)
    reg    : Group G — Regularization ablation × 3 seeds (6 runs, --base_n 필요)
    combo  : Paper combo — 선택한 best item들을 동시에 적용 (3 runs, --combo-* 필요)
    color  : Group E — Rendering color 비교 × 3 seeds (9 runs, 이미지 생성 필요)

사용법:
    # Step 1: sweep 먼저 실행 (best_n 확정 목적)
    python run_experiments_v11.py --groups sweep

    # Step 2: best_n 결정 후 C/D/F/G 실행 (예: best_n=2800)
    python run_experiments_v11.py --groups lr gc smooth reg --base_n 2800

    # Step 3: 이미지 생성 후 color 그룹
    #   python generate_images.py --config config_red.yaml
    #   python generate_images.py --config config_rednf.yaml
    python run_experiments_v11.py --groups color --base_n 2800

    # 전체 순차 실행 (base_n=700 기본값)
    python run_experiments_v11.py

    # dry-run (명령만 출력)
    python run_experiments_v11.py --dry-run

    # 결과 요약만 출력
    python run_experiments_v11.py --only-summary

    # H200 서버 (병렬 2GPU)
    python run_experiments_v11.py --server h200 --gpus 2 --groups sweep

절대 규칙:
    * 기존 logs/<run_dir>/ 절대 삭제 금지 — 존재하면 skip
    * 새 실험 = 새 폴더명 (prefix: v11_*)
    * EMA 미사용 (ema_decay=0.0 고정)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import queue
import shutil
import statistics
import subprocess
import sys
import threading
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Windows cp949 콘솔 UTF-8 출력
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
ACTIVE_DATASET_CONFIG = ROOT / "dataset.yaml"
RUN_BACKUP_DIR = ROOT / "configs" / "runs"

# ============================================================================
# 서버 프로파일
# ============================================================================

SERVER_PROFILES = {
    "laptop": {
        "extra_args": [],
        "default_gpus": 1,
        "description": "노트북 (4060 Ti 16GB, fp16, bs=32)",
    },
    "h200": {
        "extra_args": [
            "--precision", "bf16",
            "--compile",
            "--batch_size", "256",
            "--num_workers", "16",
            "--prefetch_factor", "8",
        ],
        "default_gpus": 2,
        "description": "H200 141GB × 2 (bf16+compile+bs256, 32 코어)",
    },
}

# ============================================================================
# Experiment 정의
# ============================================================================

@dataclass
class Exp:
    name: str
    group: str
    args: List[str] = field(default_factory=list)
    note: str = ""
    require_config: Optional[str] = None    # --config 경로 (None=project root dataset.yaml)
    require_image_dir: Optional[str] = None  # 필요한 image_dir 존재 확인

    def cmd(self, server_extra: Optional[List[str]] = None) -> List[str]:
        config_path = self.require_config or ACTIVE_DATASET_CONFIG.name
        base = [
            sys.executable, "-u", "train.py",
            "--log_dir", f"logs/{self.name}",
            "--config", config_path,
        ]
        if server_extra:
            base += list(server_extra)
        return base + self.args

    def log_dir(self) -> Path:
        """실제 완료된 timestamped 폴더를 반환. 없으면 기본 경로."""
        # train.py는 YYMMDD_HHMMSS_{name} 형태의 타임스탬프 폴더에 저장
        candidates = sorted(LOGS.glob(f"*_{self.name}"), reverse=True)
        for d in candidates:
            if (d / "best_info.json").exists():
                return d
        # F/R suffix가 붙은 rename된 폴더도 검색
        candidates2 = sorted(LOGS.glob(f"*_{self.name}_F*"), reverse=True)
        for d in candidates2:
            if (d / "best_info.json").exists():
                return d
        return LOGS / self.name  # fallback

    def is_done(self) -> bool:
        """완료 여부: timestamped 폴더에 best_info.json 존재 확인."""
        for pattern in (f"*_{self.name}", f"*_{self.name}_F*"):
            if any((d / "best_info.json").exists()
                   for d in LOGS.glob(pattern) if d.is_dir()):
                return True
        return False

    def is_ready(self) -> tuple[bool, str]:
        """실행 가능 여부 + 이유 반환."""
        if self.require_config and not Path(self.require_config).exists():
            return False, f"config 없음: {self.require_config}"
        if self.require_image_dir and not Path(self.require_image_dir).exists():
            return False, f"image_dir 없음: {self.require_image_dir}"
        return True, ""


# ============================================================================
# 실험 목록 생성
# ============================================================================

SWEEP_SEEDS = (42, 1, 2, 3, 4)
ABLATION_SEEDS = (42, 1, 2)
PER_CLASS_COUNTS = tuple(range(100, 1001, 100))

# 공통 winning args (EMA 없음)
WINNING_BASE = [
    "--mode", "binary",
    "--epochs", "20",
    "--patience", "5",
    "--smooth_window", "3",
    "--smooth_method", "median",
    "--lr_backbone", "2e-5",
    "--lr_head", "2e-4",
    "--warmup_epochs", "5",
    "--grad_clip", "1.0",
    "--ema_decay", "0.0",
]

LR_VARIANTS = {
    "1e5":   ["--lr_backbone", "1e-5",  "--lr_head", "1e-4",  "--warmup_epochs", "5"],
    "3e5":   ["--lr_backbone", "3e-5",  "--lr_head", "3e-4",  "--warmup_epochs", "5"],
    "5e5":   ["--lr_backbone", "5e-5",  "--lr_head", "5e-4",  "--warmup_epochs", "5"],
    "1e4":   ["--lr_backbone", "1e-4",  "--lr_head", "1e-3",  "--warmup_epochs", "5"],
    "warm3": ["--lr_backbone", "2e-5",  "--lr_head", "2e-4",  "--warmup_epochs", "3"],
    "warm8": ["--lr_backbone", "2e-5",  "--lr_head", "2e-4",  "--warmup_epochs", "8"],
}

GC_VARIANTS = {
    "05": ["--grad_clip", "0.5"],
    "20": ["--grad_clip", "2.0"],
    "50": ["--grad_clip", "5.0"],
}

SMOOTH_VARIANTS = {
    "1raw":  ["--smooth_window", "1", "--smooth_method", "median"],
    "5med":  ["--smooth_window", "5", "--smooth_method", "median"],
    "3mean": ["--smooth_window", "3", "--smooth_method", "mean"],
}

REG_VARIANTS = {
    "ls01": ["--label_smoothing", "0.1"],
    "dp02": ["--stochastic_depth_rate", "0.2"],
}


def build_sweep(base_n: int) -> List[Exp]:
    """Group B — Normal ratio × 5 seed sweep (25 runs)."""
    exps = []
    for n in (700, 1400, 2100, 2800, 3500):
        for seed in SWEEP_SEEDS:
            exps.append(Exp(
                name=f"v11_n{n}_s{seed}",
                group="sweep",
                args=WINNING_BASE + ["--normal_ratio", str(n), "--seed", str(seed)],
                note=f"winning config | n={n} seed={seed}",
            ))
    return exps


def build_perclass(base_n: int) -> List[Exp]:
    """Group B2 — Train per-original-class max_per_class × 5 seed sweep (50 runs).

    train split에서 original class(normal + 5 anomaly type) 각각을 동일 cap으로 제한한다.
    val/test 는 그대로 유지한다.
    """
    exps = []
    for count in PER_CLASS_COUNTS:
        for seed in SWEEP_SEEDS:
            exps.append(Exp(
                name=f"v11_pc{count}_s{seed}",
                group="perclass",
                args=WINNING_BASE + ["--max_per_class", str(count), "--seed", str(seed)],
                note=f"winning config | max_per_class={count} seed={seed}",
            ))
    return exps


def build_lr(base_n: int) -> List[Exp]:
    """Group C — LR sweep × 3 seeds (18 runs).

    Winning baseline lr=2e-5/warmup=5 는 sweep 그룹(B)에 이미 포함되어 있으므로 제외한다.
    """
    base_fixed = [
        "--mode", "binary", "--epochs", "20", "--patience", "5",
        "--smooth_window", "3", "--smooth_method", "median",
        "--grad_clip", "1.0", "--ema_decay", "0.0",
        "--normal_ratio", str(base_n),
    ]
    exps = []
    for tag, lr_args in LR_VARIANTS.items():
        for seed in ABLATION_SEEDS:
            exps.append(Exp(
                name=f"v11_lr{tag}_n{base_n}_s{seed}",
                group="lr",
                args=base_fixed + lr_args + ["--seed", str(seed)],
                note=f"LR={tag} | n={base_n} seed={seed}",
            ))
    return exps


def build_gc(base_n: int) -> List[Exp]:
    """Group D — Gradient clipping × 3 seeds (9 runs).

    Winning baseline gc=1.0 은 sweep 그룹(B)에 이미 포함되어 있으므로 제외한다.
    """
    base_fixed = [
        "--mode", "binary", "--epochs", "20", "--patience", "5",
        "--smooth_window", "3", "--smooth_method", "median",
        "--lr_backbone", "2e-5", "--lr_head", "2e-4", "--warmup_epochs", "5",
        "--ema_decay", "0.0",
        "--normal_ratio", str(base_n),
    ]
    exps = []
    for tag, gc_args in GC_VARIANTS.items():
        for seed in ABLATION_SEEDS:
            exps.append(Exp(
                name=f"v11_gc{tag}_n{base_n}_s{seed}",
                group="gc",
                args=base_fixed + gc_args + ["--seed", str(seed)],
                note=f"grad_clip={gc_args[-1]} | n={base_n} seed={seed}",
            ))
    return exps


def build_smooth(base_n: int) -> List[Exp]:
    """Group F — Smoothing window ablation × 3 seeds (9 runs).

    Winning baseline smooth_window=3/median 은 sweep 그룹(B)에 이미 포함되어 있으므로 제외한다.
    """
    base_fixed = [
        "--mode", "binary", "--epochs", "20", "--patience", "5",
        "--lr_backbone", "2e-5", "--lr_head", "2e-4", "--warmup_epochs", "5",
        "--grad_clip", "1.0", "--ema_decay", "0.0",
        "--normal_ratio", str(base_n),
    ]
    exps = []
    for tag, sw_args in SMOOTH_VARIANTS.items():
        for seed in ABLATION_SEEDS:
            exps.append(Exp(
                name=f"v11_sw{tag}_n{base_n}_s{seed}",
                group="smooth",
                args=base_fixed + sw_args + ["--seed", str(seed)],
                note=f"smooth={tag} | n={base_n} seed={seed}",
            ))
    return exps


def build_reg(base_n: int) -> List[Exp]:
    """Group G — Regularization ablation × 3 seeds (6 runs)."""
    base_fixed = WINNING_BASE + ["--normal_ratio", str(base_n)]
    exps = []
    for tag, reg_args in REG_VARIANTS.items():
        for seed in ABLATION_SEEDS:
            exps.append(Exp(
                name=f"v11_reg{tag}_n{base_n}_s{seed}",
                group="reg",
                args=base_fixed + reg_args + ["--seed", str(seed)],
                note=f"reg={tag} | n={base_n} seed={seed}",
            ))
    return exps


def build_color(base_n: int) -> List[Exp]:
    """Group E — Color rendering 비교 × 3 seeds (9 runs, 마지막).

    사전 작업:
      - config_red.yaml   : target_color=#E53935, image_dir=images_red
      - config_rednf.yaml : target_color=#E53935, show_fleet=false, image_dir=images_rednf
      - python generate_images.py --config config_red.yaml
      - python generate_images.py --config config_rednf.yaml
    """
    color_variants = [
        ("blue",  None,                "images",       "현재 파란색 (winning baseline)"),
        ("red",   "config_red.yaml",   "images_red",   "빨간색 target (fleet 유지)"),
        ("rednf", "config_rednf.yaml", "images_rednf", "빨간색 target + fleet 제거"),
    ]
    base_fixed = WINNING_BASE + ["--normal_ratio", str(base_n)]
    exps = []
    for tag, cfg_path, img_dir, desc in color_variants:
        for seed in ABLATION_SEEDS:
            exps.append(Exp(
                name=f"v11_{tag}_n{base_n}_s{seed}",
                group="color",
                args=base_fixed + ["--seed", str(seed)],
                note=f"{desc} | n={base_n} seed={seed}",
                require_config=cfg_path,
                require_image_dir=img_dir if tag != "blue" else None,
            ))
    return exps


def build_combo(base_n: int, lr_tag: str, gc_tag: str, smooth_tag: str, reg_tag: str) -> List[Exp]:
    """Paper-style combo: best item from each group를 동시에 적용."""
    combo_args = []
    combo_args += LR_VARIANTS[lr_tag]
    combo_args += GC_VARIANTS[gc_tag]
    combo_args += SMOOTH_VARIANTS[smooth_tag]
    combo_args += REG_VARIANTS[reg_tag]

    base_fixed = WINNING_BASE + ["--normal_ratio", str(base_n)]
    exps = []
    for seed in ABLATION_SEEDS:
        exps.append(Exp(
            name=f"v11_combo_lr{lr_tag}_gc{gc_tag}_sw{smooth_tag}_reg{reg_tag}_n{base_n}_s{seed}",
            group="combo",
            args=base_fixed + combo_args + ["--seed", str(seed)],
            note=f"combo lr={lr_tag} gc={gc_tag} smooth={smooth_tag} reg={reg_tag} | n={base_n} seed={seed}",
        ))
    return exps


def build_experiments(base_n: int, groups: List[str], name_prefix: str = "",
                      combo_tags: Optional[dict] = None) -> List[Exp]:
    combo_tags = combo_tags or {}
    builders = {
        "sweep":  lambda: build_sweep(base_n),
        "perclass": lambda: build_perclass(base_n),
        "lr":     lambda: build_lr(base_n),
        "gc":     lambda: build_gc(base_n),
        "smooth": lambda: build_smooth(base_n),
        "reg":    lambda: build_reg(base_n),
        "combo":  lambda: build_combo(
            base_n,
            combo_tags["lr_tag"],
            combo_tags["gc_tag"],
            combo_tags["smooth_tag"],
            combo_tags["reg_tag"],
        ) if {"lr_tag", "gc_tag", "smooth_tag", "reg_tag"} <= set(combo_tags.keys()) else [],
        "color":  lambda: build_color(base_n),
    }
    exps: List[Exp] = []
    for g in groups:
        exps.extend(builders[g]())
    if name_prefix:
        for exp in exps:
            exp.name = f"{name_prefix}_{exp.name}"
    return exps


def snapshot_run_spec(args, exps: List[Exp], num_gpus: int, server_extra: List[str]) -> Path:
    """실행 스펙을 configs/runs/ 에 backup 저장."""
    RUN_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    prefix = args.name_prefix or "noprefix"
    group_tag = "-".join(args.groups)
    stem = f"{ts}_{group_tag}_{prefix}"
    spec_path = RUN_BACKUP_DIR / f"{stem}.yaml"
    dataset_backup_path = RUN_BACKUP_DIR / f"{stem}_dataset.yaml"

    if ACTIVE_DATASET_CONFIG.exists():
        shutil.copy2(ACTIVE_DATASET_CONFIG, dataset_backup_path)

    payload = {
        "timestamp": ts,
        "entrypoint": "run_experiments_v11.py",
        "active_dataset_config": str(ACTIVE_DATASET_CONFIG.relative_to(ROOT)),
        "dataset_backup": str(dataset_backup_path.relative_to(ROOT)) if dataset_backup_path.exists() else None,
        "groups": list(args.groups),
        "base_n": args.base_n,
        "server": args.server,
        "server_extra": list(server_extra),
        "gpus": num_gpus,
        "name_prefix": args.name_prefix,
        "combo_tags": {
            "lr_tag": args.combo_lr_tag,
            "gc_tag": args.combo_gc_tag,
            "smooth_tag": args.combo_smooth_tag,
            "reg_tag": args.combo_reg_tag,
        },
        "dry_run": bool(args.dry_run),
        "force": bool(args.force),
        "experiments": [
            {
                "name": exp.name,
                "group": exp.group,
                "note": exp.note,
                "command": exp.cmd(server_extra=server_extra),
            }
            for exp in exps
        ],
    }
    spec_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return spec_path


# ============================================================================
# Runner
# ============================================================================

_print_lock = threading.Lock()


def _safe_print(*a, **kw):
    with _print_lock:
        print(*a, **kw, flush=True)


def run_one(exp: Exp, force: bool, dry_run: bool,
            server_extra: Optional[List[str]] = None,
            gpu_id: Optional[int] = None) -> dict:
    status = {
        "name": exp.name, "group": exp.group,
        "skipped": False, "failed": False,
        "elapsed_sec": 0.0, "rc": None, "gpu_id": gpu_id,
    }

    ready, reason = exp.is_ready()
    if not ready:
        _safe_print(f"[SKIP-NOTREADY] {exp.name} — {reason}")
        status["skipped"] = True
        status["skip_reason"] = reason
        return status

    if exp.is_done() and not force:
        status["skipped"] = True
        return status

    cmd = exp.cmd(server_extra=server_extra)
    gpu_tag = f"[GPU{gpu_id}] " if gpu_id is not None else ""

    _safe_print()
    _safe_print("=" * 80)
    _safe_print(f"{gpu_tag}[RUN] {exp.name}  ({exp.group})")
    _safe_print(f"      {exp.note}")
    _safe_print("      " + " ".join(cmd))
    _safe_print("=" * 80)

    if dry_run:
        status["skipped"] = True
        return status

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 임시 스트리밍 로그는 파일로만 저장한다.
    # 실제 결과는 train.py 가 만드는 timestamped 폴더에 들어가므로
    # logs/<exp.name>/ 같은 중복 디렉토리는 만들지 않는다.
    temp_log_file = LOGS / f"{exp.name}.run.log"

    t0 = time.time()
    with open(temp_log_file, "ab") as lf:
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=0,   # unbuffered binary — tqdm \r 포함 즉시 flush
        )
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            # 콘솔 출력은 epoch 완료 줄만 (tqdm \r 줄은 스킵 — buffer overflow 방지)
            decoded = line.decode("utf-8", errors="replace")
            if "\r" not in decoded:
                try:
                    _safe_print(gpu_tag + f"[{exp.name}] " + decoded.rstrip())
                except (BrokenPipeError, OSError):
                    pass  # stdout 닫혀도 계속 실행
        rc = proc.wait()

    # 성공 시에는 timestamped 결과 폴더 안으로 run.log 이동.
    # 실패/미완료면 root logs 에 *.run.log 파일로 남긴다.
    final_log_dir = None
    for pattern in (f"*_{exp.name}", f"*_{exp.name}_F*"):
        for d in sorted(LOGS.glob(pattern), reverse=True):
            if d.is_dir() and (d / "best_info.json").exists():
                final_log_dir = d
                break
        if final_log_dir is not None:
            break
    if final_log_dir is not None:
        final_log_file = final_log_dir / "run.log"
        try:
            if final_log_file.exists():
                final_log_file.unlink()
            shutil.move(str(temp_log_file), str(final_log_file))
        except OSError as e:
            _safe_print(f"  ! run.log move failed ({e}); keeping {temp_log_file.name}")

    status["elapsed_sec"] = round(time.time() - t0, 1)
    status["rc"] = rc
    status["failed"] = (rc != 0)
    return status


def run_all(exps: List[Exp], num_gpus: int, force: bool, dry_run: bool,
            server_extra: Optional[List[str]] = None) -> List[dict]:
    if num_gpus <= 1:
        statuses = []
        for i, exp in enumerate(exps, 1):
            _safe_print(f"[{i}/{len(exps)}]")
            s = run_one(exp, force=force, dry_run=dry_run,
                        server_extra=server_extra, gpu_id=0 if num_gpus == 1 else None)
            statuses.append(s)
            if s["skipped"] and exp.is_done():
                _safe_print(f"[SKIP] {exp.name}")
            elif s["failed"]:
                _safe_print(f"[FAIL rc={s['rc']}] {exp.name}")
        return statuses

    gpu_q: queue.Queue[int] = queue.Queue()
    for g in range(num_gpus):
        gpu_q.put(g)

    statuses: List[dict] = []

    def _task(idx: int, exp: Exp) -> dict:
        gid = gpu_q.get()
        try:
            _safe_print(f"[{idx}/{len(exps)}] start GPU{gid}: {exp.name}")
            return run_one(exp, force=force, dry_run=dry_run,
                           server_extra=server_extra, gpu_id=gid)
        finally:
            gpu_q.put(gid)

    with ThreadPoolExecutor(max_workers=num_gpus) as ex:
        futures = {ex.submit(_task, i + 1, e): e for i, e in enumerate(exps)}
        for fut in as_completed(futures):
            exp = futures[fut]
            try:
                s = fut.result()
            except Exception as e:
                s = {"name": exp.name, "group": exp.group,
                     "skipped": False, "failed": True,
                     "elapsed_sec": 0.0, "rc": None, "error": str(e)}
            statuses.append(s)
            if s.get("skipped") and exp.is_done():
                _safe_print(f"[SKIP] {exp.name}")
            elif s.get("failed"):
                _safe_print(f"[FAIL rc={s.get('rc')}] {exp.name}")
            else:
                _safe_print(f"[DONE {s['elapsed_sec']}s] {exp.name}")
    return statuses


# ============================================================================
# Summary & Reporting
# ============================================================================

def _read_best_info(exp: Exp) -> Optional[dict]:
    p = exp.log_dir() / "best_info.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_metrics(bi: dict) -> dict:
    """best_info.json → 핵심 지표 추출."""
    tm = bi.get("test_metrics", {})
    abn = tm.get("abnormal", {})
    nor = tm.get("normal", {})

    abn_r = abn.get("recall", bi.get("test_abn_recall", 0.0))
    nor_r = nor.get("recall", bi.get("test_nor_recall", 0.0))
    f1    = bi.get("test_f1", 0.0)
    ep    = bi.get("best_epoch", bi.get("epoch", "-"))

    abn_sup = abn.get("support", 750)
    nor_sup = nor.get("support", 750)
    fn = abn.get("false_negatives", int(round((1 - abn_r) * abn_sup)))
    fp = nor.get("false_positives", int(round((1 - nor_r) * nor_sup)))

    return {"ep": ep, "f1": f1, "abn_R": abn_r, "nor_R": nor_r, "FN": fn, "FP": fp}


HDR = f"{'name':<34} {'ep':>3} {'f1':>7} {'abn_R':>7} {'nor_R':>7} {'FN':>4} {'FP':>4}"


def _print_row(exp: Exp, m: dict):
    print(
        f"{exp.name:<34} {str(m['ep']):>3} {m['f1']:>7.4f} "
        f"{m['abn_R']:>7.4f} {m['nor_R']:>7.4f} "
        f"{str(m['FN']):>4} {str(m['FP']):>4}"
    )


def print_summary(exps: List[Exp], summary_tag: str = "") -> dict:
    out: dict = {"by_group": {}}

    GROUP_ORDER = ["sweep", "perclass", "lr", "gc", "smooth", "reg", "combo", "color"]
    GROUP_LABEL = {
        "sweep":  "Group B — Normal ratio × Seed sweep",
        "perclass": "Group B2 — Train per-class max_per_class × Seed sweep",
        "lr":     "Group C — LR sweep",
        "gc":     "Group D — Gradient clipping sweep",
        "smooth": "Group F — Smoothing window ablation",
        "reg":    "Group G — Regularization ablation",
        "combo":  "Paper Combo — selected items combined",
        "color":  "Group E — Color rendering 비교",
    }

    print()
    print("#" * 80)
    print("# RESULTS — v11 experiment summary")
    print("#" * 80)

    for group in GROUP_ORDER:
        group_exps = [e for e in exps if e.group == group]
        if not group_exps:
            continue

        print()
        print(f"{'─'*80}")
        print(f"  {GROUP_LABEL.get(group, group)}")
        print(f"{'─'*80}")
        print(HDR)

        rows = []
        for exp in sorted(group_exps, key=lambda e: e.name):
            bi = _read_best_info(exp)
            if bi is None:
                print(f"  {exp.name:<32} (미완료)")
                continue
            m = _extract_metrics(bi)
            _print_row(exp, m)
            rows.append({"name": exp.name, **m})
        out["by_group"][group] = rows

        # sweep 그룹: normal_ratio별 집계
        if group == "sweep" and rows:
            print()
            print("  [sweep 집계] normal_ratio별 mean ± std")
            print(f"  {'n':>5} {'seeds':>5}  {'f1 mean±std':>18}  {'abn_R mean±std':>18}  {'FN mean':>8}  {'FP mean':>8}")
            agg = {}
            for n in (700, 1400, 2100, 2800, 3500):
                xs_f1, xs_abn, xs_nor, xs_fn, xs_fp = [], [], [], [], []
                for r in rows:
                    if f"_n{n}_" in r["name"]:
                        xs_f1.append(r["f1"])
                        xs_abn.append(r["abn_R"])
                        xs_nor.append(r["nor_R"])
                        xs_fn.append(r["FN"])
                        xs_fp.append(r["FP"])
                if not xs_f1:
                    continue
                f1m = statistics.mean(xs_f1)
                f1s = statistics.stdev(xs_f1) if len(xs_f1) > 1 else 0.0
                am  = statistics.mean(xs_abn)
                as_ = statistics.stdev(xs_abn) if len(xs_abn) > 1 else 0.0
                fnm = statistics.mean(xs_fn)
                fpm = statistics.mean(xs_fp)
                marker = " ⭐" if f1m == max(
                    statistics.mean([r["f1"] for r in rows if f"_n{nn}_" in r["name"]])
                    for nn in (700, 1400, 2100, 2800, 3500)
                    if any(f"_n{nn}_" in r["name"] for r in rows)
                ) else ""
                print(f"  {n:>5} {len(xs_f1):>5}  "
                      f"{f1m:.4f} ± {f1s:.4f}    "
                      f"{am:.4f} ± {as_:.4f}  "
                      f"{fnm:>8.1f}  {fpm:>8.1f}{marker}")
                agg[n] = {"n_seeds": len(xs_f1), "f1_mean": f1m, "f1_std": f1s,
                          "abn_R_mean": am, "abn_R_std": as_,
                          "FN_mean": fnm, "FP_mean": fpm}
            out["sweep_aggregate"] = agg

        if group == "perclass" and rows:
            print()
            print("  [perclass 집계] train max_per_class별 mean ± std")
            print(f"  {'count':>7} {'seeds':>5}  {'f1 mean±std':>18}  {'abn_R mean±std':>18}  {'FN mean':>8}  {'FP mean':>8}")
            agg = {}
            best_mean = max(
                statistics.mean([r["f1"] for r in rows if f"_pc{count}_" in r["name"]])
                for count in PER_CLASS_COUNTS
                if any(f"_pc{count}_" in r["name"] for r in rows)
            )
            for count in PER_CLASS_COUNTS:
                xs_f1, xs_abn, xs_nor, xs_fn, xs_fp = [], [], [], [], []
                for r in rows:
                    if f"_pc{count}_" in r["name"]:
                        xs_f1.append(r["f1"])
                        xs_abn.append(r["abn_R"])
                        xs_nor.append(r["nor_R"])
                        xs_fn.append(r["FN"])
                        xs_fp.append(r["FP"])
                if not xs_f1:
                    continue
                f1m = statistics.mean(xs_f1)
                f1s = statistics.stdev(xs_f1) if len(xs_f1) > 1 else 0.0
                am  = statistics.mean(xs_abn)
                as_ = statistics.stdev(xs_abn) if len(xs_abn) > 1 else 0.0
                fnm = statistics.mean(xs_fn)
                fpm = statistics.mean(xs_fp)
                marker = " ⭐" if f1m == best_mean else ""
                print(f"  {count:>7} {len(xs_f1):>5}  "
                      f"{f1m:.4f} ± {f1s:.4f}    "
                      f"{am:.4f} ± {as_:.4f}  "
                      f"{fnm:>8.1f}  {fpm:>8.1f}{marker}")
                agg[count] = {
                    "n_seeds": len(xs_f1),
                    "f1_mean": f1m,
                    "f1_std": f1s,
                    "abn_R_mean": am,
                    "abn_R_std": as_,
                    "FN_mean": fnm,
                    "FP_mean": fpm,
                }
            out["perclass_aggregate"] = agg

    # 전체 집계 (각 그룹 완료된 것 기준)
    all_rows = [r for rows in out["by_group"].values() for r in rows]
    if all_rows:
        best = max(all_rows, key=lambda r: r["f1"])
        print()
        print("=" * 80)
        print(f"  Best overall: {best['name']}")
        print(f"  F1={best['f1']:.4f}  abn_R={best['abn_R']:.4f}  "
              f"nor_R={best['nor_R']:.4f}  FN={best['FN']}  FP={best['FP']}")
        print("=" * 80)
        out["best"] = best

    # JSON 저장
    summary_name = f"v11_experiments_summary_{summary_tag}.json" if summary_tag else "v11_experiments_summary.json"
    summary_path = LOGS / summary_name
    LOGS.mkdir(exist_ok=True)
    summary_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[SAVED] {summary_path}")
    return out


# ============================================================================
# CLI
# ============================================================================

ALL_GROUPS = ["sweep", "perclass", "lr", "gc", "smooth", "reg", "combo", "color"]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--groups", nargs="+", default=["sweep"],
        choices=ALL_GROUPS,
        help="실행 그룹 (default: sweep). 권장 순서: sweep/perclass → lr gc smooth reg → combo → color",
    )
    parser.add_argument(
        "--base_n", type=int, default=700,
        help="lr/gc/smooth/reg/color 그룹에서 사용할 normal_ratio (default: 700). "
             "sweep 완료 후 best_n으로 설정 권장.",
    )
    parser.add_argument(
        "--server", default="laptop", choices=list(SERVER_PROFILES.keys()),
        help="서버 환경 (default: laptop)",
    )
    parser.add_argument(
        "--gpus", type=int, default=0,
        help="병렬 GPU 수 (0=profile default, 1=순차)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="train.py DataLoader num_workers override (예: Windows pagefile 이슈 시 0)",
    )
    parser.add_argument(
        "--prefetch_factor", type=int, default=None,
        help="train.py DataLoader prefetch_factor override",
    )
    parser.add_argument("--dry-run", action="store_true", help="명령만 출력, 실제 학습 없음")
    parser.add_argument("--force", action="store_true", help="완료된 실험도 재실행")
    parser.add_argument("--only-summary", action="store_true", help="기존 결과만 요약")
    parser.add_argument(
        "--name-prefix", type=str, default="",
        help="기존 로그와 분리된 새 실험용 run name prefix",
    )
    parser.add_argument("--combo-lr-tag", type=str, default=None, choices=list(LR_VARIANTS.keys()),
                        help="combo 그룹에서 사용할 lr tag")
    parser.add_argument("--combo-gc-tag", type=str, default=None, choices=list(GC_VARIANTS.keys()),
                        help="combo 그룹에서 사용할 gc tag")
    parser.add_argument("--combo-smooth-tag", type=str, default=None, choices=list(SMOOTH_VARIANTS.keys()),
                        help="combo 그룹에서 사용할 smooth tag")
    parser.add_argument("--combo-reg-tag", type=str, default=None, choices=list(REG_VARIANTS.keys()),
                        help="combo 그룹에서 사용할 reg tag")
    args = parser.parse_args()

    profile = SERVER_PROFILES[args.server]
    server_extra = list(profile["extra_args"])
    if args.num_workers is not None:
        server_extra += ["--num_workers", str(args.num_workers)]
    if args.prefetch_factor is not None:
        server_extra += ["--prefetch_factor", str(args.prefetch_factor)]
    num_gpus = args.gpus if args.gpus > 0 else profile["default_gpus"]

    combo_tags = {
        "lr_tag": args.combo_lr_tag,
        "gc_tag": args.combo_gc_tag,
        "smooth_tag": args.combo_smooth_tag,
        "reg_tag": args.combo_reg_tag,
    }
    exps = build_experiments(
        base_n=args.base_n,
        groups=args.groups,
        name_prefix=args.name_prefix,
        combo_tags={k: v for k, v in combo_tags.items() if v is not None},
    )
    if "combo" in args.groups and not any(e.group == "combo" for e in exps):
        raise SystemExit("combo 그룹은 --combo-lr-tag/--combo-gc-tag/--combo-smooth-tag/--combo-reg-tag 가 모두 필요합니다.")

    done_count = sum(1 for e in exps if e.is_done())
    ready_count = sum(1 for e in exps if e.is_ready()[0] and not e.is_done())
    not_ready  = [(e.name, e.is_ready()[1]) for e in exps if not e.is_ready()[0]]

    print(f"[INFO] server:    {args.server} — {profile['description']}")
    print(f"[INFO] num_gpus:  {num_gpus}")
    print(f"[INFO] groups:    {args.groups}")
    print(f"[INFO] base_n:    {args.base_n}")
    print(f"[INFO] prefix:    {args.name_prefix or '(none)'}")
    print(f"[INFO] total:     {len(exps)} experiments")
    print(f"[INFO] done:      {done_count}")
    print(f"[INFO] pending:   {ready_count}")
    if not_ready:
        print(f"[WARN] not-ready: {len(not_ready)} (config/image 없음)")
        for name, reason in not_ready:
            print(f"         - {name}: {reason}")
    print()

    if args.only_summary:
        # 전체 그룹 로드해서 요약
        all_exps = build_experiments(base_n=args.base_n, groups=ALL_GROUPS, name_prefix=args.name_prefix)
        print_summary(all_exps, summary_tag=args.name_prefix)
        return

    spec_path = snapshot_run_spec(args, exps, num_gpus=num_gpus, server_extra=server_extra)
    print(f"[INFO] run spec:  {spec_path.relative_to(ROOT)}")

    t0 = time.time()
    statuses = run_all(
        exps, num_gpus=num_gpus,
        force=args.force, dry_run=args.dry_run,
        server_extra=server_extra,
    )
    total_sec = round(time.time() - t0, 1)

    print()
    print(f"[DONE] {total_sec/60:.1f} min elapsed")
    print(f"       skipped={sum(1 for s in statuses if s.get('skipped'))} "
          f"failed={sum(1 for s in statuses if s.get('failed'))} "
          f"completed={sum(1 for s in statuses if not s.get('skipped') and not s.get('failed'))}")

    all_exps = build_experiments(base_n=args.base_n, groups=ALL_GROUPS, name_prefix=args.name_prefix)
    print_summary(all_exps, summary_tag=args.name_prefix)


if __name__ == "__main__":
    main()
