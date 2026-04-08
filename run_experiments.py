"""실험 러너 — 한 번에 모두 실행 (v9 dataset)

목적:
    memory/skill에 정리된 winning config 기준으로, 아직 v9 데이터에서 검증되지 않은
    조합들을 한 스크립트로 순차 실행. 기존 결과는 건드리지 않고 skip-if-exists.

사용법:
    # 노트북 (4060 Ti, fp16, sequential)
    python run_experiments.py

    # H200 서버 (bf16 + compile + bs256 + num_workers 16, 2 GPU 병렬)
    python run_experiments.py --server h200 --gpus 2

    # 특정 그룹만
    python run_experiments.py --groups sweep reg

    # dry-run (명령만 출력)
    python run_experiments.py --dry-run

    # 강제 재실행 (기존 폴더가 있어도)
    python run_experiments.py --force

그룹:
    sweep : n=normal_ratio sweep × 3 seeds (v9 sweet spot 재검증, 15 runs)
    reg   : 정규화 ablation at n=2800 s42 (label_smooth / mixup / dropout / focal, 5 runs)
    lr    : LR 민감도 ablation at n=2800 s42 (3 runs)
    mc    : multiclass 보조 모델 (1 run) — 6-class 개별 recall 확인

서버 모드:
    laptop : 노트북 default — fp16 AMP, bs 32, num_workers 4 (4060 Ti 16GB)
    h200   : H200 서버 — bf16 + torch.compile, bs 256, num_workers 16, prefetch 8
             (141GB VRAM, 32 CPU 코어 기준 — 메모리 여유 충분, 속도 최대화)

절대 규칙:
    * 기존 logs/<run_dir>/ 절대 삭제 금지 — 이미 존재하면 skip
    * 새 실험은 새 폴더명 (본 스크립트 prefix: v9x_*, v9reg_*, v9lr_*, v9mc_*)
    * 학습 설정 변경 시 script 내 실험 리스트 업데이트 + 새 버전 tag 사용
"""
from __future__ import annotations

import argparse
import io
import json
import os
import statistics
import subprocess
import sys
import time

# Windows cp949 콘솔에서도 한글/유니코드 출력 가능하게
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"


# ============================================================================
# Server profiles — 서버 환경별 학습 args 자동 주입
# ============================================================================

SERVER_PROFILES = {
    "laptop": {
        # 노트북 default — train.py default와 동일 (변경 없음)
        # bs 32, fp16 AMP, num_workers 4, 4060 Ti 16GB 기준
        "extra_args": [],
        "default_gpus": 1,
        "description": "노트북 (4060 Ti 16GB, fp16, bs 32)",
    },
    "h200": {
        # H200 141GB × 2장, 32 CPU, 384GB RAM 폐쇄망 서버
        # bf16 (overflow 없음, scaler 불필요) + torch.compile (20~50% 가속)
        # bs 256 (28M params 모델 / 141GB VRAM 여유 충분)
        # num_workers 16 (32 코어의 절반, 2 process 동시 실행 고려)
        # prefetch_factor 8 (큰 batch + 빠른 GPU)
        "extra_args": [
            "--precision", "bf16",
            "--compile",
            "--batch_size", "256",
            "--num_workers", "16",
            "--prefetch_factor", "8",
        ],
        "default_gpus": 2,
        "description": "H200 141GB × 2 (bf16 + compile + bs 256, 32 코어)",
    },
}


# ============================================================================
# Experiment definition
# ============================================================================

@dataclass
class Exp:
    name: str                    # log_dir basename (== unique id)
    group: str                   # "sweep" | "reg" | "lr" | "mc"
    args: List[str] = field(default_factory=list)  # extra train.py args
    note: str = ""

    def cmd(self, server_extra: Optional[List[str]] = None) -> List[str]:
        base = [
            sys.executable, "-u", "train.py",  # -u: unbuffered (real-time tqdm in subprocess)
            "--log_dir", f"logs/{self.name}",
        ]
        # 순서: server profile 먼저 → exp args 마지막 (exp가 server를 override 가능)
        if server_extra:
            base = base + list(server_extra)
        return base + self.args

    def log_dir(self) -> Path:
        return LOGS / self.name

    def is_done(self) -> bool:
        return (self.log_dir() / "best_info.json").exists()


def build_experiments() -> List[Exp]:
    exps: List[Exp] = []

    # ------------------------------------------------------------------------
    # Group sweep — v9 normal_ratio × multi-seed (sweet spot 재검증)
    # ------------------------------------------------------------------------
    # v8 결과: n=2800 sweet spot (F1 0.9992 ± 0.0007, 25 trials)
    # v9는 noise 25% 강화, sparse region 증가 → 동일 sweet spot 유지되는지 확인
    # seeds 3개만 사용 (5 seeds는 v8에서 이미 패턴 확정, 확인 목적으로 3 충분)
    for n in (700, 1400, 2100, 2800, 3500):
        for seed in (1, 2, 42):
            exps.append(Exp(
                name=f"v9x_n{n}_s{seed}",
                group="sweep",
                args=[
                    "--mode", "binary",
                    "--normal_ratio", str(n),
                    "--seed", str(seed),
                ],
                note=f"winning config, n={n}, seed={seed}",
            ))

    # ------------------------------------------------------------------------
    # Group reg — regularization ablation at n=2800 s42
    # ------------------------------------------------------------------------
    # memory/project_research_overfitting.md 의 권장 기법들 검증
    # baseline (winning)은 sweep 그룹의 v9x_n2800_s42 재사용
    reg_base = ["--mode", "binary", "--normal_ratio", "2800", "--seed", "42"]

    exps.append(Exp(
        name="v9reg_ls01_n2800_s42",
        group="reg",
        args=reg_base + ["--label_smoothing", "0.1"],
        note="label_smoothing 0.1 (Szegedy 2016, Müller 2019)",
    ))
    exps.append(Exp(
        name="v9reg_mix02_n2800_s42",
        group="reg",
        args=reg_base + ["--use_mixup", "--mixup_alpha", "0.2"],
        note="mixup α=0.2 (Zhang 2018) — overlay 이미지 주의",
    ))
    exps.append(Exp(
        name="v9reg_drop02_n2800_s42",
        group="reg",
        args=reg_base + ["--dropout", "0.2"],
        note="dropout 0.2 (v8_init에서는 0.0 best, v9 재확인)",
    ))
    exps.append(Exp(
        name="v9reg_fg20_n2800_s42",
        group="reg",
        args=reg_base + ["--focal_gamma", "2.0"],
        note="focal_gamma 2.0 — v8_init에서는 CE best, 재확인",
    ))
    exps.append(Exp(
        name="v9reg_wd05_n2800_s42",
        group="reg",
        args=reg_base + ["--weight_decay", "0.05"],
        note="weight_decay 0.05 (현재 0.01) — overfitting 완화",
    ))

    # ------------------------------------------------------------------------
    # Group lr — LR sensitivity at n=2800 s42
    # ------------------------------------------------------------------------
    lr_base = ["--mode", "binary", "--normal_ratio", "2800", "--seed", "42"]

    exps.append(Exp(
        name="v9lr_bb3e5_n2800_s42",
        group="lr",
        args=lr_base + ["--lr_backbone", "3e-5", "--lr_head", "3e-4"],
        note="lr_backbone 3e-5 (더 보수적)",
    ))
    exps.append(Exp(
        name="v9lr_bb1e4_n2800_s42",
        group="lr",
        args=lr_base + ["--lr_backbone", "1e-4", "--lr_head", "1e-3"],
        note="lr_backbone 1e-4 (이전 default, collapse risk 재확인)",
    ))
    exps.append(Exp(
        name="v9lr_warm8_n2800_s42",
        group="lr",
        args=lr_base + ["--warmup_epochs", "8"],
        note="warmup 8 (현재 5)",
    ))

    # ------------------------------------------------------------------------
    # Group mc — multiclass 보조 (6-class 개별 recall 확인용)
    # ------------------------------------------------------------------------
    exps.append(Exp(
        name="v9mc_n2800_s42",
        group="mc",
        args=[
            "--mode", "multiclass",
            "--normal_ratio", "2800",
            "--seed", "42",
        ],
        note="6-class 학습 — abnormal 개별 recall 확인 (binary 주, mc 보조)",
    ))

    return exps


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
    """단일 실험 실행. done이면 skip. 반환: status dict.

    gpu_id가 주어지면 CUDA_VISIBLE_DEVICES 환경변수로 격리.
    """
    status = {
        "name": exp.name,
        "group": exp.group,
        "skipped": False,
        "failed": False,
        "elapsed_sec": 0.0,
        "rc": None,
        "gpu_id": gpu_id,
    }

    if exp.is_done() and not force:
        status["skipped"] = True
        return status

    # 절대 규칙: 기존 폴더 삭제 금지. force여도 train.py가 이어쓰기.
    exp.log_dir().mkdir(parents=True, exist_ok=True)

    cmd = exp.cmd(server_extra=server_extra)
    gpu_tag = f"[GPU {gpu_id}] " if gpu_id is not None else ""

    _safe_print()
    _safe_print("=" * 78)
    _safe_print(f"{gpu_tag}[RUN] {exp.name}  ({exp.group})")
    _safe_print(f"      {exp.note}")
    _safe_print("      " + " ".join(cmd))
    _safe_print("=" * 78)

    if dry_run:
        status["skipped"] = True
        return status

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 병렬 실행 시 process별 로그를 파일로도 저장 (출력 섞임 방지)
    log_file = exp.log_dir() / "run.log"
    t0 = time.time()
    with open(log_file, "ab") as lf:
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1,
        )
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            try:
                _safe_print(f"{gpu_tag}{exp.name}: " + line.decode("utf-8", errors="replace").rstrip())
            except Exception:
                pass
        rc = proc.wait()
    status["elapsed_sec"] = round(time.time() - t0, 1)
    status["rc"] = rc
    status["failed"] = (rc != 0)
    return status


def run_parallel(exps: List[Exp], num_gpus: int, force: bool, dry_run: bool,
                 server_extra: Optional[List[str]] = None) -> List[dict]:
    """num_gpus개 GPU에 분산 실행. GPU queue 사용 — 자유로운 GPU에 다음 작업 즉시 할당."""
    if num_gpus <= 1:
        # 순차 실행 — GPU 0 fix
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

    # 병렬 실행 — GPU queue
    gpu_q: queue.Queue[int] = queue.Queue()
    for g in range(num_gpus):
        gpu_q.put(g)

    statuses: List[dict] = []

    def _task(idx: int, exp: Exp) -> dict:
        gpu_id = gpu_q.get()
        try:
            _safe_print(f"[{idx}/{len(exps)}] start on GPU {gpu_id}: {exp.name}")
            return run_one(exp, force=force, dry_run=dry_run,
                           server_extra=server_extra, gpu_id=gpu_id)
        finally:
            gpu_q.put(gpu_id)

    with ThreadPoolExecutor(max_workers=num_gpus) as ex:
        futures = {ex.submit(_task, i + 1, e): e for i, e in enumerate(exps)}
        for fut in as_completed(futures):
            try:
                s = fut.result()
            except Exception as e:
                exp = futures[fut]
                s = {"name": exp.name, "group": exp.group, "skipped": False,
                     "failed": True, "elapsed_sec": 0.0, "rc": None, "error": str(e)}
            statuses.append(s)
            if s.get("skipped") and futures[fut].is_done():
                _safe_print(f"[SKIP] {futures[fut].name}")
            elif s.get("failed"):
                _safe_print(f"[FAIL rc={s.get('rc')}] {futures[fut].name}")
            else:
                _safe_print(f"[DONE {s['elapsed_sec']}s] {futures[fut].name}")
    return statuses


# ============================================================================
# Reporting
# ============================================================================

def _read_best_info(exp: Exp) -> dict | None:
    p = exp.log_dir() / "best_info.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _row_binary(bi: dict) -> dict:
    tm = bi.get("test_metrics", {})
    abn = tm.get("abnormal", {})
    nor = tm.get("normal", {})
    return {
        "epoch": bi.get("epoch", "-"),
        "test_f1": bi.get("test_f1", 0.0),
        "abn_R": abn.get("recall", 0.0),
        "nor_R": nor.get("recall", 0.0),
        "FN": abn.get("false_negatives",
                      int(round((1 - abn.get("recall", 0.0)) * abn.get("support", 0)))),
        "FP": nor.get("false_positives",
                      int(round((1 - nor.get("recall", 0.0)) * nor.get("support", 0)))),
    }


def print_summary(exps: List[Exp]) -> dict:
    """그룹별 결과 요약 출력 + dict 반환 (JSON 저장용)."""
    out = {"by_group": {}, "aggregate_sweep": {}}

    # Per-experiment table
    print()
    print("#" * 78)
    print("# RESULTS — per experiment")
    print("#" * 78)
    header = f"{'name':<28} {'group':<6} {'ep':>3} {'f1':>7} {'abn_R':>7} {'nor_R':>7} {'FN':>4} {'FP':>4}"
    for group in ("sweep", "reg", "lr", "mc"):
        group_exps = [e for e in exps if e.group == group]
        if not group_exps:
            continue
        print()
        print(f"-- {group} " + "-" * (72 - len(group)))
        print(header)
        rows = []
        for exp in group_exps:
            bi = _read_best_info(exp)
            if bi is None:
                print(f"{exp.name:<28} {exp.group:<6} {'(missing)':>48}")
                continue
            r = _row_binary(bi) if exp.group != "mc" else {
                "epoch": bi.get("epoch", "-"),
                "test_f1": bi.get("test_f1", 0.0),
                "abn_R": "-",
                "nor_R": "-",
                "FN": "-",
                "FP": "-",
            }
            rows.append({"name": exp.name, **r})
            abn_cell = r["abn_R"] if isinstance(r["abn_R"], str) else f"{r['abn_R']:.4f}"
            nor_cell = r["nor_R"] if isinstance(r["nor_R"], str) else f"{r['nor_R']:.4f}"
            print(
                f"{exp.name:<28} {exp.group:<6} "
                f"{str(r['epoch']):>3} {r['test_f1']:>7.4f} "
                f"{abn_cell:>7} {nor_cell:>7} "
                f"{str(r['FN']):>4} {str(r['FP']):>4}"
            )
        out["by_group"][group] = rows

    # Aggregate sweep: per-n mean ± std over seeds
    sweep_exps = [e for e in exps if e.group == "sweep"]
    if sweep_exps:
        print()
        print("-- sweep aggregate (per normal_ratio) ---------------------------------------")
        print(f"{'n':>5} {'seeds':>6}  {'f1 mean':>8} {'f1 std':>8}  {'abn_R mean':>11} {'abn_R std':>11}  {'nor_R mean':>11}")
        agg = {}
        for n in sorted({int(e.name.split("_")[1][1:]) for e in sweep_exps}):
            xs_f1, xs_abn, xs_nor = [], [], []
            for exp in sweep_exps:
                if f"_n{n}_" not in exp.name:
                    continue
                bi = _read_best_info(exp)
                if bi is None:
                    continue
                r = _row_binary(bi)
                xs_f1.append(r["test_f1"])
                xs_abn.append(r["abn_R"])
                xs_nor.append(r["nor_R"])
            if not xs_f1:
                continue
            agg[n] = {
                "n_seeds": len(xs_f1),
                "f1_mean": statistics.mean(xs_f1),
                "f1_std": statistics.stdev(xs_f1) if len(xs_f1) > 1 else 0.0,
                "abn_R_mean": statistics.mean(xs_abn),
                "abn_R_std": statistics.stdev(xs_abn) if len(xs_abn) > 1 else 0.0,
                "nor_R_mean": statistics.mean(xs_nor),
            }
            a = agg[n]
            print(f"{n:>5} {a['n_seeds']:>6}  "
                  f"{a['f1_mean']:>8.4f} {a['f1_std']:>8.4f}  "
                  f"{a['abn_R_mean']:>11.4f} {a['abn_R_std']:>11.4f}  "
                  f"{a['nor_R_mean']:>11.4f}")
        out["aggregate_sweep"] = agg

    # Write JSON
    summary_path = LOGS / "experiments_summary.json"
    LOGS.mkdir(exist_ok=True)
    summary_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    print(f"[SAVED] {summary_path}")
    return out


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", nargs="+",
                        default=["sweep", "reg", "lr", "mc"],
                        choices=["sweep", "reg", "lr", "mc"],
                        help="실행할 그룹 (default: all)")
    parser.add_argument("--server", type=str, default="laptop",
                        choices=list(SERVER_PROFILES.keys()),
                        help="서버 환경 (default: laptop)")
    parser.add_argument("--gpus", type=int, default=0,
                        help="병렬 GPU 수 (0=server profile default, 1=순차)")
    parser.add_argument("--dry-run", action="store_true", help="명령만 출력, 학습 안 함")
    parser.add_argument("--force", action="store_true", help="done 폴더도 재실행")
    parser.add_argument("--only-summary", action="store_true", help="기존 결과만 요약")
    args = parser.parse_args()

    profile = SERVER_PROFILES[args.server]
    server_extra = profile["extra_args"]
    num_gpus = args.gpus if args.gpus > 0 else profile["default_gpus"]

    exps = [e for e in build_experiments() if e.group in args.groups]

    print(f"[INFO] server profile:   {args.server} — {profile['description']}")
    print(f"[INFO] num_gpus:         {num_gpus}")
    print(f"[INFO] server args:      {' '.join(server_extra) if server_extra else '(none)'}")
    print(f"[INFO] selected groups:  {args.groups}")
    print(f"[INFO] total experiments: {len(exps)}")

    done = sum(1 for e in exps if e.is_done())
    print(f"[INFO] already done:     {done}")
    print(f"[INFO] pending:          {len(exps) - done}")
    print()

    if args.only_summary:
        print_summary(exps)
        return

    # Run (parallel or sequential per num_gpus)
    t_total = time.time()
    statuses = run_parallel(exps, num_gpus=num_gpus,
                            force=args.force, dry_run=args.dry_run,
                            server_extra=server_extra)

    total_sec = round(time.time() - t_total, 1)
    print()
    print(f"[DONE] all runs finished in {total_sec/60:.1f} min")
    print(f"[DONE] skipped: {sum(1 for s in statuses if s.get('skipped'))} / "
          f"failed: {sum(1 for s in statuses if s.get('failed'))}")

    # Summary
    print_summary(exps)


if __name__ == "__main__":
    main()
