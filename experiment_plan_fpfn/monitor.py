"""
실시간 학습 모니터 — 나쁜 run 자동 kill.

Kill 조건:
- val_loss > 0.5 (big spike, 복구 불가 가능성)
- val_f1 < 0.98 at ep >= 15 (수렴 실패)
- history.json 이 5분 이상 갱신 안 됨 (hang)

Usage:
    # 특정 log_dir 감시 (학습 병렬 실행 중 사용)
    python experiment_plan_fpfn/monitor.py --watch logs/v9_phase1_baseline_n2800_s1

    # glob pattern 여러 개 감시
    python experiment_plan_fpfn/monitor.py --watch 'logs/v9_phase1_*'
"""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

KILL_THRESHOLDS = {
    "val_loss_spike": 0.5,       # val_loss > 이면 catastrophic
    "val_f1_min_at_15": 0.98,    # ep >= 15 에서 이 미만이면 kill
    "hang_timeout_sec": 300,     # 5분 동안 history 갱신 없으면 hang
}


def find_training_pid(log_dir: Path):
    """해당 log_dir 를 쓰고 있는 train_tie.py 프로세스 찾기 (Windows/Linux 공통)."""
    try:
        # psutil 이 있으면 사용
        import psutil
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmd = p.info["cmdline"] or []
                if any("train" in c and ".py" in c for c in cmd) and any(str(log_dir) in c for c in cmd):
                    return p.info["pid"]
            except Exception:
                continue
    except ImportError:
        print("  (psutil 없음 — PID 자동 탐지 불가. kill 은 수동으로)")
    return None


def kill_run(log_dir: Path, reason: str):
    pid = find_training_pid(log_dir)
    kill_file = log_dir / "killed.txt"
    kill_file.write_text(f"Killed by monitor: {reason}\npid: {pid}\n", encoding="utf-8")
    print(f"  [KILL] {log_dir.name}: {reason} (pid={pid})")
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def check_run(log_dir: Path) -> str:
    """반환: 'ok' / 'kill:reason' / 'done' / 'waiting'"""
    h_file = log_dir / "history.json"
    if not h_file.exists():
        return "waiting"

    # 완료 여부 — best_info.json 있으면 끝
    if (log_dir / "best_info.json").exists():
        return "done"

    # hang check
    age = time.time() - h_file.stat().st_mtime
    if age > KILL_THRESHOLDS["hang_timeout_sec"]:
        return f"kill:hang ({age:.0f}s no update)"

    try:
        h = json.loads(h_file.read_text(encoding="utf-8"))
    except Exception:
        return "waiting"

    if isinstance(h, list):
        eps = h
    else:
        n = len(next(iter(h.values())))
        eps = [{k: v[i] for k, v in h.items()} for i in range(n)]

    if not eps:
        return "waiting"

    last = eps[-1]
    ep = last.get("epoch", 0)
    val_loss = last.get("val_loss", 0)
    val_f1 = last.get("val_f1", 0)

    # 1) catastrophic spike
    if val_loss > KILL_THRESHOLDS["val_loss_spike"]:
        return f"kill:spike ep{ep} val_loss={val_loss:.4f}"

    # 2) 수렴 실패
    if ep >= 15 and val_f1 < KILL_THRESHOLDS["val_f1_min_at_15"]:
        return f"kill:no_converge ep{ep} val_f1={val_f1:.4f}"

    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", nargs="+", required=True, help="log_dir 경로 or glob pattern")
    ap.add_argument("--interval", type=int, default=30, help="check 주기 (초)")
    args = ap.parse_args()

    # glob 확장
    dirs = []
    for w in args.watch:
        p = ROOT / w
        if "*" in str(p):
            dirs.extend(sorted(ROOT.glob(w)))
        elif p.exists():
            dirs.append(p)
    dirs = [d for d in dirs if d.is_dir()]
    if not dirs:
        print(f"감시 대상 없음: {args.watch}")
        return

    print(f"Monitoring {len(dirs)} log_dirs, interval={args.interval}s")
    for d in dirs:
        print(f"  - {d.name}")
    print()

    killed = set()
    done = set()
    while True:
        active = [d for d in dirs if d.name not in killed and d.name not in done]
        if not active:
            print("All runs finished or killed. exit.")
            return

        for d in active:
            status = check_run(d)
            if status == "done":
                done.add(d.name)
                print(f"  [DONE] {d.name}")
            elif status.startswith("kill:"):
                reason = status[5:]
                kill_run(d, reason)
                killed.add(d.name)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
