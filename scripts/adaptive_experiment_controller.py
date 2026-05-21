"""
Minimal adaptive experiment controller for queued train.py runs.

It launches runs sequentially, parses best_info.json/test_history.json for
F1/FN/FP, and stops once the configured target condition is satisfied.

Queue format (JSON):
{
  "defaults": {
    "seeds": [42, 1, 2, 3, 4],
    "args": {"--mode": "binary"}
  },
  "candidates": [
    {
      "name": "adaptive_gc1",
      "tag_template": "adaptive_gc1_s{seed}",
      "args": {"--grad_clip": 1.0}
    }
  ],
  "runs": [
    {
      "tag": "adaptive_gc1_s42",
      "seed": 42,
      "args": {
        "--train_config": "configs/train/winning.yaml",
        "--config": "configs/datasets/v11.yaml",
        "--grad_clip": 1.0,
        "--smooth_window": 3
      }
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
DEFAULT_SUMMARY = ROOT / "validations" / "adaptive_controller_summary.json"
DEFAULT_MARKDOWN = ROOT / "validations" / "adaptive_controller_summary.md"
DEFAULT_LIVE_SUMMARY_SCRIPT = ROOT / "scripts" / "update_live_summary_doc.py"
DEFAULT_STAGE_COMPARISON_SCRIPT = ROOT / "scripts" / "generate_stage_comparison.py"
DEFAULT_MODEL_NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k"
VALIDATED_WEIGHT_PATHS: set[Path] = set()
DDP_LISTEN_ERROR_PATTERNS = (
    "server socket has failed to listen",
    "failed to listen on any local network address",
)


def configure_output_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except Exception:
            pass


def load_queue(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        runs = payload
    else:
        runs = expand_candidates(payload) + payload.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"queue has no runs: {path}")
    for idx, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"queue run #{idx} is not an object")
        if not run.get("tag"):
            raise ValueError(f"queue run #{idx} is missing tag")
        if "seed" not in run:
            raise ValueError(f"queue run #{idx} is missing seed")
    return runs


def merge_args(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        merged.update(override)
        return merged
    if isinstance(base, list) and isinstance(override, list):
        return [*base, *override]
    if base in (None, {}, []):
        return override
    if override in (None, {}, []):
        return base
    raise ValueError("cannot merge args with different non-empty types")


def expand_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = payload.get("defaults", {})
    default_seeds = defaults.get("seeds", [42, 1, 2, 3, 4])
    default_args = defaults.get("args", {})
    runs: list[dict[str, Any]] = []
    for candidate in payload.get("candidates", []):
        if not isinstance(candidate, dict):
            raise ValueError("candidate entry is not an object")
        name = candidate.get("name")
        if not name:
            raise ValueError("candidate is missing name")
        seeds = candidate.get("seeds", default_seeds)
        tag_template = candidate.get("tag_template", f"{name}_s{{seed}}")
        args = merge_args(default_args, candidate.get("args", {}))
        for seed in seeds:
            runs.append({
                "tag": str(tag_template).format(seed=seed, name=name),
                "candidate": str(name),
                "seed": int(seed),
                "args": args,
            })
    return runs


def candidate_name(run: dict[str, Any]) -> str:
    if run.get("candidate"):
        return str(run["candidate"])
    tag = str(run["tag"])
    return re.sub(r"_s\d+$", "", tag)


def load_summary(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            text = path.read_text(encoding="utf-8")
            if text.strip():
                return json.loads(text)
        except json.JSONDecodeError:
            backup = path.with_suffix(f"{path.suffix}.corrupt-{now_iso().replace(':', '')}.bak")
            path.replace(backup)
    return {
        "created_at": now_iso(),
        "updated_at": None,
        "target": {"min": None, "max": None},
        "stop_mode": None,
        "runs": {},
        "skipped_candidates": {},
        "decision": "not_started",
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run_dir_matches_tag(path: Path, tag: str) -> bool:
    name = path.name
    prefix = re.match(r"^\d{6}_\d{6}_(.+)$", name)
    condition = prefix.group(1) if prefix else name
    return condition == tag or condition.startswith(f"{tag}_F")


def find_completed_run_dir(tag: str, log_dir_group: str = "") -> Path | None:
    if log_dir_group:
        root = LOGS_DIR / log_dir_group
        if not root.exists():
            return None
        matches = [
            path
            for path in root.glob(f"*_{tag}*")
            if path.is_dir() and (path / "best_info.json").exists() and run_dir_matches_tag(path, tag)
        ]
        direct = root / tag
        if direct.is_dir() and (direct / "best_info.json").exists():
            matches.append(direct)
        return max(matches, key=lambda path: path.stat().st_mtime) if matches else None

    # Search both logs/ and logs/<group>/ subdirectories so older flat runs and
    # new grouped runs are both visible when no explicit group was requested.
    matches: list[Path] = []
    for pattern in (f"*_{tag}*", f"*/*_{tag}*"):
        matches.extend(
            path
            for path in LOGS_DIR.glob(pattern)
            if path.is_dir() and (path / "best_info.json").exists() and run_dir_matches_tag(path, tag)
        )
    if not matches:
        for direct in (LOGS_DIR / tag, *LOGS_DIR.glob(f"*/{tag}")):
            if direct.is_dir() and (direct / "best_info.json").exists():
                return direct
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def last_metric_event(run_dir: Path, best_info: dict[str, Any]) -> dict[str, Any]:
    history_path = run_dir / "test_history.json"
    history = read_json(history_path)
    if not isinstance(history, list):
        history = best_info.get("test_history", [])
    events = [event for event in history if isinstance(event, dict)]
    metric_events = [
        event
        for event in events
        if event.get("test_f1") is not None and (event.get("fn") is not None or event.get("fp") is not None)
    ]
    if metric_events:
        new_best = [event for event in metric_events if event.get("event") in ("NEW_BEST", "TIE")]
        return (new_best or metric_events)[-1]
    return {}


def fallback_counts(best_info: dict[str, Any], event: dict[str, Any]) -> tuple[int | None, int | None]:
    fn = event.get("fn")
    fp = event.get("fp")
    if fn is not None and fp is not None:
        return int(fn), int(fp)

    metrics = best_info.get("test_metrics", {})
    abnormal = metrics.get("abnormal", {})
    normal = metrics.get("normal", {})

    abn_recall = event.get("test_abn_R", abnormal.get("recall", best_info.get("test_abn_recall")))
    nor_recall = event.get("test_nor_R", normal.get("recall", best_info.get("test_nor_recall")))
    abn_count = abnormal.get("count", abnormal.get("support"))
    nor_count = normal.get("count", normal.get("support"))

    if abn_recall is not None and abn_count is not None:
        fn = round((1.0 - float(abn_recall)) * int(abn_count))
    if nor_recall is not None and nor_count is not None:
        fp = round((1.0 - float(nor_recall)) * int(nor_count))

    return (int(fn) if fn is not None else None, int(fp) if fp is not None else None)


def parse_metrics(run_dir: Path, run: dict[str, Any], target_min: int, target_max: int) -> dict[str, Any]:
    best_info = read_json(run_dir / "best_info.json")
    if not isinstance(best_info, dict):
        raise RuntimeError(f"cannot read best_info.json: {run_dir}")

    event = last_metric_event(run_dir, best_info)
    fn, fp = fallback_counts(best_info, event)
    f1 = event.get("test_f1", best_info.get("test_f1"))
    epoch = event.get("epoch", best_info.get("epoch", best_info.get("best_epoch")))
    in_target = fn is not None and fp is not None and target_min <= fn <= target_max and target_min <= fp <= target_max

    return {
        "tag": run["tag"],
        "candidate": candidate_name(run),
        "seed": int(run["seed"]),
        "run_dir": str(run_dir.relative_to(ROOT)),
        "status": "complete",
        "epoch": epoch,
        "event": event.get("event"),
        "test_f1": float(f1) if f1 is not None else None,
        "fn": fn,
        "fp": fp,
        "target_hit": in_target,
        "completed_at": now_iso(),
    }


def normalize_config_key(raw: object) -> str:
    text = str(raw or "dataset.yaml").replace("\\", "/").strip()
    if not text:
        return "dataset.yaml"
    path = Path(text)
    try:
        resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return text


def read_train_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "train_config_used.yaml"
    if not path.exists():
        return {}
    try:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def retention_rank(record: dict[str, Any]) -> tuple[float, int, int, float]:
    f1 = record.get("test_f1")
    try:
        f1_value = float(f1)
    except (TypeError, ValueError):
        f1_value = float("-inf")
    fn = record.get("fn")
    fp = record.get("fp")
    fn_value = int(fn) if fn is not None else 1_000_000_000
    fp_value = int(fp) if fp is not None else 1_000_000_000
    return (f1_value, -fn_value, -fp_value, float(record.get("mtime", 0.0)))


def retention_record_from_dir(run_dir: Path) -> dict[str, Any] | None:
    best_info = read_json(run_dir / "best_info.json")
    if not isinstance(best_info, dict):
        return None

    event = last_metric_event(run_dir, best_info)
    fn, fp = fallback_counts(best_info, event)
    f1 = event.get("test_f1", best_info.get("test_f1"))
    train_cfg = read_train_config(run_dir)
    config_key = normalize_config_key(train_cfg.get("config"))
    model_name = str(train_cfg.get("model_name") or DEFAULT_MODEL_NAME)
    try:
        rel_dir = run_dir.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        rel_dir = str(run_dir)
    return {
        "run_dir": run_dir,
        "run_dir_key": rel_dir,
        "checkpoint": run_dir / "best_model.pth",
        "dataset_config": config_key,
        "model_name": model_name,
        "group_key": f"{config_key}|{model_name}",
        "test_f1": float(f1) if f1 is not None else None,
        "fn": fn,
        "fp": fp,
        "mtime": run_dir.stat().st_mtime,
    }


def iter_retention_run_dirs(summary: dict[str, Any], scope: str, log_dir_group: str) -> list[Path]:
    if scope == "summary":
        dirs = []
        for row in summary.get("runs", {}).values():
            if row.get("status") != "complete" or not row.get("run_dir"):
                continue
            dirs.append(ROOT / str(row["run_dir"]))
        return dirs

    if scope == "log-group" and log_dir_group:
        roots = [LOGS_DIR / log_dir_group]
    else:
        roots = [LOGS_DIR]

    dirs: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        dirs.extend(path.parent for path in root.rglob("best_info.json"))
    return dirs


def apply_checkpoint_retention(
    summary: dict[str, Any],
    policy: str,
    scope: str,
    log_dir_group: str,
) -> None:
    if policy == "all":
        summary["checkpoint_retention"] = {
            "policy": policy,
            "scope": scope,
            "updated_at": now_iso(),
            "deleted_count": 0,
        }
        return

    records = []
    seen_dirs: set[str] = set()
    for run_dir in iter_retention_run_dirs(summary, scope, log_dir_group):
        try:
            key = run_dir.resolve().as_posix()
        except Exception:
            key = str(run_dir)
        if key in seen_dirs:
            continue
        seen_dirs.add(key)
        record = retention_record_from_dir(run_dir)
        if record is not None:
            records.append(record)

    if not records:
        summary["checkpoint_retention"] = {
            "policy": policy,
            "scope": scope,
            "updated_at": now_iso(),
            "deleted_count": 0,
            "kept": [],
        }
        return

    keep_dirs: set[Path] = set()
    global_best = max(records, key=retention_rank)
    keep_dirs.add(global_best["run_dir"].resolve())

    by_group: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_group.setdefault(str(record["group_key"]), []).append(record)
    group_best: dict[str, dict[str, Any]] = {}
    for group, items in by_group.items():
        winner = max(items, key=retention_rank)
        group_best[group] = winner
        keep_dirs.add(winner["run_dir"].resolve())

    deleted: list[str] = []
    kept: list[str] = []
    for record in records:
        run_dir = record["run_dir"].resolve()
        checkpoint = record["checkpoint"]
        if run_dir in keep_dirs:
            if checkpoint.exists():
                kept.append(record["run_dir_key"])
            continue
        if checkpoint.exists():
            try:
                checkpoint.unlink()
            except OSError as exc:
                print(f"[checkpoint] delete failed: {checkpoint} ({exc})", flush=True)
                continue
            deleted.append(record["run_dir_key"])
            print(f"[checkpoint] deleted non-best checkpoint: {record['run_dir_key']}/best_model.pth", flush=True)

    record_by_dir = {record["run_dir_key"]: record for record in records}
    kept_set = set(kept)
    deleted_set = set(deleted)
    for row in summary.get("runs", {}).values():
        run_dir_key = str(row.get("run_dir", "")).replace("\\", "/")
        record = record_by_dir.get(run_dir_key)
        if record is None:
            continue
        row["checkpoint_group"] = {
            "dataset_config": record["dataset_config"],
            "model_name": record["model_name"],
        }
        row["checkpoint_retained"] = run_dir_key in kept_set
        row["checkpoint_deleted"] = run_dir_key in deleted_set

    summary["checkpoint_retention"] = {
        "policy": policy,
        "scope": scope,
        "updated_at": now_iso(),
        "global_best": global_best["run_dir_key"],
        "group_best": {group: record["run_dir_key"] for group, record in sorted(group_best.items())},
        "kept": sorted(kept),
        "deleted_count": len(deleted),
        "deleted": sorted(deleted),
    }


def flatten_train_args(run: dict[str, Any]) -> list[str]:
    args = run.get("args", {})
    if isinstance(args, list):
        return [str(item) for item in args]
    if not isinstance(args, dict):
        raise ValueError(f"run {run['tag']} args must be an object or list")

    out: list[str] = []
    for key, value in args.items():
        flag = str(key)
        if not flag.startswith("--"):
            flag = f"--{flag}"
        if isinstance(value, bool):
            if value:
                out.append(flag)
            continue
        if value is None:
            continue
        out.extend([flag, str(value)])
    return out


def train_arg_value(run: dict[str, Any], name: str) -> str | None:
    args = run.get("args", {})
    bare = name.removeprefix("--")
    flags = {f"--{bare}", bare}
    if isinstance(args, dict):
        for key, value in args.items():
            if str(key) in flags and value is not None:
                return str(value)
        return None
    if not isinstance(args, list):
        return None

    items = [str(item) for item in args]
    for idx, item in enumerate(items):
        for flag in flags:
            if item == flag and idx + 1 < len(items):
                return items[idx + 1]
            if item.startswith(f"{flag}="):
                return item.split("=", 1)[1]
    return None


def model_name_for_run(run: dict[str, Any]) -> str:
    explicit = train_arg_value(run, "model_name")
    if explicit:
        return explicit

    train_config = train_arg_value(run, "train_config")
    if train_config:
        path = Path(train_config)
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if cfg.get("model_name"):
                return str(cfg["model_name"])

    return DEFAULT_MODEL_NAME


def verify_pretrained_weight(run: dict[str, Any]) -> Path:
    model_name = model_name_for_run(run)
    path = ROOT / "weights" / f"{model_name}.pth"
    if not path.exists():
        raise FileNotFoundError(
            f"required pretrained weight is missing for tag {run['tag']}: {path}. "
            "Run `python download.py` or copy weights/*.pth before launching this queue."
        )
    resolved = path.resolve()
    if resolved not in VALIDATED_WEIGHT_PATHS:
        try:
            import torch

            state = torch.load(path, map_location="cpu")
        except Exception as exc:
            raise RuntimeError(
                f"pretrained weight exists but torch.load failed for tag {run['tag']}: {path}. "
                "The file is not a valid PyTorch state_dict or is corrupted/truncated. "
                "Run `python download.py --force` or replace the file in weights/."
            ) from exc
        if not hasattr(state, "keys") or len(state) == 0:
            raise RuntimeError(
                f"pretrained weight is not a non-empty state_dict for tag {run['tag']}: {path}. "
                "Run `python download.py --force` or replace the file in weights/."
            )
        VALIDATED_WEIGHT_PATHS.add(resolved)
    return path


def find_free_local_port(host: str = "127.0.0.1") -> int:
    """Ask the OS for a currently free local TCP port for one torchrun launch."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def ddp_listen_error_seen(line: str) -> bool:
    lowered = line.lower()
    return any(pattern in lowered for pattern in DDP_LISTEN_ERROR_PATTERNS)


def run_cleanup_sleep_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("AD_TRAIN_RUN_CLEANUP_SLEEP", "5") or "5"))
    except ValueError:
        return 5.0


def build_command(run: dict[str, Any], log_dir_group: str = "") -> list[str]:
    try:
        ddp_nproc = int(os.environ.get("AD_TRAIN_DDP_NPROC", "1") or "1")
    except ValueError:
        ddp_nproc = 1
    if ddp_nproc > 1:
        master_addr = os.environ.get("AD_TRAIN_MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("AD_TRAIN_MASTER_PORT") or str(find_free_local_port(master_addr))
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--node-rank=0",
            f"--nproc-per-node={ddp_nproc}",
            "--master-addr",
            master_addr,
            "--master-port",
            master_port,
            "train.py",
        ]
    else:
        cmd = [sys.executable, "-u", "train.py"]
    cmd.extend(flatten_train_args(run))
    cmd.extend(["--seed", str(run["seed"]), "--log_dir", str(run["tag"])])
    if log_dir_group:
        cmd.extend(["--log_dir_group", log_dir_group])
    return cmd


def _terminate_process_tree(proc: subprocess.Popen, tag: str, reason: str) -> None:
    print(f"[controller] {reason}; stopping torchrun/train process group for {tag}", flush=True)
    if os.name != "nt":
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError as exc:
            print(f"[controller] process-group terminate failed for {tag}: {exc}", flush=True)
            proc.terminate()
    else:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print(f"[controller] terminate timed out; killing torchrun/train process group for {tag}", flush=True)
        if os.name != "nt":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                print(f"[controller] process-group kill failed for {tag}: {exc}", flush=True)
                proc.kill()
        else:
            proc.kill()
        proc.wait(timeout=10)


def _cleanup_after_run(proc: subprocess.Popen, tag: str) -> None:
    if os.name != "nt":
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError as exc:
            print(f"[controller] residual process-group cleanup failed for {tag}: {exc}", flush=True)

    delay = run_cleanup_sleep_seconds()
    if delay > 0:
        print(f"[controller] cleanup pause {delay:g}s before next queued run", flush=True)
        time.sleep(delay)


def launch_run(run: dict[str, Any], dry_run: bool, completion_exit_grace: float, log_dir_group: str = "") -> int:
    try:
        max_init_retries = int(os.environ.get("AD_TRAIN_DDP_INIT_RETRIES", "2") or "2")
    except ValueError:
        max_init_retries = 2

    attempt = 0
    while True:
        cmd = build_command(run, log_dir_group=log_dir_group)
        retry_note = "" if attempt == 0 else f" retry={attempt}/{max_init_retries}"
        print(f"\n=== RUN {run['tag']} seed={run['seed']}{retry_note} ===", flush=True)
        print(" ".join(cmd), flush=True)
        if dry_run:
            return 0
        started = time.time()
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("PYTHONIOENCODING", "utf-8")
        for key in (
            "MASTER_ADDR",
            "MASTER_PORT",
            "WORLD_SIZE",
            "RANK",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_WORLD_SIZE",
            "TORCHELASTIC_MAX_RESTARTS",
            "TORCHELASTIC_RESTART_COUNT",
            "TORCHELASTIC_RUN_ID",
        ):
            env.pop(key, None)
        popen_kwargs: dict[str, Any] = {
            "cwd": ROOT,
            "env": env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
        }
        if os.name != "nt":
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(cmd, **popen_kwargs)
        output_queue: queue.Queue[str] = queue.Queue()

        def read_output() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                output_queue.put(line)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        completion_seen_at: float | None = None
        forced_after_completion = False
        listen_error = False
        returncode: int | None = None
        while True:
            try:
                line = output_queue.get(timeout=0.2)
            except queue.Empty:
                line = None
            if line is not None:
                listen_error = listen_error or ddp_listen_error_seen(line)
                print(line, end="", flush=True)
                if "학습 완료" in line:
                    completion_seen_at = completion_seen_at or time.time()

            returncode = proc.poll()
            if returncode is not None:
                break

            if completion_seen_at is not None and completion_exit_grace >= 0:
                if time.time() - completion_seen_at >= completion_exit_grace:
                    forced_after_completion = True
                    _terminate_process_tree(proc, run["tag"], "completion marker seen")
                    returncode = 0
                    break

        while True:
            try:
                line = output_queue.get_nowait()
            except queue.Empty:
                break
            listen_error = listen_error or ddp_listen_error_seen(line)
            print(line, end="", flush=True)
        elapsed = round(time.time() - started, 1)
        suffix = " forced_after_completion=1" if forced_after_completion else ""
        print(f"=== EXIT {returncode} elapsed={elapsed}s tag={run['tag']}{suffix} ===", flush=True)
        code = int(returncode or 0)
        _cleanup_after_run(proc, run["tag"])
        if code != 0 and listen_error and attempt < max_init_retries:
            attempt += 1
            wait_s = min(30, 5 * attempt)
            print(
                f"[controller] DDP listen failed before training; retrying {run['tag']} "
                f"with a fresh local port after {wait_s}s",
                flush=True,
            )
            time.sleep(wait_s)
            continue
        return code


def target_seeds(summary: dict[str, Any]) -> set[int]:
    seeds: set[int] = set()
    for record in summary.get("runs", {}).values():
        if record.get("target_hit"):
            seeds.add(int(record["seed"]))
    return seeds


def target_seeds_for_candidate(summary: dict[str, Any], candidate: str) -> set[int]:
    seeds: set[int] = set()
    for record in summary.get("runs", {}).values():
        if record.get("candidate") == candidate and record.get("target_hit"):
            seeds.add(int(record["seed"]))
    return seeds


def decide(summary: dict[str, Any], stop_mode: str, required_by_candidate: dict[str, set[int]]) -> str:
    records = list(summary.get("runs", {}).values())
    hits = [record for record in records if record.get("target_hit")]
    if stop_mode == "first-hit" and hits:
        return "stop:first_hit"
    if stop_mode == "all-seeds":
        for candidate, required_seeds in required_by_candidate.items():
            if required_seeds and required_seeds.issubset(target_seeds_for_candidate(summary, candidate)):
                return f"stop:all_seeds_hit:{candidate}"
    return "continue"


def should_skip_candidate(
    summary: dict[str, Any],
    candidate: str,
    min_runs: int,
    target_min: int,
    target_max: int,
) -> str | None:
    if min_runs <= 0:
        return None
    records = [
        r for r in summary.get("runs", {}).values()
        if r.get("candidate") == candidate and r.get("status") == "complete"
    ]
    if len(records) < min_runs or any(r.get("target_hit") for r in records):
        return None
    fns = [r["fn"] for r in records if r.get("fn") is not None]
    fps = [r["fp"] for r in records if r.get("fp") is not None]
    if len(fns) != len(records) or len(fps) != len(records):
        return None
    fn_mu = mean(fns)
    fp_mu = mean(fps)
    if fn_mu < target_min and fp_mu < target_min:
        return f"too_easy_after_{len(records)}_runs:fn_mean={fn_mu:.2f},fp_mean={fp_mu:.2f}"
    if fn_mu > target_max and fp_mu > target_max:
        return f"too_hard_after_{len(records)}_runs:fn_mean={fn_mu:.2f},fp_mean={fp_mu:.2f}"
    return None


def update_aggregates(summary: dict[str, Any]) -> None:
    records = [r for r in summary.get("runs", {}).values() if r.get("status") == "complete"]
    f1s = [r["test_f1"] for r in records if r.get("test_f1") is not None]
    fns = [r["fn"] for r in records if r.get("fn") is not None]
    fps = [r["fp"] for r in records if r.get("fp") is not None]
    summary["aggregates"] = {
        "complete": len(records),
        "target_hits": sum(1 for r in records if r.get("target_hit")),
        "f1_mean": round(mean(f1s), 6) if f1s else None,
        "f1_std": round(pstdev(f1s), 6) if len(f1s) > 1 else 0.0,
        "fn_mean": round(mean(fns), 3) if fns else None,
        "fp_mean": round(mean(fps), 3) if fps else None,
        "target_hit_seeds": sorted(target_seeds(summary)),
    }
    by_candidate: dict[str, dict[str, Any]] = {}
    for candidate in sorted({r.get("candidate", "") for r in records}):
        candidate_records = [r for r in records if r.get("candidate") == candidate]
        if not candidate:
            continue
        cf1s = [r["test_f1"] for r in candidate_records if r.get("test_f1") is not None]
        cfns = [r["fn"] for r in candidate_records if r.get("fn") is not None]
        cfps = [r["fp"] for r in candidate_records if r.get("fp") is not None]
        by_candidate[candidate] = {
            "complete": len(candidate_records),
            "target_hits": sum(1 for r in candidate_records if r.get("target_hit")),
            "target_hit_seeds": sorted(target_seeds_for_candidate(summary, candidate)),
            "f1_mean": round(mean(cf1s), 6) if cf1s else None,
            "f1_std": round(pstdev(cf1s), 6) if len(cf1s) > 1 else 0.0,
            "fn_mean": round(mean(cfns), 3) if cfns else None,
            "fp_mean": round(mean(cfps), 3) if cfps else None,
        }
    summary["aggregates"]["by_candidate"] = by_candidate


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    rows = sorted(summary.get("runs", {}).values(), key=lambda r: (r.get("seed", 0), r.get("tag", "")))
    lines = [
        "# Adaptive Controller Summary",
        "",
        f"- Decision: `{summary.get('decision')}`",
        f"- Target FN/FP: `{summary.get('target', {}).get('min')}` to `{summary.get('target', {}).get('max')}` per seed",
        f"- Stop mode: `{summary.get('stop_mode')}`",
        f"- Checkpoint retention: `{summary.get('checkpoint_retention', {}).get('policy', 'all')}` / `{summary.get('checkpoint_retention', {}).get('scope', 'summary')}`",
        "",
        "## Candidates",
        "",
        "| Candidate | Complete | Hits | Hit Seeds | F1 Mean | FN Mean | FP Mean |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    by_candidate = summary.get("aggregates", {}).get("by_candidate", {})
    for candidate, row in sorted(by_candidate.items()):
        f1 = row.get("f1_mean")
        lines.append(
            "| {candidate} | {complete} | {hits} | {seeds} | {f1} | {fn} | {fp} |".format(
                candidate=candidate,
                complete=row.get("complete", 0),
                hits=row.get("target_hits", 0),
                seeds=",".join(str(x) for x in row.get("target_hit_seeds", [])),
                f1=f"{f1:.4f}" if isinstance(f1, float) else "",
                fn=row.get("fn_mean", ""),
                fp=row.get("fp_mean", ""),
            )
        )
    lines.extend([
        "",
        "## Runs",
        "",
        "| Tag | Seed | F1 | FN | FP | Hit | Epoch | Run Dir |",
        "| --- | ---: | ---: | ---: | ---: | :--: | ---: | --- |",
    ])
    for row in rows:
        f1 = row.get("test_f1")
        lines.append(
            "| {tag} | {seed} | {f1} | {fn} | {fp} | {hit} | {epoch} | {run_dir} |".format(
                tag=row.get("tag"),
                seed=row.get("seed"),
                f1=f"{f1:.4f}" if isinstance(f1, float) else "",
                fn=row.get("fn", ""),
                fp=row.get("fp", ""),
                hit="yes" if row.get("target_hit") else "no",
                epoch=row.get("epoch", ""),
                run_dir=row.get("run_dir", ""),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)


def update_live_summary_doc(enabled: bool) -> None:
    if not enabled:
        return
    if not DEFAULT_LIVE_SUMMARY_SCRIPT.exists():
        return
    try:
        subprocess.run([sys.executable, str(DEFAULT_LIVE_SUMMARY_SCRIPT)], cwd=ROOT, check=False)
    except Exception as exc:
        print(f"[controller] live summary update failed: {exc}", flush=True)


def update_stage_comparison(args: argparse.Namespace) -> None:
    out_md = getattr(args, "stage_comparison_md", None)
    out_plot = getattr(args, "stage_comparison_plot", None)
    if not out_md or not out_plot:
        return
    if not DEFAULT_STAGE_COMPARISON_SCRIPT.exists():
        print(f"[controller] stage comparison script missing: {DEFAULT_STAGE_COMPARISON_SCRIPT}", flush=True)
        return

    cmd = [
        sys.executable,
        str(DEFAULT_STAGE_COMPARISON_SCRIPT),
        "--results",
        str(args.summary),
        "--out-md",
        str(out_md),
        "--out-plot",
        str(out_plot),
        "--title",
        str(args.stage_comparison_title or "Stage comparison"),
    ]
    if args.stage_comparison_baseline:
        cmd.extend(["--baseline", str(args.stage_comparison_baseline)])

    try:
        proc = subprocess.run(cmd, cwd=ROOT, check=False)
    except Exception as exc:
        print(f"[controller] stage comparison update failed: {exc}", flush=True)
        return
    if proc.returncode != 0:
        print(f"[controller] stage comparison update exited {proc.returncode}", flush=True)


def main() -> int:
    configure_output_encoding()
    parser = argparse.ArgumentParser(description="Sequential adaptive train.py controller")
    parser.add_argument("--queue", required=True, type=Path, help="JSON queue file")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--target-min", type=int, default=5)
    parser.add_argument("--target-max", type=int, default=15)
    parser.add_argument("--stop-mode", choices=["first-hit", "all-seeds", "never"], default="all-seeds")
    parser.add_argument(
        "--candidate-min-runs-before-skip",
        type=int,
        default=0,
        help="If >0, skip remaining runs for a candidate after this many complete runs when it is clearly too easy/hard.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching train.py")
    parser.add_argument("--force", action="store_true", help="Run even if a completed tag already exists")
    parser.add_argument(
        "--max-launched",
        type=int,
        default=0,
        help="If >0, return after launching this many new train.py runs so a master loop can inspect results.",
    )
    parser.add_argument(
        "--completion-exit-grace",
        type=float,
        default=30.0,
        help="Seconds to wait after train.py prints completion before terminating a lingering successful process. Negative disables.",
    )
    parser.add_argument("--update-live-summary", action="store_true", help="Refresh docs/summary.md after controller artifact updates.")
    parser.add_argument(
        "--stage-comparison-baseline",
        type=Path,
        default=None,
        help="Optional baseline results JSON used to rebuild a live stage comparison after each summary update.",
    )
    parser.add_argument(
        "--stage-comparison-md",
        type=Path,
        default=None,
        help="Optional markdown output for live stage comparison.",
    )
    parser.add_argument(
        "--stage-comparison-plot",
        type=Path,
        default=None,
        help="Optional plot output for live stage comparison.",
    )
    parser.add_argument(
        "--stage-comparison-title",
        default="Stage comparison",
        help="Title used by the live stage comparison report.",
    )
    parser.add_argument(
        "--log-dir-group",
        type=str,
        default="",
        help="Group all train.py runs under logs/<group>/ instead of logs/. Forwarded to train.py via --log_dir_group.",
    )
    parser.add_argument(
        "--checkpoint-retention",
        choices=["all", "dataset-backbone-best"],
        default="all",
        help=(
            "Checkpoint cleanup policy. dataset-backbone-best keeps best_model.pth "
            "only for the global best run and for each dataset config + backbone group winner."
        ),
    )
    parser.add_argument(
        "--checkpoint-retention-scope",
        choices=["summary", "log-group", "logs"],
        default="summary",
        help="Scope used when selecting checkpoints to keep.",
    )
    args = parser.parse_args()

    runs = load_queue(args.queue)
    required_by_candidate: dict[str, set[int]] = {}
    for run in runs:
        required_by_candidate.setdefault(candidate_name(run), set()).add(int(run["seed"]))
    summary = load_summary(args.summary)
    summary["updated_at"] = now_iso()
    summary["target"] = {"min": args.target_min, "max": args.target_max}
    summary["stop_mode"] = args.stop_mode
    launched = 0

    for run in runs:
        decision = "continue" if args.stop_mode == "never" else decide(summary, args.stop_mode, required_by_candidate)
        if decision != "continue":
            summary["decision"] = decision
            break

        candidate = candidate_name(run)
        skip_reason = summary.get("skipped_candidates", {}).get(candidate)
        if not skip_reason:
            skip_reason = should_skip_candidate(
                summary,
                candidate,
                args.candidate_min_runs_before_skip,
                args.target_min,
                args.target_max,
            )
            if skip_reason:
                summary.setdefault("skipped_candidates", {})[candidate] = skip_reason
        if skip_reason:
            summary["runs"][run["tag"]] = {
                "tag": run["tag"],
                "candidate": candidate,
                "seed": int(run["seed"]),
                "status": "skipped",
                "skip_reason": skip_reason,
                "target_hit": False,
            }
            update_aggregates(summary)
            apply_checkpoint_retention(
                summary, args.checkpoint_retention, args.checkpoint_retention_scope, args.log_dir_group
            )
            summary["decision"] = "continue"
            save_json(args.summary, summary)
            write_markdown(args.markdown, summary)
            update_live_summary_doc(args.update_live_summary)
            update_stage_comparison(args)
            continue

        existing = None if args.force else find_completed_run_dir(run["tag"], args.log_dir_group)
        launched_this_run = False
        if existing is None:
            if not args.dry_run:
                try:
                    weight_path = verify_pretrained_weight(run)
                except (FileNotFoundError, RuntimeError) as exc:
                    print(f"[preflight] {exc}", flush=True)
                    summary["runs"][run["tag"]] = {
                        "tag": run["tag"],
                        "candidate": candidate,
                        "seed": int(run["seed"]),
                        "status": "failed",
                        "failure_stage": "preflight_weights",
                        "error": str(exc),
                        "completed_at": now_iso(),
                        "target_hit": False,
                    }
                    summary["decision"] = f"stop:missing_weight:{run['tag']}"
                    save_json(args.summary, summary)
                    write_markdown(args.markdown, summary)
                    update_live_summary_doc(args.update_live_summary)
                    update_stage_comparison(args)
                    return 2
                print(f"[preflight] weight ok: {weight_path.relative_to(ROOT)}", flush=True)
            code = launch_run(run, args.dry_run, args.completion_exit_grace, log_dir_group=args.log_dir_group)
            launched_this_run = not args.dry_run
            if launched_this_run:
                launched += 1
            if args.dry_run:
                summary["runs"][run["tag"]] = {
                    "tag": run["tag"],
                    "candidate": candidate,
                    "seed": int(run["seed"]),
                    "status": "dry_run",
                    "command": build_command(run, log_dir_group=args.log_dir_group),
                    "target_hit": False,
                }
                continue
            if code != 0:
                existing = find_completed_run_dir(run["tag"], args.log_dir_group)
                if existing is not None:
                    print(
                        f"[controller] train.py exited with code {code}, "
                        f"but completed artifacts exist for {run['tag']}; continuing",
                        flush=True,
                    )
                    summary["runs"][run["tag"]] = parse_metrics(existing, run, args.target_min, args.target_max)
                    summary["runs"][run["tag"]]["exit_code"] = code
                    summary["runs"][run["tag"]]["recovered_from_nonzero_exit"] = True
                    update_aggregates(summary)
                    apply_checkpoint_retention(
                        summary, args.checkpoint_retention, args.checkpoint_retention_scope, args.log_dir_group
                    )
                    summary["decision"] = "continue"
                    save_json(args.summary, summary)
                    write_markdown(args.markdown, summary)
                    update_live_summary_doc(args.update_live_summary)
                    update_stage_comparison(args)
                    if args.max_launched > 0 and launched >= args.max_launched:
                        break
                    continue
                summary["runs"][run["tag"]] = {
                    "tag": run["tag"],
                    "candidate": candidate,
                    "seed": int(run["seed"]),
                    "status": "failed",
                    "exit_code": code,
                    "completed_at": now_iso(),
                    "target_hit": False,
                }
                summary["decision"] = f"stop:run_failed:{run['tag']}"
                save_json(args.summary, summary)
                write_markdown(args.markdown, summary)
                update_live_summary_doc(args.update_live_summary)
                update_stage_comparison(args)
                return code
            existing = find_completed_run_dir(run["tag"], args.log_dir_group)
            if existing is None:
                raise RuntimeError(f"train.py finished but no best_info.json was found for tag {run['tag']}")

        summary["runs"][run["tag"]] = parse_metrics(existing, run, args.target_min, args.target_max)
        update_aggregates(summary)
        apply_checkpoint_retention(
            summary, args.checkpoint_retention, args.checkpoint_retention_scope, args.log_dir_group
        )
        summary["decision"] = "continue"
        save_json(args.summary, summary)
        write_markdown(args.markdown, summary)
        update_live_summary_doc(args.update_live_summary)
        update_stage_comparison(args)
        if args.max_launched > 0 and launched >= args.max_launched:
            break

    if summary.get("decision") in ("not_started", "continue"):
        summary["decision"] = "queue_exhausted"
    update_aggregates(summary)
    apply_checkpoint_retention(
        summary, args.checkpoint_retention, args.checkpoint_retention_scope, args.log_dir_group
    )
    save_json(args.summary, summary)
    write_markdown(args.markdown, summary)
    update_live_summary_doc(args.update_live_summary)
    update_stage_comparison(args)
    print(json.dumps({"decision": summary["decision"], "aggregates": summary.get("aggregates", {})}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
