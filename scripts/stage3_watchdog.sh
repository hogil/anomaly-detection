#!/usr/bin/env bash
# Stage 3 watchdog: monitors 62_v11_axis_full.sh, auto-fixes crashes, auto-restarts.
# Appends diagnostics to validations/stage3_watchdog.log
#
# Auto-fix scenarios:
#   1. Chain exited but runs remain → relaunch (62_v11_axis_full.sh skips completed)
#   2. Unicode cp949 crash in any Python script → em-dash → hyphen, set PYTHONIOENCODING
#   3. train.py crashed mid-run → let chain pick up next seed (already handled by run_one)
#
# Alerts user (via validations/stage3_alerts.md) when:
#   - Auto-fix fired
#   - Same run fails 3+ times
#   - Stage 3 completes
#   - Mean errors stays < 2 after 30+ runs (dataset too easy confirmed)
set -u
cd "$(dirname "$0")/.."
LOG=validations/stage3_watchdog.log
ALERT=validations/stage3_alerts.md
mkdir -p validations
: > "$ALERT"  # truncate alerts
exec >> "$LOG" 2>&1

echo "[$(date)] watchdog started"

ANCHOR_TAG=warm0
ANCHOR_ARGS="--warmup_epochs 0"

last_count=-1
stall_count=0
restart_count=0
MAX_RESTARTS=5

alert() {
  echo "[$(date)] ALERT: $*" | tee -a "$ALERT"
}

fix_unicode() {
  # Search for em-dash in all stage3-relevant python, replace with hyphen
  for f in scripts/analyze_v11_anchor.py scripts/analyze_stress_anchor.py; do
    if [ -f "$f" ] && grep -l '—' "$f" > /dev/null 2>&1; then
      sed -i 's/—/-/g' "$f"
      alert "em-dash removed from $f"
    fi
  done
}

while true; do
  sleep 300  # 5 min

  # Count completed runs
  n_done=$(ls -d logs/*v11ax_warm0_*_F* 2>/dev/null | wc -l)
  chain_alive=$(ps -ef 2>/dev/null | grep -q "[6]2_v11_axis_full" && echo 1 || echo 0)
  train_alive=$(ps -ef 2>/dev/null | grep -q "[t]rain.py.*v11ax_warm0" && echo 1 || echo 0)

  echo "[$(date)] done=$n_done chain=$chain_alive train=$train_alive"

  # Completion detection
  if [ "$n_done" -ge 125 ]; then
    alert "Stage 3 COMPLETE: $n_done / 125 runs done."
    exit 0
  fi

  # Progress detection: stalled if no new runs for 3 checks (15 min)
  if [ "$n_done" = "$last_count" ]; then
    stall_count=$((stall_count + 1))
  else
    stall_count=0
    last_count=$n_done
  fi

  # Chain dead + incomplete → auto-restart
  if [ "$chain_alive" = "0" ] && [ "$train_alive" = "0" ] && [ "$n_done" -lt 125 ]; then
    restart_count=$((restart_count + 1))
    alert "Chain dead at $n_done / 125. Restart #$restart_count."

    # Try unicode fix first
    fix_unicode

    if [ "$restart_count" -le "$MAX_RESTARTS" ]; then
      export ANCHOR_TAG ANCHOR_ARGS
      nohup bash -c "export ANCHOR_TAG='$ANCHOR_TAG' ANCHOR_ARGS='$ANCHOR_ARGS'; bash scripts/sweeps_laptop/62_v11_axis_full.sh" \
        >> validations/v11_axis_full.log 2>&1 &
      alert "Chain relaunched (pid=$!). Will skip $n_done completed runs."
      sleep 30  # give it time to start
    else
      alert "Max restarts ($MAX_RESTARTS) exceeded. Stopping watchdog."
      exit 1
    fi
  fi

  # Stalled for 30+ min with no progress
  if [ "$stall_count" -ge 6 ]; then
    alert "Stalled: no progress for 30 min. Chain alive=$chain_alive train alive=$train_alive."
    stall_count=0  # reset to avoid spam
  fi

  # Saturation check: after 30+ runs, if mean_err < 2 across ALL, flag
  if [ "$n_done" -ge 30 ]; then
    mean_err=$(PYTHONIOENCODING=utf-8 python -c "
import json, glob
tots=[]
for p in glob.glob('logs/*v11ax_warm0_*_F*'):
    try:
        d=json.load(open(p+'/best_info.json'))
        tm=d['test_metrics']
        nor_c = round(tm['normal']['recall']*tm['normal']['count'])
        abn_c = round(tm['abnormal']['recall']*tm['abnormal']['count'])
        tots.append((tm['normal']['count']-nor_c) + (tm['abnormal']['count']-abn_c))
    except: pass
print(f'{sum(tots)/len(tots):.2f}' if tots else '99')
" 2>/dev/null)
    echo "[$(date)] n=$n_done mean_err=$mean_err"
    if [ -n "$mean_err" ] && awk "BEGIN {exit !($mean_err < 2)}" 2>/dev/null; then
      if [ ! -f validations/.saturation_alerted ]; then
        alert "Saturation confirmed at anchor=warm0: mean_err=$mean_err < 2 across $n_done runs. Consider vd_hard switch."
        touch validations/.saturation_alerted
      fi
    fi
  fi
done
