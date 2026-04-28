#!/usr/bin/env bash
# Team Checkpoint Daemon — runs forever, watches training progress,
# raises flag file every N new completed runs.
#
# Usage:
#   bash scripts/team_checkpoint_daemon.sh start    # launch as background
#   bash scripts/team_checkpoint_daemon.sh stop     # kill running daemon
#   bash scripts/team_checkpoint_daemon.sh status   # check if running
#   bash scripts/team_checkpoint_daemon.sh log      # tail checkpoint log
#
# Flag files (read/reset by Claude via /체크 command):
#   validations/.team_checkpoint_ready    # exists = team agent spawn requested
#   validations/.last_team_checkpoint     # integer: total runs at last checkpoint
#   validations/.team_log                 # JSONL history of checkpoints

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p validations

PID_FILE="validations/.daemon.pid"
LOG_FILE="validations/.daemon.log"
CHECKPOINT_EVERY=25    # trigger flag every N new runs
POLL_INTERVAL=600      # seconds between checks (10min)

count_runs() {
  local bc ax tie hunt gold
  bc=$(ls -d logs/*vd080_bc_*_F* 2>/dev/null | grep -v vd080_bc_tie | wc -l)
  ax=$(ls -d logs/*vd080_ax_*_F* 2>/dev/null | wc -l)
  tie=$(ls -d logs/*vd080_bc_tie_*_F* 2>/dev/null | wc -l)
  hunt=$(ls -d logs/*vd080_hunt_*_F* 2>/dev/null | wc -l)
  gold=$(ls -d logs/*vd080_control_*_F* 2>/dev/null | wc -l)
  gold2=$(ls -d logs/*vd080_golden_*_F* 2>/dev/null | wc -l)
  echo $((bc + ax + tie + hunt + gold + gold2))
}

run_daemon() {
  echo $$ > "$PID_FILE"
  echo "[daemon] started pid=$$ at $(date)" >> "$LOG_FILE"
  while true; do
    n=$(count_runs)
    last=$(cat validations/.last_team_checkpoint 2>/dev/null || echo 0)
    diff=$((n - last))
    if [ "$diff" -ge "$CHECKPOINT_EVERY" ]; then
      echo "$n" > validations/.last_team_checkpoint
      touch validations/.team_checkpoint_ready
      ts=$(date -Iseconds)
      echo "{\"ts\":\"$ts\",\"total\":$n,\"diff\":$diff,\"trigger\":\"count\"}" >> validations/.team_log
      echo "[daemon $(date +%H:%M)] CHECKPOINT total=$n, +$diff since last" >> "$LOG_FILE"
    fi
    # also set a stage-completion trigger (when an entire sweep stage just finished)
    if [ -f validations/.stage_complete ]; then
      rm -f validations/.stage_complete
      touch validations/.team_checkpoint_ready
      ts=$(date -Iseconds)
      echo "{\"ts\":\"$ts\",\"total\":$n,\"trigger\":\"stage_complete\"}" >> validations/.team_log
      echo "[daemon $(date +%H:%M)] CHECKPOINT stage complete at total=$n" >> "$LOG_FILE"
    fi
    sleep "$POLL_INTERVAL"
  done
}

cmd="${1:-status}"
case "$cmd" in
  start)
    if [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
      echo "already running (pid=$(cat $PID_FILE))"
      exit 0
    fi
    nohup bash -c "$(declare -f run_daemon count_runs); PID_FILE='$PID_FILE' LOG_FILE='$LOG_FILE' CHECKPOINT_EVERY=$CHECKPOINT_EVERY POLL_INTERVAL=$POLL_INTERVAL; cd '$ROOT'; run_daemon" >/dev/null 2>&1 &
    sleep 1
    echo "started (pid in $PID_FILE, log in $LOG_FILE)"
    ;;
  stop)
    if [ -f "$PID_FILE" ]; then
      pid=$(cat "$PID_FILE")
      kill "$pid" 2>/dev/null && echo "killed pid $pid"
      rm -f "$PID_FILE"
    else
      echo "no pid file"
    fi
    ;;
  status)
    if [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
      echo "running (pid=$(cat $PID_FILE))"
      n=$(count_runs)
      last=$(cat validations/.last_team_checkpoint 2>/dev/null || echo 0)
      echo "total runs=$n, last checkpoint=$last, diff=$((n - last))/$CHECKPOINT_EVERY"
      [ -f validations/.team_checkpoint_ready ] && echo "⚠ FLAG SET: /체크 to consume" || echo "flag: clear"
    else
      echo "not running"
    fi
    ;;
  log)
    tail -20 "$LOG_FILE" 2>/dev/null
    echo "---checkpoint log---"
    tail -10 validations/.team_log 2>/dev/null
    ;;
  *)
    echo "Usage: $0 {start|stop|status|log}"
    exit 1
    ;;
esac
