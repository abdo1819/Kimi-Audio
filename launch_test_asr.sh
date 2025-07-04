#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# launch_test_asr.sh  –  background launcher for test_asr.py
# -----------------------------------------------------------------------------
# Starts the evaluation script with `nohup` so it keeps running after the
# terminal (SSH session, tmux pane, etc.) is closed, and then immediately
# `disown`s the job so it no longer appears in the shell’s job table.
#
# All stdout/stderr are captured in a timestamped log‑file under ./logs/
# (created if it does not yet exist).
#
# Usage examples
# ---------------
#   ./launch_test_asr.sh                      # run with script defaults
#   ./launch_test_asr.sh --dataset ami --subset dev --max_samples 500
#   ./launch_test_asr.sh -h                   # see test_asr.py help
#
# You can follow the live logs with e.g.  tail -f logs/<latest>.log
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/test_asr_${TS}.log"

# Run test_asr.py in the background with nohup
nohup python -u "$SCRIPT_DIR/test_asr.py" "$@" \
     > "$LOG_FILE" 2>&1 &
PID=$!

disown "$PID"

echo "[launcher] Started test_asr.py (PID $PID)"
echo "[launcher] Logging to $LOG_FILE"
