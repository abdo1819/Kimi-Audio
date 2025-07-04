#!/usr/bin/env fish
# -----------------------------------------------------------------------------
# launch_test_asr.fish  –  background launcher for test_asr.py (Fish shell)
# -----------------------------------------------------------------------------
# Starts the evaluation script with `nohup` so it keeps running after the
# terminal (SSH session, tmux pane, etc.) is closed, and then immediately
# `disown`s the job so it no longer appears in the shell’s job table.
#
# All stdout/stderr are captured in a timestamped log-file under ./logs/
# (created if it does not yet exist).
#
# Usage examples
# ---------------
#   ./launch_test_asr.fish                      # run with script defaults
#   ./launch_test_asr.fish --dataset ami --subset dev --max_samples 500
#   ./launch_test_asr.fish -h                   # see test_asr.py help
#
# You can follow the live logs with e.g.  tail -f logs/(ls -t logs | head -n1)
# -----------------------------------------------------------------------------

# --- determine script directory ------------------------------------------------
# (status --current-filename) is this script’s path; dirname gives its folder
set SCRIPT_DIR (cd (dirname (status --current-filename)) ; pwd)

# --- set up logging ------------------------------------------------------------
set LOG_DIR "$SCRIPT_DIR/logs"
mkdir -p $LOG_DIR

set TS (date "+%Y%m%d_%H%M%S")
set LOG_FILE "$LOG_DIR/test_asr_$TS.log"

# --- run the python evaluation in background ----------------------------------
# $argv is Fish’s equivalent of "$@" from Bash
# 2>&1 merges stderr into stdout just like in Bash
nohup python -m "asr_eval.cli" $argv > "$LOG_FILE" 2>&1 &

# pid of last backgrounded command
set PID $last_pid

# detach the job so it won’t appear in `jobs`
disown $PID

# --- user feedback -------------------------------------------------------------
echo "[launcher] Started test_asr.py (PID $PID)"
echo "[launcher] Logging to $LOG_FILE"
