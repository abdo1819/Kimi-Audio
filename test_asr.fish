#!/usr/bin/env fish
# -----------------------------------------------------------------------------
# launch_test_asr.fish  –  Multi-GPU background launcher for ASR evaluation
# -----------------------------------------------------------------------------
# Launches ASR evaluation with multi-GPU support using `nohup` so it keeps 
# running after the terminal (SSH session, tmux pane, etc.) is closed.
#
# Features:
# - Multi-GPU support (auto-detects available GPUs)
# - Background execution with detailed logging
# - Auto-merge of results when completed
# - Support for both single and multi-GPU modes
#
# All stdout/stderr are captured in timestamped log-files under ./logs/
# (created if it does not yet exist).
#
# Usage examples
# ---------------
#   ./launch_test_asr.fish --dataset librispeech --subset test-clean
#   ./launch_test_asr.fish --dataset ami --subset dev --num_gpus 2 --auto_merge
#   ./launch_test_asr.fish --dataset chime8_notsofar1 --num_gpus 1 --max_samples 500
#   ./launch_test_asr.fish -h                   # see help
#
# You can follow the live logs with e.g.  tail -f logs/(ls -t logs | head -n1)
# -----------------------------------------------------------------------------

# --- Parse arguments to detect GPU settings -----------------------------------
set USE_MULTI_GPU false
set NUM_GPUS 1
set AUTO_MERGE false
set CLEANUP false

# Check for multi-GPU arguments
for i in (seq (count $argv))
    switch $argv[$i]
        case --num_gpus
            if test (math $i + 1) -le (count $argv)
                set NUM_GPUS $argv[(math $i + 1)]
                if test $NUM_GPUS -gt 1
                    set USE_MULTI_GPU true
                end
            end
        case --auto_merge
            set AUTO_MERGE true
        case --cleanup
            set CLEANUP true
        case -h --help
            echo "Multi-GPU ASR Evaluation Launcher"
            echo ""
            echo "Usage: $argv[0] [OPTIONS]"
            echo ""
            echo "Multi-GPU Options:"
            echo "  --num_gpus N     Number of GPUs to use (default: 1)"
            echo "  --auto_merge     Automatically merge parquet files after completion"
            echo "  --cleanup        Remove individual parquet files after merging"
            echo ""
            echo "All other arguments are passed to the evaluation script."
            echo "Run with --help to see evaluation script options."
            exit 0
    end
end

# --- determine script directory ------------------------------------------------
set SCRIPT_DIR (cd (dirname (status --current-filename)) ; pwd)

# --- set up logging ------------------------------------------------------------
set LOG_DIR "$SCRIPT_DIR/logs"
mkdir -p $LOG_DIR

set TS (date "+%Y%m%d_%H%M%S")
if test $USE_MULTI_GPU = true
    set LOG_FILE "$LOG_DIR/multi_gpu_asr_$TS.log"
else
    set LOG_FILE "$LOG_DIR/single_gpu_asr_$TS.log"
end

# --- Auto-detect available GPUs -----------------------------------------------
function check_gpu_availability
    if command -q nvidia-smi
        set -l gpu_count (nvidia-smi --query-gpu=name --format=csv,noheader,nounits | wc -l)
        echo $gpu_count
    else
        echo 0
    end
end

set AVAILABLE_GPUS (check_gpu_availability)

if test $AVAILABLE_GPUS -eq 0
    echo "[launcher] ❌ No CUDA GPUs detected. Falling back to CPU (if supported)."
    set USE_MULTI_GPU false
    set NUM_GPUS 1
else if test $NUM_GPUS -gt $AVAILABLE_GPUS
    echo "[launcher] ⚠️  Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available."
    echo "[launcher] 🔧 Adjusting to use $AVAILABLE_GPUS GPUs."
    set NUM_GPUS $AVAILABLE_GPUS
    if test $NUM_GPUS -gt 1
        set USE_MULTI_GPU true
    else
        set USE_MULTI_GPU false
    end
end

# --- Launch evaluation script --------------------------------------------------
echo "[launcher] 🚀 Starting ASR evaluation..."
echo "[launcher] 📊 GPUs: Using $NUM_GPUS out of $AVAILABLE_GPUS available"
echo "[launcher] 📝 Logging to: $LOG_FILE"

if test $USE_MULTI_GPU = true
    echo "[launcher] 🔥 Multi-GPU mode: Using launch_multi_gpu_asr.py"
    
    # Build command for multi-GPU launcher
    set LAUNCH_CMD "python" "launch_multi_gpu_asr.py" "--num_gpus" $NUM_GPUS
    
    if test $AUTO_MERGE = true
        set LAUNCH_CMD $LAUNCH_CMD "--auto_merge"
    end
    
    if test $CLEANUP = true
        set LAUNCH_CMD $LAUNCH_CMD "--cleanup"
    end
    
    # Add all other arguments (excluding our custom ones)
    for i in (seq (count $argv))
        switch $argv[$i]
            case --num_gpus
                # Skip this and the next argument
                set i (math $i + 1)
            case --auto_merge --cleanup
                # Skip these (already handled)
            case '*'
                set LAUNCH_CMD $LAUNCH_CMD $argv[$i]
        end
    end
    
    # Launch in background
    nohup $LAUNCH_CMD > "$LOG_FILE" 2>&1 &
    set PID $last_pid
    
else
    echo "[launcher] 🔧 Single-GPU mode: Using test_asr.py directly"
    
    # For single GPU, use the original test_asr.py
    nohup python test_asr.py $argv > "$LOG_FILE" 2>&1 &
    set PID $last_pid
end

# detach the job so it won't appear in `jobs`
disown $PID

# --- user feedback -------------------------------------------------------------
echo "[launcher] ✅ Started evaluation (PID: $PID)"
echo "[launcher] 📁 Results will be saved to 'batch_mistakes_parquet/'"

if test $USE_MULTI_GPU = true
    echo "[launcher] 🔄 Multi-GPU results will be automatically aggregated"
    if test $AUTO_MERGE = true
        echo "[launcher] 📦 Results will be auto-merged into 'merged_mistakes.parquet'"
    end
else
    echo "[launcher] 💡 Run 'python merge_batch_mistakes.py' after completion to consolidate results"
end

echo ""
echo "[launcher] 📊 Monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "[launcher] 🔍 Check status:"
echo "  ps aux | grep $PID"
echo ""
echo "[launcher] 🛑 Stop evaluation:"
echo "  kill $PID"
echo ""

# --- Optional: Show current nvidia-smi status ---------------------------------
if test $AVAILABLE_GPUS -gt 0
    echo "[launcher] 🎯 Current GPU status:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | while read -l line
        echo "  GPU $line"
    end
    echo ""
end

echo "[launcher] 🎉 Background evaluation started successfully!"
