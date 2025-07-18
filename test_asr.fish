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

# --- GPU detection function (defined first) -----------------------------------
function check_gpu_availability
    if command -q nvidia-smi
        set -l gpu_count (nvidia-smi --query-gpu=name --format=csv,noheader,nounits | wc -l)
        echo $gpu_count
    else
        echo 0
    end
end

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
            echo "🚀 Multi-GPU ASR Evaluation Launcher"
            echo "================================================"
            echo ""
            echo "DESCRIPTION:"
            echo "  Background launcher for ASR evaluation with multi-GPU support."
            echo "  Automatically detects available GPUs and splits workload across them."
            echo "  Runs in background with detailed logging and result consolidation."
            echo ""
            echo "USAGE:"
            echo "  ./test_asr.fish [LAUNCHER_OPTIONS] [EVALUATION_OPTIONS]"
            echo ""
            echo "LAUNCHER OPTIONS:"
            echo "  -h, --help       Show this help message and exit"
            echo "  --num_gpus N     Number of GPUs to use (default: 1, max: auto-detected)"
            echo "  --auto_merge     Automatically merge parquet files after completion"
            echo "  --cleanup        Remove individual parquet files after merging"
            echo ""
            echo "EVALUATION OPTIONS (passed to evaluation script):"
            echo "  --dataset NAME   Dataset to evaluate (required)"
            echo "                   Choices: librispeech, wham, ami, gigaspeech,"
            echo "                            chime8_notsofar1, chime8_chime6, chime8_mixer6, chime8_dipco"
            echo "  --subset NAME    Dataset subset (e.g., 'test-clean', 'dev')"
            echo "  --config NAME    Dataset configuration"
            echo "  --streaming      Use streaming dataset loading"
            echo "  --max_samples N  Maximum number of samples to process"
            echo ""
            echo "W&B OPTIONS:"
            echo "  --wandb_project NAME    W&B project name (default: kimi-audio-multi-eval)"
            echo "  --wandb_run_name NAME   W&B run name"
            echo ""
            echo "CHIME-8 OPTIONS:"
echo "  --notsofar_version VER  NOTSOFAR version tag (e.g., '240825.1_dev1')"
echo ""
echo "LIBRISPEECH OPTIONS:"
echo "  --librispeech_use_hf    Use HuggingFace download (may fail for large datasets)"
echo "                          Default: direct download from OpenSLR (recommended)"
echo ""
echo "ENVIRONMENT VARIABLES:"
echo "  LIBRISPEECH_AUTO_DOWNLOAD=true    Skip download confirmation prompts"
echo "  LIBRISPEECH_CACHE=/path/to/cache  Custom cache directory for LibriSpeech"
            echo ""
            echo "EXAMPLES:"
            echo "  # Single GPU evaluation"
            echo "  ./test_asr.fish --dataset librispeech --subset test-clean"
            echo ""
            echo "  # Multi-GPU with auto-merge (recommended)"
            echo "  ./test_asr.fish --num_gpus 2 --dataset librispeech --subset test-clean --auto_merge"
            echo ""
            echo "  # CHiME-8 evaluation with cleanup"
            echo "  ./test_asr.fish --num_gpus 2 --dataset chime8_notsofar1 --subset dev --auto_merge --cleanup"
            echo ""
            echo "  # Limited samples for testing"
            echo "  ./test_asr.fish --dataset ami --subset dev --max_samples 100"
            echo ""
            echo "MONITORING:"
            echo "  After launch, monitor progress with:"
            echo "    tail -f logs/(ls -t logs | head -n1)"
            echo ""
            echo "OUTPUT FILES:"
            echo "  • batch_mistakes_parquet/    - Individual GPU results (parquet format)"
            echo "  • merged_mistakes.parquet    - Consolidated results (if --auto_merge used)"
            echo "  • logs/                      - Execution logs with timestamps"
            echo ""
            echo "GPU DETECTION:"
            set gpu_count (check_gpu_availability)
            echo "  Available GPUs: $gpu_count (detected via nvidia-smi)"
            echo ""
            echo "NOTES:"
echo "  • Evaluation runs in background and survives terminal disconnection"
echo "  • Results are saved incrementally to prevent data loss"
echo "  • Multi-GPU mode splits dataset across GPUs and runs inference in parallel"
echo "  • Each GPU gets its own CUDA_VISIBLE_DEVICES assignment (0, 1, etc.)"
echo "  • Single W&B run tracks aggregated progress from all GPUs"
echo "  • Individual GPU logs available for debugging: logs/gpu_N_asr_*.log"
echo "  • Use 'kill <PID>' to stop individual GPU processes"
echo "  • LibriSpeech downloads require user confirmation (unless LIBRISPEECH_AUTO_DOWNLOAD=true)"
            echo ""
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

# --- GPU availability check ---------------------------------------------------
set AVAILABLE_GPUS (check_gpu_availability)

# --- Launch evaluation script --------------------------------------------------
echo "[launcher] 🚀 Starting ASR evaluation..."
echo "[launcher] 📊 GPUs: Using $NUM_GPUS out of $AVAILABLE_GPUS available"
echo "[launcher] 📝 Logging to: $LOG_FILE"

if test $USE_MULTI_GPU = true
    echo "[launcher] 🔥 Multi-GPU mode: Launching $NUM_GPUS parallel processes"
    
    # Validate GPU count
    if test $NUM_GPUS -gt $AVAILABLE_GPUS
        echo "[launcher] ⚠️  Warning: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
        echo "[launcher] 🔧 Adjusting to use $AVAILABLE_GPUS GPUs"
        set NUM_GPUS $AVAILABLE_GPUS
    end
    
    # Build base arguments (excluding our custom launcher ones)
    set BASE_ARGS
    set skip_next false
    for arg in $argv
        if test $skip_next = true
            # Skip this argument (it was the value for --num_gpus)
            set skip_next false
            continue
        end
        
        switch $arg
            case --num_gpus
                # Skip this argument and the next one (its value)
                set skip_next true
            case --auto_merge --cleanup
                # Skip these (launcher-specific arguments)
            case '*'
                set BASE_ARGS $BASE_ARGS $arg
        end
    end
    
    # Add multi-GPU arguments to base args
    set BASE_ARGS $BASE_ARGS "--num_gpus" $NUM_GPUS
    
    # Store PIDs for process management
    set GPU_PIDS
    
    # Launch each GPU process with separate CUDA_VISIBLE_DEVICES
    for rank in (seq 0 (math $NUM_GPUS - 1))
        set gpu_id $rank
        set rank_log_file "$LOG_DIR/gpu_$rank"_asr_"$TS.log"
        
        echo "[launcher] 🚀 Starting GPU $rank (CUDA device $gpu_id)..."
        
        # Build command for this specific GPU/rank
        set GPU_CMD env CUDA_VISIBLE_DEVICES=$gpu_id python test_asr.py $BASE_ARGS --local_rank $rank
        
        # Launch in background with separate log file
        nohup $GPU_CMD > "$rank_log_file" 2>&1 &
        set gpu_pid $last_pid
        set GPU_PIDS $GPU_PIDS $gpu_pid
        
        echo "[launcher] ✅ GPU $rank started (PID: $gpu_pid, Log: $rank_log_file)"
        
        # Small delay to avoid race conditions in distributed setup
        sleep 2
    end
    
    # Create combined log monitoring
    echo "[launcher] 📝 Creating combined log file: $LOG_FILE"
    
    # Function to merge logs in background
    begin
        echo "=== Multi-GPU ASR Evaluation Started at (date) ===" > "$LOG_FILE"
        echo "GPUs used: $NUM_GPUS" >> "$LOG_FILE"
        echo "Process PIDs: $GPU_PIDS" >> "$LOG_FILE"
        echo "Individual logs: $LOG_DIR/gpu_*_asr_$TS.log" >> "$LOG_FILE"
        echo "===========================================" >> "$LOG_FILE"
        echo "" >> "$LOG_FILE"
        
        # Monitor and merge logs from all GPU processes
        while true
            for rank in (seq 0 (math $NUM_GPUS - 1))
                set rank_log "$LOG_DIR/gpu_$rank"_asr_"$TS.log"
                if test -f "$rank_log"
                    echo "--- GPU $rank output ---" >> "$LOG_FILE"
                    tail -n 10 "$rank_log" 2>/dev/null >> "$LOG_FILE"
                end
            end
            echo "" >> "$LOG_FILE"
            sleep 30  # Update every 30 seconds
            
            # Check if any GPU process is still running
            set running_processes 0
            for pid in $GPU_PIDS
                if ps -p $pid > /dev/null 2>&1
                    set running_processes (math $running_processes + 1)
                end
            end
            
            if test $running_processes -eq 0
                echo "=== All GPU processes completed at (date) ===" >> "$LOG_FILE"
                break
            end
        end
    end &
    
    # Set main PID to first GPU process for compatibility
    set PID $GPU_PIDS[1]
    
else
    echo "[launcher] 🔧 Single-GPU mode: Using test_asr.py directly"
    
    # For single GPU, use the original test_asr.py
    nohup python test_asr.py $argv > "$LOG_FILE" 2>&1 &
    set PID $last_pid
end

# detach the job so it won't appear in `jobs`
disown $PID

# --- user feedback -------------------------------------------------------------
if test $USE_MULTI_GPU = true
    echo "[launcher] ✅ Started multi-GPU evaluation"
    echo "[launcher] 🔢 GPU PIDs: $GPU_PIDS"
    echo "[launcher] 📁 Individual logs: $LOG_DIR/gpu_*_asr_$TS.log"
    echo "[launcher] 📁 Combined log: $LOG_FILE"
    echo "[launcher] 📁 Results will be saved to 'batch_mistakes_parquet/'"
    echo "[launcher] 🔄 Multi-GPU results are automatically aggregated during evaluation"
    
    # Setup auto-merge functionality if requested
    if test $AUTO_MERGE = true
        echo "[launcher] 📦 Auto-merge enabled: Results will be consolidated after completion"
        
        # Background process to monitor completion and auto-merge
        begin
            # Wait for all GPU processes to complete
            while true
                set running_processes 0
                for pid in $GPU_PIDS
                    if ps -p $pid > /dev/null 2>&1
                        set running_processes (math $running_processes + 1)
                    end
                end
                
                if test $running_processes -eq 0
                    echo "[launcher] 🎉 All GPU processes completed, starting auto-merge..."
                    if test -f "merge_batch_mistakes.py"
                        python merge_batch_mistakes.py
                        echo "[launcher] 📦 Results merged into 'merged_mistakes.parquet'"
                        
                        if test $CLEANUP = true
                            echo "[launcher] 🧹 Cleaning up individual parquet files..."
                            rm -f batch_mistakes_parquet/mistakes_batch_*.parquet
                            echo "[launcher] ✅ Cleanup completed"
                        end
                    else
                        echo "[launcher] ⚠️  merge_batch_mistakes.py not found, skipping auto-merge"
                    end
                    break
                end
                
                sleep 30  # Check every 30 seconds
            end
        end &
    else
        echo "[launcher] 💡 Run 'python merge_batch_mistakes.py' after completion to consolidate results"
    end
else
    echo "[launcher] ✅ Started single-GPU evaluation (PID: $PID)"
    echo "[launcher] 📁 Results will be saved to 'batch_mistakes_parquet/'"
    echo "[launcher] 💡 Run 'python merge_batch_mistakes.py' after completion to consolidate results"
end

echo ""
echo "[launcher] 📊 Monitor progress:"
if test $USE_MULTI_GPU = true
    echo "  # Combined view (updates every 30s)"
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "  # Individual GPU logs (real-time)"
    for rank in (seq 0 (math $NUM_GPUS - 1))
        echo "  tail -f $LOG_DIR/gpu_$rank"_asr_"$TS.log  # GPU $rank"
    end
    echo ""
    echo "[launcher] 🔍 Check status:"
    echo "  # All GPU processes"
    for pid in $GPU_PIDS
        echo "  ps aux | grep $pid"
    end
    echo ""
    echo "[launcher] 🛑 Stop evaluation:"
    echo "  # Stop all GPU processes"
    for pid in $GPU_PIDS
        echo "  kill $pid"
    end
    echo "  # Or stop all at once"
    echo "  kill $GPU_PIDS"
else
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "[launcher] 🔍 Check status:"
    echo "  ps aux | grep $PID"
    echo ""
    echo "[launcher] 🛑 Stop evaluation:"
    echo "  kill $PID"
end
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
