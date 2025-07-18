#!/usr/bin/env python
"""
Multi-GPU ASR Evaluation Launcher
---------------------------------
Launches ASR evaluation across multiple GPUs using torch.multiprocessing.

Usage:
    python launch_multi_gpu_asr.py --num_gpus 2 --dataset librispeech --subset test-clean
    python launch_multi_gpu_asr.py --num_gpus 1 --dataset chime8_notsofar1 --subset dev
"""

import argparse
import subprocess
import sys
import os
import torch
import torch.multiprocessing as mp
from pathlib import Path


def run_evaluation_process(rank, world_size, args):
    """Run evaluation on a single GPU"""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(rank)
    
    cmd = [
        sys.executable, 'test_asr.py',
        '--num_gpus', str(world_size),
        '--local_rank', str(rank),
        '--dataset', args.dataset,
    ]
    
    # Add optional arguments
    if args.subset:
        cmd.extend(['--subset', args.subset])
    if args.config:
        cmd.extend(['--config', args.config])
    if args.streaming:
        cmd.append('--streaming')
    if args.max_samples:
        cmd.extend(['--max_samples', str(args.max_samples)])
    if args.wandb_project:
        cmd.extend(['--wandb_project', args.wandb_project])
    if args.wandb_run_name:
        cmd.extend(['--wandb_run_name', args.wandb_run_name])
    if args.notsofar_version:
        cmd.extend(['--notsofar_version', args.notsofar_version])
    if args.librispeech_use_hf:
        cmd.append('--librispeech_use_hf')
    
    print(f"🚀 Starting GPU {rank} process with command:")
    print(f"   {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, env=env, check=True)
        print(f"✅ GPU {rank} completed successfully")
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"❌ GPU {rank} failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print(f"🛑 GPU {rank} interrupted by user")
        sys.exit(130)


def main():
    parser = argparse.ArgumentParser(description="Launch multi-GPU ASR evaluation")
    
    # Multi-GPU settings
    parser.add_argument("--num_gpus", type=int, default=2, help="Number of GPUs to use")
    
    # Dataset arguments (passed through to test_asr.py)
    parser.add_argument("--dataset", required=True, help="Dataset to evaluate")
    parser.add_argument("--subset", type=str, help="Dataset subset")
    parser.add_argument("--config", type=str, help="Dataset config")
    parser.add_argument("--streaming", action="store_true", help="Use streaming dataset")
    parser.add_argument("--max_samples", type=int, help="Maximum samples to process")
    
    # W&B arguments
    parser.add_argument("--wandb_project", default="kimi-audio-multi-eval", help="W&B project name")
    parser.add_argument("--wandb_run_name", help="W&B run name")
    parser.add_argument("--notsofar_version", help="NOTSOFAR version tag")
    parser.add_argument("--librispeech_use_hf", action="store_true", 
                       help="Use HuggingFace download for LibriSpeech (may fail for large datasets)")
    
    # Additional options
    parser.add_argument("--auto_merge", action="store_true", help="Automatically merge parquet files after completion")
    parser.add_argument("--cleanup", action="store_true", help="Clean up individual parquet files after merging")
    
    args = parser.parse_args()
    
    # Validate GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        sys.exit(1)
    
    available_gpus = torch.cuda.device_count()
    if args.num_gpus > available_gpus:
        print(f"❌ Requested {args.num_gpus} GPUs but only {available_gpus} available")
        sys.exit(1)
    
    print(f"🔥 Launching ASR evaluation on {args.num_gpus} GPUs")
    print(f"   Dataset: {args.dataset}")
    if args.subset:
        print(f"   Subset: {args.subset}")
    print(f"   Available GPUs: {list(range(available_gpus))}")
    print(f"   Using GPUs: {list(range(args.num_gpus))}")
    print()
    
    if args.num_gpus == 1:
        # Single GPU - run directly
        exit_code = run_evaluation_process(0, 1, args)
    else:
        # Multi-GPU - use multiprocessing
        try:
            # Use spawn method for CUDA compatibility
            mp.set_start_method('spawn', force=True)
            
            # Create and start processes
            processes = []
            for rank in range(args.num_gpus):
                p = mp.Process(target=run_evaluation_process, args=(rank, args.num_gpus, args))
                p.start()
                processes.append(p)
                print(f"🚀 Started process for GPU {rank} (PID: {p.pid})")
            
            # Wait for all processes to complete
            exit_codes = []
            for rank, p in enumerate(processes):
                p.join()
                exit_codes.append(p.exitcode)
                print(f"🏁 GPU {rank} process finished with exit code {p.exitcode}")
            
            # Check if all processes succeeded
            if all(code == 0 for code in exit_codes):
                print("\n🎉 All GPU processes completed successfully!")
                exit_code = 0
            else:
                failed_gpus = [i for i, code in enumerate(exit_codes) if code != 0]
                print(f"\n❌ Some GPU processes failed: {failed_gpus}")
                exit_code = 1
                
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user - terminating all processes...")
            for p in processes:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
                    if p.is_alive():
                        p.kill()
            exit_code = 130
    
    # Auto-merge parquet files if requested
    if exit_code == 0 and args.auto_merge:
        print("\n📦 Auto-merging parquet files...")
        merge_cmd = ['python', 'merge_batch_mistakes.py']
        if args.cleanup:
            merge_cmd.append('--cleanup')
        
        try:
            subprocess.run(merge_cmd, check=True)
            print("✅ Parquet files merged successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to merge parquet files: {e}")
            exit_code = 1
    
    print(f"\n🏁 Evaluation completed with exit code: {exit_code}")
    if exit_code == 0:
        print("📁 Check 'batch_mistakes_parquet/' for individual GPU results")
        if args.auto_merge:
            print("📄 Check 'merged_mistakes.parquet' for consolidated results")
        else:
            print("💡 Run 'python merge_batch_mistakes.py' to consolidate results")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main() 