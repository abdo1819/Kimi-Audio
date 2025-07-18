#!/usr/bin/env python
"""
UPDATED – Whisper BasicTextNormalizer + CER & SER + CHiME‑8 datasets
------------------------------------------------------------------
* **Normalization** now uses `transformers.models.whisper.english_normalizer.BasicTextNormalizer`,
  matching Whisper's official preprocessing.
* Added **CER** (character error rate) via `evaluate.load("cer")`.
* Added **SER** (sentence/utterance error rate – a binary 0/1 per sample).
* **CHiME‑8 support** – new `chime8` dataset option with initial support for
  the NOTSOFAR‑1 dev/train sets hosted on Hugging Face (`microsoft/NOTSOFAR`).
  Place‑holders are included to plug in local prepared CHiME‑8 DASR data
  (CHiME‑6, Mixer‑6, DiPCo) generated with `chime-utils`.
* W&B logs include running and final **WER/CER/SER** plus the
  per‑250‑step *new mistakes* table as before.
* **Direct LibriSpeech download** - downloads directly from OpenSLR instead of using HF automatic download
  to avoid issues with large datasets (30GB+).
"""

from __future__ import annotations

import os, re, csv, argparse, tempfile, difflib, sys, shlex
from pathlib import Path
import subprocess, json, hashlib, itertools
from typing import Tuple, Optional
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import urllib.request
import time
import shutil

import wandb
from tqdm import tqdm
import soundfile as sf
from datasets import load_dataset, load_dataset_builder, Audio, IterableDataset, Dataset as HFDataset
import evaluate
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from kimia_infer.api.kimia import KimiAudio
from colorama import init as colorama_init, Fore, Style
import pandas as pd
from datetime import datetime

colorama_init(autoreset=True)

# ----------------------- LIBRISPEECH DIRECT DOWNLOAD ---------------------

def get_librispeech_cache_dir() -> Path:
    """Get the consistent cache directory for LibriSpeech downloads."""
    cache_dir = Path(os.environ.get("LIBRISPEECH_CACHE", "~/.cache/librispeech")).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def _download_with_progress(url: str, output_path: Path) -> None:
    """Download file with Python urllib and show progress."""
    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, (downloaded * 100) // total_size)
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            
            # Simple progress bar
            bar_length = 40
            filled_length = int(bar_length * downloaded // total_size)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            print(f'\r📥 [{bar}] {percent:3d}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)', end='', flush=True)
    
    try:
        print(f"📥 Starting download with Python urllib...")
        urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
        print()  # New line after progress bar
    except Exception as e:
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError(f"Python download failed: {e}")

def download_librispeech_subset(subset: str, cache_root: Path) -> Path:
    """
    Download LibriSpeech subset directly from OpenSLR if not already cached.
    
    Args:
        subset: LibriSpeech subset name (e.g., 'test-clean', 'train-clean-100')
        cache_root: Root directory for caching downloaded files
        
    Returns:
        Path to the extracted LibriSpeech/{subset} directory
    """
    # Map subset names to download URLs
    LIBRISPEECH_URLS = {
        "test-clean": "https://us.openslr.org/resources/12/test-clean.tar.gz",
        "test-other": "https://us.openslr.org/resources/12/test-other.tar.gz", 
        "dev-clean": "https://us.openslr.org/resources/12/dev-clean.tar.gz",
        "dev-other": "https://us.openslr.org/resources/12/dev-other.tar.gz",
        "train-clean-100": "https://us.openslr.org/resources/12/train-clean-100.tar.gz",
        "train-clean-360": "https://us.openslr.org/resources/12/train-clean-360.tar.gz", 
        "train-other-500": "https://us.openslr.org/resources/12/train-other-500.tar.gz",
    }
    
    if subset not in LIBRISPEECH_URLS:
        raise ValueError(f"Unknown LibriSpeech subset: {subset}. Available: {list(LIBRISPEECH_URLS.keys())}")
    
    cache_root.mkdir(parents=True, exist_ok=True)
    subset_dir = cache_root / "LibriSpeech" / subset
    
    # Check if already downloaded and extracted with proper validation
    if subset_dir.exists() and any(subset_dir.glob("*/*.flac")):
        # Additional check: ensure we have both audio files and transcript files
        has_audio = len(list(subset_dir.glob("*/*.flac"))) > 0
        has_transcripts = len(list(subset_dir.glob("*/*.trans.txt"))) > 0
        if has_audio and has_transcripts:
            print(f"✅ LibriSpeech {subset} already downloaded at {subset_dir}")
            return subset_dir
        else:
            print(f"⚠️  LibriSpeech {subset} partially downloaded, re-extracting...")
    
    url = LIBRISPEECH_URLS[subset]
    tar_file = cache_root / f"{subset}.tar.gz"
    
    # Subset size information for user awareness
    SUBSET_SIZES = {
        "test-clean": "~346MB",
        "test-other": "~328MB", 
        "dev-clean": "~337MB",
        "dev-other": "~314MB",
        "train-clean-100": "~6.3GB",
        "train-clean-360": "~23GB", 
        "train-other-500": "~30GB",
    }
    
    size_info = SUBSET_SIZES.get(subset, "unknown size")
    
    # Check if tar file already exists
    if tar_file.exists():
        file_size_mb = tar_file.stat().st_size / (1024 * 1024)
        print(f"📦 Found existing archive: {tar_file} ({file_size_mb:.1f} MB)")
        print(f"   Skipping download, proceeding with extraction...")
    else:
        print(f"📥 Downloading LibriSpeech {subset} ({size_info}) from {url}")
        print(f"   This may take a while for large subsets...")
        
        # Try different download methods in order of preference
        download_successful = False
        last_error = None
        
        # Method 1: wget with progress
        if not download_successful and subprocess.run(["which", "wget"], capture_output=True).returncode == 0:
            try:
                download_cmd = [
                    "wget", 
                    "-c",  # continue partial downloads
                    "--progress=bar:force:noscroll",  # show progress bar
                    "--show-progress",  # show progress even when not on tty
                    "-O", str(tar_file), 
                    url
                ]
                print(f"📥 Starting download with wget...")
                subprocess.run(download_cmd, check=True)
                download_successful = True
            except subprocess.CalledProcessError as e:
                last_error = f"wget failed: {e}"
                if tar_file.exists():
                    tar_file.unlink()
        
        # Method 2: curl with progress 
        if not download_successful and subprocess.run(["which", "curl"], capture_output=True).returncode == 0:
            try:
                download_cmd = [
                    "curl", 
                    "-L",  # follow redirects
                    "-C", "-",  # continue partial downloads
                    "--progress-bar",  # show progress bar
                    "-o", str(tar_file), 
                    url
                ]
                print(f"📥 Starting download with curl...")
                subprocess.run(download_cmd, check=True)
                download_successful = True
            except subprocess.CalledProcessError as e:
                last_error = f"curl failed: {e}"
                if tar_file.exists():
                    tar_file.unlink()
        
        # Method 3: Python urllib as fallback
        if not download_successful:
            try:
                _download_with_progress(url, tar_file)
                download_successful = True
            except Exception as e:
                last_error = f"Python download failed: {e}"
        
        if not download_successful:
            raise RuntimeError(f"All download methods failed. Last error: {last_error}")
        
        print(f"✅ Download completed: {tar_file}")
    
    # Extract the archive
    try:
        print(f"📦 Extracting {tar_file}...")
        
        # Show archive size for reference
        archive_size_mb = tar_file.stat().st_size / (1024 * 1024)
        print(f"   Archive size: {archive_size_mb:.1f} MB")
        print(f"   This may take a few minutes for large archives...")
        
        # Use tar with standard options (checkpoint options don't work consistently across systems)
        extract_cmd = ["tar", "-xzf", str(tar_file), "-C", str(cache_root)]
        
        # Start extraction and show progress
        print(f"   Extracting...")
        start_time = time.time()
        
        # Run extraction in subprocess
        process = subprocess.Popen(extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Show progress dots while extraction is running
        dot_count = 0
        while process.poll() is None:
            print(".", end="", flush=True)
            dot_count += 1
            if dot_count % 20 == 0:  # New line every 20 dots
                elapsed = time.time() - start_time
                print(f" ({elapsed:.0f}s)")
            time.sleep(2)
        
        # Wait for completion
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"\n❌ Extraction failed!")
            stderr_text = stderr.decode() if stderr else "No error details"
            stdout_text = stdout.decode() if stdout else "No output"
            
            print(f"   Error details: {stderr_text}")
            print(f"   Return code: {process.returncode}")
            
            # Check for common issues and provide helpful suggestions
            if process.returncode == 2:
                print(f"\n🔍 Diagnosing tar exit code 2 (common causes):")
                
                # Check file integrity with tar -t
                print(f"   Checking archive integrity...")
                try:
                    test_result = subprocess.run(
                        ["tar", "-tzf", str(tar_file)], 
                        capture_output=True, 
                        text=True, 
                        timeout=30
                    )
                    if test_result.returncode != 0:
                        print(f"   ❌ Archive appears corrupted (tar -t failed)")
                        print(f"   💡 Solution: Delete the corrupted file and re-download")
                        print(f"      rm {tar_file}")
                        
                                                 # Offer to delete corrupted file and retry
                        tar_file.unlink()
                        print(f"   🗑️ Automatically removed corrupted archive")
                        print(f"   🔄 Attempting fresh download...")
                        
                        # Recursive call to retry download and extraction
                        return download_librispeech_subset(subset, cache_root)
                    else:
                        print(f"   ✅ Archive integrity test passed")
                except subprocess.TimeoutExpired:
                    print(f"   ⏰ Archive integrity test timed out (very large archive)")
                except Exception as e:
                    print(f"   ⚠️ Could not test archive integrity: {e}")
                
                # Check disk space
                try:
                    free_space = shutil.disk_usage(cache_root).free
                    free_space_gb = free_space / (1024**3)
                    archive_size = tar_file.stat().st_size
                    archive_size_gb = archive_size / (1024**3)
                    
                    print(f"   Disk space: {free_space_gb:.1f} GB free, archive: {archive_size_gb:.1f} GB")
                    
                    # LibriSpeech archives typically expand to ~2-3x their compressed size
                    estimated_extracted_size = archive_size * 2.5
                    estimated_extracted_gb = estimated_extracted_size / (1024**3)
                    
                    if free_space < estimated_extracted_size:
                        print(f"   ❌ Insufficient disk space!")
                        print(f"   📊 Need ~{estimated_extracted_gb:.1f} GB for extraction, only {free_space_gb:.1f} GB available")
                        print(f"   💡 Solutions:")
                        print(f"      1. Free up disk space (delete {estimated_extracted_gb - free_space_gb:.1f} GB)")
                        print(f"      2. Use different cache location: export LIBRISPEECH_CACHE=/path/to/larger/disk")
                        print(f"      3. Use smaller subset (e.g., train-clean-100 instead of train-other-500)")
                        raise RuntimeError(f"Insufficient disk space for extraction")
                    elif free_space < estimated_extracted_size * 1.2:  # Less than 20% buffer
                        print(f"   ⚠️ Disk space is tight! Estimated extraction needs {estimated_extracted_gb:.1f} GB")
                except Exception as e:
                    print(f"   ⚠️ Could not check disk space: {e}")
                
                # Check permissions
                if not os.access(cache_root, os.W_OK):
                    print(f"   ❌ No write permission to cache directory: {cache_root}")
                    print(f"   💡 Solution: Fix permissions or use different cache location")
            
            raise subprocess.CalledProcessError(process.returncode, extract_cmd, stderr_text)
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ Extraction completed in {elapsed_time:.1f}s: {subset_dir}")
        
        # Verify extraction was successful
        audio_files = list(subset_dir.glob("*/*.flac"))
        transcript_files = list(subset_dir.glob("*/*.trans.txt"))
        
        if not audio_files:
            raise RuntimeError(f"Extraction failed - no audio files found in {subset_dir}")
        
        print(f"📊 Extracted {len(audio_files):,} audio files and {len(transcript_files):,} transcript files")
        
        # Clean up tar file to save space only after successful extraction
        tar_file.unlink()
        print(f"🗑️  Removed {tar_file} to save space")
        
        return subset_dir
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to extract LibriSpeech {subset}: {e}")
    except Exception as e:
        raise RuntimeError(f"Error processing LibriSpeech {subset}: {e}")


def load_librispeech_local(subset: str, cache_root: Path, max_samples: Optional[int] = None, streaming: bool = False) -> Tuple[HFDataset, int]:
    """
    Load LibriSpeech from locally downloaded files.
    
    Args:
        subset: LibriSpeech subset name
        cache_root: Root directory where LibriSpeech is cached
        max_samples: Maximum number of samples to load
        streaming: Whether to create a streaming dataset (for large datasets)
        
    Returns:
        Tuple of (HuggingFace Dataset, number of samples)
    """
    subset_dir = download_librispeech_subset(subset, cache_root)
    
    print(f"📊 Loading LibriSpeech {subset} from {subset_dir}")
    
    # Collect all audio files and transcripts
    records = []
    total_files = 0
    
    # First pass: count total files if needed for streaming
    if streaming:
        for speaker_dir in subset_dir.glob("*"):
            if speaker_dir.is_dir():
                for chapter_dir in speaker_dir.glob("*"):
                    if chapter_dir.is_dir():
                        total_files += len(list(chapter_dir.glob("*.flac")))
    
    sample_count = 0
    for speaker_dir in sorted(subset_dir.glob("*")):
        if not speaker_dir.is_dir():
            continue
            
        for chapter_dir in sorted(speaker_dir.glob("*")):
            if not chapter_dir.is_dir():
                continue
                
            # Find transcript file
            transcript_file = chapter_dir / f"{speaker_dir.name}-{chapter_dir.name}.trans.txt"
            if not transcript_file.exists():
                print(f"⚠️  Warning: No transcript file found for {chapter_dir}")
                continue
            
            # Load transcripts
            transcripts = {}
            with transcript_file.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        audio_id, text = parts
                        transcripts[audio_id] = text
            
            # Process FLAC files
            for flac_file in sorted(chapter_dir.glob("*.flac")):
                audio_id = flac_file.stem
                if audio_id in transcripts:
                    records.append({
                        "audio": str(flac_file),
                        "text": transcripts[audio_id],
                        "speaker_id": int(speaker_dir.name),
                        "chapter_id": int(chapter_dir.name),
                        "utterance_id": audio_id,
                    })
                    sample_count += 1
                    
                    if max_samples and sample_count >= max_samples:
                        break
            
            if max_samples and sample_count >= max_samples:
                break
        
        if max_samples and sample_count >= max_samples:
            break
    
    num_samples = len(records)
    print(f"📊 Loaded {num_samples:,} samples from LibriSpeech {subset}")
    
    # Create dataset from JSON lines for HuggingFace compatibility
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for record in records:
            json.dump(record, f)
            f.write('\n')
        temp_file = f.name
    
    try:
        # Load using HuggingFace datasets with streaming support
        ds = load_dataset('json', data_files=temp_file, split='train', streaming=streaming)
        ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
        return ds, num_samples
    finally:
        # Clean up temp file
        Path(temp_file).unlink(missing_ok=True)

# ----------------------- DATASET REGISTRY ----------------------------
# ────────────────────────── NOTSOFAR helpers ──────────────────────────


# ---------------------------------------------------------------------
# NEW – smart downloader that understands the public Azure hierarchy
# ---------------------------------------------------------------------
def _resolve_notsofar_split(split: str) -> Tuple[str, str]:
    """
    Map HF-style split names (train/dev/validation/test) to the Azure
    directory and its *current* default version-tag.  Update the dict
    when MSFT announce a newer tag.
    """
    _LATEST = {
        "train":      ("train_set", "240825.1_train"),      # :contentReference[oaicite:4]{index=4}
        "dev":        ("dev_set",   "240825.1_dev1"),       # :contentReference[oaicite:5]{index=5}
        "validation": ("dev_set",   "240825.1_dev1"),
        "test":       ("eval_set",  "240825.1_eval_full_with_GT"),  # gold text released post-challenge :contentReference[oaicite:6]{index=6}
    }
    if split not in _LATEST:
        raise ValueError(f"Unknown NOTSOFAR split: {split!r}")
    return _LATEST[split]
def download_notsofar_transcripts(
    split: str,
    dest_root: Path,
    version: Optional[str] = None,
) -> Path:
    """
    Download NOTSOFAR-1 recorded-meeting *subset* (train/dev/eval) with AzCopy.

    Expected env var:
        NOTSOFAR_SAS_URL → e.g. 'https://notsofarsa.blob.core.windows.net/benchmark-datasets'

    If the subset already exists (non-empty folder) the download step is skipped.
    Returns: Path of the local subset (…/<dest_root>/<subset>)
    """
    local_dir = dest_root / split
    if any(local_dir.glob("*/*.wav")):   # already populated
        return local_dir

    # Public container root – works for every split that already has GT
    sas_url = os.environ.get(
        "NOTSOFAR_SAS_URL",
        "https://notsofarsa.blob.core.windows.net/benchmark-datasets",
    )

    subset_name, default_ver = _resolve_notsofar_split(split)
    ver = (
        version
        or os.environ.get("NOTSOFAR_VERSION")  # user override
        or default_ver                         # baked-in latest
    )

    src_url = f"{sas_url}/{subset_name}/{ver}/MTG"
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"⌛ Downloading NOTSOFAR-1 '{split}' subset to {local_dir} …")
    try:
        # --overwrite keeps newer, --check-md5 fails if corrupted
        subprocess.run(
        [
                 "azcopy",
                 "copy",
                 src_url,
                 str(local_dir),
                 "--recursive",
                 "--overwrite",
                 "ifSourceNewer",
                 "--check-md5",
                 "FailIfDifferent",
             ],
            check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "AzCopy not found. Install it from https://aka.ms/azcopy and ensure it’s on $PATH."
        )
    print("✅  Download finished.")
    return local_dir


def _attach_notsofar_transcripts(ds, subset_dir: Path):
    """
    Add a *text* column to the NOTSOFAR dataset by reading the
    reference `transcription.json` / `.stm` files we just downloaded.
    Works in streaming mode: we build a lazy lookup dict.
    """
    def _load_meeting_transcripts(meeting_dir: Path):
        # Try JSON first (baseline format) then .stm
        json_file = meeting_dir / "transcription.json"
        if json_file.exists():
            with json_file.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
            for utt in data.get("utterances", []):
                yield utt["segment_id"], utt["segment_text"]
        else:
            # fallback: scan .stm
            for stm in meeting_dir.glob("transcriptions/*.stm"):
                with stm.open("r", encoding="utf-8") as fp:
                    for line in fp:
                        parts = line.strip().split(maxsplit=6)
                        if len(parts) == 7:
                            utt_id = parts[1]
                            text   = parts[6]
                            yield utt_id, text

    # Build a mapping {utterance_id → transcript}
    print("📜 Building transcript index …")
    id2txt = {}
    for mtg_dir in subset_dir.glob("*"):
        for utt_id, txt in _load_meeting_transcripts(mtg_dir):
            id2txt[utt_id] = txt
    print(f"↳ indexed {len(id2txt):,} utterances")

    def _add_text(ex):
        utt_path = ex["audio"]["path"]
        utt_id   = Path(utt_path).stem         # pattern: <session>_<utt-id>.wav
        return {"text": id2txt.get(utt_id, "")}

    if isinstance(ds, IterableDataset):
        return ds.map(_add_text)
    return ds.map(_add_text, num_proc=1)



def _parse_librispeech_subset(label: str) -> Tuple[str, str]:
    # Normalize label: convert dots to hyphens for LibriSpeech naming convention
    normalized_label = label.replace(".", "-")
    
    mapping = {
        "test-clean": ("clean", "test"),
        "test-other": ("other", "test"),
        "dev-clean": ("clean", "validation"),
        "dev-other": ("other", "validation"),
        "train-clean-100": ("clean", "train.100"),
        "train-clean-360": ("clean", "train.360"),
        "train-other-500": ("other", "train.500"),
    }
    
    if normalized_label not in mapping:
        available = list(mapping.keys())
        raise ValueError(f"Unknown LibriSpeech subset: '{label}'. Available subsets: {available}")
    
    return mapping[normalized_label]


REGISTRY = {
    # Existing benchmarks
    "librispeech": dict(
        local=True,  # Use direct download instead of HF
        subset_parser=_parse_librispeech_subset,
        audio="audio",
        text="text",
        def_subset="test-clean",
        sr=16_000,
    ),
    "wham": dict(
        hf="nguyenvulebinh/wham",
        audio="audio",
        text=None,
        def_subset="train",
        sr=16_000,
    ),
    "ami": dict(
        hf="edinburghcstr/ami",
        config="ihm",
        audio="audio",
        text="text",
        def_subset="train",
        sr=16_000,
    ),
    "gigaspeech": dict(
        hf="speechcolab/gigaspeech",
        audio="audio",
        text="text",
        def_subset="test_xl",
        sr=16_000,
    ),
    # ---------------- CHiME‑8 additions ----------------
    # NOTSOFAR‑1 (HF release) – dev/train sets contain transcripts.
    "chime8_notsofar1": dict(
        hf="microsoft/NOTSOFAR",  # transcripts in dev/train splits
        audio="audio",
        text="transcript",  # column present in dev/train (empty in eval)
        def_subset="dev",  # dev, train, test
        sr=16_000,
    ),
    # Place‑holders for other CHiME‑8 DASR scenarios. Require local prep via chime-utils
    "chime8_chime6": dict(
        local=True,
        def_subset="dev",
        sr=16_000,
    ),
    "chime8_mixer6": dict(
        local=True,
        def_subset="dev",
        sr=16_000,
    ),
    "chime8_dipco": dict(
        local=True,
        def_subset="dev",
        sr=16_000,
    ),
}


# --------------------------------- UTILS ---------------------------------- 

def subset_has_transcripts(ds, text_col="text", *, probe=50) -> bool:
    """
    Return True if at least one of the first `probe` samples
    has a non-empty transcript string.
    Works for both Dataset and IterableDataset.
    """
    if text_col not in ds.column_names:
        return False
    if isinstance(ds, IterableDataset):
        return any(len(ex[text_col].strip()) > 0 for ex in ds.take(probe))
    # in-memory Dataset
    sample = ds[text_col][:probe]
    return any(len(s.strip()) > 0 for s in sample)


# ---------------------------- CLI ------------------------------------
parser = argparse.ArgumentParser(description="Evaluate Kimi‑Audio‑7B‑Instruct on ASR benchmarks (now incl. CHiME‑8)")
parser.add_argument(
    "--dataset",
    choices=list(REGISTRY.keys()),
    default="librispeech",
)
parser.add_argument("--subset", type=str, help="Dataset subset / scenario (e.g. 'test-clean' or 'notsofar1_dev')")
parser.add_argument("--config", type=str)
parser.add_argument("--streaming", action="store_true")
parser.add_argument("--max_samples", type=int)
parser.add_argument("--wandb_project", default="kimi-audio-multi-eval")
parser.add_argument("--wandb_run_name")
# new:
parser.add_argument(
    "--notsofar_version",
    type=str,
    help="Optional NOTSOFAR tag like '240825.1_dev1' "
         "(overrides the built-in defaults)",
)
parser.add_argument(
    "--librispeech_use_hf", 
    action="store_true",
    help="Use HuggingFace automatic download for LibriSpeech instead of direct download "
         "(may fail for large datasets due to 30GB+ size)",
)
# Multi-GPU support
parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use (1 or 2)")
parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed processing")
args = parser.parse_args()

# -------------------------- MULTI-GPU SETUP --------------------------
def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # When using CUDA_VISIBLE_DEVICES, the visible GPU is always device 0
    torch.cuda.set_device(0)

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

# Set up multi-GPU if requested
world_size = args.num_gpus
rank = args.local_rank
is_distributed = world_size > 1

if is_distributed:
    setup_distributed(rank, world_size)
    device = torch.cuda.current_device()
    print(f"🔧 Initialized process {rank}/{world_size} on GPU {device}")
else:
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using single GPU: {device}")

# Only rank 0 handles W&B and main coordination
is_main_process = rank == 0

# Update LibriSpeech registry based on command line argument
if args.librispeech_use_hf:
    REGISTRY["librispeech"] = dict(
        hf="librispeech_asr",
        subset_parser=_parse_librispeech_subset,
        audio="audio",
        text="text",
        def_subset="test-clean",
        sr=16_000,
    )
    if is_main_process:
        print("🔄 Using HuggingFace automatic download for LibriSpeech (may be slow/fail for large datasets)")
else:
    if is_main_process:
        cache_dir = get_librispeech_cache_dir()
        print(f"📁 Using direct LibriSpeech download (cached to: {cache_dir})")

# -------------------------- W&B RESUME -------------------------------
RESUME_FILE = Path(".wandb_run_id")
prev_id = RESUME_FILE.read_text().strip() if RESUME_FILE.exists() else None

# -------------------------- PARQUET SETUP ----------------------------
# Create output directory for parquet files
PARQUET_DIR = Path("batch_mistakes_parquet")
PARQUET_DIR.mkdir(exist_ok=True)
batch_counter = 0

def save_batch_mistakes_to_parquet(batch_mistakes, batch_id, run_id, rank=0):
    """Save batch mistakes to a parquet file"""
    if not batch_mistakes:
        return None
    
    df = pd.DataFrame(batch_mistakes)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mistakes_batch_{batch_id:04d}_{timestamp}_{run_id[:8]}_rank{rank}.parquet"
    filepath = PARQUET_DIR / filename
    df.to_parquet(filepath, index=False)
    return filepath

# W&B initialization - only on main process
run = None
if is_main_process:
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        id=prev_id,
        resume="allow",
        config={**vars(args), "model": "moonshotai/Kimi-Audio-7B-Instruct"},
    )
    if prev_id is None:
        RESUME_FILE.write_text(run.id)
    print(
        f"\nResume with: WANDB_RUN_ID={run.id} WANDB_RESUME=allow python "
        + " ".join(map(shlex.quote, sys.argv))
        + "\n"
    )
else:
    # Non-main processes create a dummy run object for compatibility
    class DummyRun:
        def __init__(self):
            self.id = "dummy_run"
        def log(self, *args, **kwargs):
            pass
        def finish(self, *args, **kwargs):
            pass
    run = DummyRun()

# ---------------------------- MODEL ----------------------------------
class MultiGPUKimiAudio:
    """Wrapper for multi-GPU KimiAudio inference"""
    def __init__(self, model_path, load_detokenizer=True, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        self.model = KimiAudio(model_path=model_path, load_detokenizer=load_detokenizer)
        
    def generate(self, chats, **kwargs):
        """Generate with the model - each GPU handles its own subset"""
        return self.model.generate(chats, **kwargs)

# Initialize model
if is_main_process:
    print("🤖 Loading KimiAudio model...")
model = MultiGPUKimiAudio(
    model_path="moonshotai/Kimi-Audio-7B-Instruct", 
    load_detokenizer=True,
    rank=rank,
    world_size=world_size
)

SAMPLING = dict(
    audio_temperature=0.8,
    audio_top_k=10,
    text_temperature=0.0,
    text_top_k=5,
    audio_repetition_penalty=1.0,
    audio_repetition_window_size=64,
    text_repetition_penalty=1.0,
    text_repetition_window_size=16,
)


# -------------------------- LOADERS ----------------------------------

def _safe_num_examples(builder, split):
    splits = getattr(builder.info, "splits", None)
    return splits[split].num_examples if splits and split in splits else None


def safe_cast(ds, column, sr):
    if isinstance(ds, IterableDataset):
        # lazy, on-the-fly resample as you iterate
        return ds.map(
            lambda ex: {column: {"array": ex[column]["array"],
                                 "sampling_rate": sr,
                                 "path": ex[column]["path"]}},
            # num_proc=1  # keep it streaming-friendly
        )
    return ds.cast_column(column, Audio(sampling_rate=sr))

def _load_local_chime8(subset: str, *, max_samples: Optional[int] = None):
    """Load CHiME‑8 DASR datasets prepared with `chime-utils dgen …`.

    The environment variable **CHIME8_ROOT** must point to the folder created
    by `chime-utils dgen dasr …`, which contains sub‑folders `chime6/`,
    `mixer6/`, `dipco/`, `notsofar1/` etc.  For each scenario we expect
    `wav` files under `audio/` and transcripts in `transcript.txt` or Kaldi
    style `.stm` in `transcriptions/`.

    Only a minimal implementation is provided – it scans for `*.wav` files and
    a sibling `*.txt` transcript with the same stem.  Feel free to replace
    this with a more robust Lhotse‑based loader.
    """
    import glob, itertools
    import json

    root = Path(os.environ.get("CHIME8_ROOT", "./chime8_dasr")).expanduser()
    scenario_dir = root / subset
    if not scenario_dir.exists():
        raise FileNotFoundError(
            f"Expected scenario folder {scenario_dir} (set CHIME8_ROOT or --subset correctly)."
        )

    wavs = sorted(scenario_dir.rglob("*.wav"))
    if max_samples is not None:
        wavs = wavs[: max_samples]

    records = []
    for w in wavs:
        txt = w.with_suffix(".txt")
        if not txt.exists():
            # fallback – allow empty transcript (evaluation‑set)
            transcript = ""
        else:
            transcript = txt.read_text(encoding="utf‑8").strip()
        records.append({"audio": str(w), "text": transcript})

    # wrap into HF dataset for compatibility
    ds = load_dataset("json", data_files={"data": records}, split="data")
    ds = safe_cast(ds, "audio", 16_000)
    return ds, None if max_samples else len(records)


def load_corpus(key: str, subset: Optional[str] = None, config_override: Optional[str] = None, *, streaming=False, max_samples=None):
    """Generic loader factory with CHiME‑8 integration."""

    if key.startswith("chime8") and REGISTRY[key].get("local"):
        # local (chime-utils) scenario
        subset = subset or REGISTRY[key]["def_subset"]
        return _load_local_chime8(subset, max_samples=max_samples)

    if key == "librispeech" and REGISTRY[key].get("local"):
        # local (direct download) scenario
        cache_root = get_librispeech_cache_dir()
        return load_librispeech_local(subset or REGISTRY[key]["def_subset"],
                                     cache_root=cache_root,
                                     max_samples=max_samples,
                                     streaming=streaming)

    cfg = REGISTRY[key]

    # CHiME‑8 NOTSOFAR – normal HF path
    # ---------- CHiME-8 NOTSOFAR ---------- 
    if key == "chime8_notsofar1":
        # If the caller did *not* request a specific split, or asked for "auto",
        # pick the first split that actually contains transcripts.
        requested = (subset or cfg["def_subset"]).lower()
        if requested in {None, "", "auto"}:
            for cand in ("train", "dev", "validation"):
                tmp_ds, _ = load_corpus(key, cand, config_override,
                                        streaming=streaming, max_samples=10)
                if subset_has_transcripts(tmp_ds):
                    subset = cand
                    break
            else:
                raise ValueError("No NOTSOFAR-1 split with transcripts found.")
        else:
            subset = requested             # honour explicit --subset

        split_name = subset               # train/dev/test/etc.
        builder  = load_dataset_builder(cfg["hf"], trust_remote_code=True)
        num      = _safe_num_examples(builder, split_name)
        ds       = load_dataset(cfg["hf"],
                                split=split_name,
                                streaming=streaming,
                                trust_remote_code=True)
        # ── inject transcripts automatically if HF split has none ─────────
        if "text" not in ds.column_names or not subset_has_transcripts(ds):
            cache_root = Path(
                os.environ.get("NOTSOFAR_CACHE", "~/.cache/notsofar1")
            ).expanduser()
            subset_dir = download_notsofar_transcripts(
                split=subset,
                dest_root=cache_root,
                version=args.notsofar_version,
            )
            ds = _attach_notsofar_transcripts(ds, subset_dir)
        else:
            ds = ds.rename_column(cfg["text"], "text")   # HF kept refs (rare)

        # audio feature stays unchanged – NOTSOFAR already at 16 kHz
        if max_samples:
            ds = ds.take(max_samples)
            num = min(num or max_samples, max_samples)
        return ds, num

    # ---------- General HF dataset handling ----------
    # Handle other datasets in the registry (librispeech, ami, wham, etc.)
    subset = subset or cfg["def_subset"]
    config = config_override or cfg.get("config")
    
    # Handle LibriSpeech subset parsing
    if key == "librispeech" and "subset_parser" in cfg:
        config_name, split_name = cfg["subset_parser"](subset)
        split = split_name
        config = config_name
    else:
        split = subset
    
    # Load dataset builder to get metadata
    builder = load_dataset_builder(cfg["hf"], name=config, trust_remote_code=True)
    num = _safe_num_examples(builder, split)
    
    # Load the actual dataset
    ds = load_dataset(
        cfg["hf"],
        name=config,
        split=split,
        streaming=streaming,
        trust_remote_code=True
    )
    
    # Rename text column if needed
    if cfg.get("text") and cfg["text"] != "text":
        ds = ds.rename_column(cfg["text"], "text")
    
    # Cast audio to correct sampling rate
    if cfg.get("audio"):
        ds = safe_cast(ds, cfg["audio"], cfg.get("sr", 16_000))
    
    # Apply max_samples limit
    if max_samples:
        if streaming:
            ds = ds.take(max_samples)
        else:
            ds = ds.select(range(min(max_samples, len(ds))))
        num = min(num or max_samples, max_samples)
    
    return ds, num

# -------------------------- METRICS ----------------------------------
normalizer = BasicTextNormalizer()
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

_clean = lambda s: normalizer(s).strip()

# colour diff (unchanged)

def _diff(r: str, h: str) -> str:
    r, w = r.split(), h.split()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, r, w).get_opcodes():
        if tag == "equal":
            out += w[j1:j2]
        elif tag == "insert":
            out += [f"{Fore.GREEN}{x}{Style.RESET_ALL}" for x in w[j1:j2]]
        elif tag == "delete":
            out += [f"{Fore.RED}<{x}>{Style.RESET_ALL}" for x in r[i1:i2]]
        else:
            out += [f"{Fore.YELLOW}{x}{Style.RESET_ALL}" for x in w[j1:j2]]
    return " ".join(out)

# --------------------------- LOAD DATA -------------------------------
if is_main_process:
    print("📊 Loading dataset…")
ds, ds_len = load_corpus(
    args.dataset,
    args.subset,
    args.config,
    streaming=args.streaming,
    max_samples=args.max_samples,
)

# Split dataset across GPUs
if is_distributed and ds_len:
    # Calculate samples per GPU
    samples_per_gpu = ds_len // world_size
    start_idx = rank * samples_per_gpu
    end_idx = start_idx + samples_per_gpu if rank < world_size - 1 else ds_len
    
    if hasattr(ds, 'select'):
        # For in-memory datasets
        ds = ds.select(range(start_idx, end_idx))
        local_ds_len = end_idx - start_idx
    else:
        # For streaming datasets
        ds = ds.skip(start_idx).take(end_idx - start_idx)
        local_ds_len = end_idx - start_idx
    
    if is_main_process:
        print(f"📊 Dataset split across {world_size} GPUs:")
        for i in range(world_size):
            gpu_start = i * samples_per_gpu
            gpu_end = gpu_start + samples_per_gpu if i < world_size - 1 else ds_len
            print(f"   GPU {i}: samples {gpu_start:,} - {gpu_end:,} ({gpu_end-gpu_start:,} samples)")
else:
    local_ds_len = ds_len

if is_main_process:
    print(f"📊 Dataset ready (total≈{ds_len:,}, local≈{local_ds_len:,}).")

# ----------------------- MAIN EVAL LOOP ------------------------------
results_p, results_r = [], []
all_mistakes, batch_mistakes, failures = [], [], []
ser_errors = 0  # sentence error count

# Global counters for distributed processing
global_idx_offset = rank * (ds_len // world_size) if is_distributed else 0

try:
    desc = f"Transcribing (GPU {rank})" if is_distributed else "Transcribing"
    for idx, ex in enumerate(tqdm(ds, total=local_ds_len, desc=desc)):
        # skip ultra‑short
        if len(ex["audio"]["array"]) < 0.2 * ex["audio"]["sampling_rate"]:
            failures.append({"index": idx, "reason": "too_short"})
            results_p.append(""); results_r.append(_clean(ex["text"]))
            ser_errors += 1
            continue
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, ex["audio"]["array"], ex["audio"]["sampling_rate"])
            wav_path = tmp.name
        try:
            try:
                _, out = model.generate(
                    [
                        {"role": "user", "message_type": "text", "content": "Please transcribe the following english audio:"},
                        {"role": "user", "message_type": "audio", "content": wav_path},
                    ],
                    **SAMPLING,
                    output_type="text",
                )
            except RuntimeError as e:
                if "expected a non-empty list" in str(e):
                    failures.append({"index": idx, "reason": "whisper_empty"}); out = ""
                else:
                    raise
        finally:
            os.remove(wav_path)

        pred, ref = _clean(out), _clean(ex["text"])
        results_p.append(pred); results_r.append(ref)
        sent_err = int(pred != ref); ser_errors += sent_err

        if pred != ref:
            entry = dict(
                index=idx + global_idx_offset,  # Use global index
                reference=ref,
                prediction=pred,
                diff=_diff(ref, pred),
                sample_wer=wer_metric.compute(predictions=[pred], references=[ref]),
                sample_cer=cer_metric.compute(predictions=[pred], references=[ref]),
                gpu_rank=rank,  # Track which GPU processed this
            )
            all_mistakes.append(entry); batch_mistakes.append(entry)

        # ---- periodic logging ----
        if (idx + 1) % 250 == 0:
            # Save parquet files on each GPU
            if batch_mistakes:
                parquet_file = save_batch_mistakes_to_parquet(batch_mistakes, batch_counter, run.id, rank)
                if parquet_file:
                    print(f"💾 GPU {rank}: Saved {len(batch_mistakes)} mistakes to {parquet_file}")
                batch_mistakes = []
                batch_counter += 1
            
            # Only main process handles W&B logging and metric aggregation
            if is_main_process:
                # Gather metrics from all GPUs if distributed
                if is_distributed:
                    # Collect predictions and references from all GPUs
                    all_predictions = [None] * world_size
                    all_references = [None] * world_size
                    all_errors = [None] * world_size
                    all_failures = [None] * world_size
                    
                    dist.all_gather_object(all_predictions, results_p)
                    dist.all_gather_object(all_references, results_r)
                    dist.all_gather_object(all_errors, ser_errors)
                    dist.all_gather_object(all_failures, len(failures))
                    
                    # Flatten lists
                    global_predictions = [item for sublist in all_predictions for item in sublist]
                    global_references = [item for sublist in all_references for item in sublist]
                    global_ser_errors = sum(all_errors)
                    global_failures = sum(all_failures)
                    global_samples = len(global_predictions)
                else:
                    global_predictions = results_p
                    global_references = results_r
                    global_ser_errors = ser_errors
                    global_failures = len(failures)
                    global_samples = idx + 1
                
                if global_predictions and global_references:
                    running_wer = wer_metric.compute(predictions=global_predictions, references=global_references)
                    running_cer = cer_metric.compute(predictions=global_predictions, references=global_references)
                    running_ser = global_ser_errors / global_samples
                    
                    metrics = {
                        "running_wer": running_wer,
                        "running_cer": running_cer,
                        "running_ser": running_ser,
                        "seen_samples": global_samples,
                        "failed": global_failures,
                        "num_gpus": world_size,
                    }
                    wandb.log(metrics)
except KeyboardInterrupt:
    print(f"🛑 GPU {rank}: Interrupted – resume later!")
    if is_main_process:
        wandb.finish(exit_code=255)
    cleanup_distributed()
    sys.exit(130)

# ------------------------- FINAL METRICS -----------------------------
# Save any remaining batch_mistakes to parquet on each GPU
if batch_mistakes:
    parquet_file = save_batch_mistakes_to_parquet(batch_mistakes, batch_counter, run.id, rank)
    if parquet_file:
        print(f"💾 GPU {rank}: Saved final {len(batch_mistakes)} mistakes to {parquet_file}")

# Aggregate final results from all GPUs
if is_main_process:
    if is_distributed:
        # Gather all results from all GPUs
        all_predictions = [None] * world_size
        all_references = [None] * world_size
        all_errors = [None] * world_size
        all_failures = [None] * world_size
        all_mistakes_lists = [None] * world_size
        
        dist.all_gather_object(all_predictions, results_p)
        dist.all_gather_object(all_references, results_r)
        dist.all_gather_object(all_errors, ser_errors)
        dist.all_gather_object(all_failures, len(failures))
        dist.all_gather_object(all_mistakes_lists, all_mistakes)
        
        # Flatten and combine all results
        final_predictions = [item for sublist in all_predictions for item in sublist]
        final_references = [item for sublist in all_references for item in sublist]
        final_ser_errors = sum(all_errors)
        final_failures = sum(all_failures)
        final_all_mistakes = [item for sublist in all_mistakes_lists for item in sublist]
    else:
        final_predictions = results_p
        final_references = results_r
        final_ser_errors = ser_errors
        final_failures = len(failures)
        final_all_mistakes = all_mistakes
    
    if final_predictions and final_references:
        final_wer = wer_metric.compute(predictions=final_predictions, references=final_references)
        final_cer = cer_metric.compute(predictions=final_predictions, references=final_references)
        final_ser = final_ser_errors / len(final_predictions)
        
        wandb.log({
            "final_wer": final_wer, 
            "final_cer": final_cer, 
            "final_ser": final_ser, 
            "failed": final_failures,
            "total_samples": len(final_predictions),
            "num_gpus": world_size
        })
        
        print(f"🎯 Final Results:")
        print(f"   WER: {final_wer:.4f}")
        print(f"   CER: {final_cer:.4f}")
        print(f"   SER: {final_ser:.4f}")
        print(f"   Failed: {final_failures}")
        print(f"   Total samples: {len(final_predictions):,}")
        print(f"   GPUs used: {world_size}")
        
        # Upload consolidated error table
        if final_all_mistakes:
            tbl = wandb.Table(columns=list(final_all_mistakes[0].keys()))
            for m in final_all_mistakes:
                tbl.add_data(*m.values())
            wandb.log({"error_table": tbl})
    
    wandb.finish()
    if RESUME_FILE.exists():
        RESUME_FILE.unlink(missing_ok=True)

# Clean up distributed training
cleanup_distributed()
