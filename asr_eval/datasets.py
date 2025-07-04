# asr_eval/datasets.py
"""Generic corpus loader with CHiME‑8 integration."""
from __future__ import annotations
from typing import Optional, Tuple
from pathlib import Path
import os
from datasets import load_dataset, load_dataset_builder, Audio, IterableDataset

from .registry import REGISTRY
from .notsofar import download_notsofar_transcripts, attach_notsofar_transcripts

__all__ = ["load_corpus"]

# --- common helpers -------------------------------------------------

def _safe_num_examples(builder, split):
    splits = getattr(builder.info, "splits", None)
    return splits[split].num_examples if splits and split in splits else None


def subset_has_transcripts(ds, text_col="text", probe: int = 50) -> bool:
    if text_col not in ds.column_names:
        return False
    if isinstance(ds, IterableDataset):
        return any(len(ex[text_col].strip()) > 0 for ex in ds.take(probe))
    sample = ds[text_col][:probe]
    return any(len(s.strip()) > 0 for s in sample)


def safe_cast(ds, column, sr):
    if isinstance(ds, IterableDataset):
        return ds.map(lambda ex: {column: {**ex[column], "sampling_rate": sr}})
    return ds.cast_column(column, Audio(sampling_rate=sr))

# --- CHiME‑8 local --------------------------------------------------

def _load_local_chime8(subset: str, *, max_samples: Optional[int] = None):
    import glob, json
    from datasets import load_dataset

    root = Path(os.environ.get("CHIME8_ROOT", "./chime8_dasr")).expanduser()
    scenario_dir = root / subset
    wavs = sorted(scenario_dir.rglob("*.wav"))
    if max_samples:
        wavs = wavs[:max_samples]
    records = []
    for w in wavs:
        txt = w.with_suffix(".txt")
        transcript = txt.read_text("utf-8").strip() if txt.exists() else ""
        records.append({"audio": str(w), "text": transcript})
    ds = load_dataset("json", data_files={"data": records}, split="data")
    ds = safe_cast(ds, "audio", 16_000)
    return ds, None if max_samples else len(records)

# --- public API -----------------------------------------------------


def load_corpus(key: str, subset: Optional[str] = None, *, streaming=False, max_samples=None, notsofar_version=None):
    """Factory that wraps all dataset logic (NOTSOFAR, CHiME‑8, etc.)."""
    if key.startswith("chime8") and REGISTRY[key].get("local"):
        subset = subset or REGISTRY[key]["def_subset"]
        return _load_local_chime8(subset, max_samples=max_samples)

    cfg = REGISTRY[key]

    if key == "chime8_notsofar1":
        requested = (subset or cfg["def_subset"]).lower()
        subset = requested
        builder = load_dataset_builder(cfg["hf"], trust_remote_code=True)
        num = _safe_num_examples(builder, subset)
        ds = load_dataset(cfg["hf"], split=subset, streaming=streaming, trust_remote_code=True)
        if "text" not in ds.column_names or not subset_has_transcripts(ds):
            cache_root = Path(os.environ.get("NOTSOFAR_CACHE", "~/.cache/notsofar1")).expanduser()
            subset_dir = download_notsofar_transcripts(subset, cache_root, version=notsofar_version)
            ds = attach_notsofar_transcripts(ds, subset_dir)
        else:
            ds = ds.rename_column(cfg["text"], "text")
        if max_samples:
            ds = ds.take(max_samples)
            num = min(num or max_samples, max_samples)
        return ds, num

    # generic HF dataset
    ds = load_dataset(cfg["hf"], split=subset or cfg["def_subset"], streaming=streaming, trust_remote_code=True)
    ds = safe_cast(ds, cfg["audio"], cfg["sr"])
    return ds, None
