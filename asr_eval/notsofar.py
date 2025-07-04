# asr_eval/notsofar.py
"""NOTSOFAR‑1 specific download helpers."""
from __future__ import annotations
import os, subprocess, json
from pathlib import Path
from typing import Tuple, Optional
from asr_eval.NOTSOFAR1_Challenge.utils.azure_storage import download_meeting_subset

__all__ = [
    "_resolve_notsofar_split",
    "download_notsofar_transcripts",
    "attach_notsofar_transcripts",
]

_LATEST = {
    "train": ("train_set", "240825.1_train"),
    "dev": ("dev_set", "240825.1_dev1"),
    "validation": ("dev_set", "240825.1_dev1"),
    "test": ("eval_set", "240825.1_eval_full_with_GT"),
}


def _resolve_notsofar_split(split: str) -> Tuple[str, str]:
    try:
        return _LATEST[split]
    except KeyError:
        raise ValueError(f"Unknown NOTSOFAR split: {split!r}")


def download_notsofar_transcripts(split: str, dest_root: Path, *, version: Optional[str] = None) -> Path:
    # Caller may give us *either* the storage-account container
    # or the already-expanded meetings folder.  Detect the latter
    # and avoid concatenating paths twice – AzCopy must see the
    # *deep* folder, otherwise it will touch the container root and
    # trigger a 401/403 because anonymous listing is disabled.
    raw_base = os.environ.get(
        "NOTSOFAR_SAS_URL",
        "https://notsofarsa.blob.core.windows.net/benchmark-datasets",
    ).rstrip("/")

    subset_name, default_ver = _resolve_notsofar_split(split)
    ver = version or os.environ.get("NOTSOFAR_VERSION") or default_ver

    local_dir = dest_root / split
    local_dir.mkdir(parents=True, exist_ok=True)
    if any(local_dir.glob("*/*.wav")):
        return local_dir  # already cached

    # ------------------------------------------------------------------
    # 1) Preferred path: use the high-level helper that comes with the
    #    NOTSOFAR-1 repo.  It hides all the messy SAS / retry logic.
    # ------------------------------------------------------------------
    try:
        download_meeting_subset(
            subset_name=subset_name,
            version=ver,
            destination_dir=str(local_dir),
        )
    except ImportError:
        # --------------------------------------------------------------
        # 2) Fallback: keep the old AzCopy logic, but add the wildcard
        #    so AzCopy never tries to *list* the container (which fails
        #    anonymously) – it just downloads the blobs directly.
        # --------------------------------------------------------------

        raw_base = os.environ.get(
            "NOTSOFAR_SAS_URL",
            "https://notsofarsa.blob.core.windows.net/benchmark-datasets",
        ).rstrip("/")
        src_url = f"{raw_base}/{subset_name}/{ver}/MTG/*"

        cmd = [
            "azcopy",
            "copy",
            src_url,
            str(local_dir),
            "--recursive",
            "--overwrite=ifSourceNewer",
            "--check-md5=FailIfDifferent",
        ]
        subprocess.run(cmd, check=True)
    return local_dir


def attach_notsofar_transcripts(ds, subset_dir: Path):
    """Inject transcripts into a NOTSOFAR HF dataset in‑place."""
    from datasets import IterableDataset

    def _load_meeting_transcripts(meeting_dir: Path):
        json_file = meeting_dir / "transcription.json"
        if json_file.exists():
            data = json.loads(json_file.read_text("utf-8"))
            for utt in data.get("utterances", []):
                yield utt["segment_id"], utt["segment_text"]
        else:
            for stm in meeting_dir.glob("transcriptions/*.stm"):
                for line in stm.read_text("utf-8").splitlines():
                    parts = line.split(maxsplit=6)
                    if len(parts) == 7:
                        yield parts[1], parts[6]

    id2txt = {}
    for mtg_dir in subset_dir.glob("*"):
        for utt_id, txt in _load_meeting_transcripts(mtg_dir):
            id2txt[utt_id] = txt

    def _add_text(example):
        from pathlib import Path as _P
        utt_id = _P(example["audio"]["path"]).stem
        return {"text": id2txt.get(utt_id, "")}

    if isinstance(ds, IterableDataset):
        return ds.map(_add_text)
    return ds.map(_add_text, num_proc=1)