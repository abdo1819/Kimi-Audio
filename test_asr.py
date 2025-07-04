#!/usr/bin/env python
"""
UPDATED – Whisper BasicTextNormalizer + CER & SER + CHiME‑8 datasets
------------------------------------------------------------------
* **Normalization** now uses `transformers.models.whisper.english_normalizer.BasicTextNormalizer`,
  matching Whisper’s official preprocessing.
* Added **CER** (character error rate) via `evaluate.load("cer")`.
* Added **SER** (sentence/utterance error rate – a binary 0/1 per sample).
* **CHiME‑8 support** – new `chime8` dataset option with initial support for
  the NOTSOFAR‑1 dev/train sets hosted on Hugging Face (`microsoft/NOTSOFAR`).
  Place‑holders are included to plug in local prepared CHiME‑8 DASR data
  (CHiME‑6, Mixer‑6, DiPCo) generated with `chime-utils`.
* W&B logs include running and final **WER/CER/SER** plus the
  per‑250‑step *new mistakes* table as before.
"""

from __future__ import annotations

import os, re, csv, argparse, tempfile, difflib, sys, shlex
from pathlib import Path
import subprocess, json, hashlib, itertools
from typing import Tuple, Optional

import wandb
from tqdm import tqdm
import soundfile as sf
from datasets import load_dataset, load_dataset_builder, Audio, IterableDataset
import evaluate
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from kimia_infer.api.kimia import KimiAudio
from colorama import init as colorama_init, Fore, Style

colorama_init(autoreset=True)
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
    mapping = {
        "test-clean": ("clean", "test"),
        "test-other": ("other", "test"),
        "dev-clean": ("clean", "validation"),
        "dev-other": ("other", "validation"),
        "train-clean-100": ("clean", "train.100"),
        "train-clean-360": ("other", "train.360"),
        "train-other-500": ("other", "train.500"),
    }
    return mapping[label]


REGISTRY = {
    # Existing benchmarks
    "librispeech": dict(
        hf="librispeech_asr",
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
args = parser.parse_args()

# -------------------------- W&B RESUME -------------------------------
RESUME_FILE = Path(".wandb_run_id")
prev_id = RESUME_FILE.read_text().strip() if RESUME_FILE.exists() else None
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

# ---------------------------- MODEL ----------------------------------
model = KimiAudio(model_path="moonshotai/Kimi-Audio-7B-Instruct", load_detokenizer=True)
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
print("Loading dataset…")
ds, ds_len = load_corpus(
    args.dataset,
    args.subset,
    args.config,
    streaming=args.streaming,
    max_samples=args.max_samples,
)
print(f"Dataset ready (samples≈{ds_len}).")

# ----------------------- MAIN EVAL LOOP ------------------------------
results_p, results_r = [], []
all_mistakes, batch_mistakes, failures = [], [], []
ser_errors = 0  # sentence error count

try:
    for idx, ex in enumerate(tqdm(ds, total=ds_len, desc="Transcribing")):
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
                        {"role": "user", "message_type": "text", "content": "Please transcribe the following audio:"},
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
                index=idx,
                reference=ref,
                prediction=pred,
                diff=_diff(ref, pred),
                sample_wer=wer_metric.compute(predictions=[pred], references=[ref]),
                sample_cer=cer_metric.compute(predictions=[pred], references=[ref]),
            )
            all_mistakes.append(entry); batch_mistakes.append(entry)

        # ---- periodic logging ----
        if (idx + 1) % 250 == 0:
            running_wer = wer_metric.compute(predictions=results_p, references=results_r)
            running_cer = cer_metric.compute(predictions=results_p, references=results_r)
            running_ser = ser_errors / (idx + 1)
            metrics = {
                "running_wer": running_wer,
                "running_cer": running_cer,
                "running_ser": running_ser,
                "seen_samples": idx + 1,
                "failed": len(failures),
            }
            if batch_mistakes:
                tbl = wandb.Table(columns=list(batch_mistakes[0].keys()))
                for m in batch_mistakes:
                    tbl.add_data(*m.values())
                metrics["new_mistakes"] = tbl
                batch_mistakes = []
            wandb.log(metrics)
except KeyboardInterrupt:
    print("Interrupted – resume later!")
    wandb.finish(exit_code=255)
    sys.exit(130)

# ------------------------- FINAL METRICS -----------------------------
final_wer = wer_metric.compute(predictions=results_p, references=results_r)
final_cer = cer_metric.compute(predictions=results_p, references=results_r)
final_ser = ser_errors / len(results_p)
wandb.log({"final_wer": final_wer, "final_cer": final_cer, "final_ser": final_ser, "failed": len(failures)})

# upload full mistakes
# 'tbl' not defined
if all_mistakes:
    tbl = wandb.Table(columns=list(all_mistakes[0].keys()))
    for m in all_mistakes:
        tbl.add_data(*m.values())
    wandb.log({"error_table": tbl})

wandb.finish(); RESUME_FILE.unlink(missing_ok=True)
