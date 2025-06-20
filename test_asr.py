#!/usr/bin/env python
"""
UPDATED *again* – robust length fetching **plus RESUME‑SUPPORT**
----------------------------------------------------------------
Adds automatic W&B‑run resumption when the script crashes or is
interrupted (e.g. Ctrl‑C):

**How it works**
1.  A tiny text file (`.wandb_run_id`) stores the current `run.id`.
2.  On start‑up the script looks for that file.  If it exists we call
    `wandb.init(id=…, resume="allow")`, otherwise we start a fresh run
    and write the new run‑ID to the file.
3.  The script prints a ready‑made shell command you can copy‑paste to
   resume the run (uses the `WANDB_RUN_ID` / `WANDB_RESUME` env‑vars).
4.  When the run finishes **successfully** the file is deleted, so the
   next invocation starts a new run.

Nothing else in your training / evaluation loop needs to change.
"""

from __future__ import annotations

import os, re, csv, argparse, tempfile, difflib, sys, shlex
from pathlib import Path

import wandb
from tqdm import tqdm
import soundfile as sf
from datasets import load_dataset, load_dataset_builder, Audio, IterableDataset
import evaluate
from kimia_infer.api.kimia import KimiAudio
from colorama import init as colorama_init, Fore, Style
colorama_init(autoreset=True)

# -----------------------------------------------------------------------------
# 1. CLI ARGUMENTS
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate Kimi‑Audio‑7B‑Instruct on ASR benchmarks")
parser.add_argument("--dataset", choices=["librispeech", "wham", "ami", "gigaspeech"], default="librispeech")
parser.add_argument("--subset", type=str)
parser.add_argument("--config", type=str)
parser.add_argument("--streaming", action="store_true")
parser.add_argument("--max_samples", type=int)
parser.add_argument("--wandb_project", default="kimi-audio-multi-eval")
parser.add_argument("--wandb_run_name")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# 2. RESUME‑AWARE W&B INIT
# -----------------------------------------------------------------------------
RESUME_FILE = Path(".wandb_run_id")
prev_id = RESUME_FILE.read_text().strip() if RESUME_FILE.exists() else None

run = wandb.init(
    project=args.wandb_project,
    name=args.wandb_run_name,
    id=prev_id,
    resume="allow",              # will stitch history if id exists
    config={**vars(args), "model": "moonshotai/Kimi-Audio-7B-Instruct"},
)

# Store the run‑id for future resumes (only when starting fresh)
if prev_id is None:
    RESUME_FILE.write_text(run.id)

# Print helper command users can copy‑paste to resume the run
resume_cmd = " ".join([
    f"WANDB_RUN_ID={run.id}",
    "WANDB_RESUME=allow",
    "python", *[shlex.quote(a) for a in sys.argv]
])
print("\n>>> To resume this run if interrupted, execute:\n" + resume_cmd + "\n")

# -----------------------------------------------------------------------------
# 3. LOAD MODEL
# -----------------------------------------------------------------------------
model = KimiAudio(model_path="moonshotai/Kimi-Audio-7B-Instruct", load_detokenizer=True)
SAMPLING = dict(audio_temperature=0.8, audio_top_k=10, text_temperature=0.0,
                text_top_k=5, audio_repetition_penalty=1.0, audio_repetition_window_size=64,
                text_repetition_penalty=1.0, text_repetition_window_size=16)

# -----------------------------------------------------------------------------
# 4. DATASET REGISTRY (unchanged)
# -----------------------------------------------------------------------------

def _parse_librispeech_subset(label):
    mapping = {
        "test-clean":("clean","test"),"test-other":("other","test"),
        "dev-clean":("clean","validation"),"dev-other":("other","validation"),
        "train-clean-100":("clean","train.100"),"train-clean-360":("other","train.360"),
        "train-other-500":("other","train.500")}
    if label not in mapping:
        raise ValueError(f"Unknown LibriSpeech subset {label}")
    return mapping[label]

REGISTRY = {
    "librispeech": dict(hf="librispeech_asr", subset_parser=_parse_librispeech_subset,
                         audio="audio", text="text", def_subset="test-clean", sr=16_000),
    "wham":        dict(hf="nguyenvulebinh/wham", audio="audio", text=None,
                         def_subset="train", sr=16_000),
    "ami":         dict(hf="edinburghcstr/ami", config="ihm", audio="audio", text="text",
                         def_subset="train", sr=16_000),
    "gigaspeech":  dict(hf="speechcolab/gigaspeech", audio="audio", text="text",
                         def_subset="test_xl", sr=16_000),
}

# -----------------------------------------------------------------------------
# 5. DATASET LOADER (unchanged except cosmetic)
# -----------------------------------------------------------------------------

def _safe_num_examples(builder, split):
    """Return num_examples or None when not available."""
    try:
        splits = getattr(builder.info, "splits", None)
        if splits and split in splits:
            return splits[split].num_examples
    except Exception:
        pass
    return None


def load_corpus(key, subset=None, config_override=None, *, streaming=False, max_samples=None):
    cfg = REGISTRY[key]
    subset = subset or cfg.get("def_subset")
    config_name = config_override or cfg.get("config")

    # Resolve HF names
    if key == "librispeech":
        config_name, split_name = cfg["subset_parser"](subset)
    else:
        split_name = subset

    # Builder for length (when possible)
    num = None
    if not streaming:
        kwargs = dict(trust_remote_code=True)
        if config_name is not None:
            kwargs["name"] = config_name
        builder = load_dataset_builder(cfg["hf"], **kwargs)
        num = _safe_num_examples(builder, split_name)

    # Load dataset
    kwargs_ds = dict(split=split_name, streaming=streaming, trust_remote_code=True)
    if config_name is not None:
        kwargs_ds["name"] = config_name
    ds = load_dataset(cfg["hf"], **kwargs_ds)

    # Truncate
    if max_samples is not None:
        ds = ds.take(max_samples)
        num = min(num or max_samples, max_samples)

    # Harmonise columns
    if cfg["audio"] != "audio":
        ds = ds.rename_column(cfg["audio"], "audio")
    if cfg["text"] and cfg["text"] != "text":
        ds = ds.rename_column(cfg["text"], "text")
    elif not cfg["text"]:
        ds = ds.map(lambda x: {"text":""}, remove_columns=[c for c in ds.column_names if c!="audio"], batched=False)

    ds = ds.cast_column("audio", Audio(sampling_rate=cfg["sr"]))

    if num is None and not isinstance(ds, IterableDataset):
        num = len(ds)
    return ds, num

# -----------------------------------------------------------------------------
# 6. HELPER FUNCTIONS (WER + DIFF) – unchanged
# -----------------------------------------------------------------------------
wer_metric = evaluate.load("wer")
_clean_re = re.compile(r"[^a-z0-9\s]")

def _clean(s):
    return re.sub(r"\s+"," ", _clean_re.sub("", s.lower().strip()))

def _diff(r,h):
    r,w = r.split(), h.split()
    out=[]
    for tag,i1,i2,j1,j2 in difflib.SequenceMatcher(None,r,w).get_opcodes():
        if tag=="equal": out+=w[j1:j2]
        elif tag=="insert": out+=[f"{Fore.GREEN}{x}{Style.RESET_ALL}" for x in w[j1:j2]]
        elif tag=="delete": out+=[f"{Fore.RED}<{x}>{Style.RESET_ALL}" for x in r[i1:i2]]
        else: out+=[f"{Fore.YELLOW}{x}{Style.RESET_ALL}" for x in w[j1:j2]]
    return " ".join(out)

# -----------------------------------------------------------------------------
# 7. LOAD DATASET
# -----------------------------------------------------------------------------
print(f"\n>>> Loading {args.dataset}/{args.subset or 'default'} …")

ds, ds_len = load_corpus(args.dataset, args.subset, args.config,
                         streaming=args.streaming, max_samples=args.max_samples)
print(f"Dataset ready (samples≈{ds_len if ds_len is not None else 'unknown'}).\n")

# -----------------------------------------------------------------------------
# 8. TRANSCRIPTION + EVAL LOOP  (wrapped in try/except for clean interrupts)
# -----------------------------------------------------------------------------
results_p, results_r, mistakes = [], [], []

pbar_total = None if args.streaming or ds_len is None else ds_len

try:
    for idx, ex in enumerate(tqdm(ds, total=pbar_total, desc="Transcribing")):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, ex["audio"]["array"], ex["audio"]["sampling_rate"])
            path = tmp.name
        try:
            _, out = model.generate([
                {"role":"user","message_type":"text","content":"Please transcribe the following audio:"},
                {"role":"user","message_type":"audio","content":path}],
                **SAMPLING, output_type="text")
        finally:
            os.remove(path)
        pred, ref = _clean(out), _clean(ex["text"])
        results_p.append(pred); results_r.append(ref)

        if pred!=ref:
            w=_diff(ref,pred); s_wer = wer_metric.compute(predictions=[pred], references=[ref])
            mistakes.append(dict(index=idx, reference=ref, prediction=pred, diff=w, sample_wer=s_wer))
            if idx%50==0:
                print(f"\nSample {idx}\nREF: {ref}\nHYP: {w}\nWER: {s_wer:.2%}\n")
        if (idx+1)%250==0:
            wandb.log(dict(running_wer=wer_metric.compute(predictions=results_p, references=results_r),
                           seen_samples=idx+1))
except KeyboardInterrupt:
    print("\n>>> Interrupted!  You can resume with the command shown above.\n")
    # Leave the resume file in place so next run stitches logs.
    wandb.finish(exit_code=255)
    sys.exit(130)  # propagate Ctrl‑C

# -----------------------------------------------------------------------------
# 9. FINAL METRICS & ARTIFACTS
# -----------------------------------------------------------------------------
final_wer = wer_metric.compute(predictions=results_p, references=results_r)
print(f"\nFinal WER = {final_wer:.4%}\n"); wandb.log({"final_wer":final_wer})

if mistakes:
    path = Path(f"errors_{args.dataset}_{args.subset or 'default'}.csv")
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=mistakes[0].keys()).writeheader();
        csv.DictWriter(f, fieldnames=mistakes[0].keys()).writerows(mistakes)
    wb = wandb.Table(columns=list(mistakes[0].keys()))
    for m in mistakes: wb.add_data(*m.values())
    wandb.log({"error_table": wb})

wandb.finish()

# Remove the resume file so the next invocation starts fresh
try:
    RESUME_FILE.unlink(missing_ok=True)
except Exception:
    pass
