#!/usr/bin/env python
"""
UPDATED *again* – robust length fetching
-------------------------------------------------------------
Fixes *TypeError: 'NoneType' object is not subscriptable* that appeared when
`builder.info.splits` is *None* (e.g. on community mirrors where the Dataset
Info JSON doesn’t include split‑metadata) **or** when a config isn’t required.

Changes
=======
1. **Safe split‑size lookup** – `_safe_num_examples(builder, split)` helper
   returns `None` when metadata is missing instead of crashing.
2. **Config handling** – only passes `config_name` to HF loader when it’s not
   `None`, so datasets with a single default config (WHAM) work out‑of‑the‑box.
3. **Dataset length fallback** – if we can’t get a count from the builder we
   attempt `len(ds)` (works for non‑streaming) and finally fall back to
   `"unknown"` in progress bars.

Nothing else changes: your model, coloured diffs, W&B logs all behave as before.
"""

from __future__ import annotations

import os, re, csv, argparse, tempfile, difflib
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
# 2. INIT W&B
# -----------------------------------------------------------------------------
wandb.init(project=args.wandb_project, name=args.wandb_run_name,
           config={**vars(args), "model": "moonshotai/Kimi-Audio-7B-Instruct"})

# -----------------------------------------------------------------------------
# 3. LOAD MODEL
# -----------------------------------------------------------------------------
model = KimiAudio(model_path="moonshotai/Kimi-Audio-7B-Instruct", load_detokenizer=True)
SAMPLING = dict(audio_temperature=0.8, audio_top_k=10, text_temperature=0.0,
               text_top_k=5, audio_repetition_penalty=1.0, audio_repetition_window_size=64,
               text_repetition_penalty=1.0, text_repetition_window_size=16)

# -----------------------------------------------------------------------------
# 4. DATASET REGISTRY
# -----------------------------------------------------------------------------

def _parse_librispeech_subset(label):
    mapping = {
        "test-clean":("clean","test"),"test-other":("other","test"),
        "dev-clean":("clean","validation"),"dev-other":("other","validation"),
        "train-clean-100":("clean","train.100"),"train-clean-360":("clean","train.360"),
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
# 5. DATASET LOADER
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
# 6. HELPER FUNCTIONS (WER + DIFF)
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
# 8. TRANSCRIPTION + EVAL LOOP
# -----------------------------------------------------------------------------
results_p, results_r, mistakes = [], [], []

pbar_total = None if args.streaming or ds_len is None else ds_len
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
