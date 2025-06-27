#!/usr/bin/env python
"""
UPDATED – Whisper BasicTextNormalizer + CER & SER
-------------------------------------------------
* **Normalization** now uses `transformers.models.whisper.english_normalizer.BasicTextNormalizer`,
  matching Whisper’s official preprocessing.
* Added **CER** (character error rate) via `evaluate.load("cer")`.
* Added **SER** (sentence/utterance error rate – a binary 0/1 per sample).
* W&B logs now include running and final **WER**, **CER**, and **SER** plus the
  per‑250‑step *new mistakes* table as before.
"""

from __future__ import annotations

import os, re, csv, argparse, tempfile, difflib, sys, shlex
from pathlib import Path

import wandb
from tqdm import tqdm
import soundfile as sf
from datasets import load_dataset, load_dataset_builder, Audio, IterableDataset
import evaluate
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from kimia_infer.api.kimia import KimiAudio
from colorama import init as colorama_init, Fore, Style
colorama_init(autoreset=True)

# ---------------------------- CLI ------------------------------------
parser = argparse.ArgumentParser(description="Evaluate Kimi‑Audio‑7B‑Instruct on ASR benchmarks")
parser.add_argument("--dataset", choices=["librispeech", "wham", "ami", "gigaspeech"], default="librispeech")
parser.add_argument("--subset", type=str)
parser.add_argument("--config", type=str)
parser.add_argument("--streaming", action="store_true")
parser.add_argument("--max_samples", type=int)
parser.add_argument("--wandb_project", default="kimi-audio-multi-eval")
parser.add_argument("--wandb_run_name")
args = parser.parse_args()

# -------------------------- W&B RESUME -------------------------------
RESUME_FILE = Path(".wandb_run_id")
prev_id = RESUME_FILE.read_text().strip() if RESUME_FILE.exists() else None
run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, id=prev_id,
                 resume="allow", config={**vars(args), "model": "moonshotai/Kimi-Audio-7B-Instruct"})
if prev_id is None:
    RESUME_FILE.write_text(run.id)
print(f"\nResume with: WANDB_RUN_ID={run.id} WANDB_RESUME=allow python " + " ".join(map(shlex.quote, sys.argv)) + "\n")

# ---------------------------- MODEL ----------------------------------
model = KimiAudio(model_path="moonshotai/Kimi-Audio-7B-Instruct", load_detokenizer=True)
SAMPLING = dict(audio_temperature=0.8, audio_top_k=10, text_temperature=0.0,
                text_top_k=5, audio_repetition_penalty=1.0, audio_repetition_window_size=64,
                text_repetition_penalty=1.0, text_repetition_window_size=16)

# ----------------------- DATASET REGISTRY ----------------------------

def _parse_librispeech_subset(label):
    mapping = {
        "test-clean":("clean","test"),"test-other":("other","test"),
        "dev-clean":("clean","validation"),"dev-other":("other","validation"),
        "train-clean-100":("clean","train.100"),"train-clean-360":("other","train.360"),
        "train-other-500":("other","train.500")}
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

# -------------------------- LOADERS ----------------------------------

def _safe_num_examples(builder, split):
    splits = getattr(builder.info, "splits", None)
    return splits[split].num_examples if splits and split in splits else None


def load_corpus(key, subset=None, config_override=None, *, streaming=False, max_samples=None):
    cfg = REGISTRY[key]
    subset = subset or cfg["def_subset"]
    config_name = config_override or cfg.get("config")

    if key == "librispeech":
        config_name, split_name = cfg["subset_parser"](subset)
    else:
        split_name = subset

    num = None
    if not streaming:
        kw = {"trust_remote_code": True}
        if config_name is not None:
            kw["name"] = config_name
        builder = load_dataset_builder(cfg["hf"], **kw)
        num = _safe_num_examples(builder, split_name)

    ds_kw = dict(split=split_name, streaming=streaming, trust_remote_code=True)
    if config_name is not None:
        ds_kw["name"] = config_name
    ds = load_dataset(cfg["hf"], **ds_kw)

    if max_samples is not None:
        ds = ds.take(max_samples)
        num = min(num or max_samples, max_samples)

    if cfg["audio"] != "audio":
        ds = ds.rename_column(cfg["audio"], "audio")
    if cfg["text"] and cfg["text"] != "text":
        ds = ds.rename_column(cfg["text"], "text")
    elif not cfg["text"]:
        ds = ds.map(lambda _: {"text": ""}, remove_columns=[c for c in ds.column_names if c != "audio"], batched=False)

    ds = ds.cast_column("audio", Audio(sampling_rate=cfg["sr"]))
    if num is None and not isinstance(ds, IterableDataset):
        num = len(ds)
    return ds, num

# -------------------------- METRICS ----------------------------------
normalizer = BasicTextNormalizer()
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

_clean = lambda s: normalizer(s).strip()

# colour diff (unchanged)
def _diff(r,h):
    r,w=r.split(),h.split();out=[]
    for tag,i1,i2,j1,j2 in difflib.SequenceMatcher(None,r,w).get_opcodes():
        if tag=="equal":out+=w[j1:j2]
        elif tag=="insert":out+=[f"{Fore.GREEN}{x}{Style.RESET_ALL}" for x in w[j1:j2]]
        elif tag=="delete":out+=[f"{Fore.RED}<{x}>{Style.RESET_ALL}" for x in r[i1:i2]]
        else:out+=[f"{Fore.YELLOW}{x}{Style.RESET_ALL}" for x in w[j1:j2]]
    return " ".join(out)

# --------------------------- LOAD DATA -------------------------------
ds, ds_len = load_corpus(args.dataset, args.subset, args.config,
                         streaming=args.streaming, max_samples=args.max_samples)
print(f"Dataset ready (samples≈{ds_len}).")

# ----------------------- MAIN EVAL LOOP ------------------------------
results_p, results_r = [], []
all_mistakes, batch_mistakes, failures = [], [], []
ser_errors = 0  # sentence error count

try:
    for idx, ex in enumerate(tqdm(ds, total=ds_len, desc="Transcribing")):
        # skip ultra‑short
        if len(ex["audio"]["array"]) < 0.2*ex["audio"]["sampling_rate"]:
            failures.append({"index": idx, "reason": "too_short"})
            results_p.append(""); results_r.append(_clean(ex["text"]))
            ser_errors += 1
            continue
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, ex["audio"]["array"], ex["audio"]["sampling_rate"])
            wav_path = tmp.name
        try:
            try:
                _, out = model.generate([
                    {"role":"user","message_type":"text","content":"Please transcribe the following audio:"},
                    {"role":"user","message_type":"audio","content":wav_path}],
                    **SAMPLING, output_type="text")
            except RuntimeError as e:
                if "expected a non-empty list" in str(e):
                    failures.append({"index": idx, "reason": "whisper_empty"}); out=""
                else: raise
        finally:
            os.remove(wav_path)

        pred, ref = _clean(out), _clean(ex["text"])
        results_p.append(pred); results_r.append(ref)
        sent_err = int(pred != ref); ser_errors += sent_err

        if pred != ref:
            entry = dict(index=idx, reference=ref, prediction=pred,
                          diff=_diff(ref, pred),
                          sample_wer=wer_metric.compute(predictions=[pred], references=[ref]),
                          sample_cer=cer_metric.compute(predictions=[pred], references=[ref]))
            all_mistakes.append(entry); batch_mistakes.append(entry)

        # ---- periodic logging ----
        if (idx+1) % 250 == 0:
            running_wer = wer_metric.compute(predictions=results_p, references=results_r)
            running_cer = cer_metric.compute(predictions=results_p, references=results_r)
            running_ser = ser_errors / (idx+1)
            metrics = {"running_wer": running_wer,
                       "running_cer": running_cer,
                       "running_ser": running_ser,
                       "seen_samples": idx+1,
                       "failed": len(failures)}
            if batch_mistakes:
                tbl = wandb.Table(columns=list(batch_mistakes[0].keys()))
                for m in batch_mistakes: tbl.add_data(*m.values())
                metrics["new_mistakes"] = tbl
                batch_mistakes = []
            wandb.log(metrics)
except KeyboardInterrupt:
    print("Interrupted – resume later!"); wandb.finish(exit_code=255); sys.exit(130)

# ------------------------- FINAL METRICS -----------------------------
final_wer = wer_metric.compute(predictions=results_p, references=results_r)
final_cer = cer_metric.compute(predictions=results_p, references=results_r)
final_ser = ser_errors / len(results_p)
wandb.log({"final_wer": final_wer, "final_cer": final_cer, "final_ser": final_ser, "failed": len(failures)})

# upload full mistakes
if all_mistakes:
    tbl = wandb.Table(columns=list(all_mistakes[0].keys()))
    for m in all_mistakes: tbl.add_data(*m.values())
    wandb.log({"error_table": tbl})

wandb.finish(); RESUME_FILE.unlink(missing_ok=True)
