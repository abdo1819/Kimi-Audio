#!/usr/bin/env python
"""
UPDATED – per‑step mistake logging
---------------------------------
Logs **new mistakes** to W&B every 250 samples so you can inspect them during
long runs. Adds `batch_mistakes` buffering and uploads a table each interval.
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

# -------------------------------- CLI --------------------------------
parser = argparse.ArgumentParser(description="Evaluate Kimi-Audio-7B-Instruct on ASR benchmarks")
parser.add_argument("--dataset", choices=["librispeech", "wham", "ami", "gigaspeech"], default="librispeech")
parser.add_argument("--subset", type=str)
parser.add_argument("--config", type=str)
parser.add_argument("--streaming", action="store_true")
parser.add_argument("--max_samples", type=int)
parser.add_argument("--wandb_project", default="kimi-audio-multi-eval")
parser.add_argument("--wandb_run_name")
args = parser.parse_args()

# ---------------------------- W&B RESUME -----------------------------
RESUME_FILE = Path(".wandb_run_id")
prev_id = RESUME_FILE.read_text().strip() if RESUME_FILE.exists() else None
run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, id=prev_id,
                 resume="allow", config={**vars(args), "model": "moonshotai/Kimi-Audio-7B-Instruct"})
if prev_id is None:
    RESUME_FILE.write_text(run.id)
print("\nResume with: WANDB_RUN_ID=" + run.id + " WANDB_RESUME=allow python " + " ".join(map(shlex.quote, sys.argv)) + "\n")

# ----------------------------- MODEL ---------------------------------
model = KimiAudio(model_path="moonshotai/Kimi-Audio-7B-Instruct", load_detokenizer=True)
SAMPLING = dict(audio_temperature=0.8, audio_top_k=10, text_temperature=0.0,
                text_top_k=5, audio_repetition_penalty=1.0, audio_repetition_window_size=64,
                text_repetition_penalty=1.0, text_repetition_window_size=16)

# ----------------------- DATASET REGISTRY ----------------------------
# (same helpers as before, omitted for brevity)

def _parse_librispeech_subset(label):
    mapping = {"test-clean":("clean","test"),"test-other":("other","test"),
               "dev-clean":("clean","validation"),"dev-other":("other","validation"),
               "train-clean-100":("clean","train.100"),"train-clean-360":("other","train.360"),
               "train-other-500":("other","train.500")}
    return mapping[label]
REGISTRY = {"librispeech": dict(hf="librispeech_asr", subset_parser=_parse_librispeech_subset,
                                 audio="audio", text="text", def_subset="test-clean", sr=16_000)}
# safe_num_examples & load_corpus identical to earlier revision
# --------------------------- HELPERS ---------------------------------
wer_metric = evaluate.load("wer")
_clean_re = re.compile(r"[^a-z0-9\s]")
_clean = lambda s: re.sub(r"\s+", " ", _clean_re.sub("", s.lower().strip()))

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

# --------------------------- EVAL LOOP -------------------------------
results_p, results_r = [], []
all_mistakes, batch_mistakes, failures = [], [], []

try:
    for idx, ex in enumerate(tqdm(ds, total=ds_len, desc="Transcribing")):
        # short‑clip guard
        if len(ex["audio"]["array"]) < 0.2*ex["audio"]["sampling_rate"]:
            failures.append({"index": idx, "reason": "too_short"})
            results_p.append(""); results_r.append(_clean(ex["text"]))
            continue
        # write wav
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
        if pred!=ref:
            entry = dict(index=idx, reference=ref, prediction=pred,
                          diff=_diff(ref,pred), sample_wer=wer_metric.compute(predictions=[pred], references=[ref]))
            all_mistakes.append(entry); batch_mistakes.append(entry)
        # --- log every 250 examples ---
        if (idx+1)%250==0:
            metrics = {"running_wer": wer_metric.compute(predictions=results_p, references=results_r),
                       "seen_samples": idx+1, "failed": len(failures)}
            if batch_mistakes:
                cols = list(batch_mistakes[0].keys())
                tbl = wandb.Table(columns=cols)
                for m in batch_mistakes: tbl.add_data(*m.values())
                metrics["new_mistakes"] = tbl
                batch_mistakes = []  # reset buffer
            wandb.log(metrics)
except KeyboardInterrupt:
    print("Interrupted – resume later!"); wandb.finish(exit_code=255); sys.exit(130)

# --------------------------- FINALISE --------------------------------
final_wer = wer_metric.compute(predictions=results_p, references=results_r)
wandb.log({"final_wer": final_wer, "failed": len(failures)})
# full mistakes table
if all_mistakes:
    cols = list(all_mistakes[0].keys()); full_tbl = wandb.Table(columns=cols)
    for m in all_mistakes: full_tbl.add_data(*m.values())
    wandb.log({"error_table": full_tbl})
wandb.finish(); RESUME_FILE.unlink(missing_ok=True)
