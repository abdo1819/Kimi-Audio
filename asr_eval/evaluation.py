# asr_eval/evaluation.py
"""Core evaluation loop (model‑agnostic)."""
from __future__ import annotations
from pathlib import Path
import tempfile, difflib, os
from typing import Iterable, Tuple, List
import soundfile as sf
from tqdm import tqdm
from colorama import Fore, Style

from .metrics import clean, wer_metric, cer_metric

__all__ = ["evaluate_model", "diff"]

def diff(r: str, h: str) -> str:
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


def _transcribe_example(model, ex, *, sampling):
    if len(ex["audio"]["array"]) < 0.2 * ex["audio"]["sampling_rate"]:
        return ""  # treat as failure
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, ex["audio"]["array"], ex["audio"]["sampling_rate"])
        wav_path = tmp.name
    try:
        _, out = model.generate(
            [
                {"role": "user", "message_type": "text", "content": "Please transcribe the following audio:"},
                {"role": "user", "message_type": "audio", "content": wav_path},
            ],
            **sampling,
            output_type="text",
        )
        return out
    finally:
        os.remove(wav_path)


def evaluate_model(model, dataset, *, sampling) -> Tuple[float, float, float]:
    preds, refs = [], []
    ser_errors = 0
    for ex in tqdm(dataset, desc="Transcribing"):
        pred = clean(_transcribe_example(model, ex, sampling=sampling))
        ref = clean(ex["text"])
        preds.append(pred); refs.append(ref)
        ser_errors += int(pred != ref)
    wer = wer_metric.compute(predictions=preds, references=refs)
    cer = cer_metric.compute(predictions=preds, references=refs)
    ser = ser_errors / len(preds)
    return wer, cer, ser