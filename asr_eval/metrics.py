# asr_eval/metrics.py
"""Shared metric and normalizer utilities."""
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import evaluate

__all__ = ["clean", "wer_metric", "cer_metric"]

normalizer = BasicTextNormalizer()
clean = lambda s: normalizer(s).strip()
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

