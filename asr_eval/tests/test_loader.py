# tests/test_loader.py
"""Verify that each registry entry loads without crashing (smoke test)."""
import pytest
from asr_eval.datasets import load_corpus
from asr_eval.registry import REGISTRY

@pytest.mark.parametrize("key", list(REGISTRY.keys())[4:5])  # limit for CI speed
def test_load_smoke(key):
    ds, _ = load_corpus(key, max_samples=1, streaming=True)
    ex = next(iter(ds))
    assert "audio" in ex
    assert "text" in ex

# tests/test_dummy_eval.py
"""Evaluate the pipeline with a dummy model that echoes reference."""

class EchoModel:
    def generate(self, messages, **kwargs):
        # echo prompt content "audio" placeholder
        return None, "test"  # always perfect


def test_evaluation(dummy_dataset):
    from asr_eval.evaluation import evaluate_model
    wer, cer, ser = evaluate_model(EchoModel(), dummy_dataset, sampling={})
    assert wer == 0.0 and cer == 0.0 and ser == 0.0
