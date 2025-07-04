# tests/conftest.py
"""Pytest fixtures."""
import pytest
from datasets import Dataset
import numpy as np

@pytest.fixture
def dummy_dataset(tmp_path):
    """Creates a tiny 1‑sample dataset with a 1‑second sine wave and ground‑truth."""
    sr = 16_000
    t  = np.linspace(0, 1, sr, False)
    sine = 0.1 * np.sin(2 * np.pi * 440 * t)
    wav_path = tmp_path / "sine.wav"
    import soundfile as sf
    sf.write(wav_path, sine, sr)

    data = [{"audio": str(wav_path), "text": "test"}]
    ds = Dataset.from_list(data)
    return ds.cast_column("audio", {"path": str(wav_path), "array": sine, "sampling_rate": sr})

