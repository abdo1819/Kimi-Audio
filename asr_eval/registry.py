# asr_eval/registry.py
"""Dataset registry and related helpers (split parsers, etc.)."""
from __future__ import annotations
from typing import Tuple

# ---- helpers ------------------------------------------------------

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

# ---- central registry --------------------------------------------
REGISTRY = {
    # Existing benchmarks
    "librispeech": dict(
        hf="librispeech_asr",
        subset_parser=_parse_librispeech_subset,
        audio="audio",
        text="text",
        def_subset="test-clean",
        sr=16_000,
        subsets=[  # HF split names (per clean/other configs)
            "train.100", "train.360", "train.500",
            "validation", "test"
        ],
    ),  # :contentReference[oaicite:0]{index=0}

    "wham": dict(
        hf="nguyenvulebinh/wham",
        audio="audio",
        text=None,
        def_subset="train",
        sr=16_000,
        subsets=["train", "validation", "test"],  # WHAM! spec 30 h/10 h/5 h splits
    ),  # :contentReference[oaicite:1]{index=1}

    "ami": dict(
        hf="edinburghcstr/ami",
        config="ihm",          # also “sdm”, “mdm”, etc. – ihm used here
        audio="audio",
        text="text",
        def_subset="train",
        sr=16_000,
        subsets=["train", "validation", "test"],  # HF viewer shows all three
    ),  # :contentReference[oaicite:2]{index=2}

    "gigaspeech": dict(
        hf="speechcolab/gigaspeech",
        audio="audio",
        text="text",
        def_subset="test_xl",
        sr=16_000,
        subsets=[              # HF configs + evaluation splits
            "train_xs", "train_s", "train_m", "train_l", "train_xl",
            "dev", "test"
        ],
    ),  # :contentReference[oaicite:3]{index=3}

    # ---------------- CHiME-8 additions ----------------
    # NOTSOFAR-1 (HF release) – dev/train sets contain transcripts.
    "chime8_notsofar1": dict(
        hf="microsoft/NOTSOFAR",
        audio="audio",
        text="transcript",
        def_subset="train",           # train / validation / test
        sr=16_000,
        subsets=["dev_set", "eval_set", "train_set"],
    ),  # :contentReference[oaicite:4]{index=4}

    # Place-holders for other CHiME-8 DASR scenarios. Require local prep via chime-utils
    "chime8_chime6": dict(
        local=True,
        def_subset="train",
        sr=16_000,
        subsets=["train", "dev", "eval"],  # official CHiME-6 nomenclature
    ),
    "chime8_mixer6": dict(
        local=True,
        def_subset="train",
        sr=16_000,
        subsets=["train", "dev", "eval"],
    ),
    "chime8_dipco": dict(
        local=True,
        def_subset="train",
        sr=16_000,
        subsets=["train", "dev", "eval"],
    ),
}
