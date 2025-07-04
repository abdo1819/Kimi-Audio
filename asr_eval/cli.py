

# asr_eval/cli.py
"""Command‑line entry point."""
import argparse, shlex, sys
from pathlib import Path
import wandb

from .constants import RESUME_FILE_NAME, DEFAULT_WANDB_PROJECT, DEFAULT_MODEL
from .datasets import load_corpus
from .evaluation import evaluate_model
from kimia_infer.api.kimia import KimiAudio

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

def main():
    parser = argparse.ArgumentParser(description="ASR evaluation toolkit")
    parser.add_argument("--dataset", default="librispeech")
    parser.add_argument("--subset")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--wandb_project", default=DEFAULT_WANDB_PROJECT)
    args = parser.parse_args()

    ds, ds_len = load_corpus(args.dataset, args.subset, streaming=args.streaming, max_samples=args.max_samples)
    model = KimiAudio(model_path=DEFAULT_MODEL, load_detokenizer=True)

    resume_path = Path(RESUME_FILE_NAME)
    run_id = resume_path.read_text().strip() if resume_path.exists() else None
    run = wandb.init(project=args.wandb_project, id=run_id, resume="allow")
    if run_id is None:
        resume_path.write_text(run.id)

    wer, cer, ser = evaluate_model(model, ds, sampling=SAMPLING)
    wandb.log({"final_wer": wer, "final_cer": cer, "final_ser": ser})
    wandb.finish()
    resume_path.unlink(missing_ok=True)

if __name__ == "__main__":
    main()



