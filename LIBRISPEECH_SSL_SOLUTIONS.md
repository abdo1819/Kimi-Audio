# LibriSpeech SSL Certificate Solutions

You're encountering SSL certificate issues when downloading LibriSpeech from OpenSLR. Here are several solutions:

## Solution 1: Use Fixed Download (Recommended)

I've already fixed the SSL issues in `test_asr.py`. The download should now work:

```bash
conda activate torch_gpu
python run_reasoning_experiment.py --preset quick_test --max_samples 2 --wandb_project ""
```

The fixes include:
- `wget --no-check-certificate` for wget downloads
- `curl -k` for curl downloads  
- SSL context with verification disabled for Python urllib

## Solution 2: Skip LibriSpeech (Demo Mode)

Use the demo configuration that only uses synthetic datasets:

```bash
python run_reasoning_experiment.py --config experiments/configs/demo_no_librispeech.yaml
```

This avoids LibriSpeech entirely and uses only synthetic audio generation.

## Solution 3: Manual LibriSpeech Download

Download LibriSpeech manually and place it in the cache directory:

```bash
# Create cache directory
mkdir -p ~/.cache/librispeech/LibriSpeech

# Download with insecure SSL (one-time)
wget --no-check-certificate https://us.openslr.org/resources/12/test-clean.tar.gz

# Extract to cache directory
tar -xzf test-clean.tar.gz -C ~/.cache/librispeech/

# Now run the experiment
python run_reasoning_experiment.py --preset quick_test
```

## Solution 4: Use HuggingFace LibriSpeech

Modify the experiment to use HuggingFace's LibriSpeech instead:

```bash
python run_reasoning_experiment.py --preset quick_test --librispeech_use_hf
```

## Solution 5: Set Environment Variable

Set the auto-download environment variable to skip prompts:

```bash
export LIBRISPEECH_AUTO_DOWNLOAD=true
python run_reasoning_experiment.py --preset quick_test
```

## Recommended Approach

For immediate testing, use **Solution 2** (demo mode):

```bash
python run_reasoning_experiment.py --config experiments/configs/demo_no_librispeech.yaml
```

This will run the complete CoT vs Latent comparison using synthetic datasets, demonstrating all the framework capabilities without needing LibriSpeech downloads.

## What You'll See

The demo will:
1. Generate synthetic counting and temporal reasoning tasks
2. Test both CoT (zero-shot) and Latent (silent) reasoning
3. Compare accuracy, efficiency, and generation time
4. Save results to `experiments/results/demo_no_librispeech_[timestamp]/`
5. Generate plots and analysis

This gives you the full experiment experience without download issues!
