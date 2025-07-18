# LibriSpeech Direct Download

This document explains the new LibriSpeech direct download functionality that avoids Hugging Face's automatic download issues with large datasets (30GB+).

## Overview

The evaluation script now supports downloading LibriSpeech directly from the official OpenSLR servers instead of relying on Hugging Face's automatic download. This solves issues with:

- Large dataset downloads (30GB+ for full LibriSpeech)
- Network timeouts during HF downloads
- Resume capability for interrupted downloads
- Better progress tracking and error handling

## Usage

### Default Behavior (Direct Download)

By default, the script now uses direct download:

```bash
# Downloads directly from OpenSLR
python test_asr.py --dataset librispeech --subset test-clean
```

### Using HuggingFace Download (Legacy)

If you prefer the old HuggingFace automatic download:

```bash
# Uses HuggingFace (may fail for large datasets)
python test_asr.py --dataset librispeech --subset test-clean --librispeech_use_hf
```

### Multi-GPU Usage

```bash
# Direct download with multi-GPU
python launch_multi_gpu_asr.py --num_gpus 2 --dataset librispeech --subset test-clean

# HuggingFace download with multi-GPU  
python launch_multi_gpu_asr.py --num_gpus 2 --dataset librispeech --subset test-clean --librispeech_use_hf
```

### Fish Script Usage

```bash
# Direct download (default)
./test_asr.fish --dataset librispeech --subset test-clean --num_gpus 2

# HuggingFace download
./test_asr.fish --dataset librispeech --subset test-clean --librispeech_use_hf
```

## Cache Configuration

Downloads are cached to avoid re-downloading. Configure the cache location:

```bash
# Set custom cache directory
export LIBRISPEECH_CACHE="/path/to/your/cache"
python test_asr.py --dataset librispeech --subset test-clean

# Default cache location: ~/.cache/librispeech
```

## Supported Subsets

All official LibriSpeech subsets are supported:

| Subset | Size | Description |
|--------|------|-------------|
| `test-clean` | ~346MB | Test set, clean speech |
| `test-other` | ~328MB | Test set, other conditions |
| `dev-clean` | ~337MB | Development set, clean speech |
| `dev-other` | ~314MB | Development set, other conditions |
| `train-clean-100` | ~6.3GB | Training set, clean speech, 100h |
| `train-clean-360` | ~23GB | Training set, clean speech, 360h |
| `train-other-500` | ~30GB | Training set, other conditions, 500h |

## Features

### Resume Downloads
- Supports resume for interrupted downloads using `wget -c` or `curl -C -`
- Automatically detects existing partial downloads

### Smart Caching
- Only downloads if not already cached
- Verifies existing downloads before skipping
- Automatically extracts and organizes files

### Progress Tracking
- Shows download progress and file sizes
- Clear status messages for each step
- Error handling with informative messages

### Streaming Support
- Supports both streaming and in-memory datasets
- Efficient for large datasets with `--streaming` flag
- Compatible with multi-GPU data splitting

## Testing

Test the download functionality without running full evaluation:

```bash
# Test with small subset
python test_librispeech_download.py --subset test-clean --max_samples 5

# Test with different subset
python test_librispeech_download.py --subset dev-clean --max_samples 10
```

## Implementation Details

### Download Process
1. Check if subset already exists in cache
2. Download `.tar.gz` from OpenSLR if needed
3. Extract to `{cache}/LibriSpeech/{subset}/`
4. Parse transcript files and organize metadata
5. Create HuggingFace-compatible dataset

### File Structure
```
~/.cache/librispeech/
└── LibriSpeech/
    ├── test-clean/
    │   ├── 1089/
    │   │   ├── 134686/
    │   │   │   ├── 1089-134686.trans.txt
    │   │   │   ├── 1089-134686-0001.flac
    │   │   │   └── ...
    │   │   └── ...
    │   └── ...
    └── ...
```

### Dependencies
- `wget` or `curl` for downloading
- `tar` for extraction
- Standard Python libraries: `pathlib`, `subprocess`, `json`

## Troubleshooting

### Common Issues

**Download fails:**
```bash
# Check if wget/curl is installed
which wget || which curl

# Check network connectivity
ping us.openslr.org

# Try manual download
wget https://us.openslr.org/resources/12/test-clean.tar.gz
```

**Cache permissions:**
```bash
# Fix cache permissions
chmod -R 755 ~/.cache/librispeech

# Use custom cache location
export LIBRISPEECH_CACHE="/tmp/librispeech_cache"
```

**Large dataset issues:**
```bash
# Use streaming for large datasets
python test_asr.py --dataset librispeech --subset train-other-500 --streaming

# Limit samples for testing
python test_asr.py --dataset librispeech --subset train-other-500 --max_samples 1000
```

### Performance Tips

1. **Use SSD for cache** - Faster audio loading
2. **Sufficient disk space** - Up to 60GB for all subsets
3. **Stable internet** - Large downloads benefit from stable connection
4. **Use streaming** - For datasets larger than available RAM

## Migration from HuggingFace

If you were previously using HuggingFace automatic download:

1. **No code changes needed** - Direct download is now default
2. **Cache location differs** - HF uses different cache structure
3. **Add `--librispeech_use_hf`** - To keep using HF download
4. **Performance improvement** - Direct download is typically faster

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LIBRISPEECH_CACHE` | `~/.cache/librispeech` | Cache directory for downloads |

## Examples

### Quick Test (Small Dataset)
```bash
python test_asr.py --dataset librispeech --subset test-clean --max_samples 100
```

### Full Evaluation (Large Dataset)
```bash
python test_asr.py --dataset librispeech --subset train-clean-360 --streaming
```

### Multi-GPU Large Dataset
```bash
./test_asr.fish --num_gpus 2 --dataset librispeech --subset train-other-500 --streaming --auto_merge
``` 