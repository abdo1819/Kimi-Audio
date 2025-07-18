# Batch Mistakes Parquet Export

This functionality allows you to save ASR evaluation mistakes to parquet files for easier analysis and long-term storage.

## How it works

### Modified `test_asr.py`
The main evaluation script now:
- Saves `batch_mistakes` to individual parquet files every 250 samples
- Files are saved to `batch_mistakes_parquet/` directory with the naming pattern:
  ```
  mistakes_batch_XXXX_YYYYMMDD_HHMMSS_<run_id>.parquet
  ```
- Each file contains the same data as before: index, reference, prediction, diff, sample_wer, sample_cer
- Additional metadata columns: `source_file` and `batch_id` (added during merge)

### `merge_batch_mistakes.py`
A separate script to consolidate all individual parquet files:

```bash
# Basic usage - merge all files in batch_mistakes_parquet/
python merge_batch_mistakes.py

# Custom input/output
python merge_batch_mistakes.py --input-dir my_mistakes/ --output final_mistakes.parquet

# Merge and clean up individual files
python merge_batch_mistakes.py --cleanup
```

## Benefits

1. **Incremental saves**: Mistakes are saved as they occur, preventing data loss on interruption
2. **Efficient storage**: Parquet format is compressed and efficient for columnar data
3. **Easy analysis**: Load with pandas for detailed error analysis
4. **Scalable**: Handle large evaluation runs without memory issues

## Multi-GPU Support

The system now supports multi-GPU evaluation for faster processing:

### Using the Launcher Script (Recommended)
```bash
# Use 2 GPUs for evaluation
python launch_multi_gpu_asr.py --num_gpus 2 --dataset librispeech --subset test-clean --auto_merge

# Use 1 GPU (fallback)
python launch_multi_gpu_asr.py --num_gpus 1 --dataset chime8_notsofar1 --subset dev

# Auto-merge and cleanup
python launch_multi_gpu_asr.py --num_gpus 2 --dataset librispeech --auto_merge --cleanup
```

### Manual Multi-GPU Usage
```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python test_asr.py --num_gpus 2 --local_rank 0 --dataset librispeech &

# GPU 1  
CUDA_VISIBLE_DEVICES=1 python test_asr.py --num_gpus 2 --local_rank 1 --dataset librispeech &

# Wait for completion, then merge
wait
python merge_batch_mistakes.py
```

### Multi-GPU Features
- **Dataset splitting**: Automatically divides samples across GPUs
- **Single W&B tracking**: Only rank 0 logs to W&B to avoid conflicts
- **Distributed error collection**: Aggregates results from all GPUs
- **GPU-specific parquet files**: Each GPU saves its own mistakes with rank suffix

## Usage Example

```python
import pandas as pd

# Load merged mistakes
df = pd.read_parquet('merged_mistakes.parquet')

# Analyze errors
print(f"Total mistakes: {len(df)}")
print(f"Average WER: {df['sample_wer'].mean():.4f}")
print(f"Worst mistakes (highest WER):")
print(df.nlargest(5, 'sample_wer')[['reference', 'prediction', 'sample_wer']])

# Analyze by batch
batch_stats = df.groupby('batch_id').agg({
    'sample_wer': ['count', 'mean'],
    'sample_cer': 'mean'
}).round(4)
print(batch_stats)

# Multi-GPU analysis (if applicable)
if 'gpu_rank' in df.columns and df['gpu_rank'].nunique() > 1:
    gpu_stats = df.groupby('gpu_rank').agg({
        'sample_wer': ['count', 'mean'],
        'sample_cer': 'mean'
    }).round(4)
    print("Performance by GPU:")
    print(gpu_stats)
```

## File Structure

### Single GPU
```
batch_mistakes_parquet/
├── mistakes_batch_0000_20241208_143022_abc12345.parquet
├── mistakes_batch_0001_20241208_143522_abc12345.parquet
└── ...

merged_mistakes.parquet  # After running merge script
```

### Multi-GPU (2 GPUs)
```
batch_mistakes_parquet/
├── mistakes_batch_0000_20241208_143022_abc12345_rank0.parquet
├── mistakes_batch_0000_20241208_143022_abc12345_rank1.parquet
├── mistakes_batch_0001_20241208_143522_abc12345_rank0.parquet
├── mistakes_batch_0001_20241208_143522_abc12345_rank1.parquet
└── ...

merged_mistakes.parquet  # Consolidated results from all GPUs
```

The parquet files preserve all the original mistake data while adding efficient storage and analysis capabilities. Multi-GPU setups include GPU rank information for tracking which GPU processed each sample. 