#!/usr/bin/env python
"""
Merge Batch Mistakes Parquet Files
----------------------------------
This script merges all the individual batch mistakes parquet files generated
by test_asr.py into a single consolidated parquet file for easier analysis.

Usage:
    python merge_batch_mistakes.py [--input-dir batch_mistakes_parquet] [--output merged_mistakes.parquet]
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import List
import sys


def find_parquet_files(input_dir: Path) -> List[Path]:
    """Find all parquet files in the input directory."""
    parquet_files = list(input_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"❌ No parquet files found in {input_dir}")
        return []
    
    print(f"📁 Found {len(parquet_files)} parquet files:")
    for file in sorted(parquet_files):
        print(f"   • {file.name}")
    
    return sorted(parquet_files)


def merge_parquet_files(parquet_files: List[Path], output_file: Path) -> bool:
    """Merge multiple parquet files into a single file."""
    if not parquet_files:
        return False
    
    print(f"\n🔄 Merging {len(parquet_files)} files...")
    
    try:
        # Read all parquet files and concatenate
        dfs = []
        total_rows = 0
        
        for file_path in parquet_files:
            df = pd.read_parquet(file_path)
            
            # Add metadata columns to track source
            df['source_file'] = file_path.name
            name_parts = file_path.name.split('_')
            df['batch_id'] = name_parts[2]  # Extract batch ID from filename
            # Extract GPU rank if present in filename (format: ...rankX.parquet)
            if 'rank' in file_path.name:
                rank_part = [part for part in name_parts if part.startswith('rank')][0]
                df['gpu_rank'] = int(rank_part.replace('rank', '').replace('.parquet', ''))
            else:
                df['gpu_rank'] = 0  # Default for single GPU files
            
            dfs.append(df)
            total_rows += len(df)
            print(f"   ✓ Loaded {file_path.name}: {len(df)} rows")
        
        # Concatenate all DataFrames
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # Sort by batch_id, gpu_rank, and index for consistent ordering
        merged_df = merged_df.sort_values(['batch_id', 'gpu_rank', 'index']).reset_index(drop=True)
        
        # Save merged file
        merged_df.to_parquet(output_file, index=False)
        
        print(f"\n✅ Successfully merged {total_rows} mistakes from {len(parquet_files)} files")
        print(f"💾 Saved to: {output_file}")
        
        # Display summary statistics
        print(f"\n📊 Summary:")
        print(f"   • Total mistakes: {len(merged_df):,}")
        print(f"   • Unique batches: {merged_df['batch_id'].nunique()}")
        print(f"   • GPUs used: {merged_df['gpu_rank'].nunique()}")
        print(f"   • Average WER per mistake: {merged_df['sample_wer'].mean():.4f}")
        print(f"   • Average CER per mistake: {merged_df['sample_cer'].mean():.4f}")
        
        # Show distribution of mistakes per batch
        batch_counts = merged_df['batch_id'].value_counts().sort_index()
        print(f"   • Mistakes per batch: min={batch_counts.min()}, max={batch_counts.max()}, avg={batch_counts.mean():.1f}")
        
        # Show distribution per GPU if multi-GPU
        if merged_df['gpu_rank'].nunique() > 1:
            gpu_counts = merged_df['gpu_rank'].value_counts().sort_index()
            print(f"   • Mistakes per GPU: {dict(gpu_counts)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error merging files: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Merge batch mistakes parquet files into a single file")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="batch_mistakes_parquet",
        help="Directory containing parquet files to merge (default: batch_mistakes_parquet)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="merged_mistakes.parquet",
        help="Output file name (default: merged_mistakes.parquet)"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove individual parquet files after successful merge"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_file = Path(args.output)
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"❌ Input directory {input_dir} does not exist")
        sys.exit(1)
    
    # Find parquet files
    parquet_files = find_parquet_files(input_dir)
    if not parquet_files:
        sys.exit(1)
    
    # Merge files
    success = merge_parquet_files(parquet_files, output_file)
    if not success:
        sys.exit(1)
    
    # Optional cleanup
    if args.cleanup:
        print(f"\n🧹 Cleaning up individual files...")
        for file_path in parquet_files:
            file_path.unlink()
            print(f"   🗑️  Removed {file_path.name}")
        print("✅ Cleanup complete")
    
    print(f"\n🎉 Merge complete! Use the following to analyze your data:")
    print(f"   import pandas as pd")
    print(f"   df = pd.read_parquet('{output_file}')")
    print(f"   print(df.head())")


if __name__ == "__main__":
    main() 