#!/usr/bin/env python
"""
Test script for LibriSpeech direct download functionality.
Tests the new direct download without running the full ASR evaluation.
"""

import os
import sys
from pathlib import Path

# Add the current directory to the path so we can import from test_asr.py
sys.path.insert(0, str(Path(__file__).parent))

from test_asr import download_librispeech_subset, load_librispeech_local, get_librispeech_cache_dir

def test_download_and_load(subset="test-clean", max_samples=5):
    """Test downloading and loading a small subset of LibriSpeech"""
    
    print(f"🧪 Testing LibriSpeech direct download with subset: {subset}")
    print(f"   Max samples: {max_samples}")
    
    # Use consistent cache directory (same as main script)
    cache_root = get_librispeech_cache_dir()
    print(f"   Using cache directory: {cache_root}")
    
    try:
        # Test download
        print("\n1️⃣ Testing download...")
        subset_dir = download_librispeech_subset(subset, cache_root)
        print(f"✅ Download successful: {subset_dir}")
        
        # Test loading 
        print("\n2️⃣ Testing loading...")
        ds, num_samples = load_librispeech_local(subset, cache_root, max_samples=max_samples)
        print(f"✅ Loading successful: {num_samples} samples")
        
        # Test a few samples
        print("\n3️⃣ Testing sample access...")
        sample_count = 0
        for i, sample in enumerate(ds):
            print(f"   Sample {i+1}: {sample['utterance_id']} - '{sample['text'][:50]}...'")
            sample_count += 1
            if sample_count >= 3:  # Just show first 3
                break
        
        print(f"\n✅ All tests passed! Successfully processed {sample_count} samples.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False
    
    finally:
        # Note: Downloads are preserved in the consistent cache location
        # This allows reuse across test runs and with the main evaluation script
        print(f"💾 Downloads preserved in cache: {cache_root}")
        print(f"   Set LIBRISPEECH_CACHE environment variable to use a different location")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LibriSpeech direct download")
    parser.add_argument("--subset", default="test-clean", help="LibriSpeech subset to test")
    parser.add_argument("--max_samples", type=int, default=5, help="Max samples to test")
    
    args = parser.parse_args()
    
    success = test_download_and_load(args.subset, args.max_samples)
    sys.exit(0 if success else 1) 