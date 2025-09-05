#!/usr/bin/env python3
"""
Main entry point for CoT vs Latent Reasoning Comparison Experiment

This script provides a convenient way to run the reasoning comparison experiment
from the project root directory.

Usage:
    python run_reasoning_experiment.py --preset quick_test
    python run_reasoning_experiment.py --config experiments/configs/full_comparison.yaml
"""

import sys
from pathlib import Path

# Add experiments directory to path
sys.path.insert(0, str(Path(__file__).parent / "experiments"))

# Import and run the experiment
from run_reasoning_compare import main

if __name__ == "__main__":
    main()
