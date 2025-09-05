"""
Utility functions for the reasoning comparison experiment
"""

import random
import numpy as np
import torch
import os
from typing import Dict, Any, List
import json
from pathlib import Path

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_device_info() -> Dict[str, Any]:
    """Get information about available compute devices"""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            info[f"device_{i}"] = {
                "name": device_props.name,
                "total_memory": device_props.total_memory,
                "major": device_props.major,
                "minor": device_props.minor,
            }
    
    return info

def cleanup_temp_files(file_paths: List[str]):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            pass  # Ignore cleanup errors

def format_results_table(results: Dict[str, Any]) -> str:
    """Format results as a readable table"""
    lines = []
    lines.append("=" * 80)
    lines.append("REASONING COMPARISON RESULTS")
    lines.append("=" * 80)
    
    if "aggregated_results" in results:
        agg = results["aggregated_results"]
        lines.append(f"Condition: {agg.condition}")
        lines.append(f"Dataset: {agg.dataset}")
        lines.append(f"Samples: {agg.num_samples}")
        lines.append("-" * 40)
        lines.append(f"Accuracy: {agg.accuracy:.4f} ± {agg.accuracy_std:.4f}")
        lines.append(f"Avg Generation Time: {agg.avg_generation_time:.3f}s")
        lines.append(f"Avg Tokens: {agg.avg_tokens:.1f}")
        lines.append(f"Token Efficiency: {agg.avg_token_efficiency:.1f} tok/s")
        
        if agg.accuracy_by_type:
            lines.append("\nAccuracy by Task Type:")
            for task_type, acc in agg.accuracy_by_type.items():
                lines.append(f"  {task_type}: {acc:.4f}")
        
        if agg.accuracy_by_difficulty:
            lines.append("\nAccuracy by Difficulty:")
            for difficulty, acc in agg.accuracy_by_difficulty.items():
                lines.append(f"  {difficulty}: {acc:.4f}")
    
    lines.append("=" * 80)
    return "\n".join(lines)

def save_json(data: Any, filepath: str, indent: int = 2):
    """Save data as JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)

def load_json(filepath: str) -> Any:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def ensure_dir(directory: str):
    """Ensure directory exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)
