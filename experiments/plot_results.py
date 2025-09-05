#!/usr/bin/env python3
"""
Plotting utilities for CoT vs Latent Reasoning Comparison Results

Creates visualizations for:
- Accuracy vs reasoning steps
- Efficiency frontiers (accuracy per token/FLOP)
- Performance breakdown by task type and difficulty
- Calibration plots
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Import plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('seaborn-v0_8')
    PLOTTING_AVAILABLE = True
except ImportError:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
        PLOTTING_AVAILABLE = True
    except ImportError:
        PLOTTING_AVAILABLE = False

def load_results(results_dir: str) -> Dict[str, Any]:
    """Load results from experiment directory"""
    results_path = Path(results_dir)
    
    # Load raw results
    raw_file = results_path / "raw_results.json"
    aggregated_file = results_path / "aggregated_results.json"
    csv_file = results_path / "results.csv"
    
    data = {}
    
    if raw_file.exists():
        with open(raw_file, 'r') as f:
            data['raw'] = json.load(f)
    
    if aggregated_file.exists():
        with open(aggregated_file, 'r') as f:
            data['aggregated'] = json.load(f)
    
    if csv_file.exists():
        data['df'] = pd.read_csv(csv_file)
    
    return data

def plot_accuracy_vs_steps(results: Dict[str, Any], output_dir: str):
    """Plot accuracy vs reasoning steps for CoT and latent conditions"""
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available. Skipping plots.")
        return
    
    df = results.get('df')
    if df is None:
        print("No DataFrame available for plotting")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # CoT: Accuracy vs Chain Length
    cot_data = df[df['condition'].str.startswith('cot')]
    if not cot_data.empty:
        # Group by chain length (would need to extract from condition or add to data)
        cot_grouped = cot_data.groupby('condition')['accuracy'].mean().reset_index()
        ax1.bar(range(len(cot_grouped)), cot_grouped['accuracy'])
        ax1.set_xlabel('CoT Condition')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('CoT: Accuracy by Condition')
        ax1.set_xticks(range(len(cot_grouped)))
        ax1.set_xticklabels(cot_grouped['condition'], rotation=45)
    
    # Latent: Accuracy vs Loop Depth
    latent_data = df[df['condition'].str.startswith('latent')]
    if not latent_data.empty:
        latent_grouped = latent_data.groupby('condition')['accuracy'].mean().reset_index()
        ax2.bar(range(len(latent_grouped)), latent_grouped['accuracy'])
        ax2.set_xlabel('Latent Condition')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Latent: Accuracy by Condition')
        ax2.set_xticks(range(len(latent_grouped)))
        ax2.set_xticklabels(latent_grouped['condition'], rotation=45)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'accuracy_vs_steps.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_efficiency_frontier(results: Dict[str, Any], output_dir: str):
    """Plot efficiency frontier: accuracy vs computational cost"""
    if not PLOTTING_AVAILABLE:
        return
    
    df = results.get('df')
    if df is None or 'accuracy' not in df.columns:
        print("No suitable data for efficiency frontier plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy vs Token Efficiency
    if 'token_efficiency' in df.columns:
        for condition in df['condition'].unique():
            condition_data = df[df['condition'] == condition]
            ax1.scatter(condition_data['token_efficiency'], condition_data['accuracy'], 
                       label=condition, alpha=0.7, s=50)
        
        ax1.set_xlabel('Token Efficiency (tokens/sec)')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy vs Token Efficiency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Accuracy vs Generation Time
    if 'generation_time' in df.columns:
        for condition in df['condition'].unique():
            condition_data = df[df['condition'] == condition]
            ax2.scatter(condition_data['generation_time'], condition_data['accuracy'], 
                       label=condition, alpha=0.7, s=50)
        
        ax2.set_xlabel('Generation Time (seconds)')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy vs Generation Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'efficiency_frontier.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_breakdown(results: Dict[str, Any], output_dir: str):
    """Plot performance breakdown by task type and difficulty"""
    if not PLOTTING_AVAILABLE:
        return
    
    df = results.get('df')
    if df is None:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy by Task Type
    if 'task_type' in df.columns:
        task_accuracy = df.groupby(['condition', 'task_type'])['is_correct'].mean().unstack()
        task_accuracy.plot(kind='bar', ax=ax1)
        ax1.set_title('Accuracy by Task Type')
        ax1.set_ylabel('Accuracy')
        ax1.legend(title='Task Type')
        ax1.tick_params(axis='x', rotation=45)
    
    # Accuracy by Difficulty
    if 'difficulty' in df.columns:
        diff_accuracy = df.groupby(['condition', 'difficulty'])['is_correct'].mean().unstack()
        diff_accuracy.plot(kind='bar', ax=ax2)
        ax2.set_title('Accuracy by Difficulty')
        ax2.set_ylabel('Accuracy')
        ax2.legend(title='Difficulty')
        ax2.tick_params(axis='x', rotation=45)
    
    # Token Usage by Condition
    if 'num_tokens' in df.columns:
        token_usage = df.groupby('condition')['num_tokens'].mean()
        token_usage.plot(kind='bar', ax=ax3)
        ax3.set_title('Average Token Usage by Condition')
        ax3.set_ylabel('Number of Tokens')
        ax3.tick_params(axis='x', rotation=45)
    
    # Generation Time by Condition
    if 'generation_time' in df.columns:
        time_usage = df.groupby('condition')['generation_time'].mean()
        time_usage.plot(kind='bar', ax=ax4)
        ax4.set_title('Average Generation Time by Condition')
        ax4.set_ylabel('Time (seconds)')
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'performance_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_calibration(results: Dict[str, Any], output_dir: str):
    """Plot calibration curves for confidence vs accuracy"""
    if not PLOTTING_AVAILABLE:
        return
    
    df = results.get('df')
    if df is None or 'confidence' not in df.columns:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Calibration plot
    for i, condition in enumerate(df['condition'].unique()):
        condition_data = df[df['condition'] == condition]
        if condition_data['confidence'].sum() == 0:
            continue
            
        # Create bins for confidence
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        binned_accuracy = []
        binned_confidence = []
        bin_counts = []
        
        for j in range(len(bins) - 1):
            bin_mask = (condition_data['confidence'] >= bins[j]) & (condition_data['confidence'] < bins[j+1])
            bin_data = condition_data[bin_mask]
            
            if len(bin_data) > 0:
                binned_accuracy.append(bin_data['is_correct'].mean())
                binned_confidence.append(bin_data['confidence'].mean())
                bin_counts.append(len(bin_data))
            else:
                binned_accuracy.append(np.nan)
                binned_confidence.append(np.nan)
                bin_counts.append(0)
        
        # Plot calibration curve
        valid_indices = ~np.isnan(binned_accuracy)
        if np.any(valid_indices):
            axes[0].plot(np.array(binned_confidence)[valid_indices], 
                        np.array(binned_accuracy)[valid_indices], 
                        'o-', label=condition, alpha=0.7)
    
    # Perfect calibration line
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    axes[0].set_xlabel('Confidence')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Calibration Plot')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Confidence distribution
    for condition in df['condition'].unique():
        condition_data = df[df['condition'] == condition]
        if condition_data['confidence'].sum() > 0:
            axes[1].hist(condition_data['confidence'], bins=20, alpha=0.6, label=condition)
    
    axes[1].set_xlabel('Confidence')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Confidence Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'calibration.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_report(results: Dict[str, Any], output_dir: str):
    """Create a summary report with key findings"""
    report_lines = []
    report_lines.append("# CoT vs Latent Reasoning Comparison - Summary Report")
    report_lines.append("=" * 60)
    
    df = results.get('df')
    aggregated = results.get('aggregated')
    
    if df is not None:
        report_lines.append(f"\n## Dataset Overview")
        report_lines.append(f"- Total samples: {len(df)}")
        report_lines.append(f"- Conditions tested: {df['condition'].nunique()}")
        report_lines.append(f"- Task types: {', '.join(df['task_type'].unique()) if 'task_type' in df.columns else 'N/A'}")
        
        report_lines.append(f"\n## Overall Results")
        overall_accuracy = df.groupby('condition')['is_correct'].mean()
        for condition, accuracy in overall_accuracy.items():
            report_lines.append(f"- {condition}: {accuracy:.4f} accuracy")
        
        if 'generation_time' in df.columns:
            report_lines.append(f"\n## Efficiency Metrics")
            avg_times = df.groupby('condition')['generation_time'].mean()
            for condition, time in avg_times.items():
                report_lines.append(f"- {condition}: {time:.3f}s avg generation time")
        
        # Best performing condition
        best_condition = overall_accuracy.idxmax()
        best_accuracy = overall_accuracy.max()
        report_lines.append(f"\n## Key Findings")
        report_lines.append(f"- Best performing condition: {best_condition} ({best_accuracy:.4f} accuracy)")
        
        # Efficiency analysis
        if 'num_tokens' in df.columns and 'generation_time' in df.columns:
            df_with_efficiency = df.copy()
            df_with_efficiency['tokens_per_second'] = df_with_efficiency['num_tokens'] / df_with_efficiency['generation_time']
            efficiency_by_condition = df_with_efficiency.groupby('condition')['tokens_per_second'].mean()
            most_efficient = efficiency_by_condition.idxmax()
            report_lines.append(f"- Most efficient condition: {most_efficient} ({efficiency_by_condition.max():.1f} tokens/sec)")
    
    # Save report
    report_path = Path(output_dir) / 'summary_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Summary report saved to: {report_path}")

def main():
    """Main plotting script"""
    parser = argparse.ArgumentParser(description="Plot results from CoT vs Latent reasoning experiment")
    parser.add_argument("results_dir", help="Directory containing experiment results")
    parser.add_argument("--output_dir", help="Directory to save plots", default=None)
    parser.add_argument("--plots", nargs="+", 
                       choices=["accuracy", "efficiency", "breakdown", "calibration", "all"],
                       default=["all"], help="Which plots to generate")
    
    args = parser.parse_args()
    
    if not PLOTTING_AVAILABLE:
        print("Error: matplotlib and seaborn are required for plotting")
        print("Install with: pip install matplotlib seaborn")
        sys.exit(1)
    
    # Load results
    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        sys.exit(1)
    
    # Set output directory
    output_dir = args.output_dir or args.results_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate requested plots
    plots_to_generate = args.plots
    if "all" in plots_to_generate:
        plots_to_generate = ["accuracy", "efficiency", "breakdown", "calibration"]
    
    if "accuracy" in plots_to_generate:
        print("Generating accuracy vs steps plot...")
        plot_accuracy_vs_steps(results, output_dir)
    
    if "efficiency" in plots_to_generate:
        print("Generating efficiency frontier plot...")
        plot_efficiency_frontier(results, output_dir)
    
    if "breakdown" in plots_to_generate:
        print("Generating performance breakdown plot...")
        plot_performance_breakdown(results, output_dir)
    
    if "calibration" in plots_to_generate:
        print("Generating calibration plot...")
        plot_calibration(results, output_dir)
    
    # Always generate summary report
    print("Generating summary report...")
    create_summary_report(results, output_dir)
    
    print(f"Plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
