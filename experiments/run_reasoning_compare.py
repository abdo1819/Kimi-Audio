#!/usr/bin/env python3
"""
CoT vs Latent Reasoning Comparison Experiment for Audio-Text LLMs

Implements the experiment design from the research document to compare
Chain-of-Thought (CoT) reasoning vs Latent-space reasoning on audio understanding tasks.

Usage:
    python experiments/run_reasoning_compare.py --config experiments/configs/basic_comparison.yaml
    python experiments/run_reasoning_compare.py --preset quick_test
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import yaml
from dataclasses import dataclass, asdict
import wandb
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kimia_infer.api.kimia import KimiAudio
from experiments.reasoning_framework.model_backends import AudioTextModelBackend, KimiaBackend
from experiments.reasoning_framework.prompting import CoTPrompter, LatentPrompter
from experiments.reasoning_framework.datasets import ReasoningDatasetLoader
from experiments.reasoning_framework.evaluation import ReasoningEvaluator
from experiments.reasoning_framework.budget_controller import ComputeBudgetController
from experiments.reasoning_framework.utils import set_seed, get_device_info

logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for the CoT vs Latent reasoning experiment"""
    
    # Experiment metadata
    name: str = "cot_vs_latent_comparison"
    description: str = "Compare CoT and latent reasoning on audio tasks"
    output_dir: str = "experiments/results"
    
    # Model configuration
    model_backend: str = "kimia"  # kimia, qwen2_audio
    model_path: str = "moonshotai/Kimi-Audio-7B-Instruct"
    load_detokenizer: bool = True
    
    # Reasoning conditions
    conditions: List[str] = None  # ["cot_zero_shot", "cot_few_shot", "cot_descriptive", "latent_silent", "latent_loops"]
    
    # CoT parameters
    cot_chain_lengths: List[int] = None  # [0, 16, 64] tokens
    cot_self_consistency_k: List[int] = None  # [1, 5] samples
    cot_use_skill_retrieval: bool = True
    
    # Latent parameters  
    latent_loop_depths: List[int] = None  # [0, 2, 4, 8] loops
    latent_use_timestep_embeddings: bool = False
    
    # Compute budget matching
    budget_matching: str = "flops"  # "flops", "tokens", "time"
    max_total_tokens: int = 1000
    max_generation_time: float = 30.0  # seconds
    
    # Dataset configuration
    datasets: List[str] = None  # ["librispeech_qa", "temporal_reasoning", "synthetic_count"]
    max_samples_per_dataset: int = 100
    noise_conditions: List[str] = None  # ["clean", "snr_10", "snr_5"]
    
    # Evaluation parameters
    metrics: List[str] = None  # ["accuracy", "wer", "cer", "token_efficiency", "latency", "calibration"]
    eval_batch_size: int = 1
    
    # Sampling parameters
    temperature: float = 0.0
    top_k: int = 5
    seeds: List[int] = None  # [42, 123, 456] for reproducibility
    
    # Logging and output
    wandb_project: str = "audio_reasoning_comparison"
    wandb_run_name: Optional[str] = None
    save_predictions: bool = True
    save_traces: bool = True
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Set defaults for list fields"""
        if self.conditions is None:
            self.conditions = ["cot_zero_shot", "cot_few_shot", "latent_silent"]
        if self.cot_chain_lengths is None:
            self.cot_chain_lengths = [0, 16, 64]
        if self.cot_self_consistency_k is None:
            self.cot_self_consistency_k = [1, 3]
        if self.latent_loop_depths is None:
            self.latent_loop_depths = [0, 2, 4]
        if self.datasets is None:
            self.datasets = ["librispeech_qa", "synthetic_count"]
        if self.noise_conditions is None:
            self.noise_conditions = ["clean"]
        if self.metrics is None:
            self.metrics = ["accuracy", "token_efficiency", "latency"]
        if self.seeds is None:
            self.seeds = [42, 123, 456]

# Configuration presets
PRESETS = {
    "quick_test": ExperimentConfig(
        name="quick_test",
        description="Quick test with minimal settings",
        conditions=["cot_zero_shot", "latent_silent"],
        cot_chain_lengths=[0, 16],
        latent_loop_depths=[0, 2],
        datasets=["synthetic_count"],
        max_samples_per_dataset=10,
        seeds=[42],
        max_total_tokens=200,
    ),
    
    "full_comparison": ExperimentConfig(
        name="full_comparison",
        description="Full comparison across all conditions",
        conditions=["cot_zero_shot", "cot_few_shot", "cot_descriptive", "latent_silent", "latent_loops"],
        cot_chain_lengths=[0, 16, 64],
        cot_self_consistency_k=[1, 5],
        latent_loop_depths=[0, 2, 4, 8],
        datasets=["librispeech_qa", "temporal_reasoning", "synthetic_count"],
        max_samples_per_dataset=500,
        noise_conditions=["clean", "snr_10"],
        seeds=[42, 123, 456],
    ),
    
    "efficiency_focus": ExperimentConfig(
        name="efficiency_focus",
        description="Focus on compute efficiency comparisons",
        conditions=["cot_zero_shot", "latent_silent"],
        budget_matching="flops",
        max_total_tokens=500,
        datasets=["synthetic_count", "temporal_reasoning"],
        max_samples_per_dataset=200,
        metrics=["accuracy", "token_efficiency", "latency", "flops"],
    )
}

class ReasoningExperiment:
    """Main experiment runner for CoT vs Latent reasoning comparison"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {}
        self.setup_logging()
        self.setup_output_dirs()
        
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger.info(f"Initialized experiment: {self.config.name}")
        
    def setup_output_dirs(self):
        """Create output directories"""
        self.output_dir = Path(self.config.output_dir) / self.config.name / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / "config.yaml", "w") as f:
            yaml.dump(asdict(self.config), f, default_flow_style=False)
        
        logger.info(f"Output directory: {self.output_dir}")
    
    def initialize_wandb(self):
        """Initialize Weights & Biases logging"""
        if self.config.wandb_project:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name or self.config.name,
                config=asdict(self.config),
                dir=str(self.output_dir)
            )
            logger.info("Initialized W&B logging")
    
    def load_model_backend(self) -> AudioTextModelBackend:
        """Load the appropriate model backend"""
        logger.info(f"Loading model backend: {self.config.model_backend}")
        
        if self.config.model_backend == "kimia":
            return KimiaBackend(
                model_path=self.config.model_path,
                load_detokenizer=self.config.load_detokenizer
            )
        else:
            raise ValueError(f"Unknown model backend: {self.config.model_backend}")
    
    def load_datasets(self) -> Dict[str, Any]:
        """Load all configured datasets"""
        logger.info("Loading datasets...")
        loader = ReasoningDatasetLoader()
        datasets = {}
        
        for dataset_name in self.config.datasets:
            for noise_condition in self.config.noise_conditions:
                key = f"{dataset_name}_{noise_condition}"
                datasets[key] = loader.load_dataset(
                    dataset_name=dataset_name,
                    noise_condition=noise_condition,
                    max_samples=self.config.max_samples_per_dataset
                )
                logger.info(f"Loaded {len(datasets[key])} samples for {key}")
        
        return datasets
    
    def run_condition(self, condition: str, dataset_key: str, dataset: Any, 
                     model_backend: AudioTextModelBackend, seed: int) -> Dict[str, Any]:
        """Run a single experimental condition"""
        logger.info(f"Running condition: {condition} on {dataset_key} (seed={seed})")
        set_seed(seed)
        
        # Initialize prompter based on condition
        if condition.startswith("cot"):
            prompter = CoTPrompter(
                condition_type=condition,
                chain_lengths=self.config.cot_chain_lengths,
                self_consistency_k=self.config.cot_self_consistency_k,
                use_skill_retrieval=self.config.cot_use_skill_retrieval
            )
        elif condition.startswith("latent"):
            prompter = LatentPrompter(
                condition_type=condition,
                loop_depths=self.config.latent_loop_depths,
                use_timestep_embeddings=self.config.latent_use_timestep_embeddings
            )
        else:
            raise ValueError(f"Unknown condition: {condition}")
        
        # Initialize budget controller
        budget_controller = ComputeBudgetController(
            matching_strategy=self.config.budget_matching,
            max_tokens=self.config.max_total_tokens,
            max_time=self.config.max_generation_time
        )
        
        # Initialize evaluator
        evaluator = ReasoningEvaluator(
            metrics=self.config.metrics,
            save_predictions=self.config.save_predictions,
            save_traces=self.config.save_traces
        )
        
        # Run evaluation
        results = evaluator.evaluate(
            model_backend=model_backend,
            prompter=prompter,
            dataset=dataset,
            budget_controller=budget_controller,
            batch_size=self.config.eval_batch_size,
            temperature=self.config.temperature,
            top_k=self.config.top_k
        )
        
        # Add metadata
        results["condition"] = condition
        results["dataset"] = dataset_key
        results["seed"] = seed
        results["timestamp"] = datetime.now().isoformat()
        
        return results
    
    def aggregate_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across seeds and conditions"""
        logger.info("Aggregating results...")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(all_results)
        
        # Group by condition and dataset
        aggregated = {}
        
        for (condition, dataset), group in df.groupby(["condition", "dataset"]):
            key = f"{condition}_{dataset}"
            
            # Aggregate metrics across seeds
            agg_metrics = {}
            for metric in self.config.metrics:
                if metric in group.columns:
                    values = group[metric].dropna()
                    if len(values) > 0:
                        agg_metrics[f"{metric}_mean"] = float(values.mean())
                        agg_metrics[f"{metric}_std"] = float(values.std())
                        agg_metrics[f"{metric}_min"] = float(values.min())
                        agg_metrics[f"{metric}_max"] = float(values.max())
            
            aggregated[key] = {
                "condition": condition,
                "dataset": dataset,
                "num_seeds": len(group),
                "metrics": agg_metrics
            }
        
        return aggregated
    
    def save_results(self, all_results: List[Dict[str, Any]], aggregated_results: Dict[str, Any]):
        """Save results to files"""
        logger.info("Saving results...")
        
        # Save raw results
        results_file = self.output_dir / "raw_results.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Save aggregated results
        agg_file = self.output_dir / "aggregated_results.json"
        with open(agg_file, "w") as f:
            json.dump(aggregated_results, f, indent=2, default=str)
        
        # Save as CSV for easy analysis
        df = pd.DataFrame(all_results)
        csv_file = self.output_dir / "results.csv"
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def log_to_wandb(self, aggregated_results: Dict[str, Any]):
        """Log aggregated results to W&B"""
        if not self.config.wandb_project:
            return
            
        logger.info("Logging results to W&B...")
        
        # Log summary metrics
        for key, result in aggregated_results.items():
            condition = result["condition"]
            dataset = result["dataset"]
            
            # Log each metric
            for metric_name, value in result["metrics"].items():
                wandb.log({
                    f"{condition}/{dataset}/{metric_name}": value
                })
        
        # Create comparison tables
        self.create_wandb_tables(aggregated_results)
    
    def create_wandb_tables(self, aggregated_results: Dict[str, Any]):
        """Create W&B tables for result comparison"""
        # Accuracy comparison table
        accuracy_data = []
        for key, result in aggregated_results.items():
            if "accuracy_mean" in result["metrics"]:
                accuracy_data.append([
                    result["condition"],
                    result["dataset"], 
                    result["metrics"]["accuracy_mean"],
                    result["metrics"].get("accuracy_std", 0.0)
                ])
        
        if accuracy_data:
            accuracy_table = wandb.Table(
                columns=["Condition", "Dataset", "Accuracy_Mean", "Accuracy_Std"],
                data=accuracy_data
            )
            wandb.log({"accuracy_comparison": accuracy_table})
        
        # Efficiency comparison table
        efficiency_data = []
        for key, result in aggregated_results.items():
            if "token_efficiency_mean" in result["metrics"]:
                efficiency_data.append([
                    result["condition"],
                    result["dataset"],
                    result["metrics"]["token_efficiency_mean"],
                    result["metrics"].get("latency_mean", 0.0)
                ])
        
        if efficiency_data:
            efficiency_table = wandb.Table(
                columns=["Condition", "Dataset", "Token_Efficiency", "Latency"],
                data=efficiency_data
            )
            wandb.log({"efficiency_comparison": efficiency_table})
    
    def run(self):
        """Run the complete experiment"""
        logger.info(f"Starting experiment: {self.config.name}")
        start_time = time.time()
        
        # Initialize W&B
        self.initialize_wandb()
        
        # Load model backend
        model_backend = self.load_model_backend()
        
        # Load datasets
        datasets = self.load_datasets()
        
        # Run all conditions
        all_results = []
        total_runs = len(self.config.conditions) * len(datasets) * len(self.config.seeds)
        run_count = 0
        
        for condition in self.config.conditions:
            for dataset_key, dataset in datasets.items():
                for seed in self.config.seeds:
                    run_count += 1
                    logger.info(f"Progress: {run_count}/{total_runs}")
                    
                    try:
                        result = self.run_condition(
                            condition=condition,
                            dataset_key=dataset_key,
                            dataset=dataset,
                            model_backend=model_backend,
                            seed=seed
                        )
                        all_results.append(result)
                        
                        # Log intermediate results to W&B
                        if self.config.wandb_project:
                            wandb.log({
                                f"progress/{condition}_{dataset_key}_seed{seed}": result.get("accuracy", 0.0)
                            })
                    
                    except Exception as e:
                        logger.error(f"Error in condition {condition}, dataset {dataset_key}, seed {seed}: {e}")
                        continue
        
        # Aggregate and save results
        aggregated_results = self.aggregate_results(all_results)
        self.save_results(all_results, aggregated_results)
        self.log_to_wandb(aggregated_results)
        
        # Print summary
        total_time = time.time() - start_time
        logger.info(f"Experiment completed in {total_time:.1f}s")
        logger.info(f"Results saved to: {self.output_dir}")
        
        if self.config.wandb_project:
            wandb.finish()
        
        return aggregated_results

def load_config(config_path: str) -> ExperimentConfig:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return ExperimentConfig(**config_dict)

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="CoT vs Latent Reasoning Comparison Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/run_reasoning_compare.py --preset quick_test
  python experiments/run_reasoning_compare.py --config experiments/configs/full_comparison.yaml
  python experiments/run_reasoning_compare.py --preset efficiency_focus --wandb_project my_project
        """
    )
    
    # Configuration options
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--config", type=str, help="Path to YAML config file")
    config_group.add_argument("--preset", choices=list(PRESETS.keys()), help="Use predefined preset")
    
    # Override options
    parser.add_argument("--name", type=str, help="Override experiment name")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--wandb_project", type=str, help="Override W&B project name")
    parser.add_argument("--wandb_run_name", type=str, help="Override W&B run name")
    parser.add_argument("--max_samples", type=int, help="Override max samples per dataset")
    parser.add_argument("--seeds", type=int, nargs="+", help="Override random seeds")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Set logging level")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = PRESETS[args.preset]
    
    # Apply overrides
    if args.name:
        config.name = args.name
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.wandb_project:
        config.wandb_project = args.wandb_project
    if args.wandb_run_name:
        config.wandb_run_name = args.wandb_run_name
    if args.max_samples:
        config.max_samples_per_dataset = args.max_samples
    if args.seeds:
        config.seeds = args.seeds
    config.log_level = args.log_level
    
    # Run experiment
    experiment = ReasoningExperiment(config)
    results = experiment.run()
    
    print(f"\n🎯 Experiment completed successfully!")
    print(f"📊 Results saved to: {experiment.output_dir}")
    
    if config.wandb_project:
        print(f"🔗 W&B dashboard: https://wandb.ai/{config.wandb_project}")

if __name__ == "__main__":
    main()
