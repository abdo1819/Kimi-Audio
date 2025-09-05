#!/usr/bin/env python3
"""
Simple test script for the CoT vs Latent Reasoning Experiment Framework

Tests basic functionality without requiring the full Kimia model.
"""

import sys
import tempfile
import os
from pathlib import Path
import json

# Add experiments to path
sys.path.insert(0, str(Path(__file__).parent / "experiments"))

def test_config_loading():
    """Test configuration loading"""
    print("Testing configuration loading...")
    
    from run_reasoning_compare import ExperimentConfig, PRESETS, load_config
    
    # Test preset loading
    config = PRESETS["quick_test"]
    assert config.name == "quick_test"
    assert "cot_zero_shot" in config.conditions
    assert "latent_silent" in config.conditions
    print("✓ Preset loading works")
    
    # Test YAML config loading
    config_path = Path("experiments/configs/quick_test.yaml")
    if config_path.exists():
        config = load_config(str(config_path))
        assert config.name == "quick_test"
        print("✓ YAML config loading works")
    else:
        print("⚠ YAML config file not found, skipping YAML test")

def test_dataset_loading():
    """Test dataset loading (mock mode)"""
    print("\nTesting dataset loading...")
    
    from reasoning_framework.datasets import ReasoningDatasetLoader
    
    loader = ReasoningDatasetLoader()
    
    # Test synthetic counting dataset
    tasks = loader.load_dataset("synthetic_count", "clean", max_samples=3)
    assert len(tasks) == 3
    assert all(task.task_type == "counting" for task in tasks)
    print("✓ Synthetic counting dataset works")
    
    # Test temporal reasoning dataset
    tasks = loader.load_dataset("temporal_reasoning", "clean", max_samples=2)
    assert len(tasks) == 2
    assert all(task.task_type == "temporal" for task in tasks)
    print("✓ Temporal reasoning dataset works")

def test_prompting():
    """Test prompting strategies"""
    print("\nTesting prompting strategies...")
    
    from reasoning_framework.prompting import CoTPrompter, LatentPrompter
    
    # Test CoT prompter
    cot_prompter = CoTPrompter("cot_zero_shot")
    prompt = cot_prompter.generate_prompt(
        "How many beeps do you hear?", 
        {"audio_description": "Three beeps in sequence"}
    )
    assert "step by step" in prompt.lower()
    print("✓ CoT prompting works")
    
    # Test latent prompter
    latent_prompter = LatentPrompter("latent_silent")
    prompt = latent_prompter.generate_prompt(
        "How many beeps do you hear?",
        {"audio_description": "Three beeps in sequence"}
    )
    assert "internal" in prompt.lower() or "reasoning" in prompt.lower()
    print("✓ Latent prompting works")

def test_budget_controller():
    """Test budget controller"""
    print("\nTesting budget controller...")
    
    from reasoning_framework.budget_controller import ComputeBudgetController
    
    controller = ComputeBudgetController("tokens", max_tokens=100)
    
    # Test budget calculation
    cot_budget = controller.calculate_cot_budget(chain_length=32, self_consistency_k=1)
    latent_budget = controller.calculate_latent_budget(loop_depth=2)
    
    assert cot_budget.max_tokens > 0
    assert latent_budget.max_tokens > 0
    print("✓ Budget controller works")

def test_evaluation():
    """Test evaluation framework"""
    print("\nTesting evaluation framework...")
    
    from reasoning_framework.evaluation import ReasoningEvaluator, EvaluationResult
    from reasoning_framework.datasets import ReasoningTask
    
    evaluator = ReasoningEvaluator()
    
    # Test correctness checking
    result = EvaluationResult(
        task_id="test_1",
        predicted_answer="3",
        true_answer="3",
        is_correct=True,
        task_type="counting"
    )
    
    # Test answer extraction and normalization
    correct = evaluator._check_correctness("Three", "3", "counting")
    assert correct  # Should handle number word conversion
    print("✓ Evaluation framework works")

def test_mock_experiment():
    """Test running a minimal mock experiment"""
    print("\nTesting mock experiment run...")
    
    # Create a minimal mock backend for testing
    class MockBackend:
        def __init__(self):
            self.model_name = "MockModel"
        
        def generate(self, audio_path, text_prompt, **kwargs):
            from reasoning_framework.model_backends import GenerationStats
            return GenerationStats(
                output_text="3",
                num_output_tokens=1,
                generation_time=0.1,
                reasoning_steps=kwargs.get("reasoning_steps", 0)
            )
        
        def get_model_info(self):
            return {"model_name": "MockModel"}
    
    from reasoning_framework.datasets import ReasoningDatasetLoader
    from reasoning_framework.prompting import CoTPrompter
    from reasoning_framework.evaluation import ReasoningEvaluator
    from reasoning_framework.budget_controller import ComputeBudgetController
    
    # Create components
    backend = MockBackend()
    prompter = CoTPrompter("cot_zero_shot")
    evaluator = ReasoningEvaluator()
    budget_controller = ComputeBudgetController()
    
    # Load small dataset
    loader = ReasoningDatasetLoader()
    dataset = loader.load_dataset("synthetic_count", "clean", max_samples=2)
    
    # Run evaluation
    results = evaluator.evaluate(
        model_backend=backend,
        prompter=prompter,
        dataset=dataset,
        budget_controller=budget_controller,
        batch_size=1
    )
    
    assert "individual_results" in results
    assert "aggregated_results" in results
    assert len(results["individual_results"]) == 2
    print("✓ Mock experiment run works")

def main():
    """Run all tests"""
    print("🧪 Testing CoT vs Latent Reasoning Experiment Framework")
    print("=" * 60)
    
    try:
        test_config_loading()
        test_dataset_loading()
        test_prompting()
        test_budget_controller()
        test_evaluation()
        test_mock_experiment()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed! The framework is ready to use.")
        print("\nTo run a real experiment:")
        print("  python run_reasoning_experiment.py --preset quick_test")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
