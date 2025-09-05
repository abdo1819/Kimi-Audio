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

def test_detailed_logging():
    """Test detailed JSON logging functionality"""
    print("\nTesting detailed JSON logging...")
    
    import tempfile
    import os
    from reasoning_framework.detailed_logger import create_detailed_logger
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize detailed logger
        logger = create_detailed_logger(temp_dir, "test_experiment")
        
        # Start a trace
        trace_id = logger.start_evaluation_trace(
            task_id="test_task_1",
            condition="cot_zero_shot",
            dataset="test_dataset",
            seed=42,
            audio_path="/fake/audio.wav",
            question="How many beeps?",
            ground_truth="3",
            task_type="counting",
            difficulty="easy"
        )
        
        # Log some steps
        logger.log_prompt_generation(
            trace_id=trace_id,
            prompt_type="cot_zero_shot",
            raw_prompt="How many beeps?",
            processed_prompt="Let's think step by step. How many beeps?",
            prompt_tokens=10,
            generation_time=0.1
        )
        
        logger.log_reasoning_chain_step(
            trace_id=trace_id,
            step_number=1,
            reasoning_text="I need to count the beeps in the audio",
            intermediate_conclusion="Starting to count",
            confidence=0.7,
            tokens_used=8
        )
        
        logger.log_budget_usage(
            trace_id=trace_id,
            budget_type="tokens",
            allocated=100,
            used=18,
            efficiency=0.18
        )
        
        # Complete the trace
        logger.complete_evaluation_trace(
            trace_id=trace_id,
            final_answer="3",
            is_correct=True,
            confidence_score=0.8,
            total_tokens=18,
            memory_peak_mb=512.0
        )
        
        # Save experiment summary
        logger.save_experiment_summary()
        
        # Verify files were created
        traces_dir = Path(temp_dir) / "detailed_traces"
        assert traces_dir.exists()
        assert (traces_dir / "traces_index.json").exists()
        assert (traces_dir / f"trace_{trace_id}.json").exists()
        assert (Path(temp_dir) / "test_experiment_detailed_log.json").exists()
        
        # Verify trace content
        with open(traces_dir / f"trace_{trace_id}.json", 'r') as f:
            trace_data = json.load(f)
            assert trace_data["task_id"] == "test_task_1"
            assert trace_data["condition"] == "cot_zero_shot"
            assert trace_data["final_answer"] == "3"
            assert trace_data["is_correct"] == True
            assert len(trace_data["reasoning_steps"]) > 0
        
        print("✓ Detailed JSON logging works")

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
        test_detailed_logging()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed! The framework is ready to use.")
        print("\nTo run experiments with detailed JSON logging:")
        print("  python run_reasoning_experiment.py --preset quick_test")
        print("  python run_reasoning_experiment.py --config experiments/configs/detailed_logging_test.yaml")
        print("\nTo analyze detailed traces:")
        print("  python experiments/analyze_traces.py experiments/results/[experiment_dir]/")
        print("  python experiments/analyze_traces.py experiments/results/exp1/ --compare experiments/results/exp2/")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
