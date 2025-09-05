# 📊 Comprehensive JSON Logging & Multi-Evaluation Analysis Guide

## 🎯 Overview

The enhanced framework now provides **comprehensive JSON logging** that captures every detail of the reasoning process, enabling thorough inspection and multi-evaluation analysis as requested.

## 🔍 What Gets Logged in JSON

### 📝 Complete Trace Structure

Each evaluation generates a detailed JSON trace containing:

```json
{
  "trace_id": "uuid-unique-identifier",
  "task_id": "task_0001",
  "condition": "cot_zero_shot",
  "dataset": "synthetic_count_clean",
  "seed": 42,
  
  // Input Details
  "audio_path": "/path/to/audio.wav",
  "audio_duration": 3.5,
  "question": "How many beeps do you hear?",
  "ground_truth": "3",
  "task_type": "counting",
  "difficulty": "easy",
  
  // Prompt Generation
  "prompt_generation": {
    "prompt_type": "cot_zero_shot",
    "raw_prompt": "How many beeps do you hear?",
    "processed_prompt": "Let's think step by step. How many beeps do you hear?",
    "prompt_tokens": 12,
    "generation_time": 0.05,
    "timestamp": 1704123456.789
  },
  
  // Detailed Reasoning Steps
  "reasoning_steps": [
    {
      "step_id": "uuid_step_0",
      "step_type": "transcription",
      "timestamp": 1704123456.890,
      "content": "I hear audio with beeping sounds",
      "confidence": 0.85,
      "tokens_used": 8,
      "processing_time": 0.12,
      "metadata": {
        "model_info": {"model_name": "Kimi-Audio-7B"},
        "audio_segment_info": {"duration": 3.5},
        "transcription_quality": {"confidence": 0.85}
      }
    },
    {
      "step_id": "uuid_step_1", 
      "step_type": "reasoning",
      "content": "I need to count each distinct beep sound",
      "confidence": 0.75,
      "tokens_used": 9,
      "metadata": {
        "step_number": 1,
        "intermediate_conclusion": "Starting to count beeps",
        "reasoning_type": "chain_of_thought"
      }
    },
    {
      "step_id": "uuid_step_2",
      "step_type": "latent_iteration", 
      "content": "Latent reasoning iteration 2",
      "confidence": 0.80,
      "metadata": {
        "iteration": 2,
        "internal_state": {"processing_stage": "latent_loop_2"},
        "confidence_evolution": [0.3, 0.5, 0.7, 0.8],
        "reasoning_type": "latent"
      }
    }
  ],
  
  // Model Interactions
  "model_interactions": [
    {
      "interaction_id": "uuid_interaction_0",
      "interaction_type": "generation_start",
      "timestamp": 1704123456.900,
      "input_data": {
        "audio_path": "/path/to/audio.wav",
        "prompt": "...",
        "temperature": 0.0,
        "top_k": 5
      },
      "output_data": {"tokens_generated": 25},
      "timing": {"start_time": 1704123456.900, "end_time": 1704123457.100},
      "resource_usage": {"memory_mb": 512.5, "gpu_utilization": 0.85}
    }
  ],
  
  // Final Results
  "final_answer": "3",
  "is_correct": true,
  "confidence_score": 0.82,
  
  // Performance Metrics
  "total_processing_time": 1.25,
  "total_tokens": 34,
  "memory_peak_mb": 612.3,
  
  // Budget Tracking
  "budget_allocated": {"tokens": 100, "flops": 1e12, "time": 5.0},
  "budget_used": {"tokens": 34, "flops": 3.4e11, "time": 1.25},
  "budget_efficiency": 0.34,
  
  // Timestamps
  "start_time": 1704123456.789,
  "end_time": 1704123458.039,
  "saved_at": "2024-01-01T12:34:58.123456"
}
```

### 📁 File Structure

After running an experiment, you'll find:

```
experiments/results/[experiment_name]_[timestamp]/
├── config.yaml                           # Experiment configuration
├── raw_results.json                      # Standard results
├── aggregated_results.json               # Summary statistics  
├── results.csv                           # Analysis-ready data
├── [experiment_name]_detailed_log.json   # Comprehensive experiment summary
└── detailed_traces/
    ├── traces_index.json                 # Index of all traces
    ├── trace_[uuid1].json                # Individual trace files
    ├── trace_[uuid2].json
    └── trace_[uuid3].json
```

## 🚀 Running Experiments with Detailed Logging

### Quick Test with Logging
```bash
conda activate torch_gpu
python run_reasoning_experiment.py --preset quick_test
```

### Comprehensive Logging Test
```bash
python run_reasoning_experiment.py --config experiments/configs/detailed_logging_test.yaml
```

### Full Research Run with Detailed Traces
```bash
python run_reasoning_experiment.py --preset full_comparison --wandb_project my_research
```

## 🔬 Multi-Evaluation Analysis

### Single Experiment Analysis
```bash
# Analyze traces from one experiment
python experiments/analyze_traces.py experiments/results/quick_test_20240101_120000/

# Generate JSON output instead of markdown
python experiments/analyze_traces.py experiments/results/quick_test_20240101_120000/ --format json
```

### Multi-Experiment Comparison
```bash
# Compare multiple experiment runs
python experiments/analyze_traces.py \
  experiments/results/experiment1_20240101_120000/ \
  --compare experiments/results/experiment2_20240101_130000/ \
            experiments/results/experiment3_20240101_140000/ \
  --output comprehensive_comparison.md
```

### Programmatic Analysis
```python
# Load and analyze traces programmatically
from experiments.analyze_traces import load_experiment_traces, analyze_reasoning_patterns

# Load experiment data
experiment_data = load_experiment_traces("experiments/results/my_experiment_20240101_120000/")

# Analyze reasoning patterns
patterns = analyze_reasoning_patterns(experiment_data["traces"])

# Access detailed traces
for trace_id, trace in experiment_data["traces"].items():
    print(f"Task: {trace['task_id']}")
    print(f"Condition: {trace['condition']}")
    print(f"Correct: {trace['is_correct']}")
    print(f"Reasoning steps: {len(trace['reasoning_steps'])}")
    
    # Examine each reasoning step
    for step in trace["reasoning_steps"]:
        print(f"  {step['step_type']}: {step['content'][:50]}...")
```

## 📊 Analysis Capabilities

### 🧠 Reasoning Pattern Analysis

- **CoT Chain Analysis**: Step-by-step reasoning progression
- **Latent Iteration Tracking**: Internal processing loops
- **Confidence Evolution**: How confidence changes through reasoning
- **Intermediate Conclusions**: Key decision points
- **Token Usage Patterns**: Efficiency analysis per step

### 📈 Performance Deep Dive

- **Processing Time Breakdown**: Time spent in each reasoning phase
- **Memory Usage Tracking**: Peak memory consumption
- **Budget Efficiency**: How well computational budgets were used
- **Error Pattern Analysis**: What goes wrong and when

### 🔄 Multi-Evaluation Insights

- **Consistency Analysis**: How stable are results across runs?
- **Condition Comparison**: Detailed CoT vs Latent analysis
- **Scaling Patterns**: How performance changes with reasoning depth
- **Robustness Testing**: Performance under different noise conditions

## 📋 Generated Analysis Reports

### Automatic Report Sections

1. **Experiment Overview**
   - Total traces, conditions tested, datasets used
   - Task type distribution and difficulty breakdown

2. **Reasoning Patterns Analysis**
   - CoT: Average steps, step types, reasoning depth
   - Latent: Iteration counts, internal states, confidence trends
   - Comparative analysis between approaches

3. **Transcription Quality Analysis**  
   - Intermediate transcription accuracy
   - Confidence vs actual performance correlation
   - Processing time vs audio duration analysis

4. **Multi-Evaluation Comparison**
   - Accuracy trends across experiments
   - Efficiency evolution over time
   - Cross-run consistency metrics

5. **Key Findings & Recommendations**
   - Data-driven insights about reasoning approaches
   - Performance optimization suggestions

## 🎯 Use Cases for Detailed Logging

### 🔍 Research Analysis
- **Understand reasoning mechanisms**: How does each approach actually work?
- **Identify failure modes**: What causes errors in CoT vs Latent reasoning?
- **Optimize prompting strategies**: Which prompt variations work best?
- **Budget allocation**: How to balance accuracy vs computational cost?

### 🛠 Development & Debugging
- **Model behavior inspection**: What is the model actually doing?
- **Performance bottlenecks**: Where does processing time go?
- **Confidence calibration**: How well does the model know when it's right?
- **Comparative debugging**: Why does one approach work better than another?

### 📊 Publication & Reporting
- **Comprehensive evidence**: Detailed data for research papers
- **Reproducible results**: All traces saved for verification
- **Statistical analysis**: Rich data for significance testing
- **Visualization ready**: JSON data easily converted to plots

## 🔧 Advanced Usage

### Custom Trace Analysis
```python
# Custom analysis of specific patterns
def analyze_cot_reasoning_depth(traces):
    depth_analysis = {}
    for trace_id, trace in traces.items():
        if trace["condition"].startswith("cot"):
            steps = [s for s in trace["reasoning_steps"] if s["step_type"] == "reasoning"]
            depth_analysis[trace_id] = {
                "depth": len(steps),
                "accuracy": trace["is_correct"],
                "confidence_progression": [s["confidence"] for s in steps],
                "intermediate_conclusions": [
                    s["metadata"].get("intermediate_conclusion", "") 
                    for s in steps
                ]
            }
    return depth_analysis
```

### Batch Processing Multiple Experiments
```python
import glob
from pathlib import Path

# Analyze all experiments in results directory
experiment_dirs = glob.glob("experiments/results/*/")
comparative_analysis = {}

for exp_dir in experiment_dirs:
    exp_name = Path(exp_dir).name
    try:
        exp_data = load_experiment_traces(exp_dir)
        comparative_analysis[exp_name] = analyze_reasoning_patterns(exp_data["traces"])
    except Exception as e:
        print(f"Skipped {exp_name}: {e}")

# Generate cross-experiment insights
print("Accuracy comparison across experiments:")
for exp_name, analysis in comparative_analysis.items():
    if "comparison" in analysis:
        comp = analysis["comparison"]["accuracy_comparison"]
        print(f"{exp_name}: CoT={comp['cot_accuracy']:.3f}, Latent={comp['latent_accuracy']:.3f}")
```

## ✅ Validation & Testing

The detailed logging system has been thoroughly tested:

```bash
# Run comprehensive test suite
python test_reasoning_experiment.py
# ✅ All tests passed! Including detailed JSON logging validation
```

## 🎉 Summary

The enhanced framework now provides:

✅ **Complete JSON logging** of every reasoning step and decision  
✅ **Intermediate transcription capture** with quality metrics  
✅ **Multi-evaluation comparison** tools for longitudinal analysis  
✅ **Automated analysis scripts** for pattern detection  
✅ **Rich trace inspection** capabilities for debugging  
✅ **Research-ready data** for publication and reproducibility  

**You can now run multiple evaluations and inspect every detail of the reasoning process to understand exactly how CoT vs Latent approaches differ in their decision-making patterns!** 🚀
