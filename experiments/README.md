# CoT vs Latent Reasoning Comparison Experiment

This experiment framework implements the research design described in the document "Latent vs Chain-of-Thought Reasoning in Audio-Text LLMs" to compare Chain-of-Thought (CoT) reasoning with latent-space reasoning on audio understanding tasks.

## Overview

The framework provides:
- **Fair comparison** between CoT and latent reasoning with compute budget matching
- **Comprehensive evaluation** on audio QA, temporal reasoning, and counting tasks
- **Robust metrics** including accuracy, efficiency, calibration, and statistical analysis
- **Flexible configuration** with presets for different experimental scenarios

## Quick Start

### 1. Installation

```bash
# Install additional dependencies for the experiment framework
pip install wandb matplotlib seaborn evaluate datasets
```

### 2. Run a Quick Test

```bash
# Run a minimal test to verify the setup
python experiments/run_reasoning_compare.py --preset quick_test

# Run with W&B logging
python experiments/run_reasoning_compare.py --preset quick_test --wandb_project my_audio_reasoning
```

### 3. Full Comparison

```bash
# Run comprehensive comparison across all conditions
python experiments/run_reasoning_compare.py --preset full_comparison
```

## Experimental Conditions

### Chain-of-Thought (CoT) Conditions
- **`cot_zero_shot`**: "Let's think step by step" prompting
- **`cot_few_shot`**: Few-shot examples with reasoning chains
- **`cot_descriptive`**: Audio description followed by reasoning (Desp-CoT)

### Latent Reasoning Conditions
- **`latent_silent`**: Internal reasoning without explicit steps
- **`latent_loops`**: Emulated reasoning loops with depth control

## Datasets

### Available Datasets
1. **`librispeech_qa`**: QA tasks generated from LibriSpeech audio
2. **`synthetic_count`**: Counting tasks with synthetic audio
3. **`temporal_reasoning`**: Temporal sequence understanding
4. **`speaker_reasoning`**: Multi-speaker identification tasks

### Noise Conditions
- `clean`: No added noise
- `snr_15`, `snr_10`, `snr_5`: Different signal-to-noise ratios

## Configuration

### Using Presets

```bash
# Quick development test
python experiments/run_reasoning_compare.py --preset quick_test

# Full research comparison
python experiments/run_reasoning_compare.py --preset full_comparison

# Efficiency-focused evaluation
python experiments/run_reasoning_compare.py --preset efficiency_focus
```

### Custom Configuration

Create a YAML configuration file:

```yaml
# experiments/configs/my_experiment.yaml
name: "my_custom_experiment"
description: "Custom experiment description"

# Model settings
model_backend: "kimia"
model_path: "moonshotai/Kimi-Audio-7B-Instruct"

# Conditions to test
conditions:
  - "cot_zero_shot"
  - "latent_silent"

# Budget matching
budget_matching: "flops"  # "tokens", "flops", or "time"
max_total_tokens: 500

# Datasets
datasets:
  - "synthetic_count"
  - "temporal_reasoning"
max_samples_per_dataset: 50

# Evaluation
metrics:
  - "accuracy"
  - "token_efficiency"
  - "latency"

# Reproducibility
seeds: [42, 123, 456]
```

Run with:
```bash
python experiments/run_reasoning_compare.py --config experiments/configs/my_experiment.yaml
```

## Budget Matching

The framework ensures fair comparison by matching computational budgets:

- **Token Budget**: Match total number of output tokens
- **FLOP Budget**: Match estimated floating-point operations
- **Time Budget**: Match wall-clock generation time

Example: If CoT uses 100 tokens for reasoning chains, latent reasoning gets equivalent computational budget through internal loops.

## Evaluation Metrics

### Accuracy Metrics
- **Overall accuracy**: Percentage of correct answers
- **Accuracy by task type**: Performance on different reasoning tasks
- **Accuracy by difficulty**: Performance on easy/medium/hard tasks

### Efficiency Metrics
- **Token efficiency**: Tokens generated per second
- **Generation time**: Wall-clock time for inference
- **Memory usage**: Peak GPU memory consumption

### Quality Metrics
- **WER/CER**: Word/Character Error Rate for text similarity
- **Calibration**: Alignment between confidence and correctness
- **Statistical significance**: Confidence intervals and p-values

## Results Analysis

### Automatic Plotting

```bash
# Generate all plots from experiment results
python experiments/plot_results.py experiments/results/my_experiment_20240101_120000/

# Generate specific plots
python experiments/plot_results.py experiments/results/my_experiment_20240101_120000/ --plots accuracy efficiency
```

### Generated Visualizations
- **Accuracy vs Steps**: Performance scaling with reasoning depth
- **Efficiency Frontier**: Accuracy per computational unit
- **Performance Breakdown**: Results by task type and difficulty
- **Calibration Curves**: Confidence vs actual accuracy

### W&B Integration

The framework automatically logs to Weights & Biases:
- Real-time metrics during training
- Comparison tables across conditions
- Interactive plots and dashboards

## Expected Results

Based on the research hypothesis:

### CoT Reasoning
- ✅ **Strengths**: Interpretable, good on easier tasks, benefits from self-consistency
- ⚠️ **Weaknesses**: Verbose, higher latency, may degrade on complex audio tasks

### Latent Reasoning  
- ✅ **Strengths**: Efficient, stable on hard tasks, better token/accuracy ratio
- ⚠️ **Weaknesses**: Less interpretable, requires careful prompt design

### Key Comparisons
1. **Accuracy**: Similar performance with different scaling patterns
2. **Efficiency**: Latent reasoning achieves better accuracy per token
3. **Robustness**: Latent reasoning more stable under noise
4. **Scalability**: Both benefit from more "reasoning steps" but differently

## Advanced Usage

### Custom Model Backend

```python
# experiments/reasoning_framework/model_backends.py
class MyModelBackend(AudioTextModelBackend):
    def generate(self, audio_path: str, text_prompt: str, **kwargs):
        # Implement your model's generation logic
        pass
```

### Custom Dataset

```python
# experiments/reasoning_framework/datasets.py
def load_my_dataset(noise_condition: str, max_samples: int):
    # Load your custom reasoning dataset
    tasks = []
    # ... create ReasoningTask objects
    return tasks
```

### Custom Evaluation Metrics

```python
# experiments/reasoning_framework/evaluation.py
def my_custom_metric(predicted: str, true: str) -> float:
    # Implement your custom evaluation metric
    pass
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   ```bash
   # Reduce batch size or token limits
   python experiments/run_reasoning_compare.py --preset quick_test --max_samples 10
   ```

2. **LibriSpeech download fails**:
   ```bash
   # Set auto-download environment variable
   export LIBRISPEECH_AUTO_DOWNLOAD=true
   ```

3. **Missing dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install wandb matplotlib seaborn evaluate datasets
   ```

### Debug Mode

```bash
# Run with debug logging
python experiments/run_reasoning_compare.py --preset quick_test --log_level DEBUG
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{cot_latent_audio_comparison,
  title={CoT vs Latent Reasoning Comparison Framework for Audio-Text LLMs},
  author={Audio Reasoning Research},
  year={2024},
  url={https://github.com/your-repo/audio-reasoning-comparison}
}
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
