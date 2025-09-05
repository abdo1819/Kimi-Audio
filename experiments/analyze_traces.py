#!/usr/bin/env python3
"""
JSON Trace Analysis Tool for CoT vs Latent Reasoning Experiments

Analyze detailed traces from experiments to extract insights:
- Compare reasoning patterns between conditions
- Analyze intermediate transcriptions and decisions
- Generate detailed comparison reports
- Support multi-evaluation inspection
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

def load_experiment_traces(experiment_dir: str) -> Dict[str, Any]:
    """Load all traces from an experiment directory"""
    
    exp_path = Path(experiment_dir)
    traces_dir = exp_path / "detailed_traces"
    
    if not traces_dir.exists():
        raise FileNotFoundError(f"No detailed traces found in {experiment_dir}")
    
    # Load traces index
    index_file = traces_dir / "traces_index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"Traces index not found: {index_file}")
    
    with open(index_file, 'r') as f:
        index_data = json.load(f)
    
    # Load individual traces
    traces = {}
    for trace_summary in index_data["traces"]:
        trace_id = trace_summary["trace_id"]
        trace_file = traces_dir / trace_summary["file_path"]
        
        if trace_file.exists():
            with open(trace_file, 'r') as f:
                trace_data = json.load(f)
                traces[trace_id] = trace_data
    
    # Load experiment summary
    summary_file = exp_path / f"{index_data['experiment_name']}_detailed_log.json"
    experiment_summary = {}
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            experiment_summary = json.load(f)
    
    return {
        "index": index_data,
        "traces": traces,
        "summary": experiment_summary,
        "experiment_dir": str(exp_path)
    }

def analyze_reasoning_patterns(traces: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze reasoning patterns across different conditions"""
    
    patterns = {
        "cot_patterns": {},
        "latent_patterns": {},
        "comparison": {}
    }
    
    # Group traces by condition
    cot_traces = []
    latent_traces = []
    
    for trace_id, trace in traces.items():
        if trace["condition"].startswith("cot"):
            cot_traces.append(trace)
        elif trace["condition"].startswith("latent"):
            latent_traces.append(trace)
    
    # Analyze CoT patterns
    if cot_traces:
        patterns["cot_patterns"] = {
            "avg_reasoning_steps": np.mean([len(t["reasoning_steps"]) for t in cot_traces]),
            "step_types": _analyze_step_types(cot_traces),
            "reasoning_depth": _analyze_reasoning_depth(cot_traces),
            "intermediate_conclusions": _extract_intermediate_conclusions(cot_traces),
            "confidence_evolution": _analyze_confidence_evolution(cot_traces)
        }
    
    # Analyze Latent patterns
    if latent_traces:
        patterns["latent_patterns"] = {
            "avg_iterations": np.mean([len([s for s in t["reasoning_steps"] if s["step_type"] == "latent_iteration"]) for t in latent_traces]),
            "internal_states": _analyze_internal_states(latent_traces),
            "confidence_trends": _analyze_latent_confidence_trends(latent_traces),
            "processing_efficiency": _analyze_processing_efficiency(latent_traces)
        }
    
    # Compare patterns
    if cot_traces and latent_traces:
        patterns["comparison"] = {
            "accuracy_comparison": {
                "cot_accuracy": np.mean([t["is_correct"] for t in cot_traces]),
                "latent_accuracy": np.mean([t["is_correct"] for t in latent_traces])
            },
            "efficiency_comparison": {
                "cot_avg_time": np.mean([t["total_processing_time"] for t in cot_traces]),
                "latent_avg_time": np.mean([t["total_processing_time"] for t in latent_traces]),
                "cot_avg_tokens": np.mean([t["total_tokens"] for t in cot_traces]),
                "latent_avg_tokens": np.mean([t["total_tokens"] for t in latent_traces])
            },
            "reasoning_style_differences": _compare_reasoning_styles(cot_traces, latent_traces)
        }
    
    return patterns

def analyze_transcription_quality(traces: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze intermediate transcription quality and patterns"""
    
    transcription_analysis = {
        "transcription_steps": [],
        "quality_metrics": {},
        "error_patterns": []
    }
    
    for trace_id, trace in traces.items():
        transcription_steps = [s for s in trace["reasoning_steps"] if s["step_type"] == "transcription"]
        
        for step in transcription_steps:
            step_analysis = {
                "trace_id": trace_id,
                "condition": trace["condition"],
                "task_type": trace["task_type"],
                "content": step["content"],
                "confidence": step["confidence"],
                "processing_time": step["processing_time"],
                "audio_duration": trace["audio_duration"],
                "is_correct": trace["is_correct"]
            }
            transcription_analysis["transcription_steps"].append(step_analysis)
    
    # Calculate quality metrics
    if transcription_analysis["transcription_steps"]:
        steps = transcription_analysis["transcription_steps"]
        
        transcription_analysis["quality_metrics"] = {
            "avg_confidence": np.mean([s["confidence"] for s in steps]),
            "avg_processing_time": np.mean([s["processing_time"] for s in steps]),
            "confidence_vs_accuracy": _analyze_confidence_accuracy_correlation(steps),
            "processing_time_vs_duration": _analyze_time_duration_correlation(steps)
        }
    
    return transcription_analysis

def compare_multi_evaluations(experiment_dirs: List[str]) -> Dict[str, Any]:
    """Compare results across multiple experiment runs"""
    
    all_experiments = {}
    
    for exp_dir in experiment_dirs:
        try:
            exp_data = load_experiment_traces(exp_dir)
            exp_name = Path(exp_dir).name
            all_experiments[exp_name] = exp_data
        except Exception as e:
            print(f"Warning: Could not load experiment {exp_dir}: {e}")
            continue
    
    if len(all_experiments) < 2:
        return {"error": "Need at least 2 experiments for comparison"}
    
    comparison = {
        "experiments_compared": list(all_experiments.keys()),
        "accuracy_trends": {},
        "efficiency_trends": {},
        "reasoning_pattern_evolution": {},
        "consistency_analysis": {}
    }
    
    # Compare accuracy trends
    for exp_name, exp_data in all_experiments.items():
        traces = exp_data["traces"]
        
        condition_accuracies = {}
        for trace in traces.values():
            condition = trace["condition"]
            if condition not in condition_accuracies:
                condition_accuracies[condition] = []
            condition_accuracies[condition].append(trace["is_correct"])
        
        comparison["accuracy_trends"][exp_name] = {
            cond: np.mean(accs) for cond, accs in condition_accuracies.items()
        }
    
    # Compare efficiency trends
    for exp_name, exp_data in all_experiments.items():
        traces = exp_data["traces"]
        
        condition_efficiency = {}
        for trace in traces.values():
            condition = trace["condition"]
            if condition not in condition_efficiency:
                condition_efficiency[condition] = []
            
            if trace["total_tokens"] > 0:
                efficiency = trace["is_correct"] / trace["total_tokens"]
                condition_efficiency[condition].append(efficiency)
        
        comparison["efficiency_trends"][exp_name] = {
            cond: np.mean(effs) if effs else 0.0 
            for cond, effs in condition_efficiency.items()
        }
    
    # Analyze consistency across runs
    comparison["consistency_analysis"] = _analyze_cross_run_consistency(all_experiments)
    
    return comparison

def generate_detailed_report(analysis_data: Dict[str, Any], output_file: str):
    """Generate comprehensive analysis report"""
    
    report_lines = []
    report_lines.append("# Detailed CoT vs Latent Reasoning Analysis Report")
    report_lines.append(f"Generated: {datetime.now().isoformat()}")
    report_lines.append("=" * 80)
    
    # Experiment overview
    if "index" in analysis_data:
        index = analysis_data["index"]
        report_lines.append(f"\n## Experiment Overview")
        report_lines.append(f"- Experiment: {index['experiment_name']}")
        report_lines.append(f"- Total traces: {index['summary']['total_traces']}")
        report_lines.append(f"- Conditions: {', '.join(index['summary']['conditions'])}")
        report_lines.append(f"- Datasets: {', '.join(index['summary']['datasets'])}")
        report_lines.append(f"- Task types: {', '.join(index['summary']['task_types'])}")
    
    # Reasoning patterns analysis
    if "reasoning_patterns" in analysis_data:
        patterns = analysis_data["reasoning_patterns"]
        report_lines.append(f"\n## Reasoning Patterns Analysis")
        
        if "cot_patterns" in patterns:
            cot = patterns["cot_patterns"]
            report_lines.append(f"\n### Chain-of-Thought Patterns")
            report_lines.append(f"- Average reasoning steps: {cot.get('avg_reasoning_steps', 0):.2f}")
            report_lines.append(f"- Step types distribution: {cot.get('step_types', {})}")
            
        if "latent_patterns" in patterns:
            latent = patterns["latent_patterns"]
            report_lines.append(f"\n### Latent Reasoning Patterns")
            report_lines.append(f"- Average iterations: {latent.get('avg_iterations', 0):.2f}")
            report_lines.append(f"- Processing efficiency: {latent.get('processing_efficiency', {})}")
        
        if "comparison" in patterns:
            comp = patterns["comparison"]
            report_lines.append(f"\n### Pattern Comparison")
            
            if "accuracy_comparison" in comp:
                acc = comp["accuracy_comparison"]
                report_lines.append(f"- CoT Accuracy: {acc.get('cot_accuracy', 0):.4f}")
                report_lines.append(f"- Latent Accuracy: {acc.get('latent_accuracy', 0):.4f}")
                
            if "efficiency_comparison" in comp:
                eff = comp["efficiency_comparison"]
                report_lines.append(f"- CoT Avg Time: {eff.get('cot_avg_time', 0):.3f}s")
                report_lines.append(f"- Latent Avg Time: {eff.get('latent_avg_time', 0):.3f}s")
                report_lines.append(f"- CoT Avg Tokens: {eff.get('cot_avg_tokens', 0):.1f}")
                report_lines.append(f"- Latent Avg Tokens: {eff.get('latent_avg_tokens', 0):.1f}")
    
    # Transcription analysis
    if "transcription_analysis" in analysis_data:
        trans = analysis_data["transcription_analysis"]
        report_lines.append(f"\n## Transcription Quality Analysis")
        
        if "quality_metrics" in trans:
            metrics = trans["quality_metrics"]
            report_lines.append(f"- Average confidence: {metrics.get('avg_confidence', 0):.3f}")
            report_lines.append(f"- Average processing time: {metrics.get('avg_processing_time', 0):.3f}s")
            report_lines.append(f"- Transcription steps analyzed: {len(trans.get('transcription_steps', []))}")
    
    # Multi-evaluation comparison
    if "multi_evaluation" in analysis_data:
        multi = analysis_data["multi_evaluation"]
        report_lines.append(f"\n## Multi-Evaluation Comparison")
        report_lines.append(f"- Experiments compared: {len(multi.get('experiments_compared', []))}")
        
        if "consistency_analysis" in multi:
            consistency = multi["consistency_analysis"]
            report_lines.append(f"- Consistency score: {consistency.get('overall_consistency', 0):.3f}")
    
    # Key findings and recommendations
    report_lines.append(f"\n## Key Findings")
    report_lines.append("- [Analysis findings would be generated based on data patterns]")
    
    report_lines.append(f"\n## Recommendations")
    report_lines.append("- [Recommendations would be generated based on analysis]")
    
    report_lines.append("=" * 80)
    
    # Save report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))

# Helper functions for analysis

def _analyze_step_types(traces: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyze distribution of reasoning step types"""
    step_counts = {}
    for trace in traces:
        for step in trace["reasoning_steps"]:
            step_type = step["step_type"]
            step_counts[step_type] = step_counts.get(step_type, 0) + 1
    return step_counts

def _analyze_reasoning_depth(traces: List[Dict[str, Any]]) -> Dict[str, float]:
    """Analyze reasoning depth patterns"""
    depths = [len(trace["reasoning_steps"]) for trace in traces]
    return {
        "min_depth": min(depths) if depths else 0,
        "max_depth": max(depths) if depths else 0,
        "avg_depth": np.mean(depths) if depths else 0,
        "std_depth": np.std(depths) if depths else 0
    }

def _extract_intermediate_conclusions(traces: List[Dict[str, Any]]) -> List[str]:
    """Extract intermediate conclusions from CoT traces"""
    conclusions = []
    for trace in traces:
        for step in trace["reasoning_steps"]:
            if step["step_type"] == "reasoning" and "intermediate_conclusion" in step.get("metadata", {}):
                conclusions.append(step["metadata"]["intermediate_conclusion"])
    return conclusions

def _analyze_confidence_evolution(traces: List[Dict[str, Any]]) -> Dict[str, float]:
    """Analyze how confidence evolves through reasoning steps"""
    confidence_trends = []
    for trace in traces:
        confidences = [step["confidence"] for step in trace["reasoning_steps"] if step["confidence"] > 0]
        if len(confidences) > 1:
            trend = confidences[-1] - confidences[0]  # Final - Initial confidence
            confidence_trends.append(trend)
    
    return {
        "avg_confidence_change": np.mean(confidence_trends) if confidence_trends else 0.0,
        "confidence_increase_rate": sum(1 for t in confidence_trends if t > 0) / len(confidence_trends) if confidence_trends else 0.0
    }

def _analyze_internal_states(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze latent reasoning internal states"""
    internal_states = []
    for trace in traces:
        for step in trace["reasoning_steps"]:
            if step["step_type"] == "latent_iteration" and "internal_state" in step.get("metadata", {}):
                internal_states.append(step["metadata"]["internal_state"])
    
    return {
        "total_states_captured": len(internal_states),
        "unique_processing_stages": len(set(s.get("processing_stage", "") for s in internal_states))
    }

def _analyze_latent_confidence_trends(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze confidence trends in latent reasoning"""
    trends = []
    for trace in traces:
        latent_steps = [s for s in trace["reasoning_steps"] if s["step_type"] == "latent_iteration"]
        if len(latent_steps) > 1:
            confidences = [s["confidence"] for s in latent_steps]
            trend = "increasing" if confidences[-1] > confidences[0] else "decreasing"
            trends.append(trend)
    
    return {
        "increasing_trends": trends.count("increasing"),
        "decreasing_trends": trends.count("decreasing"),
        "trend_consistency": max(trends.count("increasing"), trends.count("decreasing")) / len(trends) if trends else 0.0
    }

def _analyze_processing_efficiency(traces: List[Dict[str, Any]]) -> Dict[str, float]:
    """Analyze processing efficiency in latent reasoning"""
    efficiencies = []
    for trace in traces:
        if trace["total_processing_time"] > 0 and trace["total_tokens"] > 0:
            efficiency = trace["is_correct"] / (trace["total_processing_time"] * trace["total_tokens"])
            efficiencies.append(efficiency)
    
    return {
        "avg_efficiency": np.mean(efficiencies) if efficiencies else 0.0,
        "efficiency_std": np.std(efficiencies) if efficiencies else 0.0
    }

def _compare_reasoning_styles(cot_traces: List[Dict], latent_traces: List[Dict]) -> Dict[str, Any]:
    """Compare reasoning styles between CoT and latent approaches"""
    
    # Compare step patterns
    cot_steps = [len(t["reasoning_steps"]) for t in cot_traces]
    latent_steps = [len(t["reasoning_steps"]) for t in latent_traces]
    
    # Compare processing times
    cot_times = [t["total_processing_time"] for t in cot_traces]
    latent_times = [t["total_processing_time"] for t in latent_traces]
    
    return {
        "step_count_difference": np.mean(cot_steps) - np.mean(latent_steps),
        "processing_time_difference": np.mean(cot_times) - np.mean(latent_times),
        "style_consistency": {
            "cot_step_variance": np.var(cot_steps),
            "latent_step_variance": np.var(latent_steps)
        }
    }

def _analyze_confidence_accuracy_correlation(steps: List[Dict]) -> float:
    """Analyze correlation between confidence and accuracy"""
    confidences = [s["confidence"] for s in steps if s["confidence"] > 0]
    accuracies = [float(s["is_correct"]) for s in steps if s["confidence"] > 0]
    
    if len(confidences) > 1:
        return np.corrcoef(confidences, accuracies)[0, 1]
    return 0.0

def _analyze_time_duration_correlation(steps: List[Dict]) -> float:
    """Analyze correlation between processing time and audio duration"""
    times = [s["processing_time"] for s in steps if s["processing_time"] > 0]
    durations = [s["audio_duration"] for s in steps if s["processing_time"] > 0]
    
    if len(times) > 1:
        return np.corrcoef(times, durations)[0, 1]
    return 0.0

def _analyze_cross_run_consistency(all_experiments: Dict[str, Any]) -> Dict[str, float]:
    """Analyze consistency across multiple experiment runs"""
    
    # Collect accuracy by condition across runs
    condition_accuracies = {}
    
    for exp_name, exp_data in all_experiments.items():
        traces = exp_data["traces"]
        
        for trace in traces.values():
            condition = trace["condition"]
            if condition not in condition_accuracies:
                condition_accuracies[condition] = []
            condition_accuracies[condition].append(trace["is_correct"])
    
    # Calculate consistency (inverse of variance)
    consistencies = {}
    for condition, accuracies in condition_accuracies.items():
        if len(accuracies) > 1:
            consistency = 1.0 / (1.0 + np.var(accuracies))  # Higher consistency = lower variance
            consistencies[condition] = consistency
    
    overall_consistency = np.mean(list(consistencies.values())) if consistencies else 0.0
    
    return {
        "condition_consistencies": consistencies,
        "overall_consistency": overall_consistency
    }

def main():
    """Main CLI for trace analysis"""
    parser = argparse.ArgumentParser(description="Analyze detailed JSON traces from reasoning experiments")
    parser.add_argument("experiment_dir", help="Directory containing experiment results")
    parser.add_argument("--compare", nargs="+", help="Additional experiment directories to compare")
    parser.add_argument("--output", "-o", default="trace_analysis_report.md", help="Output report file")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown", help="Output format")
    
    args = parser.parse_args()
    
    try:
        # Load primary experiment
        print(f"Loading experiment traces from {args.experiment_dir}...")
        experiment_data = load_experiment_traces(args.experiment_dir)
        
        # Analyze reasoning patterns
        print("Analyzing reasoning patterns...")
        reasoning_analysis = analyze_reasoning_patterns(experiment_data["traces"])
        
        # Analyze transcription quality
        print("Analyzing transcription quality...")
        transcription_analysis = analyze_transcription_quality(experiment_data["traces"])
        
        analysis_results = {
            **experiment_data,
            "reasoning_patterns": reasoning_analysis,
            "transcription_analysis": transcription_analysis
        }
        
        # Multi-evaluation comparison if requested
        if args.compare:
            print(f"Comparing with {len(args.compare)} additional experiments...")
            all_dirs = [args.experiment_dir] + args.compare
            multi_eval_analysis = compare_multi_evaluations(all_dirs)
            analysis_results["multi_evaluation"] = multi_eval_analysis
        
        # Generate report
        if args.format == "markdown":
            print(f"Generating report: {args.output}")
            generate_detailed_report(analysis_results, args.output)
            print(f"✅ Analysis complete! Report saved to: {args.output}")
        else:
            # JSON output
            json_output = args.output.replace('.md', '.json')
            with open(json_output, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            print(f"✅ Analysis complete! JSON data saved to: {json_output}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
