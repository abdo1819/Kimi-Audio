"""
Evaluation Framework for CoT vs Latent Reasoning Comparison

Implements comprehensive evaluation metrics:
- Accuracy, WER, CER
- Token efficiency and latency
- Calibration and confidence
- Statistical analysis
"""

import time
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict
import json

# Import evaluation metrics
try:
    import evaluate
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

from .model_backends import AudioTextModelBackend
from .prompting import ReasoningPrompter
from .datasets import ReasoningTask
from .budget_controller import ComputeBudgetController

@dataclass
class EvaluationResult:
    """Results from evaluating a single task"""
    task_id: str
    predicted_answer: str
    true_answer: str
    is_correct: bool
    confidence: float = 0.0
    
    # Performance metrics
    generation_time: float = 0.0
    num_tokens: int = 0
    token_efficiency: float = 0.0  # tokens/second
    
    # Text similarity metrics
    wer: float = 0.0  # Word Error Rate
    cer: float = 0.0  # Character Error Rate
    
    # Reasoning trace
    full_response: str = ""
    reasoning_trace: str = ""
    
    # Context
    task_type: str = ""
    difficulty: str = ""
    condition: str = ""

@dataclass 
class AggregatedResults:
    """Aggregated results across multiple tasks"""
    condition: str
    dataset: str
    
    # Accuracy metrics
    accuracy: float = 0.0
    accuracy_by_type: Dict[str, float] = None
    accuracy_by_difficulty: Dict[str, float] = None
    
    # Efficiency metrics
    avg_generation_time: float = 0.0
    avg_tokens: float = 0.0
    avg_token_efficiency: float = 0.0
    
    # Text quality metrics
    avg_wer: float = 0.0
    avg_cer: float = 0.0
    
    # Confidence and calibration
    avg_confidence: float = 0.0
    calibration_error: float = 0.0  # Expected Calibration Error
    
    # Statistical measures
    num_samples: int = 0
    accuracy_std: float = 0.0
    
    def __post_init__(self):
        if self.accuracy_by_type is None:
            self.accuracy_by_type = {}
        if self.accuracy_by_difficulty is None:
            self.accuracy_by_difficulty = {}

class ReasoningEvaluator:
    """Evaluator for audio reasoning tasks"""
    
    def __init__(self, 
                 metrics: List[str] = None,
                 save_predictions: bool = True,
                 save_traces: bool = True):
        self.metrics = metrics or ["accuracy", "token_efficiency", "latency"]
        self.save_predictions = save_predictions
        self.save_traces = save_traces
        self.results_history = []
    
    def evaluate(self,
                model_backend: AudioTextModelBackend,
                prompter: ReasoningPrompter,
                dataset: List[ReasoningTask],
                budget_controller: ComputeBudgetController,
                batch_size: int = 1,
                temperature: float = 0.0,
                top_k: int = 5) -> Dict[str, Any]:
        """Evaluate model on dataset with given prompting strategy"""
        
        individual_results = []
        
        for i, task in enumerate(dataset):
            # Generate prompt
            prompt = prompter.generate_prompt(task.question, task.context)
            
            # Get budget constraints
            budget = budget_controller.constraints
            
            # Evaluate single task
            result = self._evaluate_single_task(
                task=task,
                prompt=prompt,
                model_backend=model_backend,
                prompter=prompter,
                budget=budget,
                temperature=temperature,
                top_k=top_k,
                task_id=f"task_{i:04d}"
            )
            
            individual_results.append(result)
            
            # Monitor budget usage
            if hasattr(result, 'generation_stats'):
                budget_controller.monitor_usage(
                    budget=budget,
                    actual_tokens=result.num_tokens,
                    actual_flops=getattr(result, 'flops', 0.0),
                    actual_time=result.generation_time
                )
        
        # Aggregate results
        aggregated = self._aggregate_results(individual_results, prompter.condition_type)
        
        # Store results
        self.results_history.extend(individual_results)
        
        return {
            "individual_results": individual_results,
            "aggregated_results": aggregated,
            "budget_stats": budget_controller.get_efficiency_stats()
        }
    
    def _evaluate_single_task(self,
                             task: ReasoningTask,
                             prompt: str,
                             model_backend: AudioTextModelBackend,
                             prompter: ReasoningPrompter,
                             budget: Any,
                             temperature: float,
                             top_k: int,
                             task_id: str) -> EvaluationResult:
        """Evaluate a single task"""
        
        start_time = time.time()
        
        try:
            # Generate response
            generation_stats = model_backend.generate(
                audio_path=task.audio_path,
                text_prompt=prompt,
                temperature=temperature,
                top_k=top_k,
                max_new_tokens=getattr(budget, 'max_tokens', 500)
            )
            
            generation_time = time.time() - start_time
            
            # Extract answer
            predicted_answer = prompter.extract_answer(generation_stats.output_text)
            
            # Calculate correctness
            is_correct = self._check_correctness(predicted_answer, task.answer, task.task_type)
            
            # Calculate confidence (simple heuristic for now)
            confidence = self._estimate_confidence(generation_stats.output_text, predicted_answer)
            
            # Calculate text similarity metrics
            wer, cer = self._calculate_text_metrics(predicted_answer, task.answer)
            
            result = EvaluationResult(
                task_id=task_id,
                predicted_answer=predicted_answer,
                true_answer=task.answer,
                is_correct=is_correct,
                confidence=confidence,
                generation_time=generation_time,
                num_tokens=generation_stats.num_output_tokens,
                token_efficiency=generation_stats.token_efficiency,
                wer=wer,
                cer=cer,
                full_response=generation_stats.output_text,
                reasoning_trace=self._extract_reasoning_trace(generation_stats.output_text),
                task_type=task.task_type,
                difficulty=task.difficulty,
                condition=prompter.condition_type
            )
            
        except Exception as e:
            # Handle errors gracefully
            generation_time = time.time() - start_time
            result = EvaluationResult(
                task_id=task_id,
                predicted_answer=f"ERROR: {str(e)}",
                true_answer=task.answer,
                is_correct=False,
                generation_time=generation_time,
                full_response=f"ERROR: {str(e)}",
                task_type=task.task_type,
                difficulty=task.difficulty,
                condition=prompter.condition_type
            )
        
        return result
    
    def _check_correctness(self, predicted: str, true: str, task_type: str) -> bool:
        """Check if prediction is correct"""
        # Normalize answers
        pred_clean = self._normalize_answer(predicted)
        true_clean = self._normalize_answer(true)
        
        # Exact match
        if pred_clean == true_clean:
            return True
        
        # Task-specific matching
        if task_type == "counting":
            # Extract numbers
            pred_num = self._extract_number(predicted)
            true_num = self._extract_number(true)
            return pred_num is not None and pred_num == true_num
        
        elif task_type in ["temporal", "speaker_id", "content"]:
            # Fuzzy matching for text answers
            return self._fuzzy_match(pred_clean, true_clean)
        
        return False
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        return answer.lower().strip().replace(".", "").replace(",", "")
    
    def _extract_number(self, text: str) -> Optional[int]:
        """Extract number from text"""
        # Look for digits
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[0])
        
        # Look for written numbers
        number_words = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10
        }
        
        text_lower = text.lower()
        for word, num in number_words.items():
            if word in text_lower:
                return num
        
        return None
    
    def _fuzzy_match(self, pred: str, true: str, threshold: float = 0.8) -> bool:
        """Fuzzy string matching"""
        if not pred or not true:
            return False
        
        # Simple overlap-based matching
        pred_words = set(pred.split())
        true_words = set(true.split())
        
        if not true_words:
            return False
        
        overlap = len(pred_words & true_words)
        overlap_ratio = overlap / len(true_words)
        
        return overlap_ratio >= threshold
    
    def _estimate_confidence(self, full_response: str, answer: str) -> float:
        """Estimate confidence in the answer (simple heuristic)"""
        # Look for confidence indicators
        confidence_indicators = {
            "certain": 0.9,
            "sure": 0.8,
            "confident": 0.8,
            "clearly": 0.7,
            "obviously": 0.7,
            "likely": 0.6,
            "probably": 0.6,
            "maybe": 0.4,
            "possibly": 0.4,
            "unsure": 0.3,
            "uncertain": 0.3,
        }
        
        response_lower = full_response.lower()
        confidence = 0.5  # default
        
        for indicator, conf_value in confidence_indicators.items():
            if indicator in response_lower:
                confidence = max(confidence, conf_value)
        
        # Adjust based on answer length and completeness
        if len(answer.strip()) == 0:
            confidence *= 0.5
        elif len(answer.split()) == 1:
            confidence *= 1.1  # Short, definitive answers get slight boost
        
        return min(1.0, confidence)
    
    def _calculate_text_metrics(self, predicted: str, true: str) -> Tuple[float, float]:
        """Calculate WER and CER"""
        if not METRICS_AVAILABLE:
            return 0.0, 0.0
        
        try:
            wer = wer_metric.compute(predictions=[predicted], references=[true])
            cer = cer_metric.compute(predictions=[predicted], references=[true])
            return wer, cer
        except:
            return 0.0, 0.0
    
    def _extract_reasoning_trace(self, full_response: str) -> str:
        """Extract reasoning steps from response"""
        # Look for reasoning patterns
        reasoning_patterns = [
            r"(Let me think.*?Final Answer:)",
            r"(Step \d+:.*?)(?=Step \d+:|Final Answer:|$)",
            r"(First,.*?Then,.*?)(?=Final Answer:|$)",
        ]
        
        for pattern in reasoning_patterns:
            matches = re.findall(pattern, full_response, re.DOTALL | re.IGNORECASE)
            if matches:
                return "\n".join(matches)
        
        # Fallback: everything before "Final Answer:"
        if "Final Answer:" in full_response:
            return full_response.split("Final Answer:")[0].strip()
        
        return ""
    
    def _aggregate_results(self, results: List[EvaluationResult], condition: str) -> AggregatedResults:
        """Aggregate individual results"""
        if not results:
            return AggregatedResults(condition=condition, dataset="unknown")
        
        # Basic accuracy
        correct = sum(1 for r in results if r.is_correct)
        accuracy = correct / len(results)
        
        # Accuracy by task type
        accuracy_by_type = {}
        type_counts = defaultdict(list)
        for r in results:
            type_counts[r.task_type].append(r.is_correct)
        
        for task_type, correct_list in type_counts.items():
            accuracy_by_type[task_type] = sum(correct_list) / len(correct_list)
        
        # Accuracy by difficulty
        accuracy_by_difficulty = {}
        diff_counts = defaultdict(list)
        for r in results:
            diff_counts[r.difficulty].append(r.is_correct)
        
        for difficulty, correct_list in diff_counts.items():
            accuracy_by_difficulty[difficulty] = sum(correct_list) / len(correct_list)
        
        # Efficiency metrics
        avg_generation_time = np.mean([r.generation_time for r in results])
        avg_tokens = np.mean([r.num_tokens for r in results])
        avg_token_efficiency = np.mean([r.token_efficiency for r in results if r.token_efficiency > 0])
        
        # Text quality metrics
        avg_wer = np.mean([r.wer for r in results])
        avg_cer = np.mean([r.cer for r in results])
        
        # Confidence metrics
        confidences = [r.confidence for r in results if r.confidence > 0]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Calibration error (Expected Calibration Error)
        calibration_error = self._calculate_calibration_error(results)
        
        # Statistical measures
        accuracy_values = [float(r.is_correct) for r in results]
        accuracy_std = np.std(accuracy_values)
        
        return AggregatedResults(
            condition=condition,
            dataset="mixed",  # Could be improved to track actual dataset
            accuracy=accuracy,
            accuracy_by_type=accuracy_by_type,
            accuracy_by_difficulty=accuracy_by_difficulty,
            avg_generation_time=avg_generation_time,
            avg_tokens=avg_tokens,
            avg_token_efficiency=avg_token_efficiency,
            avg_wer=avg_wer,
            avg_cer=avg_cer,
            avg_confidence=avg_confidence,
            calibration_error=calibration_error,
            num_samples=len(results),
            accuracy_std=accuracy_std
        )
    
    def _calculate_calibration_error(self, results: List[EvaluationResult], num_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
        if not results or not any(r.confidence > 0 for r in results):
            return 0.0
        
        # Filter results with valid confidence scores
        valid_results = [r for r in results if r.confidence > 0]
        if len(valid_results) < 2:
            return 0.0
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        total_samples = len(valid_results)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find results in this bin
            in_bin = [r for r in valid_results if bin_lower < r.confidence <= bin_upper]
            
            if len(in_bin) > 0:
                # Calculate accuracy and confidence for this bin
                bin_accuracy = sum(r.is_correct for r in in_bin) / len(in_bin)
                bin_confidence = sum(r.confidence for r in in_bin) / len(in_bin)
                bin_weight = len(in_bin) / total_samples
                
                # Add to ECE
                ece += bin_weight * abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def compare_conditions(self, 
                          results_by_condition: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare results across different conditions"""
        comparison = {
            "accuracy_comparison": {},
            "efficiency_comparison": {},
            "statistical_tests": {}
        }
        
        # Extract accuracy values for each condition
        condition_accuracies = {}
        for condition, result_dict in results_by_condition.items():
            agg_results = result_dict.get("aggregated_results")
            if agg_results:
                condition_accuracies[condition] = agg_results.accuracy
        
        comparison["accuracy_comparison"] = condition_accuracies
        
        # Efficiency comparison (accuracy per token)
        efficiency_scores = {}
        for condition, result_dict in results_by_condition.items():
            agg_results = result_dict.get("aggregated_results")
            if agg_results and agg_results.avg_tokens > 0:
                efficiency = agg_results.accuracy / agg_results.avg_tokens * 1000  # per 1K tokens
                efficiency_scores[condition] = efficiency
        
        comparison["efficiency_comparison"] = efficiency_scores
        
        # TODO: Add statistical significance tests
        comparison["statistical_tests"] = {"note": "Statistical tests not implemented yet"}
        
        return comparison
