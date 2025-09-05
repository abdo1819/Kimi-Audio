"""
Detailed JSON Logger for CoT vs Latent Reasoning Experiments

Captures comprehensive traces including:
- Intermediate transcriptions and reasoning steps
- Model internal states and decisions
- Timing and resource usage details
- Multi-evaluation comparison data
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import uuid

@dataclass
class ReasoningStep:
    """Single step in reasoning process"""
    step_id: str
    step_type: str  # "transcription", "reasoning", "decision", "verification"
    timestamp: float
    content: str
    confidence: float = 0.0
    tokens_used: int = 0
    processing_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class DetailedEvaluationTrace:
    """Comprehensive trace of a single evaluation"""
    trace_id: str
    task_id: str
    condition: str
    dataset: str
    seed: int
    
    # Input information
    audio_path: str
    audio_duration: float
    question: str
    ground_truth: str
    task_type: str
    difficulty: str
    
    # Processing traces
    prompt_generation: Dict[str, Any]
    reasoning_steps: List[ReasoningStep]
    model_interactions: List[Dict[str, Any]]
    
    # Output and evaluation
    final_answer: str
    is_correct: bool
    confidence_score: float
    
    # Performance metrics
    total_processing_time: float
    total_tokens: int
    memory_peak_mb: float
    
    # Budget tracking
    budget_allocated: Dict[str, float]
    budget_used: Dict[str, float]
    budget_efficiency: float
    
    # Timestamps
    start_time: float
    end_time: float
    
    def __post_init__(self):
        if not hasattr(self, 'trace_id') or not self.trace_id:
            self.trace_id = str(uuid.uuid4())

class DetailedJSONLogger:
    """Comprehensive JSON logger for reasoning experiments"""
    
    def __init__(self, output_dir: str, experiment_name: str):
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.traces_dir = self.output_dir / "detailed_traces"
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log files
        self.experiment_log_file = self.output_dir / f"{experiment_name}_detailed_log.json"
        self.traces_index_file = self.traces_dir / "traces_index.json"
        
        # In-memory storage for current session
        self.current_traces = []
        self.experiment_metadata = {
            "experiment_name": experiment_name,
            "start_time": time.time(),
            "traces_logged": 0,
            "conditions_tested": set(),
            "datasets_used": set()
        }
        
        # Initialize index file
        self._initialize_traces_index()
    
    def _initialize_traces_index(self):
        """Initialize the traces index file"""
        if not self.traces_index_file.exists():
            index_data = {
                "experiment_name": self.experiment_name,
                "created_at": datetime.now().isoformat(),
                "traces": [],
                "summary": {
                    "total_traces": 0,
                    "conditions": [],
                    "datasets": [],
                    "task_types": [],
                    "last_updated": datetime.now().isoformat()
                }
            }
            with open(self.traces_index_file, 'w') as f:
                json.dump(index_data, f, indent=2)
    
    def start_evaluation_trace(self, 
                             task_id: str,
                             condition: str,
                             dataset: str,
                             seed: int,
                             audio_path: str,
                             question: str,
                             ground_truth: str,
                             task_type: str,
                             difficulty: str) -> str:
        """Start a new detailed evaluation trace"""
        
        trace_id = str(uuid.uuid4())
        
        # Get audio duration
        audio_duration = self._get_audio_duration(audio_path)
        
        trace = DetailedEvaluationTrace(
            trace_id=trace_id,
            task_id=task_id,
            condition=condition,
            dataset=dataset,
            seed=seed,
            audio_path=audio_path,
            audio_duration=audio_duration,
            question=question,
            ground_truth=ground_truth,
            task_type=task_type,
            difficulty=difficulty,
            prompt_generation={},
            reasoning_steps=[],
            model_interactions=[],
            final_answer="",
            is_correct=False,
            confidence_score=0.0,
            total_processing_time=0.0,
            total_tokens=0,
            memory_peak_mb=0.0,
            budget_allocated={},
            budget_used={},
            budget_efficiency=0.0,
            start_time=time.time(),
            end_time=0.0
        )
        
        self.current_traces.append(trace)
        
        # Update experiment metadata
        self.experiment_metadata["conditions_tested"].add(condition)
        self.experiment_metadata["datasets_used"].add(dataset)
        
        return trace_id
    
    def log_prompt_generation(self, 
                            trace_id: str,
                            prompt_type: str,
                            raw_prompt: str,
                            processed_prompt: str,
                            prompt_tokens: int,
                            generation_time: float,
                            metadata: Dict[str, Any] = None):
        """Log prompt generation details"""
        
        trace = self._find_trace(trace_id)
        if not trace:
            return
        
        trace.prompt_generation = {
            "prompt_type": prompt_type,
            "raw_prompt": raw_prompt,
            "processed_prompt": processed_prompt,
            "prompt_tokens": prompt_tokens,
            "generation_time": generation_time,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
    
    def log_reasoning_step(self,
                          trace_id: str,
                          step_type: str,
                          content: str,
                          confidence: float = 0.0,
                          tokens_used: int = 0,
                          processing_time: float = 0.0,
                          metadata: Dict[str, Any] = None) -> str:
        """Log a single reasoning step"""
        
        trace = self._find_trace(trace_id)
        if not trace:
            return ""
        
        step_id = f"{trace_id}_step_{len(trace.reasoning_steps)}"
        
        step = ReasoningStep(
            step_id=step_id,
            step_type=step_type,
            timestamp=time.time(),
            content=content,
            confidence=confidence,
            tokens_used=tokens_used,
            processing_time=processing_time,
            metadata=metadata or {}
        )
        
        trace.reasoning_steps.append(step)
        return step_id
    
    def log_model_interaction(self,
                            trace_id: str,
                            interaction_type: str,
                            input_data: Dict[str, Any],
                            output_data: Dict[str, Any],
                            timing: Dict[str, float],
                            resource_usage: Dict[str, float],
                            metadata: Dict[str, Any] = None):
        """Log detailed model interaction"""
        
        trace = self._find_trace(trace_id)
        if not trace:
            return
        
        interaction = {
            "interaction_id": f"{trace_id}_interaction_{len(trace.model_interactions)}",
            "interaction_type": interaction_type,
            "timestamp": time.time(),
            "input_data": input_data,
            "output_data": output_data,
            "timing": timing,
            "resource_usage": resource_usage,
            "metadata": metadata or {}
        }
        
        trace.model_interactions.append(interaction)
    
    def log_intermediate_transcription(self,
                                     trace_id: str,
                                     transcription_text: str,
                                     confidence: float,
                                     processing_time: float,
                                     model_info: Dict[str, Any],
                                     audio_segment_info: Dict[str, Any] = None):
        """Log intermediate transcription results"""
        
        return self.log_reasoning_step(
            trace_id=trace_id,
            step_type="transcription",
            content=transcription_text,
            confidence=confidence,
            processing_time=processing_time,
            metadata={
                "model_info": model_info,
                "audio_segment_info": audio_segment_info or {},
                "transcription_quality": {
                    "confidence": confidence,
                    "processing_time": processing_time
                }
            }
        )
    
    def log_reasoning_chain_step(self,
                               trace_id: str,
                               step_number: int,
                               reasoning_text: str,
                               intermediate_conclusion: str,
                               confidence: float,
                               tokens_used: int):
        """Log a step in CoT reasoning chain"""
        
        return self.log_reasoning_step(
            trace_id=trace_id,
            step_type="reasoning",
            content=reasoning_text,
            confidence=confidence,
            tokens_used=tokens_used,
            metadata={
                "step_number": step_number,
                "intermediate_conclusion": intermediate_conclusion,
                "reasoning_type": "chain_of_thought"
            }
        )
    
    def log_latent_reasoning_iteration(self,
                                     trace_id: str,
                                     iteration: int,
                                     internal_state: Dict[str, Any],
                                     confidence_evolution: List[float],
                                     processing_time: float):
        """Log latent reasoning iteration"""
        
        return self.log_reasoning_step(
            trace_id=trace_id,
            step_type="latent_iteration",
            content=f"Latent reasoning iteration {iteration}",
            confidence=confidence_evolution[-1] if confidence_evolution else 0.0,
            processing_time=processing_time,
            metadata={
                "iteration": iteration,
                "internal_state": internal_state,
                "confidence_evolution": confidence_evolution,
                "reasoning_type": "latent"
            }
        )
    
    def log_budget_usage(self,
                        trace_id: str,
                        budget_type: str,
                        allocated: float,
                        used: float,
                        efficiency: float):
        """Log budget allocation and usage"""
        
        trace = self._find_trace(trace_id)
        if not trace:
            return
        
        trace.budget_allocated[budget_type] = allocated
        trace.budget_used[budget_type] = used
        trace.budget_efficiency = efficiency
    
    def complete_evaluation_trace(self,
                                trace_id: str,
                                final_answer: str,
                                is_correct: bool,
                                confidence_score: float,
                                total_tokens: int,
                                memory_peak_mb: float):
        """Complete and finalize an evaluation trace"""
        
        trace = self._find_trace(trace_id)
        if not trace:
            return
        
        trace.final_answer = final_answer
        trace.is_correct = is_correct
        trace.confidence_score = confidence_score
        trace.total_tokens = total_tokens
        trace.memory_peak_mb = memory_peak_mb
        trace.end_time = time.time()
        trace.total_processing_time = trace.end_time - trace.start_time
        
        # Save individual trace file
        self._save_individual_trace(trace)
        
        # Update traces index
        self._update_traces_index(trace)
        
        self.experiment_metadata["traces_logged"] += 1
    
    def save_experiment_summary(self):
        """Save comprehensive experiment summary"""
        
        summary = {
            "experiment_metadata": {
                **self.experiment_metadata,
                "conditions_tested": list(self.experiment_metadata["conditions_tested"]),
                "datasets_used": list(self.experiment_metadata["datasets_used"]),
                "end_time": time.time(),
                "total_duration": time.time() - self.experiment_metadata["start_time"]
            },
            "traces_summary": self._generate_traces_summary(),
            "performance_analysis": self._generate_performance_analysis(),
            "comparison_analysis": self._generate_comparison_analysis()
        }
        
        with open(self.experiment_log_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def _find_trace(self, trace_id: str) -> Optional[DetailedEvaluationTrace]:
        """Find trace by ID"""
        for trace in self.current_traces:
            if trace.trace_id == trace_id:
                return trace
        return None
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds"""
        try:
            import soundfile as sf
            data, sr = sf.read(audio_path)
            return len(data) / sr
        except:
            return 0.0
    
    def _save_individual_trace(self, trace: DetailedEvaluationTrace):
        """Save individual trace to separate JSON file"""
        
        trace_file = self.traces_dir / f"trace_{trace.trace_id}.json"
        
        # Convert trace to dict for JSON serialization
        trace_dict = asdict(trace)
        trace_dict["saved_at"] = datetime.now().isoformat()
        
        with open(trace_file, 'w') as f:
            json.dump(trace_dict, f, indent=2, default=str)
    
    def _update_traces_index(self, trace: DetailedEvaluationTrace):
        """Update the traces index with new trace info"""
        
        # Load current index
        with open(self.traces_index_file, 'r') as f:
            index_data = json.load(f)
        
        # Add trace summary
        trace_summary = {
            "trace_id": trace.trace_id,
            "task_id": trace.task_id,
            "condition": trace.condition,
            "dataset": trace.dataset,
            "task_type": trace.task_type,
            "difficulty": trace.difficulty,
            "is_correct": trace.is_correct,
            "confidence_score": trace.confidence_score,
            "total_processing_time": trace.total_processing_time,
            "total_tokens": trace.total_tokens,
            "reasoning_steps_count": len(trace.reasoning_steps),
            "file_path": f"trace_{trace.trace_id}.json",
            "completed_at": datetime.now().isoformat()
        }
        
        index_data["traces"].append(trace_summary)
        
        # Update summary statistics
        index_data["summary"]["total_traces"] += 1
        index_data["summary"]["last_updated"] = datetime.now().isoformat()
        
        # Update unique lists
        if trace.condition not in index_data["summary"]["conditions"]:
            index_data["summary"]["conditions"].append(trace.condition)
        if trace.dataset not in index_data["summary"]["datasets"]:
            index_data["summary"]["datasets"].append(trace.dataset)
        if trace.task_type not in index_data["summary"]["task_types"]:
            index_data["summary"]["task_types"].append(trace.task_type)
        
        # Save updated index
        with open(self.traces_index_file, 'w') as f:
            json.dump(index_data, f, indent=2)
    
    def _generate_traces_summary(self) -> Dict[str, Any]:
        """Generate summary of all traces"""
        
        if not self.current_traces:
            return {}
        
        total_traces = len(self.current_traces)
        correct_traces = sum(1 for t in self.current_traces if t.is_correct)
        
        return {
            "total_traces": total_traces,
            "correct_traces": correct_traces,
            "accuracy": correct_traces / total_traces if total_traces > 0 else 0.0,
            "avg_processing_time": sum(t.total_processing_time for t in self.current_traces) / total_traces,
            "avg_tokens": sum(t.total_tokens for t in self.current_traces) / total_traces,
            "avg_confidence": sum(t.confidence_score for t in self.current_traces) / total_traces,
            "conditions_breakdown": self._breakdown_by_condition(),
            "task_type_breakdown": self._breakdown_by_task_type(),
            "difficulty_breakdown": self._breakdown_by_difficulty()
        }
    
    def _breakdown_by_condition(self) -> Dict[str, Dict[str, float]]:
        """Break down performance by condition"""
        
        condition_stats = {}
        
        for condition in self.experiment_metadata["conditions_tested"]:
            condition_traces = [t for t in self.current_traces if t.condition == condition]
            
            if condition_traces:
                total = len(condition_traces)
                correct = sum(1 for t in condition_traces if t.is_correct)
                
                condition_stats[condition] = {
                    "total_traces": total,
                    "accuracy": correct / total,
                    "avg_processing_time": sum(t.total_processing_time for t in condition_traces) / total,
                    "avg_tokens": sum(t.total_tokens for t in condition_traces) / total,
                    "avg_confidence": sum(t.confidence_score for t in condition_traces) / total
                }
        
        return condition_stats
    
    def _breakdown_by_task_type(self) -> Dict[str, Dict[str, float]]:
        """Break down performance by task type"""
        
        task_stats = {}
        task_types = set(t.task_type for t in self.current_traces)
        
        for task_type in task_types:
            task_traces = [t for t in self.current_traces if t.task_type == task_type]
            
            if task_traces:
                total = len(task_traces)
                correct = sum(1 for t in task_traces if t.is_correct)
                
                task_stats[task_type] = {
                    "total_traces": total,
                    "accuracy": correct / total,
                    "avg_processing_time": sum(t.total_processing_time for t in task_traces) / total,
                    "avg_tokens": sum(t.total_tokens for t in task_traces) / total
                }
        
        return task_stats
    
    def _breakdown_by_difficulty(self) -> Dict[str, Dict[str, float]]:
        """Break down performance by difficulty"""
        
        difficulty_stats = {}
        difficulties = set(t.difficulty for t in self.current_traces)
        
        for difficulty in difficulties:
            diff_traces = [t for t in self.current_traces if t.difficulty == difficulty]
            
            if diff_traces:
                total = len(diff_traces)
                correct = sum(1 for t in diff_traces if t.is_correct)
                
                difficulty_stats[difficulty] = {
                    "total_traces": total,
                    "accuracy": correct / total,
                    "avg_processing_time": sum(t.total_processing_time for t in diff_traces) / total
                }
        
        return difficulty_stats
    
    def _generate_performance_analysis(self) -> Dict[str, Any]:
        """Generate detailed performance analysis"""
        
        if not self.current_traces:
            return {}
        
        # Token efficiency analysis
        token_efficiency = []
        for trace in self.current_traces:
            if trace.total_tokens > 0:
                efficiency = trace.is_correct / trace.total_tokens
                token_efficiency.append(efficiency)
        
        # Time efficiency analysis
        time_efficiency = []
        for trace in self.current_traces:
            if trace.total_processing_time > 0:
                efficiency = trace.is_correct / trace.total_processing_time
                time_efficiency.append(efficiency)
        
        return {
            "token_efficiency": {
                "mean": sum(token_efficiency) / len(token_efficiency) if token_efficiency else 0.0,
                "std": self._calculate_std(token_efficiency),
                "min": min(token_efficiency) if token_efficiency else 0.0,
                "max": max(token_efficiency) if token_efficiency else 0.0
            },
            "time_efficiency": {
                "mean": sum(time_efficiency) / len(time_efficiency) if time_efficiency else 0.0,
                "std": self._calculate_std(time_efficiency),
                "min": min(time_efficiency) if time_efficiency else 0.0,
                "max": max(time_efficiency) if time_efficiency else 0.0
            },
            "reasoning_steps_analysis": self._analyze_reasoning_steps(),
            "confidence_calibration": self._analyze_confidence_calibration()
        }
    
    def _generate_comparison_analysis(self) -> Dict[str, Any]:
        """Generate comparison analysis between conditions"""
        
        conditions = list(self.experiment_metadata["conditions_tested"])
        
        if len(conditions) < 2:
            return {"note": "Need at least 2 conditions for comparison"}
        
        comparisons = {}
        
        for i, cond1 in enumerate(conditions):
            for cond2 in conditions[i+1:]:
                comp_key = f"{cond1}_vs_{cond2}"
                
                traces1 = [t for t in self.current_traces if t.condition == cond1]
                traces2 = [t for t in self.current_traces if t.condition == cond2]
                
                if traces1 and traces2:
                    acc1 = sum(t.is_correct for t in traces1) / len(traces1)
                    acc2 = sum(t.is_correct for t in traces2) / len(traces2)
                    
                    tokens1 = sum(t.total_tokens for t in traces1) / len(traces1)
                    tokens2 = sum(t.total_tokens for t in traces2) / len(traces2)
                    
                    time1 = sum(t.total_processing_time for t in traces1) / len(traces1)
                    time2 = sum(t.total_processing_time for t in traces2) / len(traces2)
                    
                    comparisons[comp_key] = {
                        "accuracy_difference": acc1 - acc2,
                        "token_difference": tokens1 - tokens2,
                        "time_difference": time1 - time2,
                        "efficiency_ratio": (acc1 / tokens1) / (acc2 / tokens2) if tokens2 > 0 else 0.0
                    }
        
        return comparisons
    
    def _analyze_reasoning_steps(self) -> Dict[str, Any]:
        """Analyze reasoning steps patterns"""
        
        steps_by_condition = {}
        
        for trace in self.current_traces:
            condition = trace.condition
            if condition not in steps_by_condition:
                steps_by_condition[condition] = []
            
            steps_by_condition[condition].append({
                "total_steps": len(trace.reasoning_steps),
                "step_types": [step.step_type for step in trace.reasoning_steps],
                "avg_confidence": sum(step.confidence for step in trace.reasoning_steps) / len(trace.reasoning_steps) if trace.reasoning_steps else 0.0,
                "is_correct": trace.is_correct
            })
        
        return steps_by_condition
    
    def _analyze_confidence_calibration(self) -> Dict[str, Any]:
        """Analyze confidence calibration"""
        
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        calibration_data = []
        
        for i in range(len(bins) - 1):
            bin_traces = [t for t in self.current_traces 
                         if bins[i] <= t.confidence_score < bins[i+1]]
            
            if bin_traces:
                bin_accuracy = sum(t.is_correct for t in bin_traces) / len(bin_traces)
                bin_confidence = sum(t.confidence_score for t in bin_traces) / len(bin_traces)
                
                calibration_data.append({
                    "confidence_range": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                    "count": len(bin_traces),
                    "avg_confidence": bin_confidence,
                    "accuracy": bin_accuracy,
                    "calibration_error": abs(bin_confidence - bin_accuracy)
                })
        
        return {
            "calibration_bins": calibration_data,
            "expected_calibration_error": sum(d["calibration_error"] * d["count"] for d in calibration_data) / len(self.current_traces) if calibration_data else 0.0
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

def create_detailed_logger(output_dir: str, experiment_name: str) -> DetailedJSONLogger:
    """Factory function to create detailed logger"""
    return DetailedJSONLogger(output_dir, experiment_name)
