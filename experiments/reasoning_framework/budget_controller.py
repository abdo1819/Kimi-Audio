"""
Compute Budget Controller for Fair CoT vs Latent Comparison

Ensures fair comparison by matching computational budgets across different reasoning approaches:
- Token budget matching
- FLOPs budget matching  
- Time budget matching
"""

import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class BudgetType(Enum):
    TOKENS = "tokens"
    FLOPS = "flops" 
    TIME = "time"

@dataclass
class BudgetConstraints:
    """Budget constraints for generation"""
    max_tokens: Optional[int] = None
    max_flops: Optional[float] = None
    max_time: Optional[float] = None
    target_accuracy: Optional[float] = None  # For adaptive budgeting

@dataclass
class BudgetUsage:
    """Actual budget usage during generation"""
    tokens_used: int = 0
    flops_used: float = 0.0
    time_used: float = 0.0
    within_budget: bool = True
    budget_efficiency: float = 0.0  # performance per unit budget

class ComputeBudgetController:
    """Controller for managing compute budgets across reasoning conditions"""
    
    def __init__(self, 
                 matching_strategy: str = "flops",
                 max_tokens: int = 1000,
                 max_flops: float = 1e12,
                 max_time: float = 30.0):
        self.matching_strategy = BudgetType(matching_strategy)
        self.constraints = BudgetConstraints(
            max_tokens=max_tokens,
            max_flops=max_flops,
            max_time=max_time
        )
        self.usage_history = []
    
    def calculate_cot_budget(self, 
                           chain_length: int, 
                           self_consistency_k: int = 1) -> BudgetConstraints:
        """Calculate budget for CoT condition"""
        base_tokens = self.constraints.max_tokens or 500
        
        # CoT uses more tokens for reasoning chains
        cot_tokens = base_tokens + chain_length
        # Self-consistency multiplies the budget
        total_tokens = cot_tokens * self_consistency_k
        
        # Estimate FLOPs and time based on tokens
        estimated_flops = self._tokens_to_flops(total_tokens)
        estimated_time = self._tokens_to_time(total_tokens) * self_consistency_k
        
        return BudgetConstraints(
            max_tokens=total_tokens,
            max_flops=estimated_flops,
            max_time=estimated_time
        )
    
    def calculate_latent_budget(self, 
                              loop_depth: int,
                              base_tokens: Optional[int] = None) -> BudgetConstraints:
        """Calculate equivalent budget for latent reasoning condition"""
        if base_tokens is None:
            base_tokens = self.constraints.max_tokens or 500
        
        # Latent reasoning uses fewer output tokens but more internal computation
        # Each loop adds internal computation cost
        latent_tokens = base_tokens // 2  # Fewer output tokens
        
        # Internal loops add computational cost
        internal_compute_multiplier = 1 + (loop_depth * 0.5)  # Each loop adds 50% compute
        
        estimated_flops = self._tokens_to_flops(latent_tokens) * internal_compute_multiplier
        estimated_time = self._tokens_to_time(latent_tokens) * internal_compute_multiplier
        
        return BudgetConstraints(
            max_tokens=latent_tokens,
            max_flops=estimated_flops,
            max_time=estimated_time
        )
    
    def match_budgets(self, 
                     cot_config: Dict[str, Any], 
                     latent_config: Dict[str, Any]) -> Tuple[BudgetConstraints, BudgetConstraints]:
        """Match budgets between CoT and latent conditions"""
        
        # Calculate individual budgets
        cot_budget = self.calculate_cot_budget(
            chain_length=cot_config.get("chain_length", 32),
            self_consistency_k=cot_config.get("self_consistency_k", 1)
        )
        
        latent_budget = self.calculate_latent_budget(
            loop_depth=latent_config.get("loop_depth", 2)
        )
        
        # Match based on strategy
        if self.matching_strategy == BudgetType.TOKENS:
            # Match token budgets
            target_tokens = min(cot_budget.max_tokens, latent_budget.max_tokens)
            cot_budget.max_tokens = target_tokens
            latent_budget.max_tokens = target_tokens
            
        elif self.matching_strategy == BudgetType.FLOPS:
            # Match FLOP budgets
            target_flops = min(cot_budget.max_flops, latent_budget.max_flops)
            cot_budget.max_flops = target_flops
            latent_budget.max_flops = target_flops
            
            # Adjust token budgets to match FLOPs
            cot_budget.max_tokens = self._flops_to_tokens(target_flops)
            latent_budget.max_tokens = self._flops_to_tokens(target_flops)
            
        elif self.matching_strategy == BudgetType.TIME:
            # Match time budgets
            target_time = min(cot_budget.max_time, latent_budget.max_time)
            cot_budget.max_time = target_time
            latent_budget.max_time = target_time
            
            # Adjust other budgets to match time
            cot_budget.max_tokens = self._time_to_tokens(target_time)
            latent_budget.max_tokens = self._time_to_tokens(target_time)
        
        return cot_budget, latent_budget
    
    def monitor_usage(self, 
                     budget: BudgetConstraints,
                     actual_tokens: int,
                     actual_flops: float,
                     actual_time: float) -> BudgetUsage:
        """Monitor actual budget usage vs constraints"""
        
        within_budget = True
        violations = []
        
        # Check token budget
        if budget.max_tokens and actual_tokens > budget.max_tokens:
            within_budget = False
            violations.append(f"tokens: {actual_tokens} > {budget.max_tokens}")
        
        # Check FLOP budget
        if budget.max_flops and actual_flops > budget.max_flops:
            within_budget = False
            violations.append(f"flops: {actual_flops:.2e} > {budget.max_flops:.2e}")
        
        # Check time budget
        if budget.max_time and actual_time > budget.max_time:
            within_budget = False
            violations.append(f"time: {actual_time:.2f}s > {budget.max_time:.2f}s")
        
        # Calculate efficiency based on primary matching strategy
        if self.matching_strategy == BudgetType.TOKENS:
            efficiency = actual_tokens / (budget.max_tokens or 1)
        elif self.matching_strategy == BudgetType.FLOPS:
            efficiency = actual_flops / (budget.max_flops or 1)
        else:  # TIME
            efficiency = actual_time / (budget.max_time or 1)
        
        usage = BudgetUsage(
            tokens_used=actual_tokens,
            flops_used=actual_flops,
            time_used=actual_time,
            within_budget=within_budget,
            budget_efficiency=efficiency
        )
        
        self.usage_history.append(usage)
        
        if not within_budget:
            print(f"⚠️ Budget exceeded: {', '.join(violations)}")
        
        return usage
    
    def _tokens_to_flops(self, tokens: int) -> float:
        """Estimate FLOPs from token count"""
        # Rough approximation: ~6 * model_params * tokens for transformer
        # Assuming 7B parameter model
        model_params = 7e9
        return 6 * model_params * tokens
    
    def _tokens_to_time(self, tokens: int) -> float:
        """Estimate time from token count"""
        # Rough approximation: ~20 tokens per second for large models
        tokens_per_second = 20
        return tokens / tokens_per_second
    
    def _flops_to_tokens(self, flops: float) -> int:
        """Convert FLOP budget to token budget"""
        model_params = 7e9
        return int(flops / (6 * model_params))
    
    def _time_to_tokens(self, time_seconds: float) -> int:
        """Convert time budget to token budget"""
        tokens_per_second = 20
        return int(time_seconds * tokens_per_second)
    
    def get_efficiency_stats(self) -> Dict[str, float]:
        """Get efficiency statistics across all usage history"""
        if not self.usage_history:
            return {}
        
        efficiencies = [usage.budget_efficiency for usage in self.usage_history]
        
        return {
            "mean_efficiency": sum(efficiencies) / len(efficiencies),
            "max_efficiency": max(efficiencies),
            "min_efficiency": min(efficiencies),
            "budget_violations": sum(1 for usage in self.usage_history if not usage.within_budget),
            "total_samples": len(self.usage_history)
        }
    
    def adaptive_budget_adjustment(self, 
                                 current_performance: float,
                                 target_performance: float,
                                 adjustment_factor: float = 0.1) -> BudgetConstraints:
        """Adaptively adjust budget based on performance"""
        if target_performance <= 0:
            return self.constraints
        
        performance_ratio = current_performance / target_performance
        
        # If performance is below target, increase budget
        if performance_ratio < 1.0:
            budget_multiplier = 1 + adjustment_factor * (1 - performance_ratio)
        # If performance exceeds target, we can reduce budget
        else:
            budget_multiplier = 1 - adjustment_factor * (performance_ratio - 1) * 0.5
        
        # Apply multiplier to constraints
        adjusted_constraints = BudgetConstraints(
            max_tokens=int((self.constraints.max_tokens or 500) * budget_multiplier),
            max_flops=(self.constraints.max_flops or 1e12) * budget_multiplier,
            max_time=(self.constraints.max_time or 30.0) * budget_multiplier
        )
        
        return adjusted_constraints
