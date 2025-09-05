"""
Prompting Strategies for CoT vs Latent Reasoning Comparison

Implements different prompting approaches:
- Chain-of-Thought (CoT): Zero-shot, Few-shot, Descriptive
- Latent reasoning: Silent steps, Loop emulation
"""

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json

@dataclass
class PromptTemplate:
    """Template for generating prompts"""
    system_prompt: str = ""
    user_prompt: str = ""
    reasoning_instruction: str = ""
    output_format: str = ""
    examples: List[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.examples is None:
            self.examples = []

class ReasoningPrompter(ABC):
    """Abstract base class for reasoning prompters"""
    
    def __init__(self, condition_type: str):
        self.condition_type = condition_type
        self.templates = self._load_templates()
    
    @abstractmethod
    def _load_templates(self) -> Dict[str, PromptTemplate]:
        """Load prompt templates for this prompter type"""
        pass
    
    @abstractmethod
    def generate_prompt(self, task: str, context: Dict[str, Any]) -> str:
        """Generate a prompt for the given task and context"""
        pass
    
    @abstractmethod
    def extract_answer(self, response: str) -> str:
        """Extract the final answer from the model response"""
        pass

class CoTPrompter(ReasoningPrompter):
    """Chain-of-Thought prompting strategies"""
    
    def __init__(self, 
                 condition_type: str,
                 chain_lengths: List[int] = None,
                 self_consistency_k: List[int] = None,
                 use_skill_retrieval: bool = False):
        super().__init__(condition_type)
        self.chain_lengths = chain_lengths or [0, 16, 64]
        self.self_consistency_k = self_consistency_k or [1, 3]
        self.use_skill_retrieval = use_skill_retrieval
        self.rationale_bank = self._build_rationale_bank()
    
    def _load_templates(self) -> Dict[str, PromptTemplate]:
        """Load CoT prompt templates"""
        templates = {}
        
        # Zero-shot CoT template
        templates["zero_shot"] = PromptTemplate(
            system_prompt="You are an expert at audio analysis and reasoning.",
            reasoning_instruction="Let's think step by step about this audio.",
            output_format="Provide your reasoning, then give your final answer after 'Final Answer:'",
        )
        
        # Few-shot CoT template
        templates["few_shot"] = PromptTemplate(
            system_prompt="You are an expert at audio analysis and reasoning. Here are some examples:",
            reasoning_instruction="Following the examples above, analyze this audio step by step.",
            output_format="Show your reasoning process, then give your final answer after 'Final Answer:'",
            examples=[
                {
                    "audio_description": "Audio contains a person counting: 'one, two, three, four'",
                    "question": "How many numbers are spoken?",
                    "reasoning": "I need to count the numbers spoken in the audio. I hear: 'one' (1), 'two' (2), 'three' (3), 'four' (4). That's 4 numbers total.",
                    "answer": "4"
                },
                {
                    "audio_description": "Audio has two speakers: Speaker A says 'Hello', then Speaker B says 'How are you?', then Speaker A says 'Fine, thanks'",
                    "question": "Who spoke last?",
                    "reasoning": "Let me trace the conversation: First, Speaker A says 'Hello'. Then Speaker B says 'How are you?'. Finally, Speaker A says 'Fine, thanks'. So Speaker A spoke last.",
                    "answer": "Speaker A"
                }
            ]
        )
        
        # Descriptive CoT template (Audio-CoT style)
        templates["descriptive"] = PromptTemplate(
            system_prompt="You are an expert at audio analysis. First describe the audio in detail, then reason about it.",
            reasoning_instruction="First, provide a detailed description of what you hear in the audio. Then use this description to reason step by step.",
            output_format="Description: [describe the audio]\nReasoning: [step by step reasoning]\nFinal Answer: [answer]",
        )
        
        return templates
    
    def _build_rationale_bank(self) -> List[Dict[str, Any]]:
        """Build a bank of reasoning examples for skill retrieval (LaRS-style)"""
        return [
            {
                "skill": "counting",
                "audio_type": "speech",
                "rationale": "To count items in audio, I listen for each distinct item and keep track: first item (1), second item (2), etc.",
                "example_question": "How many times is X mentioned?",
            },
            {
                "skill": "temporal_ordering", 
                "audio_type": "conversation",
                "rationale": "To determine order, I track the sequence: first event happens at time T1, second event at T2 where T2 > T1, etc.",
                "example_question": "What happened first/last?",
            },
            {
                "skill": "speaker_identification",
                "audio_type": "multi_speaker",
                "rationale": "To identify speakers, I listen for voice characteristics and track who says what in sequence.",
                "example_question": "Who said X?",
            },
            {
                "skill": "arithmetic",
                "audio_type": "numbers",
                "rationale": "For arithmetic on audio numbers, I first identify all numbers, then perform the calculation step by step.",
                "example_question": "What is the sum/difference/product?",
            }
        ]
    
    def _retrieve_relevant_rationales(self, task: str, context: Dict[str, Any], k: int = 2) -> List[Dict[str, Any]]:
        """Retrieve relevant rationales using skill similarity (LaRS-style)"""
        if not self.use_skill_retrieval:
            return []
        
        # Simple keyword-based retrieval (could be enhanced with embeddings)
        task_lower = task.lower()
        context_text = str(context).lower()
        combined_text = f"{task_lower} {context_text}"
        
        scored_rationales = []
        for rationale in self.rationale_bank:
            score = 0
            # Score based on skill relevance
            if any(keyword in combined_text for keyword in rationale["skill"].split("_")):
                score += 2
            if rationale["audio_type"] in combined_text:
                score += 1
            
            scored_rationales.append((score, rationale))
        
        # Return top-k rationales
        scored_rationales.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in scored_rationales[:k]]
    
    def generate_prompt(self, task: str, context: Dict[str, Any]) -> str:
        """Generate CoT prompt based on condition type"""
        template = self.templates.get(self.condition_type.replace("cot_", ""), self.templates["zero_shot"])
        
        # Build prompt components
        parts = []
        
        # System prompt
        if template.system_prompt:
            parts.append(template.system_prompt)
        
        # Add examples for few-shot
        if template.examples:
            parts.append("\nExamples:")
            for i, example in enumerate(template.examples, 1):
                parts.append(f"\nExample {i}:")
                parts.append(f"Audio: {example['audio_description']}")
                parts.append(f"Question: {example['question']}")
                parts.append(f"Reasoning: {example['reasoning']}")
                parts.append(f"Final Answer: {example['answer']}")
        
        # Add skill-based rationales if using retrieval
        if self.use_skill_retrieval:
            rationales = self._retrieve_relevant_rationales(task, context)
            if rationales:
                parts.append("\nRelevant reasoning strategies:")
                for rationale in rationales:
                    parts.append(f"- {rationale['skill']}: {rationale['rationale']}")
        
        # Main task
        parts.append(f"\nTask: {task}")
        
        # Add context information
        if context:
            parts.append("Context:")
            for key, value in context.items():
                if key != "audio_path":  # Don't include file paths in prompt
                    parts.append(f"- {key}: {value}")
        
        # Reasoning instruction
        parts.append(f"\n{template.reasoning_instruction}")
        
        # Output format
        parts.append(f"\n{template.output_format}")
        
        return "\n".join(parts)
    
    def extract_answer(self, response: str) -> str:
        """Extract final answer from CoT response"""
        # Look for "Final Answer:" pattern
        if "Final Answer:" in response:
            answer_part = response.split("Final Answer:")[-1].strip()
            # Take first line of the answer
            return answer_part.split("\n")[0].strip()
        
        # Fallback: take last non-empty line
        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
        return lines[-1] if lines else ""

class LatentPrompter(ReasoningPrompter):
    """Latent reasoning prompting strategies"""
    
    def __init__(self, 
                 condition_type: str,
                 loop_depths: List[int] = None,
                 use_timestep_embeddings: bool = False):
        super().__init__(condition_type)
        self.loop_depths = loop_depths or [0, 2, 4, 8]
        self.use_timestep_embeddings = use_timestep_embeddings
    
    def _load_templates(self) -> Dict[str, PromptTemplate]:
        """Load latent reasoning prompt templates"""
        templates = {}
        
        # Silent reasoning template
        templates["silent"] = PromptTemplate(
            system_prompt="You are an expert at audio analysis. Think internally before responding.",
            reasoning_instruction="Consider the audio carefully through internal reasoning steps, then provide your final answer.",
            output_format="Give only your final answer without showing intermediate steps.",
        )
        
        # Loop emulation template
        templates["loops"] = PromptTemplate(
            system_prompt="You are an expert at audio analysis with iterative reasoning capabilities.",
            reasoning_instruction="Process this audio through multiple internal reasoning iterations to refine your understanding.",
            output_format="After internal processing, provide your final answer.",
        )
        
        return templates
    
    def generate_prompt(self, task: str, context: Dict[str, Any]) -> str:
        """Generate latent reasoning prompt"""
        template_key = self.condition_type.replace("latent_", "")
        template = self.templates.get(template_key, self.templates["silent"])
        
        # Determine number of reasoning loops for this instance
        loop_depth = self._select_loop_depth(context)
        
        # Build prompt
        parts = []
        
        # System prompt
        parts.append(template.system_prompt)
        
        # Add loop instruction if using loops
        if "loops" in self.condition_type and loop_depth > 0:
            parts.append(f"\nUse {loop_depth} internal reasoning iterations to analyze the audio.")
            if self.use_timestep_embeddings:
                parts.append("Consider each iteration as a refinement step building on the previous.")
        
        # Main task
        parts.append(f"\nTask: {task}")
        
        # Context (without revealing it's for latent reasoning)
        if context:
            context_items = [f"- {k}: {v}" for k, v in context.items() if k != "audio_path"]
            if context_items:
                parts.append("Context:")
                parts.extend(context_items)
        
        # Reasoning instruction
        parts.append(f"\n{template.reasoning_instruction}")
        
        # Output format
        parts.append(f"\n{template.output_format}")
        
        # Add a constraint to prevent explicit reasoning output
        if "silent" in self.condition_type:
            parts.append("\nDo not show your reasoning process - only provide the final answer.")
        
        return "\n".join(parts)
    
    def _select_loop_depth(self, context: Dict[str, Any]) -> int:
        """Select appropriate loop depth based on task complexity"""
        if not self.loop_depths:
            return 0
        
        # Simple heuristic: more complex tasks get more loops
        complexity_indicators = [
            "count", "how many", "sum", "total",  # counting tasks
            "first", "last", "before", "after",   # temporal reasoning
            "who", "which speaker",               # speaker identification
        ]
        
        task_text = str(context).lower()
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in task_text)
        
        # Map complexity to loop depth
        if complexity_score >= 3:
            return max(self.loop_depths)
        elif complexity_score >= 2:
            return self.loop_depths[len(self.loop_depths)//2] if len(self.loop_depths) > 2 else self.loop_depths[-1]
        elif complexity_score >= 1:
            return self.loop_depths[1] if len(self.loop_depths) > 1 else self.loop_depths[0]
        else:
            return self.loop_depths[0]
    
    def extract_answer(self, response: str) -> str:
        """Extract answer from latent reasoning response"""
        # For latent reasoning, the entire response should be the answer
        # But clean it up by taking the first substantive line
        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
        
        # Remove any leaked reasoning patterns
        cleaned_lines = []
        for line in lines:
            # Skip lines that look like reasoning steps
            if not any(pattern in line.lower() for pattern in [
                "let me think", "first,", "second,", "then,", "therefore,", 
                "step 1", "step 2", "reasoning:", "because"
            ]):
                cleaned_lines.append(line)
        
        # Return the first clean line, or original first line if no clean lines
        if cleaned_lines:
            return cleaned_lines[0]
        elif lines:
            return lines[0]
        else:
            return ""

def get_prompter(condition: str, **kwargs) -> ReasoningPrompter:
    """Factory function to get appropriate prompter"""
    if condition.startswith("cot"):
        return CoTPrompter(condition, **kwargs)
    elif condition.startswith("latent"):
        return LatentPrompter(condition, **kwargs)
    else:
        raise ValueError(f"Unknown prompting condition: {condition}")
