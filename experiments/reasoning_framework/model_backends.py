"""
Model Backend Abstraction for Audio-Text LLMs

Provides a unified interface for different audio-text models (Kimia, Qwen2-Audio, etc.)
to enable consistent experimentation across different model architectures.
"""

import time
import tempfile
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import torch
import soundfile as sf
from dataclasses import dataclass

from kimia_infer.api.kimia import KimiAudio

@dataclass
class GenerationStats:
    """Statistics from model generation"""
    output_text: str
    output_audio: Optional[torch.Tensor] = None
    num_input_tokens: int = 0
    num_output_tokens: int = 0
    generation_time: float = 0.0
    memory_usage: float = 0.0  # MB
    flops_estimate: float = 0.0  # Estimated FLOPs
    reasoning_steps: int = 0  # Number of reasoning steps used
    
    @property
    def token_efficiency(self) -> float:
        """Tokens per second"""
        if self.generation_time > 0:
            return self.num_output_tokens / self.generation_time
        return 0.0

class AudioTextModelBackend(ABC):
    """Abstract base class for audio-text model backends"""
    
    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self.model_name = self.__class__.__name__
        self.generation_stats_history = []
    
    @abstractmethod
    def generate(self, 
                 audio_path: str, 
                 text_prompt: str,
                 **generation_kwargs) -> GenerationStats:
        """
        Generate response given audio input and text prompt
        
        Args:
            audio_path: Path to audio file
            text_prompt: Text prompt/instruction
            **generation_kwargs: Model-specific generation parameters
            
        Returns:
            GenerationStats object with generation results and statistics
        """
        pass
    
    @abstractmethod
    def estimate_flops(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate FLOPs for given input/output token counts"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information (size, architecture, etc.)"""
        pass
    
    def reset_stats(self):
        """Reset generation statistics history"""
        self.generation_stats_history = []
    
    def get_average_stats(self) -> Dict[str, float]:
        """Get average statistics across all generations"""
        if not self.generation_stats_history:
            return {}
        
        stats = {}
        stats["avg_generation_time"] = sum(s.generation_time for s in self.generation_stats_history) / len(self.generation_stats_history)
        stats["avg_output_tokens"] = sum(s.num_output_tokens for s in self.generation_stats_history) / len(self.generation_stats_history)
        stats["avg_token_efficiency"] = sum(s.token_efficiency for s in self.generation_stats_history) / len(self.generation_stats_history)
        stats["avg_memory_usage"] = sum(s.memory_usage for s in self.generation_stats_history) / len(self.generation_stats_history)
        stats["total_generations"] = len(self.generation_stats_history)
        
        return stats

class KimiaBackend(AudioTextModelBackend):
    """Backend for Kimia Audio model"""
    
    def __init__(self, model_path: str = "moonshotai/Kimi-Audio-7B-Instruct", 
                 load_detokenizer: bool = True, **kwargs):
        super().__init__(model_path, **kwargs)
        self.load_detokenizer = load_detokenizer
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Kimia model"""
        self.model = KimiAudio(
            model_path=self.model_path,
            load_detokenizer=self.load_detokenizer
        )
    
    def generate(self, 
                 audio_path: str, 
                 text_prompt: str,
                 temperature: float = 0.0,
                 top_k: int = 5,
                 max_new_tokens: int = -1,
                 output_type: str = "text",
                 reasoning_steps: int = 0,  # For latent loop emulation
                 **kwargs) -> GenerationStats:
        """Generate response using Kimia model"""
        
        start_time = time.time()
        
        # Get memory usage before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            memory_before = 0.0
        
        # Prepare sampling parameters
        sampling_params = {
            "audio_temperature": 0.8,  # Kimia-specific defaults
            "audio_top_k": 10,
            "text_temperature": temperature,
            "text_top_k": top_k,
            "audio_repetition_penalty": 1.0,
            "audio_repetition_window_size": 64,
            "text_repetition_penalty": 1.0,
            "text_repetition_window_size": 16,
        }
        
        # Override with any provided kwargs
        sampling_params.update(kwargs)
        
        # Prepare messages
        messages = [
            {"role": "user", "message_type": "text", "content": text_prompt},
            {"role": "user", "message_type": "audio", "content": audio_path},
        ]
        
        try:
            # Generate response
            wav, text = self.model.generate(
                messages, 
                **sampling_params,
                output_type=output_type,
                max_new_tokens=max_new_tokens
            )
            
            generation_time = time.time() - start_time
            
            # Get memory usage after generation
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                memory_usage = memory_after - memory_before
            else:
                memory_usage = 0.0
            
            # Estimate token counts (rough approximation)
            # Input tokens: audio tokens + text tokens
            # Audio is typically ~50 tokens per second at 16kHz
            audio_duration = self._get_audio_duration(audio_path)
            estimated_audio_tokens = int(audio_duration * 50)
            estimated_text_input_tokens = len(text_prompt.split()) * 1.3  # rough word-to-token ratio
            
            total_input_tokens = int(estimated_audio_tokens + estimated_text_input_tokens)
            output_tokens = len(text.split()) * 1.3 if text else 0  # rough approximation
            
            # Estimate FLOPs
            flops_estimate = self.estimate_flops(total_input_tokens, int(output_tokens))
            
            # Create stats object
            stats = GenerationStats(
                output_text=text or "",
                output_audio=wav,
                num_input_tokens=total_input_tokens,
                num_output_tokens=int(output_tokens),
                generation_time=generation_time,
                memory_usage=memory_usage,
                flops_estimate=flops_estimate,
                reasoning_steps=reasoning_steps
            )
            
            # Store in history
            self.generation_stats_history.append(stats)
            
            return stats
            
        except Exception as e:
            # Return error stats
            generation_time = time.time() - start_time
            stats = GenerationStats(
                output_text=f"ERROR: {str(e)}",
                generation_time=generation_time,
                reasoning_steps=reasoning_steps
            )
            self.generation_stats_history.append(stats)
            return stats
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds"""
        try:
            data, sr = sf.read(audio_path)
            return len(data) / sr
        except:
            return 1.0  # fallback duration
    
    def estimate_flops(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate FLOPs for Kimia model
        Rough approximation: ~6 * num_params * (input_tokens + output_tokens)
        Assuming 7B parameters
        """
        num_params = 7e9  # 7B parameters
        total_tokens = input_tokens + output_tokens
        # Forward pass: 2 * params * tokens, backward pass: 4 * params * tokens (but we're inference only)
        flops = 2 * num_params * total_tokens
        return flops
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Kimia model information"""
        return {
            "model_name": "Kimia-Audio-7B-Instruct",
            "model_path": self.model_path,
            "estimated_parameters": "7B",
            "architecture": "Audio-Text Multimodal LLM",
            "supports_audio_input": True,
            "supports_audio_output": self.load_detokenizer,
            "max_context_length": 8192,  # approximate
        }

class Qwen2AudioBackend(AudioTextModelBackend):
    """Backend for Qwen2-Audio model (placeholder implementation)"""
    
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)
        # TODO: Implement Qwen2-Audio loading
        raise NotImplementedError("Qwen2-Audio backend not yet implemented")
    
    def generate(self, audio_path: str, text_prompt: str, **kwargs) -> GenerationStats:
        # TODO: Implement Qwen2-Audio generation
        raise NotImplementedError("Qwen2-Audio generation not yet implemented")
    
    def estimate_flops(self, input_tokens: int, output_tokens: int) -> float:
        # TODO: Implement FLOPs estimation for Qwen2-Audio
        return 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": "Qwen2-Audio",
            "model_path": self.model_path,
            "status": "not_implemented"
        }

def get_backend(backend_name: str, **kwargs) -> AudioTextModelBackend:
    """Factory function to get model backend by name"""
    backends = {
        "kimia": KimiaBackend,
        "qwen2_audio": Qwen2AudioBackend,
    }
    
    if backend_name not in backends:
        raise ValueError(f"Unknown backend: {backend_name}. Available: {list(backends.keys())}")
    
    return backends[backend_name](**kwargs)
