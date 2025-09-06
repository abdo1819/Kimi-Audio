"""
Chain of Thought (CoT) tokens for Kimi Audio model
Simplified implementation with basic thinking/response separation
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CoTTokens:
    """Chain of Thought special tokens - simplified version"""
    thinking_start: int  # <|im_kimia_thinking_start|>
    thinking_end: int    # <|im_kimia_thinking_end|>
    response_start: int  # <|im_kimia_response_start|>


def instantiate_cot_tokens(tokenizer) -> CoTTokens:
    """Instantiate CoT tokens from tokenizer"""
    if hasattr(tokenizer, "special_tokens"):
        map_fn = lambda x: tokenizer.special_tokens[x]
    elif hasattr(tokenizer, "convert_tokens_to_ids"):
        map_fn = lambda x: tokenizer.convert_tokens_to_ids(x)
    else:
        raise ValueError(f"Invalid tokenizer type: {type(tokenizer)}")
    
    return CoTTokens(
        thinking_start=map_fn("<|im_kimia_thinking_start|>"),
        thinking_end=map_fn("<|im_kimia_thinking_end|>"),
        response_start=map_fn("<|im_kimia_response_start|>"),
    )


def create_cot_start_template() -> str:
    """
    Create a simple CoT start template that enforces thinking at the beginning
    
    Returns:
        Simple CoT start template
    """
    return "<|im_kimia_thinking_start|>"


def parse_cot_response(full_response: str) -> tuple[str, str]:
    """
    Parse a response that contains both thinking and final response
    
    Args:
        full_response: Complete response with thinking and response parts
        
    Returns:
        Tuple of (thinking_part, response_part)
    """
    thinking_start = "<|im_kimia_thinking_start|>"
    thinking_end = "<|im_kimia_thinking_end|>"
    response_start = "<|im_kimia_response_start|>"
    
    thinking_part = ""
    response_part = ""
    
    # Extract thinking part
    if thinking_start in full_response and thinking_end in full_response:
        start_idx = full_response.find(thinking_start) + len(thinking_start)
        end_idx = full_response.find(thinking_end)
        thinking_part = full_response[start_idx:end_idx].strip()
    
    # Extract response part
    if response_start in full_response:
        start_idx = full_response.find(response_start) + len(response_start)
        response_part = full_response[start_idx:].strip()
    else:
        # If no response start token, everything after thinking is response
        if thinking_end in full_response:
            start_idx = full_response.find(thinking_end) + len(thinking_end)
            response_part = full_response[start_idx:].strip()
        else:
            response_part = full_response
    
    return thinking_part, response_part
