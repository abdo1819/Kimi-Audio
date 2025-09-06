"""
Chain of Thought enhanced Kimi Audio model
Extends the base KimiAudio class to support thinking/response separation
"""

import torch
from typing import List, Dict, Tuple, Optional
from loguru import logger

from kimia_infer.api.kimia import KimiAudio
from kimia_infer.utils.cot_tokens import CoTTokens, instantiate_cot_tokens, create_cot_start_template, parse_cot_response


class CoTKimiAudio(KimiAudio):
    """
    Chain of Thought enhanced Kimi Audio model
    Supports separating internal reasoning from final responses
    """
    
    def __init__(self, model_path: str, load_detokenizer: bool = True, enable_cot: bool = True):
        super().__init__(model_path, load_detokenizer)
        
        self.enable_cot = enable_cot
        if enable_cot:
            try:
                self.cot_tokens = instantiate_cot_tokens(self.prompt_manager.text_tokenizer)
                logger.info("Chain of Thought tokens loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load CoT tokens: {e}. CoT functionality disabled.")
                self.enable_cot = False
                self.cot_tokens = None
    
    def generate_with_cot(
        self,
        chats: List[Dict],
        output_type: str = "text",
        max_thinking_tokens: int = 200,
        show_thinking: bool = False,
        **generation_kwargs
    ) -> Tuple[Optional[torch.Tensor], str, Optional[str]]:
        """
        Generate response with chain of thought reasoning
        
        Args:
            chats: Conversation history
            output_type: "text" or "both"
            max_thinking_tokens: Maximum tokens for thinking before auto-ending
            show_thinking: Whether to return thinking process
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Tuple of (audio_output, text_response, thinking_process)
        """
        if not self.enable_cot:
            logger.warning("CoT not enabled, falling back to standard generation")
            audio_output, text_output = self.generate(chats, output_type=output_type, **generation_kwargs)
            return audio_output, text_output, None
        
        # Create CoT-enhanced conversation with enforced start
        enhanced_chats = self._enforce_cot_start(chats)
        
        # Generate with CoT and max thinking tokens
        audio_output, full_text_output = self._generate_with_thinking_limit(
            enhanced_chats, 
            output_type=output_type,
            max_thinking_tokens=max_thinking_tokens,
            **generation_kwargs
        )
        
        # Parse thinking and response
        thinking_process, final_response = parse_cot_response(full_text_output)
        
        if show_thinking:
            return audio_output, final_response, thinking_process
        else:
            return audio_output, final_response, None
    
    def _enforce_cot_start(self, chats: List[Dict]) -> List[Dict]:
        """
        Enforce CoT start token at the beginning of generation
        
        Args:
            chats: Original conversation
            
        Returns:
            Enhanced conversation with CoT start enforced
        """
        if not self.enable_cot:
            return chats
        
        # Create CoT start template
        cot_start = create_cot_start_template()
        
        # Add CoT start to the conversation
        enhanced_chats = chats.copy()
        
        # Add CoT start as a new user message
        enhanced_chats.append({
            "role": "user",
            "message_type": "text", 
            "content": cot_start
        })
        
        return enhanced_chats
    
    def _generate_with_thinking_limit(
        self,
        chats: List[Dict],
        output_type: str = "text",
        max_thinking_tokens: int = 200,
        **generation_kwargs
    ) -> Tuple[Optional[torch.Tensor], str]:
        """
        Generate with thinking token limit that auto-ends thinking
        
        Args:
            chats: Conversation history
            output_type: "text" or "both"
            max_thinking_tokens: Maximum tokens for thinking
            **generation_kwargs: Generation parameters
            
        Returns:
            Tuple of (audio_output, text_output)
        """
        # Get the base generation parameters
        max_new_tokens = generation_kwargs.get('max_new_tokens', -1)
        
        # Calculate total max tokens including thinking
        if max_new_tokens == -1:
            if output_type == "both":
                total_max_tokens = int(12.5 * 120)  # Audio generation limit
            else:
                total_max_tokens = 7500  # Text generation limit
        else:
            total_max_tokens = max_new_tokens
        
        # Set max tokens for generation
        generation_kwargs['max_new_tokens'] = total_max_tokens
        
        # Generate normally first
        audio_output, text_output = self.generate(
            chats,
            output_type=output_type,
            **generation_kwargs
        )
        
        # Post-process to enforce thinking end if needed
        if self.enable_cot and "<|im_kimia_thinking_start|>" in text_output:
            text_output = self._enforce_thinking_end(text_output, max_thinking_tokens)
        
        return audio_output, text_output
    
    def _enforce_thinking_end(self, text_output: str, max_thinking_tokens: int) -> str:
        """
        Enforce thinking end token if thinking is too long
        
        Args:
            text_output: Generated text output
            max_thinking_tokens: Maximum tokens for thinking
            
        Returns:
            Modified text output with enforced thinking end
        """
        thinking_start = "<|im_kimia_thinking_start|>"
        thinking_end = "<|im_kimia_thinking_end|>"
        
        if thinking_start not in text_output:
            return text_output
        
        # Find thinking start position
        start_pos = text_output.find(thinking_start)
        thinking_content_start = start_pos + len(thinking_start)
        
        # Check if thinking end already exists
        if thinking_end in text_output:
            return text_output
        
        # Count tokens in thinking section (rough estimation)
        thinking_section = text_output[thinking_content_start:]
        # Simple token count estimation (spaces + 1)
        thinking_tokens = len(thinking_section.split())
        
        # If thinking is too long, force end
        if thinking_tokens > max_thinking_tokens:
            # Find a good place to end (end of sentence or word)
            end_pos = thinking_content_start + max_thinking_tokens * 4  # Rough character estimation
            if end_pos < len(text_output):
                # Try to end at sentence boundary
                for i in range(min(50, len(text_output) - end_pos)):
                    if text_output[end_pos + i] in '.!?':
                        end_pos = end_pos + i + 1
                        break
                
                # Insert thinking end token
                text_output = (text_output[:end_pos] + 
                             f" {thinking_end} <|im_kimia_response_start|>" + 
                             text_output[end_pos:])
        
        return text_output
    
    def generate_thinking_only(
        self,
        chats: List[Dict],
        max_thinking_tokens: int = 200,
        **generation_kwargs
    ) -> str:
        """
        Generate only the thinking process without final response
        
        Args:
            chats: Conversation history
            max_thinking_tokens: Maximum tokens for thinking
            **generation_kwargs: Generation parameters
            
        Returns:
            Thinking process text
        """
        if not self.enable_cot:
            raise ValueError("CoT not enabled")
        
        # Use the main CoT generation but only return thinking
        _, _, thinking = self.generate_with_cot(
            chats,
            output_type="text",
            max_thinking_tokens=max_thinking_tokens,
            show_thinking=True,
            **generation_kwargs
        )
        
        return thinking or ""
