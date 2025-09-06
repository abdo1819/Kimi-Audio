"""
Example usage of Chain of Thought (CoT) functionality in Kimi Audio model
Demonstrates how to separate thinking from response generation
"""

import os
import soundfile as sf
from kimia_infer.api.cot_kimia import CoTKimiAudio


def main():
    # Initialize CoT-enhanced Kimi Audio model
    model = CoTKimiAudio(
        model_path="moonshotai/Kimi-Audio-7B-Instruct",
        load_detokenizer=True,
        enable_cot=True
    )
    
    # Generation parameters
    sampling_params = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
    }
    
    print("=== Chain of Thought Examples ===\n")
    
    # Example 1: ASR with thinking process
    print("1. ASR with Chain of Thought:")
    messages_asr = [
        {"role": "user", "message_type": "text", "content": "Please transcribe the spoken content into written text."},
        {"role": "user", "message_type": "audio", "content": "test_audios/asr_example.wav"},
    ]
    
    # Generate with thinking process visible
    audio_output, text_response, thinking = model.generate_with_cot(
        messages_asr,
        output_type="text",
        max_thinking_tokens=150,
        show_thinking=True,
        **sampling_params
    )
    
    print(f"Thinking process: {thinking}")
    print(f"Final response: {text_response}\n")
    
    # Example 2: Audio Q&A with thinking
    print("2. Audio Q&A with Thinking:")
    messages_qa = [
        {"role": "user", "message_type": "audio", "content": "test_audios/qa_example.wav"}
    ]
    
    # Generate with thinking process
    audio_output, text_response, thinking = model.generate_with_cot(
        messages_qa,
        output_type="text",
        max_thinking_tokens=200,
        show_thinking=True,
        **sampling_params
    )
    
    print(f"Thinking process: {thinking}")
    print(f"Response: {text_response}\n")
    
    # Example 3: Thinking-only generation
    print("3. Thinking-only Generation:")
    messages_thinking = [
        {"role": "user", "message_type": "text", "content": "What are the main themes in this audio content?"},
        {"role": "user", "message_type": "audio", "content": "test_audios/qa_example.wav"},
    ]
    
    thinking_process = model.generate_thinking_only(
        messages_thinking,
        max_thinking_tokens=100,
        **sampling_params
    )
    
    print(f"Thinking process: {thinking_process}\n")
    
    # Example 4: Audio-to-Audio with CoT
    print("4. Audio-to-Audio with Chain of Thought:")
    messages_audio = [
        {"role": "user", "message_type": "audio", "content": "test_audios/qa_example.wav"}
    ]
    
    # Generate both audio and text with thinking
    audio_output, text_response, thinking = model.generate_with_cot(
        messages_audio,
        output_type="both",
        max_thinking_tokens=150,
        show_thinking=True,
        **sampling_params
    )
    
    # Save generated audio
    output_dir = "test_audios/cot_output"
    os.makedirs(output_dir, exist_ok=True)
    sf.write(
        os.path.join(output_dir, "cot_response.wav"),
        audio_output.detach().cpu().view(-1).numpy(),
        24000,
    )
    
    print(f"Thinking process: {thinking}")
    print(f"Text response: {text_response}")
    print(f"Audio saved to: {os.path.join(output_dir, 'cot_response.wav')}\n")
    
    # Example 5: Different max thinking tokens
    print("5. Different Max Thinking Tokens:")
    
    # Short thinking
    audio_output, text_response, thinking = model.generate_with_cot(
        messages_asr,
        output_type="text",
        max_thinking_tokens=50,
        show_thinking=True,
        **sampling_params
    )
    
    print(f"Short thinking (50 tokens): {thinking}")
    print(f"Response: {text_response}\n")
    
    # Long thinking
    audio_output, text_response, thinking = model.generate_with_cot(
        messages_asr,
        output_type="text",
        max_thinking_tokens=300,
        show_thinking=True,
        **sampling_params
    )
    
    print(f"Long thinking (300 tokens): {thinking}")
    print(f"Response: {text_response}")


if __name__ == "__main__":
    main()
