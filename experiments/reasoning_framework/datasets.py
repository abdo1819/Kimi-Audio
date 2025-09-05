"""
Dataset Loaders for Audio Reasoning Tasks

Provides datasets for testing CoT vs Latent reasoning:
- LibriSpeech QA tasks
- Synthetic temporal/counting reasoning
- Multi-hop reasoning tasks
- Noise robustness evaluation
"""

import os
import json
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import soundfile as sf
from datasets import load_dataset, Dataset

# Import existing LibriSpeech utilities if available
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from test_asr import get_librispeech_cache_dir, load_librispeech_local
    LIBRISPEECH_AVAILABLE = True
except ImportError:
    LIBRISPEECH_AVAILABLE = False

@dataclass
class ReasoningTask:
    """A single reasoning task"""
    audio_path: str
    question: str
    answer: str
    task_type: str  # "counting", "temporal", "speaker_id", "arithmetic", etc.
    difficulty: str  # "easy", "medium", "hard"
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}

class ReasoningDatasetLoader:
    """Loader for audio reasoning datasets"""
    
    def __init__(self):
        self.cache_dir = Path.home() / ".cache" / "audio_reasoning"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self, 
                    dataset_name: str, 
                    noise_condition: str = "clean",
                    max_samples: int = 100) -> List[ReasoningTask]:
        """Load a reasoning dataset"""
        
        if dataset_name == "librispeech_qa":
            return self.load_librispeech_qa(noise_condition, max_samples)
        elif dataset_name == "synthetic_count":
            return self.load_synthetic_counting(noise_condition, max_samples)
        elif dataset_name == "temporal_reasoning":
            return self.load_temporal_reasoning(noise_condition, max_samples)
        elif dataset_name == "speaker_reasoning":
            return self.load_speaker_reasoning(noise_condition, max_samples)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def load_librispeech_qa(self, 
                           noise_condition: str = "clean", 
                           max_samples: int = 100) -> List[ReasoningTask]:
        """Load LibriSpeech-based QA tasks"""
        tasks = []
        
        if not LIBRISPEECH_AVAILABLE:
            # Create mock tasks for testing
            return self._create_mock_librispeech_tasks(max_samples)
        
        try:
            # Load LibriSpeech test-clean subset
            cache_root = get_librispeech_cache_dir()
            ds, _ = load_librispeech_local("test-clean", cache_root, max_samples=max_samples*2)
            
            # Convert to reasoning tasks
            sample_count = 0
            for item in ds:
                if sample_count >= max_samples:
                    break
                
                # Create QA pairs from LibriSpeech samples
                audio_path = self._save_audio_to_temp(item["audio"])
                text = item["text"]
                
                # Generate different types of questions
                qa_pairs = self._generate_librispeech_questions(text, audio_path)
                
                for question, answer, task_type, difficulty in qa_pairs:
                    if sample_count >= max_samples:
                        break
                    
                    # Apply noise if needed
                    if noise_condition != "clean":
                        audio_path = self._add_noise(audio_path, noise_condition)
                    
                    tasks.append(ReasoningTask(
                        audio_path=audio_path,
                        question=question,
                        answer=answer,
                        task_type=task_type,
                        difficulty=difficulty,
                        context={"original_text": text, "speaker_id": item.get("speaker_id", 0)}
                    ))
                    sample_count += 1
                    
        except Exception as e:
            print(f"Warning: Could not load LibriSpeech data: {e}")
            return self._create_mock_librispeech_tasks(max_samples)
        
        return tasks[:max_samples]
    
    def load_synthetic_counting(self, 
                               noise_condition: str = "clean", 
                               max_samples: int = 100) -> List[ReasoningTask]:
        """Generate synthetic counting tasks"""
        tasks = []
        
        for i in range(max_samples):
            # Generate counting task
            count = random.randint(2, 10)
            items = random.choice([
                "beeps", "clicks", "tones", "words", "numbers", "sounds"
            ])
            
            # Create synthetic audio with the specified count
            audio_path = self._create_counting_audio(items, count)
            
            # Apply noise if needed
            if noise_condition != "clean":
                audio_path = self._add_noise(audio_path, noise_condition)
            
            # Generate question variations
            question_templates = [
                f"How many {items} do you hear?",
                f"Count the number of {items} in the audio.",
                f"How many times do you hear {items[:-1]}?",  # singular form
            ]
            
            question = random.choice(question_templates)
            answer = str(count)
            
            # Determine difficulty based on count and noise
            if count <= 3 and noise_condition == "clean":
                difficulty = "easy"
            elif count <= 6 or noise_condition in ["snr_10", "snr_15"]:
                difficulty = "medium"
            else:
                difficulty = "hard"
            
            tasks.append(ReasoningTask(
                audio_path=audio_path,
                question=question,
                answer=answer,
                task_type="counting",
                difficulty=difficulty,
                context={"true_count": count, "item_type": items}
            ))
        
        return tasks
    
    def load_temporal_reasoning(self, 
                               noise_condition: str = "clean", 
                               max_samples: int = 100) -> List[ReasoningTask]:
        """Generate temporal reasoning tasks"""
        tasks = []
        
        for i in range(max_samples):
            # Generate sequence of events
            num_events = random.randint(3, 6)
            events = []
            event_types = ["beep", "tone", "click", "voice", "music", "noise"]
            
            for j in range(num_events):
                event = {
                    "type": random.choice(event_types),
                    "duration": random.uniform(0.5, 1.5),
                    "position": j
                }
                events.append(event)
            
            # Create synthetic audio with temporal sequence
            audio_path = self._create_temporal_audio(events)
            
            # Apply noise if needed
            if noise_condition != "clean":
                audio_path = self._add_noise(audio_path, noise_condition)
            
            # Generate temporal reasoning questions
            question_types = [
                ("first", "What type of sound occurs first?", events[0]["type"]),
                ("last", "What type of sound occurs last?", events[-1]["type"]),
                ("before", f"What comes before the {events[2]['type']}?", events[1]["type"]),
                ("after", f"What comes after the {events[1]['type']}?", events[2]["type"]),
            ]
            
            question_type, question, answer = random.choice(question_types)
            
            # Determine difficulty
            if num_events <= 3 and noise_condition == "clean":
                difficulty = "easy"
            elif num_events <= 4 or noise_condition in ["snr_10", "snr_15"]:
                difficulty = "medium"
            else:
                difficulty = "hard"
            
            tasks.append(ReasoningTask(
                audio_path=audio_path,
                question=question,
                answer=answer,
                task_type="temporal",
                difficulty=difficulty,
                context={"events": events, "num_events": num_events}
            ))
        
        return tasks
    
    def load_speaker_reasoning(self, 
                              noise_condition: str = "clean", 
                              max_samples: int = 100) -> List[ReasoningTask]:
        """Generate speaker identification reasoning tasks"""
        tasks = []
        
        # For now, create mock speaker tasks
        # In a real implementation, this would use multi-speaker audio
        for i in range(max_samples):
            num_speakers = random.randint(2, 4)
            speakers = [f"Speaker_{chr(65+j)}" for j in range(num_speakers)]
            
            # Mock conversation structure
            conversation = []
            for turn in range(random.randint(3, 6)):
                speaker = random.choice(speakers)
                utterance = f"Utterance {turn+1}"
                conversation.append({"speaker": speaker, "text": utterance, "turn": turn})
            
            # Create mock audio path (would be real audio in full implementation)
            audio_path = self._create_mock_speaker_audio(conversation)
            
            # Apply noise if needed
            if noise_condition != "clean":
                audio_path = self._add_noise(audio_path, noise_condition)
            
            # Generate speaker reasoning questions
            question_types = [
                ("who_first", "Who spoke first?", conversation[0]["speaker"]),
                ("who_last", "Who spoke last?", conversation[-1]["speaker"]),
                ("count_speakers", "How many different speakers are there?", str(num_speakers)),
            ]
            
            question_type, question, answer = random.choice(question_types)
            
            difficulty = "medium" if num_speakers <= 3 else "hard"
            if noise_condition not in ["clean", "snr_15"]:
                difficulty = "hard"
            
            tasks.append(ReasoningTask(
                audio_path=audio_path,
                question=question,
                answer=answer,
                task_type="speaker_id",
                difficulty=difficulty,
                context={"conversation": conversation, "num_speakers": num_speakers}
            ))
        
        return tasks
    
    def _create_mock_librispeech_tasks(self, max_samples: int) -> List[ReasoningTask]:
        """Create mock LibriSpeech tasks for testing"""
        tasks = []
        
        for i in range(max_samples):
            # Create a simple mock audio file
            audio_path = self._create_mock_audio()
            
            # Mock questions about the audio
            questions = [
                ("What is the main topic of the speech?", "reading", "content", "easy"),
                ("How many sentences are spoken?", "3", "counting", "medium"),
                ("What is the speaker's gender?", "male", "classification", "easy"),
            ]
            
            question, answer, task_type, difficulty = random.choice(questions)
            
            tasks.append(ReasoningTask(
                audio_path=audio_path,
                question=question,
                answer=answer,
                task_type=task_type,
                difficulty=difficulty,
                context={"mock": True}
            ))
        
        return tasks
    
    def _save_audio_to_temp(self, audio_data: Dict[str, Any]) -> str:
        """Save audio data to temporary file"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_data["array"], audio_data["sampling_rate"])
            return tmp.name
    
    def _create_mock_audio(self, duration: float = 3.0, sr: int = 16000) -> str:
        """Create mock audio file"""
        # Generate simple sine wave
        t = np.linspace(0, duration, int(duration * sr))
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sr)
            return tmp.name
    
    def _create_counting_audio(self, item_type: str, count: int, sr: int = 16000) -> str:
        """Create synthetic audio for counting tasks"""
        duration_per_item = 0.5
        pause_duration = 0.3
        total_duration = count * (duration_per_item + pause_duration)
        
        # Create audio with specified number of items
        t = np.linspace(0, total_duration, int(total_duration * sr))
        audio = np.zeros_like(t)
        
        for i in range(count):
            start_time = i * (duration_per_item + pause_duration)
            start_idx = int(start_time * sr)
            end_idx = int((start_time + duration_per_item) * sr)
            
            if end_idx < len(audio):
                # Create different sounds for different item types
                if "beep" in item_type:
                    freq = 800 + i * 100  # Different pitch for each beep
                    item_audio = 0.3 * np.sin(2 * np.pi * freq * np.linspace(0, duration_per_item, end_idx - start_idx))
                else:
                    # Default to clicks
                    item_audio = 0.3 * np.random.normal(0, 0.1, end_idx - start_idx)
                
                audio[start_idx:end_idx] = item_audio
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sr)
            return tmp.name
    
    def _create_temporal_audio(self, events: List[Dict[str, Any]], sr: int = 16000) -> str:
        """Create synthetic audio with temporal sequence"""
        total_duration = sum(event["duration"] for event in events) + len(events) * 0.2
        audio = np.zeros(int(total_duration * sr))
        
        current_time = 0
        for event in events:
            start_idx = int(current_time * sr)
            duration = event["duration"]
            end_idx = int((current_time + duration) * sr)
            
            if end_idx < len(audio):
                # Create different sounds for different event types
                if event["type"] == "beep":
                    freq = 800
                    t = np.linspace(0, duration, end_idx - start_idx)
                    event_audio = 0.3 * np.sin(2 * np.pi * freq * t)
                elif event["type"] == "tone":
                    freq = 1200
                    t = np.linspace(0, duration, end_idx - start_idx)
                    event_audio = 0.3 * np.sin(2 * np.pi * freq * t)
                else:
                    # Default noise
                    event_audio = 0.2 * np.random.normal(0, 0.1, end_idx - start_idx)
                
                audio[start_idx:end_idx] = event_audio
            
            current_time += duration + 0.2  # Add pause between events
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sr)
            return tmp.name
    
    def _create_mock_speaker_audio(self, conversation: List[Dict[str, Any]], sr: int = 16000) -> str:
        """Create mock multi-speaker audio"""
        # For now, just create a simple tone sequence
        # In a real implementation, this would use TTS or real speaker audio
        duration_per_turn = 1.0
        pause_duration = 0.3
        total_duration = len(conversation) * (duration_per_turn + pause_duration)
        
        audio = np.zeros(int(total_duration * sr))
        
        # Different frequencies for different speakers
        speaker_freqs = {"Speaker_A": 300, "Speaker_B": 400, "Speaker_C": 500, "Speaker_D": 600}
        
        for i, turn in enumerate(conversation):
            start_time = i * (duration_per_turn + pause_duration)
            start_idx = int(start_time * sr)
            end_idx = int((start_time + duration_per_turn) * sr)
            
            if end_idx < len(audio):
                freq = speaker_freqs.get(turn["speaker"], 350)
                t = np.linspace(0, duration_per_turn, end_idx - start_idx)
                turn_audio = 0.2 * np.sin(2 * np.pi * freq * t)
                audio[start_idx:end_idx] = turn_audio
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sr)
            return tmp.name
    
    def _add_noise(self, audio_path: str, noise_condition: str) -> str:
        """Add noise to audio file"""
        # Load original audio
        audio, sr = sf.read(audio_path)
        
        # Determine SNR based on condition
        snr_db = {
            "snr_20": 20,
            "snr_15": 15,
            "snr_10": 10,
            "snr_5": 5,
            "snr_0": 0,
        }.get(noise_condition, 20)
        
        # Add white noise
        signal_power = np.mean(audio ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power), audio.shape)
        
        noisy_audio = audio + noise
        
        # Save noisy audio to new temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, noisy_audio, sr)
            return tmp.name
    
    def _generate_librispeech_questions(self, text: str, audio_path: str) -> List[Tuple[str, str, str, str]]:
        """Generate QA pairs from LibriSpeech text"""
        questions = []
        words = text.split()
        
        # Word count question
        questions.append((
            "How many words are spoken in this audio?",
            str(len(words)),
            "counting",
            "easy"
        ))
        
        # First/last word questions
        if len(words) >= 2:
            questions.append((
                "What is the first word spoken?",
                words[0].lower(),
                "temporal",
                "easy"
            ))
            questions.append((
                "What is the last word spoken?",
                words[-1].lower(),
                "temporal",
                "easy"
            ))
        
        # Content questions
        if len(words) >= 5:
            # Pick a word from the middle
            middle_idx = len(words) // 2
            middle_word = words[middle_idx].lower()
            questions.append((
                f"Is the word '{middle_word}' mentioned in the audio?",
                "yes",
                "content",
                "medium"
            ))
        
        return questions[:2]  # Return max 2 questions per sample
