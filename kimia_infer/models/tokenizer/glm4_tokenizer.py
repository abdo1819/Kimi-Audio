import torch
import librosa
import os

from transformers import WhisperFeatureExtractor
from .glm4.speech_tokenizer.modeling_whisper import WhisperVQEncoder
from .glm4_utils import extract_speech_token
from torch import nn


class Glm4Tokenizer(nn.Module):
    def __init__(self, tokenizer_path):
        super().__init__()
        self.whisper_model = WhisperVQEncoder.from_pretrained(tokenizer_path).eval()
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)

    def tokenize(self, speech=None, audio_path=None, sr=16000, language=None):
        """
        Tokenize speech into discrete tokens using GLM4 speech tokenizer.
        
        Args:
            speech: Raw audio tensor (optional)
            audio_path: Path to audio file (optional)
            sr: Sample rate (default: 16000)
            language: Language hint (e.g., 'en' for English) - COMPATIBILITY PARAMETER
            
        Note:
            The language parameter is accepted for API compatibility but has limited
            effect in the GLM4 approach. Unlike standard Whisper which generates text
            tokens that can be language-specific, GLM4 uses vector quantization to
            create discrete representations of speech features themselves.
            
            This means the tokens represent acoustic/phonetic characteristics rather
            than language-specific text, so traditional language forcing mechanisms
            don't apply in the same way.
            
        Returns:
            torch.Tensor: Quantized speech tokens
        """
        if audio_path:
            audio, sr = librosa.load(audio_path, sr=16000)
            audio = torch.tensor(audio).unsqueeze(0)
            audio_info = (audio, sr)
        else:
            assert speech is not None
            assert sr
            if isinstance(speech, list):
                speech = torch.tensor(speech).unsqueeze(0)
            if len(speech.shape) == 1:
                speech = speech.unsqueeze(0)
            audio_info = (speech, sr)

        # Note: GLM4 tokenizer doesn't natively support language specification
        # The language parameter is accepted for compatibility but not used
        audio_tokens = extract_speech_token(
            self.whisper_model, self.feature_extractor, [audio_info], language=language
        )[0]
        audio_tokens = torch.tensor(audio_tokens).unsqueeze(0)
        return audio_tokens
