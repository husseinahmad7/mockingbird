"""ASR (Automatic Speech Recognition) service implementation using faster-whisper."""

import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import tempfile

try:
    from faster_whisper import WhisperModel
    from faster_whisper.transcribe import Segment as WhisperSegment
except ImportError:
    # Fallback for when faster-whisper is not available
    WhisperModel = None
    WhisperSegment = None

from .base import BaseASRService
from ..models.core import Segment, ProcessingConfig


logger = logging.getLogger(__name__)


class ASRService(BaseASRService):
    """ASR service implementation using faster-whisper."""
    
    def __init__(self, config: ProcessingConfig):
        """Initialize the ASR service with configuration."""
        self.config = config
        self.model: Optional[WhisperModel] = None
        self.current_model_size: Optional[str] = None
        
        # Language code mapping for faster-whisper
        self.language_codes = {
            'english': 'en',
            'spanish': 'es',
            'french': 'fr',
            'german': 'de',
            'italian': 'it',
            'portuguese': 'pt',
            'russian': 'ru',
            'japanese': 'ja',
            'korean': 'ko',
            'chinese': 'zh',
            'arabic': 'ar',
            'hindi': 'hi',
        }
    
    def load_model(self, model_size: str = "base") -> None:
        """Load the faster-whisper model with specified size.
        
        Args:
            model_size: Model size ('tiny', 'base', 'small', 'medium', 'large')
        
        Raises:
            RuntimeError: If faster-whisper is not available or model loading fails
        """
        if WhisperModel is None:
            raise RuntimeError("faster-whisper is not available. Please install it.")
        
        if self.model is not None and self.current_model_size == model_size:
            logger.info(f"Model {model_size} already loaded")
            return
        
        try:
            logger.info(f"Loading faster-whisper model: {model_size}")
            
            # Use CPU with int8 quantization for optimization as per requirements
            self.model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8"
            )
            self.current_model_size = model_size
            logger.info(f"Successfully loaded model: {model_size}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_size}: {e}")
            raise RuntimeError(f"Failed to load ASR model: {e}")
    
    def detect_language(self, audio_path: str) -> str:
        """Detect the language of the audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Detected language code (e.g., 'en', 'es', 'fr')
            
        Raises:
            RuntimeError: If model is not loaded or detection fails
        """
        if self.model is None:
            self.load_model(self.config.whisper_model_size)
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            logger.info(f"Detecting language for: {audio_path}")
            
            # Use faster-whisper's language detection
            segments, info = self.model.transcribe(
                audio_path,
                language=None,  # Auto-detect
                task="transcribe",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            detected_language = info.language
            confidence = info.language_probability
            
            logger.info(f"Detected language: {detected_language} (confidence: {confidence:.2f})")
            
            return detected_language
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            raise RuntimeError(f"Language detection failed: {e}")
    
    def transcribe(self, audio_path: str, source_language: Optional[str] = None) -> List[Segment]:
        """Transcribe audio file to segments with timestamps.
        
        Args:
            audio_path: Path to the audio file
            source_language: Optional source language code
            
        Returns:
            List of transcribed segments with timestamps
            
        Raises:
            RuntimeError: If model is not loaded or transcription fails
        """
        if self.model is None:
            self.load_model(self.config.whisper_model_size)
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            logger.info(f"Transcribing audio: {audio_path}")
            
            # Convert language name to code if needed
            language = None
            if source_language:
                language = self.language_codes.get(source_language.lower(), source_language)
            
            # Configure transcription parameters
            transcribe_params = {
                "language": language,
                "task": "transcribe",
                "vad_filter": True,
                "vad_parameters": dict(min_silence_duration_ms=500),
                "word_timestamps": True,
            }
            
            # Add speaker detection if enabled
            if self.config.enable_speaker_detection:
                # Note: faster-whisper doesn't have built-in speaker diarization
                # This would require additional libraries like pyannote.audio
                # For now, we'll set speaker_id to None and log a warning
                logger.warning("Speaker detection requested but not implemented in this version")
            
            segments, info = self.model.transcribe(audio_path, **transcribe_params)
            
            # Convert faster-whisper segments to our Segment model
            result_segments = []
            speaker_counter = 0
            
            for segment in segments:
                # Simple speaker detection based on silence gaps
                # This is a basic implementation - real speaker diarization would be more complex
                speaker_id = None
                if self.config.enable_speaker_detection:
                    # Assign speaker based on segment gaps (very basic approach)
                    if len(result_segments) == 0 or (segment.start - result_segments[-1].end_time) > 2.0:
                        speaker_counter += 1
                    speaker_id = f"speaker_{speaker_counter % 4 + 1}"  # Assume max 4 speakers
                
                result_segment = Segment(
                    start_time=segment.start,
                    end_time=segment.end,
                    text=segment.text.strip(),
                    speaker_id=speaker_id,
                    confidence=getattr(segment, 'avg_logprob', 0.0)
                )
                
                # Validate segment timing
                if result_segment.start_time < result_segment.end_time:
                    result_segments.append(result_segment)
                else:
                    logger.warning(f"Invalid segment timing: {result_segment.start_time} >= {result_segment.end_time}")
            
            logger.info(f"Transcription completed: {len(result_segments)} segments")
            return result_segments
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Transcription failed: {e}")
    
    def _validate_audio_file(self, audio_path: str) -> bool:
        """Validate that the audio file exists and is readable.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            path = Path(audio_path)
            if not path.exists():
                logger.error(f"Audio file does not exist: {audio_path}")
                return False
            
            if not path.is_file():
                logger.error(f"Path is not a file: {audio_path}")
                return False
            
            if path.stat().st_size == 0:
                logger.error(f"Audio file is empty: {audio_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Audio file validation failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "model_size": self.current_model_size,
            "device": "cpu",
            "compute_type": "int8"
        }