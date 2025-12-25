"""Base service interfaces and abstract classes."""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..models.core import Segment, AudioFile, ProcessingConfig


class BaseASRService(ABC):
    """Abstract base class for Automatic Speech Recognition services."""
    
    @abstractmethod
    def transcribe(self, audio_path: str, source_language: Optional[str] = None) -> List[Segment]:
        """Transcribe audio file to segments with timestamps."""
        pass
    
    @abstractmethod
    def load_model(self, model_size: str = "base") -> None:
        """Load the ASR model with specified size."""
        pass
    
    @abstractmethod
    def detect_language(self, audio_path: str) -> str:
        """Detect the language of the audio file."""
        pass


class BaseTranslationService(ABC):
    """Abstract base class for translation services."""
    
    @abstractmethod
    def translate_segments(self, segments: List[Segment], target_language: str) -> List[Segment]:
        """Translate a list of segments to the target language."""
        pass
    
    @abstractmethod
    def translate_batch(self, texts: List[str], target_language: str) -> List[str]:
        """Translate a batch of texts to the target language."""
        pass
    
    @abstractmethod
    def fallback_translate(self, text: str, target_language: str) -> str:
        """Fallback translation method for when primary service fails."""
        pass


class BaseTTSService(ABC):
    """Abstract base class for Text-to-Speech services."""
    
    @abstractmethod
    def generate_speech(self, segment: Segment, voice: str, speed_factor: float) -> AudioFile:
        """Generate speech audio from a text segment."""
        pass
    
    @abstractmethod
    def calculate_speed_adjustment(self, text: str, target_duration: float) -> float:
        """Calculate the speed adjustment needed to fit text in target duration."""
        pass
    
    @abstractmethod
    def get_available_voices(self, language: str) -> List[str]:
        """Get list of available voices for a language."""
        pass


class BaseAudioProcessingService(ABC):
    """Abstract base class for audio processing services."""
    
    @abstractmethod
    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video file and return audio file path."""
        pass
    
    @abstractmethod
    def mix_audio_tracks(self, original: str, tts_segments: List[AudioFile]) -> str:
        """Mix original audio with TTS segments and return mixed audio path."""
        pass
    
    @abstractmethod
    def apply_volume_ducking(self, background: str, speech_segments: List[AudioFile]) -> str:
        """Apply volume ducking to background audio during speech segments."""
        pass
    
    @abstractmethod
    def create_final_video(self, video_path: str, dubbed_audio: str) -> str:
        """Create final video with dubbed audio track."""
        pass


class BaseFileHandler(ABC):
    """Abstract base class for file handling services."""
    
    @abstractmethod
    def validate_file(self, file_path: str) -> bool:
        """Validate if file is acceptable for processing."""
        pass
    
    @abstractmethod
    def validate_url(self, url: str) -> bool:
        """Validate if URL is from a supported streaming service."""
        pass
    
    @abstractmethod
    def get_file_info(self, file_path: str) -> dict:
        """Get file information including size, format, duration."""
        pass
    
    @abstractmethod
    def download_from_url(self, url: str) -> str:
        """Download content from URL and return local file path."""
        pass
    
    @abstractmethod
    def create_temp_file(self, suffix: str = '.tmp') -> str:
        """Create a temporary file and return its path."""
        pass
    
    @abstractmethod
    def cleanup_temp_files(self) -> None:
        """Clean up all temporary files and directories."""
        pass