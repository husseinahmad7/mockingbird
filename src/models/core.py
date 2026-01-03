"""Core data models for the Video Translator System."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional


@dataclass
class Segment:
    """Represents a timestamped segment of transcribed text."""
    start_time: float
    end_time: float
    text: str
    speaker_id: Optional[str] = None
    confidence: float = 0.0
    
    @property
    def duration(self) -> float:
        """Calculate the duration of the segment."""
        return self.end_time - self.start_time


class JobStatus(Enum):
    """Status enumeration for translation jobs."""
    PENDING = "pending"
    EXTRACTING_AUDIO = "extracting_audio"
    TRANSCRIBING = "transcribing"
    TRANSLATING = "translating"
    GENERATING_SPEECH = "generating_speech"
    MIXING_AUDIO = "mixing_audio"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TranslationJob:
    """Represents a video translation job."""
    id: str
    source_file: str
    source_language: str
    target_language: str
    status: JobStatus
    segments: List[Segment]
    created_at: datetime
    completed_at: Optional[datetime] = None


@dataclass
class AudioFile:
    """Represents an audio file with metadata."""
    path: str
    duration: float
    sample_rate: int
    channels: int


@dataclass
class ProcessingConfig:
    """Configuration settings for video processing."""
    whisper_model_size: str = "base"
    enable_speaker_detection: bool = True
    max_speed_adjustment: float = 1.5
    min_speed_adjustment: float = 0.8
    volume_ducking_level: float = -10.0
    gemini_api_key: str = ""
    gemini_model: str = "gemma-3-27b-it"  # Default model, can be overridden
    batch_size: int = 20
    background_preservation_mode: str = "ducking"  # "ducking" or "separator"
    hf_token: Optional[str] = None  # Hugging Face token for pyannote.audio
    separation_model: str = "UVR-MDX-NET-Inst_HQ_4.onnx"  # Audio separation model
    save_separated_audio: bool = False  # Save vocals and background files
    use_serial_separation: bool = False  # Apply two separation models in series
    min_speaker_sample_duration: float = 6.0  # Minimum duration for speaker samples in seconds
    suppress_tf32_warning: bool = True  # Suppress TensorFloat-32 warnings from pyannote