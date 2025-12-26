"""Text-to-Speech service implementation using XTTS-v2 for voice cloning."""

import os
import tempfile
import logging
from typing import Dict, List, Optional
import torch
import torchaudio

try:
    from TTS.api import TTS
except ImportError:
    TTS = None

from .base import BaseTTSService
from ..models.core import Segment, AudioFile, ProcessingConfig


logger = logging.getLogger(__name__)


class XTTSService(BaseTTSService):
    """Text-to-Speech service using Coqui XTTS-v2 for voice cloning."""
    
    def __init__(self, config: ProcessingConfig):
        """Initialize XTTS service with configuration."""
        if TTS is None:
            raise ImportError("TTS package is required for XTTS functionality. Install with: pip install TTS")
        
        self.config = config
        self.temp_files: List[str] = []
        self.speaker_samples: Dict[str, str] = {}  # speaker_id -> audio sample path
        self.tts_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing XTTS-v2 on device: {self.device}")
        
    def _load_model(self):
        """Lazy load the XTTS model."""
        if self.tts_model is None:
            logger.info("Loading XTTS-v2 model...")
            self.tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            logger.info("XTTS-v2 model loaded successfully")
    
    def set_speaker_sample(self, speaker_id: str, audio_path: str):
        """Set a voice sample for a specific speaker for cloning.
        
        Args:
            speaker_id: Unique identifier for the speaker
            audio_path: Path to audio file containing the speaker's voice (3-10 seconds recommended)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Speaker sample not found: {audio_path}")
        
        self.speaker_samples[speaker_id] = audio_path
        logger.info(f"Set voice sample for speaker '{speaker_id}': {audio_path}")
    
    def extract_speaker_sample(self, video_audio_path: str, start_time: float, duration: float = 5.0) -> str:
        """Extract a speaker sample from video audio for voice cloning.
        
        Args:
            video_audio_path: Path to the video's audio file
            start_time: Start time in seconds
            duration: Duration of sample in seconds (default: 5.0)
            
        Returns:
            Path to extracted audio sample
        """
        import subprocess
        
        output_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        self.temp_files.append(output_path)
        
        # Use FFmpeg to extract the audio segment
        cmd = [
            'ffmpeg', '-i', video_audio_path,
            '-ss', str(start_time),
            '-t', str(duration),
            '-ar', '22050',  # XTTS works best with 22050 Hz
            '-ac', '1',  # Mono
            '-y', output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to extract speaker sample: {result.stderr}")
        
        logger.info(f"Extracted speaker sample: {start_time}s - {start_time + duration}s")
        return output_path
    
    def generate_speech(self, segment: Segment, voice: str, speed_factor: float = 1.0) -> AudioFile:
        """Generate speech audio from a text segment using voice cloning.
        
        Args:
            segment: Text segment to synthesize
            voice: Speaker ID (must have a sample set via set_speaker_sample)
            speed_factor: Speed adjustment factor (1.0 = normal speed)
            
        Returns:
            AudioFile object with generated speech
        """
        if not segment.text.strip():
            raise ValueError("Cannot generate speech from empty text")
        
        # Load model if not already loaded
        self._load_model()
        
        # Get speaker sample
        speaker_sample = self.speaker_samples.get(voice)
        if not speaker_sample:
            raise ValueError(f"No voice sample set for speaker '{voice}'. Use set_speaker_sample() first.")
        
        # Create temporary file for output
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_file.close()
        self.temp_files.append(temp_file.name)
        
        try:
            # Generate speech with voice cloning
            logger.info(f"Generating speech with XTTS for speaker '{voice}': {segment.text[:50]}...")
            
            self.tts_model.tts_to_file(
                text=segment.text,
                speaker_wav=speaker_sample,
                language=self._get_language_code(segment),
                file_path=temp_file.name,
                speed=speed_factor
            )
            
            # Get audio duration
            duration = self._get_audio_duration(temp_file.name)
            
            return AudioFile(
                path=temp_file.name,
                duration=duration,
                sample_rate=22050,
                channels=1
            )
            
        except Exception as e:
            logger.error(f"XTTS generation failed: {e}")
            # Clean up on failure
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            raise RuntimeError(f"Failed to generate speech with XTTS: {e}")

    def _get_language_code(self, segment: Segment) -> str:
        """Get XTTS language code from segment.

        XTTS supports: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko
        """
        # Map common language codes to XTTS codes
        lang_map = {
            'en': 'en', 'es': 'es', 'fr': 'fr', 'de': 'de', 'it': 'it',
            'pt': 'pt', 'pl': 'pl', 'tr': 'tr', 'ru': 'ru', 'nl': 'nl',
            'cs': 'cs', 'ar': 'ar', 'zh': 'zh-cn', 'ja': 'ja', 'hu': 'hu', 'ko': 'ko'
        }

        # Try to get language from segment metadata
        lang = getattr(segment, 'language', 'en')
        if isinstance(lang, str):
            lang = lang.lower()[:2]  # Get first 2 chars

        return lang_map.get(lang, 'en')

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file in seconds."""
        try:
            import wave
            with wave.open(audio_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                return frames / float(rate)
        except Exception:
            # Fallback: estimate based on file size
            file_size = os.path.getsize(audio_path)
            # Rough estimate: 16-bit mono at 22050 Hz = ~44KB per second
            return file_size / 44100.0

    def calculate_speed_adjustment(self, text: str, target_duration: float) -> float:
        """Calculate speed adjustment factor to fit text in target duration.

        Args:
            text: Text to be synthesized
            target_duration: Target duration in seconds

        Returns:
            Speed factor (1.0 = normal, >1.0 = faster, <1.0 = slower)
        """
        # Estimate natural duration (rough estimate: 150 words per minute)
        words = len(text.split())
        estimated_duration = (words / 150.0) * 60.0

        if estimated_duration == 0:
            return 1.0

        # Calculate required speed
        speed_factor = estimated_duration / target_duration

        # Clamp to reasonable bounds
        speed_factor = max(self.config.min_speed_adjustment,
                          min(self.config.max_speed_adjustment, speed_factor))

        return speed_factor

    def get_available_voices(self, language: str) -> List[str]:
        """Get list of available voices for a language.

        For XTTS, this returns the list of speaker IDs that have samples set.
        """
        return list(self.speaker_samples.keys())

    def cleanup_temp_files(self):
        """Clean up temporary files created by TTS service."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_file}: {e}")
        self.temp_files.clear()

    def cleanup(self):
        """Alias for cleanup_temp_files for consistency."""
        self.cleanup_temp_files()

    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'temp_files'):
            self.cleanup_temp_files()

