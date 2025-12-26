"""Text-to-Speech service implementation using Edge-TTS."""

import asyncio
import os
import tempfile
import time
from typing import Dict, List, Optional
import logging

try:
    import edge_tts
except ImportError:
    edge_tts = None

from .base import BaseTTSService
from ..models.core import Segment, AudioFile, ProcessingConfig


logger = logging.getLogger(__name__)


class TTSService(BaseTTSService):
    """Text-to-Speech service using Microsoft Edge TTS."""
    
    def __init__(self, config: ProcessingConfig):
        """Initialize TTS service with configuration."""
        if edge_tts is None:
            raise ImportError("edge-tts package is required for TTS functionality")
        
        self.config = config
        self.voice_cache: Dict[str, List[str]] = {}
        self.speaker_voice_mapping: Dict[str, str] = {}
        self.temp_files: List[str] = []
        
    def generate_speech(self, segment: Segment, voice: str, speed_factor: float = 1.0) -> AudioFile:
        """Generate speech audio from a text segment."""
        if not segment.text.strip():
            raise ValueError("Cannot generate speech from empty text")
            
        # Validate speed factor bounds
        speed_factor = max(self.config.min_speed_adjustment, 
                          min(self.config.max_speed_adjustment, speed_factor))
        
        # Create temporary file for audio output
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_file.close()
        self.temp_files.append(temp_file.name)
        
        try:
            # Generate speech using edge-tts
            asyncio.run(self._generate_speech_async(segment.text, voice, speed_factor, temp_file.name))
            
            # Get audio file info
            duration = self._get_audio_duration(temp_file.name)
            
            return AudioFile(
                path=temp_file.name,
                duration=duration,
                sample_rate=22050,  # Edge-TTS default
                channels=1
            )
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
                self.temp_files.remove(temp_file.name)
            raise RuntimeError(f"TTS generation failed: {str(e)}")
    
    async def _generate_speech_async(self, text: str, voice: str, speed_factor: float, output_path: str):
        """Async helper for speech generation."""
        # Calculate rate adjustment for Edge-TTS
        # Edge-TTS rate format: +X% or -X%
        rate_percent = int((speed_factor - 1.0) * 100)
        rate_str = f"+{rate_percent}%" if rate_percent >= 0 else f"{rate_percent}%"
        
        communicate = edge_tts.Communicate(text, voice, rate=rate_str)
        await communicate.save(output_path)
    
    def calculate_speed_adjustment(self, text: str, target_duration: float) -> float:
        """Calculate the speed adjustment needed to fit text in target duration."""
        if target_duration <= 0:
            return 1.0
            
        # Estimate speech duration based on text length
        # Rough estimate: ~150 words per minute for normal speech
        words = len(text.split())
        estimated_duration = (words / 150.0) * 60.0  # Convert to seconds
        
        if estimated_duration <= 0:
            return 1.0
            
        # Calculate required speed factor
        speed_factor = estimated_duration / target_duration
        
        # Clamp to acceptable bounds
        return max(self.config.min_speed_adjustment, 
                  min(self.config.max_speed_adjustment, speed_factor))
    
    def get_available_voices(self, language: str) -> List[str]:
        """Get list of available voices for a language."""
        if language in self.voice_cache:
            return self.voice_cache[language]
        
        try:
            # Get all available voices
            voices = asyncio.run(edge_tts.list_voices())
            
            # Filter by language
            language_voices = []
            for voice in voices:
                voice_locale = voice.get('Locale', '').lower()
                if language.lower() in voice_locale or voice_locale.startswith(language.lower()[:2]):
                    language_voices.append(voice['ShortName'])
            
            # Cache the result
            self.voice_cache[language] = language_voices
            return language_voices
            
        except Exception as e:
            logger.warning(f"Failed to get voices for language {language}: {e}")
            # Return some common voices as fallback
            return self._get_fallback_voices(language)
    
    def _get_fallback_voices(self, language: str) -> List[str]:
        """Get fallback voices when API call fails."""
        fallback_voices = {
            'en': ['en-US-AriaNeural', 'en-US-JennyNeural', 'en-US-GuyNeural'],
            'es': ['es-ES-ElviraNeural', 'es-ES-AlvaroNeural'],
            'fr': ['fr-FR-DeniseNeural', 'fr-FR-HenriNeural'],
            'de': ['de-DE-KatjaNeural', 'de-DE-ConradNeural'],
            'it': ['it-IT-ElsaNeural', 'it-IT-DiegoNeural'],
            'pt': ['pt-BR-FranciscaNeural', 'pt-BR-AntonioNeural'],
            'ja': ['ja-JP-NanamiNeural', 'ja-JP-KeitaNeural'],
            'ko': ['ko-KR-SunHiNeural', 'ko-KR-InJoonNeural'],
            'zh': ['zh-CN-XiaoxiaoNeural', 'zh-CN-YunxiNeural'],
        }
        
        # Try exact match first, then prefix match
        lang_key = language.lower()
        if lang_key in fallback_voices:
            return fallback_voices[lang_key]
        
        # Try prefix match (e.g., 'en-US' -> 'en')
        for key, voices in fallback_voices.items():
            if lang_key.startswith(key):
                return voices
        
        # Default to English if no match
        return fallback_voices['en']
    
    def map_speaker_to_voice(self, speaker_id: str, language: str, preferred_gender: Optional[str] = None) -> str:
        """Map a speaker ID to a consistent voice."""
        if speaker_id in self.speaker_voice_mapping:
            return self.speaker_voice_mapping[speaker_id]
        
        available_voices = self.get_available_voices(language)
        if not available_voices:
            raise RuntimeError(f"No voices available for language: {language}")
        
        # Simple mapping strategy: use speaker_id hash to select voice consistently
        voice_index = hash(speaker_id) % len(available_voices)
        selected_voice = available_voices[voice_index]
        
        # Cache the mapping
        self.speaker_voice_mapping[speaker_id] = selected_voice
        return selected_voice
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file."""
        try:
            # Try to use a simple method to get duration
            # This is a placeholder - in a real implementation you'd use
            # a library like librosa or pydub
            import wave
            with wave.open(audio_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                return frames / float(sample_rate)
        except Exception:
            # Fallback: estimate based on file size (very rough)
            file_size = os.path.getsize(audio_path)
            # Rough estimate: 16-bit mono at 22050 Hz = ~44KB per second
            return file_size / 44100.0
    
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