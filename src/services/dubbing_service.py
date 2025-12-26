"""Dubbing service for creating dubbed videos with TTS and audio mixing."""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from ..models.core import Segment, AudioFile, ProcessingConfig
from .tts_service import TTSService
from .audio_processing import AudioProcessingService

logger = logging.getLogger(__name__)


class DubbingService:
    """Service for creating dubbed videos with TTS and audio mixing."""
    
    def __init__(self, config: ProcessingConfig):
        """Initialize the dubbing service.
        
        Args:
            config: Processing configuration
        """
        self.config = config
        self.tts_service = TTSService(config)
        self.audio_service = AudioProcessingService()
        
    def create_dubbed_video(
        self,
        video_path: str,
        translated_segments: List[Segment],
        output_path: str,
        target_language: str = "en",
        voice: Optional[str] = None,
        ducking_level: float = 0.3,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> str:
        """Create a dubbed video with translated speech.

        Args:
            video_path: Path to original video file
            translated_segments: List of translated text segments with timing
            output_path: Path for output dubbed video
            target_language: Target language code (e.g., 'ar', 'es', 'fr')
            voice: Voice to use for TTS (auto-selected if None)
            ducking_level: Background audio reduction level (0.0-1.0)
            progress_callback: Optional callback for progress updates (message, progress)

        Returns:
            Path to dubbed video file

        Raises:
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If dubbing fails
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if not translated_segments:
            raise ValueError("No translated segments provided")

        try:
            # Step 1: Extract original audio
            if progress_callback:
                progress_callback("Extracting audio from video...", 0.1)

            logger.info(f"Extracting audio from {video_path}")
            original_audio = self.audio_service.extract_audio(video_path)

            # Step 2: Select voice if not provided
            if not voice:
                voice = self._select_voice(target_language)
                logger.info(f"Auto-selected voice: {voice}")
            
            # Step 3: Generate TTS for each segment
            if progress_callback:
                progress_callback("Generating speech for translated segments...", 0.2)
            
            tts_segments = self._generate_tts_segments(
                translated_segments,
                voice,
                progress_callback
            )
            
            # Step 4: Apply volume ducking to original audio
            if progress_callback:
                progress_callback("Applying volume ducking...", 0.6)
            
            speech_timings = [
                {'start_time': seg['start_time'], 'end_time': seg['end_time']}
                for seg in tts_segments
            ]
            ducked_audio = self.audio_service.apply_volume_ducking(
                original_audio,
                speech_timings,
                ducking_level
            )
            
            # Step 5: Mix ducked background with TTS segments
            if progress_callback:
                progress_callback("Mixing audio tracks...", 0.7)
            
            mixed_audio = self.audio_service.mix_audio_tracks(
                ducked_audio,
                tts_segments
            )
            
            # Step 6: Create final video with dubbed audio
            if progress_callback:
                progress_callback("Creating final video...", 0.9)
            
            final_video = self.audio_service.create_final_video(
                video_path,
                mixed_audio
            )
            
            # Step 7: Move to output path
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(final_video, output_path)
            
            if progress_callback:
                progress_callback("Dubbing complete!", 1.0)
            
            logger.info(f"Successfully created dubbed video: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Dubbing failed: {str(e)}")
            raise RuntimeError(f"Failed to create dubbed video: {str(e)}")
    
    def _generate_tts_segments(
        self,
        segments: List[Segment],
        voice: str,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> List[Dict[str, Any]]:
        """Generate TTS audio for all segments.
        
        Args:
            segments: List of text segments
            voice: Voice to use for TTS
            progress_callback: Optional progress callback
            
        Returns:
            List of dicts with 'audio_file', 'start_time', 'end_time'
        """
        tts_segments = []
        total = len(segments)
        
        for i, segment in enumerate(segments):
            if progress_callback:
                progress = 0.2 + (0.4 * (i / total))
                progress_callback(f"Generating speech {i+1}/{total}...", progress)
            
            # Calculate target duration for this segment
            target_duration = segment.end_time - segment.start_time
            
            # Calculate speed adjustment to fit duration
            speed_factor = self.tts_service.calculate_speed_adjustment(
                segment.text,
                target_duration
            )
            
            # Generate TTS audio
            audio_file = self.tts_service.generate_speech(
                segment,
                voice,
                speed_factor
            )
            
            tts_segments.append({
                'audio_file': audio_file,
                'start_time': segment.start_time,
                'end_time': segment.end_time
            })
        
        return tts_segments

    def _select_voice(self, language: str) -> str:
        """Select an appropriate voice for the target language.

        Args:
            language: Target language code

        Returns:
            Voice identifier for TTS
        """
        # Get available voices for the language
        voices = self.tts_service.get_available_voices(language)

        if not voices:
            # Fallback to English if no voices found
            logger.warning(f"No voices found for language {language}, using English")
            voices = self.tts_service.get_available_voices('en')

        # Return first available voice
        # In production, you might want to let users choose or use gender/style preferences
        return voices[0] if voices else 'en-US-AriaNeural'

    def cleanup(self):
        """Clean up temporary files."""
        self.tts_service.cleanup()
        self.audio_service.cleanup_temp_files()

