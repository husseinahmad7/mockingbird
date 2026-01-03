"""Dubbing service for creating dubbed videos with TTS and audio mixing."""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from ..models.core import Segment, AudioFile, ProcessingConfig
from .tts_service import TTSService
from .tts_xtts_service import XTTSService
from .audio_processing import AudioProcessingService
from .audio_separator_service import AudioSeparatorService

logger = logging.getLogger(__name__)


class DubbingService:
    """Service for creating dubbed videos with TTS and audio mixing."""

    def __init__(self, config: ProcessingConfig, use_voice_cloning: bool = False):
        """Initialize the dubbing service.

        Args:
            config: Processing configuration
            use_voice_cloning: If True, use XTTS for voice cloning; if False, use Edge TTS
        """
        self.config = config
        self.use_voice_cloning = use_voice_cloning

        if use_voice_cloning:
            try:
                self.tts_service = XTTSService(config)
                logger.info("Using XTTS voice cloning for TTS")
            except ImportError as e:
                logger.warning(f"XTTS not available: {e}. Falling back to Edge TTS")
                self.tts_service = TTSService(config)
                self.use_voice_cloning = False
        else:
            self.tts_service = TTSService(config)
            logger.info("Using Edge TTS for speech synthesis")

        self.audio_service = AudioProcessingService()
        self.separator_service = None  # Lazy initialize if needed
        self.original_audio_path = None  # Track for async operations
        
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
            self.original_audio_path = original_audio  # Store for async access

            # Step 2: Setup voice cloning if enabled
            if self.use_voice_cloning and isinstance(self.tts_service, XTTSService):
                if progress_callback:
                    progress_callback("Extracting speaker voice samples...", 0.15)

                # Extract voice samples for each unique speaker
                self._setup_multi_speaker_cloning(
                    original_audio,
                    translated_segments
                )

            # Step 3: Select voice if not provided (for Edge TTS)
            if not voice:
                voice = self._select_voice(target_language)
                logger.info(f"Auto-selected voice: {voice}")

            # Step 4: Generate TTS for each segment
            if progress_callback:
                progress_callback("Generating speech for translated segments...", 0.2)
            
            tts_segments = self._generate_tts_segments(
                translated_segments,
                voice,
                progress_callback
            )
            
            # Step 4: Prepare background audio based on preservation mode
            if progress_callback:
                progress_callback("Preparing background audio...", 0.6)

            background_audio = self._prepare_background_audio(
                original_audio,
                tts_segments,
                ducking_level,
                progress_callback
            )

            # Step 5: Mix background with TTS segments
            if progress_callback:
                progress_callback("Mixing audio tracks...", 0.7)

            mixed_audio = self.audio_service.mix_audio_tracks(
                background_audio,
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

    def _prepare_background_audio(
        self,
        original_audio: str,
        tts_segments: List[Dict[str, Any]],
        ducking_level: float,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> str:
        """Prepare background audio based on preservation mode.

        Args:
            original_audio: Path to original audio file
            tts_segments: List of TTS segment dictionaries
            ducking_level: Volume reduction level in dB
            progress_callback: Optional progress callback

        Returns:
            Path to prepared background audio file
        """
        mode = self.config.background_preservation_mode

        if mode == "separator":
            # Mode 1: Use audio-separator to extract background only
            logger.info("Using audio-separator to extract background audio")

            if self.separator_service is None:
                try:
                    self.separator_service = AudioSeparatorService(
                        model_name=self.config.separation_model,
                        save_files=self.config.save_separated_audio
                    )
                except ImportError as e:
                    logger.warning(f"Audio separator not available: {e}. Falling back to ducking mode")
                    mode = "ducking"

            if mode == "separator":
                try:
                    # Use serial separation if configured
                    if self.config.use_serial_separation:
                        logger.info("Using serial separation with multiple models")
                        # Apply two models in series for better quality
                        models = [
                            "UVR-MDX-NET-Inst_HQ_3.onnx",
                            self.config.separation_model
                        ]
                        vocals_path, background_audio = self.separator_service.separate_audio_serial(
                            original_audio,
                            models=models
                        )
                    else:
                        # Single model separation
                        background_audio = self.separator_service.remove_vocals(
                            original_audio,
                            model_name=self.config.separation_model
                        )

                    logger.info(f"Background audio extracted: {background_audio}")

                    # Log saved files if configured
                    if self.config.save_separated_audio:
                        saved_files = self.separator_service.get_saved_files()
                        logger.info(f"Separated audio files saved: {saved_files}")

                    return background_audio
                except Exception as e:
                    logger.warning(f"Audio separation failed: {e}. Falling back to ducking mode")
                    mode = "ducking"

        # Mode 2 (default): Apply volume ducking to entire original audio
        logger.info("Using volume ducking for background preservation")

        speech_timings = [
            {'start_time': seg['start_time'], 'end_time': seg['end_time']}
            for seg in tts_segments
        ]

        # Reduce volume to 10% (-20dB) during speech
        ducking_level_db = -20.0  # 10% volume
        ducked_audio = self.audio_service.apply_volume_ducking(
            original_audio,
            speech_timings,
            ducking_level_db
        )

        return ducked_audio

    def _setup_multi_speaker_cloning(
        self,
        original_audio: str,
        segments: List[Segment]
    ):
        """Setup voice cloning for multiple speakers.

        Args:
            original_audio: Path to original audio file
            segments: List of segments with speaker IDs
        """
        if not segments:
            return

        # Find all segments for each unique speaker
        speaker_segments_map = {}
        for segment in segments:
            speaker_id = segment.speaker_id or "speaker_0"
            if speaker_id not in speaker_segments_map:
                speaker_segments_map[speaker_id] = []
            speaker_segments_map[speaker_id].append(segment)

        logger.info(f"Detected {len(speaker_segments_map)} unique speaker(s)")

        # Extract voice sample for each speaker
        for speaker_id, speaker_segs in speaker_segments_map.items():
            try:
                # Get minimum required duration from config
                min_duration = self.config.min_speaker_sample_duration

                # Try to find a single segment that's long enough
                suitable_segment = None
                for seg in speaker_segs:
                    seg_duration = seg.end_time - seg.start_time
                    if seg_duration >= min_duration:
                        suitable_segment = seg
                        break

                if suitable_segment:
                    # Use the suitable segment
                    sample_duration = min(10.0, suitable_segment.end_time - suitable_segment.start_time)
                    speaker_sample = self.tts_service.extract_speaker_sample(
                        original_audio,
                        suitable_segment.start_time,
                        sample_duration
                    )
                    logger.info(f"Voice sample extracted for speaker '{speaker_id}': {sample_duration:.2f}s from single segment")
                else:
                    # Combine multiple segments to reach minimum duration
                    logger.info(f"No single segment long enough for speaker '{speaker_id}', combining segments...")
                    speaker_sample = self._combine_speaker_segments(
                        original_audio,
                        speaker_segs,
                        min_duration
                    )
                    logger.info(f"Voice sample created for speaker '{speaker_id}' by combining {len(speaker_segs)} segments")

                self.tts_service.set_speaker_sample(speaker_id, speaker_sample)

            except Exception as e:
                logger.warning(f"Failed to extract sample for speaker {speaker_id}: {e}")
                # Continue with other speakers even if one fails

    def _combine_speaker_segments(
        self,
        original_audio: str,
        segments: List[Segment],
        target_duration: float
    ) -> str:
        """Combine multiple speaker segments to create a longer sample.

        Args:
            original_audio: Path to original audio file
            segments: List of segments from the same speaker
            target_duration: Target duration in seconds

        Returns:
            Path to combined audio sample
        """
        import subprocess
        import tempfile

        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda s: s.start_time)

        # Extract individual segments
        segment_files = []
        accumulated_duration = 0.0

        for seg in sorted_segments:
            if accumulated_duration >= target_duration:
                break

            seg_duration = min(seg.end_time - seg.start_time, target_duration - accumulated_duration)

            # Extract this segment
            temp_seg = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            cmd = [
                'ffmpeg', '-i', original_audio,
                '-ss', str(seg.start_time),
                '-t', str(seg_duration),
                '-ar', '22050',
                '-ac', '1',
                '-y', temp_seg
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                segment_files.append(temp_seg)
                accumulated_duration += seg_duration

        if not segment_files:
            raise RuntimeError("No valid segments found to combine")

        # Concatenate segments using FFmpeg
        output_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name

        if len(segment_files) == 1:
            # Just use the single file
            os.rename(segment_files[0], output_path)
        else:
            # Create concat file list
            concat_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            for seg_file in segment_files:
                concat_file.write(f"file '{seg_file}'\n")
            concat_file.close()

            # Concatenate
            cmd = [
                'ffmpeg', '-f', 'concat', '-safe', '0',
                '-i', concat_file.name,
                '-ar', '22050',
                '-ac', '1',
                '-y', output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            # Cleanup
            os.unlink(concat_file.name)
            for seg_file in segment_files:
                if os.path.exists(seg_file):
                    os.unlink(seg_file)

            if result.returncode != 0:
                raise RuntimeError(f"Failed to concatenate segments: {result.stderr}")

        # Track for cleanup
        if isinstance(self.tts_service, XTTSService):
            self.tts_service.temp_files.append(output_path)

        logger.info(f"Combined {len(segment_files)} segments into {accumulated_duration:.2f}s sample")
        return output_path

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
        skipped_count = 0

        for i, segment in enumerate(segments):
            if progress_callback:
                progress = 0.2 + (0.4 * (i / total))
                progress_callback(f"Generating speech {i+1}/{total}...", progress)

            # Skip empty or whitespace-only segments
            if not segment.text or not segment.text.strip():
                logger.warning(f"Skipping segment {i+1}/{total}: empty text")
                skipped_count += 1
                continue

            # Calculate target duration for this segment
            target_duration = segment.end_time - segment.start_time

            # Calculate speed adjustment to fit duration
            speed_factor = self.tts_service.calculate_speed_adjustment(
                segment.text,
                target_duration
            )

            # Use speaker-specific voice for XTTS, or default voice for Edge TTS
            segment_voice = voice
            if self.use_voice_cloning and isinstance(self.tts_service, XTTSService):
                segment_voice = segment.speaker_id or "speaker_0"

                # Check if speaker sample exists
                if segment_voice not in self.tts_service.speaker_samples:
                    logger.warning(f"No voice sample for speaker '{segment_voice}', using default")
                    segment_voice = "speaker_0"

                    # If even speaker_0 doesn't exist, skip this segment
                    if segment_voice not in self.tts_service.speaker_samples:
                        logger.error(f"No voice samples available, skipping segment {i+1}/{total}")
                        skipped_count += 1
                        continue

            try:
                # Generate TTS audio
                audio_file = self.tts_service.generate_speech(
                    segment,
                    segment_voice,
                    speed_factor
                )

                tts_segments.append({
                    'audio_file': audio_file,
                    'start_time': segment.start_time,
                    'end_time': segment.end_time
                })
            except Exception as e:
                logger.error(f"Failed to generate speech for segment {i+1}/{total}: {e}")
                skipped_count += 1
                continue

        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count}/{total} segments due to errors or empty text")

        if not tts_segments:
            raise RuntimeError("No valid TTS segments were generated. Check that segments have non-empty text and voice samples are set.")

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
        if self.separator_service is not None:
            self.separator_service.cleanup()

