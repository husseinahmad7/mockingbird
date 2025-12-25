"""Processing pipeline for the Streamlit UI."""

import logging
from pathlib import Path
from typing import List, Optional, Callable
import streamlit as st

from src.models.core import ProcessingConfig, Segment
from src.services.asr_service import ASRService
from src.services.translation_service import TranslationService
from src.services.subtitle_exporter import SubtitleExporter

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handles the video processing pipeline for the UI."""
    
    def __init__(self, config: ProcessingConfig):
        """Initialize the video processor with configuration."""
        self.config = config
        self.asr_service = None
        self.translation_service = None
        self.subtitle_exporter = None
    
    def transcribe_video(
        self,
        video_path: str,
        source_language: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[Segment]:
        """Transcribe video to segments.
        
        Args:
            video_path: Path to the video file
            source_language: Optional source language code
            progress_callback: Optional callback for progress updates (progress, message)
            
        Returns:
            List of transcribed segments
        """
        try:
            if progress_callback:
                progress_callback(0.1, "Initializing ASR service...")
            
            # Initialize ASR service
            if not self.asr_service:
                self.asr_service = ASRService(self.config)
            
            if progress_callback:
                progress_callback(0.2, "Loading Whisper model...")
            
            # Load model
            self.asr_service.load_model(self.config.whisper_model_size)
            
            # Detect language if needed
            if not source_language or source_language == "auto":
                if progress_callback:
                    progress_callback(0.3, "Detecting language...")
                source_language = self.asr_service.detect_language(video_path)
                logger.info(f"Detected language: {source_language}")
            
            if progress_callback:
                progress_callback(0.4, f"Transcribing audio ({source_language})...")
            
            # Transcribe
            segments = self.asr_service.transcribe(video_path, source_language)
            
            if progress_callback:
                progress_callback(1.0, f"Transcription complete! Found {len(segments)} segments")
            
            return segments
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def translate_segments(
        self,
        segments: List[Segment],
        target_language: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[Segment]:
        """Translate segments to target language.
        
        Args:
            segments: List of segments to translate
            target_language: Target language code
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of translated segments
        """
        try:
            if progress_callback:
                progress_callback(0.1, "Initializing translation service...")
            
            # Initialize translation service
            if not self.translation_service:
                self.translation_service = TranslationService(self.config)
            
            if progress_callback:
                progress_callback(0.3, f"Translating to {target_language}...")
            
            # Translate
            translated_segments = self.translation_service.translate_segments(
                segments, target_language
            )
            
            if progress_callback:
                progress_callback(1.0, f"Translation complete! Translated {len(translated_segments)} segments")
            
            return translated_segments
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise
    
    def export_subtitles(
        self,
        segments: List[Segment],
        output_path: str,
        format: str = "srt",
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> str:
        """Export segments as subtitle file.
        
        Args:
            segments: List of segments to export
            output_path: Output file path
            format: Subtitle format ('srt' or 'ass')
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to the exported subtitle file
        """
        try:
            if progress_callback:
                progress_callback(0.3, f"Exporting {format.upper()} subtitles...")
            
            # Initialize subtitle exporter
            if not self.subtitle_exporter:
                self.subtitle_exporter = SubtitleExporter()
            
            # Export
            if format.lower() == "srt":
                subtitle_path = self.subtitle_exporter.export_srt(segments, output_path)
            elif format.lower() == "ass":
                subtitle_path = self.subtitle_exporter.export_ass(segments, output_path)
            else:
                raise ValueError(f"Unsupported subtitle format: {format}")
            
            if progress_callback:
                progress_callback(1.0, f"Subtitles exported to {subtitle_path}")
            
            return subtitle_path
            
        except Exception as e:
            logger.error(f"Subtitle export failed: {e}")
            raise

