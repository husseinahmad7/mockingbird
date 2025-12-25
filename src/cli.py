"""Command-line interface for Video Translator System.

This module provides a CLI for batch processing and automation.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, List
import json

from src.services.file_handler import FileHandler
from src.services.error_handler import ErrorHandler
from src.services.asr_service import ASRService
from src.services.translation_service import TranslationService
from src.services.subtitle_exporter import SubtitleExporter
from src.services.dubbing_service import DubbingService
from src.models.core import ProcessingConfig, Segment


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoTranslatorCLI:
    """Command-line interface for video translation."""
    
    def __init__(self):
        """Initialize CLI components."""
        self.error_handler = ErrorHandler()
        self.file_handler = FileHandler()
        self.subtitle_exporter = SubtitleExporter()
    
    def process_video(
        self,
        input_path: str,
        output_dir: str,
        source_lang: str = "auto",
        target_langs: List[str] = None,
        whisper_model: str = "small",
        enable_speaker_detection: bool = False,
        subtitle_only: bool = False,
        formats: List[str] = None
    ) -> bool:
        """Process a video file for translation.
        
        Args:
            input_path: Path to input video/audio file
            output_dir: Directory for output files
            source_lang: Source language code (auto for detection)
            target_langs: List of target language codes
            whisper_model: Whisper model size
            enable_speaker_detection: Enable speaker detection
            subtitle_only: Only generate subtitles, no dubbing
            formats: Subtitle formats to export (srt, ass)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Processing video: {input_path}")
            
            # Validate input file
            if not Path(input_path).exists():
                logger.error(f"Input file not found: {input_path}")
                return False
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create processing config
            config = ProcessingConfig(
                whisper_model_size=whisper_model,
                enable_speaker_detection=enable_speaker_detection
            )
            
            # Step 1: Transcription
            logger.info("Step 1/3: Transcribing audio...")
            asr_service = ASRService(config)
            
            # Detect language if auto
            if source_lang == "auto":
                detected_lang = asr_service.detect_language(input_path)
                logger.info(f"Detected language: {detected_lang}")
                config.source_language = detected_lang
            
            # Transcribe
            segments = asr_service.transcribe(input_path, config.source_language)
            logger.info(f"Transcription complete: {len(segments)} segments")
            
            # Export original transcription
            base_name = Path(input_path).stem
            original_srt = output_path / f"{base_name}_original.srt"
            self.subtitle_exporter.export_srt(segments, str(original_srt))
            logger.info(f"Original subtitles saved: {original_srt}")
            
            # Step 2: Translation
            if target_langs and not subtitle_only:
                logger.info("Step 2/3: Translating segments...")
                translation_service = TranslationService(config)

                for target_lang in target_langs:
                    logger.info(f"Translating to {target_lang}...")

                    # Translate segments using the service method
                    translated_segments = translation_service.translate_segments(
                        segments,
                        target_lang
                    )
                    
                    # Export translated subtitles
                    if not formats:
                        formats = ["srt", "ass"]

                    for fmt in formats:
                        if fmt == "srt":
                            output_file = output_path / f"{base_name}_{target_lang}.srt"
                            self.subtitle_exporter.export_srt(translated_segments, str(output_file))
                            logger.info(f"Translated SRT saved: {output_file}")
                        elif fmt == "ass":
                            output_file = output_path / f"{base_name}_{target_lang}.ass"
                            self.subtitle_exporter.export_ass(translated_segments, str(output_file))
                            logger.info(f"Translated ASS saved: {output_file}")

                    # Step 3: Create dubbed video (if not subtitle-only mode)
                    if not subtitle_only:
                        logger.info(f"Step 3/3: Creating dubbed video for {target_lang}...")

                        # Update config for target language
                        dub_config = ProcessingConfig(
                            whisper_model_size=whisper_model,
                            enable_speaker_detection=enable_speaker_detection,
                            target_language=target_lang
                        )

                        dubbing_service = DubbingService(dub_config)

                        # Create dubbed video
                        dubbed_video_path = output_path / f"{base_name}_dubbed_{target_lang}.mp4"

                        try:
                            dubbing_service.create_dubbed_video(
                                video_path=validated_path,
                                translated_segments=translated_segments,
                                output_path=str(dubbed_video_path),
                                progress_callback=lambda msg, prog: logger.info(f"{msg} ({prog*100:.0f}%)")
                            )
                            logger.info(f"Dubbed video saved: {dubbed_video_path}")
                        except Exception as e:
                            logger.error(f"Dubbing failed for {target_lang}: {e}")
                            logger.info("Continuing with subtitle-only output...")
                        finally:
                            dubbing_service.cleanup()

            logger.info("Processing complete!")
            return True
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            self.error_handler.log_error(e)
            return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Video Translator - Translate and dub videos automatically",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe and translate to Spanish
  python -m src.cli input.mp4 -o output/ -t es

  # Translate to multiple languages
  python -m src.cli input.mp4 -o output/ -t es fr de ar

  # Use larger model for better accuracy
  python -m src.cli input.mp4 -o output/ -t es -m large-v3

  # Generate only subtitles (no dubbing)
  python -m src.cli input.mp4 -o output/ -t es --subtitle-only

  # Export in specific formats
  python -m src.cli input.mp4 -o output/ -t es -f srt ass

  # Enable speaker detection
  python -m src.cli input.mp4 -o output/ -t es --speaker-detection
        """
    )

    # Required arguments
    parser.add_argument(
        "input",
        help="Input video or audio file path"
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory for generated files"
    )

    # Language options
    parser.add_argument(
        "-s", "--source-lang",
        default="auto",
        help="Source language code (default: auto-detect)"
    )

    parser.add_argument(
        "-t", "--target-langs",
        nargs="+",
        default=["en"],
        help="Target language codes (space-separated)"
    )

    # Model options
    parser.add_argument(
        "-m", "--model",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        default="small",
        help="Whisper model size (default: small)"
    )

    # Feature flags
    parser.add_argument(
        "--speaker-detection",
        action="store_true",
        help="Enable speaker detection"
    )

    parser.add_argument(
        "--subtitle-only",
        action="store_true",
        help="Generate only subtitles, skip dubbing"
    )

    # Export options
    parser.add_argument(
        "-f", "--formats",
        nargs="+",
        choices=["srt", "ass"],
        default=["srt"],
        help="Subtitle formats to export (default: srt)"
    )

    # Verbosity
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create CLI instance
    cli = VideoTranslatorCLI()

    # Process video
    success = cli.process_video(
        input_path=args.input,
        output_dir=args.output,
        source_lang=args.source_lang,
        target_langs=args.target_langs,
        whisper_model=args.model,
        enable_speaker_detection=args.speaker_detection,
        subtitle_only=args.subtitle_only,
        formats=args.formats
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


