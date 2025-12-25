"""Subtitle export service for Video Translator System.

This module provides subtitle export functionality in multiple formats (SRT, ASS).
Requirements: 10.2
"""

from typing import List, Optional
from pathlib import Path
from datetime import timedelta

from src.models.core import Segment
from src.services.error_handler import ErrorHandler, ErrorSeverity


class SubtitleExporter:
    """Service for exporting subtitles in various formats."""
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        """Initialize the subtitle exporter.
        
        Args:
            error_handler: Optional error handler for logging
        """
        self.error_handler = error_handler or ErrorHandler()
    
    def export_srt(
        self,
        segments: List[Segment],
        output_path: str,
        use_translation: bool = False
    ) -> bool:
        """Export subtitles in SRT format.
        
        Args:
            segments: List of transcription segments
            output_path: Path to output SRT file
            use_translation: Whether to use translation instead of original text
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for idx, segment in enumerate(segments, start=1):
                    # Subtitle index
                    f.write(f"{idx}\n")
                    
                    # Timestamp range
                    start_time = self._format_srt_timestamp(segment.start_time)
                    end_time = self._format_srt_timestamp(segment.end_time)
                    f.write(f"{start_time} --> {end_time}\n")
                    
                    # Text content
                    text = segment.translation if use_translation and hasattr(segment, 'translation') else segment.text
                    f.write(f"{text}\n")
                    
                    # Blank line separator
                    f.write("\n")
            
            self.error_handler.log_info(
                f"Successfully exported SRT subtitles to {output_path}",
                context={'num_segments': len(segments), 'use_translation': use_translation}
            )
            return True
            
        except Exception as e:
            self.error_handler.log_error(
                e,
                severity=ErrorSeverity.ERROR,
                context={'output_path': output_path, 'format': 'SRT'},
                recovery_suggestion="Check file permissions and disk space"
            )
            return False
    
    def export_ass(
        self,
        segments: List[Segment],
        output_path: str,
        use_translation: bool = False,
        style_config: Optional[dict] = None
    ) -> bool:
        """Export subtitles in ASS (Advanced SubStation Alpha) format.
        
        Args:
            segments: List of transcription segments
            output_path: Path to output ASS file
            use_translation: Whether to use translation instead of original text
            style_config: Optional style configuration
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # Default style configuration
            default_style = {
                'font_name': 'Arial',
                'font_size': 20,
                'primary_color': '&H00FFFFFF',  # White
                'secondary_color': '&H000000FF',  # Red
                'outline_color': '&H00000000',  # Black
                'back_color': '&H80000000',  # Semi-transparent black
                'bold': 0,
                'italic': 0,
                'border_style': 1,
                'outline': 2,
                'shadow': 0,
                'alignment': 2,  # Bottom center
                'margin_l': 10,
                'margin_r': 10,
                'margin_v': 10
            }
            
            # Merge with provided config
            style = {**default_style, **(style_config or {})}
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write ASS header
                f.write("[Script Info]\n")
                f.write("Title: Video Translation Subtitles\n")
                f.write("ScriptType: v4.00+\n")
                f.write("WrapStyle: 0\n")
                f.write("PlayResX: 1920\n")
                f.write("PlayResY: 1080\n")
                f.write("\n")
                
                # Write styles
                f.write("[V4+ Styles]\n")
                f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, ")
                f.write("Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, ")
                f.write("Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
                
                f.write(f"Style: Default,{style['font_name']},{style['font_size']},{style['primary_color']},")
                f.write(f"{style['secondary_color']},{style['outline_color']},{style['back_color']},")
                f.write(f"{style['bold']},{style['italic']},0,0,100,100,0,0,{style['border_style']},")
                f.write(f"{style['outline']},{style['shadow']},{style['alignment']},{style['margin_l']},")
                f.write(f"{style['margin_r']},{style['margin_v']},1\n")
                f.write("\n")
                
                # Write events
                f.write("[Events]\n")
                f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
                
                for segment in segments:
                    start_time = self._format_ass_timestamp(segment.start_time)
                    end_time = self._format_ass_timestamp(segment.end_time)
                    text = segment.translation if use_translation and hasattr(segment, 'translation') else segment.text
                    
                    # Escape special characters
                    text = text.replace('\n', '\\N')
                    
                    f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n")
            
            self.error_handler.log_info(
                f"Successfully exported ASS subtitles to {output_path}",
                context={'num_segments': len(segments), 'use_translation': use_translation}
            )
            return True
            
        except Exception as e:
            self.error_handler.log_error(
                e,
                severity=ErrorSeverity.ERROR,
                context={'output_path': output_path, 'format': 'ASS'},
                recovery_suggestion="Check file permissions and disk space"
            )
            return False

    def _format_srt_timestamp(self, seconds: float) -> str:
        """Format timestamp for SRT format (HH:MM:SS,mmm).

        Args:
            seconds: Time in seconds

        Returns:
            Formatted timestamp string
        """
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        millis = int((td.total_seconds() % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _format_ass_timestamp(self, seconds: float) -> str:
        """Format timestamp for ASS format (H:MM:SS.cc).

        Args:
            seconds: Time in seconds

        Returns:
            Formatted timestamp string
        """
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        centisecs = int((td.total_seconds() % 1) * 100)

        return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"

    def export_both_formats(
        self,
        segments: List[Segment],
        base_output_path: str,
        use_translation: bool = False
    ) -> tuple[bool, bool]:
        """Export subtitles in both SRT and ASS formats.

        Args:
            segments: List of transcription segments
            base_output_path: Base path for output files (without extension)
            use_translation: Whether to use translation instead of original text

        Returns:
            Tuple of (srt_success, ass_success)
        """
        base_path = Path(base_output_path)

        srt_path = base_path.with_suffix('.srt')
        ass_path = base_path.with_suffix('.ass')

        srt_success = self.export_srt(segments, str(srt_path), use_translation)
        ass_success = self.export_ass(segments, str(ass_path), use_translation)

        return srt_success, ass_success

    def export_multi_language(
        self,
        segments: List[Segment],
        output_dir: str,
        base_filename: str,
        languages: List[str]
    ) -> dict[str, tuple[bool, bool]]:
        """Export subtitles for multiple languages.

        Args:
            segments: List of transcription segments with translations
            output_dir: Output directory
            base_filename: Base filename (without extension)
            languages: List of language codes to export

        Returns:
            Dictionary mapping language codes to (srt_success, ass_success) tuples
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # Export original language
        if 'original' in languages:
            original_path = output_path / f"{base_filename}_original"
            results['original'] = self.export_both_formats(
                segments,
                str(original_path),
                use_translation=False
            )

        # Export translated versions
        for lang in languages:
            if lang != 'original':
                lang_path = output_path / f"{base_filename}_{lang}"
                results[lang] = self.export_both_formats(
                    segments,
                    str(lang_path),
                    use_translation=True
                )

        return results

