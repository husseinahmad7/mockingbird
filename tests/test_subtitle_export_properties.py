"""Property-based tests for subtitle export completeness.

**Feature: video-translator, Property 20: Subtitle export completeness**
**Validates: Requirements 10.2**

Property 20: Subtitle export completeness
For any set of transcription segments, the exported subtitle files should
contain all segments with correct formatting and timing.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from typing import List
import tempfile
from pathlib import Path

from src.models.core import Segment
from src.services.subtitle_exporter import SubtitleExporter


# Strategy for generating valid timestamps
@st.composite
def timestamp_pair(draw):
    """Generate a valid pair of start and end timestamps."""
    start = draw(st.floats(min_value=0.0, max_value=3600.0))
    duration = draw(st.floats(min_value=0.1, max_value=30.0))
    end = start + duration
    return start, end


# Strategy for generating transcription segments
@st.composite
def transcription_segment(draw):
    """Generate a valid transcription segment."""
    start, end = draw(timestamp_pair())
    text = draw(st.text(min_size=1, max_size=500, alphabet=st.characters(blacklist_categories=('Cs',))))
    speaker_id = draw(st.one_of(st.none(), st.text(min_size=1, max_size=20)))
    
    return Segment(
        start_time=start,
        end_time=end,
        text=text,
        speaker_id=speaker_id
    )


class TestSubtitleExportProperties:
    """Property-based tests for subtitle export completeness."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = SubtitleExporter()
        self.temp_dir = tempfile.mkdtemp()
    
    @given(
        segments=st.lists(transcription_segment(), min_size=1, max_size=50)
    )
    @settings(max_examples=100, deadline=None)
    def test_srt_export_completeness_property(self, segments):
        """Property: SRT export should contain all segments.
        
        For any list of segments, the exported SRT file should contain
        exactly the same number of subtitle entries.
        """
        output_file = Path(self.temp_dir) / "test.srt"
        
        # Export to SRT
        success = self.exporter.export_srt(segments, str(output_file))
        
        assert success, "SRT export should succeed"
        assert output_file.exists(), "SRT file should be created"
        
        # Read and verify content
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Property: Should contain all segment indices
        for idx in range(1, len(segments) + 1):
            assert f"\n{idx}\n" in content, \
                f"SRT file should contain segment index {idx}"
        
        # Property: Should contain all segment texts
        for segment in segments:
            assert segment.text in content, \
                f"SRT file should contain segment text: {segment.text[:50]}"
    
    @given(
        segments=st.lists(transcription_segment(), min_size=1, max_size=50)
    )
    @settings(max_examples=100, deadline=None)
    def test_ass_export_completeness_property(self, segments):
        """Property: ASS export should contain all segments.
        
        For any list of segments, the exported ASS file should contain
        all dialogue lines.
        """
        output_file = Path(self.temp_dir) / "test.ass"
        
        # Export to ASS
        success = self.exporter.export_ass(segments, str(output_file))
        
        assert success, "ASS export should succeed"
        assert output_file.exists(), "ASS file should be created"
        
        # Read and verify content
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Property: Should contain ASS header sections
        assert "[Script Info]" in content, "ASS file should have Script Info section"
        assert "[V4+ Styles]" in content, "ASS file should have Styles section"
        assert "[Events]" in content, "ASS file should have Events section"
        
        # Property: Should contain all dialogue lines
        dialogue_count = content.count("Dialogue:")
        assert dialogue_count == len(segments), \
            f"ASS file should contain {len(segments)} dialogue lines, found {dialogue_count}"
    
    @given(
        segments=st.lists(transcription_segment(), min_size=1, max_size=50)
    )
    @settings(max_examples=100, deadline=None)
    def test_srt_timestamp_format_property(self, segments):
        """Property: SRT timestamps should be in correct format.
        
        For any segments, the SRT file should contain timestamps in the
        format HH:MM:SS,mmm --> HH:MM:SS,mmm
        """
        output_file = Path(self.temp_dir) / "test.srt"
        
        # Export to SRT
        success = self.exporter.export_srt(segments, str(output_file))
        assert success, "SRT export should succeed"
        
        # Read content
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Property: Should contain timestamp arrows
        arrow_count = content.count(" --> ")
        assert arrow_count == len(segments), \
            f"SRT file should contain {len(segments)} timestamp ranges"
        
        # Property: Timestamps should be in correct format (HH:MM:SS,mmm)
        import re
        timestamp_pattern = r'\d{2}:\d{2}:\d{2},\d{3}'
        timestamps = re.findall(timestamp_pattern, content)
        
        # Should have 2 timestamps per segment (start and end)
        assert len(timestamps) >= len(segments) * 2, \
            f"Should have at least {len(segments) * 2} timestamps"
    
    @given(
        segments=st.lists(transcription_segment(), min_size=1, max_size=50)
    )
    @settings(max_examples=100, deadline=None)
    def test_ass_timestamp_format_property(self, segments):
        """Property: ASS timestamps should be in correct format.
        
        For any segments, the ASS file should contain timestamps in the
        format H:MM:SS.cc
        """
        output_file = Path(self.temp_dir) / "test.ass"
        
        # Export to ASS
        success = self.exporter.export_ass(segments, str(output_file))
        assert success, "ASS export should succeed"
        
        # Read content
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Property: Dialogue lines should have correct format
        import re
        dialogue_pattern = r'Dialogue: \d+,\d+:\d{2}:\d{2}\.\d{2},\d+:\d{2}:\d{2}\.\d{2}'
        dialogues = re.findall(dialogue_pattern, content)
        
        assert len(dialogues) == len(segments), \
            f"Should have {len(segments)} properly formatted dialogue lines"
    
    @given(
        segments=st.lists(transcription_segment(), min_size=1, max_size=20)
    )
    @settings(max_examples=50, deadline=None)
    def test_export_both_formats_property(self, segments):
        """Property: Exporting both formats should create both files.
        
        For any segments, exporting both formats should create both
        SRT and ASS files with the same content.
        """
        base_path = Path(self.temp_dir) / "test_both"
        
        # Export both formats
        srt_success, ass_success = self.exporter.export_both_formats(
            segments,
            str(base_path)
        )
        
        assert srt_success, "SRT export should succeed"
        assert ass_success, "ASS export should succeed"
        
        # Property: Both files should exist
        srt_file = base_path.with_suffix('.srt')
        ass_file = base_path.with_suffix('.ass')
        
        assert srt_file.exists(), "SRT file should be created"
        assert ass_file.exists(), "ASS file should be created"
        
        # Property: Both files should contain all segments
        with open(srt_file, 'r', encoding='utf-8') as f:
            srt_content = f.read()
        
        with open(ass_file, 'r', encoding='utf-8') as f:
            ass_content = f.read()
        
        for segment in segments:
            assert segment.text in srt_content, \
                "SRT should contain all segment texts"
            # ASS may escape newlines, so check for text or escaped version
            assert segment.text in ass_content or segment.text.replace('\n', '\\N') in ass_content, \
                "ASS should contain all segment texts"
    
    @given(
        segments=st.lists(transcription_segment(), min_size=1, max_size=20),
        num_segments_to_check=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=50, deadline=None)
    def test_segment_order_preservation_property(self, segments, num_segments_to_check):
        """Property: Segment order should be preserved in export.
        
        For any list of segments, the exported file should maintain
        the same order.
        """
        if len(segments) < 2:
            return
        
        output_file = Path(self.temp_dir) / "test_order.srt"
        
        # Export to SRT
        success = self.exporter.export_srt(segments, str(output_file))
        assert success, "SRT export should succeed"
        
        # Read content
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Property: Segments should appear in order
        num_to_check = min(num_segments_to_check, len(segments) - 1)
        
        for i in range(num_to_check):
            text1 = segments[i].text
            text2 = segments[i + 1].text
            
            # Find positions in content
            pos1 = content.find(text1)
            pos2 = content.find(text2)
            
            if pos1 >= 0 and pos2 >= 0:
                assert pos1 < pos2, \
                    f"Segment {i} should appear before segment {i+1}"
    
    @given(
        segments=st.lists(transcription_segment(), min_size=1, max_size=20)
    )
    @settings(max_examples=50, deadline=None)
    def test_empty_segments_handling_property(self, segments):
        """Property: Export should handle segments gracefully.
        
        For any segments (including those with special characters),
        export should succeed without data loss.
        """
        output_file = Path(self.temp_dir) / "test_special.srt"
        
        # Export to SRT
        success = self.exporter.export_srt(segments, str(output_file))
        
        # Property: Export should always succeed for valid segments
        assert success, "Export should succeed for any valid segments"
        
        # Property: File should be created
        assert output_file.exists(), "Output file should be created"
        
        # Property: File should not be empty
        assert output_file.stat().st_size > 0, "Output file should not be empty"

