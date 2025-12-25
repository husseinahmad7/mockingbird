"""Property-based tests for UI data integrity.

**Feature: video-translator, Property 16: UI data integrity**
**Validates: Requirements 6.3, 9.1, 9.3**

Property 16: UI data integrity
For any segment data displayed in the UI, the data should remain consistent
and accurate throughout the editing process without corruption or loss.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from typing import List
import pandas as pd

from src.models.core import Segment
from src.ui.components.segment_editor import SegmentEditor


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


class TestUIDataIntegrityProperties:
    """Property-based tests for UI data integrity."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.editor = SegmentEditor()
    
    @given(
        segments=st.lists(transcription_segment(), min_size=1, max_size=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_segment_to_dataframe_preserves_data_property(self, segments):
        """Property: Converting segments to DataFrame should preserve all data.
        
        For any list of segments, converting to DataFrame and back should
        preserve all segment data without loss or corruption.
        """
        # Convert to DataFrame
        df = self.editor._segments_to_dataframe(segments, show_translation=False)
        
        # Property: DataFrame should have same number of rows as segments
        assert len(df) == len(segments), \
            "DataFrame should have same number of rows as segments"
        
        # Property: All required columns should be present
        required_columns = ['ID', 'Start', 'End', 'Duration', 'Text', 'Speaker']
        for col in required_columns:
            assert col in df.columns, \
                f"DataFrame should have '{col}' column"
        
        # Property: Data should match original segments
        for idx, seg in enumerate(segments):
            row = df.iloc[idx]
            
            # Check ID
            assert row['ID'] == idx, \
                "Segment ID should match index"
            
            # Check text
            assert row['Text'] == seg.text, \
                "Text should be preserved exactly"
            
            # Check speaker
            expected_speaker = seg.speaker_id if seg.speaker_id else 'Unknown'
            assert row['Speaker'] == expected_speaker, \
                "Speaker ID should be preserved"
    
    @given(
        segments=st.lists(transcription_segment(), min_size=1, max_size=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_dataframe_to_segment_roundtrip_property(self, segments):
        """Property: DataFrame to segment conversion should be reversible.
        
        For any list of segments, converting to DataFrame and back should
        produce equivalent segments.
        """
        # Convert to DataFrame
        df = self.editor._segments_to_dataframe(segments, show_translation=False)
        
        # Convert back to segments
        restored_segments = self.editor._dataframe_to_segments(df, segments)
        
        # Property: Should have same number of segments
        assert len(restored_segments) == len(segments), \
            "Restored segments should have same count as original"
        
        # Property: Each segment should match
        for orig, restored in zip(segments, restored_segments):
            # Check timestamps (allow small floating point differences)
            assert abs(restored.start_time - orig.start_time) < 0.001, \
                "Start time should be preserved"
            assert abs(restored.end_time - orig.end_time) < 0.001, \
                "End time should be preserved"
            
            # Check text
            assert restored.text == orig.text, \
                "Text should be preserved exactly"
            
            # Check speaker (accounting for None -> 'Unknown' conversion)
            if orig.speaker_id is not None:
                assert restored.speaker_id == orig.speaker_id, \
                    "Speaker ID should be preserved"
    
    @given(
        seconds=st.floats(min_value=0.0, max_value=86400.0)
    )
    @settings(max_examples=200, deadline=None)
    def test_timestamp_formatting_roundtrip_property(self, seconds):
        """Property: Timestamp formatting should be reversible.
        
        For any valid timestamp in seconds, formatting and parsing should
        produce the same value (within floating point precision).
        """
        # Format timestamp
        formatted = self.editor._format_timestamp(seconds)
        
        # Property: Formatted string should match expected pattern
        parts = formatted.split(':')
        assert len(parts) == 3, \
            "Formatted timestamp should have HH:MM:SS.mmm format"
        
        # Parse back
        parsed = self.editor._parse_timestamp(formatted)
        
        # Property: Parsed value should match original (within precision)
        assert abs(parsed - seconds) < 0.001, \
            f"Timestamp roundtrip should preserve value: {seconds} -> {formatted} -> {parsed}"
    
    @given(
        segments=st.lists(transcription_segment(), min_size=2, max_size=15)
    )
    @settings(max_examples=50, deadline=None)
    def test_segment_ordering_preservation_property(self, segments):
        """Property: Segment ordering should be preserved through conversions.
        
        For any ordered list of segments, the order should be maintained
        through DataFrame conversion and back.
        """
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda s: s.start_time)
        
        # Convert to DataFrame
        df = self.editor._segments_to_dataframe(sorted_segments, show_translation=False)
        
        # Property: DataFrame rows should be in same order
        for idx in range(len(df) - 1):
            current_start = self.editor._parse_timestamp(df.iloc[idx]['Start'])
            next_start = self.editor._parse_timestamp(df.iloc[idx + 1]['Start'])
            
            assert current_start <= next_start, \
                "Segment ordering should be preserved in DataFrame"
        
        # Convert back to segments
        restored_segments = self.editor._dataframe_to_segments(df, sorted_segments)
        
        # Property: Restored segments should maintain order
        for idx in range(len(restored_segments) - 1):
            assert restored_segments[idx].start_time <= restored_segments[idx + 1].start_time, \
                "Segment ordering should be preserved after roundtrip"
    
    @given(
        segments=st.lists(transcription_segment(), min_size=1, max_size=20),
        show_translation=st.booleans()
    )
    @settings(max_examples=100, deadline=None)
    def test_translation_column_handling_property(self, segments, show_translation):
        """Property: Translation column should be handled correctly.
        
        For any segments with or without translations, the DataFrame should
        correctly include or exclude the translation column.
        """
        # Add translations to some segments
        for seg in segments:
            if show_translation:
                seg.translation = f"Translation of: {seg.text[:50]}"
        
        # Convert to DataFrame
        df = self.editor._segments_to_dataframe(segments, show_translation=show_translation)
        
        # Property: Translation column presence should match parameter
        if show_translation:
            assert 'Translation' in df.columns, \
                "DataFrame should have Translation column when show_translation=True"
            
            # Check translation values
            for idx, seg in enumerate(segments):
                if hasattr(seg, 'translation'):
                    assert df.iloc[idx]['Translation'] == seg.translation, \
                        "Translation should be preserved"
        else:
            assert 'Translation' not in df.columns, \
                "DataFrame should not have Translation column when show_translation=False"

