"""Property-based tests for edit preservation accuracy.

**Feature: video-translator, Property 17: Edit preservation accuracy**
**Validates: Requirements 9.2**

Property 17: Edit preservation accuracy
For any edits made to segment text, the changes should be accurately preserved
while maintaining timestamp integrity.
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


class TestEditPreservationProperties:
    """Property-based tests for edit preservation accuracy."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.editor = SegmentEditor()
    
    @given(
        segments=st.lists(transcription_segment(), min_size=1, max_size=20),
        edit_indices=st.lists(st.integers(min_value=0, max_value=19), min_size=1, max_size=5),
        new_text=st.text(min_size=1, max_size=500, alphabet=st.characters(blacklist_categories=('Cs',)))
    )
    @settings(max_examples=100, deadline=None)
    def test_text_edit_preservation_property(self, segments, edit_indices, new_text):
        """Property: Text edits should be preserved while maintaining timestamps.
        
        For any text edits made to segments, the new text should be preserved
        exactly while timestamps remain unchanged.
        """
        if not segments:
            return
        
        # Filter valid indices
        valid_indices = [idx for idx in edit_indices if idx < len(segments)]
        if not valid_indices:
            return
        
        # Convert to DataFrame
        df = self.editor._segments_to_dataframe(segments, show_translation=False)
        
        # Make edits to DataFrame
        original_timestamps = {}
        for idx in valid_indices:
            # Store original timestamps
            original_timestamps[idx] = (df.iloc[idx]['Start'], df.iloc[idx]['End'])
            
            # Edit text
            df.at[idx, 'Text'] = new_text
        
        # Convert back to segments
        edited_segments = self.editor._dataframe_to_segments(df, segments)
        
        # Property: Edited text should be preserved
        for idx in valid_indices:
            assert edited_segments[idx].text == new_text, \
                f"Edited text should be preserved at index {idx}"
        
        # Property: Timestamps should remain unchanged
        for idx in valid_indices:
            orig_start, orig_end = original_timestamps[idx]
            
            # Parse timestamps
            parsed_start = self.editor._parse_timestamp(orig_start)
            parsed_end = self.editor._parse_timestamp(orig_end)
            
            assert abs(edited_segments[idx].start_time - parsed_start) < 0.001, \
                f"Start timestamp should be preserved at index {idx}"
            assert abs(edited_segments[idx].end_time - parsed_end) < 0.001, \
                f"End timestamp should be preserved at index {idx}"
        
        # Property: Unedited segments should remain unchanged
        for idx in range(len(segments)):
            if idx not in valid_indices:
                assert edited_segments[idx].text == segments[idx].text, \
                    f"Unedited text should remain unchanged at index {idx}"
    
    @given(
        segments=st.lists(transcription_segment(), min_size=1, max_size=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_timestamp_preservation_during_text_edit_property(self, segments):
        """Property: Timestamps should never change when only text is edited.
        
        For any text-only edits, all timestamps should remain exactly the same.
        """
        if not segments:
            return
        
        # Store original timestamps
        original_timestamps = [(seg.start_time, seg.end_time) for seg in segments]
        
        # Convert to DataFrame
        df = self.editor._segments_to_dataframe(segments, show_translation=False)
        
        # Edit all text fields
        for idx in range(len(df)):
            df.at[idx, 'Text'] = f"Edited text {idx}"
        
        # Convert back to segments
        edited_segments = self.editor._dataframe_to_segments(df, segments)
        
        # Property: All timestamps should be preserved
        for idx, (orig_start, orig_end) in enumerate(original_timestamps):
            assert abs(edited_segments[idx].start_time - orig_start) < 0.001, \
                f"Start timestamp should be preserved at index {idx}"
            assert abs(edited_segments[idx].end_time - orig_end) < 0.001, \
                f"End timestamp should be preserved at index {idx}"
    
    @given(
        segments=st.lists(transcription_segment(), min_size=1, max_size=20),
        translation_text=st.text(min_size=1, max_size=500, alphabet=st.characters(blacklist_categories=('Cs',)))
    )
    @settings(max_examples=100, deadline=None)
    def test_translation_edit_preservation_property(self, segments, translation_text):
        """Property: Translation edits should be preserved independently of original text.
        
        For any translation edits, the translation should be preserved while
        original text and timestamps remain unchanged.
        """
        if not segments:
            return
        
        # Add initial translations
        for seg in segments:
            seg.translation = f"Initial translation: {seg.text[:50]}"
        
        # Store original data
        original_texts = [seg.text for seg in segments]
        original_timestamps = [(seg.start_time, seg.end_time) for seg in segments]
        
        # Convert to DataFrame with translations
        df = self.editor._segments_to_dataframe(segments, show_translation=True)
        
        # Edit translations
        for idx in range(len(df)):
            df.at[idx, 'Translation'] = translation_text
        
        # Convert back to segments
        edited_segments = self.editor._dataframe_to_segments(df, segments)
        
        # Property: Translations should be updated
        for idx, seg in enumerate(edited_segments):
            assert seg.translation == translation_text, \
                f"Translation should be preserved at index {idx}"
        
        # Property: Original text should remain unchanged
        for idx, orig_text in enumerate(original_texts):
            assert edited_segments[idx].text == orig_text, \
                f"Original text should remain unchanged at index {idx}"
        
        # Property: Timestamps should remain unchanged
        for idx, (orig_start, orig_end) in enumerate(original_timestamps):
            assert abs(edited_segments[idx].start_time - orig_start) < 0.001, \
                f"Start timestamp should be preserved at index {idx}"
            assert abs(edited_segments[idx].end_time - orig_end) < 0.001, \
                f"End timestamp should be preserved at index {idx}"
    
    @given(
        segments=st.lists(transcription_segment(), min_size=2, max_size=20),
        num_edits=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=50, deadline=None)
    def test_multiple_edits_preservation_property(self, segments, num_edits):
        """Property: Multiple sequential edits should all be preserved.
        
        For any sequence of edits, all changes should be preserved accurately.
        """
        if len(segments) < 2:
            return
        
        # Convert to DataFrame
        df = self.editor._segments_to_dataframe(segments, show_translation=False)
        
        # Make multiple edits
        edit_log = {}
        for edit_num in range(num_edits):
            idx = edit_num % len(df)
            new_text = f"Edit {edit_num}: Modified text"
            df.at[idx, 'Text'] = new_text
            edit_log[idx] = new_text
        
        # Convert back to segments
        edited_segments = self.editor._dataframe_to_segments(df, segments)
        
        # Property: All edits should be preserved
        for idx, expected_text in edit_log.items():
            assert edited_segments[idx].text == expected_text, \
                f"Edit {idx} should be preserved"
        
        # Property: Segments should still be valid
        for seg in edited_segments:
            assert seg.end_time > seg.start_time, \
                "Segment should have valid time range"
            assert len(seg.text) > 0, \
                "Segment should have non-empty text"

