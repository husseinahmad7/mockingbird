"""Property-based tests for timing validation effectiveness.

**Feature: video-translator, Property 18: Timing validation effectiveness**
**Validates: Requirements 9.4**

Property 18: Timing validation effectiveness
For any segment timing modifications, the system should detect and warn about
potential synchronization issues.
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


class TestTimingValidationProperties:
    """Property-based tests for timing validation effectiveness."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.editor = SegmentEditor()
    
    @given(
        segments=st.lists(transcription_segment(), min_size=1, max_size=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_invalid_time_range_detection_property(self, segments):
        """Property: System should detect when end time is before or equal to start time.
        
        For any segment where end_time <= start_time, validation should
        detect this as an error.
        """
        if not segments:
            return
        
        # Convert to DataFrame
        df = self.editor._segments_to_dataframe(segments, show_translation=False)
        
        # Create invalid time range (end before start)
        if len(df) > 0:
            # Make first segment invalid
            df.at[0, 'End'] = df.at[0, 'Start']  # End equals start
            
            # Validate
            issues = self.editor._validate_segments(df)
            
            # Property: Should detect the invalid time range
            assert len(issues) > 0, \
                "Validation should detect invalid time range"
            
            assert any("End time must be after start time" in issue for issue in issues), \
                "Validation should specifically mention end time issue"
    
    @given(
        segments=st.lists(transcription_segment(), min_size=1, max_size=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_very_short_segment_detection_property(self, segments):
        """Property: System should detect very short segments (< 0.1s).
        
        For any segment with duration < 0.1s, validation should warn about it.
        """
        if not segments:
            return
        
        # Convert to DataFrame
        df = self.editor._segments_to_dataframe(segments, show_translation=False)
        
        # Create very short segment
        if len(df) > 0:
            start_time = self.editor._parse_timestamp(df.at[0, 'Start'])
            # Set end time to be 0.05s after start (very short)
            df.at[0, 'End'] = self.editor._format_timestamp(start_time + 0.05)
            
            # Validate
            issues = self.editor._validate_segments(df)
            
            # Property: Should detect the very short duration
            assert len(issues) > 0, \
                "Validation should detect very short segments"
            
            assert any("Duration too short" in issue for issue in issues), \
                "Validation should specifically mention short duration"
    
    @given(
        segments=st.lists(transcription_segment(), min_size=1, max_size=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_very_long_segment_warning_property(self, segments):
        """Property: System should warn about very long segments (> 30s).
        
        For any segment with duration > 30s, validation should suggest splitting.
        """
        if not segments:
            return
        
        # Convert to DataFrame
        df = self.editor._segments_to_dataframe(segments, show_translation=False)
        
        # Create very long segment
        if len(df) > 0:
            start_time = self.editor._parse_timestamp(df.at[0, 'Start'])
            # Set end time to be 35s after start (very long)
            df.at[0, 'End'] = self.editor._format_timestamp(start_time + 35.0)
            
            # Validate
            issues = self.editor._validate_segments(df)
            
            # Property: Should warn about the long duration
            assert len(issues) > 0, \
                "Validation should warn about very long segments"
            
            assert any("Duration very long" in issue or "consider splitting" in issue for issue in issues), \
                "Validation should suggest splitting long segments"
    
    @given(
        segments=st.lists(transcription_segment(), min_size=2, max_size=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_overlapping_segments_detection_property(self, segments):
        """Property: System should detect overlapping segments.
        
        For any two consecutive segments where the first ends after the second starts,
        validation should detect the overlap.
        """
        if len(segments) < 2:
            return
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda s: s.start_time)
        
        # Convert to DataFrame
        df = self.editor._segments_to_dataframe(sorted_segments, show_translation=False)
        
        # Create overlap between first two segments
        if len(df) >= 2:
            # Make first segment end after second segment starts
            second_start = self.editor._parse_timestamp(df.at[1, 'Start'])
            df.at[0, 'End'] = self.editor._format_timestamp(second_start + 1.0)
            
            # Validate
            issues = self.editor._validate_segments(df)
            
            # Property: Should detect the overlap
            assert len(issues) > 0, \
                "Validation should detect overlapping segments"
            
            assert any("Overlaps with next segment" in issue for issue in issues), \
                "Validation should specifically mention overlap"
    
    @given(
        segments=st.lists(transcription_segment(), min_size=1, max_size=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_empty_text_detection_property(self, segments):
        """Property: System should detect segments with empty text.
        
        For any segment with empty or whitespace-only text, validation should
        detect this as an error.
        """
        if not segments:
            return
        
        # Convert to DataFrame
        df = self.editor._segments_to_dataframe(segments, show_translation=False)
        
        # Create empty text segment
        if len(df) > 0:
            df.at[0, 'Text'] = "   "  # Whitespace only
            
            # Validate
            issues = self.editor._validate_segments(df)
            
            # Property: Should detect empty text
            assert len(issues) > 0, \
                "Validation should detect empty text"
            
            assert any("Text is empty" in issue for issue in issues), \
                "Validation should specifically mention empty text"
    
    @given(
        segments=st.lists(transcription_segment(), min_size=1, max_size=20),
        num_issues=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=50, deadline=None)
    def test_multiple_issues_detection_property(self, segments, num_issues):
        """Property: System should detect multiple validation issues simultaneously.
        
        For any DataFrame with multiple validation issues, all issues should
        be detected and reported.
        """
        if not segments:
            return
        
        # Convert to DataFrame
        df = self.editor._segments_to_dataframe(segments, show_translation=False)
        
        # Introduce multiple issues
        issues_created = 0
        
        if len(df) > 0 and issues_created < num_issues:
            # Issue 1: Empty text
            df.at[0, 'Text'] = ""
            issues_created += 1
        
        if len(df) > 1 and issues_created < num_issues:
            # Issue 2: Invalid time range
            df.at[1, 'End'] = df.at[1, 'Start']
            issues_created += 1
        
        if len(df) > 2 and issues_created < num_issues:
            # Issue 3: Very short segment
            start_time = self.editor._parse_timestamp(df.at[2, 'Start'])
            df.at[2, 'End'] = self.editor._format_timestamp(start_time + 0.05)
            issues_created += 1
        
        if len(df) > 3 and issues_created < num_issues:
            # Issue 4: Very long segment
            start_time = self.editor._parse_timestamp(df.at[3, 'Start'])
            df.at[3, 'End'] = self.editor._format_timestamp(start_time + 35.0)
            issues_created += 1
        
        # Validate
        detected_issues = self.editor._validate_segments(df)
        
        # Property: Should detect at least as many issues as we created
        assert len(detected_issues) >= issues_created, \
            f"Should detect at least {issues_created} issues, found {len(detected_issues)}"
    
    @given(
        segments=st.lists(transcription_segment(), min_size=1, max_size=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_valid_segments_pass_validation_property(self, segments):
        """Property: Valid segments should pass validation without issues.
        
        For any properly formatted segments with valid timing, validation
        should not report any issues.
        """
        if not segments:
            return
        
        # Ensure all segments are valid
        valid_segments = []
        for seg in segments:
            # Ensure valid duration
            if seg.end_time - seg.start_time >= 0.1 and seg.end_time - seg.start_time <= 30.0:
                # Ensure non-empty text
                if seg.text.strip():
                    valid_segments.append(seg)
        
        if not valid_segments:
            return
        
        # Sort to avoid overlaps
        valid_segments.sort(key=lambda s: s.start_time)
        
        # Ensure no overlaps
        non_overlapping = [valid_segments[0]]
        for seg in valid_segments[1:]:
            if seg.start_time >= non_overlapping[-1].end_time:
                non_overlapping.append(seg)
        
        if not non_overlapping:
            return
        
        # Convert to DataFrame
        df = self.editor._segments_to_dataframe(non_overlapping, show_translation=False)
        
        # Validate
        issues = self.editor._validate_segments(df)
        
        # Property: Valid segments should have no issues
        assert len(issues) == 0, \
            f"Valid segments should pass validation, but found issues: {issues}"

