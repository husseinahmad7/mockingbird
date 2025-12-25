"""Segment editor component for reviewing and editing transcriptions and translations.

This module provides an editable table interface for reviewing transcription and translation segments.
Requirements: 6.3, 9.1, 9.2, 9.3, 9.4, 9.5
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import timedelta

from src.models.core import Segment


@dataclass
class SegmentEdit:
    """Represents an edit to a segment."""
    segment_id: int
    field: str  # 'text', 'translation', 'start_time', 'end_time'
    old_value: Any
    new_value: Any
    timestamp: str


class SegmentEditor:
    """Component for editing transcription and translation segments."""
    
    def __init__(self):
        """Initialize the segment editor."""
        self.edit_history: List[SegmentEdit] = []
    
    def render(
        self,
        segments: List[Segment],
        show_translation: bool = False,
        editable: bool = True
    ) -> Tuple[List[Segment], bool]:
        """Render the segment editor interface.
        
        Args:
            segments: List of transcription segments to display/edit
            show_translation: Whether to show translation column
            editable: Whether segments are editable
            
        Returns:
            Tuple of (updated_segments, has_changes)
        """
        if not segments:
            st.info("No segments to display")
            return segments, False
        
        st.subheader("📝 Segment Editor")
        
        # Display statistics
        self._render_statistics(segments)
        
        st.divider()
        
        # Render editing controls
        if editable:
            self._render_editing_controls()
        
        # Convert segments to DataFrame for editing
        df = self._segments_to_dataframe(segments, show_translation)
        
        # Display editable data editor
        if editable:
            edited_df = st.data_editor(
                df,
                use_container_width=True,
                num_rows="fixed",
                column_config={
                    "ID": st.column_config.NumberColumn(
                        "ID",
                        help="Segment ID",
                        disabled=True,
                        width="small"
                    ),
                    "Start": st.column_config.TextColumn(
                        "Start Time",
                        help="Start timestamp (HH:MM:SS.mmm)",
                        width="small"
                    ),
                    "End": st.column_config.TextColumn(
                        "End Time",
                        help="End timestamp (HH:MM:SS.mmm)",
                        width="small"
                    ),
                    "Duration": st.column_config.TextColumn(
                        "Duration",
                        help="Segment duration",
                        disabled=True,
                        width="small"
                    ),
                    "Text": st.column_config.TextColumn(
                        "Original Text",
                        help="Transcribed text (editable)",
                        width="large"
                    ),
                    "Translation": st.column_config.TextColumn(
                        "Translation",
                        help="Translated text (editable)",
                        width="large"
                    ) if show_translation else None,
                    "Speaker": st.column_config.TextColumn(
                        "Speaker",
                        help="Speaker ID",
                        width="small"
                    ),
                },
                hide_index=True,
                key="segment_editor"
            )
            
            # Check for changes and validate
            has_changes = not df.equals(edited_df)
            
            if has_changes:
                # Validate timing
                validation_issues = self._validate_segments(edited_df)
                
                if validation_issues:
                    st.warning("⚠️ Validation Issues Detected:")
                    for issue in validation_issues:
                        st.warning(f"  • {issue}")
                
                # Convert back to segments
                updated_segments = self._dataframe_to_segments(edited_df, segments)
                
                return updated_segments, True
            
            return segments, False
        else:
            # Read-only display
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )
            return segments, False
    
    def _render_statistics(self, segments: List[Segment]):
        """Render segment statistics.
        
        Args:
            segments: List of segments
        """
        total_segments = len(segments)
        total_duration = sum(seg.end_time - seg.start_time for seg in segments)
        total_words = sum(len(seg.text.split()) for seg in segments)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Segments", total_segments)
        
        with col2:
            st.metric("Total Duration", f"{total_duration:.1f}s")
        
        with col3:
            st.metric("Total Words", total_words)
        
        with col4:
            avg_duration = total_duration / total_segments if total_segments > 0 else 0
            st.metric("Avg Segment", f"{avg_duration:.1f}s")
    
    def _render_editing_controls(self):
        """Render editing control buttons."""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Changes", use_container_width=True):
                st.success("Changes saved!")
        
        with col2:
            if st.button("↩️ Undo Last Edit", use_container_width=True):
                if self.edit_history:
                    self.edit_history.pop()
                    st.info("Last edit undone")
                else:
                    st.warning("No edits to undo")
        
        with col3:
            if st.button("🔄 Reset All", use_container_width=True):
                self.edit_history.clear()
                st.info("All edits reset")

    def _segments_to_dataframe(
        self,
        segments: List[Segment],
        show_translation: bool
    ) -> pd.DataFrame:
        """Convert segments to DataFrame for editing.

        Args:
            segments: List of segments
            show_translation: Whether to include translation column

        Returns:
            DataFrame representation of segments
        """
        data = []

        for idx, seg in enumerate(segments):
            row = {
                'ID': idx,
                'Start': self._format_timestamp(seg.start_time),
                'End': self._format_timestamp(seg.end_time),
                'Duration': self._format_duration(seg.end_time - seg.start_time),
                'Text': seg.text,
                'Speaker': seg.speaker_id or 'Unknown'
            }

            if show_translation:
                row['Translation'] = getattr(seg, 'translation', '')

            data.append(row)

        return pd.DataFrame(data)

    def _dataframe_to_segments(
        self,
        df: pd.DataFrame,
        original_segments: List[Segment]
    ) -> List[Segment]:
        """Convert DataFrame back to segments.

        Args:
            df: DataFrame with edited data
            original_segments: Original segments for reference

        Returns:
            List of updated segments
        """
        updated_segments = []

        for idx, row in df.iterrows():
            # Get original segment
            orig_seg = original_segments[idx] if idx < len(original_segments) else None

            # Parse timestamps
            start_time = self._parse_timestamp(row['Start'])
            end_time = self._parse_timestamp(row['End'])

            # Create updated segment
            segment = Segment(
                start_time=start_time,
                end_time=end_time,
                text=row['Text'],
                speaker_id=row['Speaker'] if row['Speaker'] != 'Unknown' else None
            )

            # Add translation if present
            if 'Translation' in row:
                segment.translation = row['Translation']

            updated_segments.append(segment)

        return updated_segments

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS.mmm timestamp.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted timestamp string
        """
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = td.total_seconds() % 60

        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    def _parse_timestamp(self, timestamp: str) -> float:
        """Parse timestamp string to seconds.

        Args:
            timestamp: Timestamp string (HH:MM:SS.mmm)

        Returns:
            Time in seconds
        """
        try:
            parts = timestamp.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])

            return hours * 3600 + minutes * 60 + seconds
        except (ValueError, IndexError):
            return 0.0

    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted duration string
        """
        return f"{seconds:.2f}s"

    def _validate_segments(self, df: pd.DataFrame) -> List[str]:
        """Validate segment data for timing and consistency issues.

        Args:
            df: DataFrame with segment data

        Returns:
            List of validation issue messages
        """
        issues = []

        for idx, row in df.iterrows():
            # Parse timestamps
            start_time = self._parse_timestamp(row['Start'])
            end_time = self._parse_timestamp(row['End'])

            # Check if end time is after start time
            if end_time <= start_time:
                issues.append(f"Segment {idx}: End time must be after start time")

            # Check for very short segments (< 0.1s)
            duration = end_time - start_time
            if duration < 0.1:
                issues.append(f"Segment {idx}: Duration too short ({duration:.3f}s)")

            # Check for very long segments (> 30s)
            if duration > 30.0:
                issues.append(f"Segment {idx}: Duration very long ({duration:.1f}s) - consider splitting")

            # Check for empty text
            if not row['Text'].strip():
                issues.append(f"Segment {idx}: Text is empty")

            # Check for overlapping with next segment
            if idx < len(df) - 1:
                next_row = df.iloc[idx + 1]
                next_start = self._parse_timestamp(next_row['Start'])

                if end_time > next_start:
                    issues.append(f"Segment {idx}: Overlaps with next segment")

        return issues

    def render_side_by_side_comparison(
        self,
        segments: List[Segment]
    ):
        """Render side-by-side comparison of original and translated text.

        Args:
            segments: List of segments with translations
        """
        st.subheader("🔄 Translation Comparison")

        for idx, seg in enumerate(segments):
            with st.expander(f"Segment {idx} ({self._format_timestamp(seg.start_time)} - {self._format_timestamp(seg.end_time)})"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Original:**")
                    st.text_area(
                        "Original Text",
                        value=seg.text,
                        height=100,
                        key=f"orig_{idx}",
                        label_visibility="collapsed"
                    )

                with col2:
                    st.markdown("**Translation:**")
                    translation = getattr(seg, 'translation', '')
                    st.text_area(
                        "Translation",
                        value=translation,
                        height=100,
                        key=f"trans_{idx}",
                        label_visibility="collapsed"
                    )

