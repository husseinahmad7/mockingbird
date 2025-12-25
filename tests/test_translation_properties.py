"""Property-based tests for translation service."""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from datetime import datetime
from unittest.mock import Mock, patch

from src.models.core import Segment, ProcessingConfig
from src.services.translation_service import TranslationService


# Strategy for generating valid segments
@st.composite
def segment_strategy(draw):
    """Generate valid Segment objects for testing."""
    start_time = draw(st.floats(min_value=0.0, max_value=3600.0))
    duration = draw(st.floats(min_value=0.1, max_value=60.0))
    end_time = start_time + duration
    
    text = draw(st.text(min_size=1, max_size=200).filter(lambda x: x.strip()))
    speaker_id = draw(st.one_of(st.none(), st.text(min_size=1, max_size=10)))
    confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    
    return Segment(
        start_time=start_time,
        end_time=end_time,
        text=text,
        speaker_id=speaker_id,
        confidence=confidence
    )


@st.composite
def segment_list_strategy(draw):
    """Generate a list of chronologically ordered segments."""
    num_segments = draw(st.integers(min_value=1, max_value=10))
    segments = []
    
    current_time = 0.0
    for i in range(num_segments):
        # Generate a gap before this segment
        gap = draw(st.floats(min_value=0.0, max_value=5.0))
        start_time = current_time + gap
        
        # Generate duration for this segment
        duration = draw(st.floats(min_value=0.1, max_value=10.0))
        end_time = start_time + duration
        
        # Generate other properties
        text = draw(st.text(min_size=1, max_size=100).filter(lambda x: x.strip()))
        speaker_id = draw(st.one_of(st.none(), st.text(min_size=1, max_size=10)))
        confidence = draw(st.floats(min_value=0.0, max_value=1.0))
        
        segment = Segment(
            start_time=start_time,
            end_time=end_time,
            text=text,
            speaker_id=speaker_id,
            confidence=confidence
        )
        segments.append(segment)
        
        # Update current time for next segment
        current_time = end_time
    
    return segments


class TestTranslationService:
    """Property-based tests for TranslationService."""
    
    @pytest.fixture
    def translation_service(self):
        """Create a translation service instance for testing."""
        config = ProcessingConfig(
            gemini_api_key="",  # No API key for testing - will use fallback
            batch_size=5
        )
        service = TranslationService(config)
        
        # Mock the fallback translation to avoid model loading
        def mock_fallback_translate(text, target_language):
            return f"[{target_language}] {text}"
        
        service.fallback_translate = Mock(side_effect=mock_fallback_translate)
        return service
    
    @given(segment_list_strategy(), st.text(min_size=1, max_size=20))
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=5000,  # 5 second deadline
        max_examples=20  # Reduce examples for faster testing
    )
    def test_translation_structure_preservation(self, translation_service, segments, target_language):
        """
        **Feature: video-translator, Property 6: Translation structure preservation**
        **Validates: Requirements 3.3, 3.5**
        
        For any set of transcription segments, translation should preserve 
        the number of segments, their timing, and chronological order.
        """
        # Filter out any segments with empty text after stripping
        segments = [s for s in segments if s.text.strip()]
        assume(len(segments) > 0)
        
        # Perform translation
        translated_segments = translation_service.translate_segments(segments, target_language)
        
        # Property 1: Number of segments should be preserved
        assert len(translated_segments) == len(segments), \
            f"Expected {len(segments)} segments, got {len(translated_segments)}"
        
        # Property 2: Timing should be preserved
        for original, translated in zip(segments, translated_segments):
            assert translated.start_time == original.start_time, \
                f"Start time changed: {original.start_time} -> {translated.start_time}"
            assert translated.end_time == original.end_time, \
                f"End time changed: {original.end_time} -> {translated.end_time}"
            assert translated.speaker_id == original.speaker_id, \
                f"Speaker ID changed: {original.speaker_id} -> {translated.speaker_id}"
            assert translated.confidence == original.confidence, \
                f"Confidence changed: {original.confidence} -> {translated.confidence}"
        
        # Property 3: Chronological order should be preserved
        for i in range(len(translated_segments) - 1):
            assert translated_segments[i].start_time <= translated_segments[i + 1].start_time, \
                f"Chronological order violated at index {i}: " \
                f"{translated_segments[i].start_time} > {translated_segments[i + 1].start_time}"
        
        # Property 4: All segments should have valid timing
        for i, segment in enumerate(translated_segments):
            assert segment.start_time < segment.end_time, \
                f"Invalid timing at index {i}: start_time ({segment.start_time}) >= end_time ({segment.end_time})"
            assert segment.start_time >= 0, \
                f"Negative start_time at index {i}: {segment.start_time}"
        
        # Property 5: Text should be present (not empty after translation)
        for i, segment in enumerate(translated_segments):
            assert segment.text is not None, f"Text is None at index {i}"
            assert isinstance(segment.text, str), f"Text is not a string at index {i}"
    
    @given(st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=5), 
           st.text(min_size=1, max_size=20))
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=2000,  # 2 second deadline
        max_examples=10  # Reduce examples for faster testing
    )
    def test_batch_translation_consistency(self, translation_service, texts, target_language):
        """
        Test that batch translation maintains input-output correspondence.
        """
        # Filter out empty texts
        texts = [t for t in texts if t.strip()]
        assume(len(texts) > 0)
        
        translated_texts = translation_service.translate_batch(texts, target_language)
        
        # Should return same number of translations as inputs
        assert len(translated_texts) == len(texts), \
            f"Expected {len(texts)} translations, got {len(translated_texts)}"
        
        # All translations should be strings
        for i, translation in enumerate(translated_texts):
            assert isinstance(translation, str), \
                f"Translation at index {i} is not a string: {type(translation)}"