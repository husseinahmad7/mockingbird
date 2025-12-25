"""Property-based tests for TTS timing alignment."""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock

from src.services.tts_service import TTSService
from src.models.core import Segment, ProcessingConfig, AudioFile


# Mock edge_tts module for testing
class MockEdgeTTS:
    class Communicate:
        def __init__(self, text, voice, rate="+0%"):
            self.text = text
            self.voice = voice
            self.rate = rate
        
        async def save(self, path):
            # Create a dummy audio file
            with open(path, 'wb') as f:
                # Write a minimal WAV header and some dummy data
                f.write(b'RIFF')
                f.write((44 + 1000).to_bytes(4, 'little'))  # File size
                f.write(b'WAVE')
                f.write(b'fmt ')
                f.write((16).to_bytes(4, 'little'))  # Format chunk size
                f.write((1).to_bytes(2, 'little'))   # Audio format (PCM)
                f.write((1).to_bytes(2, 'little'))   # Channels
                f.write((22050).to_bytes(4, 'little'))  # Sample rate
                f.write((44100).to_bytes(4, 'little'))  # Byte rate
                f.write((2).to_bytes(2, 'little'))   # Block align
                f.write((16).to_bytes(2, 'little'))  # Bits per sample
                f.write(b'data')
                f.write((1000).to_bytes(4, 'little'))  # Data size
                f.write(b'\x00' * 1000)  # Dummy audio data
    
    @staticmethod
    async def list_voices():
        return [
            {'ShortName': 'en-US-AriaNeural', 'Locale': 'en-US'},
            {'ShortName': 'en-US-JennyNeural', 'Locale': 'en-US'},
            {'ShortName': 'es-ES-ElviraNeural', 'Locale': 'es-ES'},
        ]


def create_tts_service():
    """Create TTS service with mocked edge_tts."""
    config = ProcessingConfig(
        min_speed_adjustment=0.8,
        max_speed_adjustment=1.5
    )
    with patch('src.services.tts_service.edge_tts', MockEdgeTTS()):
        return TTSService(config)


# Strategy for generating valid text segments
text_segments = st.text(
    min_size=3, 
    max_size=200,
    alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd', 'Pe', 'Pf', 'Pi', 'Po', 'Ps', 'Zs'),
        blacklist_characters='\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x7f\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f'
    )
).filter(lambda x: x.strip() and len(x.strip()) >= 3)

# Strategy for generating valid time ranges
time_ranges = st.tuples(
    st.floats(min_value=0.0, max_value=100.0),
    st.floats(min_value=0.1, max_value=200.0)
).map(lambda x: (x[0], x[0] + x[1]))  # Ensure end > start


@given(
    text=text_segments,
    time_range=time_ranges,
    voice=st.sampled_from(['en-US-AriaNeural', 'en-US-JennyNeural'])
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=5000)
def test_tts_timing_alignment_property(text, time_range, voice):
    """
    **Feature: video-translator, Property 7: TTS timing alignment**
    **Validates: Requirements 4.2, 4.3**
    
    For any text segment, the generated TTS audio duration should match 
    the original segment duration within acceptable speed adjustment bounds (0.8x to 1.5x).
    """
    tts_service = create_tts_service()
    
    start_time, end_time = time_range
    assume(end_time > start_time)
    assume(len(text.strip()) >= 3)  # Ensure meaningful text
    
    segment = Segment(
        start_time=start_time,
        end_time=end_time,
        text=text,
        speaker_id="speaker1"
    )
    
    # Calculate expected speed adjustment
    target_duration = segment.duration
    speed_factor = tts_service.calculate_speed_adjustment(text, target_duration)
    
    # Generate speech
    audio_file = tts_service.generate_speech(segment, voice, speed_factor)
    
    try:
        # Property: Speed factor should be within acceptable bounds
        assert tts_service.config.min_speed_adjustment <= speed_factor <= tts_service.config.max_speed_adjustment
        
        # Property: Audio file should be created successfully
        assert os.path.exists(audio_file.path)
        assert audio_file.duration > 0
        
        # Property: Generated audio duration should be reasonable relative to target
        # Allow generous tolerance since TTS timing varies significantly
        tolerance = 0.8  # 80% tolerance for property testing
        min_expected = target_duration * (1 - tolerance)
        max_expected = target_duration * (1 + tolerance)
        
        # The actual duration should be within reasonable bounds of the target
        assert min_expected <= audio_file.duration <= max_expected, \
            f"Audio duration {audio_file.duration} not within bounds [{min_expected}, {max_expected}] for target {target_duration}"
    
    finally:
        # Cleanup
        if os.path.exists(audio_file.path):
            os.unlink(audio_file.path)


@given(
    text=text_segments,
    target_duration=st.floats(min_value=0.1, max_value=60.0)
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=1000)
def test_speed_adjustment_calculation_property(text, target_duration):
    """
    Property: Speed adjustment calculation should always return values within bounds.
    """
    tts_service = create_tts_service()
    assume(len(text.strip()) >= 3)  # Ensure meaningful text
    
    speed_factor = tts_service.calculate_speed_adjustment(text, target_duration)
    
    # Property: Speed factor must be within configured bounds
    assert tts_service.config.min_speed_adjustment <= speed_factor <= tts_service.config.max_speed_adjustment
    
    # Property: Speed factor should be positive
    assert speed_factor > 0


@given(
    text=text_segments,
    voice=st.sampled_from(['en-US-AriaNeural', 'en-US-JennyNeural']),
    speed_factor=st.floats(min_value=0.5, max_value=2.0)
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=5000)
def test_speech_generation_consistency_property(text, voice, speed_factor):
    """
    Property: Speech generation should consistently produce valid audio files.
    """
    tts_service = create_tts_service()
    assume(len(text.strip()) >= 3)  # Ensure meaningful text
    
    segment = Segment(
        start_time=0.0,
        end_time=5.0,
        text=text,
        speaker_id="speaker1"
    )
    
    audio_file = tts_service.generate_speech(segment, voice, speed_factor)
    
    try:
        # Property: Audio file should have valid properties
        assert isinstance(audio_file, AudioFile)
        assert os.path.exists(audio_file.path)
        assert audio_file.duration > 0
        assert audio_file.sample_rate > 0
        assert audio_file.channels > 0
        assert audio_file.path.endswith('.wav')
    
    finally:
        # Cleanup
        if os.path.exists(audio_file.path):
            os.unlink(audio_file.path)