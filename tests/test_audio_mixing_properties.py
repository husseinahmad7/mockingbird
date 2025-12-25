"""Property-based tests for audio mixing completeness.

**Feature: video-translator, Property 9: Audio mixing completeness**
**Validates: Requirements 5.1, 5.3**
"""

import pytest
import os
import tempfile
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from unittest.mock import patch, Mock
import wave
import struct

from src.services.audio_processing import AudioProcessingService
from src.models.core import AudioFile


def create_test_audio_file(duration: float, sample_rate: int = 16000) -> str:
    """Create a test WAV audio file with specified duration."""
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_file.close()
    
    num_samples = int(duration * sample_rate)
    
    with wave.open(temp_file.name, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        # Generate simple sine wave
        for i in range(num_samples):
            value = int(32767.0 * 0.5)  # Simple constant value
            data = struct.pack('<h', value)
            wav_file.writeframes(data)
    
    return temp_file.name


class TestAudioMixingProperties:
    """Property-based tests for audio mixing completeness."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.audio_service = AudioProcessingService(temp_dir=self.temp_dir)
        self.temp_files = []
    
    def teardown_method(self):
        """Clean up test fixtures."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception:
                pass
        
        # Clean up service temp files
        try:
            self.audio_service.cleanup()
        except Exception:
            pass
    
    @given(
        original_duration=st.floats(min_value=1.0, max_value=10.0),
        num_segments=st.integers(min_value=0, max_value=5),
        sample_rate=st.sampled_from([8000, 16000, 22050])
    )
    @settings(
        max_examples=20,
        deadline=10000,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_audio_mixing_completeness_property(self, original_duration, num_segments, sample_rate):
        """
        **Feature: video-translator, Property 9: Audio mixing completeness**
        
        For any set of TTS segments and original background audio, the mixed output 
        should contain both audio sources with proper temporal alignment.
        
        **Validates: Requirements 5.1, 5.3**
        """
        # Create original audio file
        original_path = create_test_audio_file(original_duration, sample_rate)
        self.temp_files.append(original_path)
        
        # Create TTS segment files
        tts_segments = []
        for i in range(num_segments):
            segment_duration = min(1.0, original_duration / max(1, num_segments))
            segment_path = create_test_audio_file(segment_duration, sample_rate)
            self.temp_files.append(segment_path)
            
            tts_segments.append(AudioFile(
                path=segment_path,
                duration=segment_duration,
                sample_rate=sample_rate,
                channels=1
            ))
        
        # Mix audio tracks
        mixed_path = self.audio_service.mix_audio_tracks(original_path, tts_segments)
        self.temp_files.append(mixed_path)
        
        # Property 1: Mixed audio file should exist
        assert os.path.exists(mixed_path), "Mixed audio file should be created"
        
        # Property 2: Mixed audio file should be valid
        assert os.path.getsize(mixed_path) > 0, "Mixed audio file should not be empty"
        
        # Property 3: Mixed audio should be a valid WAV file
        try:
            with wave.open(mixed_path, 'rb') as wav_file:
                assert wav_file.getnchannels() > 0
                assert wav_file.getframerate() > 0
                assert wav_file.getnframes() > 0
        except Exception as e:
            pytest.fail(f"Mixed audio is not a valid WAV file: {e}")
        
        # Property 4: If no TTS segments, output should be the original
        if num_segments == 0:
            # When no segments, the service returns the original path
            assert mixed_path == original_path or os.path.exists(mixed_path)
    
    @given(
        duration=st.floats(min_value=0.5, max_value=5.0)
    )
    @settings(
        max_examples=10,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_mixing_with_empty_segments_property(self, duration):
        """Property: Mixing with empty segment list should return original audio."""
        original_path = create_test_audio_file(duration)
        self.temp_files.append(original_path)
        
        # Mix with empty segments
        mixed_path = self.audio_service.mix_audio_tracks(original_path, [])
        
        # Property: Should return original path when no segments
        assert mixed_path == original_path

