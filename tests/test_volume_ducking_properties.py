"""Property-based tests for volume ducking effectiveness.

**Feature: video-translator, Property 10: Volume ducking effectiveness**
**Validates: Requirements 5.2**
"""

import pytest
import os
import tempfile
import wave
import struct
from hypothesis import given, strategies as st, assume, settings, HealthCheck

from src.services.audio_processing import AudioProcessingService
from src.models.core import AudioFile


def create_test_audio_file(duration: float, sample_rate: int = 16000, amplitude: float = 0.5) -> str:
    """Create a test WAV audio file with specified duration and amplitude."""
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_file.close()
    
    num_samples = int(duration * sample_rate)
    
    with wave.open(temp_file.name, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        # Generate simple audio with specified amplitude
        for i in range(num_samples):
            value = int(32767.0 * amplitude)
            data = struct.pack('<h', value)
            wav_file.writeframes(data)
    
    return temp_file.name


def get_audio_rms(audio_path: str) -> float:
    """Calculate RMS (root mean square) amplitude of audio file."""
    with wave.open(audio_path, 'rb') as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        samples = struct.unpack(f'<{len(frames)//2}h', frames)
        
        # Calculate RMS
        sum_squares = sum(s * s for s in samples)
        rms = (sum_squares / len(samples)) ** 0.5
        return rms


class TestVolumeDuckingProperties:
    """Property-based tests for volume ducking effectiveness."""
    
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
        
        try:
            self.audio_service.cleanup()
        except Exception:
            pass
    
    @given(
        background_duration=st.floats(min_value=1.0, max_value=10.0),
        num_speech_segments=st.integers(min_value=1, max_value=5),
        sample_rate=st.sampled_from([8000, 16000, 22050])
    )
    @settings(
        max_examples=15,
        deadline=10000,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_volume_ducking_effectiveness_property(self, background_duration, num_speech_segments, sample_rate):
        """
        **Feature: video-translator, Property 10: Volume ducking effectiveness**
        
        For any audio mixing operation, background audio levels should be reduced 
        during TTS speech segments and restored during silence.
        
        **Validates: Requirements 5.2**
        """
        # Create background audio with higher amplitude
        background_path = create_test_audio_file(background_duration, sample_rate, amplitude=0.8)
        self.temp_files.append(background_path)
        
        # Get original RMS
        original_rms = get_audio_rms(background_path)
        
        # Create speech segments
        speech_segments = []
        for i in range(num_speech_segments):
            segment_duration = min(0.5, background_duration / max(1, num_speech_segments))
            segment_path = create_test_audio_file(segment_duration, sample_rate, amplitude=0.5)
            self.temp_files.append(segment_path)
            
            speech_segments.append(AudioFile(
                path=segment_path,
                duration=segment_duration,
                sample_rate=sample_rate,
                channels=1
            ))
        
        # Apply volume ducking
        ducked_path = self.audio_service.apply_volume_ducking(background_path, speech_segments)
        self.temp_files.append(ducked_path)
        
        # Property 1: Ducked audio file should exist
        assert os.path.exists(ducked_path), "Ducked audio file should be created"
        
        # Property 2: Ducked audio file should be valid
        assert os.path.getsize(ducked_path) > 0, "Ducked audio file should not be empty"
        
        # Property 3: Ducked audio should be a valid WAV file
        try:
            with wave.open(ducked_path, 'rb') as wav_file:
                assert wav_file.getnchannels() > 0
                assert wav_file.getframerate() > 0
                assert wav_file.getnframes() > 0
        except Exception as e:
            pytest.fail(f"Ducked audio is not a valid WAV file: {e}")
        
        # Property 4: Ducked audio RMS should be lower than original
        ducked_rms = get_audio_rms(ducked_path)
        assert ducked_rms < original_rms, \
            f"Ducked audio RMS ({ducked_rms}) should be lower than original ({original_rms})"
        
        # Property 5: Ducked audio should be reduced by a reasonable factor
        # Typical ducking reduces volume to 20-40% of original
        reduction_ratio = ducked_rms / original_rms
        assert 0.1 <= reduction_ratio <= 0.6, \
            f"Ducking reduction ratio ({reduction_ratio}) should be between 0.1 and 0.6"
    
    @given(
        duration=st.floats(min_value=0.5, max_value=5.0)
    )
    @settings(
        max_examples=10,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_ducking_with_no_speech_property(self, duration):
        """Property: Ducking with no speech segments should return original audio."""
        background_path = create_test_audio_file(duration)
        self.temp_files.append(background_path)
        
        # Apply ducking with no speech segments
        ducked_path = self.audio_service.apply_volume_ducking(background_path, [])
        
        # Property: Should return original path when no speech segments
        assert ducked_path == background_path

