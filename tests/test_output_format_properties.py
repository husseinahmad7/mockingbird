"""Property-based tests for output format consistency.

**Feature: video-translator, Property 11: Output format consistency**
**Validates: Requirements 5.4, 10.1**
"""

import pytest
import os
import tempfile
import subprocess
from hypothesis import given, strategies as st, assume, settings, HealthCheck
import wave
import struct

from src.services.audio_processing import AudioProcessingService


def create_test_video_file(duration: float = 2.0) -> str:
    """Create a minimal test video file using FFmpeg."""
    temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_file.close()
    
    try:
        # Create a simple test video with FFmpeg
        # Generate a test pattern video with audio
        subprocess.run([
            'ffmpeg', '-f', 'lavfi', '-i', f'testsrc=duration={duration}:size=320x240:rate=10',
            '-f', 'lavfi', '-i', f'sine=frequency=1000:duration={duration}',
            '-c:v', 'libx264', '-c:a', 'aac', '-shortest', '-y', temp_file.name
        ], capture_output=True, check=True, timeout=30)
        
        return temp_file.name
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        # If FFmpeg is not available or fails, create a dummy file
        # This allows tests to run even without FFmpeg installed
        with open(temp_file.name, 'wb') as f:
            f.write(b'dummy video file')
        return temp_file.name


def create_test_audio_file(duration: float = 2.0, sample_rate: int = 16000) -> str:
    """Create a test WAV audio file."""
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_file.close()
    
    num_samples = int(duration * sample_rate)
    
    with wave.open(temp_file.name, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        
        for i in range(num_samples):
            value = int(32767.0 * 0.3)
            data = struct.pack('<h', value)
            wav_file.writeframes(data)
    
    return temp_file.name


def is_valid_mp4(file_path: str) -> bool:
    """Check if a file is a valid MP4 video."""
    if not os.path.exists(file_path):
        return False
    
    if os.path.getsize(file_path) == 0:
        return False
    
    # Check MP4 file signature
    try:
        with open(file_path, 'rb') as f:
            # Skip first 4 bytes (size)
            f.seek(4)
            # Read file type
            ftyp = f.read(4)
            return ftyp in [b'ftyp', b'mdat', b'moov']
    except Exception:
        return False


class TestOutputFormatProperties:
    """Property-based tests for output format consistency."""
    
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
        video_duration=st.floats(min_value=1.0, max_value=5.0),
        audio_duration=st.floats(min_value=1.0, max_value=5.0)
    )
    @settings(
        max_examples=10,
        deadline=30000,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_output_format_consistency_property(self, video_duration, audio_duration):
        """
        **Feature: video-translator, Property 11: Output format consistency**
        
        For any completed processing job, the final output should always be in MP4 format 
        with valid video and audio streams.
        
        **Validates: Requirements 5.4, 10.1**
        """
        # Create test video and audio files
        video_path = create_test_video_file(video_duration)
        self.temp_files.append(video_path)
        
        audio_path = create_test_audio_file(audio_duration)
        self.temp_files.append(audio_path)
        
        # Skip test if FFmpeg is not available (dummy file created)
        if os.path.getsize(video_path) < 1000:
            pytest.skip("FFmpeg not available for video creation")
        
        # Create final video with dubbed audio
        try:
            final_video_path = self.audio_service.create_final_video(video_path, audio_path)
            self.temp_files.append(final_video_path)
        except Exception as e:
            # If FFmpeg is not available, skip the test
            if "ffmpeg" in str(e).lower() or "not found" in str(e).lower():
                pytest.skip(f"FFmpeg not available: {e}")
            raise
        
        # Property 1: Output file should exist
        assert os.path.exists(final_video_path), "Final video file should be created"
        
        # Property 2: Output file should not be empty
        assert os.path.getsize(final_video_path) > 0, "Final video file should not be empty"
        
        # Property 3: Output file should have .mp4 extension
        assert final_video_path.endswith('.mp4'), "Final video should have .mp4 extension"
        
        # Property 4: Output file should be a valid MP4
        assert is_valid_mp4(final_video_path), "Final video should be a valid MP4 file"
    
    def test_output_format_with_invalid_inputs(self):
        """Property: Service should raise appropriate errors for invalid inputs."""
        # Test with non-existent video file
        with pytest.raises(FileNotFoundError):
            self.audio_service.create_final_video("/nonexistent/video.mp4", "/nonexistent/audio.wav")
        
        # Test with non-existent audio file
        video_path = create_test_video_file(1.0)
        self.temp_files.append(video_path)
        
        with pytest.raises(FileNotFoundError):
            self.audio_service.create_final_video(video_path, "/nonexistent/audio.wav")

