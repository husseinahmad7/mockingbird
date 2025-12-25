"""Property-based tests for audio extraction functionality.

**Feature: video-translator, Property 3: Audio extraction preservation**
**Validates: Requirements 2.1**
"""

import os
import tempfile
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from pathlib import Path

from src.services.audio_processing import AudioProcessingService
from src.models.core import AudioFile


class TestAudioExtractionProperties:
    """Property-based tests for audio extraction."""
    
    @pytest.fixture
    def audio_service(self):
        """Create AudioProcessingService instance for testing."""
        return AudioProcessingService()
    
    @pytest.fixture
    def sample_video_file(self):
        """Create a minimal sample video file for testing."""
        # Create a minimal video file using FFmpeg
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            video_path = temp_file.name
        
        try:
            # Create a 1-second test video with audio using FFmpeg
            import ffmpeg
            video_input = ffmpeg.input('testsrc2=duration=1:size=320x240:rate=1', f='lavfi')
            audio_input = ffmpeg.input('sine=frequency=440:duration=1', f='lavfi')
            (
                ffmpeg
                .output(video_input, audio_input, video_path, vcodec='libx264', acodec='aac', t=1)
                .overwrite_output()
                .run(quiet=True, capture_stdout=True)
            )
            yield video_path
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)
    
    def test_audio_extraction_preservation_basic(self, audio_service, sample_video_file):
        """Test that audio extraction preserves basic properties.
        
        **Feature: video-translator, Property 3: Audio extraction preservation**
        **Validates: Requirements 2.1**
        """
        # Extract audio from the sample video
        extracted_audio_path = audio_service.extract_audio(sample_video_file)
        
        try:
            # Verify the extracted audio file exists
            assert os.path.exists(extracted_audio_path), "Extracted audio file should exist"
            
            # Get audio metadata
            audio_info = audio_service.get_audio_info(extracted_audio_path)
            
            # Verify audio properties are preserved
            assert audio_info.duration > 0, "Audio duration should be positive"
            assert audio_info.sample_rate > 0, "Sample rate should be positive"
            assert audio_info.channels > 0, "Channel count should be positive"
            
            # Verify the audio file is valid
            assert audio_service.validate_audio_file(extracted_audio_path), "Extracted audio should be valid"
            
        finally:
            # Cleanup
            audio_service.cleanup_temp_files()
    
    @given(
        video_duration=st.floats(min_value=0.1, max_value=5.0),
        frequency=st.integers(min_value=220, max_value=880)
    )
    @settings(max_examples=10, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])  # Reduced examples for faster testing
    def test_audio_extraction_preservation_property(self, video_duration, frequency):
        """Property test: For any valid video file, extracting audio should produce a valid audio file 
        that preserves the original timing and content structure.
        
        **Feature: video-translator, Property 3: Audio extraction preservation**
        **Validates: Requirements 2.1**
        """
        # Create audio service instance for this test
        audio_service = AudioProcessingService()
        
        # Create a test video with specified parameters
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            video_path = temp_file.name
        
        try:
            # Create test video with audio using FFmpeg
            import ffmpeg
            video_input = ffmpeg.input(f'testsrc2=duration={video_duration}:size=320x240:rate=1', f='lavfi')
            audio_input = ffmpeg.input(f'sine=frequency={frequency}:duration={video_duration}', f='lavfi')
            (
                ffmpeg
                .output(video_input, audio_input, video_path, vcodec='libx264', acodec='aac', t=video_duration)
                .overwrite_output()
                .run(quiet=True, capture_stdout=True)
            )
            
            # Extract audio from the video
            extracted_audio_path = audio_service.extract_audio(video_path)
            
            # Verify the extracted audio file exists and is valid
            assert os.path.exists(extracted_audio_path), "Extracted audio file should exist"
            
            # Get audio metadata
            audio_info = audio_service.get_audio_info(extracted_audio_path)
            
            # Property: Audio extraction should preserve timing structure
            # The extracted audio duration should be approximately equal to the original video duration
            duration_tolerance = 0.1  # Allow 100ms tolerance
            assert abs(audio_info.duration - video_duration) <= duration_tolerance, \
                f"Audio duration {audio_info.duration} should match video duration {video_duration} within tolerance"
            
            # Property: Audio extraction should produce valid audio properties
            assert audio_info.sample_rate > 0, "Sample rate should be positive"
            assert audio_info.channels > 0, "Channel count should be positive"
            
            # Property: Extracted audio should be a valid audio file
            assert audio_service.validate_audio_file(extracted_audio_path), "Extracted audio should be valid"
            
        finally:
            # Cleanup
            if os.path.exists(video_path):
                os.unlink(video_path)
            audio_service.cleanup_temp_files()
    
    def test_audio_extraction_nonexistent_file(self, audio_service):
        """Test that audio extraction fails appropriately for nonexistent files."""
        nonexistent_path = "/path/that/does/not/exist.mp4"
        
        with pytest.raises(FileNotFoundError):
            audio_service.extract_audio(nonexistent_path)
    
    def test_audio_info_extraction(self, audio_service, sample_video_file):
        """Test that audio metadata extraction works correctly."""
        extracted_audio_path = audio_service.extract_audio(sample_video_file)
        
        try:
            audio_info = audio_service.get_audio_info(extracted_audio_path)
            
            # Verify AudioFile object properties
            assert isinstance(audio_info, AudioFile)
            assert audio_info.path == extracted_audio_path
            assert audio_info.duration > 0
            assert audio_info.sample_rate > 0
            assert audio_info.channels > 0
            
        finally:
            audio_service.cleanup_temp_files()
    
    def test_audio_format_conversion(self, audio_service, sample_video_file):
        """Test audio format conversion functionality."""
        extracted_audio_path = audio_service.extract_audio(sample_video_file)
        
        try:
            # Convert to MP3 format
            converted_path = audio_service.convert_audio_format(extracted_audio_path, 'mp3')
            
            # Verify converted file exists and is valid
            assert os.path.exists(converted_path)
            assert converted_path.endswith('.mp3')
            assert audio_service.validate_audio_file(converted_path)
            
        finally:
            audio_service.cleanup_temp_files()