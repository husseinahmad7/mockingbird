"""Unit tests for audio processing error handling.

**Feature: video-translator, Task 8.4**
**Tests error handling for audio extraction, mixing, and video generation**
"""

import pytest
import os
import tempfile
from pathlib import Path

from src.services.audio_processing import AudioProcessingService
from src.models.core import AudioFile


class TestAudioProcessingErrors:
    """Unit tests for audio processing error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.audio_service = AudioProcessingService(temp_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        try:
            self.audio_service.cleanup_temp_files()
        except Exception:
            pass
    
    def test_extract_audio_nonexistent_file(self):
        """Test audio extraction with non-existent video file."""
        nonexistent_file = "/path/to/nonexistent/video.mp4"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            self.audio_service.extract_audio(nonexistent_file)
        
        assert "not found" in str(exc_info.value).lower()
    
    def test_extract_audio_invalid_file(self):
        """Test audio extraction with invalid video file."""
        # Create a dummy file that's not a valid video
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_file.write(b'not a valid video file')
        temp_file.close()
        
        try:
            with pytest.raises(RuntimeError) as exc_info:
                self.audio_service.extract_audio(temp_file.name)
            
            assert "ffmpeg" in str(exc_info.value).lower() or "failed" in str(exc_info.value).lower()
        finally:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
    
    def test_get_audio_info_nonexistent_file(self):
        """Test getting audio info for non-existent file."""
        nonexistent_file = "/path/to/nonexistent/audio.wav"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            self.audio_service.get_audio_info(nonexistent_file)
        
        assert "not found" in str(exc_info.value).lower()
    
    def test_get_audio_info_invalid_file(self):
        """Test getting audio info for invalid audio file."""
        # Create a dummy file that's not a valid audio
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_file.write(b'not a valid audio file')
        temp_file.close()
        
        try:
            with pytest.raises(RuntimeError) as exc_info:
                self.audio_service.get_audio_info(temp_file.name)
            
            assert "ffmpeg" in str(exc_info.value).lower() or "failed" in str(exc_info.value).lower()
        finally:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
    
    def test_mix_audio_nonexistent_background(self):
        """Test audio mixing with non-existent background file."""
        nonexistent_bg = "/path/to/nonexistent/background.wav"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            self.audio_service.mix_audio_tracks(nonexistent_bg, [])
        
        assert "not found" in str(exc_info.value).lower()
    
    def test_mix_audio_nonexistent_overlay(self):
        """Test audio mixing with non-existent overlay file."""
        # Create a valid background file
        bg_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        bg_file.close()
        
        # Create a simple WAV file
        import wave
        import struct
        with wave.open(bg_file.name, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            for i in range(16000):  # 1 second
                wav.writeframes(struct.pack('<h', 0))
        
        try:
            nonexistent_overlay = "/path/to/nonexistent/overlay.wav"

            # Create AudioFile object for the nonexistent overlay
            overlay_segment = AudioFile(
                path=nonexistent_overlay,
                duration=1.0,
                sample_rate=16000,
                channels=1
            )

            # The service should skip nonexistent files, so this won't raise an error
            # Instead, it will just return the background audio
            result = self.audio_service.mix_audio_tracks(bg_file.name, [overlay_segment])

            # Verify result exists (it should be the mixed audio or original)
            assert os.path.exists(result)
        finally:
            if os.path.exists(bg_file.name):
                os.unlink(bg_file.name)
    
    def test_create_final_video_nonexistent_video(self):
        """Test final video creation with non-existent video file."""
        nonexistent_video = "/path/to/nonexistent/video.mp4"
        
        # Create a dummy audio file
        audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        audio_file.close()
        
        try:
            with pytest.raises(FileNotFoundError) as exc_info:
                self.audio_service.create_final_video(nonexistent_video, audio_file.name)
            
            assert "not found" in str(exc_info.value).lower()
        finally:
            if os.path.exists(audio_file.name):
                os.unlink(audio_file.name)
    
    def test_create_final_video_nonexistent_audio(self):
        """Test final video creation with non-existent audio file."""
        # Create a dummy video file
        video_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        video_file.close()
        
        try:
            nonexistent_audio = "/path/to/nonexistent/audio.wav"
            
            with pytest.raises(FileNotFoundError) as exc_info:
                self.audio_service.create_final_video(video_file.name, nonexistent_audio)
            
            assert "not found" in str(exc_info.value).lower()
        finally:
            if os.path.exists(video_file.name):
                os.unlink(video_file.name)

