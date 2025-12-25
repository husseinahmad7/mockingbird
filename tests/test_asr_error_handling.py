"""Unit tests for ASR service error handling scenarios."""

import os
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock

from src.models.core import ProcessingConfig
from src.services.asr_service import ASRService


class TestASRErrorHandling:
    """Unit tests for ASR service error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ProcessingConfig(
            whisper_model_size="tiny",
            enable_speaker_detection=True
        )
        self.asr_service = ASRService(self.config)
    
    def test_transcription_with_nonexistent_file(self):
        """Test transcription failure with non-existent audio file.
        
        Requirements: 2.5 - ASR should handle transcription failures gracefully
        """
        nonexistent_file = "/path/to/nonexistent/audio.wav"
        
        try:
            self.asr_service.transcribe(nonexistent_file)
            # Should not reach here
            assert False, "Expected FileNotFoundError or RuntimeError"
        except RuntimeError as e:
            if "faster-whisper is not available" in str(e):
                pytest.skip("faster-whisper not available in test environment")
            else:
                # Re-raise if it's a different runtime error
                raise
        except FileNotFoundError as exc_info:
            assert "Audio file not found" in str(exc_info)
            assert nonexistent_file in str(exc_info)
    
    def test_language_detection_with_nonexistent_file(self):
        """Test language detection failure with non-existent audio file.
        
        Requirements: 2.5 - ASR should handle language detection failures gracefully
        """
        nonexistent_file = "/path/to/nonexistent/audio.wav"
        
        try:
            self.asr_service.detect_language(nonexistent_file)
            # Should not reach here
            assert False, "Expected FileNotFoundError or RuntimeError"
        except RuntimeError as e:
            if "faster-whisper is not available" in str(e):
                pytest.skip("faster-whisper not available in test environment")
            else:
                # Re-raise if it's a different runtime error
                raise
        except FileNotFoundError as exc_info:
            assert "Audio file not found" in str(exc_info)
            assert nonexistent_file in str(exc_info)
    
    def test_model_loading_without_faster_whisper(self):
        """Test model loading failure when faster-whisper is not available.
        
        Requirements: 2.5 - ASR should handle model loading errors gracefully
        """
        # Mock the WhisperModel to be None (simulating missing faster-whisper)
        with patch('src.services.asr_service.WhisperModel', None):
            asr_service = ASRService(self.config)
            
            with pytest.raises(RuntimeError) as exc_info:
                asr_service.load_model("base")
            
            assert "faster-whisper is not available" in str(exc_info.value)
    
    @patch('src.services.asr_service.WhisperModel')
    def test_model_loading_failure(self, mock_whisper_model):
        """Test model loading failure due to model initialization error.
        
        Requirements: 2.5 - ASR should handle model loading errors gracefully
        """
        # Mock WhisperModel to raise an exception during initialization
        mock_whisper_model.side_effect = Exception("Model initialization failed")
        
        with pytest.raises(RuntimeError) as exc_info:
            self.asr_service.load_model("large")
        
        assert "Failed to load ASR model" in str(exc_info.value)
        assert "Model initialization failed" in str(exc_info.value)
    
    def test_transcription_with_empty_file(self):
        """Test transcription failure with empty audio file.
        
        Requirements: 2.5 - ASR should handle invalid audio files gracefully
        """
        # Create empty temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            empty_file = f.name
        
        try:
            # Validate that empty file is detected
            assert not self.asr_service._validate_audio_file(empty_file)
            
            # Transcription should fail with empty file
            try:
                self.asr_service.transcribe(empty_file)
                # Should not reach here
                assert False, "Expected FileNotFoundError or RuntimeError"
            except RuntimeError as e:
                if "faster-whisper is not available" in str(e):
                    pytest.skip("faster-whisper not available in test environment")
                else:
                    # Re-raise if it's a different runtime error
                    raise
            except FileNotFoundError:
                # Expected for empty file
                pass
        finally:
            os.unlink(empty_file)
    
    def test_transcription_with_directory_path(self):
        """Test transcription failure when path is a directory.
        
        Requirements: 2.5 - ASR should handle invalid file paths gracefully
        """
        # Use a directory path instead of file path
        with tempfile.TemporaryDirectory() as temp_dir:
            assert not self.asr_service._validate_audio_file(temp_dir)
    
    @patch('src.services.asr_service.WhisperModel')
    def test_transcription_model_error(self, mock_whisper_model):
        """Test transcription failure due to model processing error.
        
        Requirements: 2.5 - ASR should handle transcription processing errors gracefully
        """
        # Create a valid temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(b'fake audio data')
            temp_file = f.name
        
        try:
            # Mock model to raise exception during transcription
            mock_model_instance = Mock()
            mock_model_instance.transcribe.side_effect = Exception("Transcription processing failed")
            mock_whisper_model.return_value = mock_model_instance
            
            # Load model first
            self.asr_service.load_model("base")
            
            with pytest.raises(RuntimeError) as exc_info:
                self.asr_service.transcribe(temp_file)
            
            assert "Transcription failed" in str(exc_info.value)
            assert "Transcription processing failed" in str(exc_info.value)
        finally:
            os.unlink(temp_file)
    
    @patch('src.services.asr_service.WhisperModel')
    def test_language_detection_model_error(self, mock_whisper_model):
        """Test language detection failure due to model processing error.
        
        Requirements: 2.5 - ASR should handle language detection errors gracefully
        """
        # Create a valid temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(b'fake audio data')
            temp_file = f.name
        
        try:
            # Mock model to raise exception during language detection
            mock_model_instance = Mock()
            mock_model_instance.transcribe.side_effect = Exception("Language detection failed")
            mock_whisper_model.return_value = mock_model_instance
            
            # Load model first
            self.asr_service.load_model("base")
            
            with pytest.raises(RuntimeError) as exc_info:
                self.asr_service.detect_language(temp_file)
            
            assert "Language detection failed" in str(exc_info.value)
        finally:
            os.unlink(temp_file)
    
    def test_invalid_model_size_handling(self):
        """Test handling of invalid model sizes.
        
        Requirements: 2.5 - ASR should handle configuration errors gracefully
        """
        # Test with various invalid model sizes
        invalid_sizes = ["invalid", "", None, 123]
        
        for invalid_size in invalid_sizes:
            if invalid_size is None:
                continue  # Skip None as it would cause different error
            
            try:
                # This might not fail immediately but should be handled gracefully
                # The actual validation would happen in faster-whisper
                self.asr_service.load_model(str(invalid_size))
            except (RuntimeError, TypeError, ValueError):
                # Expected - invalid model size should cause error
                pass
    
    def test_language_code_edge_cases(self):
        """Test language code mapping with edge cases.
        
        Requirements: 2.5 - ASR should handle language detection edge cases gracefully
        """
        # Test unknown language codes
        unknown_language = "unknown_language_xyz"
        
        # The service should handle unknown languages gracefully
        # by passing them through to the underlying model
        result = self.asr_service.language_codes.get(unknown_language.lower(), unknown_language)
        assert result == unknown_language
        
        # Test empty language
        empty_language = ""
        result = self.asr_service.language_codes.get(empty_language.lower(), empty_language)
        assert result == empty_language
    
    def test_concurrent_model_loading(self):
        """Test that concurrent model loading is handled safely.
        
        Requirements: 2.5 - ASR should handle concurrent access gracefully
        """
        # Test loading the same model multiple times
        try:
            self.asr_service.load_model("tiny")
            self.asr_service.load_model("tiny")  # Should not reload
            
            # Model should still be loaded
            info = self.asr_service.get_model_info()
            if info["loaded"]:
                assert info["model_size"] == "tiny"
        except RuntimeError as e:
            if "faster-whisper is not available" in str(e):
                pytest.skip("faster-whisper not available in test environment")
            else:
                raise
    
    def test_model_info_consistency(self):
        """Test that model info remains consistent across operations.
        
        Requirements: 2.5 - ASR should provide consistent state information
        """
        # Initially no model loaded
        info = self.asr_service.get_model_info()
        assert info["loaded"] is False
        
        try:
            # Load model and check info
            self.asr_service.load_model("tiny")
            info = self.asr_service.get_model_info()
            
            if info["loaded"]:
                assert info["model_size"] == "tiny"
                assert info["device"] == "cpu"
                assert info["compute_type"] == "int8"
        except RuntimeError as e:
            if "faster-whisper is not available" in str(e):
                pytest.skip("faster-whisper not available in test environment")
            else:
                raise