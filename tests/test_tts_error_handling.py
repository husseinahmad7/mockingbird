"""Unit tests for TTS error handling scenarios."""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, AsyncMock

from src.services.tts_service import TTSService
from src.models.core import Segment, ProcessingConfig


class TestTTSErrorHandling:
    """Test TTS error handling scenarios."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ProcessingConfig(
            min_speed_adjustment=0.8,
            max_speed_adjustment=1.5
        )
    
    def test_missing_edge_tts_import(self):
        """Test TTS generation failures - missing edge_tts dependency."""
        config = ProcessingConfig()
        
        with patch('src.services.tts_service.edge_tts', None):
            with pytest.raises(ImportError, match="edge-tts package is required"):
                TTSService(config)
    
    def test_empty_text_generation_failure(self, config):
        """Test TTS generation failures - empty text."""
        with patch('src.services.tts_service.edge_tts') as mock_edge_tts:
            tts_service = TTSService(config)
            
            segment = Segment(
                start_time=0.0,
                end_time=5.0,
                text="",  # Empty text
                speaker_id="speaker1"
            )
            
            with pytest.raises(ValueError, match="Cannot generate speech from empty text"):
                tts_service.generate_speech(segment, "en-US-AriaNeural")
    
    def test_whitespace_only_text_generation_failure(self, config):
        """Test TTS generation failures - whitespace only text."""
        with patch('src.services.tts_service.edge_tts') as mock_edge_tts:
            tts_service = TTSService(config)
            
            segment = Segment(
                start_time=0.0,
                end_time=5.0,
                text="   \n\t  ",  # Whitespace only
                speaker_id="speaker1"
            )
            
            with pytest.raises(ValueError, match="Cannot generate speech from empty text"):
                tts_service.generate_speech(segment, "en-US-AriaNeural")
    
    def test_edge_tts_communication_failure(self, config):
        """Test TTS generation failures - Edge-TTS communication error."""
        mock_communicate = Mock()
        mock_communicate.save = AsyncMock(side_effect=Exception("Network error"))
        
        with patch('src.services.tts_service.edge_tts') as mock_edge_tts:
            mock_edge_tts.Communicate.return_value = mock_communicate
            
            tts_service = TTSService(config)
            
            segment = Segment(
                start_time=0.0,
                end_time=5.0,
                text="Hello world",
                speaker_id="speaker1"
            )
            
            with pytest.raises(RuntimeError, match="TTS generation failed: Network error"):
                tts_service.generate_speech(segment, "en-US-AriaNeural")
    
    def test_voice_selection_edge_cases(self, config):
        """Test voice selection edge cases."""
        with patch('src.services.tts_service.edge_tts') as mock_edge_tts:
            # Mock empty voice list - this should result in empty filtered list, then fallback
            mock_edge_tts.list_voices = AsyncMock(return_value=[])
            
            tts_service = TTSService(config)
            
            # When no voices match the language filter, it should still return the empty list
            # (not fallback, because the API call succeeded but returned no matching voices)
            voices = tts_service.get_available_voices("unknown-language")
            assert len(voices) == 0  # Should return empty list when no voices match
            
            # But for a known language with empty API response, should also return empty
            voices_en = tts_service.get_available_voices("en")
            assert len(voices_en) == 0  # Should return empty list when API returns empty
    
    def test_voice_api_failure_fallback(self, config):
        """Test voice selection when API fails."""
        with patch('src.services.tts_service.edge_tts') as mock_edge_tts:
            # Mock API failure
            mock_edge_tts.list_voices = AsyncMock(side_effect=Exception("API unavailable"))
            
            tts_service = TTSService(config)
            
            # Should fall back to hardcoded voices
            voices = tts_service.get_available_voices("en")
            assert len(voices) > 0
            assert "en-US-AriaNeural" in voices
    
    def test_speaker_mapping_no_voices_available(self, config):
        """Test speaker mapping when no voices are available."""
        with patch('src.services.tts_service.edge_tts') as mock_edge_tts:
            tts_service = TTSService(config)
            
            # Mock get_available_voices to return empty list
            tts_service.get_available_voices = Mock(return_value=[])
            
            with pytest.raises(RuntimeError, match="No voices available for language"):
                tts_service.map_speaker_to_voice("speaker1", "unknown-lang")
    
    def test_speed_adjustment_boundary_conditions(self, config):
        """Test speed adjustment boundary conditions."""
        with patch('src.services.tts_service.edge_tts') as mock_edge_tts:
            tts_service = TTSService(config)
            
            # Test zero duration
            speed_factor = tts_service.calculate_speed_adjustment("Hello", 0.0)
            assert speed_factor == 1.0
            
            # Test negative duration
            speed_factor = tts_service.calculate_speed_adjustment("Hello", -1.0)
            assert speed_factor == 1.0
            
            # Test very long text with short duration (should clamp to max)
            long_text = "word " * 1000  # Very long text
            speed_factor = tts_service.calculate_speed_adjustment(long_text, 0.1)
            assert speed_factor == config.max_speed_adjustment
            
            # Test short text with long duration (should clamp to min)
            speed_factor = tts_service.calculate_speed_adjustment("hi", 100.0)
            assert speed_factor == config.min_speed_adjustment
    
    def test_audio_duration_calculation_failure(self, config):
        """Test audio duration calculation when file operations fail."""
        mock_communicate = Mock()
        mock_communicate.save = AsyncMock()
        
        with patch('src.services.tts_service.edge_tts') as mock_edge_tts:
            mock_edge_tts.Communicate.return_value = mock_communicate
            
            tts_service = TTSService(config)
            
            # Mock wave.open to fail, and also mock os.path.getsize to return a reasonable size
            with patch('wave.open', side_effect=Exception("File read error")), \
                 patch('os.path.getsize', return_value=44100):  # 1 second worth of data
                segment = Segment(
                    start_time=0.0,
                    end_time=5.0,
                    text="Hello world",
                    speaker_id="speaker1"
                )
                
                # Should still work with fallback duration calculation
                audio_file = tts_service.generate_speech(segment, "en-US-AriaNeural")
                assert audio_file.duration > 0  # Should have some estimated duration
    
    def test_temp_file_cleanup_on_error(self, config):
        """Test temporary file cleanup when generation fails."""
        mock_communicate = Mock()
        mock_communicate.save = AsyncMock(side_effect=Exception("Generation failed"))
        
        with patch('src.services.tts_service.edge_tts') as mock_edge_tts:
            mock_edge_tts.Communicate.return_value = mock_communicate
            
            tts_service = TTSService(config)
            
            segment = Segment(
                start_time=0.0,
                end_time=5.0,
                text="Hello world",
                speaker_id="speaker1"
            )
            
            # Count temp files before
            initial_temp_count = len(tts_service.temp_files)
            
            with pytest.raises(RuntimeError):
                tts_service.generate_speech(segment, "en-US-AriaNeural")
            
            # Temp files should not accumulate after error
            assert len(tts_service.temp_files) == initial_temp_count
    
    def test_cleanup_temp_files_with_missing_files(self, config):
        """Test cleanup when some temp files are already missing."""
        with patch('src.services.tts_service.edge_tts') as mock_edge_tts:
            tts_service = TTSService(config)
            
            # Add some fake temp file paths
            fake_path = "/nonexistent/file.wav"
            tts_service.temp_files.append(fake_path)
            
            # Should not raise exception when cleaning up missing files
            tts_service.cleanup_temp_files()
            assert len(tts_service.temp_files) == 0
    
    def test_language_code_normalization(self, config):
        """Test language code normalization in voice selection."""
        mock_voices = [
            {'ShortName': 'en-US-AriaNeural', 'Locale': 'en-US'},
            {'ShortName': 'en-GB-LibbyNeural', 'Locale': 'en-GB'},
            {'ShortName': 'es-ES-ElviraNeural', 'Locale': 'es-ES'},
        ]
        
        with patch('src.services.tts_service.edge_tts') as mock_edge_tts:
            mock_edge_tts.list_voices = AsyncMock(return_value=mock_voices)
            
            tts_service = TTSService(config)
            
            # Test various language code formats
            en_voices = tts_service.get_available_voices("en")
            assert "en-US-AriaNeural" in en_voices
            assert "en-GB-LibbyNeural" in en_voices
            
            en_us_voices = tts_service.get_available_voices("en-US")
            assert "en-US-AriaNeural" in en_us_voices
            assert "en-GB-LibbyNeural" in en_us_voices  # Should include all English variants
            
            es_voices = tts_service.get_available_voices("es")
            assert "es-ES-ElviraNeural" in es_voices
            assert "en-US-AriaNeural" not in es_voices