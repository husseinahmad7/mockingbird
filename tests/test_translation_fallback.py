"""Unit tests for translation service fallback and rate limiting."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.models.core import Segment, ProcessingConfig
from src.services.translation_service import TranslationService


@pytest.fixture
def config_with_api_key():
    """Configuration with Gemini API key."""
    return ProcessingConfig(
        gemini_api_key="test-api-key",
        batch_size=5
    )


@pytest.fixture
def config_without_api_key():
    """Configuration without Gemini API key."""
    return ProcessingConfig(
        gemini_api_key="",
        batch_size=5
    )


@pytest.fixture
def service_with_mock_gemini(config_with_api_key):
    """Create service with mocked Gemini client."""
    with patch('google.generativeai.configure'), \
         patch('google.generativeai.GenerativeModel') as mock_model:
        
        mock_client = Mock()
        mock_model.return_value = mock_client
        
        service = TranslationService(config_with_api_key)
        service.gemini_client = mock_client
        
        return service, mock_client


@pytest.fixture
def service_without_gemini(config_without_api_key):
    """Create service without Gemini API."""
    return TranslationService(config_without_api_key)


class TestTranslationFallback:
    """Unit tests for translation fallback mechanisms."""
    
    def test_fallback_when_no_api_key(self, config_without_api_key):
        """Test that service falls back to NLLB when no API key is provided."""
        service = TranslationService(config_without_api_key)
        
        # Should not have Gemini client
        assert service.gemini_client is None
        
        # Should use fallback for translation
        with patch.object(service, 'fallback_translate', return_value="translated text") as mock_fallback:
            result = service.translate_batch(["hello world"], "spanish")
            
            mock_fallback.assert_called_once_with("hello world", "spanish")
            assert result == ["translated text"]
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_gemini_api_initialization_failure(self, mock_model, mock_configure, config_with_api_key):
        """Test graceful handling of Gemini API initialization failure."""
        # Mock API initialization failure
        mock_configure.side_effect = Exception("API initialization failed")
        
        service = TranslationService(config_with_api_key)
        
        # Should fall back gracefully
        assert service.gemini_client is None
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_gemini_api_request_failure_fallback(self, mock_model, mock_configure, config_with_api_key):
        """Test fallback to NLLB when Gemini API request fails."""
        # Mock successful initialization but failed request
        mock_client = Mock()
        mock_client.generate_content.side_effect = Exception("API request failed")
        mock_model.return_value = mock_client
        
        service = TranslationService(config_with_api_key)
        
        # Mock fallback translation
        with patch.object(service, 'fallback_translate', return_value="fallback translation") as mock_fallback:
            result = service.translate_batch(["test text"], "spanish")
            
            mock_fallback.assert_called_once_with("test text", "spanish")
            assert result == ["fallback translation"]
    
    def test_context_preservation_across_segments(self, config_without_api_key):
        """Test that context is preserved when translating multiple segments."""
        service = TranslationService(config_without_api_key)
        
        segments = [
            Segment(0.0, 2.0, "Hello, my name is John.", speaker_id="speaker1"),
            Segment(2.0, 4.0, "I am a software engineer.", speaker_id="speaker1"),
            Segment(4.0, 6.0, "Nice to meet you!", speaker_id="speaker2")
        ]
        
        # Mock the batch translation to return context-aware translations
        with patch.object(service, '_translate_texts_batch') as mock_translate:
            mock_translate.return_value = [
                "Hola, mi nombre es John.",
                "Soy ingeniero de software.",
                "¡Mucho gusto!"
            ]
            
            result = service.translate_segments(segments, "spanish")
            
            # Should call batch translation with all texts together for context
            mock_translate.assert_called_once_with(
                ["Hello, my name is John.", "I am a software engineer.", "Nice to meet you!"],
                "spanish"
            )
            
            # Verify structure preservation
            assert len(result) == 3
            assert result[0].text == "Hola, mi nombre es John."
            assert result[1].text == "Soy ingeniero de software."
            assert result[2].text == "¡Mucho gusto!"
            
            # Verify timing and metadata preservation
            for original, translated in zip(segments, result):
                assert translated.start_time == original.start_time
                assert translated.end_time == original.end_time
                assert translated.speaker_id == original.speaker_id


class TestTranslationRateLimiting:
    """Unit tests for translation service rate limiting."""
    
    def test_rate_limiting_delay(self, service_with_mock_gemini):
        """Test that rate limiting introduces appropriate delays."""
        service, mock_client = service_with_mock_gemini
        
        # Mock successful response
        mock_response = Mock()
        mock_response.text = "1. translated text"
        mock_client.generate_content.return_value = mock_response
        
        # Set initial rate limit delay
        service.rate_limit_delay = 0.5
        
        # Record start time
        start_time = time.time()
        
        # Make first request
        service._translate_batch_with_retry(["test text"], "spanish")
        first_request_time = time.time()
        
        # Make second request immediately
        service._translate_batch_with_retry(["test text 2"], "spanish")
        second_request_time = time.time()
        
        # Second request should be delayed by at least the rate limit delay
        time_between_requests = second_request_time - first_request_time
        assert time_between_requests >= service.rate_limit_delay * 0.9  # Allow small timing variance
    
    def test_exponential_backoff_on_failure(self, service_with_mock_gemini):
        """Test exponential backoff when API requests fail."""
        service, mock_client = service_with_mock_gemini
        
        # Mock API failures followed by success
        mock_client.generate_content.side_effect = [
            Exception("Rate limit exceeded"),
            Exception("Rate limit exceeded"),
            Mock(text="1. success translation")
        ]
        
        service.rate_limit_delay = 0.1  # Start with small delay for testing
        service.max_retries = 3
        
        start_time = time.time()
        result = service._translate_batch_with_retry(["test text"], "spanish")
        end_time = time.time()
        
        # Should eventually succeed
        assert result == ["success translation"]
        
        # Should have taken time due to exponential backoff
        # First retry: ~0.1s, second retry: ~0.2s + random
        assert end_time - start_time >= 0.2
    
    def test_max_retries_exceeded(self, service_with_mock_gemini):
        """Test behavior when max retries are exceeded."""
        service, mock_client = service_with_mock_gemini
        
        # Mock continuous API failures
        mock_client.generate_content.side_effect = Exception("Persistent API failure")
        
        service.max_retries = 2
        
        # Should raise exception after max retries
        with pytest.raises(Exception, match="Persistent API failure"):
            service._translate_batch_with_retry(["test text"], "spanish")
    
    def test_rate_limit_delay_adjustment(self, service_with_mock_gemini):
        """Test that rate limit delay adjusts based on success/failure."""
        service, mock_client = service_with_mock_gemini
        
        # Mock successful response
        mock_response = Mock()
        mock_response.text = "1. translated text"
        mock_client.generate_content.return_value = mock_response
        
        initial_delay = 2.0
        service.rate_limit_delay = initial_delay
        
        # Successful request should reduce delay
        service._translate_batch_with_retry(["test text"], "spanish")
        
        # Delay should be reduced (multiplied by 0.8)
        assert service.rate_limit_delay < initial_delay
        assert service.rate_limit_delay == initial_delay * 0.8
    
    def test_batch_size_limiting(self, service_with_mock_gemini):
        """Test that large batches are split to respect API limits."""
        service, mock_client = service_with_mock_gemini
        
        # Mock successful responses
        mock_response = Mock()
        mock_response.text = "1. translation"
        mock_client.generate_content.return_value = mock_response
        
        # Create large batch of texts
        large_batch = [f"text {i}" for i in range(25)]
        
        with patch.object(service, '_translate_batch_with_retry', return_value=["translation"]) as mock_batch:
            service._translate_with_gemini(large_batch, "spanish")
            
            # Should be called multiple times with smaller batches
            assert mock_batch.call_count > 1
            
            # Each call should have batch size <= 10 (API limit)
            for call in mock_batch.call_args_list:
                batch_texts = call[0][0]  # First argument is the batch
                assert len(batch_texts) <= 10


class TestNLLBFallback:
    """Unit tests for NLLB-200 fallback functionality."""
    
    def test_nllb_language_code_mapping(self, service_without_gemini):
        """Test language code mapping for NLLB-200."""
        service = service_without_gemini
        
        # Test known language mappings
        assert service._get_nllb_language_code("spanish") == "spa_Latn"
        assert service._get_nllb_language_code("french") == "fra_Latn"
        assert service._get_nllb_language_code("german") == "deu_Latn"
        assert service._get_nllb_language_code("english") == "eng_Latn"
        
        # Test case insensitivity
        assert service._get_nllb_language_code("SPANISH") == "spa_Latn"
        assert service._get_nllb_language_code("Spanish") == "spa_Latn"
        
        # Test unknown language defaults to English
        assert service._get_nllb_language_code("unknown_language") == "eng_Latn"
    
    @patch('src.services.translation_service.AutoTokenizer')
    @patch('src.services.translation_service.AutoModelForSeq2SeqLM')
    def test_nllb_model_loading(self, mock_model_class, mock_tokenizer_class, service_without_gemini):
        """Test NLLB model loading on first use."""
        service = service_without_gemini
        
        # Mock tokenizer and model instances
        mock_tokenizer_instance = Mock()
        mock_model_instance = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
        mock_model_class.from_pretrained.return_value = mock_model_instance
        
        # Should load model on first call
        service._ensure_nllb_loaded()
        
        # Verify model loading
        mock_tokenizer_class.from_pretrained.assert_called_once_with("facebook/nllb-200-distilled-600M")
        mock_model_class.from_pretrained.assert_called_once_with("facebook/nllb-200-distilled-600M")
        
        assert service.nllb_tokenizer == mock_tokenizer_instance
        assert service.nllb_model == mock_model_instance
        
        # Second call should not reload
        mock_tokenizer_class.from_pretrained.reset_mock()
        mock_model_class.from_pretrained.reset_mock()
        
        service._ensure_nllb_loaded()
        
        mock_tokenizer_class.from_pretrained.assert_not_called()
        mock_model_class.from_pretrained.assert_not_called()
    
    def test_translation_quality_validation(self, service_without_gemini):
        """Test translation quality validation scoring."""
        service = service_without_gemini
        
        # Test empty translation
        assert service.validate_translation_quality("original text", "") == 0.0
        assert service.validate_translation_quality("original text", "   ") == 0.0
        
        # Test normal translation
        score = service.validate_translation_quality("Hello world", "Hola mundo")
        assert 0.5 <= score <= 1.0
        
        # Test very short translation (should reduce score)
        score = service.validate_translation_quality("This is a long sentence", "Hi")
        assert score < 1.0
        
        # Test very long translation (should reduce score)
        score = service.validate_translation_quality("Hi", "This is an extremely long translation that seems suspicious")
        assert score < 1.0
        
        # Test translation with repeated characters (poor quality)
        score = service.validate_translation_quality("Hello", "aaaaa")
        assert score < 0.7