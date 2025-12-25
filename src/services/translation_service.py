"""Translation service implementation with Gemini API and NLLB-200 fallback."""

import asyncio
import logging
import time
from typing import List, Optional, Dict, Any
import random

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    types = None

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None

from .base import BaseTranslationService
from ..models.core import Segment, ProcessingConfig


logger = logging.getLogger(__name__)


class TranslationService(BaseTranslationService):
    """Translation service with Gemini API primary and NLLB-200 fallback."""
    
    def __init__(self, config: ProcessingConfig):
        """Initialize the translation service with configuration."""
        self.config = config
        self.gemini_client = None
        self.nllb_model = None
        self.nllb_tokenizer = None
        self.rate_limit_delay = 1.0  # Initial delay in seconds
        self.max_retries = 3
        self.last_request_time = 0.0
        
        # Initialize Gemini API if key is provided
        if config.gemini_api_key and GEMINI_AVAILABLE:
            try:
                self.gemini_client = genai.Client(api_key=config.gemini_api_key)
                logger.info("Gemini API client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini API: {e}")
                self.gemini_client = None
        else:
            if not GEMINI_AVAILABLE:
                logger.warning("Google Genai not available - install with: pip install google-genai")
            self.gemini_client = None
    
    def translate_segments(self, segments: List[Segment], target_language: str) -> List[Segment]:
        """Translate a list of segments to the target language."""
        if not segments:
            return []
        
        # Extract texts for batch translation
        texts = [segment.text for segment in segments]
        
        # Translate texts in batches
        translated_texts = self._translate_texts_batch(texts, target_language)
        
        # Create new segments with translated text but preserve original timing
        translated_segments = []
        for i, segment in enumerate(segments):
            translated_segment = Segment(
                start_time=segment.start_time,
                end_time=segment.end_time,
                text=translated_texts[i] if i < len(translated_texts) else segment.text,
                speaker_id=segment.speaker_id,
                confidence=segment.confidence
            )
            translated_segments.append(translated_segment)
        
        return translated_segments
    
    def translate_batch(self, texts: List[str], target_language: str) -> List[str]:
        """Translate a batch of texts to the target language."""
        return self._translate_texts_batch(texts, target_language)
    
    def fallback_translate(self, text: str, target_language: str) -> str:
        """Fallback translation method using NLLB-200 model."""
        try:
            self._ensure_nllb_loaded()

            # Map language codes for NLLB-200
            lang_code = self._get_nllb_language_code(target_language)

            # Get the token ID for the target language
            # NLLB tokenizer stores language codes in lang_code_to_id attribute
            if hasattr(self.nllb_tokenizer, 'lang_code_to_id'):
                forced_bos_token_id = self.nllb_tokenizer.lang_code_to_id.get(lang_code)
            else:
                # Fallback: convert the language code token to ID
                forced_bos_token_id = self.nllb_tokenizer.convert_tokens_to_ids(lang_code)

            # Tokenize and translate
            inputs = self.nllb_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            translated_tokens = self.nllb_model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512
            )

            # Decode the translation
            translated_text = self.nllb_tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )[0]

            return translated_text
            
        except Exception as e:
            logger.error(f"Fallback translation failed: {e}")
            return text  # Return original text if translation fails
    
    def _translate_texts_batch(self, texts: List[str], target_language: str) -> List[str]:
        """Translate a batch of texts with primary/fallback logic."""
        if not texts:
            return []
        
        # Try Gemini API first
        if self.gemini_client:
            try:
                return self._translate_with_gemini(texts, target_language)
            except Exception as e:
                logger.warning(f"Gemini translation failed, falling back to NLLB: {e}")
        
        # Fallback to NLLB-200
        return [self.fallback_translate(text, target_language) for text in texts]
    
    def _translate_with_gemini(self, texts: List[str], target_language: str) -> List[str]:
        """Translate texts using Gemini API with rate limiting and retry logic."""
        translated_texts = []
        
        # Process texts in batches to respect API limits
        batch_size = min(self.config.batch_size, 10)  # Limit batch size for API
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_translations = self._translate_batch_with_retry(batch, target_language)
            translated_texts.extend(batch_translations)
        
        return translated_texts
    
    def _translate_batch_with_retry(self, texts: List[str], target_language: str) -> List[str]:
        """Translate a batch with retry logic and exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                # Apply rate limiting
                self._apply_rate_limit()

                # Create context-aware prompt
                prompt = self._create_translation_prompt(texts, target_language)

                # Make API request using new google.genai API
                response = self.gemini_client.models.generate_content(
                    model='gemini-2.0-flash-exp',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=0)
                    )
                )

                # Parse response
                translations = self._parse_gemini_response(response.text, len(texts))

                # Reset rate limit delay on success
                self.rate_limit_delay = max(1.0, self.rate_limit_delay * 0.8)

                return translations
                
            except Exception as e:
                logger.warning(f"Gemini API attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.rate_limit_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                    self.rate_limit_delay = min(60.0, self.rate_limit_delay * 1.5)
                else:
                    # Final attempt failed, raise exception
                    raise
    
    def _apply_rate_limit(self):
        """Apply rate limiting between API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _create_translation_prompt(self, texts: List[str], target_language: str) -> str:
        """Create a context-aware translation prompt for Gemini."""
        prompt = f"""Translate the following text segments to {target_language}. 
        
Preserve the meaning, tone, and context. These are segments from a video transcript, so maintain natural speech patterns.

Return only the translations, one per line, in the same order as the input.

Input segments:
"""
        
        for i, text in enumerate(texts, 1):
            prompt += f"{i}. {text}\n"
        
        prompt += f"\nTranslations in {target_language}:"
        
        return prompt
    
    def _parse_gemini_response(self, response_text: str, expected_count: int) -> List[str]:
        """Parse Gemini API response to extract translations."""
        lines = response_text.strip().split('\n')
        translations = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove numbering if present (e.g., "1. Translation text")
            if line and line[0].isdigit() and '. ' in line:
                line = line.split('. ', 1)[1]
            
            translations.append(line)
        
        # Ensure we have the expected number of translations
        while len(translations) < expected_count:
            translations.append("")  # Add empty string for missing translations
        
        return translations[:expected_count]
    
    def _ensure_nllb_loaded(self):
        """Ensure NLLB-200 model is loaded for fallback translation."""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library not available - install with: pip install transformers")
            raise ImportError("Transformers library required for NLLB fallback")
            
        if self.nllb_model is None or self.nllb_tokenizer is None:
            logger.info("Loading NLLB-200 model for fallback translation...")
            
            model_name = "facebook/nllb-200-distilled-600M"
            self.nllb_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            logger.info("NLLB-200 model loaded successfully")
    
    def _get_nllb_language_code(self, language: str) -> str:
        """Map common language names and codes to NLLB-200 language codes."""
        language_map = {
            # Full language names
            "spanish": "spa_Latn",
            "french": "fra_Latn",
            "german": "deu_Latn",
            "italian": "ita_Latn",
            "portuguese": "por_Latn",
            "russian": "rus_Cyrl",
            "chinese": "zho_Hans",
            "japanese": "jpn_Jpan",
            "korean": "kor_Hang",
            "arabic": "arb_Arab",
            "hindi": "hin_Deva",
            "english": "eng_Latn",
            "dutch": "nld_Latn",
            "polish": "pol_Latn",
            "turkish": "tur_Latn",
            "vietnamese": "vie_Latn",
            "thai": "tha_Thai",
            "indonesian": "ind_Latn",
            "czech": "ces_Latn",
            "greek": "ell_Grek",
            "hebrew": "heb_Hebr",
            "ukrainian": "ukr_Cyrl",
            "romanian": "ron_Latn",
            "hungarian": "hun_Latn",
            "swedish": "swe_Latn",
            "danish": "dan_Latn",
            "finnish": "fin_Latn",
            "norwegian": "nob_Latn",
            # ISO 639-1 language codes
            "es": "spa_Latn",
            "fr": "fra_Latn",
            "de": "deu_Latn",
            "it": "ita_Latn",
            "pt": "por_Latn",
            "ru": "rus_Cyrl",
            "zh": "zho_Hans",
            "ja": "jpn_Jpan",
            "ko": "kor_Hang",
            "ar": "arb_Arab",
            "hi": "hin_Deva",
            "en": "eng_Latn",
            "nl": "nld_Latn",
            "pl": "pol_Latn",
            "tr": "tur_Latn",
            "vi": "vie_Latn",
            "th": "tha_Thai",
            "id": "ind_Latn",
            "cs": "ces_Latn",
            "el": "ell_Grek",
            "he": "heb_Hebr",
            "uk": "ukr_Cyrl",
            "ro": "ron_Latn",
            "hu": "hun_Latn",
            "sv": "swe_Latn",
            "da": "dan_Latn",
            "fi": "fin_Latn",
            "no": "nob_Latn",
        }

        # Try exact match first
        lang_lower = language.lower()
        if lang_lower in language_map:
            return language_map[lang_lower]

        # Default to English if not found
        logger.warning(f"Language '{language}' not found in mapping, defaulting to English")
        return "eng_Latn"
    
    def validate_translation_quality(self, original: str, translated: str) -> float:
        """Validate translation quality and return a confidence score."""
        if not translated or not translated.strip():
            return 0.0
        
        # Basic quality checks
        score = 1.0
        
        # Check if translation is too short compared to original
        if len(translated) < len(original) * 0.3:
            score *= 0.7
        
        # Check if translation is suspiciously long
        if len(translated) > len(original) * 3:
            score *= 0.8
        
        # Check for repeated characters (sign of poor translation)
        if any(char * 5 in translated for char in 'abcdefghijklmnopqrstuvwxyz'):
            score *= 0.5
        
        return score