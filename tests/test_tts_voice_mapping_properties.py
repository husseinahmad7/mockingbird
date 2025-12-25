"""Property-based tests for TTS voice mapping consistency."""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from unittest.mock import patch

from src.services.tts_service import TTSService
from src.models.core import ProcessingConfig


# Mock edge_tts module for testing
class MockEdgeTTS:
    @staticmethod
    async def list_voices():
        return [
            {'ShortName': 'en-US-AriaNeural', 'Locale': 'en-US'},
            {'ShortName': 'en-US-JennyNeural', 'Locale': 'en-US'},
            {'ShortName': 'en-US-GuyNeural', 'Locale': 'en-US'},
            {'ShortName': 'es-ES-ElviraNeural', 'Locale': 'es-ES'},
            {'ShortName': 'es-ES-AlvaroNeural', 'Locale': 'es-ES'},
            {'ShortName': 'fr-FR-DeniseNeural', 'Locale': 'fr-FR'},
            {'ShortName': 'fr-FR-HenriNeural', 'Locale': 'fr-FR'},
        ]


def create_tts_service():
    """Create TTS service with mocked edge_tts."""
    config = ProcessingConfig()
    with patch('src.services.tts_service.edge_tts', MockEdgeTTS()):
        return TTSService(config)


# Strategy for generating speaker IDs
speaker_ids = st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')))

# Strategy for generating language codes
languages = st.sampled_from(['en', 'en-US', 'es', 'es-ES', 'fr', 'fr-FR'])


@given(
    speaker_id=speaker_ids,
    language=languages
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_voice_mapping_consistency_property(speaker_id, language):
    """
    **Feature: video-translator, Property 8: Voice mapping consistency**
    **Validates: Requirements 4.4**
    
    For any translation job with multiple speakers, the same speaker_id should 
    consistently map to the same TTS voice throughout the entire job.
    """
    tts_service = create_tts_service()
    
    # Get voice mapping multiple times for the same speaker
    voice1 = tts_service.map_speaker_to_voice(speaker_id, language)
    voice2 = tts_service.map_speaker_to_voice(speaker_id, language)
    voice3 = tts_service.map_speaker_to_voice(speaker_id, language)
    
    # Property: Same speaker should always get the same voice
    assert voice1 == voice2 == voice3
    
    # Property: Voice should be valid for the language
    available_voices = tts_service.get_available_voices(language)
    assert voice1 in available_voices


@given(
    speaker_ids_list=st.lists(speaker_ids, min_size=2, max_size=10, unique=True),
    language=languages
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_multiple_speakers_distinct_mapping_property(speaker_ids_list, language):
    """
    Property: Different speakers should ideally get different voices when possible.
    """
    assume(len(speaker_ids_list) >= 2)
    
    tts_service = create_tts_service()
    
    # Get voice mappings for all speakers
    voice_mappings = {}
    for speaker_id in speaker_ids_list:
        voice_mappings[speaker_id] = tts_service.map_speaker_to_voice(speaker_id, language)
    
    # Property: Each speaker should get a consistent voice
    for speaker_id in speaker_ids_list:
        # Test consistency by calling again
        voice_again = tts_service.map_speaker_to_voice(speaker_id, language)
        assert voice_mappings[speaker_id] == voice_again
    
    # Property: All voices should be valid for the language
    available_voices = tts_service.get_available_voices(language)
    for voice in voice_mappings.values():
        assert voice in available_voices
    
    # Property: If we have enough voices, different speakers should get different voices
    # (This is a best-effort property since hash collisions can occur)
    unique_voices = set(voice_mappings.values())
    available_voice_count = len(available_voices)
    
    if available_voice_count >= len(speaker_ids_list):
        # We should have good distribution, but allow some hash collisions
        # At least 70% of speakers should get unique voices if enough are available
        min_unique_expected = max(1, int(len(speaker_ids_list) * 0.7))
        assert len(unique_voices) >= min_unique_expected


@given(
    speaker_id=speaker_ids,
    languages_list=st.lists(languages, min_size=2, max_size=5, unique=True)
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_voice_mapping_language_specific_property(speaker_id, languages_list):
    """
    Property: Same speaker should get appropriate voices for different languages.
    """
    assume(len(languages_list) >= 2)
    
    tts_service = create_tts_service()
    
    voice_mappings = {}
    for language in languages_list:
        voice_mappings[language] = tts_service.map_speaker_to_voice(speaker_id, language)
    
    # Property: Each language should get a valid voice for that language
    for language, voice in voice_mappings.items():
        available_voices = tts_service.get_available_voices(language)
        assert voice in available_voices
    
    # Property: Consistency within each language
    for language in languages_list:
        voice_again = tts_service.map_speaker_to_voice(speaker_id, language)
        assert voice_mappings[language] == voice_again


@given(language=languages)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_voice_availability_property(language):
    """
    Property: get_available_voices should always return at least one voice.
    """
    tts_service = create_tts_service()
    
    voices = tts_service.get_available_voices(language)
    
    # Property: Should always return at least one voice
    assert len(voices) > 0
    
    # Property: All returned voices should be strings
    for voice in voices:
        assert isinstance(voice, str)
        assert len(voice) > 0
    
    # Property: Should return the same list when called multiple times
    voices2 = tts_service.get_available_voices(language)
    assert voices == voices2


@given(
    speaker_id=speaker_ids,
    language=languages
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_voice_mapping_deterministic_property(speaker_id, language):
    """
    Property: Voice mapping should be deterministic based on speaker_id.
    """
    # Create two separate TTS service instances
    tts_service1 = create_tts_service()
    tts_service2 = create_tts_service()
    
    # Get voice mapping from both services
    voice1 = tts_service1.map_speaker_to_voice(speaker_id, language)
    voice2 = tts_service2.map_speaker_to_voice(speaker_id, language)
    
    # Property: Same speaker_id should map to same voice across different service instances
    assert voice1 == voice2