"""Property-based tests for configuration validation thoroughness.

**Feature: video-translator, Property 15: Configuration validation thoroughness**
**Validates: Requirements 8.5**

Property 15: Configuration validation thoroughness
For any system configuration, invalid settings should be detected and clear
guidance should be provided for correction.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from typing import Dict, Any

from src.services.config_manager import ConfigurationManager


class TestConfigurationValidationProperties:
    """Property-based tests for configuration validation thoroughness."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config_manager = ConfigurationManager()
    
    @given(
        api_key=st.one_of(
            st.none(),
            st.text(min_size=0, max_size=5),  # Too short
            st.text(min_size=10, max_size=100),  # Valid length
            st.integers(),  # Wrong type
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_api_key_validation_property(self, api_key):
        """Property: API key validation should detect invalid keys.
        
        For any API key value, validation should correctly identify
        invalid keys and provide clear error messages.
        """
        config = {'gemini_api_key': api_key}
        is_valid, errors = self.config_manager.validate_configuration(config)
        
        # Property: Invalid types should be rejected
        if not isinstance(api_key, (str, type(None))):
            assert not is_valid, \
                "Non-string API keys should be invalid"
            assert any('string' in err.lower() for err in errors), \
                "Error message should mention type requirement"
        
        # Property: Too short keys should be rejected
        elif isinstance(api_key, str) and 0 < len(api_key) < 10:
            assert not is_valid, \
                "Too short API keys should be invalid"
            assert any('invalid' in err.lower() or 'short' in err.lower() for err in errors), \
                "Error message should mention length issue"
        
        # Property: Valid keys should be accepted
        elif isinstance(api_key, str) and len(api_key) >= 10:
            # This specific validation should pass (other fields might fail)
            api_key_errors = [e for e in errors if 'api key' in e.lower()]
            assert len(api_key_errors) == 0, \
                "Valid API keys should not generate errors"
    
    @given(
        model_size=st.one_of(
            st.sampled_from(['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']),
            st.text(min_size=1, max_size=20).filter(
                lambda x: x not in ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']
            ),
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_model_size_validation_property(self, model_size):
        """Property: Model size validation should accept only valid sizes.
        
        For any model size value, validation should correctly identify
        valid and invalid model sizes.
        """
        config = {'whisper_model_size': model_size}
        is_valid, errors = self.config_manager.validate_configuration(config)
        
        valid_sizes = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']
        
        # Property: Valid sizes should be accepted
        if model_size in valid_sizes:
            model_errors = [e for e in errors if 'whisper' in e.lower() or 'model' in e.lower()]
            assert len(model_errors) == 0, \
                f"Valid model size '{model_size}' should not generate errors"
        
        # Property: Invalid sizes should be rejected
        else:
            assert not is_valid, \
                f"Invalid model size '{model_size}' should be rejected"
            assert any('whisper' in err.lower() or 'model' in err.lower() for err in errors), \
                "Error message should mention model size issue"
    
    @given(
        max_speed=st.one_of(
            st.floats(min_value=1.0, max_value=2.0),  # Valid
            st.floats(min_value=-10.0, max_value=0.99),  # Too low
            st.floats(min_value=2.01, max_value=10.0),  # Too high
            st.text(),  # Wrong type
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_speed_adjustment_validation_property(self, max_speed):
        """Property: Speed adjustment validation should enforce bounds.
        
        For any speed adjustment value, validation should enforce
        the acceptable range (1.0 to 2.0).
        """
        config = {'max_speed_adjustment': max_speed}
        is_valid, errors = self.config_manager.validate_configuration(config)
        
        # Property: Valid range should be accepted
        if isinstance(max_speed, (int, float)) and 1.0 <= max_speed <= 2.0:
            speed_errors = [e for e in errors if 'speed' in e.lower()]
            assert len(speed_errors) == 0, \
                f"Valid speed adjustment {max_speed} should not generate errors"
        
        # Property: Invalid values should be rejected
        else:
            assert not is_valid, \
                f"Invalid speed adjustment {max_speed} should be rejected"
            assert any('speed' in err.lower() for err in errors), \
                "Error message should mention speed adjustment issue"
    
    @given(
        batch_size=st.one_of(
            st.integers(min_value=1, max_value=100),  # Valid
            st.integers(min_value=-100, max_value=0),  # Too low
            st.integers(min_value=101, max_value=1000),  # Too high
            st.floats(),  # Wrong type
            st.text(),  # Wrong type
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_batch_size_validation_property(self, batch_size):
        """Property: Batch size validation should enforce integer bounds.
        
        For any batch size value, validation should enforce that it's
        an integer between 1 and 100.
        """
        config = {'batch_size': batch_size}
        is_valid, errors = self.config_manager.validate_configuration(config)
        
        # Property: Valid batch sizes should be accepted
        if isinstance(batch_size, int) and 1 <= batch_size <= 100:
            batch_errors = [e for e in errors if 'batch' in e.lower()]
            assert len(batch_errors) == 0, \
                f"Valid batch size {batch_size} should not generate errors"
        
        # Property: Invalid values should be rejected
        else:
            assert not is_valid, \
                f"Invalid batch size {batch_size} should be rejected"
            assert any('batch' in err.lower() for err in errors), \
                "Error message should mention batch size issue"
    
    @given(
        config_dict=st.fixed_dictionaries({
            'gemini_api_key': st.text(min_size=10, max_size=100),
            'whisper_model_size': st.sampled_from(['tiny', 'base', 'small', 'medium']),
            'max_speed_adjustment': st.floats(min_value=1.0, max_value=2.0),
            'min_speed_adjustment': st.floats(min_value=0.5, max_value=1.0),
            'batch_size': st.integers(min_value=1, max_value=100),
        })
    )
    @settings(max_examples=100, deadline=None)
    def test_valid_configuration_acceptance_property(self, config_dict):
        """Property: Valid configurations should always be accepted.
        
        For any configuration with all valid values, validation should
        pass without errors.
        """
        is_valid, errors = self.config_manager.validate_configuration(config_dict)
        
        # Property: All valid configurations should pass
        assert is_valid, \
            f"Valid configuration should be accepted. Errors: {errors}"
        assert len(errors) == 0, \
            f"Valid configuration should have no errors. Got: {errors}"
    
    @given(
        num_invalid_fields=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=50, deadline=None)
    def test_error_message_clarity_property(self, num_invalid_fields):
        """Property: Error messages should be clear and actionable.
        
        For any invalid configuration, error messages should clearly
        identify the problem and provide guidance.
        """
        # Create a config with intentionally invalid values
        config = {}
        if num_invalid_fields >= 1:
            config['whisper_model_size'] = 'invalid_size'
        if num_invalid_fields >= 2:
            config['max_speed_adjustment'] = 10.0  # Too high
        if num_invalid_fields >= 3:
            config['batch_size'] = 0  # Too low
        if num_invalid_fields >= 4:
            config['min_speed_adjustment'] = 2.0  # Too high
        if num_invalid_fields >= 5:
            config['volume_ducking_level'] = 10.0  # Should be negative
        
        is_valid, errors = self.config_manager.validate_configuration(config)
        
        # Property: Invalid config should be rejected
        assert not is_valid, \
            "Configuration with invalid fields should be rejected"
        
        # Property: Should have at least one error per invalid field
        assert len(errors) >= num_invalid_fields, \
            f"Should have at least {num_invalid_fields} errors, got {len(errors)}"
        
        # Property: Each error should be a non-empty string
        for error in errors:
            assert isinstance(error, str), \
                "Error messages should be strings"
            assert len(error) > 0, \
                "Error messages should not be empty"
            assert len(error) < 500, \
                "Error messages should be concise"

