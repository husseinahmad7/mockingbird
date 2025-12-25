"""Property-based tests for file validation.

**Feature: video-translator, Property 1: File validation correctness**
"""

import os
import tempfile
from pathlib import Path
from hypothesis import given, strategies as st, settings
import pytest

from src.services.file_handler import FileHandler


class TestFileValidationProperties:
    """Property-based tests for file validation correctness."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.file_handler = FileHandler()
    
    @given(
        file_extension=st.sampled_from(['.mp4', '.mkv', '.avi', '.mp3', '.txt', '.jpg', '.pdf', '.mov']),
        file_size_mb=st.floats(min_value=0.1, max_value=600.0)
    )
    @settings(deadline=None, max_examples=100)
    def test_file_validation_correctness(self, file_extension: str, file_size_mb: float):
        """
        **Feature: video-translator, Property 1: File validation correctness**
        
        For any uploaded file, the system should accept files with valid formats 
        (MP4, MKV, AVI, MP3) under 500MB and reject all others with appropriate error messages.
        
        **Validates: Requirements 1.1, 1.4**
        """
        # Create a temporary file with the specified extension and size
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            # For efficiency, write smaller chunks for large files
            file_size_bytes = int(file_size_mb * 1024 * 1024)
            
            # Write data efficiently - use larger chunks for big files
            chunk_size = min(1024 * 1024, file_size_bytes)  # 1MB chunks max
            remaining = file_size_bytes
            
            while remaining > 0:
                write_size = min(chunk_size, remaining)
                temp_file.write(b'0' * write_size)
                remaining -= write_size
                
            temp_file_path = temp_file.name
        
        try:
            # Test the validation
            is_valid = self.file_handler.validate_file(temp_file_path)
            
            # Determine expected result based on our criteria
            is_supported_format = file_extension.lower() in {'.mp4', '.mkv', '.avi', '.mp3'}
            is_valid_size = file_size_mb <= 500.0
            expected_valid = is_supported_format and is_valid_size
            
            # Assert that the validation result matches our expectations
            assert is_valid == expected_valid, (
                f"File validation failed for {file_extension} file of {file_size_mb:.2f}MB. "
                f"Expected {expected_valid}, got {is_valid}. "
                f"Supported format: {is_supported_format}, Valid size: {is_valid_size}"
            )
            
            # Additional verification through get_file_info
            file_info = self.file_handler.get_file_info(temp_file_path)
            assert file_info['is_supported'] == is_supported_format
            assert file_info['is_valid_size'] == is_valid_size
            assert file_info['format'] == file_extension.lower()
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def test_nonexistent_file_validation(self):
        """Test that validation correctly handles non-existent files."""
        nonexistent_path = "/path/that/does/not/exist.mp4"
        assert not self.file_handler.validate_file(nonexistent_path)
    
    def test_file_info_nonexistent_file(self):
        """Test that get_file_info raises appropriate error for non-existent files."""
        nonexistent_path = "/path/that/does/not/exist.mp4"
        with pytest.raises(FileNotFoundError):
            self.file_handler.get_file_info(nonexistent_path)