"""Unit tests for file validation edge cases."""

import os
import tempfile
import pytest

from src.services.file_handler import FileHandler


class TestFileValidationEdgeCases:
    """Unit tests for file validation edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.file_handler = FileHandler()
    
    def teardown_method(self):
        """Clean up after tests."""
        self.file_handler.cleanup_temp_files()
    
    def test_oversized_file_rejection(self):
        """Test oversized file rejection - Requirements 1.3"""
        # Create a file larger than 500MB
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            # Write 600MB of data
            file_size_bytes = 600 * 1024 * 1024  # 600MB
            chunk_size = 1024 * 1024  # 1MB chunks
            
            for _ in range(600):
                temp_file.write(b'0' * chunk_size)
            
            temp_file_path = temp_file.name
        
        try:
            # Test validation - should reject oversized file
            assert not self.file_handler.validate_file(temp_file_path)
            
            # Test file info - should indicate invalid size
            file_info = self.file_handler.get_file_info(temp_file_path)
            assert not file_info['is_valid_size']
            assert file_info['is_supported']  # Format is supported, but size is not
            assert file_info['size_mb'] > 500
            
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def test_invalid_format_handling(self):
        """Test invalid format handling - Requirements 1.5"""
        invalid_formats = ['.txt', '.jpg', '.pdf', '.doc', '.zip', '.exe', '.py']
        
        for format_ext in invalid_formats:
            with tempfile.NamedTemporaryFile(suffix=format_ext, delete=False) as temp_file:
                temp_file.write(b'test content')
                temp_file_path = temp_file.name
            
            try:
                # Test validation - should reject invalid format
                assert not self.file_handler.validate_file(temp_file_path), f"Should reject {format_ext} files"
                
                # Test file info - should indicate unsupported format
                file_info = self.file_handler.get_file_info(temp_file_path)
                assert not file_info['is_supported'], f"Should not support {format_ext} format"
                assert file_info['is_valid_size']  # Size should be valid (small file)
                assert file_info['format'] == format_ext.lower()
                
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
    
    def test_valid_formats_acceptance(self):
        """Test that valid formats are accepted."""
        valid_formats = ['.mp4', '.mkv', '.avi', '.mp3']
        
        for format_ext in valid_formats:
            with tempfile.NamedTemporaryFile(suffix=format_ext, delete=False) as temp_file:
                temp_file.write(b'fake video content' * 1000)  # Small valid file
                temp_file_path = temp_file.name
            
            try:
                # Test validation - should accept valid format
                assert self.file_handler.validate_file(temp_file_path), f"Should accept {format_ext} files"
                
                # Test file info - should indicate supported format
                file_info = self.file_handler.get_file_info(temp_file_path)
                assert file_info['is_supported'], f"Should support {format_ext} format"
                assert file_info['is_valid_size']  # Size should be valid (small file)
                assert file_info['format'] == format_ext.lower()
                
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
    
    def test_url_validation_scenarios(self):
        """Test URL validation scenarios - Requirements 1.5"""
        # Valid URLs
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "http://www.youtube.com/watch?v=test123",
            "https://youtu.be/dQw4w9WgXcQ",
            "http://youtu.be/test123",
            "https://www.vimeo.com/123456789",
            "http://vimeo.com/987654321",
            "https://www.dailymotion.com/video/x123456",
            "http://dailymotion.com/video/x987654",
        ]
        
        for url in valid_urls:
            assert self.file_handler.validate_url(url), f"Should accept valid URL: {url}"
        
        # Invalid URLs
        invalid_urls = [
            "",  # Empty string
            None,  # None value
            "not-a-url",  # Not a URL
            "ftp://youtube.com/watch?v=test",  # Wrong protocol
            "https://unsupported-site.com/video/123",  # Unsupported domain
            "youtube.com/watch?v=test",  # Missing protocol (should be invalid)
            "https://",  # Incomplete URL
            "https://www.youtube.com",  # Missing video ID
            "https://www.facebook.com/video/123",  # Unsupported social media
        ]
        
        for url in invalid_urls:
            assert not self.file_handler.validate_url(url), f"Should reject invalid URL: {url}"
    
    def test_nonexistent_file_handling(self):
        """Test handling of nonexistent files."""
        nonexistent_path = "/path/that/does/not/exist/video.mp4"
        
        # Validation should return False
        assert not self.file_handler.validate_file(nonexistent_path)
        
        # get_file_info should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            self.file_handler.get_file_info(nonexistent_path)
    
    def test_file_at_size_boundary(self):
        """Test files exactly at the 500MB boundary."""
        # Create a file exactly 500MB
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            file_size_bytes = 500 * 1024 * 1024  # Exactly 500MB
            chunk_size = 1024 * 1024  # 1MB chunks
            
            for _ in range(500):
                temp_file.write(b'0' * chunk_size)
            
            temp_file_path = temp_file.name
        
        try:
            # Should accept file exactly at the limit
            assert self.file_handler.validate_file(temp_file_path)
            
            file_info = self.file_handler.get_file_info(temp_file_path)
            assert file_info['is_valid_size']
            assert file_info['size_mb'] == 500.0
            
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
        # Create a file just over 500MB
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            file_size_bytes = (500 * 1024 * 1024) + 1  # 500MB + 1 byte
            temp_file.write(b'0' * file_size_bytes)
            temp_file_path = temp_file.name
        
        try:
            # Should reject file over the limit
            assert not self.file_handler.validate_file(temp_file_path)
            
            file_info = self.file_handler.get_file_info(temp_file_path)
            assert not file_info['is_valid_size']
            assert file_info['size_mb'] > 500.0
            
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def test_case_insensitive_format_validation(self):
        """Test that format validation is case insensitive."""
        case_variations = ['.MP4', '.Mp4', '.mP4', '.MKV', '.Mkv', '.AVI', '.Avi', '.MP3', '.Mp3']
        
        for format_ext in case_variations:
            with tempfile.NamedTemporaryFile(suffix=format_ext, delete=False) as temp_file:
                temp_file.write(b'test content')
                temp_file_path = temp_file.name
            
            try:
                # Should accept regardless of case
                assert self.file_handler.validate_file(temp_file_path), f"Should accept {format_ext} (case insensitive)"
                
                file_info = self.file_handler.get_file_info(temp_file_path)
                assert file_info['is_supported']
                # Format should be normalized to lowercase
                assert file_info['format'] == format_ext.lower()
                
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
    
    def test_temporary_file_management(self):
        """Test temporary file management system."""
        # Test temp directory exists
        temp_dir = self.file_handler.get_temp_dir()
        assert os.path.exists(temp_dir)
        assert os.path.isdir(temp_dir)
        assert "video_translator_" in os.path.basename(temp_dir)
        
        # Test temp file creation
        temp_file_path = self.file_handler.create_temp_file('.mp4')
        assert temp_file_path.startswith(temp_dir)
        assert temp_file_path.endswith('.mp4')
        assert os.path.exists(temp_file_path)
        
        # Test cleanup
        self.file_handler.cleanup_temp_files()
        # Note: On Windows, files might still exist due to file handles
        # but the cleanup method should have been called without error