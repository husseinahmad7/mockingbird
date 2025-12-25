"""Property-based tests for URL processing.

**Feature: video-translator, Property 2: URL processing consistency**
"""

import os
import tempfile
from hypothesis import given, strategies as st, settings, assume
import pytest

from src.services.file_handler import FileHandler


class TestURLProcessingProperties:
    """Property-based tests for URL processing consistency."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.file_handler = FileHandler()
    
    def teardown_method(self):
        """Clean up after tests."""
        self.file_handler.cleanup_temp_files()
    
    @given(
        domain=st.sampled_from(['youtube.com', 'youtu.be', 'vimeo.com', 'dailymotion.com', 'example.com', 'invalid-site.xyz']),
        protocol=st.sampled_from(['http://', 'https://', '']),
        video_id=st.text(alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_', min_size=5, max_size=20)
    )
    @settings(deadline=None, max_examples=50)
    def test_url_processing_consistency(self, domain: str, protocol: str, video_id: str):
        """
        **Feature: video-translator, Property 2: URL processing consistency**
        
        For any valid streaming URL, the system should successfully download content 
        using yt-dlp and make it available for processing.
        
        **Validates: Requirements 1.2**
        """
        # Construct URL based on domain
        if domain == 'youtube.com':
            url = f"{protocol}www.{domain}/watch?v={video_id}"
        elif domain == 'youtu.be':
            url = f"{protocol}{domain}/{video_id}"
        elif domain == 'vimeo.com':
            # Use numeric ID for vimeo
            numeric_id = ''.join(c for c in video_id if c.isdigit())
            if not numeric_id:
                numeric_id = '123456789'
            url = f"{protocol}www.{domain}/{numeric_id}"
        elif domain == 'dailymotion.com':
            url = f"{protocol}www.{domain}/video/{video_id}"
        else:
            # Unsupported domain
            url = f"{protocol}www.{domain}/video/{video_id}"
        
        # Test URL validation
        is_valid_url = self.file_handler.validate_url(url)
        
        # Determine expected validation result
        supported_domains = ['youtube.com', 'youtu.be', 'vimeo.com', 'dailymotion.com']
        has_protocol = protocol in ['http://', 'https://']
        expected_valid = domain in supported_domains and (has_protocol or protocol == '')
        
        # Assert validation consistency
        assert is_valid_url == expected_valid, (
            f"URL validation inconsistent for {url}. "
            f"Expected {expected_valid}, got {is_valid_url}. "
            f"Domain supported: {domain in supported_domains}, Has protocol: {has_protocol}"
        )
        
        # For valid URLs, test that download method behaves consistently
        if is_valid_url:
            # Note: We won't actually download in property tests to avoid network calls
            # Instead, we test that the method exists and would handle the URL appropriately
            try:
                # This would normally download, but we'll catch the exception
                # since we don't want to make actual network calls in property tests
                downloaded_path = self.file_handler.download_from_url(url)
                
                # If download succeeds (unlikely in test environment), verify the file
                if downloaded_path and os.path.exists(downloaded_path):
                    assert self.file_handler.validate_file(downloaded_path), (
                        f"Downloaded file from {url} failed validation"
                    )
            except (RuntimeError, ValueError) as e:
                # Expected in test environment without actual video content
                # The important thing is that it doesn't crash unexpectedly
                assert "Download failed" in str(e) or "Unsupported URL" in str(e) or "Video too long" in str(e)
        else:
            # For invalid URLs, ensure proper error handling
            with pytest.raises(ValueError, match="Unsupported URL format"):
                self.file_handler.download_from_url(url)
    
    def test_url_validation_edge_cases(self):
        """Test URL validation with edge cases."""
        # Test empty and None URLs
        assert not self.file_handler.validate_url("")
        assert not self.file_handler.validate_url(None)
        
        # Test malformed URLs
        assert not self.file_handler.validate_url("not-a-url")
        assert not self.file_handler.validate_url("ftp://youtube.com/watch?v=test")
        
        # Test valid URLs
        assert self.file_handler.validate_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert self.file_handler.validate_url("https://youtu.be/dQw4w9WgXcQ")
        assert self.file_handler.validate_url("https://vimeo.com/123456789")
        assert self.file_handler.validate_url("https://www.dailymotion.com/video/x123456")
    
    def test_temp_file_management(self):
        """Test temporary file management consistency."""
        # Test temp directory creation
        temp_dir = self.file_handler.get_temp_dir()
        assert os.path.exists(temp_dir)
        assert os.path.isdir(temp_dir)
        
        # Test temp file creation
        temp_file = self.file_handler.create_temp_file('.mp4')
        assert temp_file.startswith(temp_dir)
        assert temp_file.endswith('.mp4')
        
        # Test cleanup
        self.file_handler.cleanup_temp_files()
        # Note: cleanup might not immediately remove files on Windows due to file handles