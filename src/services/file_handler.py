"""File handling service implementation."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse
import re
import yt_dlp

from .base import BaseFileHandler


class FileHandler(BaseFileHandler):
    """Implementation of file handling service."""
    
    # Supported file formats and maximum file size
    SUPPORTED_FORMATS = {'.mp4', '.mkv', '.avi', '.mp3'}
    MAX_FILE_SIZE_MB = 500
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    
    # URL patterns for supported streaming services
    # Protocol (https?://) is required for security and clarity
    SUPPORTED_URL_PATTERNS = [
        r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
        r'https?://(?:www\.)?youtu\.be/[\w-]+',
        r'https?://(?:www\.)?vimeo\.com/\d+',
        r'https?://(?:www\.)?dailymotion\.com/video/[\w-]+',
    ]
    
    def __init__(self):
        """Initialize the file handler."""
        self.temp_dir = tempfile.mkdtemp(prefix="video_translator_")
        
    def __del__(self):
        """Clean up temporary directory on destruction."""
        self.cleanup_temp_files()
    
    def validate_file(self, file_path: str) -> bool:
        """
        Validate if file is acceptable for processing.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if file is valid, False otherwise
        """
        if not os.path.exists(file_path):
            return False
            
        # Check file extension
        file_extension = Path(file_path).suffix.lower()
        if file_extension not in self.SUPPORTED_FORMATS:
            return False
            
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > self.MAX_FILE_SIZE_BYTES:
            return False
            
        return True
    
    def validate_url(self, url: str) -> bool:
        """
        Validate if URL is from a supported streaming service.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is supported, False otherwise
        """
        if not url or not isinstance(url, str):
            return False
            
        # Check if URL matches any supported patterns
        for pattern in self.SUPPORTED_URL_PATTERNS:
            if re.match(pattern, url, re.IGNORECASE):
                return True
                
        return False
    
    def get_file_info(self, file_path: str) -> Dict[str, any]:
        """
        Get file information including size, format, duration.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file information
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_stats = os.stat(file_path)
        file_extension = Path(file_path).suffix.lower()
        
        return {
            'path': file_path,
            'size_bytes': file_stats.st_size,
            'size_mb': file_stats.st_size / (1024 * 1024),
            'format': file_extension,
            'is_supported': file_extension in self.SUPPORTED_FORMATS,
            'is_valid_size': file_stats.st_size <= self.MAX_FILE_SIZE_BYTES,
        }
    
    def download_from_url(self, url: str) -> str:
        """
        Download content from URL and return local file path.
        
        Args:
            url: URL to download from
            
        Returns:
            Path to downloaded file
            
        Raises:
            ValueError: If URL is not supported
            RuntimeError: If download fails
        """
        if not self.validate_url(url):
            raise ValueError(f"Unsupported URL format: {url}")
        
        # Configure yt-dlp options
        ydl_opts = {
            'outtmpl': os.path.join(self.temp_dir, '%(title)s.%(ext)s'),
            'format': 'best[ext=mp4]/best[ext=mkv]/best[ext=avi]/best',
            'noplaylist': True,
            'extractaudio': False,
            'quiet': True,  # Suppress output
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info first to get the filename
                info = ydl.extract_info(url, download=False)
                
                # Check if the video is too long (optional safety check)
                duration = info.get('duration', 0)
                if duration and duration > 3600:  # 1 hour limit
                    raise RuntimeError(f"Video too long: {duration} seconds (max 3600)")
                
                # Download the video
                ydl.download([url])
                
                # Find the downloaded file
                title = info.get('title', 'video')
                ext = info.get('ext', 'mp4')
                
                # Clean title for filename
                safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                expected_path = os.path.join(self.temp_dir, f"{safe_title}.{ext}")
                
                # If exact match not found, look for any file in temp dir
                if not os.path.exists(expected_path):
                    temp_files = [f for f in os.listdir(self.temp_dir) 
                                if os.path.isfile(os.path.join(self.temp_dir, f))]
                    if temp_files:
                        expected_path = os.path.join(self.temp_dir, temp_files[-1])  # Get the latest file
                
                if not os.path.exists(expected_path):
                    raise RuntimeError("Downloaded file not found")
                
                # Validate the downloaded file
                if not self.validate_file(expected_path):
                    os.remove(expected_path)
                    raise RuntimeError("Downloaded file failed validation")
                
                return expected_path
                
        except yt_dlp.DownloadError as e:
            raise RuntimeError(f"Download failed: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during download: {str(e)}")
    
    def create_temp_file(self, suffix: str = '.tmp') -> str:
        """
        Create a temporary file and return its path.
        
        Args:
            suffix: File extension/suffix
            
        Returns:
            Path to temporary file
        """
        fd, temp_path = tempfile.mkstemp(suffix=suffix, dir=self.temp_dir)
        os.close(fd)  # Close the file descriptor, but keep the file
        return temp_path
    
    def cleanup_temp_files(self) -> None:
        """Clean up all temporary files and directories."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except (OSError, PermissionError):
                # If cleanup fails, it's not critical
                pass
    
    def get_temp_dir(self) -> str:
        """Get the temporary directory path."""
        return self.temp_dir