"""Pytest configuration and fixtures for the Video Translator System tests."""

import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_video_file(temp_dir):
    """Create a sample video file for testing."""
    video_path = os.path.join(temp_dir, "sample.mp4")
    # Create a small dummy file
    with open(video_path, 'wb') as f:
        f.write(b'fake video content' * 1000)  # Small file under 500MB
    return video_path


@pytest.fixture
def large_video_file(temp_dir):
    """Create a large video file for testing size limits."""
    video_path = os.path.join(temp_dir, "large.mp4")
    # Create a file larger than 500MB
    file_size = 600 * 1024 * 1024  # 600MB
    with open(video_path, 'wb') as f:
        f.write(b'0' * file_size)
    return video_path


@pytest.fixture
def unsupported_file(temp_dir):
    """Create an unsupported file format for testing."""
    file_path = os.path.join(temp_dir, "document.txt")
    with open(file_path, 'w') as f:
        f.write("This is a text document")
    return file_path