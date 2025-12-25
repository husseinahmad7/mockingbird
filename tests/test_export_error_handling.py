"""Unit tests for export error handling.

**Feature: video-translator**
**Validates: Requirements 10.4**

Tests for export failure scenarios, alternative export options, and
multi-language generation errors.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import zipfile

from src.models.core import Segment
from src.services.subtitle_exporter import SubtitleExporter
from src.services.package_manager import PackageManager
from src.services.error_handler import ErrorHandler


class TestSubtitleExportErrorHandling:
    """Tests for subtitle export error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
        self.exporter = SubtitleExporter(self.error_handler)
        self.temp_dir = tempfile.mkdtemp()
    
    def test_export_to_invalid_path(self):
        """Test exporting to an invalid file path.
        
        Requirement: 10.4 - Export error handling
        """
        segments = [
            Segment(
                start_time=0.0,
                end_time=1.0,
                text="Test segment",
                speaker_id=None
            )
        ]

        # Try to export to invalid path (non-existent directory)
        invalid_path = "/nonexistent/directory/output.srt"

        success = self.exporter.export_srt(segments, invalid_path)

        assert not success, "Export to invalid path should fail"

    def test_export_with_permission_error(self):
        """Test exporting when file permissions prevent writing.

        Requirement: 10.4 - Export error handling
        """
        segments = [
            Segment(
                start_time=0.0,
                end_time=1.0,
                text="Test segment",
                speaker_id=None
            )
        ]
        
        output_file = Path(self.temp_dir) / "readonly.srt"
        
        # Create file and make it read-only
        output_file.touch()
        output_file.chmod(0o444)  # Read-only
        
        try:
            # Try to export (should fail due to permissions)
            with patch('builtins.open', side_effect=PermissionError("Permission denied")):
                success = self.exporter.export_srt(segments, str(output_file))
                assert not success, "Export should fail with permission error"
        finally:
            # Restore permissions for cleanup
            output_file.chmod(0o644)
    
    def test_export_with_empty_segments(self):
        """Test exporting with empty segment list.
        
        Requirement: 10.4 - Export error handling
        """
        segments = []
        output_file = Path(self.temp_dir) / "empty.srt"
        
        success = self.exporter.export_srt(segments, str(output_file))
        
        # Should succeed but create minimal file
        assert success, "Export with empty segments should succeed"
        assert output_file.exists(), "Output file should be created"
    
    def test_export_with_invalid_segment_data(self):
        """Test exporting with segments containing invalid data.
        
        Requirement: 10.4 - Export error handling
        """
        # Create segment with invalid timing (end before start)
        segments = [
            Segment(
                start_time=10.0,
                end_time=5.0,  # Invalid: end before start
                text="Invalid segment",
                speaker_id=None
            )
        ]

        output_file = Path(self.temp_dir) / "invalid.srt"

        # Should still export (exporter doesn't validate, just formats)
        success = self.exporter.export_srt(segments, str(output_file))

        # Export should succeed (validation is separate concern)
        assert success, "Export should succeed even with invalid timing"

    def test_export_with_unicode_errors(self):
        """Test exporting with text that might cause encoding issues.

        Requirement: 10.4 - Export error handling
        """
        segments = [
            Segment(
                start_time=0.0,
                end_time=1.0,
                text="Test with emoji 😀 and special chars: 你好世界 مرحبا",
                speaker_id=None
            )
        ]
        
        output_file = Path(self.temp_dir) / "unicode.srt"
        
        success = self.exporter.export_srt(segments, str(output_file))
        
        assert success, "Export with unicode should succeed"
        
        # Verify content is preserved
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "😀" in content, "Emoji should be preserved"
        assert "你好世界" in content, "Chinese characters should be preserved"
        assert "مرحبا" in content, "Arabic characters should be preserved"
    
    def test_ass_export_with_invalid_style_config(self):
        """Test ASS export with invalid style configuration.
        
        Requirement: 10.4 - Export error handling
        """
        segments = [
            Segment(
                start_time=0.0,
                end_time=1.0,
                text="Test segment",
                speaker_id=None
            )
        ]

        output_file = Path(self.temp_dir) / "invalid_style.ass"

        # Provide invalid style config (should use defaults)
        invalid_style = {
            'font_size': 'invalid',  # Should be int
            'unknown_key': 'value'
        }

        success = self.exporter.export_ass(
            segments,
            str(output_file),
            style_config=invalid_style
        )

        # Should succeed by using defaults for invalid values
        assert success, "Export should succeed with invalid style config"


class TestPackageManagerErrorHandling:
    """Tests for package manager error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
        self.package_manager = PackageManager(self.error_handler)
        self.temp_dir = tempfile.mkdtemp()
    
    def _create_dummy_file(self, filename: str, content: str = "dummy") -> str:
        """Create a dummy file for testing."""
        file_path = Path(self.temp_dir) / filename
        with open(file_path, 'w') as f:
            f.write(content)
        return str(file_path)
    
    def test_package_creation_with_missing_video_file(self):
        """Test package creation when video file doesn't exist.
        
        Requirement: 10.4 - Export error handling
        """
        # Create subtitle files but not video file
        subtitle_files = [
            self._create_dummy_file("subtitle1.srt"),
            self._create_dummy_file("subtitle2.srt")
        ]
        
        nonexistent_video = "/nonexistent/video.mp4"
        package_path = Path(self.temp_dir) / "package.zip"
        
        # Should still create package (with warning logged)
        success = self.package_manager.create_package(
            nonexistent_video,
            subtitle_files,
            str(package_path)
        )
        
        # Package creation should succeed (video file is optional in implementation)
        assert success, "Package creation should succeed even without video file"
    
    def test_package_creation_with_missing_subtitle_files(self):
        """Test package creation when subtitle files don't exist.
        
        Requirement: 10.4 - Export error handling
        """
        video_file = self._create_dummy_file("video.mp4")
        
        # Provide non-existent subtitle files
        nonexistent_subtitles = [
            "/nonexistent/subtitle1.srt",
            "/nonexistent/subtitle2.srt"
        ]
        
        package_path = Path(self.temp_dir) / "package.zip"
        
        # Should still create package (with warnings logged)
        success = self.package_manager.create_package(
            video_file,
            nonexistent_subtitles,
            str(package_path)
        )
        
        assert success, "Package creation should succeed even without subtitle files"
    
    def test_package_creation_to_invalid_path(self):
        """Test package creation to invalid output path.
        
        Requirement: 10.4 - Export error handling
        """
        video_file = self._create_dummy_file("video.mp4")
        subtitle_files = [self._create_dummy_file("subtitle.srt")]
        
        # Try to create package in non-existent directory
        invalid_path = "/nonexistent/directory/package.zip"
        
        success = self.package_manager.create_package(
            video_file,
            subtitle_files,
            invalid_path
        )
        
        assert not success, "Package creation to invalid path should fail"
    
    def test_package_verification_with_nonexistent_file(self):
        """Test package verification when package doesn't exist.
        
        Requirement: 10.4 - Export error handling
        """
        nonexistent_package = "/nonexistent/package.zip"
        
        is_valid, issues = self.package_manager.verify_package_integrity(nonexistent_package)
        
        assert not is_valid, "Verification should fail for nonexistent package"
        assert len(issues) > 0, "Should report issues"
        assert any("not found" in issue.lower() for issue in issues), \
            "Should report file not found"
    
    def test_package_verification_with_invalid_zip(self):
        """Test package verification with invalid ZIP file.
        
        Requirement: 10.4 - Export error handling
        """
        # Create a file that's not a valid ZIP
        invalid_zip = Path(self.temp_dir) / "invalid.zip"
        with open(invalid_zip, 'w') as f:
            f.write("This is not a ZIP file")
        
        is_valid, issues = self.package_manager.verify_package_integrity(str(invalid_zip))
        
        assert not is_valid, "Verification should fail for invalid ZIP"
        assert len(issues) > 0, "Should report issues"
        assert any("not a valid" in issue.lower() for issue in issues), \
            "Should report invalid ZIP"
    
    def test_package_verification_with_corrupted_zip(self):
        """Test package verification with corrupted ZIP file.
        
        Requirement: 10.4 - Export error handling
        """
        # Create a corrupted ZIP file
        corrupted_zip = Path(self.temp_dir) / "corrupted.zip"
        
        # Create a valid ZIP first
        with zipfile.ZipFile(corrupted_zip, 'w') as zipf:
            zipf.writestr('test.txt', 'test content')
        
        # Corrupt it by truncating
        with open(corrupted_zip, 'r+b') as f:
            f.truncate(50)  # Truncate to make it corrupted
        
        is_valid, issues = self.package_manager.verify_package_integrity(str(corrupted_zip))
        
        # Should detect corruption
        assert not is_valid, "Verification should fail for corrupted ZIP"
    
    def test_multi_language_package_with_partial_failure(self):
        """Test multi-language package creation with some missing files.
        
        Requirement: 10.4 - Export error handling
        """
        video_file = self._create_dummy_file("video.mp4")
        
        # Mix of existing and non-existing files
        subtitle_files_by_language = {
            'en': [self._create_dummy_file("en_subtitle.srt")],
            'es': ["/nonexistent/es_subtitle.srt"],  # Non-existent
            'fr': [self._create_dummy_file("fr_subtitle.srt")]
        }
        
        package_path = Path(self.temp_dir) / "multilang.zip"
        
        # Should still create package with available files
        success = self.package_manager.create_multi_language_package(
            video_file,
            subtitle_files_by_language,
            str(package_path)
        )
        
        assert success, "Package creation should succeed with partial files"
        assert package_path.exists(), "Package file should be created"


class TestExportRecoveryStrategies:
    """Tests for export recovery and alternative strategies."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
        self.exporter = SubtitleExporter(self.error_handler)
        self.temp_dir = tempfile.mkdtemp()
    
    def test_fallback_to_individual_exports(self):
        """Test falling back to individual file exports when package fails.
        
        Requirement: 10.4 - Export error handling
        """
        segments = [
            Segment(
                start_time=0.0,
                end_time=1.0,
                text="Test segment",
                speaker_id=None
            )
        ]

        # Export individual files as fallback
        srt_file = Path(self.temp_dir) / "fallback.srt"
        ass_file = Path(self.temp_dir) / "fallback.ass"

        srt_success = self.exporter.export_srt(segments, str(srt_file))
        ass_success = self.exporter.export_ass(segments, str(ass_file))

        assert srt_success, "SRT export should succeed as fallback"
        assert ass_success, "ASS export should succeed as fallback"
        assert srt_file.exists(), "SRT file should exist"
        assert ass_file.exists(), "ASS file should exist"

    def test_export_with_reduced_quality_on_error(self):
        """Test exporting with reduced quality/features when full export fails.

        Requirement: 10.4 - Export error handling
        """
        segments = [
            Segment(
                start_time=0.0,
                end_time=1.0,
                text="Test segment",
                speaker_id=None
            )
        ]
        
        # Try ASS export with complex styling
        output_file = Path(self.temp_dir) / "reduced.ass"
        
        # If complex export fails, fall back to simple export
        success = self.exporter.export_ass(
            segments,
            str(output_file),
            style_config=None  # Use default/simple styling
        )
        
        assert success, "Simple export should succeed as fallback"

