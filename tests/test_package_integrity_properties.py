"""Property-based tests for package integrity assurance.

**Feature: video-translator, Property 21: Package integrity assurance**
**Validates: Requirements 10.3**

Property 21: Package integrity assurance
For any export package, the system should ensure all files are included,
checksums are correct, and the package can be verified.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from typing import List
import tempfile
from pathlib import Path
import zipfile
import hashlib

from src.services.package_manager import PackageManager


class TestPackageIntegrityProperties:
    """Property-based tests for package integrity assurance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.package_manager = PackageManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def _create_dummy_file(self, filename: str, content: str) -> str:
        """Create a dummy file for testing.
        
        Args:
            filename: Name of the file
            content: Content to write
            
        Returns:
            Path to created file
        """
        file_path = Path(self.temp_dir) / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return str(file_path)
    
    @given(
        num_subtitle_files=st.integers(min_value=1, max_value=10),
        include_checksums=st.booleans()
    )
    @settings(max_examples=50, deadline=None)
    def test_package_contains_all_files_property(self, num_subtitle_files, include_checksums):
        """Property: Package should contain all input files.
        
        For any set of input files, the created package should contain
        all of them.
        """
        # Create dummy files
        video_file = self._create_dummy_file("video.mp4", "dummy video content")
        subtitle_files = [
            self._create_dummy_file(f"subtitle_{i}.srt", f"subtitle content {i}")
            for i in range(num_subtitle_files)
        ]
        
        # Create package
        package_path = Path(self.temp_dir) / "test_package.zip"
        success = self.package_manager.create_package(
            video_file,
            subtitle_files,
            str(package_path),
            include_checksums=include_checksums
        )
        
        assert success, "Package creation should succeed"
        assert package_path.exists(), "Package file should be created"
        
        # Verify contents
        with zipfile.ZipFile(package_path, 'r') as zipf:
            file_list = zipf.namelist()
            
            # Property: Should contain video file
            assert Path(video_file).name in file_list, \
                "Package should contain video file"
            
            # Property: Should contain all subtitle files
            for subtitle_file in subtitle_files:
                assert Path(subtitle_file).name in file_list, \
                    f"Package should contain {Path(subtitle_file).name}"
            
            # Property: Should contain checksums if requested
            if include_checksums:
                assert 'checksums.txt' in file_list, \
                    "Package should contain checksums file"
    
    @given(
        num_subtitle_files=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=50, deadline=None)
    def test_package_integrity_verification_property(self, num_subtitle_files):
        """Property: Created packages should pass integrity verification.
        
        For any created package, the integrity verification should succeed.
        """
        # Create dummy files
        video_file = self._create_dummy_file("video.mp4", "dummy video content")
        subtitle_files = [
            self._create_dummy_file(f"subtitle_{i}.srt", f"subtitle content {i}")
            for i in range(num_subtitle_files)
        ]
        
        # Create package
        package_path = Path(self.temp_dir) / "test_integrity.zip"
        success = self.package_manager.create_package(
            video_file,
            subtitle_files,
            str(package_path),
            include_checksums=True
        )
        
        assert success, "Package creation should succeed"
        
        # Property: Package should pass integrity verification
        is_valid, issues = self.package_manager.verify_package_integrity(str(package_path))
        
        assert is_valid, f"Package should pass integrity verification, issues: {issues}"
        assert len(issues) == 0, "Valid package should have no issues"
    
    @given(
        num_subtitle_files=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=50, deadline=None)
    def test_checksum_correctness_property(self, num_subtitle_files):
        """Property: Checksums should be correct for all files.
        
        For any package with checksums, the checksums should match
        the actual file contents.
        """
        # Create dummy files with known content
        video_file = self._create_dummy_file("video.mp4", "dummy video content")
        subtitle_files = [
            self._create_dummy_file(f"subtitle_{i}.srt", f"subtitle content {i}")
            for i in range(num_subtitle_files)
        ]
        
        # Calculate expected checksums
        expected_checksums = {}
        
        for file_path in [video_file] + subtitle_files:
            sha256_hash = hashlib.sha256()
            with open(file_path, 'rb') as f:
                sha256_hash.update(f.read())
            expected_checksums[Path(file_path).name] = sha256_hash.hexdigest()
        
        # Create package
        package_path = Path(self.temp_dir) / "test_checksums.zip"
        success = self.package_manager.create_package(
            video_file,
            subtitle_files,
            str(package_path),
            include_checksums=True
        )
        
        assert success, "Package creation should succeed"
        
        # Extract and verify checksums
        with zipfile.ZipFile(package_path, 'r') as zipf:
            checksums_content = zipf.read('checksums.txt').decode('utf-8')
            
            # Property: All files should have checksums
            for filename, expected_checksum in expected_checksums.items():
                assert expected_checksum in checksums_content, \
                    f"Checksum for {filename} should be in checksums file"
                assert filename in checksums_content, \
                    f"Filename {filename} should be in checksums file"
    
    @given(
        num_languages=st.integers(min_value=1, max_value=5),
        files_per_language=st.integers(min_value=1, max_value=3)
    )
    @settings(max_examples=50, deadline=None)
    def test_multi_language_package_organization_property(self, num_languages, files_per_language):
        """Property: Multi-language packages should organize files by language.
        
        For any multi-language package, files should be organized in
        language-specific directories.
        """
        # Create dummy files
        video_file = self._create_dummy_file("video.mp4", "dummy video content")
        
        # Create subtitle files for each language
        subtitle_files_by_language = {}
        languages = [f"lang_{i}" for i in range(num_languages)]
        
        for lang in languages:
            subtitle_files_by_language[lang] = [
                self._create_dummy_file(
                    f"{lang}_subtitle_{j}.srt",
                    f"{lang} subtitle content {j}"
                )
                for j in range(files_per_language)
            ]
        
        # Create multi-language package
        package_path = Path(self.temp_dir) / "test_multilang.zip"
        success = self.package_manager.create_multi_language_package(
            video_file,
            subtitle_files_by_language,
            str(package_path)
        )
        
        assert success, "Multi-language package creation should succeed"
        
        # Verify organization
        with zipfile.ZipFile(package_path, 'r') as zipf:
            file_list = zipf.namelist()
            
            # Property: Should contain video file at root
            assert Path(video_file).name in file_list, \
                "Package should contain video file at root"
            
            # Property: Should contain README
            assert 'README.txt' in file_list, \
                "Package should contain README"
            
            # Property: Each language should have its own directory
            for lang in languages:
                lang_files = [f for f in file_list if f.startswith(f"{lang}/")]
                assert len(lang_files) == files_per_language, \
                    f"Language {lang} should have {files_per_language} files"
    
    @given(
        num_subtitle_files=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=50, deadline=None)
    def test_package_is_valid_zip_property(self, num_subtitle_files):
        """Property: Created packages should be valid ZIP files.
        
        For any created package, it should be a valid ZIP archive that
        can be opened and extracted.
        """
        # Create dummy files
        video_file = self._create_dummy_file("video.mp4", "dummy video content")
        subtitle_files = [
            self._create_dummy_file(f"subtitle_{i}.srt", f"subtitle content {i}")
            for i in range(num_subtitle_files)
        ]
        
        # Create package
        package_path = Path(self.temp_dir) / "test_valid_zip.zip"
        success = self.package_manager.create_package(
            video_file,
            subtitle_files,
            str(package_path)
        )
        
        assert success, "Package creation should succeed"
        
        # Property: Should be a valid ZIP file
        assert zipfile.is_zipfile(package_path), \
            "Package should be a valid ZIP file"
        
        # Property: ZIP should be openable
        try:
            with zipfile.ZipFile(package_path, 'r') as zipf:
                # Property: ZIP integrity test should pass
                bad_file = zipf.testzip()
                assert bad_file is None, \
                    f"ZIP integrity test should pass, but found bad file: {bad_file}"
        except zipfile.BadZipFile:
            pytest.fail("Package should be a valid ZIP file")
    
    @given(
        num_subtitle_files=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=50, deadline=None)
    def test_package_file_content_preservation_property(self, num_subtitle_files):
        """Property: Package should preserve file contents exactly.
        
        For any files added to a package, the contents should be
        preserved exactly when extracted.
        """
        # Create dummy files with specific content
        video_content = "dummy video content with special chars: 你好世界"
        video_file = self._create_dummy_file("video.mp4", video_content)
        
        subtitle_contents = [f"subtitle {i} content: 字幕 {i}" for i in range(num_subtitle_files)]
        subtitle_files = [
            self._create_dummy_file(f"subtitle_{i}.srt", content)
            for i, content in enumerate(subtitle_contents)
        ]
        
        # Create package
        package_path = Path(self.temp_dir) / "test_content.zip"
        success = self.package_manager.create_package(
            video_file,
            subtitle_files,
            str(package_path)
        )
        
        assert success, "Package creation should succeed"
        
        # Extract and verify contents
        with zipfile.ZipFile(package_path, 'r') as zipf:
            # Property: Video content should be preserved
            extracted_video = zipf.read(Path(video_file).name).decode('utf-8')
            assert extracted_video == video_content, \
                "Video content should be preserved exactly"
            
            # Property: Subtitle contents should be preserved
            for i, subtitle_file in enumerate(subtitle_files):
                extracted_subtitle = zipf.read(Path(subtitle_file).name).decode('utf-8')
                assert extracted_subtitle == subtitle_contents[i], \
                    f"Subtitle {i} content should be preserved exactly"

