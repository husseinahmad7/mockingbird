"""Package manager for bundling export files.

This module provides functionality to package video and subtitle files together.
Requirements: 10.3
"""

import zipfile
import shutil
from pathlib import Path
from typing import List, Optional, Dict
import hashlib

from src.services.error_handler import ErrorHandler, ErrorSeverity


class PackageManager:
    """Manager for packaging export files."""
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        """Initialize the package manager.
        
        Args:
            error_handler: Optional error handler for logging
        """
        self.error_handler = error_handler or ErrorHandler()
    
    def create_package(
        self,
        video_file: str,
        subtitle_files: List[str],
        output_path: str,
        include_checksums: bool = True
    ) -> bool:
        """Create a ZIP package with video and subtitle files.
        
        Args:
            video_file: Path to video file
            subtitle_files: List of paths to subtitle files
            output_path: Path to output ZIP file
            include_checksums: Whether to include checksum file
            
        Returns:
            True if package created successfully, False otherwise
        """
        try:
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create ZIP file
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add video file
                if Path(video_file).exists():
                    zipf.write(video_file, Path(video_file).name)
                    self.error_handler.log_info(
                        f"Added video file to package: {Path(video_file).name}"
                    )
                else:
                    self.error_handler.log_warning(
                        f"Video file not found: {video_file}"
                    )
                
                # Add subtitle files
                for subtitle_file in subtitle_files:
                    if Path(subtitle_file).exists():
                        zipf.write(subtitle_file, Path(subtitle_file).name)
                        self.error_handler.log_info(
                            f"Added subtitle file to package: {Path(subtitle_file).name}"
                        )
                    else:
                        self.error_handler.log_warning(
                            f"Subtitle file not found: {subtitle_file}"
                        )
                
                # Add checksums if requested
                if include_checksums:
                    checksums = self._generate_checksums([video_file] + subtitle_files)
                    checksum_content = self._format_checksums(checksums)
                    zipf.writestr('checksums.txt', checksum_content)
                    self.error_handler.log_info("Added checksums to package")
            
            self.error_handler.log_info(
                f"Successfully created package: {output_path}",
                context={
                    'video_file': video_file,
                    'num_subtitles': len(subtitle_files),
                    'package_size': Path(output_path).stat().st_size
                }
            )
            return True
            
        except Exception as e:
            self.error_handler.log_error(
                e,
                severity=ErrorSeverity.ERROR,
                context={'output_path': output_path},
                recovery_suggestion="Check file permissions and disk space"
            )
            return False
    
    def create_multi_language_package(
        self,
        video_file: str,
        subtitle_files_by_language: Dict[str, List[str]],
        output_path: str
    ) -> bool:
        """Create a package with multi-language subtitle files organized by language.
        
        Args:
            video_file: Path to video file
            subtitle_files_by_language: Dictionary mapping language codes to subtitle file lists
            output_path: Path to output ZIP file
            
        Returns:
            True if package created successfully, False otherwise
        """
        try:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add video file
                if Path(video_file).exists():
                    zipf.write(video_file, Path(video_file).name)
                
                # Add subtitle files organized by language
                for language, subtitle_files in subtitle_files_by_language.items():
                    for subtitle_file in subtitle_files:
                        if Path(subtitle_file).exists():
                            # Create language subdirectory in ZIP
                            archive_name = f"{language}/{Path(subtitle_file).name}"
                            zipf.write(subtitle_file, archive_name)
                
                # Add README
                readme_content = self._generate_readme(subtitle_files_by_language)
                zipf.writestr('README.txt', readme_content)
            
            self.error_handler.log_info(
                f"Successfully created multi-language package: {output_path}",
                context={
                    'video_file': video_file,
                    'languages': list(subtitle_files_by_language.keys()),
                    'package_size': Path(output_path).stat().st_size
                }
            )
            return True
            
        except Exception as e:
            self.error_handler.log_error(
                e,
                severity=ErrorSeverity.ERROR,
                context={'output_path': output_path},
                recovery_suggestion="Check file permissions and disk space"
            )
            return False
    
    def _generate_checksums(self, file_paths: List[str]) -> Dict[str, str]:
        """Generate SHA256 checksums for files.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Dictionary mapping filenames to checksums
        """
        checksums = {}
        
        for file_path in file_paths:
            if Path(file_path).exists():
                sha256_hash = hashlib.sha256()
                
                with open(file_path, 'rb') as f:
                    for byte_block in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(byte_block)
                
                checksums[Path(file_path).name] = sha256_hash.hexdigest()
        
        return checksums
    
    def _format_checksums(self, checksums: Dict[str, str]) -> str:
        """Format checksums as text content.
        
        Args:
            checksums: Dictionary of filename to checksum
            
        Returns:
            Formatted checksum text
        """
        lines = ["SHA256 Checksums", "=" * 50, ""]
        
        for filename, checksum in checksums.items():
            lines.append(f"{checksum}  {filename}")
        
        return "\n".join(lines)

    def _generate_readme(self, subtitle_files_by_language: Dict[str, List[str]]) -> str:
        """Generate README content for multi-language package.

        Args:
            subtitle_files_by_language: Dictionary of language to subtitle files

        Returns:
            README text content
        """
        lines = [
            "Video Translation Package",
            "=" * 50,
            "",
            "This package contains the translated video and subtitles in multiple languages.",
            "",
            "Contents:",
            "-" * 50,
            ""
        ]

        # List languages
        lines.append("Available Languages:")
        for language in subtitle_files_by_language.keys():
            num_files = len(subtitle_files_by_language[language])
            lines.append(f"  - {language}: {num_files} subtitle file(s)")

        lines.extend([
            "",
            "File Organization:",
            "-" * 50,
            "",
            "The subtitle files are organized in language-specific folders:",
            ""
        ])

        for language in subtitle_files_by_language.keys():
            lines.append(f"  {language}/")
            for subtitle_file in subtitle_files_by_language[language]:
                lines.append(f"    - {Path(subtitle_file).name}")

        lines.extend([
            "",
            "Usage:",
            "-" * 50,
            "",
            "1. Extract the video file to your desired location",
            "2. Choose the subtitle file for your preferred language",
            "3. Load the subtitle file in your video player",
            "",
            "Supported subtitle formats:",
            "  - SRT (SubRip): Compatible with most video players",
            "  - ASS (Advanced SubStation Alpha): Supports advanced styling",
            "",
            "Generated by Video Translator System",
            "https://github.com/yourusername/video-translator"
        ])

        return "\n".join(lines)

    def verify_package_integrity(self, package_path: str) -> tuple[bool, List[str]]:
        """Verify the integrity of a package.

        Args:
            package_path: Path to ZIP package

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        try:
            if not Path(package_path).exists():
                issues.append(f"Package file not found: {package_path}")
                return False, issues

            # Check if it's a valid ZIP file
            if not zipfile.is_zipfile(package_path):
                issues.append("File is not a valid ZIP archive")
                return False, issues

            # Open and verify contents
            with zipfile.ZipFile(package_path, 'r') as zipf:
                # Test ZIP integrity
                bad_file = zipf.testzip()
                if bad_file:
                    issues.append(f"Corrupted file in archive: {bad_file}")
                    return False, issues

                # Check for required files
                file_list = zipf.namelist()

                # Should have at least one video file
                video_extensions = ['.mp4', '.mkv', '.avi']
                has_video = any(
                    any(f.lower().endswith(ext) for ext in video_extensions)
                    for f in file_list
                )

                if not has_video:
                    issues.append("No video file found in package")

                # Should have at least one subtitle file
                subtitle_extensions = ['.srt', '.ass']
                has_subtitles = any(
                    any(f.lower().endswith(ext) for ext in subtitle_extensions)
                    for f in file_list
                )

                if not has_subtitles:
                    issues.append("No subtitle files found in package")

            if issues:
                return False, issues

            self.error_handler.log_info(
                f"Package integrity verified: {package_path}",
                context={'num_files': len(file_list)}
            )
            return True, []

        except Exception as e:
            self.error_handler.log_error(
                e,
                severity=ErrorSeverity.ERROR,
                context={'package_path': package_path},
                recovery_suggestion="Package may be corrupted, try re-creating it"
            )
            issues.append(f"Error verifying package: {str(e)}")
            return False, issues

