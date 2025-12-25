"""Result viewer component for displaying and exporting results.

This module provides UI for viewing results and downloading packages.
Requirements: 6.5, 10.1, 10.3, 10.4
"""

import streamlit as st
from pathlib import Path
from typing import List, Optional, Dict
import tempfile

from src.models.core import Segment
from src.services.subtitle_exporter import SubtitleExporter
from src.services.package_manager import PackageManager
from src.services.error_handler import ErrorHandler


class ResultViewer:
    """Component for viewing and exporting translation results."""
    
    def __init__(
        self,
        subtitle_exporter: SubtitleExporter,
        package_manager: PackageManager,
        error_handler: ErrorHandler
    ):
        """Initialize the result viewer.
        
        Args:
            subtitle_exporter: SubtitleExporter instance
            package_manager: PackageManager instance
            error_handler: ErrorHandler instance
        """
        self.subtitle_exporter = subtitle_exporter
        self.package_manager = package_manager
        self.error_handler = error_handler
    
    def render(
        self,
        video_file: str,
        segments: List[Segment],
        target_languages: List[str]
    ):
        """Render the result viewer interface.
        
        Args:
            video_file: Path to processed video file
            segments: List of transcription segments with translations
            target_languages: List of target language codes
        """
        st.header("💾 Export Results")
        
        # Display video preview
        self._render_video_preview(video_file)
        
        st.divider()
        
        # Export options
        self._render_export_options(video_file, segments, target_languages)
    
    def _render_video_preview(self, video_file: str):
        """Render video preview player.
        
        Args:
            video_file: Path to video file
        """
        st.subheader("🎬 Video Preview")
        
        if Path(video_file).exists():
            try:
                # Display video player
                with open(video_file, 'rb') as f:
                    video_bytes = f.read()
                st.video(video_bytes)
                
                # Display file info
                file_size_mb = Path(video_file).stat().st_size / (1024 * 1024)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("File Name", Path(video_file).name)
                
                with col2:
                    st.metric("File Size", f"{file_size_mb:.2f} MB")
                
            except Exception as e:
                st.error(f"❌ Error loading video preview: {str(e)}")
                self.error_handler.log_error(
                    e,
                    context={'video_file': video_file},
                    recovery_suggestion="Video file may be corrupted or in unsupported format"
                )
        else:
            st.warning("⚠️ Video file not found")
    
    def _render_export_options(
        self,
        video_file: str,
        segments: List[Segment],
        target_languages: List[str]
    ):
        """Render export options interface.
        
        Args:
            video_file: Path to video file
            segments: List of segments
            target_languages: List of target languages
        """
        st.subheader("📦 Export Options")
        
        # Export format selection
        col1, col2 = st.columns(2)
        
        with col1:
            export_subtitles = st.checkbox("Export Subtitles", value=True)
            if export_subtitles:
                subtitle_formats = st.multiselect(
                    "Subtitle Formats",
                    options=["SRT", "ASS"],
                    default=["SRT", "ASS"]
                )
        
        with col2:
            create_package = st.checkbox("Create Download Package", value=True)
            if create_package:
                include_checksums = st.checkbox("Include Checksums", value=True)
        
        # Language selection
        if len(target_languages) > 1:
            st.subheader("🌐 Language Selection")
            selected_languages = st.multiselect(
                "Select languages to export",
                options=['original'] + target_languages,
                default=['original'] + target_languages
            )
        else:
            selected_languages = ['original'] + target_languages
        
        st.divider()
        
        # Export button
        if st.button("📥 Generate Export Package", use_container_width=True, type="primary"):
            self._handle_export(
                video_file,
                segments,
                selected_languages,
                subtitle_formats if export_subtitles else [],
                create_package,
                include_checksums if create_package else False
            )
    
    def _handle_export(
        self,
        video_file: str,
        segments: List[Segment],
        languages: List[str],
        subtitle_formats: List[str],
        create_package: bool,
        include_checksums: bool
    ):
        """Handle the export process.
        
        Args:
            video_file: Path to video file
            segments: List of segments
            languages: List of languages to export
            subtitle_formats: List of subtitle formats to export
            create_package: Whether to create a ZIP package
            include_checksums: Whether to include checksums
        """
        with st.spinner("Generating export files..."):
            try:
                # Create temporary directory for exports
                temp_dir = tempfile.mkdtemp()
                base_filename = Path(video_file).stem
                
                subtitle_files = []
                export_results = {}
                
                # Export subtitles for each language
                for language in languages:
                    use_translation = language != 'original'
                    
                    for fmt in subtitle_formats:
                        output_file = Path(temp_dir) / f"{base_filename}_{language}.{fmt.lower()}"
                        
                        if fmt == "SRT":
                            success = self.subtitle_exporter.export_srt(
                                segments,
                                str(output_file),
                                use_translation=use_translation
                            )
                        else:  # ASS
                            success = self.subtitle_exporter.export_ass(
                                segments,
                                str(output_file),
                                use_translation=use_translation
                            )
                        
                        if success:
                            subtitle_files.append(str(output_file))
                            export_results[f"{language}_{fmt}"] = "✅ Success"
                        else:
                            export_results[f"{language}_{fmt}"] = "❌ Failed"
                
                # Display export results
                st.success(f"✅ Exported {len(subtitle_files)} subtitle file(s)")
                
                with st.expander("📊 Export Details"):
                    for key, status in export_results.items():
                        st.text(f"{key}: {status}")
                
                # Create package if requested
                if create_package and subtitle_files:
                    package_path = Path(temp_dir) / f"{base_filename}_package.zip"
                    
                    success = self.package_manager.create_package(
                        video_file,
                        subtitle_files,
                        str(package_path),
                        include_checksums=include_checksums
                    )
                    
                    if success:
                        st.success("✅ Package created successfully!")
                        
                        # Provide download button
                        with open(package_path, 'rb') as f:
                            st.download_button(
                                label="⬇️ Download Package",
                                data=f.read(),
                                file_name=package_path.name,
                                mime="application/zip",
                                use_container_width=True
                            )
                    else:
                        st.error("❌ Failed to create package")
                        self._render_alternative_export(subtitle_files)
                else:
                    # Provide individual file downloads
                    self._render_individual_downloads(subtitle_files)
                
            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")
                self.error_handler.log_error(
                    e,
                    context={'video_file': video_file, 'languages': languages},
                    recovery_suggestion="Try exporting individual files instead of a package"
                )
                self._render_alternative_export(subtitle_files if 'subtitle_files' in locals() else [])
    
    def _render_individual_downloads(self, subtitle_files: List[str]):
        """Render individual file download buttons.
        
        Args:
            subtitle_files: List of subtitle file paths
        """
        st.subheader("📄 Individual File Downloads")
        
        for subtitle_file in subtitle_files:
            if Path(subtitle_file).exists():
                with open(subtitle_file, 'rb') as f:
                    st.download_button(
                        label=f"⬇️ {Path(subtitle_file).name}",
                        data=f.read(),
                        file_name=Path(subtitle_file).name,
                        mime="text/plain",
                        key=f"download_{Path(subtitle_file).name}"
                    )
    
    def _render_alternative_export(self, subtitle_files: List[str]):
        """Render alternative export options when package creation fails.
        
        Args:
            subtitle_files: List of subtitle file paths
        """
        st.warning("⚠️ Package creation failed. You can download individual files below:")
        self._render_individual_downloads(subtitle_files)

