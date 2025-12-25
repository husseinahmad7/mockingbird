"""File upload component for Video Translator System.

This module provides file upload functionality with drag-and-drop support.
Requirements: 6.2
"""

import streamlit as st
from pathlib import Path
from typing import Optional, Tuple
import tempfile
import os

from src.services.file_handler import FileHandler
from src.services.error_handler import ErrorHandler, ErrorSeverity


class FileUploadComponent:
    """Component for handling file uploads and URL inputs."""
    
    def __init__(self, file_handler: FileHandler, error_handler: ErrorHandler):
        """Initialize the file upload component.
        
        Args:
            file_handler: FileHandler instance for file operations
            error_handler: ErrorHandler instance for error logging
        """
        self.file_handler = file_handler
        self.error_handler = error_handler
    
    def render(self) -> Optional[str]:
        """Render the file upload component.

        Returns:
            Path to uploaded/downloaded file, or None if no file
        """
        st.header("📤 Upload Video")

        # Check if file is already uploaded in session state
        if 'uploaded_file_path' in st.session_state and st.session_state.uploaded_file_path:
            # Show current file info
            st.success(f"✅ File ready: {Path(st.session_state.uploaded_file_path).name}")

            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🗑️ Remove File", use_container_width=True):
                    st.session_state.uploaded_file_path = None
                    st.rerun()

            return st.session_state.uploaded_file_path

        # Create tabs for different input methods
        tab1, tab2 = st.tabs(["📁 File Upload", "🔗 URL"])

        file_path = None
        with tab1:
            file_path = self._render_file_upload()

        with tab2:
            url_file_path = self._render_url_input()
            if url_file_path:
                file_path = url_file_path

        # Store in session state if file was uploaded
        if file_path:
            st.session_state.uploaded_file_path = file_path

        return file_path
    
    def _render_file_upload(self) -> Optional[str]:
        """Render file upload interface.
        
        Returns:
            Path to uploaded file, or None
        """
        st.markdown("""
            Upload a video file to get started. Supported formats:
            - **Video**: MP4, MKV, AVI
            - **Audio**: MP3
            - **Maximum size**: 500 MB
        """)
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['mp4', 'mkv', 'avi', 'mp3'],
            help="Drag and drop a file or click to browse",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Display file info
            file_size_mb = uploaded_file.size / (1024 * 1024)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Name", uploaded_file.name)
            with col2:
                st.metric("File Size", f"{file_size_mb:.2f} MB")
            with col3:
                st.metric("File Type", uploaded_file.type or "Unknown")
            
            # Validate file
            is_valid, error_message = self._validate_uploaded_file(uploaded_file, file_size_mb)
            
            if not is_valid:
                st.error(f"❌ {error_message}")
                self.error_handler.log_error(
                    ValueError(error_message),
                    severity=ErrorSeverity.WARNING,
                    context={'file_name': uploaded_file.name, 'file_size_mb': file_size_mb}
                )
                return None
            
            # Save uploaded file
            try:
                st.success("✅ File validated successfully!")
                
                # Save to temporary location
                temp_file_path = self._save_uploaded_file(uploaded_file)
                
                st.info(f"📁 File saved to: {temp_file_path}")
                
                return temp_file_path
                
            except Exception as e:
                st.error(f"❌ Error saving file: {str(e)}")
                self.error_handler.log_error(
                    e,
                    severity=ErrorSeverity.ERROR,
                    context={'file_name': uploaded_file.name},
                    recovery_suggestion="Try uploading the file again"
                )
                return None
        
        return None
    
    def _render_url_input(self) -> Optional[str]:
        """Render URL input interface.
        
        Returns:
            Path to downloaded file, or None
        """
        st.markdown("""
            Provide a URL to download video content. Supports:
            - YouTube videos
            - Direct video file URLs
            - Other streaming platforms (via yt-dlp)
        """)
        
        url = st.text_input(
            "Enter URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste a video URL here"
        )
        
        if url:
            # Validate URL
            is_valid, error_message = self.file_handler.validate_url(url)
            
            if not is_valid:
                st.error(f"❌ {error_message}")
                return None
            
            # Download button
            if st.button("⬇️ Download Video", use_container_width=True):
                with st.spinner("Downloading video..."):
                    try:
                        file_path = self.file_handler.download_from_url(url)
                        st.success(f"✅ Video downloaded successfully!")
                        st.info(f"📁 File saved to: {file_path}")
                        return file_path
                    except Exception as e:
                        st.error(f"❌ Download failed: {str(e)}")
                        self.error_handler.log_error(
                            e,
                            severity=ErrorSeverity.ERROR,
                            context={'url': url},
                            recovery_suggestion="Check the URL and try again"
                        )
                        return None
        
        return None
    
    def _validate_uploaded_file(self, uploaded_file, file_size_mb: float) -> Tuple[bool, str]:
        """Validate uploaded file.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            file_size_mb: File size in megabytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file size
        if file_size_mb > self.file_handler.MAX_FILE_SIZE_MB:
            return False, f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed size ({self.file_handler.MAX_FILE_SIZE_MB} MB)"
        
        # Check file extension
        file_ext = Path(uploaded_file.name).suffix.lower()
        if file_ext not in [fmt for fmt in self.file_handler.SUPPORTED_FORMATS]:
            return False, f"File format '{file_ext}' is not supported. Supported formats: {', '.join(self.file_handler.SUPPORTED_FORMATS)}"
        
        return True, ""
    
    def _save_uploaded_file(self, uploaded_file) -> str:
        """Save uploaded file to temporary location.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            Path to saved file
        """
        # Create temp file with original extension
        file_ext = Path(uploaded_file.name).suffix
        temp_file = self.file_handler.create_temp_file(file_ext)
        
        # Write uploaded file content
        with open(temp_file, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        return temp_file

