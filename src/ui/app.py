"""Main Streamlit application for Video Translator System.

This module provides the web interface for video translation and dubbing.
Requirements: 6.1, 6.2, 6.4
"""

import streamlit as st
from pathlib import Path
from typing import Optional, Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.services.file_handler import FileHandler
from src.services.error_handler import ErrorHandler
from src.models.core import ProcessingConfig, JobStatus
from src.ui.components.file_upload import FileUploadComponent
from src.ui.processing import VideoProcessor


class SessionState:
    """Manages Streamlit session state for the application."""
    
    @staticmethod
    def initialize():
        """Initialize session state variables."""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.current_job = None
            st.session_state.job_status = JobStatus.PENDING
            st.session_state.uploaded_file_path = None
            st.session_state.processing_config = ProcessingConfig()
            st.session_state.transcription_segments = []
            st.session_state.translation_segments = []
            st.session_state.current_step = 'upload'  # upload, transcribe, translate, tts, review, export
            st.session_state.error_handler = ErrorHandler()
            st.session_state.file_handler = FileHandler()
            st.session_state.video_processor = None
            st.session_state.progress = 0.0
            st.session_state.status_message = "Ready to process video"
            st.session_state.source_language = "auto"
            st.session_state.target_language = "en"
            st.session_state.subtitle_path = None
    
    @staticmethod
    def reset():
        """Reset session state for a new job."""
        st.session_state.current_job = None
        st.session_state.job_status = JobStatus.PENDING
        st.session_state.uploaded_file_path = None
        st.session_state.transcription_segments = []
        st.session_state.translation_segments = []
        st.session_state.current_step = 'upload'
        st.session_state.progress = 0.0
        st.session_state.status_message = "Ready to process video"
        # Clean up temporary files
        if st.session_state.file_handler:
            st.session_state.file_handler.cleanup_temp_files()


def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Video Translator",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/yourusername/video-translator',
            'Report a bug': 'https://github.com/yourusername/video-translator/issues',
            'About': '# Video Translator\nFast, free, and open-source video translation and dubbing.'
        }
    )


def apply_custom_css():
    """Apply custom CSS for better UI appearance."""
    st.markdown("""
        <style>
        /* Main container styling */
        .main {
            padding: 2rem;
        }
        
        /* Progress bar styling */
        .stProgress > div > div > div > div {
            background-color: #4CAF50;
        }
        
        /* File uploader styling */
        .uploadedFile {
            border: 2px dashed #4CAF50;
            border-radius: 10px;
            padding: 20px;
        }
        
        /* Status message styling */
        .status-message {
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        
        .status-success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .status-error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .status-info {
            background-color: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }
        
        /* Step indicator styling */
        .step-indicator {
            display: flex;
            justify-content: space-between;
            margin: 2rem 0;
        }
        
        .step {
            flex: 1;
            text-align: center;
            padding: 1rem;
            border-radius: 5px;
            margin: 0 0.5rem;
        }
        
        .step-active {
            background-color: #4CAF50;
            color: white;
        }
        
        .step-completed {
            background-color: #8BC34A;
            color: white;
        }
        
        .step-pending {
            background-color: #f0f0f0;
            color: #666;
        }
        
        /* Button styling */
        .stButton > button {
            width: 100%;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            padding: 2rem 1rem;
        }
        </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render the application header."""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.title("🎬 Video Translator")
        st.markdown("*Fast, free, and open-source video translation and dubbing*")

    with col2:
        # Theme toggle (Streamlit handles this natively)
        st.markdown("###")  # Spacing


def render_step_indicator(current_step: str):
    """Render the processing step indicator.

    Args:
        current_step: Current processing step
    """
    steps = ['upload', 'transcribe', 'translate', 'export']
    step_labels = {
        'upload': '📤 Upload',
        'transcribe': '🎤 Transcribe',
        'translate': '🌐 Translate',
        'export': '💾 Export'
    }

    cols = st.columns(len(steps))

    for idx, (step, col) in enumerate(zip(steps, cols)):
        with col:
            # Determine step status
            current_idx = steps.index(current_step) if current_step in steps else 0

            if idx < current_idx:
                status = "completed"
                icon = "✅"
            elif idx == current_idx:
                status = "active"
                icon = "▶️"
            else:
                status = "pending"
                icon = "⏸️"

            # Render step
            if status == "active":
                st.markdown(f"**{icon} {step_labels[step]}**")
            elif status == "completed":
                st.markdown(f"{icon} {step_labels[step]}")
            else:
                st.markdown(f"<span style='color: #999;'>{icon} {step_labels[step]}</span>",
                          unsafe_allow_html=True)


def render_progress_bar(progress: float, status_message: str):
    """Render progress bar with status message.

    Args:
        progress: Progress value (0.0 to 1.0)
        status_message: Status message to display
    """
    st.progress(progress)
    st.info(f"📊 {status_message}")


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Language selection
        st.subheader("Languages")

        # Language options with more languages including Arabic
        language_options = {
            "auto": "Auto-detect",
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ar": "Arabic",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese (Simplified)",
            "zh-TW": "Chinese (Traditional)",
            "hi": "Hindi",
            "tr": "Turkish",
            "nl": "Dutch",
            "pl": "Polish",
            "sv": "Swedish",
            "da": "Danish",
            "fi": "Finnish",
            "no": "Norwegian",
            "cs": "Czech",
            "el": "Greek",
            "he": "Hebrew",
            "th": "Thai",
            "vi": "Vietnamese",
            "id": "Indonesian",
            "ms": "Malay",
            "uk": "Ukrainian",
            "ro": "Romanian",
            "hu": "Hungarian",
            "bg": "Bulgarian",
            "hr": "Croatian",
            "sk": "Slovak",
            "sl": "Slovenian",
            "lt": "Lithuanian",
            "lv": "Latvian",
            "et": "Estonian"
        }

        source_lang_options = ["auto"] + [k for k in language_options.keys() if k != "auto"]
        target_lang_options = [k for k in language_options.keys() if k != "auto"]

        source_lang = st.selectbox(
            "Source Language",
            options=source_lang_options,
            format_func=lambda x: language_options[x],
            index=0,
            help="Select 'auto' for automatic detection"
        )

        target_lang = st.selectbox(
            "Target Language",
            options=target_lang_options,
            format_func=lambda x: language_options[x],
            index=0,
            help="Language to translate to"
        )

        # Model configuration
        st.subheader("Model Settings")

        whisper_model = st.selectbox(
            "Whisper Model Size",
            options=["tiny", "base", "small", "medium", "large-v3"],
            index=2,
            help="Larger models are more accurate but slower"
        )

        enable_speaker_detection = st.checkbox(
            "Enable Speaker Detection",
            value=False,
            help="Attempt to identify different speakers"
        )

        # Audio settings
        st.subheader("Audio Settings")

        enable_volume_ducking = st.checkbox(
            "Enable Volume Ducking",
            value=True,
            help="Reduce background audio during speech"
        )

        ducking_amount = st.slider(
            "Ducking Amount (dB)",
            min_value=-30,
            max_value=0,
            value=-15,
            help="How much to reduce background audio"
        )

        # Update session state config
        if st.session_state.processing_config:
            st.session_state.processing_config.whisper_model_size = whisper_model
            st.session_state.processing_config.enable_speaker_detection = enable_speaker_detection
            st.session_state.source_language = source_lang
            st.session_state.target_language = target_lang

        st.divider()

        # System info
        st.subheader("ℹ️ System Info")

        try:
            from src.services.config_manager import ConfigurationManager
            config_manager = ConfigurationManager()
            hw_info = config_manager.get_hardware_info()

            st.text(f"GPU: {hw_info.gpu_name or 'Not available'}")
            st.text(f"CPU Cores: {hw_info.cpu_count}")
            st.text(f"RAM: {hw_info.total_memory_gb:.1f} GB")
        except Exception as e:
            st.text("System info unavailable")

        st.divider()

        # Reset button
        if st.button("🔄 Reset Session", use_container_width=True):
            SessionState.reset()
            st.rerun()


def main():
    """Main application entry point."""
    # Configure page
    configure_page()

    # Initialize session state
    SessionState.initialize()

    # Apply custom CSS
    apply_custom_css()

    # Render header
    render_header()

    # Render sidebar
    render_sidebar()

    # Render step indicator
    st.divider()
    render_step_indicator(st.session_state.current_step)
    st.divider()

    # Render progress bar if processing
    if st.session_state.progress > 0:
        render_progress_bar(st.session_state.progress, st.session_state.status_message)

    # Main content area
    if st.session_state.current_step == 'upload':
        render_upload_step()
    elif st.session_state.current_step == 'transcribe':
        render_transcribe_step()
    elif st.session_state.current_step == 'translate':
        render_translate_step()
    elif st.session_state.current_step == 'export':
        render_export_step()


def render_upload_step():
    """Render the file upload step."""
    st.info("👋 Welcome! Upload a video file or provide a URL to get started.")

    # Create file upload component
    file_upload = FileUploadComponent(
        st.session_state.file_handler,
        st.session_state.error_handler
    )

    # Render file upload
    file_path = file_upload.render()

    # If file is uploaded, show next step button
    if file_path:
        st.session_state.uploaded_file_path = file_path

        st.divider()

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("▶️ Start Processing", use_container_width=True, type="primary"):
                st.session_state.current_step = 'transcribe'
                st.session_state.progress = 0.1
                st.session_state.status_message = "Starting transcription..."
                st.rerun()


def render_transcribe_step():
    """Render the transcription step."""
    st.header("🎤 Transcription")

    # Check if we have a file to process
    if not st.session_state.uploaded_file_path:
        st.error("No file uploaded. Please go back and upload a file.")
        if st.button("⬅️ Back to Upload"):
            st.session_state.current_step = 'upload'
            st.rerun()
        return

    # Show file info
    st.info(f"📁 Processing: {Path(st.session_state.uploaded_file_path).name}")

    # Check if already transcribed
    if st.session_state.transcription_segments:
        st.success(f"✅ Transcription complete! Found {len(st.session_state.transcription_segments)} segments")

        # Show preview
        with st.expander("Preview Transcription", expanded=True):
            for i, segment in enumerate(st.session_state.transcription_segments[:5]):
                st.text(f"[{segment.start_time:.2f}s - {segment.end_time:.2f}s] {segment.text}")
            if len(st.session_state.transcription_segments) > 5:
                st.text(f"... and {len(st.session_state.transcription_segments) - 5} more segments")

        # Next step button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("▶️ Continue to Translation", use_container_width=True, type="primary"):
                st.session_state.current_step = 'translate'
                st.session_state.progress = 0.4
                st.rerun()
        return

    # Start transcription button
    st.write("Click the button below to start transcription:")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎤 Start Transcription", use_container_width=True, type="primary"):
            # Initialize video processor
            if not st.session_state.video_processor:
                st.session_state.video_processor = VideoProcessor(st.session_state.processing_config)

            # Create progress placeholder
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(progress: float, message: str):
                progress_bar.progress(progress)
                status_text.info(message)

            try:
                # Run transcription
                segments = st.session_state.video_processor.transcribe_video(
                    st.session_state.uploaded_file_path,
                    st.session_state.source_language,
                    update_progress
                )

                # Store results
                st.session_state.transcription_segments = segments
                st.session_state.progress = 0.3

                st.success(f"✅ Transcription complete! Found {len(segments)} segments")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Transcription failed: {str(e)}")
                st.session_state.error_handler.log_error(e)


def render_translate_step():
    """Render the translation step."""
    st.header("🌐 Translation")

    # Check if we have transcription
    if not st.session_state.transcription_segments:
        st.error("No transcription available. Please complete transcription first.")
        if st.button("⬅️ Back to Transcription"):
            st.session_state.current_step = 'transcribe'
            st.rerun()
        return

    # Show translation info
    st.info(f"🌍 Translating from {st.session_state.source_language} to {st.session_state.target_language}")

    # Check if already translated
    if st.session_state.translation_segments:
        st.success(f"✅ Translation complete! Translated {len(st.session_state.translation_segments)} segments")

        # Show preview
        with st.expander("Preview Translation", expanded=True):
            for i, (orig, trans) in enumerate(zip(
                st.session_state.transcription_segments[:5],
                st.session_state.translation_segments[:5]
            )):
                col1, col2 = st.columns(2)
                with col1:
                    st.text(f"Original:\n{orig.text}")
                with col2:
                    st.text(f"Translated:\n{trans.text}")
                st.divider()
            if len(st.session_state.translation_segments) > 5:
                st.text(f"... and {len(st.session_state.translation_segments) - 5} more segments")

        # Next step button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("▶️ Continue to Export", use_container_width=True, type="primary"):
                st.session_state.current_step = 'export'
                st.session_state.progress = 0.7
                st.rerun()
        return

    # Start translation button
    st.write(f"Click the button below to translate {len(st.session_state.transcription_segments)} segments:")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🌐 Start Translation", use_container_width=True, type="primary"):
            # Initialize video processor
            if not st.session_state.video_processor:
                st.session_state.video_processor = VideoProcessor(st.session_state.processing_config)

            # Create progress placeholder
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(progress: float, message: str):
                progress_bar.progress(progress)
                status_text.info(message)

            try:
                # Run translation
                translated_segments = st.session_state.video_processor.translate_segments(
                    st.session_state.transcription_segments,
                    st.session_state.target_language,
                    update_progress
                )

                # Store results
                st.session_state.translation_segments = translated_segments
                st.session_state.progress = 0.6

                st.success(f"✅ Translation complete! Translated {len(translated_segments)} segments")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Translation failed: {str(e)}")
                st.session_state.error_handler.log_error(e)


def render_export_step():
    """Render the export step."""
    st.header("💾 Export Results")

    # Check if we have translation
    if not st.session_state.translation_segments:
        st.error("No translation available. Please complete translation first.")
        if st.button("⬅️ Back to Translation"):
            st.session_state.current_step = 'translate'
            st.rerun()
        return

    st.success(f"✅ Processing complete! Ready to export {len(st.session_state.translation_segments)} segments")

    # Export options
    st.subheader("Export Options")

    # Export type selection
    export_type = st.radio(
        "Export Type",
        options=["Subtitles Only", "Dubbed Video"],
        help="Choose whether to export subtitles or create a dubbed video"
    )

    col1, col2 = st.columns(2)

    with col1:
        if export_type == "Subtitles Only":
            export_format = st.selectbox(
                "Subtitle Format",
                options=["SRT", "ASS"],
                help="Choose subtitle format"
            )
        else:
            st.info("Video will be exported as MP4 with dubbed audio")
            export_format = "MP4"

    with col2:
        output_filename = st.text_input(
            "Output Filename",
            value=f"translated_{st.session_state.target_language}",
            help="Filename without extension"
        )

    # Dubbing options (only for dubbed video)
    if export_type == "Dubbed Video":
        st.subheader("Dubbing Options")

        col3, col4 = st.columns(2)

        with col3:
            ducking_level = st.slider(
                "Background Audio Level",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Volume level of background audio during speech (lower = quieter)"
            )

        with col4:
            voice_selection = st.selectbox(
                "Voice",
                options=["Auto-select", "Custom"],
                help="Voice for text-to-speech"
            )

            if voice_selection == "Custom":
                custom_voice = st.text_input(
                    "Voice ID",
                    value="",
                    help="Enter Edge-TTS voice ID (e.g., ar-SA-ZariyahNeural)"
                )
            else:
                custom_voice = None

    # Export button
    button_label = "📥 Export Subtitles" if export_type == "Subtitles Only" else "🎬 Create Dubbed Video"

    if st.button(button_label, use_container_width=True, type="primary"):
        # Initialize video processor
        if not st.session_state.video_processor:
            st.session_state.video_processor = VideoProcessor(st.session_state.processing_config)

        # Create progress placeholder
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(progress: float, message: str):
            progress_bar.progress(progress)
            status_text.info(message)

        try:
            # Create output path
            import tempfile
            output_dir = Path(tempfile.gettempdir()) / "video_translator_output"
            output_dir.mkdir(exist_ok=True)

            if export_type == "Subtitles Only":
                # Export subtitles
                extension = export_format.lower()
                output_path = str(output_dir / f"{output_filename}.{extension}")

                subtitle_path = st.session_state.video_processor.export_subtitles(
                    st.session_state.translation_segments,
                    output_path,
                    export_format.lower(),
                    update_progress
                )

                st.session_state.subtitle_path = subtitle_path
                st.session_state.progress = 1.0

                st.success(f"✅ Subtitles exported successfully!")

                # Provide download button
                with open(subtitle_path, 'r', encoding='utf-8') as f:
                    subtitle_content = f.read()

                st.download_button(
                    label=f"⬇️ Download {export_format} Subtitles",
                    data=subtitle_content,
                    file_name=f"{output_filename}.{extension}",
                    mime="text/plain",
                    use_container_width=True
                )

            else:
                # Create dubbed video
                if not st.session_state.uploaded_file_path:
                    st.error("Original video file not found. Please start over.")
                    return

                output_path = str(output_dir / f"{output_filename}.mp4")

                dubbed_path = st.session_state.video_processor.create_dubbed_video(
                    video_path=st.session_state.uploaded_file_path,
                    translated_segments=st.session_state.translation_segments,
                    output_path=output_path,
                    voice=custom_voice if voice_selection == "Custom" else None,
                    ducking_level=ducking_level,
                    progress_callback=update_progress
                )

                st.session_state.progress = 1.0
                st.success(f"✅ Dubbed video created successfully!")

                # Provide download button
                with open(dubbed_path, 'rb') as f:
                    video_content = f.read()

                st.download_button(
                    label="⬇️ Download Dubbed Video",
                    data=video_content,
                    file_name=f"{output_filename}.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")
            st.session_state.error_handler.log_error(e)

    st.divider()

    # Start new job button
    if st.button("🔄 Start New Job", use_container_width=True):
        SessionState.reset()
        st.rerun()


if __name__ == "__main__":
    main()

