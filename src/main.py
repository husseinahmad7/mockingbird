"""Main entry point for the Video Translator System."""

from .models.core import ProcessingConfig, JobStatus
from .services.file_handler import FileHandler


def main():
    """Main application entry point."""
    print("Video Translator System")
    print("=======================")
    
    # Initialize core components
    file_handler = FileHandler()
    config = ProcessingConfig()
    
    print(f"Supported formats: {', '.join(file_handler.SUPPORTED_FORMATS)}")
    print(f"Maximum file size: {file_handler.MAX_FILE_SIZE_MB}MB")
    print(f"Whisper model: {config.whisper_model_size}")
    print(f"Speaker detection: {config.enable_speaker_detection}")
    
    print("\nSystem initialized successfully!")


if __name__ == "__main__":
    main()