# Video Translator System - Source Code

This directory contains the source code for the Video Translator System.

## Structure

- `models/` - Core data models and types
  - `core.py` - Main data structures (Segment, TranslationJob, ProcessingConfig, etc.)

- `services/` - Service layer implementations
  - `base.py` - Abstract base classes for all services
  - `file_handler.py` - File validation and handling service

- `ui/` - User interface components (Streamlit-based)
  - (To be implemented in future tasks)

## Key Components

### Data Models
- **Segment**: Represents a timestamped text segment with speaker information
- **TranslationJob**: Tracks the complete translation workflow
- **ProcessingConfig**: Configuration settings for the translation pipeline
- **AudioFile**: Metadata for audio files

### Services
- **FileHandler**: Validates uploaded files and handles URL downloads
- **BaseASRService**: Abstract interface for speech recognition
- **BaseTranslationService**: Abstract interface for translation
- **BaseTTSService**: Abstract interface for text-to-speech
- **BaseAudioProcessingService**: Abstract interface for audio processing

## Testing

The system uses property-based testing with Hypothesis to ensure correctness across all valid inputs. Tests are located in the `tests/` directory at the project root.