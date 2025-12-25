# Video Translator 🎬

Fast, free, and open-source video translation and dubbing system with support for 40+ languages including Arabic.

## Features

- 🎤 **Automatic Speech Recognition** - Powered by Faster Whisper
- 🌐 **Multi-language Translation** - Support for 40+ languages including Arabic
- 🗣️ **Text-to-Speech Dubbing** - Natural voice synthesis with Edge-TTS
- 👥 **Speaker Detection** - Identify and track different speakers
- 📝 **Subtitle Export** - Generate SRT and ASS subtitle files
- 🎨 **Web Interface** - User-friendly Streamlit interface
- ⚡ **CLI Support** - Command-line interface for batch processing
- 🚀 **GPU Acceleration** - CUDA support for faster processing

## Supported Languages

English, Spanish, French, German, Italian, Portuguese, Russian, **Arabic**, Japanese, Korean, Chinese (Simplified & Traditional), Hindi, Turkish, Dutch, Polish, Swedish, Danish, Finnish, Norwegian, Czech, Greek, Hebrew, Thai, Vietnamese, Indonesian, Malay, Ukrainian, Romanian, Hungarian, Bulgarian, Croatian, Slovak, Slovenian, Lithuanian, Latvian, Estonian, and more!

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/video-translator.git
cd video-translator

# Install dependencies with uv
uv sync
```

### Web Interface

```bash
# Run the Streamlit web app
uv run streamlit run src/ui/app.py
```

Then open your browser to `http://localhost:8501`

### Command Line Interface

```bash
# Translate a video to Spanish
uv run python -m src.cli input.mp4 -o output/ -t es

# Translate to multiple languages including Arabic
uv run python -m src.cli input.mp4 -o output/ -t es fr ar

# Use larger model for better accuracy
uv run python -m src.cli input.mp4 -o output/ -t ar -m large-v3

# Generate only subtitles (no dubbing)
uv run python -m src.cli input.mp4 -o output/ -t es --subtitle-only
```

See [CLI_USAGE.md](CLI_USAGE.md) for detailed CLI documentation.

## Configuration

### API Keys

Set your Gemini API key for translation:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

The system will automatically fall back to local NLLB model if the API is unavailable.

### GPU Support

For GPU acceleration, ensure CUDA is installed:

```bash
# Check GPU availability
nvidia-smi

# Set GPU device
export CUDA_VISIBLE_DEVICES=0
```

## Usage Examples

### Web Interface Workflow

1. **Upload** - Drag and drop a video file or provide a URL
2. **Configure** - Select source and target languages
3. **Transcribe** - Automatic speech recognition with timestamps
4. **Translate** - AI-powered translation with context preservation
5. **Review** - Edit transcriptions and translations
6. **Export** - Download dubbed video and subtitles

### CLI Examples

```bash
# Example 1: Translate English video to Arabic
uv run python -m src.cli english_video.mp4 -o output/ -s en -t ar

# Example 2: Multi-language translation
uv run python -m src.cli video.mp4 -o output/ -t es fr de ar ja ko

# Example 3: High-quality with speaker detection
uv run python -m src.cli meeting.mp4 -o output/ -m large-v3 --speaker-detection -t en

# Example 4: Quick subtitle generation
uv run python -m src.cli video.mp4 -o output/ -m tiny -t es --subtitle-only -f srt
```

## Project Structure

```
video-translator/
├── src/
│   ├── models/          # Data models
│   ├── services/        # Core services (ASR, Translation, TTS)
│   ├── ui/              # Streamlit web interface
│   │   ├── app.py       # Main application
│   │   └── components/  # UI components
│   ├── cli.py           # Command-line interface
│   └── __main__.py      # CLI entry point
├── tests/               # Test suite
├── .kiro/               # Project specifications
└── pyproject.toml       # Project configuration
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_subtitle_export_properties.py

# Run with coverage
uv run pytest --cov=src
```

### Code Quality

```bash
# Format code
uv run black src/

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/
```

## Requirements

- Python 3.11+
- FFmpeg (for audio/video processing)
- CUDA (optional, for GPU acceleration)

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please read CONTRIBUTING.md for guidelines.

## Acknowledgments

- Faster Whisper for ASR
- Google Gemini for translation
- Edge-TTS for voice synthesis
- Streamlit for the web interface
