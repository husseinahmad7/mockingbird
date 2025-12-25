#!/bin/bash
# Video Translation Service Setup Script

echo "Setting up Video Translation Service..."

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Check if FFmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "FFmpeg is required. Please install it:"
    echo "Ubuntu/Debian: sudo apt install ffmpeg"
    echo "macOS: brew install ffmpeg"
    echo "Windows: Download from https://ffmpeg.org/"
    exit 1
fi

echo "Setup complete! Run: streamlit run app.py"