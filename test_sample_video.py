#!/usr/bin/env python
"""Manual test script for sample video processing.

This script tests the video translation pipeline with the sample video.
Run with: uv run python test_sample_video.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.core import ProcessingConfig
from src.services.file_handler import FileHandler
from src.services.audio_processing import AudioProcessingService


def main():
    """Test the sample video."""
    print("=" * 60)
    print("Video Translator - Sample Video Test")
    print("=" * 60)
    
    # Check if sample video exists
    sample_video = Path("video_sample/Trump_vs._Bane_Inauguration_Speech_144P.mp4")
    if not sample_video.exists():
        print(f"❌ Sample video not found: {sample_video}")
        return 1
    
    print(f"✅ Sample video found: {sample_video}")
    print(f"   Size: {os.path.getsize(sample_video) / (1024*1024):.2f} MB")
    print()
    
    # Test 1: File Validation
    print("Test 1: File Validation")
    print("-" * 60)
    file_handler = FileHandler()
    is_valid = file_handler.validate_file(str(sample_video))

    if is_valid:
        print("✅ File validation passed")

        # Get additional file info
        file_info = file_handler.get_file_info(str(sample_video))
        print(f"   Format: {file_info['format']}")
        print(f"   Size: {file_info['size_mb']:.2f} MB")
        print(f"   Supported: {file_info['is_supported']}")
    else:
        print(f"❌ File validation failed")

        # Check why it failed
        file_size_mb = os.path.getsize(sample_video) / (1024 * 1024)
        file_ext = sample_video.suffix.lower()

        if file_ext not in ['.mp4', '.mkv', '.avi', '.mp3']:
            print(f"   Reason: Unsupported format '{file_ext}'")
        elif file_size_mb > 500:
            print(f"   Reason: File too large ({file_size_mb:.2f} MB > 500 MB)")
        else:
            print(f"   Reason: Unknown")

        return 1
    print()
    
    # Test 2: Audio Extraction
    print("Test 2: Audio Extraction")
    print("-" * 60)
    audio_service = AudioProcessingService()
    
    try:
        audio_path = audio_service.extract_audio(str(sample_video))
        print(f"✅ Audio extracted to: {audio_path}")
        
        # Get audio info
        audio_info = audio_service.get_audio_info(audio_path)
        print(f"   Duration: {audio_info.duration:.2f} seconds")
        print(f"   Sample rate: {audio_info.sample_rate} Hz")
        print(f"   Channels: {audio_info.channels}")
        
        # Cleanup
        if os.path.exists(audio_path):
            os.unlink(audio_path)
            print(f"   Cleaned up: {audio_path}")
        
    except Exception as e:
        print(f"❌ Audio extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        audio_service.cleanup_temp_files()
    
    print()
    
    # Test 3: Check Dependencies
    print("Test 3: Dependency Check")
    print("-" * 60)
    
    dependencies = {
        'ffmpeg-python': ('ffmpeg', False),
        'faster-whisper': ('faster_whisper', False),
        'google.genai': ('google.genai', False),
        'edge-tts': ('edge_tts', False),
        'transformers': ('transformers', False),
    }

    for dep_name, (import_name, _) in dependencies.items():
        try:
            if import_name == 'google.genai':
                from google import genai
            else:
                __import__(import_name)
            dependencies[dep_name] = (import_name, True)
            print(f"✅ {dep_name}")
        except ImportError:
            print(f"❌ {dep_name} (not installed)")
    
    print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"✅ File validation: PASSED")
    print(f"✅ Audio extraction: PASSED")
    
    installed_deps = sum(1 for _, (_, installed) in dependencies.items() if installed)
    total_deps = len(dependencies)
    print(f"📦 Dependencies: {installed_deps}/{total_deps} installed")
    
    if installed_deps < total_deps:
        print()
        print("⚠️  Some dependencies are missing. Install with:")
        print("   uv sync")
    
    print()
    print("✅ Basic tests completed successfully!")
    print()
    print("Next steps:")
    print("1. Ensure all dependencies are installed: uv sync")
    print("2. Set GEMINI_API_KEY environment variable")
    print("3. Run full test suite: uv run pytest tests/ -v")
    print("4. Implement UI layer for complete application")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

