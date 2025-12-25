"""Test that all app imports work correctly."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all critical imports."""
    errors = []
    
    # Test core models
    try:
        from src.models.core import ProcessingConfig, Segment, JobStatus
        print("✅ Core models imported successfully")
    except Exception as e:
        errors.append(f"Core models: {e}")
        print(f"❌ Core models failed: {e}")
    
    # Test services
    try:
        from src.services.asr_service import ASRService
        print("✅ ASR service imported successfully")
    except Exception as e:
        errors.append(f"ASR service: {e}")
        print(f"❌ ASR service failed: {e}")
    
    try:
        from src.services.translation_service import TranslationService
        print("✅ Translation service imported successfully")
    except Exception as e:
        errors.append(f"Translation service: {e}")
        print(f"❌ Translation service failed: {e}")
    
    try:
        from src.services.subtitle_exporter import SubtitleExporter
        print("✅ Subtitle exporter imported successfully")
    except Exception as e:
        errors.append(f"Subtitle exporter: {e}")
        print(f"❌ Subtitle exporter failed: {e}")
    
    # Test UI processing
    try:
        from src.ui.processing import VideoProcessor
        print("✅ Video processor imported successfully")
    except Exception as e:
        errors.append(f"Video processor: {e}")
        print(f"❌ Video processor failed: {e}")
    
    # Test CLI
    try:
        from src.cli import VideoTranslatorCLI
        print("✅ CLI imported successfully")
    except Exception as e:
        errors.append(f"CLI: {e}")
        print(f"❌ CLI failed: {e}")
    
    if errors:
        print(f"\n⚠️ {len(errors)} import errors found")
        return False
    else:
        print("\n🎉 All imports successful!")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

