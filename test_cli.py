"""Quick test script for CLI functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_cli_import():
    """Test that CLI can be imported."""
    try:
        from src.cli import VideoTranslatorCLI, main
        print("✅ CLI import successful")
        return True
    except Exception as e:
        print(f"❌ CLI import failed: {e}")
        return False

def test_cli_help():
    """Test CLI help message."""
    try:
        import subprocess
        result = subprocess.run(
            ["python", "-m", "src.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if "Video Translator" in result.stdout:
            print("✅ CLI help message works")
            return True
        else:
            print("❌ CLI help message not found")
            return False
    except Exception as e:
        print(f"❌ CLI help test failed: {e}")
        return False

def test_language_support():
    """Test that language options are available."""
    try:
        from src.ui.app import render_sidebar
        print("✅ Language support module loaded")
        return True
    except Exception as e:
        print(f"❌ Language support test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Video Translator CLI...\n")
    
    tests = [
        ("CLI Import", test_cli_import),
        ("Language Support", test_language_support),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nRunning: {name}")
        result = test_func()
        results.append((name, result))
    
    print("\n" + "="*50)
    print("Test Results:")
    print("="*50)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

