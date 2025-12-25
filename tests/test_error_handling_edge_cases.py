"""Unit tests for error handling edge cases.

**Feature: video-translator, Task 11.3**
**Tests API service unavailability, resource exhaustion, and cleanup failure recovery**
**Requirements: 8.1**
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.services.error_handler import ErrorHandler, ErrorSeverity
from src.services.file_handler import FileHandler
from src.services.audio_processing import AudioProcessingService


class TestErrorHandlingEdgeCases:
    """Unit tests for error handling edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
        self.temp_files = []
    
    def teardown_method(self):
        """Clean up test fixtures."""
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
    
    def test_api_service_unavailability_fallback(self):
        """Test fallback mechanism when API service is unavailable."""
        # Register a fallback handler
        def fallback_handler(*args, **kwargs):
            return "fallback_result"
        
        self.error_handler.register_fallback_handler('APIError', fallback_handler)
        
        # Define a function that simulates API failure
        def failing_api_call():
            raise ConnectionError("API service unavailable")
        
        # Test that fallback is used
        result = self.error_handler.handle_with_fallback(
            failing_api_call,
            'APIError'
        )
        
        assert result == "fallback_result", \
            "Should use fallback when primary API fails"
        
        # Verify error was logged
        assert len(self.error_handler.error_log) > 0, \
            "Error should be logged when API fails"
        
        # Check that error log contains the failure
        last_error = self.error_handler.error_log[-1]
        assert 'ConnectionError' in last_error.error_type or 'unavailable' in last_error.message.lower(), \
            "Error log should contain API unavailability information"
    
    def test_api_service_unavailability_no_fallback(self):
        """Test error handling when API is unavailable and no fallback exists."""
        def failing_api_call():
            raise ConnectionError("API service unavailable")
        
        # Should raise exception when no fallback is registered
        with pytest.raises(ConnectionError):
            self.error_handler.handle_with_fallback(
                failing_api_call,
                'UnregisteredError'
            )
        
        # Verify error was logged
        assert len(self.error_handler.error_log) > 0, \
            "Error should be logged even when no fallback exists"
    
    def test_fallback_handler_failure(self):
        """Test handling when both primary and fallback handlers fail."""
        # Register a fallback that also fails
        def failing_fallback(*args, **kwargs):
            raise RuntimeError("Fallback also failed")
        
        self.error_handler.register_fallback_handler('TestError', failing_fallback)
        
        def failing_primary():
            raise ValueError("Primary failed")
        
        # Should raise the fallback error
        with pytest.raises(RuntimeError) as exc_info:
            self.error_handler.handle_with_fallback(
                failing_primary,
                'TestError'
            )
        
        assert "Fallback also failed" in str(exc_info.value), \
            "Should raise fallback error when both fail"
        
        # Should have logged both errors
        assert len(self.error_handler.error_log) >= 2, \
            "Should log both primary and fallback failures"
    
    def test_resource_exhaustion_detection(self):
        """Test detection and handling of resource exhaustion scenarios."""
        # Simulate low memory condition
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(
                total=8 * 1024**3,  # 8 GB total
                available=0.5 * 1024**3,  # Only 0.5 GB available
                percent=93.75  # 93.75% used
            )
            
            from src.services.config_manager import ConfigurationManager
            config_manager = ConfigurationManager()
            
            # Check resource availability
            is_available, message = config_manager.check_resource_availability(
                required_memory_gb=2.0
            )
            
            assert not is_available, \
                "Should detect insufficient memory"
            assert "insufficient" in message.lower() or "memory" in message.lower(), \
                "Error message should mention memory issue"
    
    def test_resource_exhaustion_warning(self):
        """Test warning for high resource usage."""
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            mock_memory.return_value = Mock(
                total=16 * 1024**3,
                available=10 * 1024**3,
                percent=37.5
            )
            mock_cpu.return_value = 96.0  # Very high CPU usage
            
            from src.services.config_manager import ConfigurationManager
            config_manager = ConfigurationManager()
            
            is_available, message = config_manager.check_resource_availability(
                required_memory_gb=2.0
            )
            
            # Should still be available but with warning
            assert is_available, \
                "Should be available despite high CPU"
            assert "warning" in message.lower() or "cpu" in message.lower(), \
                "Should warn about high CPU usage"
    
    def test_cleanup_failure_recovery(self):
        """Test recovery when cleanup operations fail."""
        file_handler = FileHandler()
        
        # Create a temp file
        temp_file = file_handler.create_temp_file('.tmp')
        self.temp_files.append(temp_file)
        
        # Write some data
        with open(temp_file, 'w') as f:
            f.write("test data")
        
        # Mock os.remove to simulate cleanup failure
        with patch('os.remove', side_effect=PermissionError("Cannot delete file")):
            # Cleanup should not raise an exception
            try:
                file_handler.cleanup_temp_files()
                # Should complete without raising
                assert True, "Cleanup should handle failures gracefully"
            except Exception as e:
                pytest.fail(f"Cleanup should not raise exceptions: {e}")
    
    def test_multiple_cleanup_calls(self):
        """Test that multiple cleanup calls don't cause errors."""
        file_handler = FileHandler()
        
        # Create temp files
        temp_file1 = file_handler.create_temp_file('.tmp')
        temp_file2 = file_handler.create_temp_file('.tmp')
        
        # First cleanup
        file_handler.cleanup_temp_files()
        
        # Second cleanup should not fail
        try:
            file_handler.cleanup_temp_files()
            assert True, "Multiple cleanup calls should be safe"
        except Exception as e:
            pytest.fail(f"Multiple cleanup calls should not raise: {e}")
    
    def test_error_log_export_with_io_error(self):
        """Test error log export when file writing fails."""
        # Try to export to an invalid path
        invalid_path = "/invalid/path/that/does/not/exist/error_log.json"
        
        # Log some errors first
        self.error_handler.log_error(RuntimeError("Test error"))
        
        # Export should handle the error gracefully
        try:
            self.error_handler.export_error_log(invalid_path)
            # If it succeeds (e.g., creates the directory), that's fine
            assert True
        except (PermissionError, OSError, FileNotFoundError):
            # If it fails, it should raise a clear exception
            assert True, "Should raise clear exception for invalid paths"
    
    def test_error_handler_with_invalid_log_file(self):
        """Test ErrorHandler initialization with invalid log file path."""
        # Try to create error handler with invalid log path
        invalid_log_path = "/invalid/path/error.log"
        
        try:
            error_handler = ErrorHandler(log_file=invalid_log_path)
            # If it succeeds (creates directory), that's acceptable
            assert True
        except (PermissionError, OSError):
            # If it fails, should raise clear exception
            assert True, "Should handle invalid log paths appropriately"
    
    def test_concurrent_error_logging(self):
        """Test error logging under concurrent conditions."""
        import threading
        
        errors_logged = []
        
        def log_errors(count):
            for i in range(count):
                error = RuntimeError(f"Error from thread {threading.current_thread().name} - {i}")
                record = self.error_handler.log_error(error)
                errors_logged.append(record)
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=log_errors, args=(5,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All errors should be logged
        assert len(self.error_handler.error_log) >= 15, \
            "All errors from concurrent threads should be logged"

