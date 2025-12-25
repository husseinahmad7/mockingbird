"""Property-based tests for error logging completeness.

**Feature: video-translator, Property 13: Error logging completeness**
**Validates: Requirements 8.2**

Property 13: Error logging completeness
For any processing failure, the system should generate detailed error logs
with sufficient information for troubleshooting and recovery.
"""

import pytest
import tempfile
import os
from hypothesis import given, strategies as st, assume, settings
from datetime import datetime

from src.services.error_handler import ErrorHandler, ErrorSeverity


class TestErrorLoggingProperties:
    """Property-based tests for error logging completeness."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_log_file = tempfile.NamedTemporaryFile(suffix='.log', delete=False)
        self.temp_log_file.close()
        self.error_handler = ErrorHandler(log_file=self.temp_log_file.name)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_log_file.name):
            try:
                os.unlink(self.temp_log_file.name)
            except:
                pass
    
    @given(
        error_message=st.text(min_size=1, max_size=500),
        severity=st.sampled_from(list(ErrorSeverity)),
    )
    @settings(max_examples=100, deadline=None)
    def test_error_logging_creates_record_property(self, error_message, severity):
        """Property: Every logged error should create a complete error record.
        
        For any error that is logged, a complete error record should be created
        with all necessary information for troubleshooting.
        """
        # Create and log an error
        error = RuntimeError(error_message)
        
        initial_count = len(self.error_handler.error_log)
        record = self.error_handler.log_error(error, severity=severity)
        
        # Property: Error log should increase by 1
        assert len(self.error_handler.error_log) == initial_count + 1, \
            "Error log should increase by 1 for each logged error"
        
        # Property: Record should have all required fields
        assert record.timestamp is not None, \
            "Error record should have a timestamp"
        assert isinstance(record.timestamp, datetime), \
            "Timestamp should be a datetime object"
        
        assert record.severity == severity, \
            "Error record should preserve severity level"
        
        assert record.error_type == 'RuntimeError', \
            "Error record should capture error type"
        
        assert record.message == error_message, \
            "Error record should preserve error message"
        
        # Property: Record should be retrievable from log
        assert record in self.error_handler.error_log, \
            "Error record should be in the error log"
    
    @given(
        num_errors=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=50, deadline=None)
    def test_error_log_completeness_property(self, num_errors):
        """Property: Error log should contain all logged errors.
        
        For any number of errors logged, all errors should be preserved
        in the error log in chronological order.
        """
        # Log multiple errors
        logged_errors = []
        for i in range(num_errors):
            error = ValueError(f"Error {i}")
            record = self.error_handler.log_error(error)
            logged_errors.append(record)
        
        # Property: All errors should be in the log
        assert len(self.error_handler.error_log) >= num_errors, \
            f"Error log should contain at least {num_errors} errors"
        
        # Property: Errors should be in chronological order
        for i in range(len(logged_errors) - 1):
            assert logged_errors[i].timestamp <= logged_errors[i + 1].timestamp, \
                "Errors should be logged in chronological order"
        
        # Property: All logged errors should be retrievable
        for logged_error in logged_errors:
            assert logged_error in self.error_handler.error_log, \
                "All logged errors should be in the error log"
    
    @given(
        context_data=st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.one_of(st.text(), st.integers(), st.floats(), st.booleans()),
            min_size=0,
            max_size=10
        ),
        recovery_suggestion=st.one_of(st.none(), st.text(min_size=1, max_size=200))
    )
    @settings(max_examples=100, deadline=None)
    def test_error_context_preservation_property(self, context_data, recovery_suggestion):
        """Property: Error context and recovery suggestions should be preserved.
        
        For any error with context and recovery suggestions, all information
        should be preserved in the error record.
        """
        error = RuntimeError("Test error")
        record = self.error_handler.log_error(
            error,
            context=context_data,
            recovery_suggestion=recovery_suggestion
        )
        
        # Property: Context should be preserved
        assert record.context == context_data, \
            "Error context should be preserved exactly"
        
        # Property: Recovery suggestion should be preserved
        assert record.recovery_suggestion == recovery_suggestion, \
            "Recovery suggestion should be preserved exactly"
    
    @given(
        num_errors=st.integers(min_value=5, max_value=30),
    )
    @settings(max_examples=50, deadline=None)
    def test_error_summary_accuracy_property(self, num_errors):
        """Property: Error summary should accurately reflect logged errors.
        
        For any set of logged errors, the error summary should provide
        accurate statistics and information.
        """
        # Log errors with different severities and types
        error_types = ['ValueError', 'RuntimeError', 'TypeError', 'KeyError']
        severities = list(ErrorSeverity)
        
        for i in range(num_errors):
            error_type = error_types[i % len(error_types)]
            severity = severities[i % len(severities)]
            
            if error_type == 'ValueError':
                error = ValueError(f"Error {i}")
            elif error_type == 'RuntimeError':
                error = RuntimeError(f"Error {i}")
            elif error_type == 'TypeError':
                error = TypeError(f"Error {i}")
            else:
                error = KeyError(f"Error {i}")
            
            self.error_handler.log_error(error, severity=severity)
        
        summary = self.error_handler.get_error_summary()
        
        # Property: Total count should match
        assert summary['total_errors'] >= num_errors, \
            "Error summary should reflect total error count"
        
        # Property: Summary should have required keys
        assert 'by_severity' in summary, \
            "Summary should include errors by severity"
        assert 'by_type' in summary, \
            "Summary should include errors by type"
        assert 'recent_errors' in summary, \
            "Summary should include recent errors"
        
        # Property: Recent errors should be limited
        assert len(summary['recent_errors']) <= 10, \
            "Recent errors should be limited to last 10"
        
        # Property: Severity counts should sum to total
        severity_sum = sum(summary['by_severity'].values())
        assert severity_sum == summary['total_errors'], \
            "Severity counts should sum to total errors"
    
    @given(
        error_type=st.sampled_from(['ValueError', 'RuntimeError', 'TypeError']),
        num_suggestions=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=50, deadline=None)
    def test_recovery_suggestions_retrieval_property(self, error_type, num_suggestions):
        """Property: Recovery suggestions should be retrievable by error type.
        
        For any error type with recovery suggestions, all suggestions
        should be retrievable and deduplicated.
        """
        # Log errors with recovery suggestions
        suggestions = [f"Suggestion {i} for {error_type}" for i in range(num_suggestions)]
        
        for suggestion in suggestions:
            if error_type == 'ValueError':
                error = ValueError("Test error")
            elif error_type == 'RuntimeError':
                error = RuntimeError("Test error")
            else:
                error = TypeError("Test error")
            
            self.error_handler.log_error(error, recovery_suggestion=suggestion)
        
        # Get recovery suggestions for this error type
        retrieved_suggestions = self.error_handler.get_recovery_suggestions(error_type)
        
        # Property: All unique suggestions should be retrievable
        assert len(retrieved_suggestions) >= num_suggestions, \
            f"Should retrieve at least {num_suggestions} suggestions"
        
        # Property: Suggestions should be unique (no duplicates)
        assert len(retrieved_suggestions) == len(set(retrieved_suggestions)), \
            "Retrieved suggestions should be unique"

