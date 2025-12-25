"""Centralized error handling and logging service.

This module provides comprehensive error handling, logging, and recovery mechanisms.
"""

import logging
import traceback
import sys
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


class ErrorSeverity(Enum):
    """Error severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    timestamp: datetime
    severity: ErrorSeverity
    error_type: str
    message: str
    traceback: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_suggestion: Optional[str] = None


class ErrorHandler:
    """Centralized error handler with logging and recovery mechanisms."""
    
    def __init__(self, log_file: Optional[str] = None, log_level: int = logging.INFO):
        """Initialize the error handler.
        
        Args:
            log_file: Optional path to log file. If None, logs to console only.
            log_level: Logging level (default: INFO)
        """
        self.error_log: List[ErrorRecord] = []
        self.logger = self._setup_logger(log_file, log_level)
        self._fallback_handlers: Dict[str, Callable] = {}
        
    def _setup_logger(self, log_file: Optional[str], log_level: int) -> logging.Logger:
        """Set up the logging system.
        
        Args:
            log_file: Optional path to log file
            log_level: Logging level
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger('video_translator')
        logger.setLevel(log_level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def log_error(
        self,
        error: Exception,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestion: Optional[str] = None
    ) -> ErrorRecord:
        """Log an error with full context and recovery suggestions.
        
        Args:
            error: The exception that occurred
            severity: Error severity level
            context: Additional context information
            recovery_suggestion: Suggestion for recovering from the error
            
        Returns:
            ErrorRecord object
        """
        # Create error record
        record = ErrorRecord(
            timestamp=datetime.now(),
            severity=severity,
            error_type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc() if sys.exc_info()[0] is not None else None,
            context=context or {},
            recovery_suggestion=recovery_suggestion
        )
        
        # Add to error log
        self.error_log.append(record)
        
        # Log to logger
        log_message = f"{record.error_type}: {record.message}"
        if context:
            log_message += f" | Context: {context}"
        if recovery_suggestion:
            log_message += f" | Suggestion: {recovery_suggestion}"
        
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif severity == ErrorSeverity.ERROR:
            self.logger.error(log_message)
        elif severity == ErrorSeverity.WARNING:
            self.logger.warning(log_message)
        elif severity == ErrorSeverity.INFO:
            self.logger.info(log_message)
        else:
            self.logger.debug(log_message)
        
        # Log traceback if available
        if record.traceback and severity in [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL]:
            self.logger.debug(f"Traceback:\n{record.traceback}")
        
        return record
    
    def log_info(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log an informational message.
        
        Args:
            message: The message to log
            context: Additional context information
        """
        log_message = message
        if context:
            log_message += f" | Context: {context}"
        self.logger.info(log_message)
    
    def log_warning(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log a warning message.

        Args:
            message: The message to log
            context: Additional context information
        """
        log_message = message
        if context:
            log_message += f" | Context: {context}"
        self.logger.warning(log_message)

    def register_fallback_handler(self, error_type: str, handler: Callable) -> None:
        """Register a fallback handler for a specific error type.

        Args:
            error_type: The type of error to handle (e.g., 'APIError', 'NetworkError')
            handler: Callable that handles the error and returns a fallback result
        """
        self._fallback_handlers[error_type] = handler
        self.logger.info(f"Registered fallback handler for {error_type}")

    def handle_with_fallback(
        self,
        primary_func: Callable,
        error_type: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute a function with automatic fallback on error.

        Args:
            primary_func: The primary function to execute
            error_type: The type of error to handle
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Result from primary function or fallback handler

        Raises:
            Exception: If both primary and fallback fail
        """
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            self.log_error(
                e,
                severity=ErrorSeverity.WARNING,
                context={'function': primary_func.__name__, 'error_type': error_type},
                recovery_suggestion=f"Attempting fallback for {error_type}"
            )

            # Try fallback handler
            if error_type in self._fallback_handlers:
                try:
                    self.logger.info(f"Executing fallback handler for {error_type}")
                    return self._fallback_handlers[error_type](*args, **kwargs)
                except Exception as fallback_error:
                    self.log_error(
                        fallback_error,
                        severity=ErrorSeverity.ERROR,
                        context={'function': 'fallback_handler', 'error_type': error_type},
                        recovery_suggestion="Both primary and fallback methods failed"
                    )
                    raise
            else:
                # No fallback available
                self.logger.error(f"No fallback handler registered for {error_type}")
                raise

    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of all logged errors.

        Returns:
            Dictionary with error statistics and recent errors
        """
        if not self.error_log:
            return {
                'total_errors': 0,
                'by_severity': {},
                'by_type': {},
                'recent_errors': []
            }

        # Count by severity
        by_severity = {}
        for record in self.error_log:
            severity_name = record.severity.value
            by_severity[severity_name] = by_severity.get(severity_name, 0) + 1

        # Count by type
        by_type = {}
        for record in self.error_log:
            by_type[record.error_type] = by_type.get(record.error_type, 0) + 1

        # Get recent errors (last 10)
        recent_errors = [
            {
                'timestamp': record.timestamp.isoformat(),
                'severity': record.severity.value,
                'type': record.error_type,
                'message': record.message,
                'suggestion': record.recovery_suggestion
            }
            for record in self.error_log[-10:]
        ]

        return {
            'total_errors': len(self.error_log),
            'by_severity': by_severity,
            'by_type': by_type,
            'recent_errors': recent_errors
        }

    def get_recovery_suggestions(self, error_type: Optional[str] = None) -> List[str]:
        """Get recovery suggestions for errors.

        Args:
            error_type: Optional filter by error type

        Returns:
            List of recovery suggestions
        """
        suggestions = []

        for record in self.error_log:
            if error_type is None or record.error_type == error_type:
                if record.recovery_suggestion:
                    suggestions.append(record.recovery_suggestion)

        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique_suggestions.append(suggestion)

        return unique_suggestions

    def clear_error_log(self) -> None:
        """Clear the error log."""
        self.error_log.clear()
        self.logger.info("Error log cleared")

    def export_error_log(self, output_file: str) -> None:
        """Export error log to a file.

        Args:
            output_file: Path to output file
        """
        import json

        log_data = {
            'export_time': datetime.now().isoformat(),
            'total_errors': len(self.error_log),
            'errors': [
                {
                    'timestamp': record.timestamp.isoformat(),
                    'severity': record.severity.value,
                    'type': record.error_type,
                    'message': record.message,
                    'traceback': record.traceback,
                    'context': record.context,
                    'recovery_suggestion': record.recovery_suggestion
                }
                for record in self.error_log
            ]
        }

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        self.logger.info(f"Error log exported to {output_file}")

    def get_common_error_suggestions(self) -> Dict[str, str]:
        """Get common error types and their recovery suggestions.

        Returns:
            Dictionary mapping error types to recovery suggestions
        """
        return {
            'FileNotFoundError': 'Verify that the file path is correct and the file exists',
            'PermissionError': 'Check file permissions and ensure you have read/write access',
            'RuntimeError': 'Check system resources and try again with different settings',
            'APIError': 'Verify API key is valid and check network connectivity',
            'NetworkError': 'Check internet connection and try again',
            'OutOfMemoryError': 'Close other applications or use a smaller model size',
            'TimeoutError': 'Increase timeout duration or check network speed',
            'ValidationError': 'Review input parameters and ensure they meet requirements',
            'ConfigurationError': 'Check configuration file and ensure all required fields are set',
            'ModelLoadError': 'Ensure model files are downloaded and not corrupted'
        }

