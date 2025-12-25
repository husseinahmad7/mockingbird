"""Configuration management service for system settings and hardware detection.

This module provides configuration management, GPU detection, and system resource monitoring.
"""

import os
import platform
import psutil
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    """Information about system hardware capabilities."""
    has_cuda: bool = False
    has_mps: bool = False  # Apple Metal Performance Shaders
    cuda_version: Optional[str] = None
    gpu_count: int = 0
    gpu_names: List[str] = field(default_factory=list)
    cpu_count: int = 0
    total_memory_gb: float = 0.0
    available_memory_gb: float = 0.0
    platform: str = ""
    python_version: str = ""


@dataclass
class ResourceUsage:
    """Current system resource usage."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_available_gb: float = 0.0
    disk_usage_percent: float = 0.0


class ConfigurationManager:
    """Manages system configuration, hardware detection, and resource monitoring."""
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.hardware_info = self._detect_hardware()
        self._config_cache: Dict[str, Any] = {}
        
    def _detect_hardware(self) -> HardwareInfo:
        """Detect available hardware capabilities.
        
        Returns:
            HardwareInfo object with detected capabilities
        """
        info = HardwareInfo()
        
        # Basic system info
        info.platform = platform.system()
        info.python_version = platform.python_version()
        info.cpu_count = psutil.cpu_count(logical=True)
        
        # Memory info
        memory = psutil.virtual_memory()
        info.total_memory_gb = memory.total / (1024 ** 3)
        info.available_memory_gb = memory.available / (1024 ** 3)
        
        # GPU detection
        info.has_cuda = self._detect_cuda()
        info.has_mps = self._detect_mps()
        
        if info.has_cuda:
            info.cuda_version, info.gpu_count, info.gpu_names = self._get_cuda_info()
        
        logger.info(f"Hardware detected: {info}")
        return info
    
    def _detect_cuda(self) -> bool:
        """Detect if CUDA is available.
        
        Returns:
            True if CUDA is available, False otherwise
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            logger.debug("PyTorch not available, CUDA detection skipped")
            return False
    
    def _detect_mps(self) -> bool:
        """Detect if Apple Metal Performance Shaders (MPS) is available.
        
        Returns:
            True if MPS is available, False otherwise
        """
        try:
            import torch
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except (ImportError, AttributeError):
            return False
    
    def _get_cuda_info(self) -> tuple[Optional[str], int, List[str]]:
        """Get detailed CUDA information.
        
        Returns:
            Tuple of (cuda_version, gpu_count, gpu_names)
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return None, 0, []
            
            cuda_version = torch.version.cuda
            gpu_count = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            
            return cuda_version, gpu_count, gpu_names
        except Exception as e:
            logger.warning(f"Failed to get CUDA info: {e}")
            return None, 0, []
    
    def get_optimal_device(self) -> str:
        """Determine the optimal device for computation.
        
        Returns:
            Device string: 'cuda', 'mps', or 'cpu'
        """
        if self.hardware_info.has_cuda:
            return 'cuda'
        elif self.hardware_info.has_mps:
            return 'mps'
        else:
            return 'cpu'
    
    def get_resource_usage(self) -> ResourceUsage:
        """Get current system resource usage.
        
        Returns:
            ResourceUsage object with current metrics
        """
        usage = ResourceUsage()
        
        # CPU usage
        usage.cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        usage.memory_percent = memory.percent
        usage.memory_used_gb = memory.used / (1024 ** 3)
        usage.memory_available_gb = memory.available / (1024 ** 3)
        
        # Disk usage (for temp directory)
        disk = psutil.disk_usage('/')
        usage.disk_usage_percent = disk.percent
        
        return usage

    def validate_configuration(self, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate configuration settings and provide guidance.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Validate Gemini API key
        if 'gemini_api_key' in config:
            api_key = config['gemini_api_key']
            if api_key and not isinstance(api_key, str):
                errors.append("Gemini API key must be a string")
            elif api_key and len(api_key) < 10:
                errors.append("Gemini API key appears to be invalid (too short)")

        # Validate whisper model size
        if 'whisper_model_size' in config:
            valid_sizes = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']
            if config['whisper_model_size'] not in valid_sizes:
                errors.append(f"Invalid whisper model size. Must be one of: {', '.join(valid_sizes)}")

        # Validate speed adjustment bounds
        if 'max_speed_adjustment' in config:
            max_speed = config['max_speed_adjustment']
            if not isinstance(max_speed, (int, float)) or max_speed < 1.0 or max_speed > 2.0:
                errors.append("max_speed_adjustment must be between 1.0 and 2.0")

        if 'min_speed_adjustment' in config:
            min_speed = config['min_speed_adjustment']
            if not isinstance(min_speed, (int, float)) or min_speed < 0.5 or min_speed > 1.0:
                errors.append("min_speed_adjustment must be between 0.5 and 1.0")

        # Validate volume ducking level
        if 'volume_ducking_level' in config:
            ducking = config['volume_ducking_level']
            if not isinstance(ducking, (int, float)) or ducking < -30.0 or ducking > 0.0:
                errors.append("volume_ducking_level must be between -30.0 and 0.0 dB")

        # Validate batch size
        if 'batch_size' in config:
            batch_size = config['batch_size']
            if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 100:
                errors.append("batch_size must be an integer between 1 and 100")

        return len(errors) == 0, errors

    def get_optimization_suggestions(self) -> List[str]:
        """Get optimization suggestions based on current hardware.

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        # GPU suggestions
        if not self.hardware_info.has_cuda and not self.hardware_info.has_mps:
            suggestions.append(
                "No GPU detected. Consider using a GPU for faster processing. "
                "Use smaller Whisper models (tiny/base) for better CPU performance."
            )
        elif self.hardware_info.has_cuda:
            suggestions.append(
                f"CUDA GPU detected ({self.hardware_info.gpu_count} device(s)). "
                "GPU acceleration will be used automatically for supported operations."
            )

        # Memory suggestions
        if self.hardware_info.total_memory_gb < 8:
            suggestions.append(
                f"Low system memory detected ({self.hardware_info.total_memory_gb:.1f} GB). "
                "Consider using smaller models and reducing batch sizes."
            )
        elif self.hardware_info.total_memory_gb >= 16:
            suggestions.append(
                f"Sufficient memory available ({self.hardware_info.total_memory_gb:.1f} GB). "
                "You can use larger models for better accuracy."
            )

        # CPU suggestions
        if self.hardware_info.cpu_count < 4:
            suggestions.append(
                f"Limited CPU cores detected ({self.hardware_info.cpu_count}). "
                "Processing may be slower. Consider using smaller models."
            )

        # Resource usage suggestions
        usage = self.get_resource_usage()
        if usage.memory_percent > 80:
            suggestions.append(
                f"High memory usage detected ({usage.memory_percent:.1f}%). "
                "Close other applications for better performance."
            )

        if usage.disk_usage_percent > 90:
            suggestions.append(
                f"Low disk space ({100 - usage.disk_usage_percent:.1f}% free). "
                "Ensure sufficient space for temporary files."
            )

        return suggestions

    def get_recommended_config(self) -> Dict[str, Any]:
        """Get recommended configuration based on hardware.

        Returns:
            Dictionary with recommended configuration values
        """
        config = {}

        # Recommend model size based on available memory
        if self.hardware_info.total_memory_gb < 4:
            config['whisper_model_size'] = 'tiny'
        elif self.hardware_info.total_memory_gb < 8:
            config['whisper_model_size'] = 'base'
        elif self.hardware_info.total_memory_gb < 16:
            config['whisper_model_size'] = 'small'
        else:
            config['whisper_model_size'] = 'medium'

        # Recommend batch size based on memory
        if self.hardware_info.total_memory_gb < 8:
            config['batch_size'] = 10
        elif self.hardware_info.total_memory_gb < 16:
            config['batch_size'] = 20
        else:
            config['batch_size'] = 30

        # Device recommendation
        config['device'] = self.get_optimal_device()

        return config

    def check_resource_availability(self, required_memory_gb: float = 2.0) -> tuple[bool, str]:
        """Check if sufficient resources are available for processing.

        Args:
            required_memory_gb: Minimum required memory in GB

        Returns:
            Tuple of (is_available, message)
        """
        usage = self.get_resource_usage()

        # Check available memory
        if usage.memory_available_gb < required_memory_gb:
            return False, (
                f"Insufficient memory available. "
                f"Required: {required_memory_gb:.1f} GB, "
                f"Available: {usage.memory_available_gb:.1f} GB"
            )

        # Check disk space (need at least 1GB free)
        if usage.disk_usage_percent > 99:
            return False, "Insufficient disk space for temporary files"

        # Check CPU usage
        if usage.cpu_percent > 95:
            return True, (
                f"Warning: High CPU usage ({usage.cpu_percent:.1f}%). "
                "Processing may be slower than expected."
            )

        return True, "Sufficient resources available"

    def get_env_variable(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable with caching.

        Args:
            key: Environment variable name
            default: Default value if not found

        Returns:
            Environment variable value or default
        """
        if key in self._config_cache:
            return self._config_cache[key]

        value = os.environ.get(key, default)
        self._config_cache[key] = value
        return value

    def set_env_variable(self, key: str, value: str) -> None:
        """Set environment variable and update cache.

        Args:
            key: Environment variable name
            value: Value to set
        """
        os.environ[key] = value
        self._config_cache[key] = value

    def get_hardware_summary(self) -> str:
        """Get a human-readable summary of hardware capabilities.

        Returns:
            Formatted string with hardware information
        """
        info = self.hardware_info
        lines = [
            "=== Hardware Summary ===",
            f"Platform: {info.platform}",
            f"Python: {info.python_version}",
            f"CPU Cores: {info.cpu_count}",
            f"Total Memory: {info.total_memory_gb:.1f} GB",
            f"Available Memory: {info.available_memory_gb:.1f} GB",
        ]

        if info.has_cuda:
            lines.append(f"CUDA: Available (v{info.cuda_version})")
            lines.append(f"GPU Count: {info.gpu_count}")
            for i, name in enumerate(info.gpu_names):
                lines.append(f"  GPU {i}: {name}")
        elif info.has_mps:
            lines.append("MPS: Available (Apple Silicon)")
        else:
            lines.append("GPU: Not available (CPU only)")

        lines.append("=" * 24)
        return "\n".join(lines)


