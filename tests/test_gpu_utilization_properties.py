"""Property-based tests for GPU utilization optimization.

**Feature: video-translator, Property 12: GPU utilization optimization**
**Validates: Requirements 7.2**

Property 12: GPU utilization optimization
For any system with available GPU resources, the system should automatically detect
and utilize GPU acceleration when beneficial.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from unittest.mock import Mock, patch, MagicMock

from src.services.config_manager import ConfigurationManager, HardwareInfo


class TestGPUUtilizationProperties:
    """Property-based tests for GPU utilization optimization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config_manager = ConfigurationManager()
    
    @given(
        has_cuda=st.booleans(),
        has_mps=st.booleans(),
        gpu_count=st.integers(min_value=0, max_value=8),
    )
    @settings(max_examples=100, deadline=None)
    def test_gpu_detection_consistency_property(self, has_cuda, has_mps, gpu_count):
        """Property: GPU detection should be consistent with hardware state.
        
        For any hardware configuration, the detected GPU state should match
        the actual hardware capabilities.
        """
        # Mock hardware detection
        with patch.object(self.config_manager, '_detect_cuda', return_value=has_cuda), \
             patch.object(self.config_manager, '_detect_mps', return_value=has_mps), \
             patch.object(self.config_manager, '_get_cuda_info', return_value=('12.0', gpu_count if has_cuda else 0, [])):
            
            # Re-detect hardware with mocked values
            hardware_info = self.config_manager._detect_hardware()
            
            # Property: CUDA detection should match input
            assert hardware_info.has_cuda == has_cuda, \
                "CUDA detection should match hardware state"
            
            # Property: MPS detection should match input
            assert hardware_info.has_mps == has_mps, \
                "MPS detection should match hardware state"
            
            # Property: GPU count should be 0 if no CUDA
            if not has_cuda:
                assert hardware_info.gpu_count == 0, \
                    "GPU count should be 0 when CUDA is not available"
            else:
                assert hardware_info.gpu_count == gpu_count, \
                    "GPU count should match detected count when CUDA is available"
    
    @given(
        has_cuda=st.booleans(),
        has_mps=st.booleans(),
    )
    @settings(max_examples=100, deadline=None)
    def test_optimal_device_selection_property(self, has_cuda, has_mps):
        """Property: Optimal device selection should prioritize GPU over CPU.
        
        For any hardware configuration, the system should select the best
        available device in order: CUDA > MPS > CPU.
        """
        # Mock hardware info
        self.config_manager.hardware_info.has_cuda = has_cuda
        self.config_manager.hardware_info.has_mps = has_mps
        
        optimal_device = self.config_manager.get_optimal_device()
        
        # Property: CUDA should be preferred if available
        if has_cuda:
            assert optimal_device == 'cuda', \
                "CUDA should be selected when available"
        # Property: MPS should be selected if CUDA not available
        elif has_mps:
            assert optimal_device == 'mps', \
                "MPS should be selected when CUDA not available but MPS is"
        # Property: CPU should be fallback
        else:
            assert optimal_device == 'cpu', \
                "CPU should be selected when no GPU is available"
    
    @given(
        has_cuda=st.booleans(),
        has_mps=st.booleans(),
        total_memory_gb=st.floats(min_value=2.0, max_value=128.0),
    )
    @settings(max_examples=100, deadline=None)
    def test_optimization_suggestions_property(self, has_cuda, has_mps, total_memory_gb):
        """Property: Optimization suggestions should be relevant to hardware.
        
        For any hardware configuration, optimization suggestions should
        provide relevant guidance based on available resources.
        """
        # Mock hardware info
        self.config_manager.hardware_info.has_cuda = has_cuda
        self.config_manager.hardware_info.has_mps = has_mps
        self.config_manager.hardware_info.total_memory_gb = total_memory_gb
        self.config_manager.hardware_info.gpu_count = 1 if has_cuda else 0
        
        suggestions = self.config_manager.get_optimization_suggestions()
        
        # Property: Should always return a list
        assert isinstance(suggestions, list), \
            "Optimization suggestions should be a list"
        
        # Property: Should provide GPU-related suggestions
        has_gpu_suggestion = any('gpu' in s.lower() or 'cuda' in s.lower() or 'mps' in s.lower() 
                                 for s in suggestions)
        assert has_gpu_suggestion, \
            "Should provide GPU-related suggestions"
        
        # Property: Low memory should trigger memory suggestions
        if total_memory_gb < 8:
            has_memory_suggestion = any('memory' in s.lower() for s in suggestions)
            assert has_memory_suggestion, \
                "Low memory should trigger memory-related suggestions"
    
    @given(
        total_memory_gb=st.floats(min_value=2.0, max_value=64.0),
    )
    @settings(max_examples=100, deadline=None)
    def test_recommended_config_property(self, total_memory_gb):
        """Property: Recommended config should scale with available memory.
        
        For any memory configuration, the recommended model size and batch size
        should be appropriate for the available memory.
        """
        # Mock hardware info
        self.config_manager.hardware_info.total_memory_gb = total_memory_gb
        
        recommended = self.config_manager.get_recommended_config()
        
        # Property: Should always include required keys
        assert 'whisper_model_size' in recommended, \
            "Recommended config should include whisper_model_size"
        assert 'batch_size' in recommended, \
            "Recommended config should include batch_size"
        assert 'device' in recommended, \
            "Recommended config should include device"
        
        # Property: Model size should scale with memory
        model_size = recommended['whisper_model_size']
        if total_memory_gb < 4:
            assert model_size == 'tiny', \
                "Should recommend tiny model for <4GB memory"
        elif total_memory_gb < 8:
            assert model_size == 'base', \
                "Should recommend base model for 4-8GB memory"
        elif total_memory_gb < 16:
            assert model_size == 'small', \
                "Should recommend small model for 8-16GB memory"
        else:
            assert model_size == 'medium', \
                "Should recommend medium model for >=16GB memory"
        
        # Property: Batch size should scale with memory
        batch_size = recommended['batch_size']
        assert isinstance(batch_size, int), \
            "Batch size should be an integer"
        assert batch_size > 0, \
            "Batch size should be positive"
        
        # Property: Larger memory should allow larger batch sizes
        if total_memory_gb >= 16:
            assert batch_size >= 20, \
                "Larger memory should allow larger batch sizes"

