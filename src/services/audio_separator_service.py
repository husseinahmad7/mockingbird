"""Audio separation service for splitting vocals and background audio."""

import os
import tempfile
import logging
from pathlib import Path
from typing import Tuple

try:
    from audio_separator.separator import Separator
except ImportError:
    Separator = None

logger = logging.getLogger(__name__)


class AudioSeparatorService:
    """Service for separating vocals and background audio using audio-separator."""
    
    def __init__(self):
        """Initialize the audio separator service."""
        if Separator is None:
            raise ImportError("audio-separator package is required. Install with: pip install audio-separator[cpu]")
        
        self.separator = None
        self.temp_files = []
        
    def _initialize_separator(self):
        """Lazy initialize the separator."""
        if self.separator is None:
            logger.info("Initializing audio separator...")
            self.separator = Separator(
                log_level=logging.INFO,
                model_file_dir=os.path.join(tempfile.gettempdir(), "audio-separator-models")
            )
            # Use MDX-Net model for best quality
            self.separator.load_model(model_filename="UVR-MDX-NET-Inst_HQ_3.onnx")
            logger.info("Audio separator initialized successfully")
    
    def separate_audio(self, audio_path: str, output_dir: str = None) -> Tuple[str, str]:
        """Separate audio into vocals and background (instrumental).
        
        Args:
            audio_path: Path to the audio file to separate
            output_dir: Directory to save separated files (uses temp dir if None)
            
        Returns:
            Tuple of (vocals_path, background_path)
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If separation fails
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Initialize separator if needed
        self._initialize_separator()
        
        # Use temp directory if output_dir not specified
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        
        try:
            logger.info(f"Separating audio: {audio_path}")
            
            # Perform separation
            output_files = self.separator.separate(audio_path, output_dir=output_dir)
            
            # audio-separator returns [vocals, instrumental]
            if len(output_files) < 2:
                raise RuntimeError("Audio separation did not produce expected output files")
            
            vocals_path = output_files[0]  # Usually ends with (Vocals).wav
            background_path = output_files[1]  # Usually ends with (Instrumental).wav
            
            # Track temp files for cleanup
            self.temp_files.extend([vocals_path, background_path])
            
            logger.info(f"Audio separated successfully:")
            logger.info(f"  Vocals: {vocals_path}")
            logger.info(f"  Background: {background_path}")
            
            return vocals_path, background_path
            
        except Exception as e:
            logger.error(f"Audio separation failed: {e}")
            raise RuntimeError(f"Failed to separate audio: {e}")
    
    def remove_vocals(self, audio_path: str, output_path: str = None) -> str:
        """Remove vocals from audio, keeping only background/instrumental.
        
        Args:
            audio_path: Path to the audio file
            output_path: Path for output file (auto-generated if None)
            
        Returns:
            Path to background-only audio file
        """
        output_dir = os.path.dirname(output_path) if output_path else None
        vocals_path, background_path = self.separate_audio(audio_path, output_dir)
        
        if output_path and background_path != output_path:
            # Move to desired output path
            os.rename(background_path, output_path)
            background_path = output_path
        
        # Clean up vocals file
        if os.path.exists(vocals_path):
            os.unlink(vocals_path)
            if vocals_path in self.temp_files:
                self.temp_files.remove(vocals_path)
        
        return background_path
    
    def cleanup_temp_files(self):
        """Clean up temporary files created by the separator."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.debug(f"Deleted temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_file}: {e}")
        
        self.temp_files.clear()
        logger.info("Cleaned up audio separator temporary files")
    
    def cleanup(self):
        """Alias for cleanup_temp_files for consistency."""
        self.cleanup_temp_files()
    
    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'temp_files'):
            self.cleanup_temp_files()

