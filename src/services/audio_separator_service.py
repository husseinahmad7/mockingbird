"""Audio separation service for splitting vocals and background audio."""

import os
import tempfile
import logging
from pathlib import Path
from typing import Tuple, Optional, List

try:
    from audio_separator.separator import Separator
except ImportError:
    Separator = None

logger = logging.getLogger(__name__)


class AudioSeparatorService:
    """Service for separating vocals and background audio using audio-separator."""

    def __init__(self, model_name: str = "UVR-MDX-NET-Inst_HQ_4.onnx", save_files: bool = False):
        """Initialize the audio separator service.

        Args:
            model_name: Name of the separation model to use
            save_files: Whether to save separated files permanently
        """
        if Separator is None:
            raise ImportError("audio-separator package is required. Install with: pip install audio-separator[cpu]")

        self.model_name = model_name
        self.save_files = save_files
        self.separator = None
        self.temp_files = []
        self.saved_files = []  # Track permanently saved files

    def _initialize_separator(self, model_name: Optional[str] = None):
        """Lazy initialize the separator.

        Args:
            model_name: Optional model name to use (overrides default)
        """
        model_to_load = model_name or self.model_name

        if self.separator is None:
            logger.info(f"Initializing audio separator with model: {model_to_load}")
            self.separator = Separator(
                log_level=logging.INFO,
                model_file_dir=os.path.join(tempfile.gettempdir(), "audio-separator-models"),
                use_autocast=True
            )
            self.separator.load_model(model_filename=model_to_load)
            logger.info("Audio separator initialized successfully")
        elif model_to_load != self.model_name:
            # Reload with different model
            logger.info(f"Reloading audio separator with model: {model_to_load}")
            self.separator.load_model(model_filename=model_to_load)
            self.model_name = model_to_load
    
    def separate_audio(
        self,
        audio_path: str,
        output_dir: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Tuple[str, str]:
        """Separate audio into vocals and background (instrumental).

        Args:
            audio_path: Path to the audio file to separate
            output_dir: Directory to save separated files (uses temp dir if None)
            model_name: Optional model name to use for this separation

        Returns:
            Tuple of (vocals_path, background_path)

        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If separation fails
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Initialize separator if needed
        self._initialize_separator(model_name)

        try:
            logger.info(f"Separating audio: {audio_path}")

            # Perform separation
            output_files = self.separator.separate(audio_path)

            # audio-separator returns [vocals, instrumental]
            if len(output_files) < 2:
                raise RuntimeError("Audio separation did not produce expected output files")

            vocals_path = output_files[1]  # Usually ends with (Vocals).wav
            background_path = output_files[0]  # Usually ends with (Instrumental).wav

            # Handle file tracking based on save_files setting
            if self.save_files:
                # Keep files permanently
                self.saved_files.extend([vocals_path, background_path])
                logger.info(f"Separated files saved permanently:")
            else:
                # Track for cleanup
                self.temp_files.extend([vocals_path, background_path])
                logger.info(f"Audio separated successfully:")

            logger.info(f"  Vocals: {vocals_path}")
            logger.info(f"  Background: {background_path}")

            return vocals_path, background_path

        except Exception as e:
            logger.error(f"Audio separation failed: {e}")
            raise RuntimeError(f"Failed to separate audio: {e}")

    def separate_audio_serial(
        self,
        audio_path: str,
        models: List[str],
        output_dir: Optional[str] = None
    ) -> Tuple[str, str]:
        """Apply multiple separation models in series for better quality.

        Args:
            audio_path: Path to the audio file to separate
            models: List of model names to apply in order
            output_dir: Directory to save separated files

        Returns:
            Tuple of (vocals_path, background_path) from final separation
        """
        if not models:
            raise ValueError("At least one model must be specified")

        logger.info(f"Applying {len(models)} separation models in series")

        current_audio = audio_path
        vocals_path = None
        background_path = None

        for i, model_name in enumerate(models):
            logger.info(f"Applying model {i+1}/{len(models)}: {model_name}")

            # Separate with current model
            vocals_path, background_path = self.separate_audio(
                current_audio,
                output_dir=output_dir,
                model_name=model_name
            )

            # Use the background (instrumental) as input for next iteration
            # This progressively removes more vocals
            if i < len(models) - 1:
                current_audio = background_path

        logger.info("Serial separation complete")
        return vocals_path, background_path
    
    def remove_vocals(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> str:
        """Remove vocals from audio, keeping only background/instrumental.

        Args:
            audio_path: Path to the audio file
            output_path: Path for output file (auto-generated if None)
            model_name: Optional model name to use

        Returns:
            Path to background-only audio file
        """
        output_dir = os.path.dirname(output_path) if output_path else None
        vocals_path, background_path = self.separate_audio(audio_path, output_dir, model_name)

        if output_path and background_path != output_path:
            # Move to desired output path
            os.rename(background_path, output_path)
            background_path = output_path

        # Clean up vocals file unless we're saving files
        if not self.save_files and os.path.exists(vocals_path):
            os.unlink(vocals_path)
            if vocals_path in self.temp_files:
                self.temp_files.remove(vocals_path)

        return background_path

    def get_saved_files(self) -> List[str]:
        """Get list of permanently saved separated files.

        Returns:
            List of file paths that were saved
        """
        return self.saved_files.copy()
    
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

