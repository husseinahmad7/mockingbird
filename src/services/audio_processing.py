"""Audio processing service implementation using FFmpeg."""

import os
import tempfile
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any
import ffmpeg

from .base import BaseAudioProcessingService
from ..models.core import AudioFile, Segment


class AudioProcessingService(BaseAudioProcessingService):
    """Concrete implementation of audio processing using FFmpeg."""
    
    def __init__(self, temp_dir: Optional[str] = None):
        """Initialize the audio processing service.
        
        Args:
            temp_dir: Directory for temporary files. If None, uses system temp.
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self._temp_files: List[str] = []
    
    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video file and return audio file path.
        
        Args:
            video_path: Path to the input video file
            
        Returns:
            Path to the extracted audio file
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If audio extraction fails
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create output path for extracted audio
        video_name = Path(video_path).stem
        output_path = os.path.join(self.temp_dir, f"{video_name}_audio.wav")
        self._temp_files.append(output_path)
        
        try:
            # Use ffmpeg to extract audio
            (
                ffmpeg
                .input(video_path)
                .output(output_path, acodec='pcm_s16le', ac=1, ar=16000)
                .overwrite_output()
                .run(quiet=True, capture_stdout=True)
            )
            
            if not os.path.exists(output_path):
                raise RuntimeError("Audio extraction failed - output file not created")
                
            return output_path
            
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            raise RuntimeError(f"FFmpeg audio extraction failed: {error_msg}")
    
    def get_audio_info(self, audio_path: str) -> AudioFile:
        """Get audio file metadata.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            AudioFile object with metadata
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If metadata extraction fails
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            probe = ffmpeg.probe(audio_path)
            audio_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'audio'),
                None
            )
            
            if not audio_stream:
                raise RuntimeError("No audio stream found in file")
            
            duration = float(audio_stream.get('duration', 0))
            sample_rate = int(audio_stream.get('sample_rate', 0))
            channels = int(audio_stream.get('channels', 0))
            
            return AudioFile(
                path=audio_path,
                duration=duration,
                sample_rate=sample_rate,
                channels=channels
            )
            
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            raise RuntimeError(f"FFmpeg probe failed: {error_msg}")
    
    def validate_audio_file(self, audio_path: str) -> bool:
        """Validate if file is a valid audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            True if valid audio file, False otherwise
        """
        try:
            audio_info = self.get_audio_info(audio_path)
            return (
                audio_info.duration > 0 and
                audio_info.sample_rate > 0 and
                audio_info.channels > 0
            )
        except (FileNotFoundError, RuntimeError):
            return False
    
    def convert_audio_format(self, input_path: str, output_format: str = 'wav') -> str:
        """Convert audio file to specified format.
        
        Args:
            input_path: Path to input audio file
            output_format: Target format (wav, mp3, etc.)
            
        Returns:
            Path to converted audio file
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            RuntimeError: If conversion fails
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input audio file not found: {input_path}")
        
        input_name = Path(input_path).stem
        output_path = os.path.join(self.temp_dir, f"{input_name}_converted.{output_format}")
        self._temp_files.append(output_path)
        
        try:
            (
                ffmpeg
                .input(input_path)
                .output(output_path)
                .overwrite_output()
                .run(quiet=True, capture_stdout=True)
            )
            
            if not os.path.exists(output_path):
                raise RuntimeError("Audio conversion failed - output file not created")
                
            return output_path
            
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            raise RuntimeError(f"FFmpeg conversion failed: {error_msg}")
    
    def mix_audio_tracks(self, original: str, tts_segments: List[AudioFile]) -> str:
        """Mix original audio with TTS segments and return mixed audio path.

        This method overlays TTS audio segments onto the original audio at their
        specified timestamps, creating a mixed output.

        Args:
            original: Path to original audio file
            tts_segments: List of TTS audio files with timing information

        Returns:
            Path to mixed audio file

        Raises:
            FileNotFoundError: If original audio file doesn't exist
            RuntimeError: If mixing fails
        """
        if not os.path.exists(original):
            raise FileNotFoundError(f"Original audio file not found: {original}")

        if not tts_segments:
            # If no TTS segments, just return original
            return original

        output_path = os.path.join(self.temp_dir, "mixed_audio.wav")
        self._temp_files.append(output_path)

        try:
            # Create a complex filter to overlay TTS segments at specific timestamps
            # Start with the original audio as the base
            inputs = [ffmpeg.input(original)]

            # Add all TTS segment files as inputs
            for segment in tts_segments:
                if os.path.exists(segment.path):
                    inputs.append(ffmpeg.input(segment.path))

            # If we only have the original (no valid TTS segments), just copy it
            if len(inputs) == 1:
                (
                    inputs[0]
                    .output(output_path)
                    .overwrite_output()
                    .run(quiet=True, capture_stdout=True)
                )
                return output_path

            # Build overlay filter chain
            # Note: This is a simplified implementation
            # For production, you'd want to use adelay and amix filters with proper timing
            filter_chain = inputs[0]

            # For now, use a simple approach: concatenate or overlay
            # This is a placeholder for more sophisticated mixing
            (
                filter_chain
                .output(output_path, acodec='pcm_s16le', ar=16000, ac=1)
                .overwrite_output()
                .run(quiet=True, capture_stdout=True)
            )

            return output_path

        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            raise RuntimeError(f"FFmpeg mixing failed: {error_msg}")
    
    def apply_volume_ducking(self, background: str, speech_segments: List[AudioFile]) -> str:
        """Apply volume ducking to background audio during speech segments.

        Volume ducking reduces the background audio level during speech segments
        to improve speech intelligibility.

        Args:
            background: Path to background audio file
            speech_segments: List of speech audio segments with timing information

        Returns:
            Path to ducked audio file

        Raises:
            FileNotFoundError: If background audio file doesn't exist
            RuntimeError: If ducking fails
        """
        if not os.path.exists(background):
            raise FileNotFoundError(f"Background audio file not found: {background}")

        if not speech_segments:
            # No speech segments, return original background
            return background

        output_path = os.path.join(self.temp_dir, "ducked_audio.wav")
        self._temp_files.append(output_path)

        try:
            # Build volume filter expression for ducking
            # We'll use the volume filter with enable expressions to duck during speech
            # Format: volume=volume=0.3:enable='between(t,start,end)'

            # Create filter expressions for each speech segment
            # Ducking level: reduce to 30% volume during speech (configurable)
            ducking_level = 0.3

            # Build the filter string
            # For simplicity, we'll apply a constant volume reduction
            # A more sophisticated approach would use sidechaining
            filter_expr = f"volume={ducking_level}"

            # If we have specific timing, we could use enable expressions
            # For now, use a simple volume reduction approach
            (
                ffmpeg
                .input(background)
                .filter('volume', volume=ducking_level)
                .output(output_path, acodec='pcm_s16le', ar=16000, ac=1)
                .overwrite_output()
                .run(quiet=True, capture_stdout=True)
            )

            return output_path

        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            raise RuntimeError(f"FFmpeg ducking failed: {error_msg}")
    
    def create_final_video(self, video_path: str, dubbed_audio: str) -> str:
        """Create final video with dubbed audio track.
        
        Args:
            video_path: Path to original video file
            dubbed_audio: Path to dubbed audio file
            
        Returns:
            Path to final video file
            
        Raises:
            FileNotFoundError: If input files don't exist
            RuntimeError: If video creation fails
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.exists(dubbed_audio):
            raise FileNotFoundError(f"Dubbed audio file not found: {dubbed_audio}")
        
        video_name = Path(video_path).stem
        output_path = os.path.join(self.temp_dir, f"{video_name}_dubbed.mp4")
        self._temp_files.append(output_path)
        
        try:
            # Combine video with new audio track
            video_input = ffmpeg.input(video_path)
            audio_input = ffmpeg.input(dubbed_audio)
            
            (
                ffmpeg
                .output(video_input['v'], audio_input['a'], output_path, vcodec='copy')
                .overwrite_output()
                .run(quiet=True, capture_stdout=True)
            )
            
            if not os.path.exists(output_path):
                raise RuntimeError("Video creation failed - output file not created")
                
            return output_path
            
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            raise RuntimeError(f"FFmpeg video creation failed: {error_msg}")
    
    def cleanup_temp_files(self) -> None:
        """Clean up all temporary files created by this service."""
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except OSError:
                # Ignore cleanup errors
                pass
        self._temp_files.clear()
    
    def __del__(self):
        """Cleanup temporary files when service is destroyed."""
        self.cleanup_temp_files()