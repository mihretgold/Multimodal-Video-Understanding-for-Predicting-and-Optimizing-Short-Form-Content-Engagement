"""
Subtitle Service Module
=======================
Provides functionality for handling video subtitles including:
- Extraction of embedded subtitles
- Whisper-based transcription
- SRT parsing and formatting
"""

import os
import logging
from faster_whisper import WhisperModel
from typing import Optional

# Import canonical utility functions
from ..utils.video_utils import (
    generate_srt, 
    extract_subtitles, 
    parse_srt_content
)

# Import configuration
from ..config import get_config

# Configure logging
logger = logging.getLogger(__name__)


class SubtitleService:
    """
    Service class for handling subtitle-related operations.
    
    Provides methods for:
    - Extracting embedded subtitles from video files
    - Transcribing audio using Whisper
    - Parsing and formatting subtitle data
    """
    
    def __init__(self, model_size: Optional[str] = None):
        """
        Initialize the subtitle service.
        
        Args:
            model_size: Whisper model size ("tiny", "small", "medium", "large").
                       If None, uses value from configuration.
        """
        config = get_config()
        self.model_size = model_size or config.whisper.model_size
        
        # Initialize Whisper with configuration
        whisper_config = config.whisper
        self.model = WhisperModel(
            self.model_size,
            device=whisper_config.device if whisper_config.device != "auto" else "auto",
            compute_type=whisper_config.compute_type if whisper_config.compute_type != "auto" else "default"
        )
        logger.info(f"Initialized SubtitleService with Whisper model: {self.model_size}")

    def transcribe_with_whisper(self, filepath: str, output_path: str) -> bool:
        """
        Transcribe video audio using Whisper and save as SRT file.
        
        Args:
            filepath: Path to the video file to transcribe
            output_path: Path where the SRT file will be saved
            
        Returns:
            True if transcription succeeded, False otherwise
        """
        try:
            logger.info(f"Starting Whisper transcription for: {filepath}")
            
            segments, info = self.model.transcribe(filepath)
            language = info.language
            segments = list(segments)
            
            generate_srt(segments, output_path)
            
            logger.info(f"Transcription complete: {len(segments)} segments, language: {language}")
            return True
            
        except Exception as e:
            logger.error(f"Error transcribing with Whisper: {str(e)}")
            return False

    def extract_subtitles(self, filepath: str, filename: str) -> str | None:
        """
        Extract embedded subtitles from a video file.
        
        Args:
            filepath: Path to the video file
            filename: Original filename (used for output naming)
            
        Returns:
            Path to extracted SRT file, or None if no embedded subtitles
        """
        # Use the canonical extract_subtitles function with default subtitles folder
        subtitles_folder = os.path.join(os.path.dirname(filepath), '..', 'subtitles')
        subtitles_folder = os.path.normpath(subtitles_folder)
        
        # Ensure folder exists
        os.makedirs(subtitles_folder, exist_ok=True)
        
        return extract_subtitles(filepath, filename, subtitles_folder)

    def get_subtitles_json(self, filepath: str, subtitles_folder: str) -> dict:
        """
        Get subtitles in JSON format for API response.
        
        Tries embedded subtitles first, falls back to Whisper transcription.
        
        Args:
            filepath: Path to the video file
            subtitles_folder: Directory for storing/reading subtitle files
            
        Returns:
            Dictionary containing:
            - subtitles: List of {start, end, text} entries
            - language: Detected or embedded language
            - source: "whisper" or "embedded"
            
        Raises:
            Exception: If subtitles cannot be extracted or generated
        """
        try:
            # Try to extract embedded subtitles first
            subtitle_path = extract_subtitles(
                filepath, 
                os.path.basename(filepath), 
                subtitles_folder
            )
            
            # If no embedded subtitles, use Whisper
            if not subtitle_path:
                logger.info("No embedded subtitles found, using Whisper")
                
                segments, info = self.model.transcribe(filepath)
                language = info.language
                segments = list(segments)
                
                formatted_subtitles = [
                    {
                        'start': segment.start,
                        'end': segment.end,
                        'text': segment.text.strip()
                    }
                    for segment in segments
                ]
                
                return {
                    'subtitles': formatted_subtitles,
                    'language': language,
                    'source': 'whisper'
                }
            
            # Parse embedded subtitles from SRT file
            with open(subtitle_path, 'r', encoding='utf-8') as f:
                srt_content = f.read()
            
            formatted_subtitles = parse_srt_content(srt_content)
            
            return {
                'subtitles': formatted_subtitles,
                'language': 'unknown',  # SRT doesn't include language info
                'source': 'embedded'
            }
            
        except Exception as e:
            logger.error(f"Error getting subtitles JSON: {str(e)}")
            raise
