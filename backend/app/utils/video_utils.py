"""
Video Utilities Module
======================
Canonical implementation of all video-related utility functions.
This module is the single source of truth for:
- File validation
- Video metadata extraction
- Subtitle handling (extraction, parsing, formatting)
- FFmpeg/FFprobe operations

All other modules should import from here rather than defining their own versions.
"""

import os
from pathlib import Path
import subprocess
import json
import logging
from datetime import timedelta
from moviepy.video.io.VideoFileClip import VideoFileClip
import traceback

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION ACCESS
# =============================================================================

def _get_allowed_extensions() -> set:
    """Get allowed video extensions from configuration."""
    try:
        from ..config import get_config
        return get_config().video.allowed_extensions
    except ImportError:
        # Fallback if config not available (e.g., during testing)
        return {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Legacy constant for backward compatibility
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# =============================================================================
# FILE VALIDATION
# =============================================================================

def allowed_file(filename: str) -> bool:
    """
    Check if a file extension is allowed for video upload.
    
    Args:
        filename: The name of the file to check
        
    Returns:
        True if the file extension is in the allowed set, False otherwise
    """
    allowed = _get_allowed_extensions()
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed


# =============================================================================
# VIDEO METADATA
# =============================================================================

def get_video_info(filepath: str) -> dict:
    """
    Extract metadata from a video file.
    
    Args:
        filepath: Path to the video file
        
    Returns:
        Dictionary containing:
        - duration: Video length in seconds
        - fps: Frames per second
        - size: Tuple of (width, height)
        
    Raises:
        Exception: If video cannot be read or is corrupted
    """
    try:
        with VideoFileClip(filepath) as clip:
            return {
                'duration': clip.duration,
                'fps': clip.fps,
                'size': (clip.w, clip.h)
            }
    except Exception as e:
        logger.error(f"Error getting video info for {filepath}: {str(e)}")
        logger.error(traceback.format_exc())
        raise


# =============================================================================
# TIMESTAMP FORMATTING
# =============================================================================

def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).
    
    Args:
        seconds: Time in seconds (can include fractional part)
        
    Returns:
        Formatted timestamp string like "00:01:30,500"
    """
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert SRT timestamp to seconds.
    
    Args:
        timestamp: SRT format timestamp like "00:01:30,500" or "00:01:30.500"
        
    Returns:
        Time in seconds as float
    """
    h, m, s = timestamp.replace(',', '.').split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


# =============================================================================
# SUBTITLE OPERATIONS
# =============================================================================

def check_subtitles(filepath: str) -> bool:
    """
    Check if a video file has embedded subtitle streams.
    
    Uses FFprobe to inspect the video container for subtitle tracks.
    
    Args:
        filepath: Path to the video file
        
    Returns:
        True if at least one subtitle stream exists, False otherwise
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            filepath
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        
        subtitle_streams = [
            stream for stream in data.get('streams', [])
            if stream.get('codec_type') == 'subtitle'
        ]
        
        return len(subtitle_streams) > 0
    except Exception as e:
        logger.error(f"Error checking subtitles for {filepath}: {str(e)}")
        return False


def extract_subtitles(filepath: str, filename: str, subtitles_folder: str) -> str | None:
    """
    Extract embedded subtitles from a video file to an SRT file.
    
    Args:
        filepath: Path to the video file
        filename: Original filename (used to name the output file)
        subtitles_folder: Directory to save the extracted subtitles
        
    Returns:
        Path to the extracted SRT file, or None if extraction failed
    """
    try:
        subtitle_path = os.path.join(subtitles_folder, f"{os.path.splitext(filename)[0]}.srt")
        
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-i', filepath,
            '-map', '0:s:0',
            subtitle_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if os.path.exists(subtitle_path) and os.path.getsize(subtitle_path) > 0:
            logger.info(f"Successfully extracted subtitles to {subtitle_path}")
            return subtitle_path
            
        logger.info(f"No subtitles extracted from {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error extracting subtitles from {filepath}: {str(e)}")
        return None


def generate_srt(segments, output_path: str) -> None:
    """
    Generate an SRT file from Whisper transcription segments.
    
    Args:
        segments: Iterable of Whisper segment objects with .start, .end, .text attributes
        output_path: Path where the SRT file will be written
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, 1):
            start_time = format_timestamp(segment.start)
            end_time = format_timestamp(segment.end)
            text = segment.text.strip()
            f.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")
    
    logger.info(f"Generated SRT file with {i} segments at {output_path}")


def parse_srt_content(srt_content: str) -> list[dict]:
    """
    Parse SRT file content into a list of subtitle entries.
    
    Args:
        srt_content: Raw content of an SRT file
        
    Returns:
        List of dictionaries, each containing:
        - start: Start time in seconds
        - end: End time in seconds
        - text: Subtitle text
    """
    subtitle_blocks = srt_content.strip().split('\n\n')
    formatted_subtitles = []
    
    for block in subtitle_blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            # Parse timestamp line (line index 1)
            timestamp_line = lines[1]
            start_time, end_time = timestamp_line.split(' --> ')
            
            start_seconds = timestamp_to_seconds(start_time)
            end_seconds = timestamp_to_seconds(end_time)
            
            # Get subtitle text (everything after the timestamp)
            text = ' '.join(lines[2:])
            
            formatted_subtitles.append({
                'start': start_seconds,
                'end': end_seconds,
                'text': text
            })
    
    return formatted_subtitles


# =============================================================================
# VIDEO CUTTING
# =============================================================================

def validate_time_range(start_time: float, end_time: float, duration: float) -> tuple[bool, str]:
    """
    Validate that a time range is valid for cutting.
    
    Args:
        start_time: Start time in seconds
        end_time: End time in seconds
        duration: Total video duration in seconds
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if start_time < 0:
        return False, "Start time cannot be negative"
    if end_time > duration:
        return False, f"End time ({end_time}s) exceeds video duration ({duration}s)"
    if start_time >= end_time:
        return False, "Start time must be less than end time"
    return True, ""
