"""
Subtitle Routes Module
======================
Handles all subtitle-related endpoints including extraction, checking, and retrieval.
"""

from flask import Blueprint, request, jsonify, send_from_directory
import os
from pathlib import Path
import logging

from ..services.subtitle_service import SubtitleService
from ..utils.video_utils import check_subtitles

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
subtitle_bp = Blueprint('subtitles', __name__)

# Initialize services
subtitle_service = SubtitleService()

# Configure folders (relative to app directory)
UPLOAD_FOLDER = Path(__file__).parent.parent / 'uploads'
SUBTITLES_FOLDER = Path(__file__).parent.parent / 'subtitles'


@subtitle_bp.route('/check/<filename>')
def check_video_subtitles(filename):
    """
    Check if a video file has embedded subtitles or if AI transcription is available.
    
    Args:
        filename: Name of the video file to check
        
    Returns:
        JSON with 'has_subtitles' boolean
    """
    try:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Video file not found'}), 404
        
        has_subtitles = check_subtitles(filepath)
        return jsonify({'has_subtitles': has_subtitles}), 200
        
    except Exception as e:
        logger.error(f"Error checking subtitles: {str(e)}")
        return jsonify({'error': str(e)}), 500


@subtitle_bp.route('/extract/<filename>')
def get_subtitles(filename):
    """
    Extract subtitles from a video file as an SRT download.
    
    Tries embedded subtitles first, falls back to Whisper transcription.
    """
    try:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Video file not found'}), 404
        
        # Try to extract embedded subtitles first
        subtitle_path = subtitle_service.extract_subtitles(filepath, filename)
        
        # If no embedded subtitles, use Whisper
        if not subtitle_path:
            logger.info("No embedded subtitles found, using Whisper")
            subtitle_path = os.path.join(
                SUBTITLES_FOLDER, 
                f"{os.path.splitext(filename)[0]}_whisper.srt"
            )
            if subtitle_service.transcribe_with_whisper(filepath, subtitle_path):
                return send_from_directory(
                    SUBTITLES_FOLDER,
                    os.path.basename(subtitle_path),
                    as_attachment=True
                )
            else:
                return jsonify({'error': 'Failed to generate subtitles'}), 500
        
        return send_from_directory(
            SUBTITLES_FOLDER,
            os.path.basename(subtitle_path),
            as_attachment=True
        )
        
    except Exception as e:
        logger.error(f"Error extracting subtitles: {str(e)}")
        return jsonify({'error': str(e)}), 500


@subtitle_bp.route('/get/<filename>')
def get_subtitles_json(filename):
    """
    Get subtitles as JSON with timestamps for frontend display and analysis.
    """
    try:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Video file not found'}), 404
        
        result = subtitle_service.get_subtitles_json(filepath, str(SUBTITLES_FOLDER))
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error extracting subtitles: {str(e)}")
        return jsonify({'error': str(e)}), 500
