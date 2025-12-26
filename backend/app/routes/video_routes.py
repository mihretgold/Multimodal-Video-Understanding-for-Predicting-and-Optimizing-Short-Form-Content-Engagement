"""
Video Routes Module
===================
Handles video upload, cutting, and serving endpoints.
"""

from flask import Blueprint, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from pathlib import Path
from moviepy.video.io.VideoFileClip import VideoFileClip
import uuid
import logging
import traceback

# Import canonical utility functions
from ..utils.video_utils import allowed_file, get_video_info, validate_time_range

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
video_bp = Blueprint('video', __name__)

# Configure folders (relative to app directory)
UPLOAD_FOLDER = Path(__file__).parent.parent / 'uploads'
CUTS_FOLDER = Path(__file__).parent.parent / 'cuts'


@video_bp.route('/upload', methods=['POST'])
def upload_video():
    """
    Handle video file uploads via the /api/video/upload endpoint.
    
    This is a duplicate of the main /upload endpoint for API consistency.
    """
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            
            file.save(filepath)
            video_info = get_video_info(filepath)
            
            logger.info(f"Uploaded video: {filename} (duration: {video_info['duration']:.2f}s)")
            
            return jsonify({
                'message': 'Video uploaded successfully',
                'filename': filename,
                'duration': video_info['duration'],
                'fps': video_info['fps'],
                'size': video_info['size']
            }), 200
            
        except Exception as e:
            logger.error(f"Error during upload: {str(e)}")
            logger.error(traceback.format_exc())
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Error processing video: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400


@video_bp.route('/cut', methods=['POST'])
def cut_video():
    """
    Cut a video segment from an uploaded video.
    
    Expects JSON body with:
    - filename: Name of the uploaded video
    - startTime: Start time in seconds
    - endTime: End time in seconds
    """
    try:
        data = request.json
        filename = data.get('filename')
        start_time = float(data.get('startTime', 0))
        end_time = float(data.get('endTime', 0))
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        
        if not os.path.exists(input_path):
            return jsonify({'error': 'Video file not found'}), 404
        
        # Generate unique filename for the cut
        cut_filename = f"cut_{uuid.uuid4().hex[:8]}_{filename}"
        output_path = os.path.join(CUTS_FOLDER, cut_filename)
        
        logger.info(f"Starting video cut: {input_path} -> {output_path}")
        logger.info(f"Time range: {start_time:.2f}s - {end_time:.2f}s")
        
        with VideoFileClip(input_path) as video:
            # Validate time range
            is_valid, error_msg = validate_time_range(start_time, end_time, video.duration)
            if not is_valid:
                return jsonify({'error': error_msg}), 400
            
            cut = video.subclipped(start_time, end_time)
            
            cut.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                threads=4,
                preset='ultrafast'
            )
            
            logger.info(f"Video cut completed: {cut_filename}")
            
            return jsonify({
                'message': 'Video cut successfully',
                'cut_filename': cut_filename
            }), 200
            
    except Exception as e:
        logger.error(f"Error during video cut: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Error cutting video: {str(e)}'}), 500


@video_bp.route('/uploads/<filename>')
def serve_video(filename):
    """Serve uploaded video files."""
    return send_from_directory(UPLOAD_FOLDER, filename)


@video_bp.route('/cuts/<filename>')
def serve_cut(filename):
    """Serve cut video files."""
    return send_from_directory(CUTS_FOLDER, filename)
