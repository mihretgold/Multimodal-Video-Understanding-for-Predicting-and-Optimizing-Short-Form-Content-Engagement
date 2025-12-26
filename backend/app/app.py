"""
Movie Shorts - Video Processing Application
============================================
Main application entry point that configures Flask and registers blueprints.

This module:
- Creates the Flask application factory
- Registers all API blueprints
- Defines core routes (/, /health, file serving)
- Maintains backward compatibility with legacy routes

Route Organization:
- /                     -> Static frontend
- /health               -> Health check
- /uploads/<filename>   -> Serve uploaded videos
- /cuts/<filename>      -> Serve cut videos
- /api/video/*          -> Video operations (upload, cut)
- /api/subtitles/*      -> Subtitle operations
- /api/analysis/*       -> AI analysis operations
- /api/pipeline/*       -> Pipeline operations (new)

Legacy routes (/upload, /cut, etc.) redirect to /api/* equivalents.
"""

from flask import Flask, request, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from flask_cors import CORS

# Import blueprints
from app.routes.subtitle_routes import subtitle_bp
from app.routes.analysis_routes import analysis_bp
from app.routes.video_routes import video_bp
from app.routes.pipeline_routes import pipeline_bp

# Import configuration system
from app.config import get_config, apply_environment_overrides

# Import logging system
from app.logging_config import get_research_logger

# Load environment variables
load_dotenv()

# Initialize configuration with environment overrides
config = get_config()
apply_environment_overrides(config)

# Configure Google Generative AI using config
if config.gemini.api_key:
    genai.configure(api_key=config.gemini.api_key)

# Configure logging
logger = get_research_logger("app", log_to_file=False)


def create_app(config_override=None):
    """
    Application factory function.
    
    Creates and configures the Flask application with:
    - CORS support
    - Blueprint registration
    - Directory setup
    - File upload configuration
    
    Args:
        config_override: Optional AppConfig instance to use instead of global config
    
    Returns:
        Configured Flask application instance
    """
    # Use provided config or get global config
    app_config = config_override or get_config()
    
    app = Flask(__name__)
    CORS(app, origins=app_config.flask.cors_origins)
    
    # Store config in app for access in routes
    app.app_config = app_config
    
    # ==========================================================================
    # REGISTER BLUEPRINTS
    # ==========================================================================
    
    app.register_blueprint(subtitle_bp, url_prefix='/api/subtitles')
    app.register_blueprint(analysis_bp, url_prefix='/api/analysis')
    app.register_blueprint(video_bp, url_prefix='/api/video')
    app.register_blueprint(pipeline_bp, url_prefix='/api/pipeline')
    
    # ==========================================================================
    # CONFIGURE DIRECTORIES AND LIMITS
    # ==========================================================================
    
    # Ensure all directories exist
    app_config.paths.ensure_directories()
    
    # Configure Flask app
    app.config['MAX_CONTENT_LENGTH'] = app_config.video.max_file_size_bytes
    app.config['UPLOAD_FOLDER'] = app_config.paths.uploads
    app.config['CUTS_FOLDER'] = app_config.paths.cuts
    app.config['SUBTITLES_FOLDER'] = app_config.paths.subtitles
    app.config['SECRET_KEY'] = app_config.flask.secret_key
    
    logger.info(
        f"Initialized Flask app",
        extra={
            'experiment': app_config.research.experiment_name,
            'whisper_model': app_config.whisper.model_size,
            'ablation_mode': app_config.ablation.mode_name
        }
    )
    
    # ==========================================================================
    # CORE ROUTES
    # ==========================================================================
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint for monitoring."""
        return {
            'status': 'healthy',
            'experiment': app_config.research.experiment_name,
            'version': '2.0.0'
        }, 200

    @app.route('/')
    def serve_index():
        """Serve the main application page."""
        return send_from_directory('static', 'index.html')

    @app.route('/favicon.ico')
    def favicon():
        """Return empty favicon to prevent 404 errors."""
        return '', 204

    # ==========================================================================
    # FILE SERVING ROUTES
    # ==========================================================================
    
    @app.route('/uploads/<filename>')
    def serve_video(filename):
        """Serve uploaded video files."""
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    @app.route('/cuts/<filename>')
    def serve_cut(filename):
        """Serve cut video files."""
        return send_from_directory(app.config['CUTS_FOLDER'], filename)

    # ==========================================================================
    # LEGACY ROUTE COMPATIBILITY
    # These routes maintain backward compatibility with the existing frontend.
    # They delegate to the blueprint handlers.
    # ==========================================================================
    
    @app.route('/upload', methods=['POST'])
    def upload_video_legacy():
        """
        Legacy upload endpoint for backward compatibility.
        Delegates to /api/video/upload.
        """
        from app.routes.video_routes import upload_video
        return upload_video()

    @app.route('/cut', methods=['POST'])
    def cut_video_legacy():
        """
        Legacy cut endpoint for backward compatibility.
        Delegates to /api/video/cut.
        """
        from app.routes.video_routes import cut_video
        return cut_video()

    @app.route('/check-subtitles/<filename>')
    def check_subtitles_legacy(filename):
        """
        Legacy subtitle check endpoint.
        Delegates to /api/subtitles/check/<filename>.
        """
        from app.routes.subtitle_routes import check_video_subtitles
        return check_video_subtitles(filename)

    @app.route('/extract-subtitles/<filename>')
    def extract_subtitles_legacy(filename):
        """
        Legacy subtitle extraction endpoint.
        Delegates to /api/subtitles/extract/<filename>.
        """
        from app.routes.subtitle_routes import get_subtitles
        return get_subtitles(filename)

    @app.route('/get-subtitles/<filename>')
    def get_subtitles_json_legacy(filename):
        """
        Legacy subtitle JSON endpoint.
        Delegates to /api/subtitles/get/<filename>.
        """
        from app.routes.subtitle_routes import get_subtitles_json
        return get_subtitles_json(filename)

    # ==========================================================================
    # ERROR HANDLERS
    # ==========================================================================
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        """Handle file too large errors."""
        return jsonify({
            'error': f'File is too large. Maximum size is {app_config.video.max_file_size_bytes / (1024*1024*1024):.1f}GB.'
        }), 413

    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors."""
        return jsonify({'error': 'Resource not found'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors."""
        logger.error(f"Internal server error: {error}")
        return jsonify({'error': 'Internal server error'}), 500

    return app


if __name__ == '__main__':
    app = create_app()
    flask_config = get_config().flask
    app.run(
        debug=flask_config.debug, 
        host=flask_config.host, 
        port=flask_config.port
    )
