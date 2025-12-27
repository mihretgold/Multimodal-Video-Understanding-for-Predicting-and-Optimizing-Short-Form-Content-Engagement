"""
Pipeline Routes Module
======================
REST API endpoints for running the video analysis pipeline.

Endpoints:
- POST /api/pipeline/run          - Run full pipeline on a video
- POST /api/pipeline/run-stages   - Run specific pipeline stages
- GET  /api/pipeline/status/<id>  - Get status of a pipeline run
- GET  /api/pipeline/result/<id>  - Get results of a completed run
- POST /api/pipeline/ablation     - Run ablation study
"""

from flask import Blueprint, request, jsonify, current_app
import os
import logging
import traceback
from pathlib import Path

from ..pipeline import VideoPipeline, PipelineContext
from ..pipeline.pipeline import run_ablation
from ..config import get_config, AblationConfig
from ..logging_config import get_research_logger, log_pipeline_decision

# Configure logging
logger = get_research_logger("routes.pipeline", log_to_file=False)

# Create blueprint
pipeline_bp = Blueprint('pipeline', __name__)


@pipeline_bp.route('/run', methods=['POST'])
def run_pipeline():
    """
    Run the full analysis pipeline on an uploaded video.
    
    Request JSON:
        {
            "filename": "video.mp4",           # Required: uploaded video filename
            "use_cache": true,                 # Optional: use cached results (default: true)
            "save_results": true,              # Optional: save results to disk (default: true)
            "ablation_mode": "full_multimodal" # Optional: ablation mode
        }
        
    Response JSON:
        {
            "success": true,
            "result_id": "abc123...",
            "segments": [...],
            "processing_time_seconds": 45.2,
            "ablation_mode": "full_multimodal"
        }
    """
    try:
        data = request.json or {}
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        # Get config
        config = get_config()
        
        # Apply ablation mode if specified
        ablation_mode = data.get('ablation_mode', 'full_multimodal')
        ablation_map = {
            'text_only': AblationConfig.text_only,
            'audio_only': AblationConfig.audio_only,
            'visual_only': AblationConfig.visual_only,
            'visual_signal_only': AblationConfig.visual_signal_only,  # Visual without CV
            'text_audio': AblationConfig.text_audio,
            'full_no_cv': AblationConfig.full_no_cv,  # All modalities, no CV features
            'full_multimodal': AblationConfig.full_multimodal,
        }
        if ablation_mode in ablation_map:
            config.ablation = ablation_map[ablation_mode]()
        
        # Build video path
        video_path = os.path.join(config.paths.uploads, filename)
        
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video file not found'}), 404
        
        # Run pipeline
        use_cache = data.get('use_cache', True)
        save_results = data.get('save_results', True)
        
        pipeline = VideoPipeline(use_cache=use_cache, save_results=save_results)
        
        logger.info(f"Starting pipeline for {filename}", extra={
            'video_filename': filename,
            'ablation_mode': ablation_mode,
            'use_cache': use_cache
        })
        
        result = pipeline.run(video_path, config=config)
        
        log_pipeline_decision(
            "pipeline_complete",
            {
                'video_filename': filename,
                'segment_count': result.segment_count,
                'ablation_mode': ablation_mode
            },
            result_id=result.result_id
        )
        
        return jsonify({
            'success': True,
            'result_id': result.result_id,
            'segments': [s.to_dict() for s in result.segments],
            'segment_count': result.segment_count,
            'processing_time_seconds': result.processing_time_seconds,
            'ablation_mode': result.ablation_mode
        }), 200
        
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Pipeline error: {str(e)}'}), 500


@pipeline_bp.route('/run-stages', methods=['POST'])
def run_pipeline_stages():
    """
    Run specific pipeline stages on a video.
    
    Request JSON:
        {
            "filename": "video.mp4",
            "stages": ["video_ingest", "transcription", "segment_detection"]
        }
        
    Response JSON:
        {
            "success": true,
            "result_id": "abc123...",
            "completed_stages": ["video_ingest", "transcription", "segment_detection"],
            "stage_results": [...]
        }
    """
    try:
        data = request.json or {}
        filename = data.get('filename')
        stages = data.get('stages', [])
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        if not stages:
            return jsonify({'error': 'No stages specified'}), 400
        
        config = get_config()
        video_path = os.path.join(config.paths.uploads, filename)
        
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video file not found'}), 404
        
        # Run pipeline with specific stages
        pipeline = VideoPipeline(use_cache=True, save_results=False)
        result = pipeline.run(video_path, config=config, stage_names=stages)
        
        return jsonify({
            'success': True,
            'result_id': result.result_id,
            'completed_stages': stages,
            'processing_time_seconds': result.processing_time_seconds
        }), 200
        
    except Exception as e:
        logger.error(f"Pipeline stages error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@pipeline_bp.route('/ablation', methods=['POST'])
def run_ablation_study():
    """
    Run ablation study with multiple modality configurations.
    
    Request JSON:
        {
            "filename": "video.mp4",
            "modes": ["text_only", "audio_only", "full_multimodal"]  # Optional
        }
        
    Response JSON:
        {
            "success": true,
            "results": {
                "text_only": { "result_id": "...", "segment_count": 5, ... },
                "audio_only": { "result_id": "...", "segment_count": 4, ... },
                ...
            }
        }
    """
    try:
        data = request.json or {}
        filename = data.get('filename')
        modes = data.get('modes')  # None means all modes
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        config = get_config()
        video_path = os.path.join(config.paths.uploads, filename)
        
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video file not found'}), 404
        
        logger.info(f"Starting ablation study for {filename}", extra={
            'video_filename': filename,
            'modes': modes or 'all'
        })
        
        # Run ablation
        results = run_ablation(video_path, ablation_modes=modes)
        
        # Format results
        formatted_results = {}
        for mode_name, result in results.items():
            if result is not None:
                formatted_results[mode_name] = {
                    'result_id': result.result_id,
                    'segment_count': result.segment_count,
                    'processing_time_seconds': result.processing_time_seconds,
                    'top_segment': result.top_segment.to_dict() if result.top_segment else None
                }
            else:
                formatted_results[mode_name] = {'error': 'Failed'}
        
        log_pipeline_decision(
            "ablation_study_complete",
            {
                'video_filename': filename,
                'modes_run': list(results.keys()),
                'modes_succeeded': [k for k, v in results.items() if v is not None]
            }
        )
        
        return jsonify({
            'success': True,
            'filename': filename,
            'results': formatted_results
        }), 200
        
    except Exception as e:
        logger.error(f"Ablation study error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@pipeline_bp.route('/result/<result_id>', methods=['GET'])
def get_pipeline_result(result_id):
    """
    Get the results of a completed pipeline run.
    
    Args:
        result_id: The unique result ID from a pipeline run
        
    Response JSON:
        Full AnalysisResult object
    """
    try:
        config = get_config()
        results_dir = config.paths.experiments / "results"
        result_file = results_dir / f"{result_id}.json"
        
        if not result_file.exists():
            return jsonify({'error': 'Result not found'}), 404
        
        from ..models import AnalysisResult
        result = AnalysisResult.load(str(result_file))
        
        return jsonify(result.to_dict()), 200
        
    except Exception as e:
        logger.error(f"Error loading result: {str(e)}")
        return jsonify({'error': str(e)}), 500


@pipeline_bp.route('/results', methods=['GET'])
def list_pipeline_results():
    """
    List all pipeline results.
    
    Query params:
        limit: Maximum number of results to return (default: 20)
        
    Response JSON:
        {
            "results": [
                {"result_id": "...", "filename": "...", "created_at": "..."},
                ...
            ]
        }
    """
    try:
        config = get_config()
        results_dir = config.paths.experiments / "results"
        
        if not results_dir.exists():
            return jsonify({'results': []}), 200
        
        limit = request.args.get('limit', 20, type=int)
        
        # Get all result files, sorted by modification time
        result_files = sorted(
            results_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )[:limit]
        
        results = []
        from ..models import AnalysisResult
        
        for result_file in result_files:
            try:
                result = AnalysisResult.load(str(result_file))
                results.append({
                    'result_id': result.result_id,
                    'filename': result.video_metadata.filename,
                    'segment_count': result.segment_count,
                    'ablation_mode': result.ablation_mode,
                    'created_at': result.created_at
                })
            except Exception:
                continue
        
        return jsonify({'results': results}), 200
        
    except Exception as e:
        logger.error(f"Error listing results: {str(e)}")
        return jsonify({'error': str(e)}), 500


@pipeline_bp.route('/stages', methods=['GET'])
def list_pipeline_stages():
    """
    List all available pipeline stages.
    
    Response JSON:
        {
            "stages": [
                {"name": "video_ingest", "description": "..."},
                ...
            ]
        }
    """
    from ..pipeline.stages import ALL_STAGES
    
    stages = []
    for stage_class in ALL_STAGES:
        stage = stage_class() if stage_class.__name__ != 'OutputStage' else stage_class(save_to_disk=False)
        stages.append({
            'name': stage.name,
            'description': stage.description,
            'cacheable': stage.cacheable
        })
    
    return jsonify({'stages': stages}), 200

