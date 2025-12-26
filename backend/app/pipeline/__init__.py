"""
Pipeline Package
================
Modular, composable pipeline for video analysis.

This package provides:
- Individual pipeline stages that can run independently
- A composed pipeline that runs all stages in sequence
- Caching support for expensive operations
- Progress tracking and logging

Usage:
    from app.pipeline import VideoPipeline, PipelineContext
    
    # Run full pipeline
    pipeline = VideoPipeline()
    result = pipeline.run("video.mp4")
    
    # Run individual stages
    from app.pipeline.stages import TranscriptionStage
    stage = TranscriptionStage()
    subtitles = stage.run(context)
"""

from .context import PipelineContext
from .base import PipelineStage
from .pipeline import VideoPipeline

__all__ = [
    'PipelineContext',
    'PipelineStage', 
    'VideoPipeline',
]

