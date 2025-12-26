"""
Temporal Segmentation Module
============================
Research-grade temporal segmentation for video analysis.

This module implements multiple segmentation strategies:
- Pause-based: Uses speech pauses from transcription timestamps
- Fixed-window: Regular intervals with boundary optimization
- Semantic: Groups semantically coherent subtitle blocks

Usage:
    from app.segmentation import TemporalSegmenter, PauseBasedStrategy
    
    segmenter = TemporalSegmenter(strategy=PauseBasedStrategy())
    segments = segmenter.segment(subtitle_data, video_duration)
"""

from .strategies import (
    SegmentationStrategy,
    PauseBasedStrategy,
    FixedWindowStrategy,
    SemanticBoundaryStrategy,
    HybridStrategy,
)
from .segmenter import TemporalSegmenter
from .boundaries import BoundaryDetector, SpeechBoundary

__all__ = [
    'TemporalSegmenter',
    'SegmentationStrategy',
    'PauseBasedStrategy',
    'FixedWindowStrategy',
    'SemanticBoundaryStrategy',
    'HybridStrategy',
    'BoundaryDetector',
    'SpeechBoundary',
]

