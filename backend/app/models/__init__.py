"""
Data Models Package
===================
Exports all data model classes for the Movie Shorts pipeline.

Usage:
    from app.models import VideoMetadata, Segment, AnalysisResult
    from app.models import TextFeatures, AudioFeatures, VisualFeatures
"""

from .schemas import (
    # Enums
    SegmentType,
    ModalityType,
    ProcessingStatus,
    
    # Base
    BaseModel,
    
    # Video
    VideoMetadata,
    
    # Subtitles
    SubtitleEntry,
    SubtitleData,
    
    # Features
    TextFeatures,
    AudioFeatures,
    VisualFeatures,
    SegmentFeatures,
    
    # Scoring
    ScoreBreakdown,
    Segment,
    
    # Pipeline outputs
    AnalysisResult,
    
    # Evaluation
    GroundTruthSegment,
    EvaluationResult,
    
    # Utilities
    generate_segment_id,
    generate_result_id,
)

__all__ = [
    # Enums
    'SegmentType',
    'ModalityType', 
    'ProcessingStatus',
    
    # Base
    'BaseModel',
    
    # Video
    'VideoMetadata',
    
    # Subtitles
    'SubtitleEntry',
    'SubtitleData',
    
    # Features
    'TextFeatures',
    'AudioFeatures',
    'VisualFeatures',
    'SegmentFeatures',
    
    # Scoring
    'ScoreBreakdown',
    'Segment',
    
    # Pipeline outputs
    'AnalysisResult',
    
    # Evaluation
    'GroundTruthSegment',
    'EvaluationResult',
    
    # Utilities
    'generate_segment_id',
    'generate_result_id',
]

