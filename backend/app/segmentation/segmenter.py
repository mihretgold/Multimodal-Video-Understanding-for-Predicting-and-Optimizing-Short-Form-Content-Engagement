"""
Temporal Segmenter Module
=========================
Main interface for temporal video segmentation.

The TemporalSegmenter provides a unified interface for running
different segmentation strategies on video content.
"""

import logging
from typing import List, Optional, Dict, Any

from .strategies import (
    SegmentationStrategy,
    SegmentationParams,
    PauseBasedStrategy,
    FixedWindowStrategy,
    SemanticBoundaryStrategy,
    HybridStrategy,
)
from .boundaries import BoundaryDetector, SpeechBoundary
from ..models import SubtitleData, Segment
from ..config import SegmentationConfig, get_config
from ..logging_config import get_research_logger, log_pipeline_decision

logger = get_research_logger("segmentation")


# Strategy registry
STRATEGY_REGISTRY = {
    'pause_based': PauseBasedStrategy,
    'fixed_window': FixedWindowStrategy,
    'semantic_boundary': SemanticBoundaryStrategy,
    'hybrid': HybridStrategy,
}


class TemporalSegmenter:
    """
    Main class for temporal video segmentation.
    
    Provides a unified interface for:
    - Running segmentation with configurable strategies
    - Validating segment constraints
    - Logging segmentation decisions
    
    Usage:
        segmenter = TemporalSegmenter()
        segments = segmenter.segment(subtitle_data, video_duration)
        
        # Or with specific strategy
        segmenter = TemporalSegmenter(strategy='pause_based')
        segments = segmenter.segment(subtitle_data, video_duration)
    """
    
    def __init__(
        self,
        strategy: Optional[str] = None,
        config: Optional[SegmentationConfig] = None,
        custom_strategy: Optional[SegmentationStrategy] = None
    ):
        """
        Initialize the temporal segmenter.
        
        Args:
            strategy: Strategy name ('pause_based', 'fixed_window', 'semantic_boundary', 'hybrid')
            config: Segmentation configuration (default: from global config)
            custom_strategy: Custom strategy instance (overrides strategy name)
        """
        self.config = config or get_config().segmentation
        
        # Initialize strategy
        if custom_strategy:
            self.strategy = custom_strategy
        elif strategy:
            if strategy not in STRATEGY_REGISTRY:
                raise ValueError(f"Unknown strategy: {strategy}. Available: {list(STRATEGY_REGISTRY.keys())}")
            self.strategy = STRATEGY_REGISTRY[strategy]()
        else:
            # Default to pause-based
            self.strategy = PauseBasedStrategy()
        
        # Create parameters from config
        self.params = SegmentationParams.from_config(self.config)
        
        logger.info(f"Initialized TemporalSegmenter with strategy '{self.strategy.name}'")
    
    def segment(
        self,
        subtitle_data: SubtitleData,
        video_duration: float,
        result_id: Optional[str] = None
    ) -> List[Segment]:
        """
        Segment video content using the configured strategy.
        
        Args:
            subtitle_data: Transcription data with timestamps
            video_duration: Total video duration in seconds
            result_id: Optional result ID for logging
            
        Returns:
            List of Segment objects
        """
        logger.info(
            f"Starting segmentation",
            extra={
                'strategy': self.strategy.name,
                'video_duration': video_duration,
                'subtitle_count': len(subtitle_data.entries),
                'result_id': result_id
            }
        )
        
        # Run strategy
        segments = self.strategy.segment(
            subtitle_data,
            video_duration,
            self.params
        )
        
        # Validate and post-process
        segments = self._validate_segments(segments, video_duration)
        
        # Log decision
        log_pipeline_decision(
            "segmentation_complete",
            {
                'strategy': self.strategy.name,
                'segment_count': len(segments),
                'total_coverage': sum(s.duration_seconds for s in segments),
                'video_duration': video_duration,
                'segments': [
                    {
                        'id': s.segment_id,
                        'start': s.start_seconds,
                        'end': s.end_seconds,
                        'duration': s.duration_seconds
                    }
                    for s in segments
                ]
            },
            result_id=result_id
        )
        
        return segments
    
    def _validate_segments(
        self,
        segments: List[Segment],
        video_duration: float
    ) -> List[Segment]:
        """
        Validate and fix segment constraints.
        
        - Ensures segments are within video bounds
        - Removes invalid segments
        - Logs any issues
        """
        valid = []
        
        for segment in segments:
            # Clamp to video bounds
            start = max(0, segment.start_seconds)
            end = min(video_duration, segment.end_seconds)
            
            # Update segment if clamped
            if start != segment.start_seconds or end != segment.end_seconds:
                segment.start_seconds = start
                segment.end_seconds = end
                logger.debug(f"Clamped segment {segment.segment_id} to video bounds")
            
            # Check duration
            duration = segment.duration_seconds
            if duration < self.params.min_duration_seconds:
                logger.warning(
                    f"Segment {segment.segment_id} too short ({duration:.1f}s < {self.params.min_duration_seconds}s)"
                )
                continue
            
            if duration > self.params.max_duration_seconds:
                logger.warning(
                    f"Segment {segment.segment_id} too long ({duration:.1f}s > {self.params.max_duration_seconds}s)"
                )
                # Keep it but log warning
            
            valid.append(segment)
        
        # Check segment count
        if len(valid) < self.params.min_segments:
            logger.warning(
                f"Too few segments: {len(valid)} < {self.params.min_segments}"
            )
        
        if len(valid) > self.params.max_segments:
            logger.info(
                f"Limiting segments from {len(valid)} to {self.params.max_segments}"
            )
            # Keep the best segments (for now, just truncate)
            # In future, could use scoring to select
            valid = valid[:self.params.max_segments]
        
        return valid
    
    def detect_boundaries(
        self,
        subtitle_data: SubtitleData,
        video_duration: float
    ) -> List[SpeechBoundary]:
        """
        Detect potential segment boundaries without creating segments.
        
        Useful for visualization or custom segmentation logic.
        
        Args:
            subtitle_data: Transcription data
            video_duration: Video duration
            
        Returns:
            List of SpeechBoundary objects
        """
        detector = BoundaryDetector()
        return detector.detect_from_subtitles(
            subtitle_data.entries,
            video_duration
        )
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about the current strategy."""
        return {
            'name': self.strategy.name,
            'description': self.strategy.description,
            'params': {
                'min_duration': self.params.min_duration_seconds,
                'target_duration': self.params.target_duration_seconds,
                'max_duration': self.params.max_duration_seconds,
                'min_segments': self.params.min_segments,
                'max_segments': self.params.max_segments,
            }
        }


def create_segmenter(
    strategy: str = 'pause_based',
    **kwargs
) -> TemporalSegmenter:
    """
    Factory function to create a segmenter.
    
    Args:
        strategy: Strategy name
        **kwargs: Additional parameters for the segmenter
        
    Returns:
        Configured TemporalSegmenter instance
    """
    return TemporalSegmenter(strategy=strategy, **kwargs)


def segment_video(
    subtitle_data: SubtitleData,
    video_duration: float,
    strategy: str = 'pause_based',
    config: Optional[SegmentationConfig] = None
) -> List[Segment]:
    """
    Convenience function to segment a video.
    
    Args:
        subtitle_data: Transcription data
        video_duration: Video duration in seconds
        strategy: Segmentation strategy name
        config: Optional configuration
        
    Returns:
        List of Segment objects
    """
    segmenter = TemporalSegmenter(strategy=strategy, config=config)
    return segmenter.segment(subtitle_data, video_duration)

