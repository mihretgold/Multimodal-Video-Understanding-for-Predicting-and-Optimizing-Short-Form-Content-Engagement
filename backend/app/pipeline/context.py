"""
Pipeline Context Module
=======================
Defines the shared context that flows through all pipeline stages.

The PipelineContext holds:
- Input parameters (video path, configuration)
- Intermediate results (metadata, subtitles, features)
- Final outputs (segments, scores, analysis result)
- Execution metadata (timing, stage completion status)
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List

from ..config import AppConfig, get_config
from ..models import (
    VideoMetadata,
    SubtitleData,
    Segment,
    SegmentFeatures,
    AnalysisResult,
    generate_result_id,
)

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Result of a single pipeline stage execution."""
    stage_name: str
    success: bool
    duration_seconds: float
    error_message: Optional[str] = None
    output_summary: Optional[str] = None


@dataclass
class PipelineContext:
    """
    Shared context that flows through all pipeline stages.
    
    This object is passed between stages and accumulates results.
    Each stage reads what it needs and writes its outputs.
    
    Attributes:
        # Input parameters
        video_path: Path to the input video file
        config: Configuration to use for this pipeline run
        
        # Video information
        video_metadata: Extracted video metadata
        
        # Subtitle data
        subtitle_data: Extracted/transcribed subtitles
        
        # Segments and features
        candidate_segments: Initial segment candidates (before scoring)
        segment_features: Extracted features per segment
        scored_segments: Segments with scores and rankings
        
        # Final output
        analysis_result: Complete pipeline output
        
        # Execution tracking
        result_id: Unique identifier for this pipeline run
        stage_results: Timing and status for each stage
        started_at: Pipeline start timestamp
        completed_at: Pipeline completion timestamp
    """
    
    # Input parameters
    video_path: str
    config: AppConfig = field(default_factory=get_config)
    
    # Video information
    video_metadata: Optional[VideoMetadata] = None
    
    # Subtitle data  
    subtitle_data: Optional[SubtitleData] = None
    
    # Segments and features
    candidate_segments: List[Segment] = field(default_factory=list)
    segment_features: Dict[str, SegmentFeatures] = field(default_factory=dict)
    scored_segments: List[Segment] = field(default_factory=list)
    
    # Final output
    analysis_result: Optional[AnalysisResult] = None
    
    # Execution tracking
    result_id: str = field(default_factory=lambda: generate_result_id("pipeline"))
    stage_results: List[StageResult] = field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    def __post_init__(self):
        """Initialize timestamps and validate inputs."""
        self.started_at = datetime.now().isoformat()
        
        # Validate video path exists
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
    
    @property
    def video_filename(self) -> str:
        """Get the video filename from path."""
        return os.path.basename(self.video_path)
    
    @property
    def total_duration_seconds(self) -> float:
        """Get total pipeline execution time."""
        return sum(r.duration_seconds for r in self.stage_results)
    
    @property
    def successful_stages(self) -> List[str]:
        """Get list of successfully completed stage names."""
        return [r.stage_name for r in self.stage_results if r.success]
    
    @property
    def failed_stages(self) -> List[str]:
        """Get list of failed stage names."""
        return [r.stage_name for r in self.stage_results if not r.success]
    
    @property
    def is_complete(self) -> bool:
        """Check if pipeline completed without failures."""
        return len(self.failed_stages) == 0 and self.analysis_result is not None
    
    def record_stage(
        self, 
        stage_name: str, 
        success: bool, 
        duration: float,
        error: Optional[str] = None,
        summary: Optional[str] = None
    ) -> None:
        """Record the result of a stage execution."""
        self.stage_results.append(StageResult(
            stage_name=stage_name,
            success=success,
            duration_seconds=duration,
            error_message=error,
            output_summary=summary
        ))
        
        if success:
            logger.info(f"Stage '{stage_name}' completed in {duration:.2f}s: {summary or 'OK'}")
        else:
            logger.error(f"Stage '{stage_name}' failed after {duration:.2f}s: {error}")
    
    def get_cache_key(self, stage_name: str) -> str:
        """
        Generate a cache key for a stage based on inputs.
        
        The cache key incorporates:
        - Video file hash (first 1MB)
        - Stage name
        - Relevant configuration parameters
        """
        # Read first 1MB of video for hash
        with open(self.video_path, 'rb') as f:
            video_sample = f.read(1024 * 1024)
        video_hash = hashlib.md5(video_sample).hexdigest()[:12]
        
        # Include relevant config in cache key
        config_str = f"{self.config.whisper.model_size}_{self.config.ablation.mode_name}"
        
        return f"{stage_name}_{video_hash}_{config_str}"
    
    def get_cache_path(self, stage_name: str) -> Path:
        """Get the file path for caching a stage's output."""
        cache_key = self.get_cache_key(stage_name)
        cache_dir = self.config.paths.features / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{cache_key}.json"
    
    def save_checkpoint(self, checkpoint_name: str) -> str:
        """
        Save current context state as a checkpoint.
        
        Returns the checkpoint file path.
        """
        checkpoint_dir = self.config.paths.experiments / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{self.result_id}_{checkpoint_name}.json"
        
        checkpoint_data = {
            'result_id': self.result_id,
            'video_path': self.video_path,
            'video_filename': self.video_filename,
            'started_at': self.started_at,
            'checkpoint_at': datetime.now().isoformat(),
            'checkpoint_name': checkpoint_name,
            'video_metadata': self.video_metadata.to_dict() if self.video_metadata else None,
            'subtitle_data': self.subtitle_data.to_dict() if self.subtitle_data else None,
            'candidate_segments': [s.to_dict() for s in self.candidate_segments],
            'scored_segments': [s.to_dict() for s in self.scored_segments],
            'stage_results': [
                {
                    'stage_name': r.stage_name,
                    'success': r.success,
                    'duration_seconds': r.duration_seconds,
                    'error_message': r.error_message,
                    'output_summary': r.output_summary
                }
                for r in self.stage_results
            ],
            'config_snapshot': self.config.to_dict()
        }
        
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"Saved checkpoint '{checkpoint_name}' to {checkpoint_path}")
        return str(checkpoint_path)
    
    @classmethod
    def load_checkpoint(cls, checkpoint_path: str) -> "PipelineContext":
        """Load context from a checkpoint file."""
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        context = cls(
            video_path=data['video_path'],
        )
        context.result_id = data['result_id']
        context.started_at = data['started_at']
        
        if data.get('video_metadata'):
            context.video_metadata = VideoMetadata(**data['video_metadata'])
        
        if data.get('subtitle_data'):
            context.subtitle_data = SubtitleData.from_dict(data['subtitle_data'])
        
        context.candidate_segments = [
            Segment.from_dict(s) for s in data.get('candidate_segments', [])
        ]
        context.scored_segments = [
            Segment.from_dict(s) for s in data.get('scored_segments', [])
        ]
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return context
    
    def finalize(self) -> AnalysisResult:
        """
        Finalize the pipeline and create the AnalysisResult.
        
        This should be called after all stages complete.
        """
        self.completed_at = datetime.now().isoformat()
        
        if self.video_metadata is None:
            raise ValueError("Cannot finalize: video_metadata is missing")
        
        # Use scored segments if available, otherwise candidates
        final_segments = self.scored_segments if self.scored_segments else self.candidate_segments
        
        self.analysis_result = AnalysisResult(
            result_id=self.result_id,
            video_metadata=self.video_metadata,
            subtitle_data=self.subtitle_data,
            segments=final_segments,
            config_snapshot=self.config.to_dict(),
            processing_time_seconds=self.total_duration_seconds,
            ablation_mode=self.config.ablation.mode_name
        )
        
        return self.analysis_result
    
    def save_result(self, output_dir: Optional[str] = None) -> str:
        """
        Save the final analysis result to disk.
        
        Returns the output file path.
        """
        if self.analysis_result is None:
            self.finalize()
        
        if output_dir is None:
            output_dir = self.config.paths.experiments / "results"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        result_file = output_path / f"{self.result_id}.json"
        self.analysis_result.save(str(result_file))
        
        logger.info(f"Saved analysis result to {result_file}")
        return str(result_file)

