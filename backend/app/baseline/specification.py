"""
Baseline Specification Module
=============================
Defines the formal specification of the baseline video analysis system.

This module documents:
- Input requirements and validation
- Intermediate representations at each pipeline stage
- Output format and validation
- Evaluation criteria

The specification serves as the contract for reproducibility.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import json

from ..models import (
    VideoMetadata,
    SubtitleData,
    Segment,
    SegmentFeatures,
    AnalysisResult,
)
from ..config import AppConfig, get_config


# =============================================================================
# BASELINE SPECIFICATION
# =============================================================================

@dataclass
class InputSpec:
    """
    Specification for valid pipeline inputs.
    
    A valid input must satisfy:
    - Video file exists and is readable
    - Video format is in allowed extensions
    - Video duration is within acceptable range
    - Video has audio track (for transcription)
    """
    
    # Allowed video formats
    allowed_extensions: List[str] = field(default_factory=lambda: [
        'mp4', 'avi', 'mov', 'mkv', 'webm'
    ])
    
    # Duration constraints (seconds)
    min_duration_seconds: float = 60.0  # At least 1 minute
    max_duration_seconds: float = 7200.0  # At most 2 hours
    
    # File size constraints (bytes)
    max_file_size_bytes: int = 2 * 1024 * 1024 * 1024  # 2GB
    
    def validate(self, video_path: str, metadata: VideoMetadata = None) -> tuple:
        """
        Validate that a video meets input requirements.
        
        Returns:
            (is_valid, error_message)
        """
        path = Path(video_path)
        
        # Check file exists
        if not path.exists():
            return False, f"Video file not found: {video_path}"
        
        # Check extension
        ext = path.suffix.lower().lstrip('.')
        if ext not in self.allowed_extensions:
            return False, f"Invalid format: {ext}. Allowed: {self.allowed_extensions}"
        
        # Check file size
        file_size = path.stat().st_size
        if file_size > self.max_file_size_bytes:
            return False, f"File too large: {file_size / 1e9:.2f}GB > {self.max_file_size_bytes / 1e9:.2f}GB"
        
        # Check duration if metadata provided
        if metadata:
            if metadata.duration_seconds < self.min_duration_seconds:
                return False, f"Video too short: {metadata.duration_seconds:.1f}s < {self.min_duration_seconds}s"
            if metadata.duration_seconds > self.max_duration_seconds:
                return False, f"Video too long: {metadata.duration_seconds:.1f}s > {self.max_duration_seconds}s"
        
        return True, ""


@dataclass
class IntermediateSpec:
    """
    Specification for intermediate representations.
    
    Defines what each pipeline stage should produce.
    """
    
    # Stage 1: Video Ingest -> VideoMetadata
    video_metadata_required_fields: List[str] = field(default_factory=lambda: [
        'filename', 'filepath', 'duration_seconds', 'fps', 'width', 'height'
    ])
    
    # Stage 2: Transcription -> SubtitleData
    subtitle_data_required_fields: List[str] = field(default_factory=lambda: [
        'video_filename', 'entries', 'source', 'language'
    ])
    min_subtitle_entries: int = 1
    
    # Stage 3: Segment Detection -> List[Segment]
    segment_required_fields: List[str] = field(default_factory=lambda: [
        'segment_id', 'start_seconds', 'end_seconds', 'segment_type'
    ])
    min_segments: int = 1
    max_segments: int = 20
    
    # Stage 4: Feature Extraction -> SegmentFeatures
    # (Depends on ablation mode - may have partial features)
    
    # Stage 5: Scoring -> Segments with scores
    score_required_fields: List[str] = field(default_factory=lambda: [
        'total_score', 'text_score', 'audio_score', 'visual_score'
    ])

    def validate_video_metadata(self, metadata: VideoMetadata) -> tuple:
        """Validate VideoMetadata meets specification."""
        if metadata is None:
            return False, "VideoMetadata is None"
        
        for field_name in self.video_metadata_required_fields:
            if not hasattr(metadata, field_name):
                return False, f"Missing field: {field_name}"
            if getattr(metadata, field_name) is None:
                return False, f"Field is None: {field_name}"
        
        return True, ""
    
    def validate_subtitle_data(self, subtitle_data: SubtitleData) -> tuple:
        """Validate SubtitleData meets specification."""
        if subtitle_data is None:
            return False, "SubtitleData is None"
        
        for field_name in self.subtitle_data_required_fields:
            if not hasattr(subtitle_data, field_name):
                return False, f"Missing field: {field_name}"
        
        if len(subtitle_data.entries) < self.min_subtitle_entries:
            return False, f"Too few subtitle entries: {len(subtitle_data.entries)}"
        
        return True, ""
    
    def validate_segments(self, segments: List[Segment]) -> tuple:
        """Validate segment list meets specification."""
        if not segments:
            return False, "No segments produced"
        
        if len(segments) < self.min_segments:
            return False, f"Too few segments: {len(segments)} < {self.min_segments}"
        
        if len(segments) > self.max_segments:
            return False, f"Too many segments: {len(segments)} > {self.max_segments}"
        
        for i, segment in enumerate(segments):
            for field_name in self.segment_required_fields:
                if not hasattr(segment, field_name):
                    return False, f"Segment {i} missing field: {field_name}"
            
            # Validate segment duration
            duration = segment.end_seconds - segment.start_seconds
            if duration <= 0:
                return False, f"Segment {i} has invalid duration: {duration}"
        
        return True, ""


@dataclass
class OutputSpec:
    """
    Specification for final pipeline outputs.
    
    Defines the required structure and content of AnalysisResult.
    """
    
    # Required top-level fields
    required_fields: List[str] = field(default_factory=lambda: [
        'result_id', 'video_metadata', 'segments', 'processing_time_seconds',
        'ablation_mode', 'created_at', 'version'
    ])
    
    # Schema version for compatibility
    schema_version: str = "1.0.0"
    
    def validate(self, result: AnalysisResult) -> tuple:
        """Validate AnalysisResult meets output specification."""
        if result is None:
            return False, "AnalysisResult is None"
        
        # Check required fields
        for field_name in self.required_fields:
            if not hasattr(result, field_name):
                return False, f"Missing field: {field_name}"
        
        # Check result_id format
        if not result.result_id or len(result.result_id) < 8:
            return False, f"Invalid result_id: {result.result_id}"
        
        # Check video_metadata
        if result.video_metadata is None:
            return False, "video_metadata is None"
        
        # Check segments exist
        if not result.segments:
            return False, "No segments in result"
        
        # Check processing time is reasonable
        if result.processing_time_seconds <= 0:
            return False, "Invalid processing time"
        
        # Check version
        if result.version != self.schema_version:
            return False, f"Schema version mismatch: {result.version} != {self.schema_version}"
        
        return True, ""


@dataclass
class BaselineSpec:
    """
    Complete baseline system specification.
    
    Combines input, intermediate, and output specifications.
    """
    
    input_spec: InputSpec = field(default_factory=InputSpec)
    intermediate_spec: IntermediateSpec = field(default_factory=IntermediateSpec)
    output_spec: OutputSpec = field(default_factory=OutputSpec)
    
    # Baseline identification
    baseline_name: str = "multimodal_video_baseline_v1"
    baseline_version: str = "1.0.0"
    
    # Default configuration for baseline
    default_whisper_model: str = "small"
    default_gemini_model: str = "gemini-1.5-flash"
    default_segment_duration_range: tuple = (60.0, 70.0)  # seconds
    
    def to_dict(self) -> dict:
        """Export specification as dictionary."""
        return {
            'baseline_name': self.baseline_name,
            'baseline_version': self.baseline_version,
            'input_spec': {
                'allowed_extensions': self.input_spec.allowed_extensions,
                'min_duration_seconds': self.input_spec.min_duration_seconds,
                'max_duration_seconds': self.input_spec.max_duration_seconds,
                'max_file_size_bytes': self.input_spec.max_file_size_bytes,
            },
            'intermediate_spec': {
                'video_metadata_required_fields': self.intermediate_spec.video_metadata_required_fields,
                'subtitle_data_required_fields': self.intermediate_spec.subtitle_data_required_fields,
                'segment_required_fields': self.intermediate_spec.segment_required_fields,
                'min_segments': self.intermediate_spec.min_segments,
                'max_segments': self.intermediate_spec.max_segments,
            },
            'output_spec': {
                'required_fields': self.output_spec.required_fields,
                'schema_version': self.output_spec.schema_version,
            },
            'defaults': {
                'whisper_model': self.default_whisper_model,
                'gemini_model': self.default_gemini_model,
                'segment_duration_range': self.default_segment_duration_range,
            }
        }
    
    def save(self, filepath: str) -> None:
        """Save specification to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# BASELINE OUTPUT WRAPPER
# =============================================================================

@dataclass
class BaselineOutput:
    """
    Wrapper for baseline pipeline output with full provenance.
    
    Contains:
    - The analysis result
    - Validation status
    - Specification used
    - Execution metadata
    """
    
    # Core result
    result: AnalysisResult
    
    # Validation
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    
    # Provenance
    spec_version: str = "1.0.0"
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    # Execution metadata
    executed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    execution_host: str = ""
    
    # Stage-level details
    stage_outputs: Dict[str, Dict] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Export as dictionary for serialization."""
        return {
            'result': self.result.to_dict() if self.result else None,
            'validation': {
                'is_valid': self.is_valid,
                'errors': self.validation_errors
            },
            'provenance': {
                'spec_version': self.spec_version,
                'config_snapshot': self.config_snapshot,
                'executed_at': self.executed_at,
                'execution_host': self.execution_host
            },
            'stage_outputs': self.stage_outputs
        }
    
    def save(self, filepath: str) -> None:
        """Save baseline output to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "BaselineOutput":
        """Load baseline output from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        result = AnalysisResult.from_dict(data['result']) if data.get('result') else None
        
        return cls(
            result=result,
            is_valid=data['validation']['is_valid'],
            validation_errors=data['validation']['errors'],
            spec_version=data['provenance']['spec_version'],
            config_snapshot=data['provenance']['config_snapshot'],
            executed_at=data['provenance']['executed_at'],
            execution_host=data['provenance'].get('execution_host', ''),
            stage_outputs=data.get('stage_outputs', {})
        )

