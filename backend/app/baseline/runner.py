"""
Baseline Runner Module
======================
Executes the video analysis pipeline as a formal baseline with full logging.

The BaselineRunner:
- Validates inputs against specification
- Runs the pipeline with comprehensive logging
- Validates all intermediate outputs
- Produces reproducible, documented results
- Saves outputs in standard format
"""

import os
import socket
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from .specification import BaselineSpec, BaselineOutput, InputSpec, IntermediateSpec, OutputSpec
from ..pipeline import VideoPipeline, PipelineContext
from ..pipeline.stages import (
    VideoIngestStage,
    TranscriptionStage,
    SegmentDetectionStage,
    FeatureExtractionStage,
    ScoringStage,
    OutputStage,
)
from ..config import AppConfig, get_config
from ..models import AnalysisResult, VideoMetadata, SubtitleData, Segment
from ..logging_config import (
    get_research_logger,
    log_pipeline_decision,
    log_stage_start,
    log_stage_complete,
    log_stage_error,
    log_segment_features,
    log_segment_score,
)

logger = get_research_logger("baseline", log_to_file=True)


class BaselineRunner:
    """
    Runs the video analysis pipeline as a formal research baseline.
    
    Features:
    - Input/output validation against specification
    - Comprehensive structured logging
    - Stage-level output capture
    - Reproducible execution with provenance tracking
    
    Usage:
        runner = BaselineRunner()
        output = runner.run("video.mp4")
        
        if output.is_valid:
            print(f"Success: {output.result.segment_count} segments")
        else:
            print(f"Errors: {output.validation_errors}")
    """
    
    def __init__(
        self,
        config: Optional[AppConfig] = None,
        spec: Optional[BaselineSpec] = None,
        use_cache: bool = True,
        save_results: bool = True
    ):
        """
        Initialize the baseline runner.
        
        Args:
            config: Configuration to use (default: global config)
            spec: Baseline specification (default: standard spec)
            use_cache: Whether to cache expensive operations
            save_results: Whether to save results to disk
        """
        self.config = config or get_config()
        self.spec = spec or BaselineSpec()
        self.use_cache = use_cache
        self.save_results = save_results
        
        # Initialize pipeline with custom stages that include logging
        self.pipeline = VideoPipeline(
            stages=self._create_logged_stages(),
            use_cache=use_cache,
            save_results=save_results
        )
    
    def _create_logged_stages(self) -> List:
        """Create pipeline stages with enhanced logging."""
        return [
            LoggedVideoIngestStage(self.spec.intermediate_spec),
            LoggedTranscriptionStage(self.spec.intermediate_spec),
            LoggedSegmentDetectionStage(self.spec.intermediate_spec),
            LoggedFeatureExtractionStage(self.spec.intermediate_spec),
            LoggedScoringStage(self.spec.intermediate_spec),
            OutputStage(save_to_disk=self.save_results),
        ]
    
    def run(
        self,
        video_path: str,
        experiment_name: Optional[str] = None
    ) -> BaselineOutput:
        """
        Run the baseline pipeline on a video.
        
        Args:
            video_path: Path to the video file
            experiment_name: Optional name for this experiment run
            
        Returns:
            BaselineOutput containing results and validation status
        """
        start_time = time.time()
        validation_errors = []
        stage_outputs = {}
        
        # Set experiment name if provided
        if experiment_name:
            self.config.research.experiment_name = experiment_name
        
        logger.info(
            f"Starting baseline run",
            extra={
                'video_path': video_path,
                'experiment': self.config.research.experiment_name,
                'ablation_mode': self.config.ablation.mode_name,
                'baseline_version': self.spec.baseline_version
            }
        )
        
        # Validate input
        is_valid, error = self.spec.input_spec.validate(video_path)
        if not is_valid:
            logger.error(f"Input validation failed: {error}")
            validation_errors.append(f"Input: {error}")
            return BaselineOutput(
                result=None,
                is_valid=False,
                validation_errors=validation_errors,
                spec_version=self.spec.baseline_version,
                config_snapshot=self.config.to_dict()
            )
        
        # Run pipeline
        try:
            result = self.pipeline.run(
                video_path,
                config=self.config,
                stop_on_failure=False  # Continue to capture all errors
            )
            
            # Validate output
            output_valid, output_error = self.spec.output_spec.validate(result)
            if not output_valid:
                validation_errors.append(f"Output: {output_error}")
            
            # Capture stage outputs for detailed logging
            # Get from pipeline context if available
            stage_outputs = self._capture_stage_outputs(result)
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            validation_errors.append(f"Execution: {str(e)}")
            result = None
        
        execution_time = time.time() - start_time
        
        # Create baseline output
        output = BaselineOutput(
            result=result,
            is_valid=len(validation_errors) == 0,
            validation_errors=validation_errors,
            spec_version=self.spec.baseline_version,
            config_snapshot=self.config.to_dict(),
            executed_at=datetime.now().isoformat(),
            execution_host=socket.gethostname(),
            stage_outputs=stage_outputs
        )
        
        # Log completion
        log_pipeline_decision(
            "baseline_complete",
            {
                'is_valid': output.is_valid,
                'validation_errors': validation_errors,
                'segment_count': result.segment_count if result else 0,
                'execution_time_seconds': execution_time
            },
            result_id=result.result_id if result else None
        )
        
        logger.info(
            f"Baseline run complete",
            extra={
                'is_valid': output.is_valid,
                'segment_count': result.segment_count if result else 0,
                'execution_time': execution_time,
                'error_count': len(validation_errors)
            }
        )
        
        # Save output if requested
        if self.save_results and result:
            output_path = self._get_output_path(result.result_id)
            output.save(output_path)
            logger.info(f"Saved baseline output to {output_path}")
        
        return output
    
    def _capture_stage_outputs(self, result: AnalysisResult) -> Dict[str, Dict]:
        """Capture summary outputs from each stage."""
        outputs = {}
        
        if result and result.video_metadata:
            outputs['video_ingest'] = {
                'filename': result.video_metadata.filename,
                'duration_seconds': result.video_metadata.duration_seconds,
                'resolution': result.video_metadata.resolution,
                'fps': result.video_metadata.fps
            }
        
        if result and result.subtitle_data:
            outputs['transcription'] = {
                'entry_count': len(result.subtitle_data.entries),
                'word_count': result.subtitle_data.word_count,
                'source': result.subtitle_data.source,
                'language': result.subtitle_data.language
            }
        
        if result and result.segments:
            outputs['segment_detection'] = {
                'segment_count': len(result.segments),
                'segment_types': list(set(s.segment_type for s in result.segments)),
                'total_duration': sum(s.duration_seconds for s in result.segments)
            }
            
            outputs['scoring'] = {
                'top_score': max(s.score.total_score for s in result.segments if s.score) if result.segments else 0,
                'avg_score': sum(s.score.total_score for s in result.segments if s.score) / len(result.segments) if result.segments else 0,
                'score_range': (
                    min(s.score.total_score for s in result.segments if s.score),
                    max(s.score.total_score for s in result.segments if s.score)
                ) if result.segments and any(s.score for s in result.segments) else (0, 0)
            }
        
        return outputs
    
    def _get_output_path(self, result_id: str) -> str:
        """Get the output file path for a result."""
        output_dir = self.config.paths.experiments / "baseline_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir / f"baseline_{result_id}.json")
    
    def verify_result(self, result: AnalysisResult) -> tuple:
        """
        Verify that a result meets baseline specification.
        
        Returns:
            (is_valid, list of errors)
        """
        errors = []
        
        # Validate output
        valid, error = self.spec.output_spec.validate(result)
        if not valid:
            errors.append(error)
        
        # Validate video metadata
        if result.video_metadata:
            valid, error = self.spec.intermediate_spec.validate_video_metadata(result.video_metadata)
            if not valid:
                errors.append(error)
        
        # Validate subtitle data
        if result.subtitle_data:
            valid, error = self.spec.intermediate_spec.validate_subtitle_data(result.subtitle_data)
            if not valid:
                errors.append(error)
        
        # Validate segments
        valid, error = self.spec.intermediate_spec.validate_segments(result.segments)
        if not valid:
            errors.append(error)
        
        return len(errors) == 0, errors


# =============================================================================
# LOGGED PIPELINE STAGES
# =============================================================================

class LoggedVideoIngestStage(VideoIngestStage):
    """Video ingest stage with baseline logging."""
    
    def __init__(self, spec: IntermediateSpec):
        super().__init__()
        self.spec = spec
    
    def _execute(self, context: PipelineContext) -> None:
        log_stage_start(
            self.name,
            context.result_id,
            {'video_path': context.video_path}
        )
        
        super()._execute(context)
        
        # Validate output
        valid, error = self.spec.validate_video_metadata(context.video_metadata)
        if not valid:
            logger.warning(f"Video metadata validation: {error}")
        
        # Log details
        if context.video_metadata:
            logger.info(
                "Video metadata extracted",
                extra={
                    'result_id': context.result_id,
                    'filename': context.video_metadata.filename,
                    'duration': context.video_metadata.duration_seconds,
                    'resolution': context.video_metadata.resolution,
                    'fps': context.video_metadata.fps,
                    'has_subtitles': context.video_metadata.has_subtitles
                }
            )


class LoggedTranscriptionStage(TranscriptionStage):
    """Transcription stage with baseline logging."""
    
    def __init__(self, spec: IntermediateSpec):
        super().__init__()
        self.spec = spec
    
    def _execute(self, context: PipelineContext) -> None:
        log_stage_start(
            self.name,
            context.result_id,
            {'video_filename': context.video_filename}
        )
        
        super()._execute(context)
        
        # Validate output
        valid, error = self.spec.validate_subtitle_data(context.subtitle_data)
        if not valid:
            logger.warning(f"Subtitle data validation: {error}")
        
        # Log details
        if context.subtitle_data:
            logger.info(
                "Subtitles extracted",
                extra={
                    'result_id': context.result_id,
                    'entry_count': len(context.subtitle_data.entries),
                    'word_count': context.subtitle_data.word_count,
                    'source': context.subtitle_data.source,
                    'language': context.subtitle_data.language,
                    'duration_covered': context.subtitle_data.total_duration_seconds
                }
            )


class LoggedSegmentDetectionStage(SegmentDetectionStage):
    """Segment detection stage with baseline logging."""
    
    def __init__(self, spec: IntermediateSpec):
        super().__init__()
        self.spec = spec
    
    def _execute(self, context: PipelineContext) -> None:
        log_stage_start(
            self.name,
            context.result_id,
            {'subtitle_count': len(context.subtitle_data.entries) if context.subtitle_data else 0}
        )
        
        super()._execute(context)
        
        # Validate output
        valid, error = self.spec.validate_segments(context.candidate_segments)
        if not valid:
            logger.warning(f"Segment validation: {error}")
        
        # Log each segment
        for segment in context.candidate_segments:
            logger.info(
                f"Segment detected: {segment.segment_id}",
                extra={
                    'result_id': context.result_id,
                    'segment_id': segment.segment_id,
                    'start_seconds': segment.start_seconds,
                    'end_seconds': segment.end_seconds,
                    'duration_seconds': segment.duration_seconds,
                    'segment_type': segment.segment_type,
                    'source': segment.source
                }
            )


class LoggedFeatureExtractionStage(FeatureExtractionStage):
    """Feature extraction stage with baseline logging."""
    
    def __init__(self, spec: IntermediateSpec):
        super().__init__()
        self.spec = spec
    
    def _execute(self, context: PipelineContext) -> None:
        ablation = context.config.ablation
        log_stage_start(
            self.name,
            context.result_id,
            {
                'segment_count': len(context.candidate_segments),
                'ablation_mode': ablation.mode_name,
                'use_text': ablation.use_text,
                'use_audio': ablation.use_audio,
                'use_visual': ablation.use_visual
            }
        )
        
        super()._execute(context)
        
        # Log features for each segment
        for segment_id, features in context.segment_features.items():
            feature_summary = {}
            
            if features.text_features:
                feature_summary['text'] = {
                    'word_count': features.text_features.word_count,
                    'sentence_count': features.text_features.sentence_count,
                    'question_count': features.text_features.question_count,
                    'exclamation_count': features.text_features.exclamation_count
                }
            
            if features.audio_features:
                feature_summary['audio'] = {
                    'energy_mean': features.audio_features.energy_mean,
                    'silence_ratio': features.audio_features.silence_ratio,
                    'speech_rate': features.audio_features.speech_rate
                }
            
            if features.visual_features:
                feature_summary['visual'] = {
                    'motion_intensity': features.visual_features.motion_intensity,
                    'scene_change_count': features.visual_features.scene_change_count
                }
            
            log_segment_features(
                segment_id=segment_id,
                features=feature_summary,
                modalities=features.modalities_present
            )


class LoggedScoringStage(ScoringStage):
    """Scoring stage with baseline logging."""
    
    def __init__(self, spec: IntermediateSpec):
        super().__init__()
        self.spec = spec
    
    def _execute(self, context: PipelineContext) -> None:
        log_stage_start(
            self.name,
            context.result_id,
            {
                'segment_count': len(context.candidate_segments),
                'scoring_mode': context.config.scoring.mode
            }
        )
        
        super()._execute(context)
        
        # Log score for each segment
        for segment in context.scored_segments:
            if segment.score:
                log_segment_score(
                    segment_id=segment.segment_id,
                    total_score=segment.score.total_score,
                    text_score=segment.score.text_score,
                    audio_score=segment.score.audio_score,
                    visual_score=segment.score.visual_score,
                    weights={
                        'text': segment.score.text_weight,
                        'audio': segment.score.audio_weight,
                        'visual': segment.score.visual_weight
                    },
                    method=segment.score.scoring_method
                )
        
        # Log ranking summary
        if context.scored_segments:
            top_segment = context.scored_segments[0]
            logger.info(
                "Scoring complete",
                extra={
                    'result_id': context.result_id,
                    'segment_count': len(context.scored_segments),
                    'top_segment_id': top_segment.segment_id,
                    'top_score': top_segment.score.total_score if top_segment.score else 0,
                    'top_type': top_segment.segment_type
                }
            )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_baseline(
    video_path: str,
    experiment_name: Optional[str] = None,
    save_results: bool = True
) -> BaselineOutput:
    """
    Convenience function to run the baseline pipeline.
    
    Args:
        video_path: Path to the video file
        experiment_name: Optional experiment name
        save_results: Whether to save results to disk
        
    Returns:
        BaselineOutput with results and validation status
    """
    runner = BaselineRunner(save_results=save_results)
    return runner.run(video_path, experiment_name=experiment_name)


def export_baseline_spec(output_path: str) -> None:
    """Export the baseline specification to a JSON file."""
    spec = BaselineSpec()
    spec.save(output_path)
    logger.info(f"Exported baseline specification to {output_path}")

