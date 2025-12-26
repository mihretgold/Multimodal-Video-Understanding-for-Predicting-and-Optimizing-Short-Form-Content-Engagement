"""
Video Pipeline Module
=====================
Main pipeline class that composes and runs all stages.

Usage:
    from app.pipeline import VideoPipeline
    
    # Run full pipeline
    pipeline = VideoPipeline()
    result = pipeline.run("path/to/video.mp4")
    
    # Run with custom config
    from app.config import get_research_config, AblationConfig
    config = get_research_config("text_only_ablation")
    config.ablation = AblationConfig.text_only()
    
    result = pipeline.run("video.mp4", config=config)
    
    # Run specific stages only
    result = pipeline.run(
        "video.mp4",
        stages=["video_ingest", "transcription"]
    )
"""

import logging
import time
from typing import Optional, List, Union

from .context import PipelineContext
from .base import PipelineStage
from .stages import (
    VideoIngestStage,
    TranscriptionStage,
    SegmentDetectionStage,
    FeatureExtractionStage,
    ScoringStage,
    OutputStage,
    ALL_STAGES,
    get_stage_by_name,
)
from ..config import AppConfig, get_config
from ..models import AnalysisResult

logger = logging.getLogger(__name__)


class VideoPipeline:
    """
    Main pipeline for video analysis.
    
    Composes multiple stages and runs them in sequence.
    Supports:
    - Running the full pipeline
    - Running specific stages only
    - Caching of expensive stages
    - Custom configurations
    - Progress callbacks
    """
    
    def __init__(
        self,
        stages: Optional[List[PipelineStage]] = None,
        use_cache: bool = True,
        save_results: bool = True
    ):
        """
        Initialize the pipeline.
        
        Args:
            stages: List of stage instances to use. If None, uses default stages.
            use_cache: Whether to use caching for expensive stages.
            save_results: Whether to save results to disk.
        """
        self.use_cache = use_cache
        self.save_results = save_results
        
        # Initialize default stages if not provided
        if stages is None:
            self.stages = [
                VideoIngestStage(),
                TranscriptionStage(),
                SegmentDetectionStage(),
                FeatureExtractionStage(),
                ScoringStage(),
                OutputStage(save_to_disk=save_results),
            ]
        else:
            self.stages = stages
    
    def run(
        self,
        video_path: str,
        config: Optional[AppConfig] = None,
        stage_names: Optional[List[str]] = None,
        stop_on_failure: bool = True,
        progress_callback: Optional[callable] = None
    ) -> AnalysisResult:
        """
        Run the pipeline on a video.
        
        Args:
            video_path: Path to the input video file.
            config: Configuration to use. If None, uses global config.
            stage_names: List of stage names to run. If None, runs all stages.
            stop_on_failure: Whether to stop if a stage fails.
            progress_callback: Optional callback(stage_name, stage_index, total_stages)
            
        Returns:
            AnalysisResult containing all pipeline outputs.
            
        Raises:
            FileNotFoundError: If video file doesn't exist.
            RuntimeError: If pipeline fails and stop_on_failure is True.
        """
        # Create pipeline context
        config = config or get_config()
        context = PipelineContext(video_path=video_path, config=config)
        
        logger.info(f"Starting pipeline for {context.video_filename}")
        logger.info(f"Result ID: {context.result_id}")
        logger.info(f"Ablation mode: {config.ablation.mode_name}")
        
        start_time = time.time()
        
        # Determine which stages to run
        stages_to_run = self._get_stages_to_run(stage_names)
        total_stages = len(stages_to_run)
        
        # Run each stage
        for i, stage in enumerate(stages_to_run):
            if progress_callback:
                progress_callback(stage.name, i + 1, total_stages)
            
            success = stage.run(context, use_cache=self.use_cache)
            
            if not success and stop_on_failure:
                error_msg = f"Pipeline failed at stage: {stage.name}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        
        total_time = time.time() - start_time
        logger.info(f"Pipeline completed in {total_time:.2f}s")
        
        # Ensure we have a result
        if context.analysis_result is None:
            context.finalize()
        
        return context.analysis_result
    
    def run_from_checkpoint(
        self,
        checkpoint_path: str,
        stage_names: Optional[List[str]] = None,
        stop_on_failure: bool = True
    ) -> AnalysisResult:
        """
        Resume pipeline from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
            stage_names: Stages to run from checkpoint. If None, runs remaining stages.
            stop_on_failure: Whether to stop if a stage fails.
            
        Returns:
            AnalysisResult containing all pipeline outputs.
        """
        context = PipelineContext.load_checkpoint(checkpoint_path)
        
        logger.info(f"Resuming pipeline from checkpoint: {checkpoint_path}")
        logger.info(f"Previously completed stages: {context.successful_stages}")
        
        # Determine which stages to run (skip completed ones)
        completed = set(context.successful_stages)
        stages_to_run = [
            stage for stage in self.stages
            if stage.name not in completed
        ]
        
        if stage_names:
            stages_to_run = [
                stage for stage in stages_to_run
                if stage.name in stage_names
            ]
        
        # Run remaining stages
        for stage in stages_to_run:
            success = stage.run(context, use_cache=self.use_cache)
            
            if not success and stop_on_failure:
                error_msg = f"Pipeline failed at stage: {stage.name}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        
        if context.analysis_result is None:
            context.finalize()
        
        return context.analysis_result
    
    def _get_stages_to_run(
        self,
        stage_names: Optional[List[str]] = None
    ) -> List[PipelineStage]:
        """Get the list of stages to run."""
        if stage_names is None:
            return self.stages
        
        # Filter to requested stages, maintaining order
        name_set = set(stage_names)
        return [stage for stage in self.stages if stage.name in name_set]
    
    def get_stage(self, name: str) -> Optional[PipelineStage]:
        """Get a stage by name."""
        for stage in self.stages:
            if stage.name == name:
                return stage
        return None
    
    @property
    def stage_names(self) -> List[str]:
        """Get list of all stage names in order."""
        return [stage.name for stage in self.stages]


def run_pipeline(
    video_path: str,
    config: Optional[AppConfig] = None,
    use_cache: bool = True,
    save_results: bool = True
) -> AnalysisResult:
    """
    Convenience function to run the full pipeline.
    
    Args:
        video_path: Path to the input video file.
        config: Configuration to use. If None, uses global config.
        use_cache: Whether to use caching.
        save_results: Whether to save results to disk.
        
    Returns:
        AnalysisResult containing all pipeline outputs.
    """
    pipeline = VideoPipeline(use_cache=use_cache, save_results=save_results)
    return pipeline.run(video_path, config=config)


def run_ablation(
    video_path: str,
    ablation_modes: Optional[List[str]] = None
) -> dict:
    """
    Run pipeline with multiple ablation configurations.
    
    Args:
        video_path: Path to the input video file.
        ablation_modes: List of ablation modes to run. 
                       Options: "text_only", "audio_only", "visual_only", 
                                "text_audio", "full_multimodal"
                       If None, runs all modes.
                       
    Returns:
        Dictionary mapping ablation mode names to AnalysisResults.
    """
    from ..config import AblationConfig, get_research_config
    
    if ablation_modes is None:
        ablation_modes = [
            "text_only",
            "audio_only", 
            "visual_only",
            "text_audio",
            "full_multimodal"
        ]
    
    # Map mode names to AblationConfig factory methods
    mode_factories = {
        "text_only": AblationConfig.text_only,
        "audio_only": AblationConfig.audio_only,
        "visual_only": AblationConfig.visual_only,
        "text_audio": AblationConfig.text_audio,
        "full_multimodal": AblationConfig.full_multimodal,
    }
    
    results = {}
    
    for mode_name in ablation_modes:
        if mode_name not in mode_factories:
            logger.warning(f"Unknown ablation mode: {mode_name}")
            continue
        
        logger.info(f"Running ablation: {mode_name}")
        
        # Create config for this ablation
        config = get_research_config(f"ablation_{mode_name}")
        config.ablation = mode_factories[mode_name]()
        
        # Run pipeline
        pipeline = VideoPipeline(use_cache=True, save_results=True)
        
        try:
            result = pipeline.run(video_path, config=config)
            results[mode_name] = result
            logger.info(f"Ablation {mode_name} complete: {result.segment_count} segments")
        except Exception as e:
            logger.error(f"Ablation {mode_name} failed: {e}")
            results[mode_name] = None
    
    return results

