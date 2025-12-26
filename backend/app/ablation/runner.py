"""
Ablation Runner Module
======================
Runs systematic ablation experiments across modality configurations.

Executes the full pipeline with different ablation modes and
collects results for analysis.
"""

import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from ..config import AblationConfig, get_config, AppConfig
from ..models import AnalysisResult, Segment
from ..logging_config import get_research_logger, log_pipeline_decision

logger = get_research_logger("ablation")


# Define all ablation modes
ABLATION_MODES = {
    'text_only': AblationConfig.text_only,
    'audio_only': AblationConfig.audio_only,
    'visual_only': AblationConfig.visual_only,
    'text_audio': AblationConfig.text_audio,
    'full_multimodal': AblationConfig.full_multimodal,
}


@dataclass
class AblationResult:
    """
    Result from a single ablation run.
    
    Wraps AnalysisResult with ablation-specific metadata.
    """
    mode: str
    analysis_result: Optional[AnalysisResult] = None
    execution_time_seconds: float = 0.0
    success: bool = True
    error_message: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            'mode': self.mode,
            'analysis_result': self.analysis_result.to_dict() if self.analysis_result else None,
            'execution_time_seconds': self.execution_time_seconds,
            'success': self.success,
            'error_message': self.error_message,
            'timestamp': self.timestamp
        }
    
    @property
    def segment_count(self) -> int:
        if self.analysis_result:
            return len(self.analysis_result.segments)
        return 0
    
    @property
    def top_segment(self) -> Optional[Segment]:
        if self.analysis_result and self.analysis_result.segments:
            sorted_segments = sorted(
                self.analysis_result.segments,
                key=lambda s: s.rank
            )
            return sorted_segments[0]
        return None


class AblationRunner:
    """
    Runs ablation experiments across different modality configurations.
    
    For each mode, runs the full pipeline and collects results
    for later analysis.
    
    Usage:
        runner = AblationRunner()
        
        # Run all modes
        results = runner.run_all_modes("video.mp4", subtitle_data)
        
        # Run specific modes
        results = runner.run_modes(
            "video.mp4", 
            subtitle_data,
            modes=['text_only', 'full_multimodal']
        )
    """
    
    def __init__(self, base_config: Optional[AppConfig] = None):
        """
        Initialize the ablation runner.
        
        Args:
            base_config: Base configuration (ablation will be overridden)
        """
        self.base_config = base_config or get_config()
        
        logger.info("Initialized AblationRunner")
    
    def run_all_modes(
        self,
        video_path: str,
        save_results: bool = True
    ) -> Dict[str, AblationResult]:
        """
        Run ablation study with all modality configurations.
        
        Args:
            video_path: Path to video file
            save_results: Whether to save individual results
            
        Returns:
            Dict mapping mode name to AblationResult
        """
        return self.run_modes(
            video_path,
            modes=list(ABLATION_MODES.keys()),
            save_results=save_results
        )
    
    def run_modes(
        self,
        video_path: str,
        modes: List[str],
        save_results: bool = True
    ) -> Dict[str, AblationResult]:
        """
        Run ablation study with specific modes.
        
        Args:
            video_path: Path to video file
            modes: List of mode names to run
            save_results: Whether to save individual results
            
        Returns:
            Dict mapping mode name to AblationResult
        """
        results = {}
        
        logger.info(
            f"Starting ablation study with {len(modes)} modes",
            extra={'video_path': video_path, 'modes': modes}
        )
        
        total_start = time.time()
        
        for mode in modes:
            if mode not in ABLATION_MODES:
                logger.warning(f"Unknown ablation mode: {mode}")
                results[mode] = AblationResult(
                    mode=mode,
                    success=False,
                    error_message=f"Unknown mode: {mode}"
                )
                continue
            
            logger.info(f"Running mode: {mode}")
            result = self._run_single_mode(video_path, mode, save_results)
            results[mode] = result
            
            if result.success:
                logger.info(
                    f"Mode {mode} complete: {result.segment_count} segments, "
                    f"time: {result.execution_time_seconds:.2f}s"
                )
            else:
                logger.error(f"Mode {mode} failed: {result.error_message}")
        
        total_time = time.time() - total_start
        
        # Log completion
        log_pipeline_decision(
            "ablation_study_complete",
            {
                'modes_run': len(results),
                'successful': sum(1 for r in results.values() if r.success),
                'failed': sum(1 for r in results.values() if not r.success),
                'total_time_seconds': total_time,
                'mode_summary': {
                    mode: {
                        'success': r.success,
                        'segments': r.segment_count,
                        'time': r.execution_time_seconds
                    }
                    for mode, r in results.items()
                }
            }
        )
        
        logger.info(
            f"Ablation study complete",
            extra={
                'total_modes': len(results),
                'total_time': total_time
            }
        )
        
        return results
    
    def _run_single_mode(
        self,
        video_path: str,
        mode: str,
        save_results: bool
    ) -> AblationResult:
        """Run pipeline with a single ablation mode."""
        from ..pipeline import VideoPipeline
        
        start_time = time.time()
        
        try:
            # Create config with this ablation mode
            config = self._create_mode_config(mode)
            
            # Create and run pipeline
            pipeline = VideoPipeline(
                config=config,
                use_cache=True,
                save_results=save_results
            )
            
            analysis_result = pipeline.run(video_path)
            
            execution_time = time.time() - start_time
            
            return AblationResult(
                mode=mode,
                analysis_result=analysis_result,
                execution_time_seconds=execution_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error in mode {mode}: {str(e)}")
            return AblationResult(
                mode=mode,
                success=False,
                error_message=str(e),
                execution_time_seconds=time.time() - start_time
            )
    
    def _create_mode_config(self, mode: str) -> AppConfig:
        """Create a config with the specified ablation mode."""
        import copy
        
        # Deep copy base config
        config = AppConfig(
            paths=self.base_config.paths,
            whisper=self.base_config.whisper,
            gemini=self.base_config.gemini,
            segmentation=self.base_config.segmentation,
            features=self.base_config.features,
            scoring=self.base_config.scoring,
            video=self.base_config.video,
            flask=self.base_config.flask,
            research=self.base_config.research
        )
        
        # Set ablation mode
        config.ablation = ABLATION_MODES[mode]()
        
        return config
    
    def run_quick_comparison(
        self,
        video_path: str
    ) -> Dict[str, AblationResult]:
        """
        Run quick comparison with just text_only and full_multimodal.
        
        Useful for initial testing.
        """
        return self.run_modes(
            video_path,
            modes=['text_only', 'full_multimodal'],
            save_results=False
        )
    
    def run_unimodal_comparison(
        self,
        video_path: str
    ) -> Dict[str, AblationResult]:
        """Run comparison of all single-modality modes."""
        return self.run_modes(
            video_path,
            modes=['text_only', 'audio_only', 'visual_only'],
            save_results=False
        )


def run_ablation_study(
    video_path: str,
    modes: Optional[List[str]] = None,
    save_results: bool = True
) -> Dict[str, AblationResult]:
    """
    Convenience function to run an ablation study.
    
    Args:
        video_path: Path to video file
        modes: Optional list of modes (default: all)
        save_results: Whether to save results
        
    Returns:
        Dict of ablation results
    """
    runner = AblationRunner()
    
    if modes:
        return runner.run_modes(video_path, modes, save_results)
    return runner.run_all_modes(video_path, save_results)

