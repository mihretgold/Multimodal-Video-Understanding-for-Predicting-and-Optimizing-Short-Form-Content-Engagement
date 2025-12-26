"""
Pipeline Stages Module
======================
Implements individual pipeline stages for video analysis.

Stages:
1. VideoIngestStage - Load video and extract metadata
2. TranscriptionStage - Extract/generate subtitles
3. SegmentDetectionStage - Identify candidate segments (via Gemini)
4. FeatureExtractionStage - Extract multimodal features (placeholder for Step 3)
5. ScoringStage - Score and rank segments (placeholder for Step 4)
6. OutputStage - Finalize and save results
"""

import os
import logging
from typing import Optional

from .base import PipelineStage, ConditionalStage
from .context import PipelineContext
from ..config import get_config
from ..models import (
    VideoMetadata,
    SubtitleData,
    SubtitleEntry,
    Segment,
    SegmentFeatures,
    TextFeatures,
    AudioFeatures,
    VisualFeatures,
    generate_segment_id,
)
from ..utils.video_utils import get_video_info, check_subtitles
from ..services.subtitle_service import SubtitleService
from ..services.analysis_service import AnalysisService

logger = logging.getLogger(__name__)


# =============================================================================
# STAGE 1: VIDEO INGESTION
# =============================================================================

class VideoIngestStage(PipelineStage):
    """
    Load video file and extract metadata.
    
    Reads:
        - context.video_path
        
    Writes:
        - context.video_metadata
    """
    
    @property
    def name(self) -> str:
        return "video_ingest"
    
    @property
    def description(self) -> str:
        return "Load video file and extract metadata (duration, resolution, fps)"
    
    def _execute(self, context: PipelineContext) -> None:
        video_path = context.video_path
        
        # Get video info using utility function
        video_info = get_video_info(video_path)
        
        # Check for embedded subtitles
        has_subs = check_subtitles(video_path)
        
        # Get file size
        file_size = os.path.getsize(video_path)
        
        # Create metadata object
        context.video_metadata = VideoMetadata(
            filename=context.video_filename,
            filepath=video_path,
            duration_seconds=video_info['duration'],
            fps=video_info['fps'],
            width=video_info['size'][0],
            height=video_info['size'][1],
            file_size_bytes=file_size,
            has_subtitles=has_subs
        )
    
    def _get_output_summary(self, context: PipelineContext) -> str:
        if context.video_metadata:
            m = context.video_metadata
            return f"{m.duration_formatted} @ {m.resolution}, {m.fps:.1f}fps"
        return "metadata extracted"


# =============================================================================
# STAGE 2: TRANSCRIPTION
# =============================================================================

class TranscriptionStage(PipelineStage):
    """
    Extract embedded subtitles or transcribe using Whisper.
    
    Reads:
        - context.video_path
        - context.video_metadata
        
    Writes:
        - context.subtitle_data
    """
    
    @property
    def name(self) -> str:
        return "transcription"
    
    @property
    def description(self) -> str:
        return "Extract embedded subtitles or transcribe audio using Whisper"
    
    @property
    def cacheable(self) -> bool:
        return True  # Transcription is expensive, cache it
    
    def __init__(self):
        self._subtitle_service = None
    
    @property
    def subtitle_service(self) -> SubtitleService:
        """Lazy initialization of subtitle service."""
        if self._subtitle_service is None:
            self._subtitle_service = SubtitleService()
        return self._subtitle_service
    
    def _execute(self, context: PipelineContext) -> None:
        video_path = context.video_path
        subtitles_folder = str(context.config.paths.subtitles)
        
        # Use subtitle service to get subtitles JSON
        result = self.subtitle_service.get_subtitles_json(video_path, subtitles_folder)
        
        # Convert to SubtitleData model
        entries = [
            SubtitleEntry(
                index=i,
                start_seconds=s['start'],
                end_seconds=s['end'],
                text=s['text']
            )
            for i, s in enumerate(result['subtitles'])
        ]
        
        context.subtitle_data = SubtitleData(
            video_filename=context.video_filename,
            entries=entries,
            source=result['source'],
            language=result.get('language', 'unknown')
        )
    
    def _get_output_summary(self, context: PipelineContext) -> str:
        if context.subtitle_data:
            s = context.subtitle_data
            return f"{len(s.entries)} entries, {s.word_count} words, source: {s.source}"
        return "subtitles extracted"
    
    def _get_cache_data(self, context: PipelineContext) -> dict:
        if context.subtitle_data:
            return context.subtitle_data.to_dict()
        return {}
    
    def _restore_from_cache(self, context: PipelineContext, cached_data: dict) -> None:
        context.subtitle_data = SubtitleData.from_dict(cached_data)


# =============================================================================
# STAGE 3: SEGMENT DETECTION
# =============================================================================

class SegmentDetectionStage(PipelineStage):
    """
    Identify candidate segments using temporal segmentation algorithms.
    
    Supports multiple strategies:
    - pause_based: Uses speech pauses from transcription
    - fixed_window: Regular intervals with boundary snapping
    - semantic_boundary: Groups semantically coherent content
    - hybrid: Combines multiple strategies
    - llm: Uses Gemini for segment identification (original approach)
    
    Reads:
        - context.subtitle_data
        - context.video_metadata
        
    Writes:
        - context.candidate_segments
    """
    
    @property
    def name(self) -> str:
        return "segment_detection"
    
    @property
    def description(self) -> str:
        strategy = get_config().segmentation.strategy
        return f"Identify candidate segments using '{strategy}' strategy"
    
    def __init__(self):
        self._analysis_service = None
        self._segmenter = None
    
    @property
    def analysis_service(self) -> AnalysisService:
        """Lazy initialization of analysis service for LLM mode."""
        if self._analysis_service is None:
            self._analysis_service = AnalysisService()
        return self._analysis_service
    
    def _get_segmenter(self, config):
        """Get or create temporal segmenter."""
        from ..segmentation import TemporalSegmenter
        strategy = config.segmentation.strategy
        
        if strategy == 'llm':
            return None  # Use LLM path
        
        return TemporalSegmenter(
            strategy=strategy,
            config=config.segmentation
        )
    
    def _execute(self, context: PipelineContext) -> None:
        if context.subtitle_data is None:
            raise ValueError("No subtitle data available for segment detection")
        
        config = context.config
        strategy = config.segmentation.strategy
        
        logger.info(f"Running segment detection with strategy: {strategy}")
        
        if strategy == 'llm':
            # Original LLM-based approach
            segments = self._run_llm_segmentation(context)
        else:
            # Temporal segmentation approach
            segments = self._run_temporal_segmentation(context)
        
        # Optionally classify segments using LLM
        if config.segmentation.use_llm_classification and strategy != 'llm':
            segments = self._classify_segments(segments, context)
        
        # Add text preview for all segments
        for segment in segments:
            if not segment.text_preview:
                segment.text_preview = context.subtitle_data.get_text_in_range(
                    segment.start_seconds,
                    segment.end_seconds
                )[:200]
        
        context.candidate_segments = segments
    
    def _run_temporal_segmentation(self, context: PipelineContext) -> list:
        """Run temporal segmentation algorithms."""
        from ..segmentation import TemporalSegmenter
        
        segmenter = TemporalSegmenter(
            strategy=context.config.segmentation.strategy,
            config=context.config.segmentation
        )
        
        video_duration = context.video_metadata.duration_seconds
        
        segments = segmenter.segment(
            context.subtitle_data,
            video_duration,
            result_id=context.result_id
        )
        
        return segments
    
    def _run_llm_segmentation(self, context: PipelineContext) -> list:
        """Original LLM-based segmentation using Gemini."""
        # Convert subtitle entries to API format
        subtitles = [
            {
                'start': entry.start_seconds,
                'end': entry.end_seconds,
                'text': entry.text
            }
            for entry in context.subtitle_data.entries
        ]
        
        # Run Gemini analysis with model output
        segments = self.analysis_service.analyze_subtitles(
            subtitles,
            video_filename=context.video_filename,
            return_models=True
        )
        
        return segments
    
    def _classify_segments(self, segments: list, context: PipelineContext) -> list:
        """Classify segment types using LLM."""
        try:
            for segment in segments:
                if segment.segment_type == 'auto':
                    # Get text for classification
                    text = context.subtitle_data.get_text_in_range(
                        segment.start_seconds,
                        segment.end_seconds
                    )
                    
                    # Simple heuristic classification (placeholder for LLM)
                    segment.segment_type = self._heuristic_classify(text)
        except Exception as e:
            logger.warning(f"Segment classification failed: {e}")
        
        return segments
    
    def _heuristic_classify(self, text: str) -> str:
        """Simple heuristic classification of segment content."""
        text_lower = text.lower()
        
        # Simple keyword-based classification
        if any(w in text_lower for w in ['haha', 'lol', 'funny', 'laugh', 'joke']):
            return 'funny'
        elif any(w in text_lower for w in ['sad', 'cry', 'emotional', 'feel', 'love']):
            return 'emotional'
        elif any(w in text_lower for w in ['fight', 'action', 'run', 'chase', 'battle']):
            return 'action'
        elif any(w in text_lower for w in ['shock', 'surprise', 'twist', 'reveal']):
            return 'dramatic'
        elif '?' in text and text.count('?') >= 2:
            return 'informative'
        else:
            return 'highlight'
    
    def _get_output_summary(self, context: PipelineContext) -> str:
        count = len(context.candidate_segments)
        types = set(s.segment_type for s in context.candidate_segments)
        strategy = context.config.segmentation.strategy
        return f"{count} segments via '{strategy}', types: {', '.join(types)}"


# =============================================================================
# STAGE 4: FEATURE EXTRACTION
# =============================================================================

class FeatureExtractionStage(ConditionalStage):
    """
    Extract multimodal features for each segment.
    
    Uses the FeatureExtractor to extract:
    - Text features: sentiment, word count, speech rate
    - Audio features: energy, silence ratio, dynamics
    - Visual features: motion, scene changes, brightness
    
    Reads:
        - context.candidate_segments
        - context.subtitle_data
        - context.video_path
        
    Writes:
        - context.segment_features
        - Updates context.candidate_segments with features
    """
    
    @property
    def name(self) -> str:
        return "feature_extraction"
    
    @property
    def description(self) -> str:
        config = get_config().ablation
        modalities = []
        if config.use_text:
            modalities.append("text")
        if config.use_audio:
            modalities.append("audio")
        if config.use_visual:
            modalities.append("visual")
        return f"Extract features: {', '.join(modalities) or 'none'}"
    
    def __init__(self, parallel: bool = False):
        self._extractor = None
        self.parallel = parallel
    
    def should_run(self, context: PipelineContext) -> bool:
        """Run if any modality is enabled in ablation config."""
        config = context.config.ablation
        return config.use_text or config.use_audio or config.use_visual
    
    def _get_extractor(self, context: PipelineContext):
        """Get or create feature extractor."""
        from ..features import FeatureExtractor
        
        if self._extractor is None:
            self._extractor = FeatureExtractor(
                ablation=context.config.ablation,
                parallel=self.parallel
            )
        return self._extractor
    
    def _execute(self, context: PipelineContext) -> None:
        extractor = self._get_extractor(context)
        
        logger.info(
            f"Extracting features for {len(context.candidate_segments)} segments "
            f"(mode: {context.config.ablation.mode_name})"
        )
        
        # Extract features for each segment
        features_dict = extractor.extract_batch(
            video_path=context.video_path,
            segments=context.candidate_segments,
            subtitle_data=context.subtitle_data,
            result_id=context.result_id
        )
        
        # Update context and segments
        context.segment_features = features_dict
        
        for segment in context.candidate_segments:
            if segment.segment_id in features_dict:
                segment.features = features_dict[segment.segment_id]
    
    def _get_output_summary(self, context: PipelineContext) -> str:
        count = len(context.segment_features)
        modalities = []
        if context.config.ablation.use_text:
            modalities.append("text")
        if context.config.ablation.use_audio:
            modalities.append("audio")
        if context.config.ablation.use_visual:
            modalities.append("visual")
        return f"{count} segments, modalities: {', '.join(modalities)}"


# =============================================================================
# STAGE 5: SCORING
# =============================================================================

class ScoringStage(PipelineStage):
    """
    Score and rank segments based on multimodal features.
    
    Uses the EngagementScorer with configurable:
    - Scoring strategy (rule_based, normalized, learned)
    - Modality weights
    - Ablation settings
    
    Reads:
        - context.candidate_segments
        - context.segment_features
        
    Writes:
        - context.scored_segments
    """
    
    @property
    def name(self) -> str:
        return "scoring"
    
    @property
    def description(self) -> str:
        config = get_config().scoring
        return f"Score and rank segments using '{config.mode}' strategy"
    
    def __init__(self, strategy: Optional[str] = None):
        self._scorer = None
        self._strategy_override = strategy
    
    def _get_scorer(self, context: PipelineContext):
        """Get or create engagement scorer."""
        from ..scoring import EngagementScorer
        
        if self._scorer is None:
            strategy = self._strategy_override or context.config.scoring.mode
            self._scorer = EngagementScorer(
                strategy=strategy,
                config=context.config.scoring,
                ablation=context.config.ablation
            )
        return self._scorer
    
    def _execute(self, context: PipelineContext) -> None:
        scorer = self._get_scorer(context)
        
        logger.info(
            f"Scoring {len(context.candidate_segments)} segments "
            f"(strategy: {scorer.strategy.name}, weights: {scorer.weights.to_dict()})"
        )
        
        # Score and rank segments
        scored_segments = scorer.score_and_rank(
            context.candidate_segments,
            context.segment_features,
            result_id=context.result_id
        )
        
        context.scored_segments = scored_segments
        
        # Log summary
        if scored_segments:
            top = scored_segments[0]
            logger.info(
                f"Top segment: {top.segment_id} "
                f"(score: {top.score.total_score:.3f}, type: {top.segment_type})"
            )
    
    def _get_output_summary(self, context: PipelineContext) -> str:
        if context.scored_segments:
            top = context.scored_segments[0]
            n = len(context.scored_segments)
            return f"{n} segments, top: {top.time_range_formatted} (score: {top.score.total_score:.3f})"
        return "segments scored"


# =============================================================================
# STAGE 6: OUTPUT
# =============================================================================

class OutputStage(PipelineStage):
    """
    Finalize and save analysis results.
    
    Reads:
        - All context fields
        
    Writes:
        - context.analysis_result
        - Saves result to disk
    """
    
    @property
    def name(self) -> str:
        return "output"
    
    @property
    def description(self) -> str:
        return "Finalize analysis and save results to disk"
    
    def __init__(self, save_to_disk: bool = True):
        self.save_to_disk = save_to_disk
    
    def _execute(self, context: PipelineContext) -> None:
        # Finalize creates the AnalysisResult
        context.finalize()
        
        # Save to disk if enabled
        if self.save_to_disk:
            context.save_result()
    
    def _get_output_summary(self, context: PipelineContext) -> str:
        if context.analysis_result:
            r = context.analysis_result
            return f"result_id: {r.result_id}, {r.segment_count} segments"
        return "results finalized"


# =============================================================================
# STAGE REGISTRY
# =============================================================================

# All available stages in recommended execution order
ALL_STAGES = [
    VideoIngestStage,
    TranscriptionStage,
    SegmentDetectionStage,
    FeatureExtractionStage,
    ScoringStage,
    OutputStage,
]

def get_stage_by_name(name: str) -> type:
    """Get a stage class by its name."""
    for stage_class in ALL_STAGES:
        if stage_class().name == name:
            return stage_class
    raise ValueError(f"Unknown stage: {name}")

