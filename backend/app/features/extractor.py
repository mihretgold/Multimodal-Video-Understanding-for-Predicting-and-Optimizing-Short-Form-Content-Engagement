"""
Unified Feature Extractor Module
=================================
Orchestrates multimodal feature extraction for video segments.

Combines text, audio, and visual extractors with:
- Configurable modality selection
- Parallel extraction (if enabled)
- Segment-level multiprocessing (ProcessPoolExecutor)
- Caching of expensive operations
- Structured logging
"""

import os
import logging
import time
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from .text_features import TextFeatureExtractor, extract_text_features
from .audio_features import AudioFeatureExtractor, extract_audio_features
from .visual_features import VisualFeatureExtractor, extract_visual_features
from ..models import (
    SegmentFeatures,
    TextFeatures,
    AudioFeatures,
    VisualFeatures,
    DeepFeatures,
    Segment,
    SubtitleData,
)
from ..config import get_config, AblationConfig
from ..logging_config import get_research_logger, log_segment_features

logger = get_research_logger("features")


# =========================================================================
# MODULE-LEVEL WORKER FUNCTION (required for ProcessPoolExecutor)
# =========================================================================

def process_segment(args: Tuple) -> Tuple[str, Dict[str, Any]]:
    """
    Process a single segment in a worker process.

    This is a top-level function so it can be pickled by ProcessPoolExecutor.
    Each worker creates its own lightweight extractors (no deep models —
    those run in the main process to avoid per-worker model loading overhead).

    Args:
        args: Tuple of (video_path, segment_dict, subtitle_dict, ablation_kwargs, index, total)

    Returns:
        Tuple of (segment_id, features_dict)
    """
    video_path, segment_dict, subtitle_dict, ablation_kwargs, index, total = args

    seg_logger = logging.getLogger("features.worker")
    seg_id = segment_dict.get("segment_id", "unknown")
    seg_logger.info(f"Processing segment {index}/{total} — {seg_id}")

    try:
        segment = Segment.from_dict(segment_dict)
        subtitle_data = SubtitleData.from_dict(subtitle_dict) if subtitle_dict else None
        ablation = AblationConfig(**ablation_kwargs)

        extractor = FeatureExtractor(
            ablation=ablation,
            parallel=False,
            cache_enabled=False,
            enable_deep_features=False,
        )

        features = extractor.extract_segment_features(
            video_path, segment, subtitle_data
        )
        return (segment.segment_id, features.to_dict())

    except Exception as e:
        seg_logger.error(f"Worker failed for segment {seg_id}: {e}")
        fallback = SegmentFeatures(
            segment_id=seg_id,
            start_seconds=segment_dict.get("start_seconds", 0.0),
            end_seconds=segment_dict.get("end_seconds", 0.0),
            ablation_mode=ablation_kwargs.get("mode_name", "full_multimodal"),
        )
        return (seg_id, fallback.to_dict())


class FeatureExtractor:
    """
    Unified multimodal feature extractor.
    
    Orchestrates extraction from text, audio, and visual modalities
    with support for:
    - Ablation studies (enable/disable specific modalities)
    - Parallel extraction
    - Caching
    - Structured logging
    
    Usage:
        extractor = FeatureExtractor()
        
        # Extract all features for a segment
        features = extractor.extract_segment_features(
            video_path="video.mp4",
            segment=segment,
            subtitle_data=subtitles
        )
        
        # Or extract specific modalities
        text_features = extractor.extract_text(text, duration)
    """
    
    def __init__(
        self,
        ablation: Optional[AblationConfig] = None,
        parallel: bool = False,
        cache_enabled: bool = True,
        enable_deep_features: bool = True,
    ):
        """
        Initialize the feature extractor.
        
        Args:
            ablation: Ablation configuration (which modalities to enable)
            parallel: Whether to run modality extraction in parallel
            cache_enabled: Whether to cache results
            enable_deep_features: Whether to run deep learning feature extractors
        """
        config = get_config()
        self.ablation = ablation or config.ablation
        self.parallel = parallel
        self.cache_enabled = cache_enabled
        self.enable_deep_features = enable_deep_features
        
        # Initialize modality extractors
        self.text_extractor = TextFeatureExtractor()
        self.audio_extractor = AudioFeatureExtractor()
        self.visual_extractor = VisualFeatureExtractor()
        
        # Initialize deep feature extractor (lazy — models load on first use)
        self._deep_extractor = None
        if self.enable_deep_features:
            try:
                from ..deep_features import DeepFeatureExtractor
                self._deep_extractor = DeepFeatureExtractor()
            except Exception as e:
                logger.warning(f"Deep feature extractor unavailable: {e}")
                self._deep_extractor = None
        
        # Simple in-memory cache
        self._cache: Dict[str, SegmentFeatures] = {}
        
        logger.info(
            f"Initialized FeatureExtractor",
            extra={
                'ablation_mode': self.ablation.mode_name,
                'use_text': self.ablation.use_text,
                'use_audio': self.ablation.use_audio,
                'use_visual': self.ablation.use_visual,
                'use_cv_features': self.ablation.use_cv_features,
                'deep_features': self._deep_extractor is not None,
                'parallel': parallel
            }
        )
    
    def extract_segment_features(
        self,
        video_path: str,
        segment: Segment,
        subtitle_data: Optional[SubtitleData] = None,
        result_id: Optional[str] = None
    ) -> SegmentFeatures:
        """
        Extract all enabled modality features for a segment.
        
        Args:
            video_path: Path to video file
            segment: Segment to extract features for
            subtitle_data: Subtitle data for text features
            result_id: Result ID for logging
            
        Returns:
            SegmentFeatures object with all extracted features
        """
        start_time = time.time()
        cache_key = self._get_cache_key(video_path, segment)
        
        # Check cache
        if self.cache_enabled and cache_key in self._cache:
            logger.debug(f"Cache hit for segment {segment.segment_id}")
            return self._cache[cache_key]
        
        # Create features container
        features = SegmentFeatures(
            segment_id=segment.segment_id,
            start_seconds=segment.start_seconds,
            end_seconds=segment.end_seconds,
            ablation_mode=self.ablation.mode_name
        )
        
        # Extract based on ablation config
        if self.parallel and sum([self.ablation.use_text, self.ablation.use_audio, self.ablation.use_visual]) > 1:
            features = self._extract_parallel(video_path, segment, subtitle_data, features)
        else:
            features = self._extract_sequential(video_path, segment, subtitle_data, features)
        
        extraction_time = time.time() - start_time
        
        # Log the features
        log_segment_features(
            segment_id=segment.segment_id,
            features=self._summarize_features(features),
            modalities=features.modalities_present
        )
        
        logger.info(
            f"Extracted features for segment {segment.segment_id}",
            extra={
                'segment_id': segment.segment_id,
                'modalities': features.modalities_present,
                'extraction_time': extraction_time,
                'result_id': result_id
            }
        )
        
        # Cache result
        if self.cache_enabled:
            self._cache[cache_key] = features
        
        return features
    
    def _extract_sequential(
        self,
        video_path: str,
        segment: Segment,
        subtitle_data: Optional[SubtitleData],
        features: SegmentFeatures
    ) -> SegmentFeatures:
        """Extract features sequentially."""
        
        # Text features
        subtitle_text = ""
        if self.ablation.use_text and subtitle_data:
            try:
                subtitle_text = subtitle_data.get_text_in_range(
                    segment.start_seconds,
                    segment.end_seconds
                )
                features.text_features = self.text_extractor.extract(
                    subtitle_text,
                    duration_seconds=segment.duration_seconds
                )
            except Exception as e:
                logger.warning(f"Text feature extraction failed: {e}")
                features.text_features = TextFeatures()
        
        # Audio features
        if self.ablation.use_audio:
            try:
                features.audio_features = self.audio_extractor.extract(
                    video_path,
                    segment.start_seconds,
                    segment.end_seconds
                )
            except Exception as e:
                logger.warning(f"Audio feature extraction failed: {e}")
                features.audio_features = AudioFeatures()
        
        # Visual features
        if self.ablation.use_visual:
            try:
                features.visual_features = self.visual_extractor.extract(
                    video_path,
                    segment.start_seconds,
                    segment.end_seconds
                )
                # Zero out CV features if disabled (keep only signal-level)
                if not self.ablation.use_cv_features:
                    features.visual_features = self._zero_cv_features(features.visual_features)
            except Exception as e:
                logger.warning(f"Visual feature extraction failed: {e}")
                features.visual_features = VisualFeatures()
        
        # Deep learning features
        if self._deep_extractor is not None:
            try:
                raw_deep = self._deep_extractor.extract_all(
                    video_path,
                    segment.start_seconds,
                    segment.end_seconds,
                    subtitle_text=subtitle_text or None,
                )
                features.deep_features = DeepFeatures.from_extractor_output(raw_deep)
            except Exception as e:
                logger.warning(f"Deep feature extraction failed: {e}")
        
        return features
    
    def _extract_parallel(
        self,
        video_path: str,
        segment: Segment,
        subtitle_data: Optional[SubtitleData],
        features: SegmentFeatures
    ) -> SegmentFeatures:
        """Extract features in parallel using thread pool."""
        
        subtitle_text = ""
        if subtitle_data:
            subtitle_text = subtitle_data.get_text_in_range(
                segment.start_seconds,
                segment.end_seconds
            )
        
        max_workers = 4 if self._deep_extractor else 3
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            # Submit text extraction
            if self.ablation.use_text and subtitle_text:
                futures['text'] = executor.submit(
                    self.text_extractor.extract,
                    subtitle_text,
                    segment.duration_seconds
                )
            
            # Submit audio extraction
            if self.ablation.use_audio:
                futures['audio'] = executor.submit(
                    self.audio_extractor.extract,
                    video_path,
                    segment.start_seconds,
                    segment.end_seconds
                )
            
            # Submit visual extraction
            if self.ablation.use_visual:
                futures['visual'] = executor.submit(
                    self.visual_extractor.extract,
                    video_path,
                    segment.start_seconds,
                    segment.end_seconds
                )
            
            # Submit deep feature extraction
            if self._deep_extractor is not None:
                futures['deep'] = executor.submit(
                    self._deep_extractor.extract_all,
                    video_path,
                    segment.start_seconds,
                    segment.end_seconds,
                    subtitle_text or None,
                )
            
            # Collect results
            for modality, future in futures.items():
                try:
                    result = future.result(timeout=300)
                    if modality == 'text':
                        features.text_features = result
                    elif modality == 'audio':
                        features.audio_features = result
                    elif modality == 'visual':
                        if not self.ablation.use_cv_features:
                            result = self._zero_cv_features(result)
                        features.visual_features = result
                    elif modality == 'deep':
                        features.deep_features = DeepFeatures.from_extractor_output(result)
                except Exception as e:
                    logger.warning(f"Parallel {modality} extraction failed: {e}")
        
        return features
    
    def extract_text(
        self,
        text: str,
        duration_seconds: Optional[float] = None
    ) -> TextFeatures:
        """Extract text features only."""
        return self.text_extractor.extract(text, duration_seconds)
    
    def extract_audio(
        self,
        video_path: str,
        start_seconds: float,
        end_seconds: float
    ) -> AudioFeatures:
        """Extract audio features only."""
        return self.audio_extractor.extract(video_path, start_seconds, end_seconds)
    
    def extract_visual(
        self,
        video_path: str,
        start_seconds: float,
        end_seconds: float
    ) -> VisualFeatures:
        """Extract visual features only."""
        return self.visual_extractor.extract(video_path, start_seconds, end_seconds)
    
    def extract_batch(
        self,
        video_path: str,
        segments: List[Segment],
        subtitle_data: Optional[SubtitleData] = None,
        result_id: Optional[str] = None,
        use_multiprocessing: bool = True,
    ) -> Dict[str, SegmentFeatures]:
        """
        Extract features for multiple segments.

        When *use_multiprocessing* is True and there are enough segments,
        basic features (text, audio, visual, CV) are extracted in parallel
        across worker processes.  Deep learning features are always applied
        in the main process afterwards to avoid loading heavy models in
        every worker.

        Falls back to sequential extraction if multiprocessing fails.

        Returns dict mapping segment_id to SegmentFeatures.
        """
        n = len(segments)

        if use_multiprocessing and n >= 2:
            try:
                results = self.extract_features_parallel(
                    segments, video_path, subtitle_data, result_id
                )
                return results
            except Exception as e:
                logger.warning(
                    f"Multiprocessing extraction failed, falling back to sequential: {e}"
                )

        return self._extract_batch_sequential(
            video_path, segments, subtitle_data, result_id
        )

    # -----------------------------------------------------------------
    # Sequential batch (original path, also used as fallback)
    # -----------------------------------------------------------------

    def _extract_batch_sequential(
        self,
        video_path: str,
        segments: List[Segment],
        subtitle_data: Optional[SubtitleData] = None,
        result_id: Optional[str] = None,
    ) -> Dict[str, SegmentFeatures]:
        """Extract features for multiple segments sequentially."""
        results: Dict[str, SegmentFeatures] = {}
        n = len(segments)

        for i, segment in enumerate(segments):
            logging.info(f"Processing segment {i + 1}/{n}")

            features = self.extract_segment_features(
                video_path, segment, subtitle_data, result_id
            )
            results[segment.segment_id] = features

        return results

    # -----------------------------------------------------------------
    # Parallel batch via ProcessPoolExecutor
    # -----------------------------------------------------------------

    def extract_features_parallel(
        self,
        segments: List[Segment],
        video_path: str,
        subtitle_data: Optional[SubtitleData] = None,
        result_id: Optional[str] = None,
    ) -> Dict[str, SegmentFeatures]:
        """
        Extract features for multiple segments using multiprocessing.

        Each worker process handles one segment (text + audio + visual).
        Deep learning features are applied in the main process after the
        parallel phase completes, so heavy models are loaded only once.

        Args:
            segments: List of Segment objects.
            video_path: Path to the video file.
            subtitle_data: Optional subtitle data for text features.
            result_id: Optional result ID for logging.

        Returns:
            Dict mapping segment_id to SegmentFeatures.
        """
        max_workers = min(os.cpu_count() or 4, 8)
        n = len(segments)

        logger.info(
            f"Starting parallel feature extraction: "
            f"{n} segments, {max_workers} workers"
        )
        start_time = time.time()

        subtitle_dict = subtitle_data.to_dict() if subtitle_data else None
        ablation_kwargs = {
            "use_text": self.ablation.use_text,
            "use_audio": self.ablation.use_audio,
            "use_visual": self.ablation.use_visual,
            "use_cv_features": self.ablation.use_cv_features,
            "mode_name": self.ablation.mode_name,
        }

        worker_args = [
            (
                video_path,
                seg.to_dict(),
                subtitle_dict,
                ablation_kwargs,
                i + 1,
                n,
            )
            for i, seg in enumerate(segments)
        ]

        results: Dict[str, SegmentFeatures] = {}

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(process_segment, args): args[1]["segment_id"]
                for args in worker_args
            }

            for future in as_completed(futures):
                seg_id = futures[future]
                try:
                    returned_id, features_dict = future.result(timeout=600)
                    features = SegmentFeatures.from_dict(features_dict)
                    results[returned_id] = features
                    logger.info(
                        f"Segment {returned_id} done "
                        f"({len(results)}/{n} complete)"
                    )
                except Exception as e:
                    logger.error(f"Segment {seg_id} failed in pool: {e}")

        # Phase 2: apply deep features in the main process
        if self._deep_extractor is not None:
            logger.info("Applying deep learning features (main process)…")
            for segment in segments:
                if segment.segment_id not in results:
                    continue
                features = results[segment.segment_id]
                try:
                    subtitle_text = (
                        subtitle_data.get_text_in_range(
                            segment.start_seconds, segment.end_seconds
                        )
                        if subtitle_data
                        else None
                    )
                    raw_deep = self._deep_extractor.extract_all(
                        video_path,
                        segment.start_seconds,
                        segment.end_seconds,
                        subtitle_text=subtitle_text,
                    )
                    features.deep_features = DeepFeatures.from_extractor_output(
                        raw_deep
                    )
                except Exception as e:
                    logger.warning(
                        f"Deep feature extraction failed for {segment.segment_id}: {e}"
                    )

        elapsed = time.time() - start_time
        logger.info(
            f"Parallel extraction complete: {len(results)}/{n} segments "
            f"in {elapsed:.1f}s"
        )

        return results
    
    def _zero_cv_features(self, visual_features: VisualFeatures) -> VisualFeatures:
        """
        Zero out classical CV features, keeping only signal-level features.
        
        This is used for ablation studies to measure the contribution of
        CV features (edge detection, histogram analysis, etc.) vs. basic
        signal-level features (raw pixel statistics).
        
        Signal-level features preserved:
        - motion_intensity, scene_change_count/rate
        - brightness_mean/std, color_variance
        
        CV features zeroed:
        - contrast, edge_density, edge_intensity
        - motion_magnitude, histogram_diff_mean, scene_boundaries
        """
        return VisualFeatures(
            # Preserve signal-level features
            motion_intensity=visual_features.motion_intensity,
            scene_change_count=visual_features.scene_change_count,
            scene_change_rate=visual_features.scene_change_rate,
            brightness_mean=visual_features.brightness_mean,
            brightness_std=visual_features.brightness_std,
            color_variance=visual_features.color_variance,
            face_presence_ratio=visual_features.face_presence_ratio,
            text_overlay_ratio=visual_features.text_overlay_ratio,
            # Zero out CV features
            contrast=0.0,
            edge_density=0.0,
            edge_intensity=0.0,
            motion_magnitude=0.0,
            histogram_diff_mean=0.0,
            scene_boundaries=[]
        )
    
    def _get_cache_key(self, video_path: str, segment: Segment) -> str:
        """Generate cache key for a segment."""
        return f"{video_path}:{segment.start_seconds}:{segment.end_seconds}:{self.ablation.mode_name}"
    
    def _summarize_features(self, features: SegmentFeatures) -> Dict[str, Any]:
        """Create summary of features for logging."""
        summary = {}
        
        if features.text_features:
            tf = features.text_features
            summary['text'] = {
                'word_count': tf.word_count,
                'sentiment': round(tf.sentiment_score, 3),
                'speech_rate': round(tf.word_count / max(1, features.duration_seconds), 2)
            }
        
        if features.audio_features:
            af = features.audio_features
            summary['audio'] = {
                'energy': round(af.energy_mean, 3),
                'silence_ratio': round(af.silence_ratio, 3),
                'dynamics': round(af.volume_dynamics, 2)
            }
        
        if features.visual_features:
            vf = features.visual_features
            summary['visual'] = {
                'motion': round(vf.motion_intensity, 3),
                'scene_changes': vf.scene_change_count,
                'brightness': round(vf.brightness_mean, 3)
            }
        
        if features.deep_features:
            df = features.deep_features
            summary['deep'] = {
                'clip_variance': round(df.clip_semantic_variance, 4),
                'audio_emotion': df.audio_emotion_label,
                'text_coherence': round(df.text_coherence_score, 3),
                'face_emotion': df.face_emotion_label,
            }
        
        return summary
    
    def clear_cache(self):
        """Clear the feature cache."""
        self._cache.clear()
        logger.info("Cleared feature cache")


def create_extractor(
    ablation_mode: str = 'full_multimodal',
    parallel: bool = False,
    enable_deep_features: bool = True,
) -> FeatureExtractor:
    """
    Factory function to create a feature extractor.
    
    Args:
        ablation_mode: Ablation mode name
        parallel: Whether to use parallel extraction
        enable_deep_features: Whether to enable deep learning feature extraction
        
    Returns:
        Configured FeatureExtractor instance
    """
    ablation_map = {
        'text_only': AblationConfig.text_only,
        'audio_only': AblationConfig.audio_only,
        'visual_only': AblationConfig.visual_only,
        'visual_signal_only': AblationConfig.visual_signal_only,  # Visual without CV
        'text_audio': AblationConfig.text_audio,
        'full_no_cv': AblationConfig.full_no_cv,  # All modalities, no CV features
        'full_multimodal': AblationConfig.full_multimodal,
    }
    
    ablation = ablation_map.get(ablation_mode, AblationConfig.full_multimodal)()
    return FeatureExtractor(
        ablation=ablation,
        parallel=parallel,
        enable_deep_features=enable_deep_features,
    )

