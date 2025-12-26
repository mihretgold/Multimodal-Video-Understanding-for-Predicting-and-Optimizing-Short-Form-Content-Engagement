"""
Unified Feature Extractor Module
=================================
Orchestrates multimodal feature extraction for video segments.

Combines text, audio, and visual extractors with:
- Configurable modality selection
- Parallel extraction (if enabled)
- Caching of expensive operations
- Structured logging
"""

import logging
import time
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from .text_features import TextFeatureExtractor, extract_text_features
from .audio_features import AudioFeatureExtractor, extract_audio_features
from .visual_features import VisualFeatureExtractor, extract_visual_features
from ..models import (
    SegmentFeatures,
    TextFeatures,
    AudioFeatures,
    VisualFeatures,
    Segment,
    SubtitleData,
)
from ..config import get_config, AblationConfig
from ..logging_config import get_research_logger, log_segment_features

logger = get_research_logger("features")


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
        cache_enabled: bool = True
    ):
        """
        Initialize the feature extractor.
        
        Args:
            ablation: Ablation configuration (which modalities to enable)
            parallel: Whether to run modality extraction in parallel
            cache_enabled: Whether to cache results
        """
        config = get_config()
        self.ablation = ablation or config.ablation
        self.parallel = parallel
        self.cache_enabled = cache_enabled
        
        # Initialize modality extractors
        self.text_extractor = TextFeatureExtractor()
        self.audio_extractor = AudioFeatureExtractor()
        self.visual_extractor = VisualFeatureExtractor()
        
        # Simple in-memory cache
        self._cache: Dict[str, SegmentFeatures] = {}
        
        logger.info(
            f"Initialized FeatureExtractor",
            extra={
                'ablation_mode': self.ablation.mode_name,
                'use_text': self.ablation.use_text,
                'use_audio': self.ablation.use_audio,
                'use_visual': self.ablation.use_visual,
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
        if self.ablation.use_text and subtitle_data:
            try:
                text = subtitle_data.get_text_in_range(
                    segment.start_seconds,
                    segment.end_seconds
                )
                features.text_features = self.text_extractor.extract(
                    text,
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
            except Exception as e:
                logger.warning(f"Visual feature extraction failed: {e}")
                features.visual_features = VisualFeatures()
        
        return features
    
    def _extract_parallel(
        self,
        video_path: str,
        segment: Segment,
        subtitle_data: Optional[SubtitleData],
        features: SegmentFeatures
    ) -> SegmentFeatures:
        """Extract features in parallel using thread pool."""
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            
            # Submit text extraction
            if self.ablation.use_text and subtitle_data:
                text = subtitle_data.get_text_in_range(
                    segment.start_seconds,
                    segment.end_seconds
                )
                futures['text'] = executor.submit(
                    self.text_extractor.extract,
                    text,
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
            
            # Collect results
            for modality, future in futures.items():
                try:
                    result = future.result(timeout=120)
                    if modality == 'text':
                        features.text_features = result
                    elif modality == 'audio':
                        features.audio_features = result
                    elif modality == 'visual':
                        features.visual_features = result
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
        result_id: Optional[str] = None
    ) -> Dict[str, SegmentFeatures]:
        """
        Extract features for multiple segments.
        
        Returns dict mapping segment_id to features.
        """
        results = {}
        
        for i, segment in enumerate(segments):
            logger.debug(f"Extracting features for segment {i+1}/{len(segments)}")
            
            features = self.extract_segment_features(
                video_path,
                segment,
                subtitle_data,
                result_id
            )
            results[segment.segment_id] = features
        
        return results
    
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
        
        return summary
    
    def clear_cache(self):
        """Clear the feature cache."""
        self._cache.clear()
        logger.info("Cleared feature cache")


def create_extractor(
    ablation_mode: str = 'full_multimodal',
    parallel: bool = False
) -> FeatureExtractor:
    """
    Factory function to create a feature extractor.
    
    Args:
        ablation_mode: Ablation mode name
        parallel: Whether to use parallel extraction
        
    Returns:
        Configured FeatureExtractor instance
    """
    ablation_map = {
        'text_only': AblationConfig.text_only,
        'audio_only': AblationConfig.audio_only,
        'visual_only': AblationConfig.visual_only,
        'text_audio': AblationConfig.text_audio,
        'full_multimodal': AblationConfig.full_multimodal,
    }
    
    ablation = ablation_map.get(ablation_mode, AblationConfig.full_multimodal)()
    return FeatureExtractor(ablation=ablation, parallel=parallel)

