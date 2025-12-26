"""
Feature Normalization Module
============================
Normalizes multimodal features to comparable scales.

Normalization is critical for fair comparison across modalities.
Different approaches:
- MinMax: Scale to [0, 1] range
- ZScore: Standardize to mean=0, std=1
- Percentile: Map to percentile ranks
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ..models import SegmentFeatures, TextFeatures, AudioFeatures, VisualFeatures

logger = logging.getLogger(__name__)


@dataclass
class FeatureStats:
    """Statistics for a single feature dimension."""
    name: str
    min_val: float = 0.0
    max_val: float = 1.0
    mean_val: float = 0.5
    std_val: float = 0.25
    count: int = 0
    
    def update(self, value: float) -> None:
        """Update running statistics with a new value."""
        self.count += 1
        if self.count == 1:
            self.min_val = self.max_val = self.mean_val = value
            self.std_val = 0.0
        else:
            self.min_val = min(self.min_val, value)
            self.max_val = max(self.max_val, value)
            # Running mean update
            delta = value - self.mean_val
            self.mean_val += delta / self.count


@dataclass
class NormalizationConfig:
    """Configuration for feature normalization."""
    
    # Text feature ranges (expected values)
    text_word_count_range: Tuple[float, float] = (0, 200)
    text_sentiment_range: Tuple[float, float] = (-1, 1)
    text_keyword_density_range: Tuple[float, float] = (0, 0.3)
    
    # Audio feature ranges
    audio_energy_range: Tuple[float, float] = (0, 1)
    audio_silence_range: Tuple[float, float] = (0, 1)
    audio_dynamics_range: Tuple[float, float] = (1, 10)
    
    # Visual feature ranges
    visual_motion_range: Tuple[float, float] = (0, 1)
    visual_scene_change_range: Tuple[float, float] = (0, 10)
    visual_brightness_range: Tuple[float, float] = (0, 1)


class FeatureNormalizer:
    """
    Normalizes features to [0, 1] range for fair comparison.
    
    Supports two modes:
    1. Fixed ranges: Uses predefined expected ranges
    2. Adaptive: Learns ranges from the data
    
    Usage:
        normalizer = FeatureNormalizer()
        
        # Normalize a single value
        normalized = normalizer.normalize_value(raw_value, 'text_sentiment')
        
        # Normalize all features for a segment
        normalized_features = normalizer.normalize_segment(features)
    """
    
    def __init__(
        self,
        config: Optional[NormalizationConfig] = None,
        adaptive: bool = False
    ):
        """
        Initialize the normalizer.
        
        Args:
            config: Configuration with expected ranges
            adaptive: Whether to learn ranges from data
        """
        self.config = config or NormalizationConfig()
        self.adaptive = adaptive
        
        # Feature statistics for adaptive normalization
        self._stats: Dict[str, FeatureStats] = {}
        
        # Build range lookup
        self._ranges = self._build_ranges()
    
    def _build_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Build lookup table for feature ranges."""
        return {
            # Text features
            'word_count': self.config.text_word_count_range,
            'sentence_count': (0, 20),
            'sentiment_score': self.config.text_sentiment_range,
            'keyword_density': self.config.text_keyword_density_range,
            'question_count': (0, 10),
            'exclamation_count': (0, 10),
            'avg_word_length': (3, 10),
            
            # Audio features
            'energy_mean': self.config.audio_energy_range,
            'energy_std': (0, 0.5),
            'silence_ratio': self.config.audio_silence_range,
            'volume_dynamics': self.config.audio_dynamics_range,
            'spectral_centroid': (0, 8000),
            'speech_rate': (0, 5),
            
            # Visual features
            'motion_intensity': self.config.visual_motion_range,
            'scene_change_count': self.config.visual_scene_change_range,
            'scene_change_rate': (0, 0.5),
            'brightness_mean': self.config.visual_brightness_range,
            'brightness_std': (0, 0.3),
            'color_variance': (0, 1),
        }
    
    def normalize_value(
        self,
        value: float,
        feature_name: str,
        clip: bool = True
    ) -> float:
        """
        Normalize a single feature value to [0, 1] range.
        
        Args:
            value: Raw feature value
            feature_name: Name of the feature
            clip: Whether to clip values outside expected range
            
        Returns:
            Normalized value in [0, 1]
        """
        if feature_name not in self._ranges:
            logger.warning(f"Unknown feature: {feature_name}, returning as-is")
            return value
        
        min_val, max_val = self._ranges[feature_name]
        
        # Handle edge cases
        if max_val == min_val:
            return 0.5
        
        # Normalize
        normalized = (value - min_val) / (max_val - min_val)
        
        # Optionally clip to [0, 1]
        if clip:
            normalized = max(0.0, min(1.0, normalized))
        
        # Update adaptive stats
        if self.adaptive:
            self._update_stats(feature_name, value)
        
        return normalized
    
    def _update_stats(self, feature_name: str, value: float) -> None:
        """Update adaptive statistics for a feature."""
        if feature_name not in self._stats:
            self._stats[feature_name] = FeatureStats(name=feature_name)
        self._stats[feature_name].update(value)
    
    def normalize_text_features(self, features: TextFeatures) -> Dict[str, float]:
        """
        Normalize all text features.
        
        Returns dict of normalized values.
        """
        if features is None:
            return {}
        
        return {
            'word_count': self.normalize_value(features.word_count, 'word_count'),
            'sentence_count': self.normalize_value(features.sentence_count, 'sentence_count'),
            'sentiment_score': self.normalize_value(features.sentiment_score, 'sentiment_score'),
            'keyword_density': self.normalize_value(features.keyword_density, 'keyword_density'),
            'question_count': self.normalize_value(features.question_count, 'question_count'),
            'exclamation_count': self.normalize_value(features.exclamation_count, 'exclamation_count'),
        }
    
    def normalize_audio_features(self, features: AudioFeatures) -> Dict[str, float]:
        """Normalize all audio features."""
        if features is None:
            return {}
        
        return {
            'energy_mean': self.normalize_value(features.energy_mean, 'energy_mean'),
            'energy_std': self.normalize_value(features.energy_std, 'energy_std'),
            'silence_ratio': self.normalize_value(features.silence_ratio, 'silence_ratio'),
            'volume_dynamics': self.normalize_value(features.volume_dynamics, 'volume_dynamics'),
            'spectral_centroid': self.normalize_value(features.spectral_centroid, 'spectral_centroid'),
        }
    
    def normalize_visual_features(self, features: VisualFeatures) -> Dict[str, float]:
        """Normalize all visual features."""
        if features is None:
            return {}
        
        return {
            'motion_intensity': self.normalize_value(features.motion_intensity, 'motion_intensity'),
            'scene_change_count': self.normalize_value(features.scene_change_count, 'scene_change_count'),
            'scene_change_rate': self.normalize_value(features.scene_change_rate, 'scene_change_rate'),
            'brightness_mean': self.normalize_value(features.brightness_mean, 'brightness_mean'),
            'brightness_std': self.normalize_value(features.brightness_std, 'brightness_std'),
            'color_variance': self.normalize_value(features.color_variance, 'color_variance'),
        }
    
    def normalize_segment(self, features: SegmentFeatures) -> Dict[str, Dict[str, float]]:
        """
        Normalize all features for a segment.
        
        Returns nested dict of {modality: {feature: normalized_value}}.
        """
        result = {}
        
        if features.text_features:
            result['text'] = self.normalize_text_features(features.text_features)
        
        if features.audio_features:
            result['audio'] = self.normalize_audio_features(features.audio_features)
        
        if features.visual_features:
            result['visual'] = self.normalize_visual_features(features.visual_features)
        
        return result
    
    def get_stats(self) -> Dict[str, Dict]:
        """Get learned statistics for all features."""
        return {
            name: {
                'min': stats.min_val,
                'max': stats.max_val,
                'mean': stats.mean_val,
                'count': stats.count
            }
            for name, stats in self._stats.items()
        }

