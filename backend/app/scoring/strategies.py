"""
Scoring Strategies Module
=========================
Different approaches for scoring segment engagement.

RESEARCH CONTEXT:
-----------------
Engagement prediction is challenging because:
1. "Engagement" is multifaceted (views, likes, shares, completion rate)
2. No ground truth labels available for this project
3. Engagement varies by platform, audience, and content type

OUR APPROACH:
We implement multiple scoring strategies to enable:
- Interpretable baselines (rule-based)
- Statistical normalization (z-scores)
- Future learned models (regression on labels)

Strategies:
1. RuleBasedScoring - Heuristic weights based on engagement research
2. NormalizedScoring - Weighted combination of z-score normalized features
3. LearnedScoring - Placeholder for trained regressor (future work)

The scoring function follows the general form:
    E(S) = Σ_m w_m * f_m(S)  where m ∈ {text, audio, visual}

Each modality score is itself a weighted combination of features within that modality.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field

from .normalizers import FeatureNormalizer
from ..models import SegmentFeatures, TextFeatures, AudioFeatures, VisualFeatures, ScoreBreakdown
from ..config import ScoringConfig, AblationConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class ScoringWeights:
    """
    Weights for combining scores across modalities and features.
    
    RESEARCH DECISION: Weight selection rationale.
    
    MODALITY WEIGHTS (text=0.4, audio=0.3, visual=0.3):
    - Text highest: dialogue content is most reliable engagement signal
    - Audio/visual equal: both contribute but are less reliable alone
    
    INTRA-MODALITY WEIGHTS:
    
    Text (sentiment=0.3, keyword=0.25, question=0.15, exclamation=0.15, density=0.15):
    - Sentiment dominates: emotional content drives shares
    - Keywords matter: trending topics increase discoverability
    - Questions/exclamations: hooks and emphasis
    - Word density: pacing indicator
    
    Audio (energy=0.35, dynamics=0.25, silence=-0.2, variation=0.2):
    - Energy dominates: loud = exciting in most contexts
    - Dynamics: variation maintains attention
    - Silence penalized: dead air loses viewers
    - Variation: predictability is boring
    
    Visual (motion=0.35, scene=0.3, brightness=0.15, color=0.2):
    - Motion highest: movement captures attention
    - Scene changes: editing pace matters
    - Brightness/color: secondary visual interest
    
    These weights are HYPERPARAMETERS - future work can tune via grid search
    with human-annotated engagement labels.
    """
    
    # Modality weights (should sum to 1.0)
    text_weight: float = 0.4   # Dialogue content most reliable
    audio_weight: float = 0.3  # Energy/dynamics contribute
    visual_weight: float = 0.3 # Motion/scenes contribute
    
    # Text feature weights (within text modality, sum to 1.0)
    text_sentiment_weight: float = 0.3      # Emotional content
    text_keyword_weight: float = 0.25       # Topic relevance
    text_question_weight: float = 0.15      # Hooks ("did you know?")
    text_exclamation_weight: float = 0.15   # Emphasis/excitement
    text_word_density_weight: float = 0.15  # Pacing (words per second)
    
    # Audio feature weights (sum to 1.0)
    audio_energy_weight: float = 0.35       # Loudness = excitement
    audio_dynamics_weight: float = 0.25     # Volume variation
    audio_silence_penalty: float = 0.2      # Penalize dead air
    audio_variation_weight: float = 0.2     # Unpredictability
    
    # Visual feature weights (sum to 1.0)
    visual_motion_weight: float = 0.35      # Movement captures attention
    visual_scene_change_weight: float = 0.3 # Editing pace
    visual_brightness_weight: float = 0.15  # Visibility on mobile
    visual_color_weight: float = 0.2        # Visual richness
    
    @classmethod
    def from_config(cls, config: ScoringConfig) -> "ScoringWeights":
        """Create weights from ScoringConfig."""
        return cls(
            text_weight=config.text_weight,
            audio_weight=config.audio_weight,
            visual_weight=config.visual_weight
        )
    
    @classmethod
    def for_ablation(cls, ablation: AblationConfig) -> "ScoringWeights":
        """Create weights respecting ablation mode."""
        weights = cls()
        
        # Zero out disabled modalities
        if not ablation.use_text:
            weights.text_weight = 0.0
        if not ablation.use_audio:
            weights.audio_weight = 0.0
        if not ablation.use_visual:
            weights.visual_weight = 0.0
        
        # Renormalize remaining weights
        total = weights.text_weight + weights.audio_weight + weights.visual_weight
        if total > 0:
            weights.text_weight /= total
            weights.audio_weight /= total
            weights.visual_weight /= total
        
        return weights
    
    def to_dict(self) -> Dict[str, float]:
        """Export weights as dictionary."""
        return {
            'text_weight': self.text_weight,
            'audio_weight': self.audio_weight,
            'visual_weight': self.visual_weight,
        }


class ScoringStrategy(ABC):
    """
    Abstract base class for scoring strategies.
    
    Each strategy implements a different approach to combining
    multimodal features into an engagement score.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging."""
        pass
    
    @abstractmethod
    def score(
        self,
        features: SegmentFeatures,
        weights: ScoringWeights
    ) -> ScoreBreakdown:
        """
        Score a segment based on its features.
        
        Args:
            features: Extracted multimodal features
            weights: Scoring weights configuration
            
        Returns:
            ScoreBreakdown with total and component scores
        """
        pass
    
    def score_batch(
        self,
        features_list: List[SegmentFeatures],
        weights: ScoringWeights
    ) -> List[ScoreBreakdown]:
        """Score multiple segments."""
        return [self.score(f, weights) for f in features_list]


class RuleBasedScoring(ScoringStrategy):
    """
    Rule-based scoring using heuristic weights.
    
    This strategy uses domain knowledge about what makes
    content engaging to weight different features:
    
    - High sentiment (positive or negative) = more engaging
    - Questions/exclamations = attention grabbing
    - Higher audio energy = more dynamic
    - Motion = visual interest
    - Scene changes = variety
    """
    
    @property
    def name(self) -> str:
        return "rule_based"
    
    def __init__(self):
        self.normalizer = FeatureNormalizer()
    
    def score(
        self,
        features: SegmentFeatures,
        weights: ScoringWeights
    ) -> ScoreBreakdown:
        """Calculate engagement score using heuristic rules."""
        
        text_score = 0.0
        audio_score = 0.0
        visual_score = 0.0
        
        # Score text features
        if features.text_features and weights.text_weight > 0:
            text_score = self._score_text(features.text_features, weights)
        
        # Score audio features
        if features.audio_features and weights.audio_weight > 0:
            audio_score = self._score_audio(features.audio_features, weights)
        
        # Score visual features
        if features.visual_features and weights.visual_weight > 0:
            visual_score = self._score_visual(features.visual_features, weights)
        
        # Weighted combination
        total_score = (
            text_score * weights.text_weight +
            audio_score * weights.audio_weight +
            visual_score * weights.visual_weight
        )
        
        return ScoreBreakdown(
            total_score=total_score,
            text_score=text_score,
            audio_score=audio_score,
            visual_score=visual_score,
            text_weight=weights.text_weight,
            audio_weight=weights.audio_weight,
            visual_weight=weights.visual_weight,
            scoring_method=self.name
        )
    
    def _score_text(self, features: TextFeatures, weights: ScoringWeights) -> float:
        """Score text features."""
        score = 0.0
        
        # Sentiment: Both strong positive AND negative are engaging
        # Using absolute value so both extremes score high
        sentiment_magnitude = abs(features.sentiment_score)
        score += sentiment_magnitude * weights.text_sentiment_weight
        
        # Keyword density (engagement words)
        keyword_score = self.normalizer.normalize_value(
            features.keyword_density, 'keyword_density'
        )
        score += keyword_score * weights.text_keyword_weight
        
        # Questions are engaging (rhetorical, to audience, etc.)
        question_score = min(1.0, features.question_count / 3)  # Cap at 3
        score += question_score * weights.text_question_weight
        
        # Exclamations indicate excitement/emphasis
        exclamation_score = min(1.0, features.exclamation_count / 3)
        score += exclamation_score * weights.text_exclamation_weight
        
        # Word density (moderate speech rate is engaging)
        # Too slow is boring, too fast is overwhelming
        # Optimal around 2-3 words/second
        word_count_norm = self.normalizer.normalize_value(
            features.word_count, 'word_count'
        )
        score += word_count_norm * weights.text_word_density_weight
        
        return min(1.0, score)
    
    def _score_audio(self, features: AudioFeatures, weights: ScoringWeights) -> float:
        """Score audio features."""
        score = 0.0
        
        # Higher energy = more dynamic/engaging
        energy_score = self.normalizer.normalize_value(
            features.energy_mean, 'energy_mean'
        )
        score += energy_score * weights.audio_energy_weight
        
        # Volume dynamics (variation = interest)
        dynamics_score = self.normalizer.normalize_value(
            features.volume_dynamics, 'volume_dynamics'
        )
        score += dynamics_score * weights.audio_dynamics_weight
        
        # Silence penalty (too much silence = less engaging)
        silence_penalty = features.silence_ratio * weights.audio_silence_penalty
        score -= silence_penalty
        
        # Energy variation (std) indicates dynamic content
        variation_score = self.normalizer.normalize_value(
            features.energy_std, 'energy_std'
        )
        score += variation_score * weights.audio_variation_weight
        
        return max(0.0, min(1.0, score))
    
    def _score_visual(self, features: VisualFeatures, weights: ScoringWeights) -> float:
        """Score visual features."""
        score = 0.0
        
        # Motion intensity (movement = visual interest)
        motion_score = self.normalizer.normalize_value(
            features.motion_intensity, 'motion_intensity'
        )
        score += motion_score * weights.visual_motion_weight
        
        # Scene changes (variety, but not too many)
        # Optimal is moderate scene changes
        scene_norm = self.normalizer.normalize_value(
            features.scene_change_count, 'scene_change_count'
        )
        # Penalty for too many scene changes (jarring)
        if scene_norm > 0.7:
            scene_norm = 1.0 - (scene_norm - 0.7) / 0.3
        score += scene_norm * weights.visual_scene_change_weight
        
        # Brightness (mid-range is often most engaging)
        brightness = features.brightness_mean
        # Penalty for very dark or very bright
        brightness_score = 1.0 - abs(brightness - 0.5) * 2
        brightness_score = max(0.0, brightness_score)
        score += brightness_score * weights.visual_brightness_weight
        
        # Color variance (more colorful = more visually interesting)
        color_score = self.normalizer.normalize_value(
            features.color_variance, 'color_variance'
        )
        score += color_score * weights.visual_color_weight
        
        return max(0.0, min(1.0, score))


class NormalizedScoring(ScoringStrategy):
    """
    Scoring using normalized feature values.
    
    All features are normalized to [0, 1] and combined
    with configurable weights. Simpler than rule-based
    but less interpretable.
    """
    
    @property
    def name(self) -> str:
        return "normalized"
    
    def __init__(self):
        self.normalizer = FeatureNormalizer()
    
    def score(
        self,
        features: SegmentFeatures,
        weights: ScoringWeights
    ) -> ScoreBreakdown:
        """Calculate score from normalized features."""
        
        # Normalize all features
        normalized = self.normalizer.normalize_segment(features)
        
        # Average within each modality
        text_score = self._average_modality(normalized.get('text', {}))
        audio_score = self._average_modality(normalized.get('audio', {}))
        visual_score = self._average_modality(normalized.get('visual', {}))
        
        # Apply modality weights
        total_score = (
            text_score * weights.text_weight +
            audio_score * weights.audio_weight +
            visual_score * weights.visual_weight
        )
        
        return ScoreBreakdown(
            total_score=total_score,
            text_score=text_score,
            audio_score=audio_score,
            visual_score=visual_score,
            text_weight=weights.text_weight,
            audio_weight=weights.audio_weight,
            visual_weight=weights.visual_weight,
            scoring_method=self.name
        )
    
    def _average_modality(self, features: Dict[str, float]) -> float:
        """Calculate average of normalized features."""
        if not features:
            return 0.0
        return sum(features.values()) / len(features)


class LearnedScoring(ScoringStrategy):
    """
    Placeholder for learned scoring model.
    
    Would be trained on engagement labels (likes, views, etc.)
    to learn optimal feature weights.
    
    For now, falls back to rule-based scoring.
    """
    
    @property
    def name(self) -> str:
        return "learned"
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self._fallback = RuleBasedScoring()
        
        # Load model if path provided
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, path: str) -> None:
        """Load a trained model."""
        # Placeholder - would load a trained model here
        logger.info(f"Would load model from: {path}")
        self.model = None
    
    def score(
        self,
        features: SegmentFeatures,
        weights: ScoringWeights
    ) -> ScoreBreakdown:
        """Score using learned model (or fallback)."""
        
        if self.model is None:
            # Fallback to rule-based
            result = self._fallback.score(features, weights)
            result.scoring_method = "learned_fallback"
            return result
        
        # Would use model.predict here
        # For now, return fallback
        return self._fallback.score(features, weights)
    
    def train(
        self,
        features_list: List[SegmentFeatures],
        labels: List[float]
    ) -> None:
        """
        Train the model on labeled data.
        
        Args:
            features_list: List of segment features
            labels: Engagement labels (e.g., normalized view counts)
        """
        # Placeholder - would train a model here
        logger.info(f"Would train on {len(features_list)} samples")


# Strategy registry
STRATEGY_REGISTRY = {
    'rule_based': RuleBasedScoring,
    'normalized': NormalizedScoring,
    'learned': LearnedScoring,
}


def get_strategy(name: str) -> ScoringStrategy:
    """Get a scoring strategy by name."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown scoring strategy: {name}")
    return STRATEGY_REGISTRY[name]()

