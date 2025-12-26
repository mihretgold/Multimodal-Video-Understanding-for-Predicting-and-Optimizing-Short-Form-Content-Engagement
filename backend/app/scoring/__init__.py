"""
Engagement Scoring Module
=========================
Scores and ranks video segments based on multimodal features.

This module implements:
- Rule-based scoring with configurable weights
- Feature normalization for fair comparison
- Segment ranking
- Score explanations for interpretability

Usage:
    from app.scoring import EngagementScorer, RuleBasedScoring
    
    scorer = EngagementScorer()
    scored_segments = scorer.score_segments(segments, features_dict)
    
    # Get ranked list
    ranked = scorer.rank_segments(scored_segments)
"""

from .scorer import EngagementScorer
from .strategies import (
    ScoringStrategy,
    RuleBasedScoring,
    NormalizedScoring,
    LearnedScoring,
)
from .normalizers import FeatureNormalizer
from .ranker import SegmentRanker

__all__ = [
    'EngagementScorer',
    'ScoringStrategy',
    'RuleBasedScoring',
    'NormalizedScoring',
    'LearnedScoring',
    'FeatureNormalizer',
    'SegmentRanker',
]

