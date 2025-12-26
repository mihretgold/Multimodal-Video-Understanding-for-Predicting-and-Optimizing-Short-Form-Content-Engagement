"""
Engagement Scorer Module
========================
Main interface for scoring video segments.

Combines scoring strategies, normalization, and ranking
into a unified workflow.
"""

import logging
from typing import Dict, List, Optional, Any

from .strategies import (
    ScoringStrategy,
    ScoringWeights,
    RuleBasedScoring,
    NormalizedScoring,
    LearnedScoring,
    get_strategy,
)
from .normalizers import FeatureNormalizer
from .ranker import SegmentRanker, RankingResult
from ..models import Segment, SegmentFeatures, ScoreBreakdown
from ..config import ScoringConfig, AblationConfig, get_config
from ..logging_config import get_research_logger, log_segment_score, log_pipeline_decision

logger = get_research_logger("scoring")


class EngagementScorer:
    """
    Main class for scoring segment engagement.
    
    Orchestrates:
    - Scoring strategy selection
    - Weight configuration
    - Score computation
    - Ranking
    
    Usage:
        scorer = EngagementScorer()
        
        # Score a single segment
        score = scorer.score_segment(segment, features)
        
        # Score and rank multiple segments
        ranked = scorer.score_and_rank(segments, features_dict)
    """
    
    def __init__(
        self,
        strategy: Optional[str] = None,
        config: Optional[ScoringConfig] = None,
        ablation: Optional[AblationConfig] = None,
        custom_weights: Optional[ScoringWeights] = None
    ):
        """
        Initialize the engagement scorer.
        
        Args:
            strategy: Scoring strategy name ('rule_based', 'normalized', 'learned')
            config: Scoring configuration
            ablation: Ablation configuration (affects weights)
            custom_weights: Override weights
        """
        self.config = config or get_config().scoring
        self.ablation = ablation or get_config().ablation
        
        # Get or create weights
        if custom_weights:
            self.weights = custom_weights
        elif ablation:
            # Create weights respecting ablation
            self.weights = ScoringWeights.for_ablation(self.ablation)
            # Override modality weights from config
            base_weights = ScoringWeights.from_config(self.config)
            if self.ablation.use_text:
                self.weights.text_sentiment_weight = base_weights.text_sentiment_weight
                self.weights.text_keyword_weight = base_weights.text_keyword_weight
            if self.ablation.use_audio:
                self.weights.audio_energy_weight = base_weights.audio_energy_weight
                self.weights.audio_dynamics_weight = base_weights.audio_dynamics_weight
            if self.ablation.use_visual:
                self.weights.visual_motion_weight = base_weights.visual_motion_weight
                self.weights.visual_scene_change_weight = base_weights.visual_scene_change_weight
        else:
            self.weights = ScoringWeights.from_config(self.config)
        
        # Initialize strategy
        strategy_name = strategy or self.config.mode
        self.strategy = get_strategy(strategy_name)
        
        # Initialize ranker
        self.ranker = SegmentRanker()
        
        logger.info(
            f"Initialized EngagementScorer",
            extra={
                'strategy': self.strategy.name,
                'text_weight': self.weights.text_weight,
                'audio_weight': self.weights.audio_weight,
                'visual_weight': self.weights.visual_weight,
                'ablation_mode': self.ablation.mode_name
            }
        )
    
    def score_segment(
        self,
        segment: Segment,
        features: Optional[SegmentFeatures] = None
    ) -> ScoreBreakdown:
        """
        Score a single segment.
        
        Args:
            segment: Segment to score
            features: Optional features (uses segment.features if not provided)
            
        Returns:
            ScoreBreakdown with scores
        """
        # Get features
        feat = features or segment.features
        
        if feat is None:
            logger.warning(f"No features for segment {segment.segment_id}")
            return ScoreBreakdown(total_score=0.0, scoring_method="no_features")
        
        # Compute score
        score = self.strategy.score(feat, self.weights)
        
        # Update segment
        segment.score = score
        
        # Log
        log_segment_score(
            segment_id=segment.segment_id,
            total_score=score.total_score,
            text_score=score.text_score,
            audio_score=score.audio_score,
            visual_score=score.visual_score,
            weights=self.weights.to_dict(),
            method=score.scoring_method
        )
        
        return score
    
    def score_segments(
        self,
        segments: List[Segment],
        features_dict: Optional[Dict[str, SegmentFeatures]] = None
    ) -> List[Segment]:
        """
        Score multiple segments.
        
        Args:
            segments: List of segments
            features_dict: Optional mapping of segment_id to features
            
        Returns:
            Segments with scores attached
        """
        for segment in segments:
            features = None
            if features_dict:
                features = features_dict.get(segment.segment_id)
            
            self.score_segment(segment, features)
        
        return segments
    
    def score_and_rank(
        self,
        segments: List[Segment],
        features_dict: Optional[Dict[str, SegmentFeatures]] = None,
        top_k: Optional[int] = None,
        result_id: Optional[str] = None
    ) -> List[Segment]:
        """
        Score and rank segments.
        
        Args:
            segments: List of segments
            features_dict: Optional features mapping
            top_k: Optional limit on results
            result_id: For logging
            
        Returns:
            Ranked segments (best first)
        """
        # Score all segments
        scored = self.score_segments(segments, features_dict)
        
        # Rank
        if top_k:
            ranked = self.ranker.top_k(scored, top_k)
        else:
            ranked = self.ranker.rank(scored)
        
        # Log decision
        if ranked:
            log_pipeline_decision(
                "scoring_complete",
                {
                    'segment_count': len(ranked),
                    'top_segment': ranked[0].segment_id,
                    'top_score': ranked[0].score.total_score if ranked[0].score else 0,
                    'score_range': (
                        ranked[-1].score.total_score if ranked[-1].score else 0,
                        ranked[0].score.total_score if ranked[0].score else 0
                    ),
                    'strategy': self.strategy.name,
                    'weights': self.weights.to_dict()
                },
                result_id=result_id
            )
        
        return ranked
    
    def get_ranking_stats(self, segments: List[Segment]) -> RankingResult:
        """Get detailed ranking statistics."""
        return self.ranker.get_ranking_result(segments)
    
    def explain_score(self, segment: Segment) -> Dict[str, Any]:
        """
        Explain how a segment's score was computed.
        
        Returns detailed breakdown for interpretability.
        """
        if not segment.score:
            return {'error': 'Segment has no score'}
        
        score = segment.score
        
        explanation = {
            'segment_id': segment.segment_id,
            'total_score': score.total_score,
            'strategy': score.scoring_method,
            'components': {
                'text': {
                    'score': score.text_score,
                    'weight': score.text_weight,
                    'contribution': score.text_score * score.text_weight
                },
                'audio': {
                    'score': score.audio_score,
                    'weight': score.audio_weight,
                    'contribution': score.audio_score * score.audio_weight
                },
                'visual': {
                    'score': score.visual_score,
                    'weight': score.visual_weight,
                    'contribution': score.visual_score * score.visual_weight
                }
            },
            'interpretation': self._interpret_score(score)
        }
        
        return explanation
    
    def _interpret_score(self, score: ScoreBreakdown) -> str:
        """Generate human-readable interpretation of score."""
        total = score.total_score
        
        if total >= 0.8:
            quality = "Excellent"
        elif total >= 0.6:
            quality = "Good"
        elif total >= 0.4:
            quality = "Average"
        elif total >= 0.2:
            quality = "Below average"
        else:
            quality = "Low"
        
        # Find dominant contributor
        contributions = [
            ('text', score.text_score * score.text_weight),
            ('audio', score.audio_score * score.audio_weight),
            ('visual', score.visual_score * score.visual_weight)
        ]
        dominant = max(contributions, key=lambda x: x[1])
        
        return f"{quality} engagement ({total:.2f}). Primary driver: {dominant[0]} ({dominant[1]:.2f})"
    
    def compare_strategies(
        self,
        segments: List[Segment],
        features_dict: Dict[str, SegmentFeatures],
        strategies: List[str] = None
    ) -> Dict[str, Dict]:
        """
        Compare rankings from different scoring strategies.
        
        Useful for ablation studies and strategy selection.
        """
        strategies = strategies or ['rule_based', 'normalized']
        
        results = {}
        rankings = {}
        
        for strategy_name in strategies:
            # Create scorer with this strategy
            scorer = EngagementScorer(
                strategy=strategy_name,
                ablation=self.ablation
            )
            
            # Score and rank
            # Clone segments to avoid modifying originals
            segment_copies = [
                Segment(
                    segment_id=s.segment_id,
                    start_seconds=s.start_seconds,
                    end_seconds=s.end_seconds,
                    segment_type=s.segment_type,
                    features=s.features
                )
                for s in segments
            ]
            
            ranked = scorer.score_and_rank(segment_copies, features_dict)
            rankings[strategy_name] = ranked
            
            # Get stats
            stats = scorer.get_ranking_stats(ranked)
            results[strategy_name] = {
                'top_segment': ranked[0].segment_id if ranked else None,
                'top_score': ranked[0].score.total_score if ranked and ranked[0].score else 0,
                'score_mean': stats.score_mean,
                'score_std': stats.score_std,
                'score_range': stats.score_range
            }
        
        # Compare rankings pairwise
        strategy_list = list(rankings.keys())
        for i, s1 in enumerate(strategy_list):
            for s2 in strategy_list[i+1:]:
                comparison = self.ranker.compare_rankings(rankings[s1], rankings[s2])
                results[f'{s1}_vs_{s2}'] = comparison
        
        return results


def create_scorer(
    strategy: str = 'rule_based',
    ablation_mode: str = 'full_multimodal'
) -> EngagementScorer:
    """
    Factory function to create an engagement scorer.
    
    Args:
        strategy: Scoring strategy name
        ablation_mode: Ablation mode
        
    Returns:
        Configured EngagementScorer
    """
    ablation_map = {
        'text_only': AblationConfig.text_only,
        'audio_only': AblationConfig.audio_only,
        'visual_only': AblationConfig.visual_only,
        'text_audio': AblationConfig.text_audio,
        'full_multimodal': AblationConfig.full_multimodal,
    }
    
    ablation = ablation_map.get(ablation_mode, AblationConfig.full_multimodal)()
    return EngagementScorer(strategy=strategy, ablation=ablation)

