"""
Segment Ranking Module
======================
Ranks segments based on their engagement scores.

Features:
- Score-based ranking
- Tie-breaking strategies
- Top-K selection
- Rank statistics
"""

import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

from ..models import Segment, ScoreBreakdown

logger = logging.getLogger(__name__)


@dataclass
class RankingResult:
    """Result of ranking operation."""
    ranked_segments: List[Segment]
    total_count: int
    score_range: Tuple[float, float]  # (min, max)
    score_mean: float
    score_std: float


class SegmentRanker:
    """
    Ranks segments based on engagement scores.
    
    Features:
    - Descending score ranking (highest = best)
    - Multiple tie-breaking strategies
    - Top-K filtering
    - Score threshold filtering
    
    Usage:
        ranker = SegmentRanker()
        ranked = ranker.rank(segments)
        top_5 = ranker.top_k(segments, k=5)
    """
    
    def __init__(
        self,
        tie_breaker: str = 'duration',
        ascending: bool = False
    ):
        """
        Initialize the ranker.
        
        Args:
            tie_breaker: How to break ties ('duration', 'position', 'random')
            ascending: If True, lower scores rank higher
        """
        self.tie_breaker = tie_breaker
        self.ascending = ascending
    
    def rank(
        self,
        segments: List[Segment],
        update_rank_field: bool = True
    ) -> List[Segment]:
        """
        Rank segments by their engagement scores.
        
        Args:
            segments: List of segments with scores
            update_rank_field: Whether to update segment.rank
            
        Returns:
            Segments sorted by rank (best first)
        """
        if not segments:
            return []
        
        # Sort by score (and tie-breaker)
        sorted_segments = sorted(
            segments,
            key=lambda s: self._sort_key(s),
            reverse=not self.ascending
        )
        
        # Update rank field
        if update_rank_field:
            for rank, segment in enumerate(sorted_segments, 1):
                segment.rank = rank
        
        return sorted_segments
    
    def _sort_key(self, segment: Segment) -> tuple:
        """
        Generate sort key for a segment.
        
        Returns tuple of (primary_score, tie_breaker_value).
        """
        primary = segment.score.total_score if segment.score else 0.0
        
        if self.tie_breaker == 'duration':
            # Prefer longer segments on tie
            secondary = segment.duration_seconds
        elif self.tie_breaker == 'position':
            # Prefer earlier segments on tie
            secondary = -segment.start_seconds
        else:
            # No secondary sort
            secondary = 0
        
        return (primary, secondary)
    
    def top_k(
        self,
        segments: List[Segment],
        k: int,
        min_score: Optional[float] = None
    ) -> List[Segment]:
        """
        Get top K segments by score.
        
        Args:
            segments: List of segments
            k: Number of top segments to return
            min_score: Optional minimum score threshold
            
        Returns:
            Top K segments, ranked
        """
        ranked = self.rank(segments, update_rank_field=False)
        
        # Apply minimum score filter
        if min_score is not None:
            ranked = [
                s for s in ranked
                if s.score and s.score.total_score >= min_score
            ]
        
        # Take top K
        top = ranked[:k]
        
        # Update ranks for selected segments
        for rank, segment in enumerate(top, 1):
            segment.rank = rank
        
        return top
    
    def filter_by_score(
        self,
        segments: List[Segment],
        min_score: float = 0.0,
        max_score: float = 1.0
    ) -> List[Segment]:
        """
        Filter segments by score range.
        
        Args:
            segments: List of segments
            min_score: Minimum score (inclusive)
            max_score: Maximum score (inclusive)
            
        Returns:
            Filtered and ranked segments
        """
        filtered = [
            s for s in segments
            if s.score and min_score <= s.score.total_score <= max_score
        ]
        return self.rank(filtered)
    
    def get_ranking_result(self, segments: List[Segment]) -> RankingResult:
        """
        Get detailed ranking statistics.
        
        Args:
            segments: List of segments to rank
            
        Returns:
            RankingResult with statistics
        """
        ranked = self.rank(segments)
        
        if not ranked:
            return RankingResult(
                ranked_segments=[],
                total_count=0,
                score_range=(0.0, 0.0),
                score_mean=0.0,
                score_std=0.0
            )
        
        # Extract scores
        scores = [s.score.total_score for s in ranked if s.score]
        
        if not scores:
            return RankingResult(
                ranked_segments=ranked,
                total_count=len(ranked),
                score_range=(0.0, 0.0),
                score_mean=0.0,
                score_std=0.0
            )
        
        # Calculate statistics
        score_min = min(scores)
        score_max = max(scores)
        score_mean = sum(scores) / len(scores)
        
        # Calculate std
        variance = sum((s - score_mean) ** 2 for s in scores) / len(scores)
        score_std = variance ** 0.5
        
        return RankingResult(
            ranked_segments=ranked,
            total_count=len(ranked),
            score_range=(score_min, score_max),
            score_mean=score_mean,
            score_std=score_std
        )
    
    def compare_rankings(
        self,
        ranking1: List[Segment],
        ranking2: List[Segment]
    ) -> dict:
        """
        Compare two rankings.
        
        Useful for comparing different scoring methods.
        
        Returns dict with comparison metrics.
        """
        # Get segment IDs in order
        ids1 = [s.segment_id for s in ranking1]
        ids2 = [s.segment_id for s in ranking2]
        
        # Find common segments
        common = set(ids1) & set(ids2)
        
        if not common:
            return {
                'common_segments': 0,
                'rank_correlation': None,
                'top_1_agreement': False,
                'top_3_agreement': 0
            }
        
        # Calculate rank correlation (Spearman's rho approximation)
        rank_diffs = []
        for seg_id in common:
            rank1 = ids1.index(seg_id) + 1
            rank2 = ids2.index(seg_id) + 1
            rank_diffs.append((rank1 - rank2) ** 2)
        
        n = len(rank_diffs)
        d_squared_sum = sum(rank_diffs)
        rho = 1 - (6 * d_squared_sum) / (n * (n**2 - 1)) if n > 1 else 1.0
        
        # Top-K agreement
        top1_agree = ids1[0] == ids2[0] if ids1 and ids2 else False
        top3_agree = len(set(ids1[:3]) & set(ids2[:3]))
        
        return {
            'common_segments': len(common),
            'rank_correlation': rho,
            'top_1_agreement': top1_agree,
            'top_3_agreement': top3_agree
        }


def rank_segments(
    segments: List[Segment],
    top_k: Optional[int] = None
) -> List[Segment]:
    """
    Convenience function to rank segments.
    
    Args:
        segments: List of segments with scores
        top_k: Optional limit on results
        
    Returns:
        Ranked segments
    """
    ranker = SegmentRanker()
    
    if top_k:
        return ranker.top_k(segments, top_k)
    return ranker.rank(segments)

