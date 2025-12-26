"""
Modality Analysis Module
========================
Statistical analysis of modality contributions.

Implements:
- Rank correlation (Spearman's rho)
- Score distribution comparison
- Segment agreement analysis
- Modality importance metrics
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import math

from ..models import Segment, AnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class ComparisonMetrics:
    """
    Metrics comparing two ablation modes.
    
    Attributes:
        mode_a: First ablation mode name
        mode_b: Second ablation mode name
        rank_correlation: Spearman's rho (-1 to 1)
        top_1_agreement: Whether top segment is the same
        top_3_agreement: Number of common segments in top 3
        top_5_agreement: Number of common segments in top 5
        score_correlation: Pearson correlation of scores
        mean_rank_difference: Average rank difference for common segments
        kendall_tau: Kendall's tau rank correlation
    """
    mode_a: str
    mode_b: str
    rank_correlation: float = 0.0
    top_1_agreement: bool = False
    top_3_agreement: int = 0
    top_5_agreement: int = 0
    score_correlation: float = 0.0
    mean_rank_difference: float = 0.0
    kendall_tau: float = 0.0
    common_segments: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'mode_a': self.mode_a,
            'mode_b': self.mode_b,
            'rank_correlation': self.rank_correlation,
            'top_1_agreement': self.top_1_agreement,
            'top_3_agreement': self.top_3_agreement,
            'top_5_agreement': self.top_5_agreement,
            'score_correlation': self.score_correlation,
            'mean_rank_difference': self.mean_rank_difference,
            'kendall_tau': self.kendall_tau,
            'common_segments': self.common_segments
        }


@dataclass
class ScoreDistribution:
    """Statistics about score distribution for a mode."""
    mode: str
    min_score: float = 0.0
    max_score: float = 0.0
    mean_score: float = 0.0
    median_score: float = 0.0
    std_score: float = 0.0
    segment_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'mode': self.mode,
            'min': self.min_score,
            'max': self.max_score,
            'mean': self.mean_score,
            'median': self.median_score,
            'std': self.std_score,
            'count': self.segment_count
        }


@dataclass
class ModalityContribution:
    """
    Measures a single modality's contribution.
    
    Calculated by comparing full multimodal to ablated versions.
    """
    modality: str
    contribution_score: float = 0.0  # How much this modality contributes
    rank_change_impact: float = 0.0  # How much rankings change without it
    top_segment_impact: float = 0.0  # Impact on top segment selection
    unique_value: float = 0.0  # Information not captured by others
    
    def to_dict(self) -> Dict:
        return {
            'modality': self.modality,
            'contribution_score': self.contribution_score,
            'rank_change_impact': self.rank_change_impact,
            'top_segment_impact': self.top_segment_impact,
            'unique_value': self.unique_value
        }


class ModalityAnalyzer:
    """
    Analyzes contributions of different modalities.
    
    Compares rankings and scores across ablation modes to
    quantify each modality's importance.
    
    Usage:
        analyzer = ModalityAnalyzer()
        
        # Compare two modes
        metrics = analyzer.compare_modes(results['text_only'], results['audio_only'])
        
        # Analyze full results
        contributions = analyzer.analyze_contributions(results)
    """
    
    def __init__(self):
        pass
    
    def compare_modes(
        self,
        result_a: AnalysisResult,
        result_b: AnalysisResult
    ) -> ComparisonMetrics:
        """
        Compare rankings from two ablation modes.
        
        Args:
            result_a: First ablation result
            result_b: Second ablation result
            
        Returns:
            ComparisonMetrics with all comparison statistics
        """
        mode_a = result_a.ablation_mode
        mode_b = result_b.ablation_mode
        
        # Get segment rankings
        segments_a = {s.segment_id: s for s in result_a.segments}
        segments_b = {s.segment_id: s for s in result_b.segments}
        
        # Find common segments
        common_ids = set(segments_a.keys()) & set(segments_b.keys())
        
        if not common_ids:
            return ComparisonMetrics(
                mode_a=mode_a,
                mode_b=mode_b,
                common_segments=0
            )
        
        # Extract ranks and scores for common segments
        ranks_a = {sid: segments_a[sid].rank for sid in common_ids}
        ranks_b = {sid: segments_b[sid].rank for sid in common_ids}
        
        scores_a = {
            sid: segments_a[sid].score.total_score 
            for sid in common_ids 
            if segments_a[sid].score
        }
        scores_b = {
            sid: segments_b[sid].score.total_score 
            for sid in common_ids 
            if segments_b[sid].score
        }
        
        # Calculate Spearman's rho
        rank_correlation = self._spearman_correlation(ranks_a, ranks_b)
        
        # Calculate Kendall's tau
        kendall_tau = self._kendall_tau(ranks_a, ranks_b)
        
        # Calculate score correlation
        score_correlation = self._pearson_correlation(scores_a, scores_b)
        
        # Top-K agreement
        top_a = sorted(result_a.segments, key=lambda s: s.rank)
        top_b = sorted(result_b.segments, key=lambda s: s.rank)
        
        top_1_agreement = (top_a[0].segment_id == top_b[0].segment_id) if top_a and top_b else False
        
        top_3_ids_a = {s.segment_id for s in top_a[:3]}
        top_3_ids_b = {s.segment_id for s in top_b[:3]}
        top_3_agreement = len(top_3_ids_a & top_3_ids_b)
        
        top_5_ids_a = {s.segment_id for s in top_a[:5]}
        top_5_ids_b = {s.segment_id for s in top_b[:5]}
        top_5_agreement = len(top_5_ids_a & top_5_ids_b)
        
        # Mean rank difference
        rank_diffs = [abs(ranks_a[sid] - ranks_b[sid]) for sid in common_ids]
        mean_rank_diff = sum(rank_diffs) / len(rank_diffs) if rank_diffs else 0.0
        
        return ComparisonMetrics(
            mode_a=mode_a,
            mode_b=mode_b,
            rank_correlation=rank_correlation,
            top_1_agreement=top_1_agreement,
            top_3_agreement=top_3_agreement,
            top_5_agreement=top_5_agreement,
            score_correlation=score_correlation,
            mean_rank_difference=mean_rank_diff,
            kendall_tau=kendall_tau,
            common_segments=len(common_ids)
        )
    
    def _spearman_correlation(
        self,
        ranks_a: Dict[str, int],
        ranks_b: Dict[str, int]
    ) -> float:
        """Calculate Spearman's rank correlation coefficient."""
        if len(ranks_a) < 2:
            return 0.0
        
        common = set(ranks_a.keys()) & set(ranks_b.keys())
        n = len(common)
        
        if n < 2:
            return 0.0
        
        # Calculate sum of squared rank differences
        d_squared_sum = sum(
            (ranks_a[sid] - ranks_b[sid]) ** 2
            for sid in common
        )
        
        # Spearman's rho formula
        rho = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
        
        return max(-1.0, min(1.0, rho))
    
    def _kendall_tau(
        self,
        ranks_a: Dict[str, int],
        ranks_b: Dict[str, int]
    ) -> float:
        """Calculate Kendall's tau rank correlation."""
        common = list(set(ranks_a.keys()) & set(ranks_b.keys()))
        n = len(common)
        
        if n < 2:
            return 0.0
        
        concordant = 0
        discordant = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                id_i, id_j = common[i], common[j]
                
                sign_a = ranks_a[id_i] - ranks_a[id_j]
                sign_b = ranks_b[id_i] - ranks_b[id_j]
                
                if sign_a * sign_b > 0:
                    concordant += 1
                elif sign_a * sign_b < 0:
                    discordant += 1
        
        total_pairs = n * (n - 1) / 2
        
        if total_pairs == 0:
            return 0.0
        
        tau = (concordant - discordant) / total_pairs
        return tau
    
    def _pearson_correlation(
        self,
        values_a: Dict[str, float],
        values_b: Dict[str, float]
    ) -> float:
        """Calculate Pearson correlation coefficient."""
        common = set(values_a.keys()) & set(values_b.keys())
        n = len(common)
        
        if n < 2:
            return 0.0
        
        # Extract values
        x = [values_a[sid] for sid in common]
        y = [values_b[sid] for sid in common]
        
        # Means
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        # Covariance and standard deviations
        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
        std_x = math.sqrt(sum((xi - mean_x)**2 for xi in x) / n)
        std_y = math.sqrt(sum((yi - mean_y)**2 for yi in y) / n)
        
        if std_x == 0 or std_y == 0:
            return 0.0
        
        return cov / (std_x * std_y)
    
    def get_score_distribution(self, result: AnalysisResult) -> ScoreDistribution:
        """Calculate score distribution statistics for a mode."""
        scores = [
            s.score.total_score 
            for s in result.segments 
            if s.score
        ]
        
        if not scores:
            return ScoreDistribution(mode=result.ablation_mode)
        
        scores_sorted = sorted(scores)
        n = len(scores)
        
        return ScoreDistribution(
            mode=result.ablation_mode,
            min_score=min(scores),
            max_score=max(scores),
            mean_score=sum(scores) / n,
            median_score=scores_sorted[n // 2],
            std_score=math.sqrt(sum((s - sum(scores)/n)**2 for s in scores) / n),
            segment_count=n
        )
    
    def analyze_contributions(
        self,
        results: Dict[str, AnalysisResult]
    ) -> Dict[str, ModalityContribution]:
        """
        Analyze contribution of each modality.
        
        Compares full multimodal to single-modality versions.
        
        Args:
            results: Dict mapping mode name to AnalysisResult
            
        Returns:
            Dict mapping modality name to ModalityContribution
        """
        contributions = {}
        
        full = results.get('full_multimodal')
        if not full:
            logger.warning("No full_multimodal result for contribution analysis")
            return contributions
        
        # Map ablation modes to modalities
        modality_modes = {
            'text': 'text_only',
            'audio': 'audio_only',
            'visual': 'visual_only'
        }
        
        for modality, mode_name in modality_modes.items():
            single_result = results.get(mode_name)
            
            if single_result:
                contribution = self._calculate_contribution(
                    modality, full, single_result, results
                )
                contributions[modality] = contribution
            else:
                contributions[modality] = ModalityContribution(modality=modality)
        
        return contributions
    
    def _calculate_contribution(
        self,
        modality: str,
        full_result: AnalysisResult,
        single_result: AnalysisResult,
        all_results: Dict[str, AnalysisResult]
    ) -> ModalityContribution:
        """Calculate contribution metrics for a single modality."""
        
        # Compare single modality to full
        single_vs_full = self.compare_modes(single_result, full_result)
        
        # Contribution score: How well does this modality alone predict full ranking?
        contribution_score = single_vs_full.rank_correlation
        
        # Rank change impact: How different is single-modality ranking from full?
        rank_change_impact = single_vs_full.mean_rank_difference
        
        # Top segment impact: Does this modality alone get the top segment right?
        top_segment_impact = 1.0 if single_vs_full.top_1_agreement else 0.0
        
        # Unique value: Compare to other single modalities
        # Higher unique value means this modality provides information others don't
        unique_scores = []
        other_modes = {
            'text': ['audio_only', 'visual_only'],
            'audio': ['text_only', 'visual_only'],
            'visual': ['text_only', 'audio_only']
        }
        
        for other_mode in other_modes.get(modality, []):
            other_result = all_results.get(other_mode)
            if other_result:
                cross_compare = self.compare_modes(single_result, other_result)
                # Low correlation with others = high unique value
                unique_scores.append(1.0 - abs(cross_compare.rank_correlation))
        
        unique_value = sum(unique_scores) / len(unique_scores) if unique_scores else 0.5
        
        return ModalityContribution(
            modality=modality,
            contribution_score=contribution_score,
            rank_change_impact=rank_change_impact,
            top_segment_impact=top_segment_impact,
            unique_value=unique_value
        )
    
    def compare_all_modes(
        self,
        results: Dict[str, AnalysisResult]
    ) -> Dict[str, ComparisonMetrics]:
        """
        Compare all pairs of ablation modes.
        
        Returns dict with keys like "text_only_vs_audio_only".
        """
        comparisons = {}
        modes = list(results.keys())
        
        for i, mode_a in enumerate(modes):
            for mode_b in modes[i+1:]:
                key = f"{mode_a}_vs_{mode_b}"
                comparisons[key] = self.compare_modes(
                    results[mode_a],
                    results[mode_b]
                )
        
        return comparisons
    
    def get_all_distributions(
        self,
        results: Dict[str, AnalysisResult]
    ) -> Dict[str, ScoreDistribution]:
        """Get score distributions for all modes."""
        return {
            mode: self.get_score_distribution(result)
            for mode, result in results.items()
        }

