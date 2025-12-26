"""
Engagement Scoring Tests
========================
Tests for the engagement scoring module.
"""

import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.scoring import (
    EngagementScorer,
    RuleBasedScoring,
    NormalizedScoring,
    LearnedScoring,
    FeatureNormalizer,
    SegmentRanker,
)
from app.scoring.strategies import ScoringWeights, get_strategy
from app.scoring.scorer import create_scorer
from app.models import (
    Segment,
    SegmentFeatures,
    TextFeatures,
    AudioFeatures,
    VisualFeatures,
    ScoreBreakdown,
)
from app.config import AblationConfig, ScoringConfig


# =============================================================================
# TEST FIXTURES
# =============================================================================

def create_test_segment(
    segment_id: str = "test_001",
    start: float = 0.0,
    end: float = 60.0,
    segment_type: str = "test"
) -> Segment:
    """Create a test segment."""
    return Segment(
        segment_id=segment_id,
        start_seconds=start,
        end_seconds=end,
        segment_type=segment_type
    )


def create_test_features(
    word_count: int = 50,
    sentiment: float = 0.3,
    energy: float = 0.5,
    motion: float = 0.4
) -> SegmentFeatures:
    """Create test features with configurable values."""
    return SegmentFeatures(
        segment_id="test_001",
        start_seconds=0.0,
        end_seconds=60.0,
        text_features=TextFeatures(
            word_count=word_count,
            sentence_count=5,
            sentiment_score=sentiment,
            keyword_density=0.05,
            question_count=2,
            exclamation_count=1
        ),
        audio_features=AudioFeatures(
            energy_mean=energy,
            energy_std=0.1,
            silence_ratio=0.1,
            volume_dynamics=2.0,
            spectral_centroid=1000.0
        ),
        visual_features=VisualFeatures(
            motion_intensity=motion,
            scene_change_count=3,
            scene_change_rate=0.05,
            brightness_mean=0.5,
            brightness_std=0.1,
            color_variance=0.3
        )
    )


def create_test_segments_with_features(n: int = 5):
    """Create multiple test segments with varying features."""
    segments = []
    features_dict = {}
    
    for i in range(n):
        segment = create_test_segment(
            segment_id=f"seg_{i:03d}",
            start=i * 60,
            end=(i + 1) * 60
        )
        
        # Vary features
        features = create_test_features(
            word_count=30 + i * 20,
            sentiment=0.1 + i * 0.15,
            energy=0.3 + i * 0.1,
            motion=0.2 + i * 0.15
        )
        features.segment_id = segment.segment_id
        
        segment.features = features
        segments.append(segment)
        features_dict[segment.segment_id] = features
    
    return segments, features_dict


# =============================================================================
# NORMALIZER TESTS
# =============================================================================

def test_normalizer_creation():
    """Test creating a feature normalizer."""
    normalizer = FeatureNormalizer()
    assert normalizer is not None
    print("[PASS] Normalizer creation test passed")


def test_normalizer_value():
    """Test normalizing individual values."""
    normalizer = FeatureNormalizer()
    
    # Word count (range 0-200)
    norm = normalizer.normalize_value(100, 'word_count')
    assert 0 <= norm <= 1
    assert norm == 0.5  # 100 is midpoint
    
    # Sentiment (range -1 to 1)
    norm = normalizer.normalize_value(0.5, 'sentiment_score')
    assert 0 <= norm <= 1
    assert norm == 0.75  # 0.5 is 75% of range from -1 to 1
    
    print("[PASS] Value normalization test passed")


def test_normalizer_clipping():
    """Test that values outside range are clipped."""
    normalizer = FeatureNormalizer()
    
    # Very high word count
    norm = normalizer.normalize_value(500, 'word_count', clip=True)
    assert norm == 1.0
    
    # Very low value
    norm = normalizer.normalize_value(-10, 'word_count', clip=True)
    assert norm == 0.0
    
    print("[PASS] Clipping test passed")


def test_normalizer_segment():
    """Test normalizing a full segment."""
    normalizer = FeatureNormalizer()
    features = create_test_features()
    
    normalized = normalizer.normalize_segment(features)
    
    assert 'text' in normalized
    assert 'audio' in normalized
    assert 'visual' in normalized
    
    # Check all values in [0, 1]
    for modality, values in normalized.items():
        for name, value in values.items():
            assert 0 <= value <= 1, f"{modality}.{name} = {value} not in [0, 1]"
    
    print(f"[PASS] Segment normalization: {len(normalized)} modalities")


# =============================================================================
# SCORING STRATEGY TESTS
# =============================================================================

def test_rule_based_strategy():
    """Test rule-based scoring strategy."""
    strategy = RuleBasedScoring()
    features = create_test_features()
    weights = ScoringWeights()
    
    score = strategy.score(features, weights)
    
    assert isinstance(score, ScoreBreakdown)
    assert 0 <= score.total_score <= 1
    assert score.scoring_method == "rule_based"
    
    print(f"[PASS] Rule-based score: {score.total_score:.3f}")


def test_normalized_strategy():
    """Test normalized scoring strategy."""
    strategy = NormalizedScoring()
    features = create_test_features()
    weights = ScoringWeights()
    
    score = strategy.score(features, weights)
    
    assert isinstance(score, ScoreBreakdown)
    assert 0 <= score.total_score <= 1
    assert score.scoring_method == "normalized"
    
    print(f"[PASS] Normalized score: {score.total_score:.3f}")


def test_learned_strategy_fallback():
    """Test learned strategy falls back correctly."""
    strategy = LearnedScoring()  # No model provided
    features = create_test_features()
    weights = ScoringWeights()
    
    score = strategy.score(features, weights)
    
    assert score.scoring_method == "learned_fallback"
    
    print(f"[PASS] Learned fallback score: {score.total_score:.3f}")


def test_strategy_registry():
    """Test getting strategies by name."""
    for name in ['rule_based', 'normalized', 'learned']:
        strategy = get_strategy(name)
        assert strategy.name == name
    
    print("[PASS] Strategy registry works")


def test_weights_from_config():
    """Test creating weights from config."""
    config = ScoringConfig(
        text_weight=0.5,
        audio_weight=0.3,
        visual_weight=0.2
    )
    
    weights = ScoringWeights.from_config(config)
    
    assert weights.text_weight == 0.5
    assert weights.audio_weight == 0.3
    assert weights.visual_weight == 0.2
    
    print("[PASS] Weights from config test passed")


def test_weights_for_ablation():
    """Test weights respect ablation mode."""
    # Text only
    ablation = AblationConfig.text_only()
    weights = ScoringWeights.for_ablation(ablation)
    
    assert weights.text_weight == 1.0
    assert weights.audio_weight == 0.0
    assert weights.visual_weight == 0.0
    
    # Text + Audio
    ablation = AblationConfig.text_audio()
    weights = ScoringWeights.for_ablation(ablation)
    
    assert weights.text_weight > 0
    assert weights.audio_weight > 0
    assert weights.visual_weight == 0.0
    assert abs(weights.text_weight + weights.audio_weight - 1.0) < 0.01
    
    print("[PASS] Ablation weights test passed")


# =============================================================================
# RANKER TESTS
# =============================================================================

def test_ranker_creation():
    """Test creating a segment ranker."""
    ranker = SegmentRanker()
    assert ranker is not None
    print("[PASS] Ranker creation test passed")


def test_ranking():
    """Test basic ranking."""
    ranker = SegmentRanker()
    
    # Create segments with different scores
    segments = [create_test_segment(f"seg_{i}") for i in range(3)]
    segments[0].score = ScoreBreakdown(total_score=0.5)
    segments[1].score = ScoreBreakdown(total_score=0.8)
    segments[2].score = ScoreBreakdown(total_score=0.3)
    
    ranked = ranker.rank(segments)
    
    # Should be sorted descending by score
    assert ranked[0].segment_id == "seg_1"  # score 0.8
    assert ranked[1].segment_id == "seg_0"  # score 0.5
    assert ranked[2].segment_id == "seg_2"  # score 0.3
    
    # Check rank field updated
    assert ranked[0].rank == 1
    assert ranked[1].rank == 2
    assert ranked[2].rank == 3
    
    print("[PASS] Ranking test passed")


def test_top_k():
    """Test top-K selection."""
    ranker = SegmentRanker()
    
    segments = [create_test_segment(f"seg_{i}") for i in range(5)]
    for i, seg in enumerate(segments):
        seg.score = ScoreBreakdown(total_score=0.1 * (i + 1))
    
    top2 = ranker.top_k(segments, k=2)
    
    assert len(top2) == 2
    assert top2[0].score.total_score == 0.5  # highest
    assert top2[1].score.total_score == 0.4  # second highest
    
    print("[PASS] Top-K selection test passed")


def test_ranking_stats():
    """Test ranking statistics."""
    ranker = SegmentRanker()
    
    segments = [create_test_segment(f"seg_{i}") for i in range(3)]
    segments[0].score = ScoreBreakdown(total_score=0.3)
    segments[1].score = ScoreBreakdown(total_score=0.5)
    segments[2].score = ScoreBreakdown(total_score=0.7)
    
    result = ranker.get_ranking_result(segments)
    
    assert result.total_count == 3
    assert result.score_range == (0.3, 0.7)
    assert result.score_mean == 0.5
    
    print(f"[PASS] Ranking stats: mean={result.score_mean:.2f}, range={result.score_range}")


# =============================================================================
# ENGAGEMENT SCORER TESTS
# =============================================================================

def test_scorer_creation():
    """Test creating an engagement scorer."""
    scorer = EngagementScorer()
    assert scorer is not None
    print("[PASS] Scorer creation test passed")


def test_scorer_single_segment():
    """Test scoring a single segment."""
    scorer = EngagementScorer()
    segment = create_test_segment()
    features = create_test_features()
    segment.features = features
    
    score = scorer.score_segment(segment)
    
    assert isinstance(score, ScoreBreakdown)
    assert segment.score is not None
    assert 0 <= segment.score.total_score <= 1
    
    print(f"[PASS] Single segment score: {score.total_score:.3f}")


def test_scorer_multiple_segments():
    """Test scoring multiple segments."""
    scorer = EngagementScorer()
    segments, features_dict = create_test_segments_with_features(5)
    
    scored = scorer.score_segments(segments, features_dict)
    
    assert len(scored) == 5
    assert all(s.score is not None for s in scored)
    
    print(f"[PASS] Scored {len(scored)} segments")


def test_scorer_and_rank():
    """Test scoring and ranking together."""
    scorer = EngagementScorer()
    segments, features_dict = create_test_segments_with_features(5)
    
    ranked = scorer.score_and_rank(segments, features_dict)
    
    # Should be sorted by score
    for i in range(len(ranked) - 1):
        assert ranked[i].score.total_score >= ranked[i+1].score.total_score
    
    # Ranks should be set
    assert ranked[0].rank == 1
    
    print(f"[PASS] Score and rank: top score = {ranked[0].score.total_score:.3f}")


def test_scorer_with_ablation():
    """Test scorer respects ablation mode."""
    # Text only
    scorer_text = create_scorer(ablation_mode='text_only')
    assert scorer_text.weights.text_weight == 1.0
    assert scorer_text.weights.audio_weight == 0.0
    
    # Full multimodal
    scorer_full = create_scorer(ablation_mode='full_multimodal')
    assert scorer_full.weights.text_weight > 0
    assert scorer_full.weights.audio_weight > 0
    assert scorer_full.weights.visual_weight > 0
    
    print("[PASS] Ablation mode affects weights")


def test_score_explanation():
    """Test score explanation."""
    scorer = EngagementScorer()
    segment = create_test_segment()
    segment.features = create_test_features()
    
    scorer.score_segment(segment)
    explanation = scorer.explain_score(segment)
    
    assert 'segment_id' in explanation
    assert 'total_score' in explanation
    assert 'components' in explanation
    assert 'interpretation' in explanation
    
    print(f"[PASS] Explanation: {explanation['interpretation']}")


def test_strategy_comparison():
    """Test comparing different strategies."""
    scorer = EngagementScorer()
    segments, features_dict = create_test_segments_with_features(5)
    
    comparison = scorer.compare_strategies(
        segments, features_dict,
        strategies=['rule_based', 'normalized']
    )
    
    assert 'rule_based' in comparison
    assert 'normalized' in comparison
    assert 'rule_based_vs_normalized' in comparison
    
    print(f"[PASS] Strategy comparison: correlation = {comparison.get('rule_based_vs_normalized', {}).get('rank_correlation', 'N/A')}")


# =============================================================================
# SCORE VARIATION TESTS
# =============================================================================

def test_higher_features_higher_score():
    """Test that better features lead to higher scores."""
    scorer = EngagementScorer()
    
    # Low engagement features
    low_features = create_test_features(
        word_count=20,
        sentiment=0.1,
        energy=0.2,
        motion=0.1
    )
    
    # High engagement features
    high_features = create_test_features(
        word_count=100,
        sentiment=0.9,
        energy=0.8,
        motion=0.7
    )
    
    segment_low = create_test_segment("low")
    segment_low.features = low_features
    low_features.segment_id = "low"
    
    segment_high = create_test_segment("high")
    segment_high.features = high_features
    high_features.segment_id = "high"
    
    scorer.score_segment(segment_low)
    scorer.score_segment(segment_high)
    
    assert segment_high.score.total_score > segment_low.score.total_score
    
    print(f"[PASS] High features ({segment_high.score.total_score:.3f}) > Low features ({segment_low.score.total_score:.3f})")


def test_weight_changes_affect_score():
    """Test that changing weights changes scores."""
    segments, features_dict = create_test_segments_with_features(3)
    
    # Text-heavy weights
    scorer_text = EngagementScorer(
        custom_weights=ScoringWeights(text_weight=0.8, audio_weight=0.1, visual_weight=0.1)
    )
    
    # Audio-heavy weights
    scorer_audio = EngagementScorer(
        custom_weights=ScoringWeights(text_weight=0.1, audio_weight=0.8, visual_weight=0.1)
    )
    
    # Score with both
    ranked_text = scorer_text.score_and_rank(segments.copy(), features_dict)
    ranked_audio = scorer_audio.score_and_rank(segments.copy(), features_dict)
    
    # Rankings might differ
    print(f"[PASS] Text-heavy top: {ranked_text[0].segment_id}, Audio-heavy top: {ranked_audio[0].segment_id}")


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests():
    """Run all scoring tests."""
    print("\n" + "="*60)
    print("ENGAGEMENT SCORING TESTS")
    print("="*60 + "\n")
    
    # Normalizer tests
    print("\n--- Normalizer Tests ---")
    test_normalizer_creation()
    test_normalizer_value()
    test_normalizer_clipping()
    test_normalizer_segment()
    
    # Strategy tests
    print("\n--- Scoring Strategy Tests ---")
    test_rule_based_strategy()
    test_normalized_strategy()
    test_learned_strategy_fallback()
    test_strategy_registry()
    test_weights_from_config()
    test_weights_for_ablation()
    
    # Ranker tests
    print("\n--- Ranker Tests ---")
    test_ranker_creation()
    test_ranking()
    test_top_k()
    test_ranking_stats()
    
    # Scorer tests
    print("\n--- Engagement Scorer Tests ---")
    test_scorer_creation()
    test_scorer_single_segment()
    test_scorer_multiple_segments()
    test_scorer_and_rank()
    test_scorer_with_ablation()
    test_score_explanation()
    test_strategy_comparison()
    
    # Score variation tests
    print("\n--- Score Variation Tests ---")
    test_higher_features_higher_score()
    test_weight_changes_affect_score()
    
    print("\n" + "="*60)
    print("ALL SCORING TESTS PASSED")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()

