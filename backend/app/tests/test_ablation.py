"""
Ablation & Modality Analysis Tests
==================================
Tests for the ablation study module.
"""

import os
import sys
import tempfile

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.ablation import (
    AblationRunner,
    AblationResult,
    ModalityAnalyzer,
    ComparisonMetrics,
    AblationReport,
    generate_ablation_report,
)
from app.ablation.analyzer import ScoreDistribution, ModalityContribution
from app.models import (
    AnalysisResult,
    Segment,
    VideoMetadata,
    ScoreBreakdown,
)
from app.config import AblationConfig


# =============================================================================
# TEST FIXTURES
# =============================================================================

def create_test_analysis_result(
    mode: str = "full_multimodal",
    num_segments: int = 5,
    base_score: float = 0.5
) -> AnalysisResult:
    """Create a test analysis result."""
    metadata = VideoMetadata(
        filename="test_video.mp4",
        filepath="/path/to/test_video.mp4",
        duration_seconds=300.0,
        fps=30.0,
        width=1920,
        height=1080
    )
    
    segments = []
    for i in range(num_segments):
        score = base_score + i * 0.1
        segment = Segment(
            segment_id=f"seg_{i:03d}",
            start_seconds=i * 60,
            end_seconds=(i + 1) * 60,
            segment_type="test",
            rank=num_segments - i,  # Higher score = lower rank
            score=ScoreBreakdown(
                total_score=score,
                text_score=score * 0.4,
                audio_score=score * 0.3,
                visual_score=score * 0.3
            )
        )
        segments.append(segment)
    
    # Sort by rank
    segments.sort(key=lambda s: s.rank)
    
    return AnalysisResult(
        result_id=f"result_{mode}",
        video_metadata=metadata,
        segments=segments,
        processing_time_seconds=10.0,
        ablation_mode=mode
    )


def create_test_ablation_results() -> dict:
    """Create a set of test ablation results."""
    results = {}
    
    modes = {
        'text_only': 0.4,
        'audio_only': 0.35,
        'visual_only': 0.3,
        'text_audio': 0.45,
        'full_multimodal': 0.5
    }
    
    for mode, base_score in modes.items():
        analysis = create_test_analysis_result(mode, 5, base_score)
        results[mode] = AblationResult(
            mode=mode,
            analysis_result=analysis,
            execution_time_seconds=10.0,
            success=True
        )
    
    return results


# =============================================================================
# ANALYZER TESTS
# =============================================================================

def test_analyzer_creation():
    """Test creating a modality analyzer."""
    analyzer = ModalityAnalyzer()
    assert analyzer is not None
    print("[PASS] Analyzer creation test passed")


def test_spearman_correlation():
    """Test Spearman correlation calculation."""
    analyzer = ModalityAnalyzer()
    
    # Perfect correlation
    result_a = create_test_analysis_result("mode_a")
    result_b = create_test_analysis_result("mode_b")
    
    metrics = analyzer.compare_modes(result_a, result_b)
    
    # Same rankings should give correlation of 1
    assert abs(metrics.rank_correlation - 1.0) < 0.01
    
    print(f"[PASS] Spearman correlation: {metrics.rank_correlation:.3f}")


def test_kendall_tau():
    """Test Kendall's tau calculation."""
    analyzer = ModalityAnalyzer()
    
    result_a = create_test_analysis_result("mode_a")
    result_b = create_test_analysis_result("mode_b")
    
    metrics = analyzer.compare_modes(result_a, result_b)
    
    # Same rankings should give tau of 1
    assert abs(metrics.kendall_tau - 1.0) < 0.01
    
    print(f"[PASS] Kendall's tau: {metrics.kendall_tau:.3f}")


def test_top_k_agreement():
    """Test top-K agreement calculation."""
    analyzer = ModalityAnalyzer()
    
    result_a = create_test_analysis_result("mode_a", 5, 0.5)
    result_b = create_test_analysis_result("mode_b", 5, 0.5)
    
    metrics = analyzer.compare_modes(result_a, result_b)
    
    # Same results should have full agreement
    assert metrics.top_1_agreement is True
    assert metrics.top_3_agreement == 3
    assert metrics.top_5_agreement == 5
    
    print(f"[PASS] Top-K agreement: top1={metrics.top_1_agreement}, top3={metrics.top_3_agreement}")


def test_score_distribution():
    """Test score distribution calculation."""
    analyzer = ModalityAnalyzer()
    
    result = create_test_analysis_result("test_mode", 5, 0.5)
    dist = analyzer.get_score_distribution(result)
    
    assert dist.segment_count == 5
    assert dist.min_score > 0
    assert dist.max_score <= 1
    assert dist.min_score <= dist.mean_score <= dist.max_score
    
    print(f"[PASS] Score distribution: mean={dist.mean_score:.3f}, range=[{dist.min_score:.3f}, {dist.max_score:.3f}]")


def test_compare_all_modes():
    """Test comparing all mode pairs."""
    analyzer = ModalityAnalyzer()
    
    results = create_test_ablation_results()
    analysis_results = {
        mode: r.analysis_result
        for mode, r in results.items()
        if r.success and r.analysis_result
    }
    
    comparisons = analyzer.compare_all_modes(analysis_results)
    
    # Should have n*(n-1)/2 comparisons for n modes
    n = len(analysis_results)
    expected_comparisons = n * (n - 1) // 2
    assert len(comparisons) == expected_comparisons
    
    print(f"[PASS] Compared {len(comparisons)} mode pairs")


def test_modality_contributions():
    """Test modality contribution analysis."""
    analyzer = ModalityAnalyzer()
    
    results = create_test_ablation_results()
    analysis_results = {
        mode: r.analysis_result
        for mode, r in results.items()
        if r.success and r.analysis_result
    }
    
    contributions = analyzer.analyze_contributions(analysis_results)
    
    assert 'text' in contributions
    assert 'audio' in contributions
    assert 'visual' in contributions
    
    for modality, contrib in contributions.items():
        assert isinstance(contrib, ModalityContribution)
        assert 0 <= contrib.contribution_score <= 1 or contrib.contribution_score >= -1
    
    print(f"[PASS] Analyzed contributions for {len(contributions)} modalities")


# =============================================================================
# RUNNER TESTS
# =============================================================================

def test_runner_creation():
    """Test creating an ablation runner."""
    runner = AblationRunner()
    assert runner is not None
    print("[PASS] Runner creation test passed")


def test_ablation_result_creation():
    """Test creating an ablation result."""
    analysis = create_test_analysis_result("test_mode")
    
    result = AblationResult(
        mode="test_mode",
        analysis_result=analysis,
        execution_time_seconds=5.0,
        success=True
    )
    
    assert result.mode == "test_mode"
    assert result.success is True
    assert result.segment_count == 5
    assert result.top_segment is not None
    
    print(f"[PASS] AblationResult: {result.segment_count} segments, top={result.top_segment.segment_id}")


def test_ablation_result_serialization():
    """Test ablation result serialization."""
    analysis = create_test_analysis_result("test_mode")
    
    result = AblationResult(
        mode="test_mode",
        analysis_result=analysis,
        execution_time_seconds=5.0,
        success=True
    )
    
    result_dict = result.to_dict()
    
    assert result_dict['mode'] == "test_mode"
    assert result_dict['success'] is True
    assert result_dict['analysis_result'] is not None
    
    print("[PASS] AblationResult serialization test passed")


# =============================================================================
# REPORT TESTS
# =============================================================================

def test_report_creation():
    """Test creating an ablation report."""
    report = AblationReport(
        video_filename="test.mp4",
        experiment_name="test_experiment"
    )
    
    assert report.video_filename == "test.mp4"
    assert report.experiment_name == "test_experiment"
    
    print("[PASS] Report creation test passed")


def test_generate_report():
    """Test generating a full report."""
    results = create_test_ablation_results()
    
    report = generate_ablation_report(
        results,
        video_filename="test_video.mp4",
        experiment_name="test_ablation"
    )
    
    assert report.video_filename == "test_video.mp4"
    assert len(report.mode_results) == 5
    assert len(report.pairwise_comparisons) > 0
    assert len(report.modality_contributions) == 3
    assert len(report.findings) > 0
    
    print(f"[PASS] Generated report with {len(report.findings)} findings")


def test_report_serialization():
    """Test saving and loading a report."""
    results = create_test_ablation_results()
    report = generate_ablation_report(results, "test.mp4", "test")
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        report.save(temp_path)
        
        loaded = AblationReport.load(temp_path)
        
        assert loaded.video_filename == report.video_filename
        assert len(loaded.mode_results) == len(report.mode_results)
        
        print("[PASS] Report save/load test passed")
    finally:
        os.unlink(temp_path)


def test_report_markdown():
    """Test generating Markdown report."""
    results = create_test_ablation_results()
    report = generate_ablation_report(results, "test.mp4", "test")
    
    markdown = report.to_markdown()
    
    assert "# Ablation Study Report" in markdown
    assert "## Summary" in markdown
    assert "## Mode Results" in markdown
    assert "## Key Findings" in markdown
    
    # Check tables are formatted
    assert "| Mode |" in markdown
    assert "| Modality |" in markdown
    
    print(f"[PASS] Generated {len(markdown)} chars of Markdown")


def test_report_summary():
    """Test report summary generation."""
    results = create_test_ablation_results()
    report = generate_ablation_report(results, "test.mp4", "test")
    
    assert 'modes_tested' in report.summary
    assert 'best_mode' in report.summary
    assert report.summary['modes_tested'] == 5
    
    print(f"[PASS] Summary: best mode = {report.summary.get('best_mode')}")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_full_analysis_workflow():
    """Test complete analysis workflow."""
    # Create results
    results = create_test_ablation_results()
    
    # Create analyzer
    analyzer = ModalityAnalyzer()
    
    # Get analysis results
    analysis_results = {
        mode: r.analysis_result
        for mode, r in results.items()
        if r.success and r.analysis_result
    }
    
    # Analyze
    contributions = analyzer.analyze_contributions(analysis_results)
    comparisons = analyzer.compare_all_modes(analysis_results)
    distributions = analyzer.get_all_distributions(analysis_results)
    
    # Generate report
    report = generate_ablation_report(results, "test.mp4", "integration_test")
    
    # Verify all pieces are present
    assert len(contributions) == 3
    assert len(comparisons) == 10  # 5 choose 2
    assert len(distributions) == 5
    assert report is not None
    
    print("[PASS] Full analysis workflow completed")


def test_findings_generation():
    """Test that findings are meaningful."""
    results = create_test_ablation_results()
    report = generate_ablation_report(results, "test.mp4", "test")
    
    # Should have at least some findings
    assert len(report.findings) >= 1
    
    # Findings should mention modalities
    finding_text = " ".join(report.findings)
    has_modality_mention = any(
        m in finding_text.lower()
        for m in ['text', 'audio', 'visual', 'multimodal']
    )
    assert has_modality_mention
    
    print(f"[PASS] Generated {len(report.findings)} meaningful findings")


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests():
    """Run all ablation tests."""
    print("\n" + "="*60)
    print("ABLATION & MODALITY ANALYSIS TESTS")
    print("="*60 + "\n")
    
    # Analyzer tests
    print("\n--- Analyzer Tests ---")
    test_analyzer_creation()
    test_spearman_correlation()
    test_kendall_tau()
    test_top_k_agreement()
    test_score_distribution()
    test_compare_all_modes()
    test_modality_contributions()
    
    # Runner tests
    print("\n--- Runner Tests ---")
    test_runner_creation()
    test_ablation_result_creation()
    test_ablation_result_serialization()
    
    # Report tests
    print("\n--- Report Tests ---")
    test_report_creation()
    test_generate_report()
    test_report_serialization()
    test_report_markdown()
    test_report_summary()
    
    # Integration tests
    print("\n--- Integration Tests ---")
    test_full_analysis_workflow()
    test_findings_generation()
    
    print("\n" + "="*60)
    print("ALL ABLATION TESTS PASSED")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()

