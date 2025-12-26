"""
Ablation Report Module
======================
Generates research reports from ablation study results.

Report formats:
- JSON: Machine-readable structured data
- Markdown: Human-readable research summary
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .runner import AblationResult
from .analyzer import ModalityAnalyzer, ComparisonMetrics, ScoreDistribution, ModalityContribution
from ..models import AnalysisResult
from ..config import get_config

logger = logging.getLogger(__name__)


@dataclass
class AblationReport:
    """
    Comprehensive ablation study report.
    
    Contains all analysis results in structured format
    suitable for research documentation.
    """
    
    # Metadata
    video_filename: str = ""
    experiment_name: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Results
    mode_results: Dict[str, Dict] = field(default_factory=dict)
    
    # Comparisons
    pairwise_comparisons: Dict[str, Dict] = field(default_factory=dict)
    
    # Distributions
    score_distributions: Dict[str, Dict] = field(default_factory=dict)
    
    # Contributions
    modality_contributions: Dict[str, Dict] = field(default_factory=dict)
    
    # Summary statistics
    summary: Dict[str, Any] = field(default_factory=dict)
    
    # Research findings
    findings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Export report as dictionary."""
        return {
            'metadata': {
                'video_filename': self.video_filename,
                'experiment_name': self.experiment_name,
                'created_at': self.created_at
            },
            'mode_results': self.mode_results,
            'pairwise_comparisons': self.pairwise_comparisons,
            'score_distributions': self.score_distributions,
            'modality_contributions': self.modality_contributions,
            'summary': self.summary,
            'findings': self.findings
        }
    
    def save(self, filepath: str) -> None:
        """Save report to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Report saved to {filepath}")
    
    def save_markdown(self, filepath: str) -> None:
        """Save report as Markdown."""
        md = self.to_markdown()
        with open(filepath, 'w') as f:
            f.write(md)
        logger.info(f"Markdown report saved to {filepath}")
    
    def to_markdown(self) -> str:
        """Generate Markdown report."""
        lines = []
        
        # Header
        lines.append("# Ablation Study Report")
        lines.append("")
        lines.append(f"**Video:** {self.video_filename}")
        lines.append(f"**Experiment:** {self.experiment_name}")
        lines.append(f"**Generated:** {self.created_at}")
        lines.append("")
        
        # Summary
        lines.append("## Summary")
        lines.append("")
        if self.summary:
            lines.append(f"- **Modes tested:** {self.summary.get('modes_tested', 'N/A')}")
            lines.append(f"- **Best mode:** {self.summary.get('best_mode', 'N/A')}")
            lines.append(f"- **Text contribution:** {self.summary.get('text_contribution', 'N/A'):.2%}")
            lines.append(f"- **Audio contribution:** {self.summary.get('audio_contribution', 'N/A'):.2%}")
            lines.append(f"- **Visual contribution:** {self.summary.get('visual_contribution', 'N/A'):.2%}")
        lines.append("")
        
        # Mode Results Table
        lines.append("## Mode Results")
        lines.append("")
        lines.append("| Mode | Segments | Top Score | Mean Score | Time (s) |")
        lines.append("|------|----------|-----------|------------|----------|")
        
        for mode, result in self.mode_results.items():
            segments = result.get('segment_count', 0)
            top_score = result.get('top_score', 0)
            mean_score = result.get('mean_score', 0)
            time_s = result.get('execution_time', 0)
            lines.append(f"| {mode} | {segments} | {top_score:.3f} | {mean_score:.3f} | {time_s:.1f} |")
        
        lines.append("")
        
        # Score Distributions
        lines.append("## Score Distributions")
        lines.append("")
        lines.append("| Mode | Min | Max | Mean | Std |")
        lines.append("|------|-----|-----|------|-----|")
        
        for mode, dist in self.score_distributions.items():
            lines.append(
                f"| {mode} | {dist.get('min', 0):.3f} | {dist.get('max', 0):.3f} | "
                f"{dist.get('mean', 0):.3f} | {dist.get('std', 0):.3f} |"
            )
        
        lines.append("")
        
        # Modality Contributions
        lines.append("## Modality Contributions")
        lines.append("")
        lines.append("| Modality | Contribution | Rank Impact | Top Segment | Unique Value |")
        lines.append("|----------|--------------|-------------|-------------|--------------|")
        
        for modality, contrib in self.modality_contributions.items():
            lines.append(
                f"| {modality} | {contrib.get('contribution_score', 0):.3f} | "
                f"{contrib.get('rank_change_impact', 0):.2f} | "
                f"{contrib.get('top_segment_impact', 0):.0%} | "
                f"{contrib.get('unique_value', 0):.3f} |"
            )
        
        lines.append("")
        
        # Pairwise Comparisons
        lines.append("## Pairwise Comparisons")
        lines.append("")
        lines.append("| Comparison | Spearman ρ | Kendall τ | Top-1 | Top-3 |")
        lines.append("|------------|------------|-----------|-------|-------|")
        
        for comparison, metrics in self.pairwise_comparisons.items():
            rho = metrics.get('rank_correlation', 0)
            tau = metrics.get('kendall_tau', 0)
            top1 = "✓" if metrics.get('top_1_agreement') else "✗"
            top3 = metrics.get('top_3_agreement', 0)
            lines.append(f"| {comparison} | {rho:.3f} | {tau:.3f} | {top1} | {top3}/3 |")
        
        lines.append("")
        
        # Findings
        lines.append("## Key Findings")
        lines.append("")
        for i, finding in enumerate(self.findings, 1):
            lines.append(f"{i}. {finding}")
        
        lines.append("")
        lines.append("---")
        lines.append("*Report generated by Movie-Shorts Multimodal Video Understanding System*")
        
        return "\n".join(lines)
    
    @classmethod
    def load(cls, filepath: str) -> "AblationReport":
        """Load report from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls(
            video_filename=data['metadata']['video_filename'],
            experiment_name=data['metadata']['experiment_name'],
            created_at=data['metadata']['created_at'],
            mode_results=data.get('mode_results', {}),
            pairwise_comparisons=data.get('pairwise_comparisons', {}),
            score_distributions=data.get('score_distributions', {}),
            modality_contributions=data.get('modality_contributions', {}),
            summary=data.get('summary', {}),
            findings=data.get('findings', [])
        )


def generate_ablation_report(
    results: Dict[str, AblationResult],
    video_filename: str = "",
    experiment_name: str = ""
) -> AblationReport:
    """
    Generate a comprehensive ablation report from results.
    
    Args:
        results: Dict mapping mode name to AblationResult
        video_filename: Source video filename
        experiment_name: Experiment name
        
    Returns:
        AblationReport with all analysis
    """
    config = get_config()
    analyzer = ModalityAnalyzer()
    
    report = AblationReport(
        video_filename=video_filename,
        experiment_name=experiment_name or config.research.experiment_name
    )
    
    # Extract successful analysis results
    analysis_results = {
        mode: r.analysis_result
        for mode, r in results.items()
        if r.success and r.analysis_result
    }
    
    if not analysis_results:
        report.findings.append("No successful ablation runs to analyze.")
        return report
    
    # Mode results
    for mode, ablation_result in results.items():
        top_score = 0.0
        mean_score = 0.0
        
        if ablation_result.success and ablation_result.analysis_result:
            segments = ablation_result.analysis_result.segments
            scores = [s.score.total_score for s in segments if s.score]
            if scores:
                top_score = max(scores)
                mean_score = sum(scores) / len(scores)
        
        report.mode_results[mode] = {
            'success': ablation_result.success,
            'segment_count': ablation_result.segment_count,
            'top_score': top_score,
            'mean_score': mean_score,
            'execution_time': ablation_result.execution_time_seconds,
            'error': ablation_result.error_message if not ablation_result.success else None
        }
    
    # Score distributions
    for mode, result in analysis_results.items():
        dist = analyzer.get_score_distribution(result)
        report.score_distributions[mode] = dist.to_dict()
    
    # Pairwise comparisons
    comparisons = analyzer.compare_all_modes(analysis_results)
    report.pairwise_comparisons = {
        key: metrics.to_dict()
        for key, metrics in comparisons.items()
    }
    
    # Modality contributions
    contributions = analyzer.analyze_contributions(analysis_results)
    report.modality_contributions = {
        modality: contrib.to_dict()
        for modality, contrib in contributions.items()
    }
    
    # Generate summary
    report.summary = _generate_summary(report, analysis_results, contributions)
    
    # Generate findings
    report.findings = _generate_findings(report, analysis_results, contributions, comparisons)
    
    return report


def _generate_summary(
    report: AblationReport,
    results: Dict[str, AnalysisResult],
    contributions: Dict[str, ModalityContribution]
) -> Dict[str, Any]:
    """Generate summary statistics."""
    summary = {
        'modes_tested': len(report.mode_results),
        'successful_modes': sum(1 for r in report.mode_results.values() if r.get('success')),
    }
    
    # Best mode by mean score
    best_mode = None
    best_score = -1
    for mode, mode_result in report.mode_results.items():
        if mode_result.get('success') and mode_result.get('mean_score', 0) > best_score:
            best_score = mode_result['mean_score']
            best_mode = mode
    
    summary['best_mode'] = best_mode
    summary['best_mean_score'] = best_score
    
    # Modality contributions
    for modality, contrib in contributions.items():
        summary[f'{modality}_contribution'] = contrib.contribution_score
    
    # Full multimodal comparison
    if 'full_multimodal' in results:
        full = report.mode_results.get('full_multimodal', {})
        summary['full_multimodal_score'] = full.get('mean_score', 0)
    
    return summary


def _generate_findings(
    report: AblationReport,
    results: Dict[str, AnalysisResult],
    contributions: Dict[str, ModalityContribution],
    comparisons: Dict[str, ComparisonMetrics]
) -> List[str]:
    """Generate key research findings."""
    findings = []
    
    # Finding 1: Which modality is most important?
    if contributions:
        sorted_contribs = sorted(
            contributions.items(),
            key=lambda x: x[1].contribution_score,
            reverse=True
        )
        if sorted_contribs:
            top_modality = sorted_contribs[0][0]
            top_score = sorted_contribs[0][1].contribution_score
            findings.append(
                f"**{top_modality.title()}** shows the highest correlation (ρ={top_score:.3f}) "
                f"with full multimodal ranking, suggesting it is the most important single modality."
            )
    
    # Finding 2: Does multimodal outperform unimodal?
    full_score = report.mode_results.get('full_multimodal', {}).get('mean_score', 0)
    unimodal_scores = [
        (mode, r.get('mean_score', 0))
        for mode, r in report.mode_results.items()
        if mode in ['text_only', 'audio_only', 'visual_only']
    ]
    
    if unimodal_scores and full_score:
        best_unimodal = max(unimodal_scores, key=lambda x: x[1])
        if full_score > best_unimodal[1]:
            improvement = ((full_score - best_unimodal[1]) / best_unimodal[1]) * 100
            findings.append(
                f"Full multimodal achieves {improvement:.1f}% higher mean score than the best "
                f"single modality ({best_unimodal[0]}), demonstrating the value of fusion."
            )
        else:
            findings.append(
                f"Interestingly, {best_unimodal[0]} achieves comparable or better performance "
                f"than full multimodal, suggesting potential for simpler models."
            )
    
    # Finding 3: Modality agreement
    for key, metrics in comparisons.items():
        if 'full_multimodal' in key:
            other_mode = key.replace('_vs_full_multimodal', '').replace('full_multimodal_vs_', '')
            if metrics.top_1_agreement:
                findings.append(
                    f"{other_mode} agrees with full multimodal on the top-ranked segment, "
                    f"with rank correlation ρ={metrics.rank_correlation:.3f}."
                )
            elif metrics.rank_correlation > 0.7:
                findings.append(
                    f"{other_mode} shows strong agreement with full multimodal "
                    f"(ρ={metrics.rank_correlation:.3f}) despite different top selections."
                )
    
    # Finding 4: Unique contributions
    for modality, contrib in contributions.items():
        if contrib.unique_value > 0.6:
            findings.append(
                f"{modality.title()} provides unique information (uniqueness={contrib.unique_value:.3f}) "
                f"not captured by other modalities."
            )
    
    # Finding 5: Processing time
    times = {
        mode: r.get('execution_time', 0)
        for mode, r in report.mode_results.items()
        if r.get('success')
    }
    if times:
        fastest = min(times.items(), key=lambda x: x[1])
        slowest = max(times.items(), key=lambda x: x[1])
        if slowest[1] > fastest[1] * 2:
            findings.append(
                f"{fastest[0]} is {slowest[1]/fastest[1]:.1f}x faster than {slowest[0]}, "
                f"offering a speed-quality tradeoff."
            )
    
    if not findings:
        findings.append("Insufficient data for meaningful findings. Run more ablation modes.")
    
    return findings

