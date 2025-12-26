"""
Ablation & Modality Analysis Package
====================================
Systematic analysis of modality contributions to engagement prediction.

This package provides:
- Ablation study runner
- Modality comparison metrics
- Research report generation
- Visualization utilities

Usage:
    from app.ablation import AblationRunner, AblationReport
    
    runner = AblationRunner()
    results = runner.run_all_modes(video_path)
    
    report = runner.generate_report(results)
    report.save("ablation_report.json")
"""

from .runner import AblationRunner, AblationResult
from .analyzer import ModalityAnalyzer, ComparisonMetrics
from .report import AblationReport, generate_ablation_report

__all__ = [
    'AblationRunner',
    'AblationResult',
    'ModalityAnalyzer',
    'ComparisonMetrics',
    'AblationReport',
    'generate_ablation_report',
]

