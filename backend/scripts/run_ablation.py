#!/usr/bin/env python3
"""
Run Ablation Study Script
=========================
Command-line interface for running ablation experiments.

Usage:
    python scripts/run_ablation.py --video path/to/video.mp4
    python scripts/run_ablation.py --video video.mp4 --modes text_only full_multimodal
    python scripts/run_ablation.py --video video.mp4 --output report.json --markdown report.md
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ablation import AblationRunner, generate_ablation_report
from app.config import get_config


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation study on a video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all ablation modes
    python scripts/run_ablation.py --video uploads/my_video.mp4
    
    # Run specific modes
    python scripts/run_ablation.py --video video.mp4 --modes text_only audio_only full_multimodal
    
    # Save reports
    python scripts/run_ablation.py --video video.mp4 --output report.json --markdown report.md
    
    # Quick comparison (text vs full)
    python scripts/run_ablation.py --video video.mp4 --quick
        """
    )
    
    parser.add_argument(
        '--video', '-v',
        type=str,
        required=True,
        help='Path to video file to analyze'
    )
    
    parser.add_argument(
        '--modes', '-m',
        nargs='+',
        choices=['text_only', 'audio_only', 'visual_only', 'text_audio', 'full_multimodal'],
        default=None,
        help='Ablation modes to run (default: all)'
    )
    
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick comparison: text_only vs full_multimodal'
    )
    
    parser.add_argument(
        '--unimodal', '-u',
        action='store_true',
        help='Run only single-modality modes'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output path for JSON report'
    )
    
    parser.add_argument(
        '--markdown', '--md',
        type=str,
        default=None,
        help='Output path for Markdown report'
    )
    
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        default='ablation_study',
        help='Experiment name for logging'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save individual results'
    )
    
    args = parser.parse_args()
    
    # Resolve video path
    video_path = args.video
    if not os.path.isabs(video_path):
        config = get_config()
        uploads_path = config.paths.uploads / video_path
        if uploads_path.exists():
            video_path = str(uploads_path)
        elif not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return 1
    
    # Determine modes
    if args.quick:
        modes = ['text_only', 'full_multimodal']
    elif args.unimodal:
        modes = ['text_only', 'audio_only', 'visual_only']
    else:
        modes = args.modes  # None = all modes
    
    print("="*60)
    print("ABLATION STUDY")
    print("="*60)
    print(f"Video: {video_path}")
    print(f"Modes: {modes or 'all'}")
    print(f"Experiment: {args.experiment}")
    print("="*60)
    print()
    
    # Run ablation study
    runner = AblationRunner()
    
    if modes:
        results = runner.run_modes(video_path, modes, save_results=not args.no_save)
    else:
        results = runner.run_all_modes(video_path, save_results=not args.no_save)
    
    # Generate report
    print("\nGenerating report...")
    video_filename = os.path.basename(video_path)
    report = generate_ablation_report(
        results,
        video_filename=video_filename,
        experiment_name=args.experiment
    )
    
    # Display summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print()
    
    print("Mode Results:")
    for mode, result in results.items():
        status = "OK" if result.success else "FAILED"
        if result.success:
            top = result.top_segment
            top_score = top.score.total_score if top and top.score else 0
            print(f"  [{status}] {mode}: {result.segment_count} segments, "
                  f"top score={top_score:.3f}, time={result.execution_time_seconds:.1f}s")
        else:
            print(f"  [{status}] {mode}: {result.error_message}")
    
    print()
    print("Modality Contributions:")
    for modality, contrib in report.modality_contributions.items():
        print(f"  {modality}: contribution={contrib.get('contribution_score', 0):.3f}, "
              f"unique={contrib.get('unique_value', 0):.3f}")
    
    print()
    print("Key Findings:")
    for i, finding in enumerate(report.findings[:5], 1):
        # Truncate long findings
        if len(finding) > 100:
            finding = finding[:100] + "..."
        print(f"  {i}. {finding}")
    
    # Save reports
    if args.output:
        report.save(args.output)
        print(f"\nJSON report saved to: {args.output}")
    
    if args.markdown:
        report.save_markdown(args.markdown)
        print(f"Markdown report saved to: {args.markdown}")
    
    # Default save
    if not args.output and not args.markdown:
        config = get_config()
        default_path = config.paths.experiments / f"ablation_{args.experiment}.json"
        report.save(str(default_path))
        print(f"\nReport saved to: {default_path}")
    
    print()
    print("="*60)
    print("ABLATION STUDY COMPLETE")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

