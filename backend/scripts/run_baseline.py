#!/usr/bin/env python3
"""
Run Baseline Script
===================
Command-line interface for running the baseline video analysis system.

Usage:
    python scripts/run_baseline.py --video path/to/video.mp4
    python scripts/run_baseline.py --video video.mp4 --experiment my_experiment
    python scripts/run_baseline.py --export-spec baseline_spec.json
    python scripts/run_baseline.py --list-results
"""

import argparse
import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.baseline import BaselineRunner, BaselineSpec
from app.baseline.runner import run_baseline, export_baseline_spec
from app.config import get_config, AblationConfig
from app.logging_config import get_research_logger

logger = get_research_logger("cli", log_to_file=False)


def main():
    parser = argparse.ArgumentParser(
        description="Run the baseline video analysis system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run baseline on a video
    python scripts/run_baseline.py --video uploads/my_video.mp4
    
    # Run with custom experiment name
    python scripts/run_baseline.py --video video.mp4 --experiment exp_001
    
    # Run with text-only ablation
    python scripts/run_baseline.py --video video.mp4 --ablation text_only
    
    # Export specification
    python scripts/run_baseline.py --export-spec spec.json
    
    # List previous results
    python scripts/run_baseline.py --list-results
        """
    )
    
    parser.add_argument(
        '--video', '-v',
        type=str,
        help='Path to video file to analyze'
    )
    
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        default=None,
        help='Experiment name for this run'
    )
    
    parser.add_argument(
        '--ablation', '-a',
        type=str,
        choices=['text_only', 'audio_only', 'visual_only', 'text_audio', 'full_multimodal'],
        default='full_multimodal',
        help='Ablation mode (default: full_multimodal)'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching of expensive operations'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to disk'
    )
    
    parser.add_argument(
        '--export-spec',
        type=str,
        metavar='FILE',
        help='Export baseline specification to JSON file'
    )
    
    parser.add_argument(
        '--list-results',
        action='store_true',
        help='List previous baseline results'
    )
    
    parser.add_argument(
        '--show-result',
        type=str,
        metavar='RESULT_ID',
        help='Show details of a specific result'
    )
    
    parser.add_argument(
        '--verify-result',
        type=str,
        metavar='RESULT_FILE',
        help='Verify a result file against specification'
    )
    
    args = parser.parse_args()
    
    # Handle export spec
    if args.export_spec:
        export_baseline_spec(args.export_spec)
        print(f"Exported specification to: {args.export_spec}")
        return 0
    
    # Handle list results
    if args.list_results:
        return list_results()
    
    # Handle show result
    if args.show_result:
        return show_result(args.show_result)
    
    # Handle verify result
    if args.verify_result:
        return verify_result(args.verify_result)
    
    # Run baseline (requires video)
    if not args.video:
        parser.error("--video is required to run the baseline")
    
    return run_baseline_cli(
        video_path=args.video,
        experiment_name=args.experiment,
        ablation_mode=args.ablation,
        use_cache=not args.no_cache,
        save_results=not args.no_save
    )


def run_baseline_cli(
    video_path: str,
    experiment_name: str = None,
    ablation_mode: str = 'full_multimodal',
    use_cache: bool = True,
    save_results: bool = True
) -> int:
    """Run the baseline and display results."""
    
    # Resolve video path
    if not os.path.isabs(video_path):
        config = get_config()
        # Try relative to uploads folder
        uploads_path = config.paths.uploads / video_path
        if uploads_path.exists():
            video_path = str(uploads_path)
        elif not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return 1
    
    print("="*60)
    print("BASELINE VIDEO ANALYSIS")
    print("="*60)
    print(f"Video: {video_path}")
    print(f"Ablation Mode: {ablation_mode}")
    print(f"Experiment: {experiment_name or 'default'}")
    print(f"Cache: {'enabled' if use_cache else 'disabled'}")
    print("="*60)
    print()
    
    # Configure ablation
    config = get_config()
    ablation_map = {
        'text_only': AblationConfig.text_only,
        'audio_only': AblationConfig.audio_only,
        'visual_only': AblationConfig.visual_only,
        'text_audio': AblationConfig.text_audio,
        'full_multimodal': AblationConfig.full_multimodal,
    }
    config.ablation = ablation_map[ablation_mode]()
    
    # Run baseline
    runner = BaselineRunner(
        config=config,
        use_cache=use_cache,
        save_results=save_results
    )
    
    print("Running baseline pipeline...")
    print()
    
    output = runner.run(video_path, experiment_name=experiment_name)
    
    # Display results
    print()
    print("="*60)
    print("RESULTS")
    print("="*60)
    
    if output.is_valid:
        result = output.result
        print(f"Status: SUCCESS")
        print(f"Result ID: {result.result_id}")
        print(f"Processing Time: {result.processing_time_seconds:.2f}s")
        print()
        
        print("VIDEO METADATA:")
        print(f"  - Filename: {result.video_metadata.filename}")
        print(f"  - Duration: {result.video_metadata.duration_formatted}")
        print(f"  - Resolution: {result.video_metadata.resolution}")
        print(f"  - FPS: {result.video_metadata.fps:.1f}")
        print()
        
        if result.subtitle_data:
            print("SUBTITLES:")
            print(f"  - Source: {result.subtitle_data.source}")
            print(f"  - Language: {result.subtitle_data.language}")
            print(f"  - Entries: {len(result.subtitle_data.entries)}")
            print(f"  - Word Count: {result.subtitle_data.word_count}")
            print()
        
        print(f"SEGMENTS ({result.segment_count}):")
        for i, segment in enumerate(result.segments):
            score_str = f", score={segment.score.total_score:.3f}" if segment.score else ""
            print(f"  {i+1}. [{segment.segment_type}] {segment.time_range_formatted} ({segment.duration_formatted}){score_str}")
        
        print()
        print("="*60)
        return 0
    
    else:
        print(f"Status: FAILED")
        print()
        print("VALIDATION ERRORS:")
        for error in output.validation_errors:
            print(f"  - {error}")
        print()
        print("="*60)
        return 1


def list_results() -> int:
    """List previous baseline results."""
    config = get_config()
    results_dir = config.paths.experiments / "baseline_outputs"
    
    if not results_dir.exists():
        print("No baseline results found.")
        return 0
    
    result_files = sorted(results_dir.glob("baseline_*.json"), reverse=True)
    
    if not result_files:
        print("No baseline results found.")
        return 0
    
    print("="*60)
    print("BASELINE RESULTS")
    print("="*60)
    print()
    
    from app.baseline.specification import BaselineOutput
    
    for result_file in result_files[:20]:  # Show last 20
        try:
            output = BaselineOutput.load(str(result_file))
            result = output.result
            
            status = "OK" if output.is_valid else "FAILED"
            filename = result.video_metadata.filename if result else "N/A"
            segments = result.segment_count if result else 0
            
            print(f"[{status}] {result_file.name}")
            print(f"      Video: {filename}")
            print(f"      Segments: {segments}")
            print(f"      Created: {output.executed_at}")
            print()
            
        except Exception as e:
            print(f"[ERROR] {result_file.name}: {e}")
            print()
    
    return 0


def show_result(result_id: str) -> int:
    """Show details of a specific result."""
    config = get_config()
    results_dir = config.paths.experiments / "baseline_outputs"
    
    # Find the result file
    result_file = results_dir / f"baseline_{result_id}.json"
    
    if not result_file.exists():
        # Try searching
        matches = list(results_dir.glob(f"*{result_id}*.json"))
        if not matches:
            print(f"Result not found: {result_id}")
            return 1
        result_file = matches[0]
    
    from app.baseline.specification import BaselineOutput
    
    output = BaselineOutput.load(str(result_file))
    
    print("="*60)
    print("BASELINE RESULT DETAILS")
    print("="*60)
    print()
    print(json.dumps(output.to_dict(), indent=2))
    
    return 0


def verify_result(result_file: str) -> int:
    """Verify a result file against specification."""
    from app.baseline.specification import BaselineOutput
    
    if not os.path.exists(result_file):
        print(f"File not found: {result_file}")
        return 1
    
    output = BaselineOutput.load(result_file)
    
    print("="*60)
    print("RESULT VERIFICATION")
    print("="*60)
    print()
    
    runner = BaselineRunner()
    
    if output.result:
        valid, errors = runner.verify_result(output.result)
        
        if valid:
            print("Status: VALID")
            print("All checks passed.")
        else:
            print("Status: INVALID")
            print()
            print("Errors:")
            for error in errors:
                print(f"  - {error}")
    else:
        print("Status: NO RESULT")
        print("The output file contains no result.")
    
    print()
    return 0 if (output.result and valid) else 1


if __name__ == "__main__":
    sys.exit(main())

