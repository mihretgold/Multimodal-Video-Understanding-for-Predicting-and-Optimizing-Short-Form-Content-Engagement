#!/usr/bin/env python3
"""
Segment Visualization Script
============================
Visualizes segment boundaries overlaid on video timeline.

Creates ASCII-art timeline showing:
- Segment boundaries
- Segment durations
- Boundary strength indicators

Usage:
    python scripts/visualize_segments.py --subtitle-file subtitles.srt --duration 300
    python scripts/visualize_segments.py --test  # Use test data
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.segmentation import TemporalSegmenter, BoundaryDetector
from app.models import SubtitleData, SubtitleEntry


def create_test_subtitles(duration: float = 300.0):
    """Create test subtitle data."""
    entries = []
    current_time = 0.0
    index = 0
    
    phrases = [
        "This is an interesting point.",
        "Let me explain further.",
        "You see, it works like this.",
        "And that's really important!",
        "Now, here's a question for you?",
        "Think about it carefully.",
        "The answer might surprise you.",
        "Moving on to the next topic.",
    ]
    
    while current_time < duration - 5:
        phrase = phrases[index % len(phrases)]
        entry_duration = 3.0 + (index % 3)
        
        entries.append(SubtitleEntry(
            index=index,
            start_seconds=current_time,
            end_seconds=current_time + entry_duration,
            text=phrase
        ))
        
        current_time += entry_duration
        
        if index % 10 == 0:
            current_time += 2.5
        elif index % 5 == 0:
            current_time += 1.0
        else:
            current_time += 0.3
        
        index += 1
    
    return SubtitleData(
        video_filename="test_video.mp4",
        entries=entries,
        source="test",
        language="en"
    )


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{mins:02d}:{secs:02d}"


def visualize_timeline(
    subtitle_data: SubtitleData,
    duration: float,
    width: int = 80,
    strategy: str = 'pause_based'
):
    """
    Create ASCII visualization of segmentation.
    
    Args:
        subtitle_data: Subtitle data
        duration: Video duration in seconds
        width: Timeline width in characters
        strategy: Segmentation strategy
    """
    print("\n" + "="*width)
    print(f"TEMPORAL SEGMENTATION VISUALIZATION")
    print(f"Video Duration: {format_time(duration)} ({duration:.1f}s)")
    print(f"Strategy: {strategy}")
    print("="*width + "\n")
    
    # Detect boundaries
    detector = BoundaryDetector()
    boundaries = detector.detect_from_subtitles(subtitle_data.entries, duration)
    
    # Run segmentation
    segmenter = TemporalSegmenter(strategy=strategy)
    segments = segmenter.segment(subtitle_data, duration)
    
    # === TIMELINE VISUALIZATION ===
    print("TIMELINE (boundaries shown as |, strength indicated by height)")
    print("-" * width)
    
    # Create timeline rows
    timeline_chars = width - 10  # Leave space for time labels
    scale = duration / timeline_chars
    
    # Boundary indicator row (shows strength)
    strong_row = [' '] * timeline_chars
    medium_row = [' '] * timeline_chars
    weak_row = [' '] * timeline_chars
    
    for b in boundaries:
        pos = min(int(b.timestamp / scale), timeline_chars - 1)
        if b.strength >= 0.7:
            strong_row[pos] = '|'
            medium_row[pos] = '|'
            weak_row[pos] = '|'
        elif b.strength >= 0.5:
            medium_row[pos] = '|'
            weak_row[pos] = '|'
        else:
            weak_row[pos] = '.'
    
    print("Strong:  " + "".join(strong_row))
    print("Medium:  " + "".join(medium_row))
    print("Weak:    " + "".join(weak_row))
    
    # Time axis
    print("-" * width)
    axis_row = ['-'] * timeline_chars
    labels = []
    for t in range(0, int(duration) + 1, 60):
        pos = min(int(t / scale), timeline_chars - 1)
        axis_row[pos] = '+'
        labels.append((pos, format_time(t)))
    
    print("Time:    " + "".join(axis_row))
    
    # Print time labels
    label_row = [' '] * timeline_chars
    for pos, label in labels:
        if pos + len(label) <= timeline_chars:
            for i, c in enumerate(label):
                if pos + i < timeline_chars:
                    label_row[pos + i] = c
    print("         " + "".join(label_row))
    
    # === SEGMENT VISUALIZATION ===
    print("\n" + "-"*width)
    print("DETECTED SEGMENTS")
    print("-"*width)
    
    for i, seg in enumerate(segments):
        print(f"\nSegment {i+1}: {format_time(seg.start_seconds)} -> {format_time(seg.end_seconds)}")
        print(f"  Duration: {seg.duration_seconds:.1f}s")
        print(f"  Type: {seg.segment_type}")
        
        # Create mini-timeline for segment
        seg_chars = 50
        seg_start_pos = int((seg.start_seconds / duration) * seg_chars)
        seg_end_pos = int((seg.end_seconds / duration) * seg_chars)
        
        seg_row = ['.'] * seg_chars
        for p in range(seg_start_pos, min(seg_end_pos + 1, seg_chars)):
            seg_row[p] = '#'
        
        print(f"  |{''.join(seg_row)}|")
        print(f"   0{' '*(seg_chars-5)}{format_time(duration)}")
        
        # Show text preview
        if seg.text_preview:
            preview = seg.text_preview[:70] + "..." if len(seg.text_preview) > 70 else seg.text_preview
            print(f"  Preview: \"{preview}\"")
    
    # === COVERAGE STATISTICS ===
    print("\n" + "-"*width)
    print("COVERAGE STATISTICS")
    print("-"*width)
    
    total_coverage = sum(s.duration_seconds for s in segments)
    coverage_pct = (total_coverage / duration) * 100 if duration > 0 else 0
    
    print(f"  Total segments: {len(segments)}")
    print(f"  Total coverage: {total_coverage:.1f}s ({coverage_pct:.1f}%)")
    print(f"  Average segment: {total_coverage/len(segments):.1f}s" if segments else "  No segments")
    print(f"  Boundaries detected: {len(boundaries)}")
    print(f"  Strong boundaries: {len([b for b in boundaries if b.strength >= 0.7])}")
    
    print("\n" + "="*width + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize temporal segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Use test data instead of real subtitles'
    )
    
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=300.0,
        help='Video duration in seconds (default: 300)'
    )
    
    parser.add_argument(
        '--strategy', '-s',
        choices=['pause_based', 'fixed_window', 'semantic_boundary', 'hybrid'],
        default='pause_based',
        help='Segmentation strategy (default: pause_based)'
    )
    
    parser.add_argument(
        '--width', '-w',
        type=int,
        default=80,
        help='Output width in characters (default: 80)'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all strategies'
    )
    
    args = parser.parse_args()
    
    # Create test data
    subtitle_data = create_test_subtitles(args.duration)
    
    if args.compare:
        # Compare all strategies
        for strategy in ['pause_based', 'fixed_window', 'semantic_boundary', 'hybrid']:
            visualize_timeline(
                subtitle_data,
                args.duration,
                args.width,
                strategy
            )
    else:
        visualize_timeline(
            subtitle_data,
            args.duration,
            args.width,
            args.strategy
        )


if __name__ == "__main__":
    main()

