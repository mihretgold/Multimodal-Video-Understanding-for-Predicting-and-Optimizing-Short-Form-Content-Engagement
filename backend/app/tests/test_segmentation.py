"""
Temporal Segmentation Tests
===========================
Tests for the temporal segmentation module.
"""

import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.segmentation import (
    TemporalSegmenter,
    PauseBasedStrategy,
    FixedWindowStrategy,
    SemanticBoundaryStrategy,
    HybridStrategy,
    BoundaryDetector,
    SpeechBoundary,
)
from app.segmentation.strategies import SegmentationParams
from app.segmentation.boundaries import BoundaryType
from app.models import SubtitleData, SubtitleEntry


# =============================================================================
# TEST FIXTURES
# =============================================================================

def create_test_subtitles():
    """Create test subtitle data with realistic gaps."""
    entries = [
        SubtitleEntry(index=0, start_seconds=0.0, end_seconds=3.5, 
                      text="Hello and welcome to this video."),
        SubtitleEntry(index=1, start_seconds=3.7, end_seconds=7.2, 
                      text="Today we're going to discuss something interesting."),
        SubtitleEntry(index=2, start_seconds=8.5, end_seconds=12.0,  # 1.3s gap - strong boundary
                      text="Let me start with a question."),
        SubtitleEntry(index=3, start_seconds=12.2, end_seconds=16.0, 
                      text="What do you think about this topic?"),
        SubtitleEntry(index=4, start_seconds=16.5, end_seconds=20.0, 
                      text="It's really fascinating when you think about it."),
        SubtitleEntry(index=5, start_seconds=25.0, end_seconds=30.0,  # 5s gap - very strong
                      text="Now let's move on to the second part!"),
        SubtitleEntry(index=6, start_seconds=30.5, end_seconds=35.0, 
                      text="This is where things get really exciting."),
        SubtitleEntry(index=7, start_seconds=35.5, end_seconds=40.0, 
                      text="Watch closely as we demonstrate."),
        SubtitleEntry(index=8, start_seconds=40.5, end_seconds=45.0, 
                      text="See how this works in practice?"),
        SubtitleEntry(index=9, start_seconds=50.0, end_seconds=55.0,  # 5s gap
                      text="And here's the dramatic conclusion."),
        SubtitleEntry(index=10, start_seconds=55.5, end_seconds=60.0, 
                      text="Thank you for watching this video!"),
    ]
    
    return SubtitleData(
        video_filename="test_video.mp4",
        entries=entries,
        source="test",
        language="en"
    )


def create_long_subtitles(duration_seconds: float = 300.0):
    """Create longer subtitle data for testing segmentation."""
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
    
    while current_time < duration_seconds - 5:
        # Add entry
        phrase = phrases[index % len(phrases)]
        entry_duration = 3.0 + (index % 3)  # 3-5 seconds
        
        entries.append(SubtitleEntry(
            index=index,
            start_seconds=current_time,
            end_seconds=current_time + entry_duration,
            text=phrase
        ))
        
        current_time += entry_duration
        
        # Add gap (varies by position)
        if index % 10 == 0:
            current_time += 2.5  # Strong boundary every 10 entries
        elif index % 5 == 0:
            current_time += 1.0  # Medium boundary every 5 entries
        else:
            current_time += 0.3  # Small gap
        
        index += 1
    
    return SubtitleData(
        video_filename="long_video.mp4",
        entries=entries,
        source="test",
        language="en"
    )


# =============================================================================
# BOUNDARY DETECTOR TESTS
# =============================================================================

def test_boundary_detector_creation():
    """Test creating a boundary detector."""
    detector = BoundaryDetector()
    assert detector.min_gap_seconds == 0.5
    assert detector.strong_gap_seconds == 1.5
    print("[PASS] Boundary detector creation test passed")


def test_boundary_detection():
    """Test detecting boundaries from subtitles."""
    detector = BoundaryDetector()
    subtitles = create_test_subtitles()
    
    boundaries = detector.detect_from_subtitles(subtitles.entries, 60.0)
    
    assert len(boundaries) > 0
    assert all(isinstance(b, SpeechBoundary) for b in boundaries)
    
    # Should detect strong boundaries (5s gaps at ~20s and ~45s)
    strong_boundaries = [b for b in boundaries if b.strength > 0.7]
    assert len(strong_boundaries) >= 2, f"Expected at least 2 strong boundaries, got {len(strong_boundaries)}"
    
    print(f"[PASS] Detected {len(boundaries)} boundaries ({len(strong_boundaries)} strong)")


def test_boundary_sorting():
    """Test that boundaries are sorted by timestamp."""
    detector = BoundaryDetector()
    subtitles = create_test_subtitles()
    
    boundaries = detector.detect_from_subtitles(subtitles.entries, 60.0)
    
    for i in range(len(boundaries) - 1):
        assert boundaries[i].timestamp <= boundaries[i + 1].timestamp
    
    print("[PASS] Boundary sorting test passed")


def test_top_boundaries_selection():
    """Test selecting top boundaries with minimum separation."""
    detector = BoundaryDetector()
    subtitles = create_test_subtitles()
    
    boundaries = detector.detect_from_subtitles(subtitles.entries, 60.0)
    top = detector.get_top_boundaries(boundaries, count=3, min_separation=10.0)
    
    assert len(top) <= 3
    
    # Check minimum separation
    for i in range(len(top) - 1):
        separation = abs(top[i + 1].timestamp - top[i].timestamp)
        assert separation >= 10.0, f"Separation {separation} < 10.0"
    
    print(f"[PASS] Selected {len(top)} top boundaries with separation >= 10s")


# =============================================================================
# SEGMENTATION STRATEGY TESTS
# =============================================================================

def test_pause_based_strategy():
    """Test pause-based segmentation strategy."""
    strategy = PauseBasedStrategy()
    subtitles = create_long_subtitles(180.0)  # 3 min video
    
    params = SegmentationParams(
        min_duration_seconds=30.0,
        target_duration_seconds=60.0,
        max_duration_seconds=70.0,
        min_segments=1,
        max_segments=5
    )
    
    segments = strategy.segment(subtitles, 180.0, params)
    
    assert len(segments) >= 1
    assert len(segments) <= 5
    
    # Check duration constraints
    for seg in segments:
        assert seg.duration_seconds >= params.min_duration_seconds, \
            f"Segment too short: {seg.duration_seconds}s"
        assert seg.duration_seconds <= params.max_duration_seconds + 5, \
            f"Segment too long: {seg.duration_seconds}s"
    
    print(f"[PASS] Pause-based strategy created {len(segments)} segments")


def test_fixed_window_strategy():
    """Test fixed-window segmentation strategy."""
    strategy = FixedWindowStrategy(snap_to_boundaries=True)
    subtitles = create_long_subtitles(180.0)
    
    params = SegmentationParams(
        min_duration_seconds=55.0,
        target_duration_seconds=60.0,
        max_duration_seconds=70.0,
        min_segments=1,
        max_segments=5
    )
    
    segments = strategy.segment(subtitles, 180.0, params)
    
    assert len(segments) >= 1
    
    print(f"[PASS] Fixed-window strategy created {len(segments)} segments")


def test_semantic_boundary_strategy():
    """Test semantic boundary segmentation strategy."""
    strategy = SemanticBoundaryStrategy()
    subtitles = create_long_subtitles(180.0)
    
    params = SegmentationParams(
        min_duration_seconds=30.0,
        target_duration_seconds=60.0,
        max_duration_seconds=90.0,
        min_segments=1,
        max_segments=5
    )
    
    segments = strategy.segment(subtitles, 180.0, params)
    
    assert len(segments) >= 1
    
    print(f"[PASS] Semantic boundary strategy created {len(segments)} segments")


def test_hybrid_strategy():
    """Test hybrid segmentation strategy."""
    strategy = HybridStrategy()
    subtitles = create_long_subtitles(180.0)
    
    params = SegmentationParams(
        min_duration_seconds=30.0,
        target_duration_seconds=60.0,
        max_duration_seconds=90.0,
        min_segments=1,
        max_segments=5
    )
    
    segments = strategy.segment(subtitles, 180.0, params)
    
    assert len(segments) >= 1
    
    print(f"[PASS] Hybrid strategy created {len(segments)} segments")


# =============================================================================
# TEMPORAL SEGMENTER TESTS
# =============================================================================

def test_temporal_segmenter_creation():
    """Test creating a TemporalSegmenter."""
    segmenter = TemporalSegmenter()
    
    assert segmenter.strategy is not None
    assert segmenter.params is not None
    
    info = segmenter.get_strategy_info()
    assert 'name' in info
    assert 'params' in info
    
    print(f"[PASS] Created segmenter with strategy '{info['name']}'")


def test_temporal_segmenter_with_strategy():
    """Test creating a segmenter with specific strategy."""
    for strategy_name in ['pause_based', 'fixed_window', 'semantic_boundary', 'hybrid']:
        segmenter = TemporalSegmenter(strategy=strategy_name)
        assert segmenter.strategy.name == strategy_name
    
    print("[PASS] Created segmenters with all strategies")


def test_temporal_segmenter_segment():
    """Test running segmentation."""
    segmenter = TemporalSegmenter(strategy='pause_based')
    subtitles = create_long_subtitles(300.0)  # 5 min video
    
    segments = segmenter.segment(subtitles, 300.0)
    
    assert len(segments) >= 1
    
    # Check all segments have required attributes
    for seg in segments:
        assert seg.segment_id is not None
        assert seg.start_seconds >= 0
        assert seg.end_seconds > seg.start_seconds
        assert seg.duration_seconds > 0
    
    print(f"[PASS] Segmented 5-min video into {len(segments)} segments")


def test_segmenter_respects_config():
    """Test that segmenter respects configuration."""
    from app.config import SegmentationConfig
    
    config = SegmentationConfig(
        min_duration_seconds=40.0,
        target_duration_seconds=50.0,
        max_duration_seconds=60.0,
        max_segments=3
    )
    
    segmenter = TemporalSegmenter(config=config)
    assert segmenter.params.target_duration_seconds == 50.0
    assert segmenter.params.max_segments == 3
    
    print("[PASS] Segmenter respects configuration")


def test_boundary_detection_via_segmenter():
    """Test detecting boundaries through segmenter."""
    segmenter = TemporalSegmenter()
    subtitles = create_test_subtitles()
    
    boundaries = segmenter.detect_boundaries(subtitles, 60.0)
    
    assert len(boundaries) > 0
    assert all(isinstance(b, SpeechBoundary) for b in boundaries)
    
    print(f"[PASS] Detected {len(boundaries)} boundaries via segmenter")


# =============================================================================
# SEGMENT VALIDATION TESTS
# =============================================================================

def test_segment_validation():
    """Test segment validation in segmenter."""
    segmenter = TemporalSegmenter(strategy='fixed_window')
    subtitles = create_long_subtitles(120.0)  # 2 min video
    
    segments = segmenter.segment(subtitles, 120.0)
    
    # All segments should be within video bounds
    for seg in segments:
        assert seg.start_seconds >= 0
        assert seg.end_seconds <= 120.0
    
    print(f"[PASS] All {len(segments)} segments within bounds")


def test_no_overlap():
    """Test that segments don't overlap."""
    segmenter = TemporalSegmenter(strategy='pause_based')
    subtitles = create_long_subtitles(300.0)
    
    segments = segmenter.segment(subtitles, 300.0)
    
    # Sort by start time
    segments_sorted = sorted(segments, key=lambda s: s.start_seconds)
    
    for i in range(len(segments_sorted) - 1):
        current_end = segments_sorted[i].end_seconds
        next_start = segments_sorted[i + 1].start_seconds
        assert current_end <= next_start + 0.1, \
            f"Segments overlap: {current_end} > {next_start}"
    
    print("[PASS] No segment overlap detected")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_segment_text_preview():
    """Test that segments have text previews."""
    segmenter = TemporalSegmenter()
    subtitles = create_long_subtitles(180.0)
    
    segments = segmenter.segment(subtitles, 180.0)
    
    for seg in segments:
        # Segments should have text preview set
        assert seg.text_preview is not None or seg.text_preview == ""
    
    print("[PASS] Segments have text previews")


def test_empty_subtitles():
    """Test handling of empty subtitles."""
    segmenter = TemporalSegmenter()
    empty_subtitles = SubtitleData(
        video_filename="empty.mp4",
        entries=[],
        source="test",
        language="en"
    )
    
    segments = segmenter.segment(empty_subtitles, 60.0)
    
    # Should handle gracefully
    assert segments is not None
    
    print("[PASS] Empty subtitles handled gracefully")


def test_short_video():
    """Test segmentation of video shorter than target duration."""
    segmenter = TemporalSegmenter()
    subtitles = SubtitleData(
        video_filename="short.mp4",
        entries=[
            SubtitleEntry(index=0, start_seconds=0.0, end_seconds=10.0, text="Short video content.")
        ],
        source="test",
        language="en"
    )
    
    # Video is 30 seconds, target is 65 seconds
    segments = segmenter.segment(subtitles, 30.0)
    
    # Should return at least something or empty list
    assert segments is not None
    
    print(f"[PASS] Short video handling: {len(segments)} segments")


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests():
    """Run all segmentation tests."""
    print("\n" + "="*60)
    print("TEMPORAL SEGMENTATION TESTS")
    print("="*60 + "\n")
    
    # Boundary detector tests
    print("\n--- Boundary Detector Tests ---")
    test_boundary_detector_creation()
    test_boundary_detection()
    test_boundary_sorting()
    test_top_boundaries_selection()
    
    # Strategy tests
    print("\n--- Segmentation Strategy Tests ---")
    test_pause_based_strategy()
    test_fixed_window_strategy()
    test_semantic_boundary_strategy()
    test_hybrid_strategy()
    
    # Segmenter tests
    print("\n--- Temporal Segmenter Tests ---")
    test_temporal_segmenter_creation()
    test_temporal_segmenter_with_strategy()
    test_temporal_segmenter_segment()
    test_segmenter_respects_config()
    test_boundary_detection_via_segmenter()
    
    # Validation tests
    print("\n--- Validation Tests ---")
    test_segment_validation()
    test_no_overlap()
    
    # Integration tests
    print("\n--- Integration Tests ---")
    test_segment_text_preview()
    test_empty_subtitles()
    test_short_video()
    
    print("\n" + "="*60)
    print("ALL SEGMENTATION TESTS PASSED")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()

