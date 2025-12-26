"""
Data Models Tests
=================
Verifies that all data models serialize, deserialize, and validate correctly.
"""

import os
import sys
import tempfile
import json

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.models import (
    VideoMetadata,
    SubtitleEntry,
    SubtitleData,
    TextFeatures,
    AudioFeatures,
    VisualFeatures,
    SegmentFeatures,
    ScoreBreakdown,
    Segment,
    AnalysisResult,
    GroundTruthSegment,
    EvaluationResult,
    SegmentType,
    generate_segment_id,
    generate_result_id,
)


def test_video_metadata():
    """Test VideoMetadata creation and serialization."""
    metadata = VideoMetadata(
        filename="test.mp4",
        filepath="/path/to/test.mp4",
        duration_seconds=120.5,
        fps=30.0,
        width=1920,
        height=1080
    )
    
    assert metadata.duration_formatted == "00:02:00"
    assert metadata.resolution == "1920x1080"
    
    # Test serialization
    data = metadata.to_dict()
    assert data['filename'] == "test.mp4"
    assert data['duration_seconds'] == 120.5
    
    # Test JSON round-trip
    json_str = metadata.to_json()
    loaded = VideoMetadata.from_json(json_str)
    assert loaded.filename == metadata.filename
    
    print("[PASS] VideoMetadata test passed")


def test_subtitle_entry():
    """Test SubtitleEntry creation and properties."""
    entry = SubtitleEntry(
        index=0,
        start_seconds=10.0,
        end_seconds=15.0,
        text="Hello world, how are you?"
    )
    
    assert entry.duration_seconds == 5.0
    assert entry.word_count == 5
    assert len(entry.words) == 5
    
    print("[PASS] SubtitleEntry test passed")


def test_subtitle_data():
    """Test SubtitleData creation and methods."""
    entries = [
        SubtitleEntry(index=0, start_seconds=0.0, end_seconds=5.0, text="First subtitle"),
        SubtitleEntry(index=1, start_seconds=5.0, end_seconds=10.0, text="Second subtitle"),
        SubtitleEntry(index=2, start_seconds=10.0, end_seconds=15.0, text="Third subtitle"),
    ]
    
    subtitle_data = SubtitleData(
        video_filename="test.mp4",
        entries=entries,
        source="whisper"
    )
    
    assert subtitle_data.total_duration_seconds == 15.0
    assert subtitle_data.word_count == 6  # 2 + 2 + 2
    
    # Test get_text_in_range
    text = subtitle_data.get_text_in_range(3.0, 12.0)
    assert "First" in text
    assert "Second" in text
    assert "Third" in text
    
    # Test serialization round-trip
    data = subtitle_data.to_dict()
    loaded = SubtitleData.from_dict(data)
    assert len(loaded.entries) == 3
    assert loaded.source == "whisper"
    
    print("[PASS] SubtitleData test passed")


def test_subtitle_data_from_list():
    """Test creating SubtitleData from raw API format."""
    raw_subtitles = [
        {'start': 0.0, 'end': 5.0, 'text': 'Hello'},
        {'start': 5.0, 'end': 10.0, 'text': 'World'},
    ]
    
    subtitle_data = SubtitleData.from_subtitle_list(
        video_filename="test.mp4",
        subtitles=raw_subtitles,
        source="embedded"
    )
    
    assert len(subtitle_data.entries) == 2
    assert subtitle_data.entries[0].text == "Hello"
    assert subtitle_data.source == "embedded"
    
    print("[PASS] SubtitleData from_subtitle_list test passed")


def test_feature_classes():
    """Test feature dataclasses."""
    text_features = TextFeatures(
        word_count=100,
        sentence_count=10,
        sentiment_score=0.5
    )
    
    audio_features = AudioFeatures(
        energy_mean=0.7,
        silence_ratio=0.2,
        speech_rate=2.5
    )
    
    visual_features = VisualFeatures(
        motion_intensity=0.8,
        scene_change_count=5,
        brightness_mean=0.6
    )
    
    assert text_features.word_count == 100
    assert audio_features.silence_ratio == 0.2
    assert visual_features.scene_change_count == 5
    
    print("[PASS] Feature classes test passed")


def test_segment_features():
    """Test SegmentFeatures with nested feature objects."""
    features = SegmentFeatures(
        segment_id="test_001",
        start_seconds=10.0,
        end_seconds=70.0,
        text_features=TextFeatures(word_count=50),
        audio_features=AudioFeatures(energy_mean=0.5),
        visual_features=None  # Simulating ablation
    )
    
    assert features.duration_seconds == 60.0
    assert features.has_text is True
    assert features.has_audio is True
    assert features.has_visual is False
    assert features.modalities_present == ["text", "audio"]
    
    # Test serialization
    data = features.to_dict()
    assert data['text_features']['word_count'] == 50
    assert data['visual_features'] is None
    
    # Test round-trip
    loaded = SegmentFeatures.from_dict(data)
    assert loaded.has_text is True
    assert loaded.has_visual is False
    
    print("[PASS] SegmentFeatures test passed")


def test_score_breakdown():
    """Test ScoreBreakdown properties."""
    score = ScoreBreakdown(
        total_score=0.75,
        text_score=0.8,
        audio_score=0.6,
        visual_score=0.7,
        text_weight=0.4,
        audio_weight=0.3,
        visual_weight=0.3
    )
    
    # Use approximate comparison for floating point
    assert abs(score.weighted_text - 0.32) < 0.001  # 0.8 * 0.4
    assert abs(score.weighted_audio - 0.18) < 0.001  # 0.6 * 0.3
    assert abs(score.weighted_visual - 0.21) < 0.001  # 0.7 * 0.3
    
    print("[PASS] ScoreBreakdown test passed")


def test_segment():
    """Test Segment creation and methods."""
    segment = Segment(
        segment_id="seg_001",
        start_seconds=30.0,
        end_seconds=95.0,
        segment_type="funny",
        confidence=0.85,
        rank=1
    )
    
    assert segment.duration_seconds == 65.0
    assert segment.duration_formatted == "01:05"
    assert segment.time_range_formatted == "00:30 - 01:35"
    
    # Test from_gemini_response
    gemini_response = {'start': 100, 'end': 165, 'type': 'emotional'}
    segment2 = Segment.from_gemini_response(gemini_response, "seg_002")
    
    assert segment2.start_seconds == 100.0
    assert segment2.end_seconds == 165.0
    assert segment2.segment_type == "emotional"
    assert segment2.source == "gemini"
    
    print("[PASS] Segment test passed")


def test_analysis_result():
    """Test AnalysisResult creation and methods."""
    metadata = VideoMetadata(
        filename="test.mp4",
        filepath="/path/test.mp4",
        duration_seconds=300.0,
        fps=30.0,
        width=1920,
        height=1080
    )
    
    segments = [
        Segment(segment_id="s1", start_seconds=10, end_seconds=70, segment_type="funny", rank=2),
        Segment(segment_id="s2", start_seconds=100, end_seconds=160, segment_type="emotional", rank=1),
        Segment(segment_id="s3", start_seconds=200, end_seconds=260, segment_type="funny", rank=3),
    ]
    
    result = AnalysisResult(
        result_id="result_001",
        video_metadata=metadata,
        segments=segments,
        processing_time_seconds=45.5
    )
    
    assert result.segment_count == 3
    assert result.top_segment.segment_id == "s2"  # rank=1
    assert len(result.get_segments_by_type("funny")) == 2
    assert len(result.get_top_n_segments(2)) == 2
    
    # Test serialization
    data = result.to_dict()
    # segment_count is a property, not in to_dict
    assert len(data['segments']) == 3
    
    # Test round-trip
    loaded = AnalysisResult.from_dict(data)
    assert loaded.segment_count == 3
    assert loaded.video_metadata.filename == "test.mp4"
    
    print("[PASS] AnalysisResult test passed")


def test_analysis_result_file_io():
    """Test saving and loading AnalysisResult to file."""
    metadata = VideoMetadata(
        filename="test.mp4",
        filepath="/path/test.mp4",
        duration_seconds=120.0,
        fps=24.0,
        width=1280,
        height=720
    )
    
    result = AnalysisResult(
        result_id="io_test",
        video_metadata=metadata,
        segments=[
            Segment(segment_id="s1", start_seconds=0, end_seconds=60, rank=1)
        ]
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        result.save(temp_path)
        loaded = AnalysisResult.load(temp_path)
        
        assert loaded.result_id == "io_test"
        assert loaded.segment_count == 1
        assert loaded.video_metadata.fps == 24.0
        
        print("[PASS] AnalysisResult file I/O test passed")
    finally:
        os.unlink(temp_path)


def test_segment_id_generation():
    """Test unique ID generation functions."""
    id1 = generate_segment_id("video.mp4", 10.0, 70.0)
    id2 = generate_segment_id("video.mp4", 10.0, 70.0)
    id3 = generate_segment_id("video.mp4", 10.0, 71.0)
    
    assert id1 == id2  # Same inputs = same ID
    assert id1 != id3  # Different inputs = different ID
    assert len(id1) == 12
    
    result_id = generate_result_id("video.mp4")
    assert len(result_id) == 16
    
    print("[PASS] ID generation test passed")


def test_segment_type_enum():
    """Test SegmentType enum."""
    assert SegmentType.FUNNY.value == "funny"
    assert SegmentType.EMOTIONAL.value == "emotional"
    assert SegmentType("funny") == SegmentType.FUNNY
    
    print("[PASS] SegmentType enum test passed")


def run_all_tests():
    """Run all model tests."""
    print("\n" + "="*60)
    print("DATA MODELS TESTS")
    print("="*60 + "\n")
    
    test_video_metadata()
    test_subtitle_entry()
    test_subtitle_data()
    test_subtitle_data_from_list()
    test_feature_classes()
    test_segment_features()
    test_score_breakdown()
    test_segment()
    test_analysis_result()
    test_analysis_result_file_io()
    test_segment_id_generation()
    test_segment_type_enum()
    
    print("\n" + "="*60)
    print("ALL DATA MODEL TESTS PASSED")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()

