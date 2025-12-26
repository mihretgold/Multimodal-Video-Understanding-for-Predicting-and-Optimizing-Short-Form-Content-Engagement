"""
Baseline System Tests
=====================
Verifies that the baseline formalization works correctly.
"""

import os
import sys
import tempfile
import json

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.baseline.specification import (
    BaselineSpec,
    InputSpec,
    IntermediateSpec,
    OutputSpec,
    BaselineOutput,
)
from app.baseline.runner import BaselineRunner
from app.models import (
    VideoMetadata,
    SubtitleData,
    SubtitleEntry,
    Segment,
    AnalysisResult,
    ScoreBreakdown,
)
from app.config import get_config


# =============================================================================
# SPEC TESTS
# =============================================================================

def test_input_spec_validation():
    """Test input specification validation."""
    spec = InputSpec()
    
    # Test with non-existent file
    valid, error = spec.validate("/nonexistent/video.mp4")
    assert valid is False
    assert "not found" in error.lower()
    
    # Test with existing file (create temp)
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(b"fake video content")
        temp_path = f.name
    
    try:
        valid, error = spec.validate(temp_path)
        assert valid is True
        assert error == ""
        
        # Test with wrong extension
        wrong_ext = temp_path.replace('.mp4', '.txt')
        os.rename(temp_path, wrong_ext)
        valid, error = spec.validate(wrong_ext)
        assert valid is False
        assert "invalid format" in error.lower()
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        if os.path.exists(wrong_ext):
            os.unlink(wrong_ext)
    
    print("[PASS] Input spec validation test passed")


def test_intermediate_spec_video_metadata():
    """Test intermediate spec validation for VideoMetadata."""
    spec = IntermediateSpec()
    
    # Valid metadata
    metadata = VideoMetadata(
        filename="test.mp4",
        filepath="/path/test.mp4",
        duration_seconds=300.0,
        fps=30.0,
        width=1920,
        height=1080
    )
    
    valid, error = spec.validate_video_metadata(metadata)
    assert valid is True
    
    # None metadata
    valid, error = spec.validate_video_metadata(None)
    assert valid is False
    
    print("[PASS] Intermediate spec video metadata test passed")


def test_intermediate_spec_subtitle_data():
    """Test intermediate spec validation for SubtitleData."""
    spec = IntermediateSpec()
    
    # Valid subtitle data
    entries = [
        SubtitleEntry(index=0, start_seconds=0.0, end_seconds=5.0, text="Hello")
    ]
    subtitle_data = SubtitleData(
        video_filename="test.mp4",
        entries=entries,
        source="whisper",
        language="en"
    )
    
    valid, error = spec.validate_subtitle_data(subtitle_data)
    assert valid is True
    
    # Empty entries
    empty_data = SubtitleData(
        video_filename="test.mp4",
        entries=[],
        source="whisper",
        language="en"
    )
    valid, error = spec.validate_subtitle_data(empty_data)
    assert valid is False
    assert "too few" in error.lower()
    
    print("[PASS] Intermediate spec subtitle data test passed")


def test_intermediate_spec_segments():
    """Test intermediate spec validation for segments."""
    spec = IntermediateSpec()
    
    # Valid segments
    segments = [
        Segment(
            segment_id="seg1",
            start_seconds=0.0,
            end_seconds=65.0,
            segment_type="funny"
        )
    ]
    
    valid, error = spec.validate_segments(segments)
    assert valid is True
    
    # Empty segments
    valid, error = spec.validate_segments([])
    assert valid is False
    assert "no segments" in error.lower()
    
    # Invalid segment (end before start)
    invalid_segments = [
        Segment(
            segment_id="bad",
            start_seconds=100.0,
            end_seconds=50.0,
            segment_type="funny"
        )
    ]
    valid, error = spec.validate_segments(invalid_segments)
    assert valid is False
    assert "invalid duration" in error.lower()
    
    print("[PASS] Intermediate spec segments test passed")


def test_output_spec_validation():
    """Test output specification validation."""
    spec = OutputSpec()
    
    # Valid result
    metadata = VideoMetadata(
        filename="test.mp4",
        filepath="/path/test.mp4",
        duration_seconds=300.0,
        fps=30.0,
        width=1920,
        height=1080
    )
    
    segments = [
        Segment(
            segment_id="seg1",
            start_seconds=0.0,
            end_seconds=65.0,
            segment_type="funny"
        )
    ]
    
    result = AnalysisResult(
        result_id="test_result_12345678",
        video_metadata=metadata,
        segments=segments,
        processing_time_seconds=30.0,
        ablation_mode="full_multimodal"
    )
    
    valid, error = spec.validate(result)
    assert valid is True
    
    # Invalid result (None)
    valid, error = spec.validate(None)
    assert valid is False
    
    print("[PASS] Output spec validation test passed")


def test_baseline_spec_to_dict():
    """Test baseline spec export to dictionary."""
    spec = BaselineSpec()
    data = spec.to_dict()
    
    assert 'baseline_name' in data
    assert 'baseline_version' in data
    assert 'input_spec' in data
    assert 'intermediate_spec' in data
    assert 'output_spec' in data
    assert 'defaults' in data
    
    print("[PASS] Baseline spec to_dict test passed")


def test_baseline_spec_save_load():
    """Test baseline spec save and load."""
    spec = BaselineSpec()
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        spec.save(temp_path)
        
        # Load and verify
        with open(temp_path, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data['baseline_name'] == spec.baseline_name
        assert loaded_data['baseline_version'] == spec.baseline_version
        
        print("[PASS] Baseline spec save/load test passed")
    finally:
        os.unlink(temp_path)


# =============================================================================
# BASELINE OUTPUT TESTS
# =============================================================================

def test_baseline_output_creation():
    """Test creating a BaselineOutput."""
    metadata = VideoMetadata(
        filename="test.mp4",
        filepath="/path/test.mp4",
        duration_seconds=300.0,
        fps=30.0,
        width=1920,
        height=1080
    )
    
    result = AnalysisResult(
        result_id="test_result_001",
        video_metadata=metadata,
        segments=[],
        processing_time_seconds=30.0
    )
    
    output = BaselineOutput(
        result=result,
        is_valid=True,
        validation_errors=[],
        spec_version="1.0.0",
        config_snapshot={'test': 'config'}
    )
    
    assert output.is_valid is True
    assert output.result.result_id == "test_result_001"
    assert output.spec_version == "1.0.0"
    
    print("[PASS] BaselineOutput creation test passed")


def test_baseline_output_serialization():
    """Test BaselineOutput save and load."""
    metadata = VideoMetadata(
        filename="test.mp4",
        filepath="/path/test.mp4",
        duration_seconds=300.0,
        fps=30.0,
        width=1920,
        height=1080
    )
    
    result = AnalysisResult(
        result_id="test_result_002",
        video_metadata=metadata,
        segments=[
            Segment(
                segment_id="seg1",
                start_seconds=0.0,
                end_seconds=65.0,
                segment_type="funny"
            )
        ],
        processing_time_seconds=30.0
    )
    
    output = BaselineOutput(
        result=result,
        is_valid=True,
        validation_errors=[],
        spec_version="1.0.0",
        config_snapshot={'experiment': 'test'}
    )
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        output.save(temp_path)
        
        loaded = BaselineOutput.load(temp_path)
        
        assert loaded.is_valid == output.is_valid
        assert loaded.result.result_id == output.result.result_id
        assert loaded.spec_version == output.spec_version
        assert len(loaded.result.segments) == 1
        
        print("[PASS] BaselineOutput serialization test passed")
    finally:
        os.unlink(temp_path)


# =============================================================================
# RUNNER TESTS
# =============================================================================

def test_baseline_runner_creation():
    """Test creating a BaselineRunner."""
    runner = BaselineRunner(use_cache=False, save_results=False)
    
    assert runner.spec is not None
    assert runner.config is not None
    assert runner.pipeline is not None
    assert len(runner.pipeline.stages) == 6
    
    print("[PASS] BaselineRunner creation test passed")


def test_baseline_runner_verify_result():
    """Test result verification."""
    runner = BaselineRunner()
    
    # Create valid result
    metadata = VideoMetadata(
        filename="test.mp4",
        filepath="/path/test.mp4",
        duration_seconds=300.0,
        fps=30.0,
        width=1920,
        height=1080
    )
    
    segments = [
        Segment(
            segment_id="seg1",
            start_seconds=0.0,
            end_seconds=65.0,
            segment_type="funny",
            score=ScoreBreakdown(total_score=0.8)
        )
    ]
    
    entries = [
        SubtitleEntry(index=0, start_seconds=0.0, end_seconds=5.0, text="Hello")
    ]
    
    subtitle_data = SubtitleData(
        video_filename="test.mp4",
        entries=entries,
        source="whisper",
        language="en"
    )
    
    result = AnalysisResult(
        result_id="verify_test_12345678",
        video_metadata=metadata,
        subtitle_data=subtitle_data,
        segments=segments,
        processing_time_seconds=30.0,
        ablation_mode="full_multimodal"
    )
    
    valid, errors = runner.verify_result(result)
    assert valid is True
    assert len(errors) == 0
    
    print("[PASS] BaselineRunner verify_result test passed")


def test_baseline_runner_invalid_input():
    """Test runner with invalid input."""
    runner = BaselineRunner(save_results=False)
    
    # Run with non-existent file
    output = runner.run("/nonexistent/video.mp4")
    
    assert output.is_valid is False
    assert len(output.validation_errors) > 0
    assert any("not found" in e.lower() for e in output.validation_errors)
    
    print("[PASS] BaselineRunner invalid input test passed")


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests():
    """Run all baseline tests."""
    print("\n" + "="*60)
    print("BASELINE SYSTEM TESTS")
    print("="*60 + "\n")
    
    # Spec tests
    test_input_spec_validation()
    test_intermediate_spec_video_metadata()
    test_intermediate_spec_subtitle_data()
    test_intermediate_spec_segments()
    test_output_spec_validation()
    test_baseline_spec_to_dict()
    test_baseline_spec_save_load()
    
    # Output tests
    test_baseline_output_creation()
    test_baseline_output_serialization()
    
    # Runner tests
    test_baseline_runner_creation()
    test_baseline_runner_verify_result()
    test_baseline_runner_invalid_input()
    
    print("\n" + "="*60)
    print("ALL BASELINE TESTS PASSED")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()

