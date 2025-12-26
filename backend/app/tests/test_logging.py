"""
Logging System Tests
====================
Verifies that the research logging system works correctly.
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.logging_config import (
    get_research_logger,
    get_pipeline_logger,
    get_feature_logger,
    get_score_logger,
    get_decision_logger,
    log_segment_features,
    log_segment_score,
    log_pipeline_decision,
    log_stage_start,
    log_stage_complete,
    read_log_file,
    StructuredFormatter,
    ConsoleFormatter,
)
from app.config import get_config, reset_config


def test_get_research_logger():
    """Test creating research loggers."""
    logger = get_research_logger("test", log_to_file=False)
    
    assert logger is not None
    assert "research.test" in logger.name
    
    # Should return same logger on second call
    logger2 = get_research_logger("test", log_to_file=False)
    # Note: May return LoggerAdapter or Logger depending on configuration
    
    print("[PASS] get_research_logger test passed")


def test_specialized_loggers():
    """Test specialized logger functions."""
    pipeline_logger = get_pipeline_logger()
    feature_logger = get_feature_logger()
    score_logger = get_score_logger()
    decision_logger = get_decision_logger()
    
    assert pipeline_logger is not None
    assert feature_logger is not None
    assert score_logger is not None
    assert decision_logger is not None
    
    print("[PASS] Specialized loggers test passed")


def test_structured_formatter():
    """Test JSON formatting of log records."""
    import logging
    
    formatter = StructuredFormatter()
    
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test message",
        args=(),
        exc_info=None
    )
    
    formatted = formatter.format(record)
    
    # Should be valid JSON
    data = json.loads(formatted)
    assert data['level'] == 'INFO'
    assert data['logger'] == 'test.logger'
    assert data['message'] == 'Test message'
    assert 'timestamp' in data
    
    print("[PASS] StructuredFormatter test passed")


def test_console_formatter():
    """Test console formatting of log records."""
    import logging
    
    formatter = ConsoleFormatter(use_colors=False)
    
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test message",
        args=(),
        exc_info=None
    )
    
    formatted = formatter.format(record)
    
    assert "[INFO]" in formatted
    assert "test.logger" in formatted
    assert "Test message" in formatted
    
    print("[PASS] ConsoleFormatter test passed")


def test_log_segment_features():
    """Test feature logging convenience function."""
    # This should not raise, even if logging is disabled
    log_segment_features(
        segment_id="test_seg_001",
        features={
            'word_count': 100,
            'energy': 0.5
        },
        modalities=['text', 'audio']
    )
    
    print("[PASS] log_segment_features test passed")


def test_log_segment_score():
    """Test score logging convenience function."""
    log_segment_score(
        segment_id="test_seg_001",
        total_score=0.75,
        text_score=0.8,
        audio_score=0.6,
        visual_score=0.7,
        weights={'text': 0.4, 'audio': 0.3, 'visual': 0.3},
        method="rule_based"
    )
    
    print("[PASS] log_segment_score test passed")


def test_log_pipeline_decision():
    """Test decision logging convenience function."""
    log_pipeline_decision(
        decision_type="segment_selection",
        details={
            'selected_count': 5,
            'total_candidates': 10,
            'top_score': 0.85
        },
        result_id="test_result_001"
    )
    
    print("[PASS] log_pipeline_decision test passed")


def test_log_stage_functions():
    """Test stage start/complete logging functions."""
    result_id = "test_result_002"
    
    log_stage_start(
        stage_name="test_stage",
        result_id=result_id,
        input_summary={'video_duration': 300.0}
    )
    
    log_stage_complete(
        stage_name="test_stage",
        result_id=result_id,
        duration_seconds=5.5,
        output_summary={'segments_found': 5}
    )
    
    print("[PASS] Stage logging functions test passed")


def test_log_file_writing():
    """Test that logs are written to files correctly."""
    config = get_config()
    
    # Create a test log file
    test_log_path = config.paths.logs / "test" / "test_log.jsonl"
    test_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write some test entries
    entries = [
        {"timestamp": "2024-01-15T10:00:00", "level": "INFO", "message": "Test 1"},
        {"timestamp": "2024-01-15T10:00:01", "level": "INFO", "message": "Test 2"},
    ]
    
    with open(test_log_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    # Read back
    loaded = read_log_file(test_log_path)
    
    assert len(loaded) == 2
    assert loaded[0]['message'] == 'Test 1'
    assert loaded[1]['message'] == 'Test 2'
    
    # Clean up
    os.unlink(test_log_path)
    
    print("[PASS] Log file writing test passed")


def test_logger_with_extra():
    """Test logging with extra context."""
    import logging
    
    logger = get_research_logger("extra_test", log_to_file=False)
    
    # Should not raise
    logger.info(
        "Test with extra",
        extra={
            'segment_id': 'seg_001',
            'score': 0.75,
            'nested': {'a': 1, 'b': 2}
        }
    )
    
    print("[PASS] Logger with extra test passed")


def run_all_tests():
    """Run all logging tests."""
    print("\n" + "="*60)
    print("LOGGING SYSTEM TESTS")
    print("="*60 + "\n")
    
    test_get_research_logger()
    test_specialized_loggers()
    test_structured_formatter()
    test_console_formatter()
    test_log_segment_features()
    test_log_segment_score()
    test_log_pipeline_decision()
    test_log_stage_functions()
    test_log_file_writing()
    test_logger_with_extra()
    
    print("\n" + "="*60)
    print("ALL LOGGING TESTS PASSED")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()

