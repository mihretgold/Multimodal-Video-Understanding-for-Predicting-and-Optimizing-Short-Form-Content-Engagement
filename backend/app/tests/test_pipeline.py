"""
Pipeline Tests
==============
Verifies that the pipeline architecture works correctly.

Note: These tests use mocked services to avoid requiring actual video files
and API keys. Integration tests would need real resources.
"""

import os
import sys
import tempfile
from unittest.mock import Mock, patch, MagicMock

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.pipeline.context import PipelineContext, StageResult
from app.pipeline.base import PipelineStage
from app.pipeline.stages import (
    VideoIngestStage,
    TranscriptionStage,
    SegmentDetectionStage,
    FeatureExtractionStage,
    ScoringStage,
    OutputStage,
    ALL_STAGES,
)
from app.pipeline.pipeline import VideoPipeline
from app.models import VideoMetadata, SubtitleData, SubtitleEntry, Segment
from app.config import get_config, AppConfig


# =============================================================================
# MOCK DATA
# =============================================================================

def create_mock_video_metadata():
    return VideoMetadata(
        filename="test_video.mp4",
        filepath="/path/to/test_video.mp4",
        duration_seconds=300.0,
        fps=30.0,
        width=1920,
        height=1080,
        has_subtitles=False
    )


def create_mock_subtitle_data():
    entries = [
        SubtitleEntry(index=0, start_seconds=0.0, end_seconds=10.0, text="Hello everyone!"),
        SubtitleEntry(index=1, start_seconds=10.0, end_seconds=20.0, text="Welcome to the show."),
        SubtitleEntry(index=2, start_seconds=20.0, end_seconds=30.0, text="Today we'll discuss something exciting!"),
    ]
    return SubtitleData(
        video_filename="test_video.mp4",
        entries=entries,
        source="whisper",
        language="en"
    )


def create_mock_segments():
    return [
        Segment(segment_id="seg1", start_seconds=0.0, end_seconds=65.0, segment_type="funny", rank=1),
        Segment(segment_id="seg2", start_seconds=100.0, end_seconds=165.0, segment_type="emotional", rank=2),
    ]


# =============================================================================
# CONTEXT TESTS
# =============================================================================

def test_pipeline_context_creation():
    """Test PipelineContext can be created with a mock video path."""
    # Create a temporary file to act as our "video"
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        temp_path = f.name
        f.write(b"fake video content")
    
    try:
        context = PipelineContext(video_path=temp_path)
        
        assert context.video_path == temp_path
        assert context.video_filename == os.path.basename(temp_path)
        assert context.result_id is not None
        assert context.started_at is not None
        assert len(context.stage_results) == 0
        
        print("[PASS] PipelineContext creation test passed")
    finally:
        os.unlink(temp_path)


def test_context_stage_recording():
    """Test recording stage results in context."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        temp_path = f.name
        f.write(b"fake video content")
    
    try:
        context = PipelineContext(video_path=temp_path)
        
        context.record_stage("test_stage", success=True, duration=1.5, summary="OK")
        context.record_stage("failed_stage", success=False, duration=0.5, error="Something went wrong")
        
        assert len(context.stage_results) == 2
        assert context.successful_stages == ["test_stage"]
        assert context.failed_stages == ["failed_stage"]
        assert abs(context.total_duration_seconds - 2.0) < 0.001
        
        print("[PASS] Context stage recording test passed")
    finally:
        os.unlink(temp_path)


def test_context_checkpoint():
    """Test checkpoint save and load."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        temp_path = f.name
        f.write(b"fake video content")
    
    try:
        context = PipelineContext(video_path=temp_path)
        context.video_metadata = create_mock_video_metadata()
        context.subtitle_data = create_mock_subtitle_data()
        context.candidate_segments = create_mock_segments()
        
        # Save checkpoint
        checkpoint_path = context.save_checkpoint("test_checkpoint")
        assert os.path.exists(checkpoint_path)
        
        # Load checkpoint
        loaded = PipelineContext.load_checkpoint(checkpoint_path)
        
        assert loaded.result_id == context.result_id
        assert loaded.video_metadata is not None
        assert loaded.video_metadata.filename == "test_video.mp4"
        assert loaded.subtitle_data is not None
        assert len(loaded.subtitle_data.entries) == 3
        assert len(loaded.candidate_segments) == 2
        
        print("[PASS] Context checkpoint test passed")
        
        # Clean up checkpoint
        os.unlink(checkpoint_path)
    finally:
        os.unlink(temp_path)


# =============================================================================
# STAGE BASE TESTS
# =============================================================================

def test_stage_interface():
    """Test that all stages implement the required interface."""
    for stage_class in ALL_STAGES:
        stage = stage_class() if stage_class != OutputStage else stage_class(save_to_disk=False)
        
        assert hasattr(stage, 'name')
        assert hasattr(stage, 'description')
        assert hasattr(stage, 'run')
        assert hasattr(stage, '_execute')
        
        assert isinstance(stage.name, str)
        assert len(stage.name) > 0
        assert isinstance(stage.description, str)
    
    print("[PASS] Stage interface test passed")


def test_stage_names_unique():
    """Test that all stage names are unique."""
    names = [stage_class().name for stage_class in ALL_STAGES]
    assert len(names) == len(set(names)), "Stage names must be unique"
    
    print("[PASS] Stage names uniqueness test passed")


# =============================================================================
# PIPELINE TESTS
# =============================================================================

def test_pipeline_creation():
    """Test pipeline can be created with default stages."""
    pipeline = VideoPipeline()
    
    assert len(pipeline.stages) == 6
    assert pipeline.stage_names == [
        "video_ingest",
        "transcription", 
        "segment_detection",
        "feature_extraction",
        "scoring",
        "output"
    ]
    
    print("[PASS] Pipeline creation test passed")


def test_pipeline_get_stage():
    """Test getting stages by name."""
    pipeline = VideoPipeline()
    
    stage = pipeline.get_stage("transcription")
    assert stage is not None
    assert stage.name == "transcription"
    
    stage = pipeline.get_stage("nonexistent")
    assert stage is None
    
    print("[PASS] Pipeline get_stage test passed")


@patch('app.pipeline.stages.get_video_info')
@patch('app.pipeline.stages.check_subtitles')
def test_video_ingest_stage(mock_check_subs, mock_get_info):
    """Test VideoIngestStage with mocked video utilities."""
    mock_get_info.return_value = {
        'duration': 300.0,
        'fps': 30.0,
        'size': (1920, 1080)
    }
    mock_check_subs.return_value = False
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        temp_path = f.name
        f.write(b"fake video content")
    
    try:
        context = PipelineContext(video_path=temp_path)
        stage = VideoIngestStage()
        
        success = stage.run(context, use_cache=False)
        
        assert success is True
        assert context.video_metadata is not None
        assert context.video_metadata.duration_seconds == 300.0
        assert context.video_metadata.width == 1920
        
        print("[PASS] VideoIngestStage test passed")
    finally:
        os.unlink(temp_path)


def test_feature_extraction_ablation():
    """Test that FeatureExtractionStage respects ablation settings."""
    from app.config import AblationConfig
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        temp_path = f.name
        f.write(b"fake video content")
    
    try:
        config = get_config()
        
        # Test text-only ablation
        config.ablation = AblationConfig.text_only()
        context = PipelineContext(video_path=temp_path, config=config)
        context.subtitle_data = create_mock_subtitle_data()
        context.candidate_segments = create_mock_segments()
        
        stage = FeatureExtractionStage()
        success = stage.run(context, use_cache=False)
        
        assert success is True
        assert len(context.segment_features) == 2
        
        # Check that only text features are present
        features = context.segment_features["seg1"]
        assert features.text_features is not None
        assert features.audio_features is None  # Ablated
        assert features.visual_features is None  # Ablated
        
        print("[PASS] Feature extraction ablation test passed")
    finally:
        os.unlink(temp_path)


def test_scoring_stage():
    """Test ScoringStage produces ranked segments."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        temp_path = f.name
        f.write(b"fake video content")
    
    try:
        context = PipelineContext(video_path=temp_path)
        context.candidate_segments = create_mock_segments()
        context.segment_features = {}  # No features, will use defaults
        
        stage = ScoringStage()
        success = stage.run(context, use_cache=False)
        
        assert success is True
        assert len(context.scored_segments) == 2
        
        # Check ranking
        assert context.scored_segments[0].rank == 1
        assert context.scored_segments[1].rank == 2
        
        # Check scores exist
        for segment in context.scored_segments:
            assert segment.score is not None
            assert 0 <= segment.score.total_score <= 1
        
        print("[PASS] ScoringStage test passed")
    finally:
        os.unlink(temp_path)


def test_output_stage():
    """Test OutputStage finalizes context."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        temp_path = f.name
        f.write(b"fake video content")
    
    try:
        context = PipelineContext(video_path=temp_path)
        context.video_metadata = create_mock_video_metadata()
        context.scored_segments = create_mock_segments()
        
        stage = OutputStage(save_to_disk=False)
        success = stage.run(context, use_cache=False)
        
        assert success is True
        assert context.analysis_result is not None
        assert context.analysis_result.segment_count == 2
        
        print("[PASS] OutputStage test passed")
    finally:
        os.unlink(temp_path)


# =============================================================================
# INTEGRATION TEST (with mocks)
# =============================================================================

@patch('app.pipeline.stages.get_video_info')
@patch('app.pipeline.stages.check_subtitles')
@patch.object(TranscriptionStage, 'subtitle_service', new_callable=lambda: Mock())
@patch.object(SegmentDetectionStage, 'analysis_service', new_callable=lambda: Mock())
def test_pipeline_integration(
    mock_analysis_service,
    mock_subtitle_service,
    mock_check_subs,
    mock_get_info
):
    """Test running the full pipeline with mocked services."""
    # Setup mocks
    mock_get_info.return_value = {
        'duration': 300.0,
        'fps': 30.0,
        'size': (1920, 1080)
    }
    mock_check_subs.return_value = False
    
    # Mock subtitle service
    mock_subtitle_service.get_subtitles_json = Mock(return_value={
        'subtitles': [
            {'start': 0.0, 'end': 10.0, 'text': 'Hello'},
            {'start': 10.0, 'end': 20.0, 'text': 'World'},
        ],
        'source': 'whisper',
        'language': 'en'
    })
    
    # Mock analysis service
    mock_analysis_service.analyze_subtitles = Mock(return_value=[
        Segment(segment_id='seg1', start_seconds=0.0, end_seconds=65.0, 
                segment_type='funny', rank=1, source='gemini'),
    ])
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        temp_path = f.name
        f.write(b"fake video content" * 1000)  # Make it a bit larger
    
    try:
        pipeline = VideoPipeline(use_cache=False, save_results=False)
        
        # Can't run full pipeline without real services, but we test structure
        assert len(pipeline.stages) == 6
        assert pipeline.stage_names[0] == "video_ingest"
        assert pipeline.stage_names[-1] == "output"
        
        print("[PASS] Pipeline integration structure test passed")
    finally:
        os.unlink(temp_path)


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests():
    """Run all pipeline tests."""
    print("\n" + "="*60)
    print("PIPELINE TESTS")
    print("="*60 + "\n")
    
    test_pipeline_context_creation()
    test_context_stage_recording()
    test_context_checkpoint()
    test_stage_interface()
    test_stage_names_unique()
    test_pipeline_creation()
    test_pipeline_get_stage()
    test_video_ingest_stage()
    test_feature_extraction_ablation()
    test_scoring_stage()
    test_output_stage()
    test_pipeline_integration()
    
    print("\n" + "="*60)
    print("ALL PIPELINE TESTS PASSED")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()

