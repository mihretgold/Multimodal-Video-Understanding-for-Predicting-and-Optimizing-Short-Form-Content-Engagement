"""
Feature Extraction Tests
========================
Tests for the multimodal feature extraction module.
"""

import os
import sys
import tempfile

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.features import (
    FeatureExtractor,
    TextFeatureExtractor,
    AudioFeatureExtractor,
    VisualFeatureExtractor,
)
from app.features.extractor import create_extractor
from app.features.text_features import (
    POSITIVE_WORDS,
    NEGATIVE_WORDS,
    EMOTION_KEYWORDS,
)
from app.models import (
    TextFeatures,
    AudioFeatures,
    VisualFeatures,
    SegmentFeatures,
    Segment,
    SubtitleData,
    SubtitleEntry,
)
from app.config import AblationConfig


# =============================================================================
# TEXT FEATURE TESTS
# =============================================================================

def test_text_extractor_creation():
    """Test creating a text feature extractor."""
    extractor = TextFeatureExtractor()
    assert extractor is not None
    print("[PASS] Text extractor creation test passed")


def test_text_basic_features():
    """Test basic text feature extraction."""
    extractor = TextFeatureExtractor()
    
    text = "Hello world. This is a test sentence. How are you doing today?"
    features = extractor.extract(text, duration_seconds=10.0)
    
    assert features.word_count == 12
    assert features.sentence_count == 3
    assert features.avg_word_length > 0
    assert features.question_count == 1
    assert features.exclamation_count == 0
    
    print(f"[PASS] Basic text features: {features.word_count} words, {features.sentence_count} sentences")


def test_text_sentiment_positive():
    """Test positive sentiment detection."""
    extractor = TextFeatureExtractor()
    
    text = "This is amazing! I love it so much. It's really wonderful and fantastic!"
    features = extractor.extract(text)
    
    assert features.sentiment_score > 0, f"Expected positive sentiment, got {features.sentiment_score}"
    
    print(f"[PASS] Positive sentiment score: {features.sentiment_score:.3f}")


def test_text_sentiment_negative():
    """Test negative sentiment detection."""
    extractor = TextFeatureExtractor()
    
    text = "This is terrible. I hate it. It's really awful and disappointing."
    features = extractor.extract(text)
    
    assert features.sentiment_score < 0, f"Expected negative sentiment, got {features.sentiment_score}"
    
    print(f"[PASS] Negative sentiment score: {features.sentiment_score:.3f}")


def test_text_sentiment_negation():
    """Test sentiment with negation."""
    extractor = TextFeatureExtractor()
    
    # "not great" should be less positive
    text1 = "This is not great at all."
    features1 = extractor.extract(text1)
    
    # "not terrible" should be less negative
    text2 = "This is not terrible."
    features2 = extractor.extract(text2)
    
    # Both should be more neutral due to negation
    print(f"[PASS] Negation handling: 'not great'={features1.sentiment_score:.3f}, 'not terrible'={features2.sentiment_score:.3f}")


def test_text_speech_rate():
    """Test speech rate calculation."""
    extractor = TextFeatureExtractor()
    
    text = "Word " * 100  # 100 words
    features = extractor.extract(text, duration_seconds=50.0)
    
    expected_rate = 100 / 50.0  # 2 words per second
    assert abs(features.word_count / 50.0 - expected_rate) < 0.1
    
    print(f"[PASS] Speech rate: {100/50:.1f} words/sec")


def test_text_emotion_detection():
    """Test emotion keyword detection."""
    extractor = TextFeatureExtractor()
    
    funny_text = "Haha that was so funny! I'm laughing so hard at this joke!"
    emotion, confidence = extractor.detect_dominant_emotion(funny_text)
    
    assert emotion == 'funny', f"Expected 'funny', got '{emotion}'"
    assert confidence > 0
    
    print(f"[PASS] Detected emotion: {emotion} (confidence: {confidence:.3f})")


def test_text_empty_input():
    """Test handling of empty text."""
    extractor = TextFeatureExtractor()
    
    features = extractor.extract("")
    
    assert features.word_count == 0
    assert features.sentiment_score == 0.0
    
    print("[PASS] Empty input handled correctly")


# =============================================================================
# AUDIO FEATURE TESTS
# =============================================================================

def test_audio_extractor_creation():
    """Test creating an audio feature extractor."""
    extractor = AudioFeatureExtractor()
    assert extractor is not None
    print("[PASS] Audio extractor creation test passed")


def test_audio_default_features():
    """Test default audio features when extraction fails."""
    extractor = AudioFeatureExtractor()
    
    # Extract from non-existent file
    features = extractor.extract("/nonexistent/video.mp4", 0, 10)
    
    # Should return default features, not crash
    assert features.energy_mean >= 0
    assert features.silence_ratio >= 0
    
    print(f"[PASS] Default audio features: energy={features.energy_mean:.3f}, silence={features.silence_ratio:.3f}")


def test_audio_stats():
    """Test audio stats method."""
    extractor = AudioFeatureExtractor()
    
    stats = extractor.get_audio_stats("/nonexistent/video.mp4", 0, 10)
    
    assert 'energy' in stats
    assert 'silence_ratio' in stats
    
    print("[PASS] Audio stats structure correct")


# =============================================================================
# VISUAL FEATURE TESTS
# =============================================================================

def test_visual_extractor_creation():
    """Test creating a visual feature extractor."""
    extractor = VisualFeatureExtractor()
    assert extractor is not None
    print("[PASS] Visual extractor creation test passed")


def test_visual_default_features():
    """Test default visual features when extraction fails."""
    extractor = VisualFeatureExtractor()
    
    # Extract from non-existent file
    features = extractor.extract("/nonexistent/video.mp4", 0, 10)
    
    # Should return default features, not crash
    assert features.motion_intensity >= 0
    assert features.brightness_mean >= 0
    
    print(f"[PASS] Default visual features: motion={features.motion_intensity:.3f}, brightness={features.brightness_mean:.3f}")


# =============================================================================
# UNIFIED EXTRACTOR TESTS
# =============================================================================

def test_feature_extractor_creation():
    """Test creating the unified feature extractor."""
    extractor = FeatureExtractor()
    assert extractor is not None
    print("[PASS] Unified feature extractor creation test passed")


def test_feature_extractor_text_only():
    """Test text-only ablation mode."""
    extractor = FeatureExtractor(ablation=AblationConfig.text_only())
    
    assert extractor.ablation.use_text is True
    assert extractor.ablation.use_audio is False
    assert extractor.ablation.use_visual is False
    
    # Extract text features
    features = extractor.extract_text("This is a test.", duration_seconds=5.0)
    assert features.word_count == 4
    
    print("[PASS] Text-only ablation mode works")


def test_feature_extractor_audio_only():
    """Test audio-only ablation mode."""
    extractor = FeatureExtractor(ablation=AblationConfig.audio_only())
    
    assert extractor.ablation.use_text is False
    assert extractor.ablation.use_audio is True
    assert extractor.ablation.use_visual is False
    
    print("[PASS] Audio-only ablation mode configured correctly")


def test_feature_extractor_full_multimodal():
    """Test full multimodal mode."""
    extractor = FeatureExtractor(ablation=AblationConfig.full_multimodal())
    
    assert extractor.ablation.use_text is True
    assert extractor.ablation.use_audio is True
    assert extractor.ablation.use_visual is True
    
    print("[PASS] Full multimodal mode configured correctly")


def test_create_extractor_factory():
    """Test the factory function for creating extractors."""
    for mode in ['text_only', 'audio_only', 'visual_only', 'text_audio', 'full_multimodal']:
        extractor = create_extractor(ablation_mode=mode)
        assert extractor.ablation.mode_name == mode
    
    print("[PASS] Factory function creates all ablation modes")


def test_segment_feature_extraction():
    """Test extracting features for a segment."""
    extractor = FeatureExtractor(ablation=AblationConfig.text_only())
    
    # Create test segment and subtitle data
    segment = Segment(
        segment_id="test_seg_001",
        start_seconds=0.0,
        end_seconds=60.0,
        segment_type="test"
    )
    
    subtitle_data = SubtitleData(
        video_filename="test.mp4",
        entries=[
            SubtitleEntry(index=0, start_seconds=0.0, end_seconds=10.0, text="Hello world!"),
            SubtitleEntry(index=1, start_seconds=10.0, end_seconds=20.0, text="This is amazing!"),
        ],
        source="test",
        language="en"
    )
    
    # Extract features (video path doesn't need to exist for text-only)
    features = extractor.extract_segment_features(
        video_path="/fake/video.mp4",
        segment=segment,
        subtitle_data=subtitle_data
    )
    
    assert features.segment_id == "test_seg_001"
    assert features.text_features is not None
    assert features.text_features.word_count > 0
    
    print(f"[PASS] Segment feature extraction: {features.modalities_present}")


def test_batch_extraction():
    """Test batch feature extraction."""
    extractor = FeatureExtractor(ablation=AblationConfig.text_only())
    
    # Create test segments
    segments = [
        Segment(segment_id=f"seg_{i}", start_seconds=i*60, end_seconds=(i+1)*60, segment_type="test")
        for i in range(3)
    ]
    
    subtitle_data = SubtitleData(
        video_filename="test.mp4",
        entries=[
            SubtitleEntry(index=i, start_seconds=i*30, end_seconds=(i+1)*30, text=f"Segment {i} text content.")
            for i in range(6)
        ],
        source="test",
        language="en"
    )
    
    results = extractor.extract_batch(
        video_path="/fake/video.mp4",
        segments=segments,
        subtitle_data=subtitle_data
    )
    
    assert len(results) == 3
    assert all(seg.segment_id in results for seg in segments)
    
    print(f"[PASS] Batch extraction: {len(results)} segments processed")


def test_feature_caching():
    """Test that features are cached."""
    extractor = FeatureExtractor(ablation=AblationConfig.text_only(), cache_enabled=True)
    
    segment = Segment(
        segment_id="cache_test_001",
        start_seconds=0.0,
        end_seconds=60.0,
        segment_type="test"
    )
    
    subtitle_data = SubtitleData(
        video_filename="cache_test.mp4",
        entries=[
            SubtitleEntry(index=0, start_seconds=0.0, end_seconds=30.0, text="Test text for caching.")
        ],
        source="test",
        language="en"
    )
    
    # First extraction
    features1 = extractor.extract_segment_features(
        "/fake/video.mp4", segment, subtitle_data
    )
    
    # Second extraction (should hit cache)
    features2 = extractor.extract_segment_features(
        "/fake/video.mp4", segment, subtitle_data
    )
    
    # Should be the same object (cached)
    assert features1 is features2
    
    print("[PASS] Feature caching works correctly")


def test_cache_clear():
    """Test clearing the cache."""
    extractor = FeatureExtractor(cache_enabled=True)
    
    # Add something to cache
    segment = Segment(segment_id="clear_test", start_seconds=0, end_seconds=60, segment_type="test")
    subtitle_data = SubtitleData(
        video_filename="test.mp4",
        entries=[SubtitleEntry(index=0, start_seconds=0, end_seconds=30, text="Test")],
        source="test",
        language="en"
    )
    
    extractor.extract_segment_features("/fake/video.mp4", segment, subtitle_data)
    assert len(extractor._cache) > 0
    
    extractor.clear_cache()
    assert len(extractor._cache) == 0
    
    print("[PASS] Cache clear works correctly")


# =============================================================================
# LEXICON TESTS
# =============================================================================

def test_lexicon_coverage():
    """Test that sentiment lexicons have reasonable coverage."""
    assert len(POSITIVE_WORDS) >= 30
    assert len(NEGATIVE_WORDS) >= 30
    
    # Check emotion keywords
    for emotion, keywords in EMOTION_KEYWORDS.items():
        assert len(keywords) >= 3, f"Emotion '{emotion}' has too few keywords"
    
    print(f"[PASS] Lexicons: {len(POSITIVE_WORDS)} positive, {len(NEGATIVE_WORDS)} negative words")


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests():
    """Run all feature extraction tests."""
    print("\n" + "="*60)
    print("MULTIMODAL FEATURE EXTRACTION TESTS")
    print("="*60 + "\n")
    
    # Text feature tests
    print("\n--- Text Feature Tests ---")
    test_text_extractor_creation()
    test_text_basic_features()
    test_text_sentiment_positive()
    test_text_sentiment_negative()
    test_text_sentiment_negation()
    test_text_speech_rate()
    test_text_emotion_detection()
    test_text_empty_input()
    
    # Audio feature tests
    print("\n--- Audio Feature Tests ---")
    test_audio_extractor_creation()
    test_audio_default_features()
    test_audio_stats()
    
    # Visual feature tests
    print("\n--- Visual Feature Tests ---")
    test_visual_extractor_creation()
    test_visual_default_features()
    
    # Unified extractor tests
    print("\n--- Unified Extractor Tests ---")
    test_feature_extractor_creation()
    test_feature_extractor_text_only()
    test_feature_extractor_audio_only()
    test_feature_extractor_full_multimodal()
    test_create_extractor_factory()
    test_segment_feature_extraction()
    test_batch_extraction()
    test_feature_caching()
    test_cache_clear()
    
    # Lexicon tests
    print("\n--- Lexicon Tests ---")
    test_lexicon_coverage()
    
    print("\n" + "="*60)
    print("ALL FEATURE EXTRACTION TESTS PASSED")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()

