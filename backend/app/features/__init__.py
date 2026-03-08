"""
Multimodal Feature Extraction Package
======================================
Research-grade feature extraction for video segments.

This package implements extraction of:
- Text features: sentiment, emotion, speech rate
- Audio features: energy, silence ratio, dynamics
- Visual features: motion, scene changes, brightness
- Deep features: CLIP, Wav2Vec2 emotion, Sentence Transformers, FER

Supports segment-level parallelism via ProcessPoolExecutor.

Usage:
    from app.features import FeatureExtractor
    
    extractor = FeatureExtractor()
    features = extractor.extract_all(video_path, segment)
    
    # Parallel batch extraction
    results = extractor.extract_batch(video_path, segments, subtitle_data)
"""

from .text_features import TextFeatureExtractor
from .audio_features import AudioFeatureExtractor
from .visual_features import VisualFeatureExtractor
from .extractor import FeatureExtractor, create_extractor, process_segment

__all__ = [
    'FeatureExtractor',
    'TextFeatureExtractor',
    'AudioFeatureExtractor',
    'VisualFeatureExtractor',
    'create_extractor',
    'process_segment',
]

