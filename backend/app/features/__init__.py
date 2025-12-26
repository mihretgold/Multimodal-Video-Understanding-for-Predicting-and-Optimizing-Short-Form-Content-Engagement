"""
Multimodal Feature Extraction Package
======================================
Research-grade feature extraction for video segments.

This package implements extraction of:
- Text features: sentiment, emotion, speech rate
- Audio features: energy, silence ratio, dynamics
- Visual features: motion, scene changes, brightness

Usage:
    from app.features import FeatureExtractor
    
    extractor = FeatureExtractor()
    features = extractor.extract_all(video_path, segment)
    
    # Or extract specific modalities
    text_features = extractor.extract_text(subtitle_text)
    audio_features = extractor.extract_audio(video_path, start, end)
    visual_features = extractor.extract_visual(video_path, start, end)
"""

from .text_features import TextFeatureExtractor
from .audio_features import AudioFeatureExtractor
from .visual_features import VisualFeatureExtractor
from .extractor import FeatureExtractor

__all__ = [
    'FeatureExtractor',
    'TextFeatureExtractor',
    'AudioFeatureExtractor',
    'VisualFeatureExtractor',
]

