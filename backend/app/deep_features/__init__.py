"""
Deep Learning Feature Extraction Package
==========================================
CPU-friendly deep learning models for enhanced feature extraction.

This package provides four extractors that run efficiently on CPU:

- CLIPFeatureExtractor: Semantic visual understanding via CLIP ViT-B/32
- AudioEmotionExtractor: Speech emotion detection via Wav2Vec2
- TextEmbeddingExtractor: Semantic text embeddings via Sentence Transformers
- FaceEmotionExtractor: Facial emotion detection via FER / MediaPipe

All models use lazy loading (loaded on first call, kept in memory).

Usage:
    from app.deep_features import DeepFeatureExtractor

    extractor = DeepFeatureExtractor()
    features = extractor.extract_all(video_path, start, end, subtitle_text)
"""

import logging
import time
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .clip_features import CLIPFeatureExtractor
from .audio_emotion import AudioEmotionExtractor
from .text_embeddings import TextEmbeddingExtractor
from .face_emotion import FaceEmotionExtractor

logger = logging.getLogger(__name__)


class DeepFeatureExtractor:
    """
    Unified orchestrator for all deep learning feature extractors.

    Wraps CLIP, Wav2Vec2 audio emotion, Sentence Transformer text embeddings,
    and face emotion detection into a single interface. All sub-extractors
    run **in parallel** via ThreadPoolExecutor since they are independent.
    Each is fault-tolerant: if one fails, the others still produce results.
    """

    def __init__(
        self,
        enable_clip: bool = True,
        enable_audio_emotion: bool = True,
        enable_text_embeddings: bool = True,
        enable_face_emotion: bool = True,
    ):
        self.enable_clip = enable_clip
        self.enable_audio_emotion = enable_audio_emotion
        self.enable_text_embeddings = enable_text_embeddings
        self.enable_face_emotion = enable_face_emotion

        self._clip = CLIPFeatureExtractor() if enable_clip else None
        self._audio_emotion = AudioEmotionExtractor() if enable_audio_emotion else None
        self._text_embeddings = TextEmbeddingExtractor() if enable_text_embeddings else None
        self._face_emotion = FaceEmotionExtractor() if enable_face_emotion else None

        logger.info(
            "DeepFeatureExtractor initialized — "
            f"CLIP={enable_clip}, AudioEmotion={enable_audio_emotion}, "
            f"TextEmbed={enable_text_embeddings}, FaceEmotion={enable_face_emotion}"
        )

    def extract_all(
        self,
        video_path: str,
        start_seconds: float,
        end_seconds: float,
        subtitle_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run all enabled deep feature extractors **in parallel**.

        Returns a dictionary with keys:
            visual_deep_features, audio_deep_features,
            text_deep_features, face_deep_features
        """
        result: Dict[str, Any] = {}
        tasks: Dict[str, Any] = {}

        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="deep") as pool:
            if self._clip:
                tasks["visual_deep_features"] = pool.submit(
                    self._safe_extract, self._clip, "CLIP",
                    CLIPFeatureExtractor._default_features,
                    video_path, start_seconds, end_seconds,
                )

            if self._audio_emotion:
                tasks["audio_deep_features"] = pool.submit(
                    self._safe_extract, self._audio_emotion, "AudioEmotion",
                    AudioEmotionExtractor._default_features,
                    video_path, start_seconds, end_seconds,
                )

            if self._text_embeddings and subtitle_text:
                tasks["text_deep_features"] = pool.submit(
                    self._safe_extract_text, self._text_embeddings,
                    TextEmbeddingExtractor._default_features,
                    subtitle_text,
                )

            if self._face_emotion:
                tasks["face_deep_features"] = pool.submit(
                    self._safe_extract, self._face_emotion, "FaceEmotion",
                    FaceEmotionExtractor._default_features,
                    video_path, start_seconds, end_seconds,
                )

            for key, future in tasks.items():
                try:
                    result[key] = future.result(timeout=300)
                except Exception as e:
                    logger.warning(f"Deep feature '{key}' timed out or crashed: {e}")

        return result

    @staticmethod
    def _safe_extract(extractor, name, default_fn, video_path, start, end):
        try:
            t0 = time.time()
            out = extractor.extract(video_path, start, end)
            logger.debug(f"{name} extracted in {time.time()-t0:.1f}s")
            return out
        except Exception as e:
            logger.warning(f"{name} extraction failed: {e}")
            return default_fn()

    @staticmethod
    def _safe_extract_text(extractor, default_fn, text):
        try:
            t0 = time.time()
            out = extractor.extract(text)
            logger.debug(f"TextEmbed extracted in {time.time()-t0:.1f}s")
            return out
        except Exception as e:
            logger.warning(f"Text embedding extraction failed: {e}")
            return default_fn()


__all__ = [
    "DeepFeatureExtractor",
    "CLIPFeatureExtractor",
    "AudioEmotionExtractor",
    "TextEmbeddingExtractor",
    "FaceEmotionExtractor",
]
