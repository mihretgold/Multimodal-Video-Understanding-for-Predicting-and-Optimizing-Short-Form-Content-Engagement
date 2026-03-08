"""
CLIP Visual Feature Extraction Module
======================================
Extracts semantic visual features using OpenAI's CLIP (ViT-B/32).

CLIP (Contrastive Language-Image Pre-training) maps images and text into a
shared embedding space, enabling semantic understanding of visual content
without task-specific fine-tuning.

Features extracted per segment:
- clip_embedding_mean: Mean CLIP embedding across sampled frames
- clip_semantic_variance: Variance of embeddings (visual diversity)
- semantic_scene_change_rate: Rate of large embedding shifts between frames
- object_richness_score: Embedding norm as a proxy for visual complexity

Frames are sampled every 2 seconds to reduce compute cost on CPU.
Model is loaded lazily (singleton) so it initializes only once.

Dependencies: transformers, torch, Pillow
"""

import os
import logging
import subprocess
import tempfile
from typing import Optional, Dict, Any, List

import numpy as np

from ..config import get_config
from ..utils.video_utils import get_ffmpeg_path

logger = logging.getLogger(__name__)

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
SAMPLE_INTERVAL_SECONDS = 2.0
SCENE_CHANGE_COSINE_THRESHOLD = 0.3

_clip_model = None
_clip_processor = None


def _get_clip_model():
    """Lazy-load the CLIP model and processor (singleton)."""
    global _clip_model, _clip_processor
    if _clip_model is None:
        try:
            from transformers import CLIPModel, CLIPProcessor
            import torch

            logger.info(f"Loading CLIP model: {CLIP_MODEL_NAME} (CPU)")
            _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
            _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
            _clip_model.eval()
            _clip_model.to("cpu")
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    return _clip_model, _clip_processor


class CLIPFeatureExtractor:
    """
    Extracts semantic visual features from video segments using CLIP ViT-B/32.

    The model runs on CPU by default. Frames are sampled every 2 seconds
    to keep compute tractable for long segments.
    """

    def __init__(self):
        self.device = "cpu"
        self.sample_interval = SAMPLE_INTERVAL_SECONDS
        self.max_frames = 15

    def extract(
        self,
        video_path: str,
        start_seconds: float,
        end_seconds: float,
    ) -> Dict[str, Any]:
        """
        Extract CLIP-based visual features for a video segment.

        Returns:
            {
                "clip_embedding_mean": List[float],
                "clip_semantic_variance": float,
                "semantic_scene_change_rate": float,
                "object_richness_score": float,
            }
        """
        if not os.path.exists(video_path):
            logger.warning(f"Video not found: {video_path}")
            return self._default_features()

        try:
            frames = self._sample_frames(video_path, start_seconds, end_seconds)
            if not frames:
                logger.warning("No frames sampled for CLIP extraction")
                return self._default_features()

            embeddings = self._encode_frames(frames)
            if embeddings is None or len(embeddings) == 0:
                return self._default_features()

            return self._compute_features(embeddings, end_seconds - start_seconds)

        except Exception as e:
            logger.warning(f"CLIP feature extraction failed: {e}")
            return self._default_features()

    def _sample_frames(
        self,
        video_path: str,
        start_seconds: float,
        end_seconds: float,
    ) -> list:
        """Sample frames from the segment at the configured interval using FFmpeg."""
        from PIL import Image
        from io import BytesIO

        duration = end_seconds - start_seconds
        n_frames = max(1, int(duration / self.sample_interval))
        n_frames = min(n_frames, self.max_frames)

        effective_interval = duration / n_frames if n_frames > 0 else self.sample_interval
        fps_filter = 1.0 / max(effective_interval, 0.5)

        try:
            with tempfile.TemporaryDirectory() as tmp:
                pattern = os.path.join(tmp, "frame_%04d.jpg")
                cmd = [
                    get_ffmpeg_path(), "-y",
                    "-ss", str(start_seconds),
                    "-t", str(duration),
                    "-i", video_path,
                    "-vf", f"fps={fps_filter},scale=224:224",
                    "-frames:v", str(n_frames),
                    "-q:v", "2",
                    pattern,
                ]
                subprocess.run(cmd, capture_output=True, timeout=120)

                images = []
                for i in range(1, n_frames + 1):
                    fpath = os.path.join(tmp, f"frame_{i:04d}.jpg")
                    if os.path.exists(fpath):
                        images.append(Image.open(fpath).convert("RGB"))
                return images
        except Exception as e:
            logger.warning(f"Frame sampling failed: {e}")
            return []

    def _encode_frames(self, frames: list) -> Optional[np.ndarray]:
        """Encode PIL images into CLIP embedding vectors."""
        try:
            import torch

            model, processor = _get_clip_model()
            inputs = processor(images=frames, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)

            with torch.no_grad():
                image_features = model.get_image_features(pixel_values=pixel_values)

            if not isinstance(image_features, torch.Tensor):
                vision_out = model.vision_model(pixel_values=pixel_values)
                image_features = model.visual_projection(vision_out.pooler_output)

            embeddings = image_features.detach().cpu().numpy()
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-8, None)
            embeddings = embeddings / norms
            return embeddings
        except Exception as e:
            logger.warning(f"CLIP encoding failed: {e}")
            return None

    def _compute_features(
        self, embeddings: np.ndarray, duration: float
    ) -> Dict[str, Any]:
        """Derive aggregate features from per-frame CLIP embeddings."""
        mean_emb = np.mean(embeddings, axis=0)
        variance = float(np.mean(np.var(embeddings, axis=0)))

        scene_changes = 0
        if len(embeddings) > 1:
            for i in range(1, len(embeddings)):
                cosine_sim = float(np.dot(embeddings[i], embeddings[i - 1]))
                if (1.0 - cosine_sim) > SCENE_CHANGE_COSINE_THRESHOLD:
                    scene_changes += 1

        scene_change_rate = scene_changes / max(duration, 0.1)
        richness = float(np.mean(np.linalg.norm(embeddings, axis=1)))

        return {
            "clip_embedding_mean": mean_emb.tolist(),
            "clip_semantic_variance": round(variance, 6),
            "semantic_scene_change_rate": round(scene_change_rate, 4),
            "object_richness_score": round(richness, 4),
        }

    @staticmethod
    def _default_features() -> Dict[str, Any]:
        return {
            "clip_embedding_mean": [],
            "clip_semantic_variance": 0.0,
            "semantic_scene_change_rate": 0.0,
            "object_richness_score": 0.0,
        }
