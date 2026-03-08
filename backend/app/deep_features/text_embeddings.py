"""
Text Embedding Module
=====================
Generates semantic text embeddings using Sentence Transformers.

Sentence Transformers produce dense vector representations that capture
the meaning of text. Similar sentences map to nearby points in the
embedding space, enabling semantic similarity comparisons.

Features extracted per segment:
- text_embedding: Dense vector (384-d for MiniLM-L6-v2)
- text_semantic_density: Embedding norm as proxy for information density
- text_embedding_variance: Variance across sentence embeddings (topic diversity)
- text_coherence_score: Average cosine similarity between consecutive sentences

Model is loaded lazily and runs on CPU.

Dependencies: sentence-transformers, torch
"""

import logging
from typing import Optional, Dict, Any, List

import numpy as np

logger = logging.getLogger(__name__)

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

_sbert_model = None


def _get_sbert_model():
    """Lazy-load the Sentence Transformer model (singleton)."""
    global _sbert_model
    if _sbert_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading Sentence Transformer: {SBERT_MODEL_NAME} (CPU)")
            _sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device="cpu")
            logger.info("Sentence Transformer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer: {e}")
            raise
    return _sbert_model


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using simple heuristics."""
    import re
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


class TextEmbeddingExtractor:
    """
    Generates semantic embeddings for subtitle text using Sentence Transformers.

    The model maps each sentence to a 384-dimensional dense vector.
    Aggregate statistics over sentences measure topic diversity and
    coherence within a segment.
    """

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract text embedding features from subtitle text.

        Args:
            text: Concatenated subtitle text for a segment.

        Returns:
            {
                "text_embedding": List[float],
                "text_semantic_density": float,
                "text_embedding_variance": float,
                "text_coherence_score": float,
            }
        """
        if not text or not text.strip():
            return self._default_features()

        try:
            model = _get_sbert_model()
            sentences = _split_sentences(text)

            if not sentences:
                return self._default_features()

            embeddings = model.encode(sentences, show_progress_bar=False)
            embeddings = np.array(embeddings)

            return self._compute_features(embeddings)
        except Exception as e:
            logger.warning(f"Text embedding extraction failed: {e}")
            return self._default_features()

    def _compute_features(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Derive aggregate features from per-sentence embeddings."""
        mean_emb = np.mean(embeddings, axis=0)
        density = float(np.linalg.norm(mean_emb))
        variance = float(np.mean(np.var(embeddings, axis=0))) if len(embeddings) > 1 else 0.0

        coherence = 0.0
        if len(embeddings) > 1:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-8, None)
            normed = embeddings / norms
            sims = []
            for i in range(1, len(normed)):
                sims.append(float(np.dot(normed[i], normed[i - 1])))
            coherence = float(np.mean(sims)) if sims else 0.0

        return {
            "text_embedding": mean_emb.tolist(),
            "text_semantic_density": round(density, 4),
            "text_embedding_variance": round(variance, 6),
            "text_coherence_score": round(coherence, 4),
        }

    @staticmethod
    def _default_features() -> Dict[str, Any]:
        return {
            "text_embedding": [],
            "text_semantic_density": 0.0,
            "text_embedding_variance": 0.0,
            "text_coherence_score": 0.0,
        }
