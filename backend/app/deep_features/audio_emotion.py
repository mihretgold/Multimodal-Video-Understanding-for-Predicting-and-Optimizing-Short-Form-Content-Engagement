"""
Audio Emotion Detection Module
==============================
Detects emotional content in speech using Wav2Vec2 (small).

Wav2Vec2 is a self-supervised speech representation model. We use the
emotion-recognition fine-tuned variant to classify voice segments into
emotion categories (angry, happy, sad, neutral, etc.).

Features extracted per segment:
- audio_emotion_label: Dominant detected emotion
- audio_emotion_confidence: Confidence of the dominant emotion
- audio_emotion_valence: Positive/negative emotional valence (-1 to 1)
- audio_excitement_score: Composite excitement measure

Model is loaded lazily and runs on CPU.

Dependencies: transformers, torch, torchaudio
"""

import os
import logging
import subprocess
import tempfile
from typing import Optional, Dict, Any

import numpy as np

from ..utils.video_utils import get_ffmpeg_path

logger = logging.getLogger(__name__)

WAV2VEC2_EMOTION_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

EMOTION_VALENCE = {
    "angry": -0.6,
    "disgust": -0.7,
    "fear": -0.5,
    "happy": 0.8,
    "neutral": 0.0,
    "sad": -0.8,
    "surprise": 0.4,
    "ps": 0.4,  # positive surprise
}

EXCITEMENT_EMOTIONS = {"angry", "happy", "surprise", "ps", "fear"}

_audio_emotion_pipeline = None


def _get_audio_emotion_pipeline():
    """Lazy-load the audio emotion classification pipeline (singleton)."""
    global _audio_emotion_pipeline
    if _audio_emotion_pipeline is None:
        try:
            from transformers import pipeline

            logger.info(f"Loading audio emotion model: {WAV2VEC2_EMOTION_MODEL} (CPU)")
            _audio_emotion_pipeline = pipeline(
                "audio-classification",
                model=WAV2VEC2_EMOTION_MODEL,
                device=-1,
            )
            logger.info("Audio emotion model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load audio emotion model: {e}")
            raise
    return _audio_emotion_pipeline


class AudioEmotionExtractor:
    """
    Detects emotional tone from speech audio using Wav2Vec2.

    Processes the audio track of a video segment and returns emotion
    classification results. Runs on CPU.
    """

    def __init__(self):
        self.sample_rate = 16000

    def extract(
        self,
        video_path: str,
        start_seconds: float,
        end_seconds: float,
    ) -> Dict[str, Any]:
        """
        Extract audio emotion features for a video segment.

        Returns:
            {
                "audio_emotion_label": str,
                "audio_emotion_confidence": float,
                "audio_emotion_valence": float,
                "audio_excitement_score": float,
            }
        """
        if not os.path.exists(video_path):
            logger.warning(f"Video not found: {video_path}")
            return self._default_features()

        try:
            waveform = self._extract_audio(video_path, start_seconds, end_seconds)
            if waveform is None or len(waveform) == 0:
                return self._default_features()

            return self._classify_emotion(waveform)
        except Exception as e:
            logger.warning(f"Audio emotion extraction failed: {e}")
            return self._default_features()

    def _extract_audio(
        self,
        video_path: str,
        start_seconds: float,
        end_seconds: float,
    ) -> Optional[np.ndarray]:
        """Extract mono 16 kHz WAV audio from a video segment via FFmpeg."""
        duration = end_seconds - start_seconds
        max_duration = min(duration, 30.0)

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            cmd = [
                get_ffmpeg_path(), "-y",
                "-ss", str(start_seconds),
                "-t", str(max_duration),
                "-i", video_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", str(self.sample_rate),
                "-ac", "1",
                tmp_path,
            ]
            subprocess.run(cmd, capture_output=True, timeout=60)

            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 100:
                return None

            import struct
            with open(tmp_path, "rb") as f:
                raw = f.read()

            os.unlink(tmp_path)

            header_end = raw.find(b"data")
            if header_end == -1:
                return None
            data_start = header_end + 8
            pcm = raw[data_start:]

            n_samples = len(pcm) // 2
            if n_samples == 0:
                return None
            samples = np.array(
                struct.unpack(f"{n_samples}h", pcm[:n_samples * 2]),
                dtype=np.float32,
            )
            samples = samples / 32768.0
            return samples
        except Exception as e:
            logger.warning(f"Audio extraction failed: {e}")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return None

    def _classify_emotion(self, waveform: np.ndarray) -> Dict[str, Any]:
        """Run emotion classification on the waveform."""
        try:
            pipe = _get_audio_emotion_pipeline()

            max_samples = self.sample_rate * 10
            if len(waveform) > max_samples:
                mid = len(waveform) // 2
                half = max_samples // 2
                waveform = waveform[mid - half : mid + half]

            results = pipe(
                {"raw": waveform, "sampling_rate": self.sample_rate},
                top_k=5,
            )

            if not results:
                return self._default_features()

            top = results[0]
            label = top["label"].lower()
            confidence = float(top["score"])

            valence = EMOTION_VALENCE.get(label, 0.0)

            excitement = sum(
                r["score"]
                for r in results
                if r["label"].lower() in EXCITEMENT_EMOTIONS
            )
            excitement = min(float(excitement), 1.0)

            return {
                "audio_emotion_label": label,
                "audio_emotion_confidence": round(confidence, 4),
                "audio_emotion_valence": round(valence, 4),
                "audio_excitement_score": round(excitement, 4),
            }
        except Exception as e:
            logger.warning(f"Emotion classification failed: {e}")
            return self._default_features()

    @staticmethod
    def _default_features() -> Dict[str, Any]:
        return {
            "audio_emotion_label": "neutral",
            "audio_emotion_confidence": 0.0,
            "audio_emotion_valence": 0.0,
            "audio_excitement_score": 0.0,
        }
