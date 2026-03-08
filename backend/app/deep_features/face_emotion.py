"""
Face Emotion Detection Module
==============================
Detects facial emotions from video frames using a lightweight CNN approach.

This module uses MediaPipe for face detection and a small CNN (FER-style)
for emotion classification. When the deep model is unavailable it falls
back to MediaPipe face-mesh landmarks for basic valence estimation.

Features extracted per segment:
- face_emotion_label: Dominant facial emotion across sampled frames
- face_emotion_confidence: Confidence of the dominant emotion
- face_count_mean: Average number of faces per frame
- face_emotion_diversity: Number of distinct emotions detected

Frames are sampled every 2 seconds. Runs on CPU.

Dependencies: mediapipe, Pillow, numpy, opencv-python-headless
"""

import os
import logging
import subprocess
import tempfile
from typing import Optional, Dict, Any, List
from collections import Counter

import numpy as np

from ..utils.video_utils import get_ffmpeg_path

logger = logging.getLogger(__name__)

SAMPLE_INTERVAL_SECONDS = 2.0

_fer_model = None
_fer_available: Optional[bool] = None


def _check_fer():
    """Check and lazily load the FER emotion classifier."""
    global _fer_model, _fer_available
    if _fer_available is not None:
        return _fer_available
    try:
        from fer import FER
        _fer_model = FER(mtcnn=False)
        _fer_available = True
        logger.info("FER face emotion model loaded successfully")
    except Exception:
        _fer_available = False
        logger.info("FER library not available — using MediaPipe face detection only")
    return _fer_available


def _detect_faces_mediapipe(image_array: np.ndarray) -> int:
    """Count faces using MediaPipe Face Detection (lightweight fallback)."""
    try:
        import mediapipe as mp
        with mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        ) as detector:
            results = detector.process(image_array)
            if results.detections:
                return len(results.detections)
    except Exception as e:
        logger.debug(f"MediaPipe face detection failed: {e}")
    return 0


class FaceEmotionExtractor:
    """
    Detects facial emotions from video frames.

    Uses FER (Facial Expression Recognition) when available, otherwise
    falls back to MediaPipe for face counting only.
    """

    def __init__(self):
        self.sample_interval = SAMPLE_INTERVAL_SECONDS
        self.max_frames = 10

    def extract(
        self,
        video_path: str,
        start_seconds: float,
        end_seconds: float,
    ) -> Dict[str, Any]:
        """
        Extract face emotion features for a video segment.

        Returns:
            {
                "face_emotion_label": str,
                "face_emotion_confidence": float,
                "face_count_mean": float,
                "face_emotion_diversity": int,
            }
        """
        if not os.path.exists(video_path):
            logger.warning(f"Video not found: {video_path}")
            return self._default_features()

        try:
            frames = self._sample_frames(video_path, start_seconds, end_seconds)
            if not frames:
                return self._default_features()

            return self._analyze_faces(frames)
        except Exception as e:
            logger.warning(f"Face emotion extraction failed: {e}")
            return self._default_features()

    def _sample_frames(
        self,
        video_path: str,
        start_seconds: float,
        end_seconds: float,
    ) -> List[np.ndarray]:
        """Sample frames as numpy RGB arrays."""
        from PIL import Image
        from io import BytesIO

        duration = end_seconds - start_seconds
        n_frames = max(1, int(duration / self.sample_interval))
        n_frames = min(n_frames, self.max_frames)
        effective_interval = duration / n_frames if n_frames > 0 else self.sample_interval
        fps_filter = 1.0 / max(effective_interval, 0.5)

        try:
            with tempfile.TemporaryDirectory() as tmp:
                pattern = os.path.join(tmp, "face_%04d.jpg")
                cmd = [
                    get_ffmpeg_path(), "-y",
                    "-ss", str(start_seconds),
                    "-t", str(duration),
                    "-i", video_path,
                    "-vf", f"fps={fps_filter},scale=320:-1",
                    "-frames:v", str(n_frames),
                    "-q:v", "3",
                    pattern,
                ]
                subprocess.run(cmd, capture_output=True, timeout=60)

                arrays = []
                for i in range(1, n_frames + 1):
                    fpath = os.path.join(tmp, f"face_{i:04d}.jpg")
                    if os.path.exists(fpath):
                        img = Image.open(fpath).convert("RGB")
                        arrays.append(np.array(img))
                return arrays
        except Exception as e:
            logger.warning(f"Frame sampling for face detection failed: {e}")
            return []

    def _analyze_faces(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze faces across all sampled frames."""
        use_fer = _check_fer()
        all_emotions: List[str] = []
        all_confidences: List[float] = []
        face_counts: List[int] = []

        for frame in frames:
            if use_fer and _fer_model is not None:
                try:
                    detections = _fer_model.detect_emotions(frame)
                    face_counts.append(len(detections))
                    for det in detections:
                        emotions = det.get("emotions", {})
                        if emotions:
                            top_emotion = max(emotions, key=emotions.get)
                            all_emotions.append(top_emotion)
                            all_confidences.append(emotions[top_emotion])
                except Exception:
                    face_counts.append(_detect_faces_mediapipe(frame))
            else:
                face_counts.append(_detect_faces_mediapipe(frame))

        face_count_mean = float(np.mean(face_counts)) if face_counts else 0.0

        if all_emotions:
            counter = Counter(all_emotions)
            dominant_emotion, count = counter.most_common(1)[0]
            dominant_confidence = float(np.mean(
                [c for e, c in zip(all_emotions, all_confidences) if e == dominant_emotion]
            ))
            diversity = len(counter)
        else:
            dominant_emotion = "unknown"
            dominant_confidence = 0.0
            diversity = 0

        return {
            "face_emotion_label": dominant_emotion,
            "face_emotion_confidence": round(dominant_confidence, 4),
            "face_count_mean": round(face_count_mean, 2),
            "face_emotion_diversity": diversity,
        }

    @staticmethod
    def _default_features() -> Dict[str, Any]:
        return {
            "face_emotion_label": "unknown",
            "face_emotion_confidence": 0.0,
            "face_count_mean": 0.0,
            "face_emotion_diversity": 0,
        }
