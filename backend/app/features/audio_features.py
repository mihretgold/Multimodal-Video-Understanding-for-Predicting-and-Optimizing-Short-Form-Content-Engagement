"""
Audio Feature Extraction Module
===============================
Extracts audio features from video segments.

Features extracted:
- Energy statistics (mean, std, dynamics)
- Silence ratio
- Spectral features (centroid)
- Volume dynamics

Note: This module uses numpy for basic audio analysis.
For more advanced features, consider adding librosa as a dependency.
"""

import os
import subprocess
import tempfile
import logging
from typing import Optional, Tuple
import struct

from ..models import AudioFeatures
from ..config import get_config

logger = logging.getLogger(__name__)


class AudioFeatureExtractor:
    """
    Extracts audio features from video segments.
    
    Uses FFmpeg to extract audio and numpy for signal analysis.
    Falls back to reasonable defaults if extraction fails.
    """
    
    def __init__(self):
        """Initialize the audio feature extractor."""
        self.config = get_config().features
        self.sample_rate = self.config.audio_sample_rate
        
        # Check for numpy availability
        try:
            import numpy as np
            self.np = np
            self._has_numpy = True
        except ImportError:
            self._has_numpy = False
            logger.warning("NumPy not available, audio features will use simplified analysis")
    
    def extract(
        self,
        video_path: str,
        start_seconds: float,
        end_seconds: float
    ) -> AudioFeatures:
        """
        Extract audio features from a video segment.
        
        Args:
            video_path: Path to video file
            start_seconds: Segment start time
            end_seconds: Segment end time
            
        Returns:
            AudioFeatures object with computed features
        """
        duration = end_seconds - start_seconds
        
        if not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            return self._default_features()
        
        try:
            # Extract audio segment to raw PCM
            audio_data = self._extract_audio_segment(
                video_path, start_seconds, end_seconds
            )
            
            if audio_data is None or len(audio_data) == 0:
                logger.warning("No audio data extracted")
                return self._default_features()
            
            # Analyze the audio
            return self._analyze_audio(audio_data, duration)
            
        except Exception as e:
            logger.warning(f"Audio feature extraction failed: {e}")
            return self._default_features()
    
    def _extract_audio_segment(
        self,
        video_path: str,
        start_seconds: float,
        end_seconds: float
    ) -> Optional[bytes]:
        """
        Extract audio segment using FFmpeg.
        
        Returns raw PCM audio data.
        """
        try:
            duration = end_seconds - start_seconds
            
            # Use FFmpeg to extract audio as raw PCM (16-bit signed, mono)
            cmd = [
                'ffmpeg',
                '-y',
                '-ss', str(start_seconds),
                '-t', str(duration),
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # 16-bit PCM
                '-ar', str(self.sample_rate),  # Sample rate
                '-ac', '1',  # Mono
                '-f', 's16le',  # Raw format
                'pipe:1'  # Output to stdout
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30
            )
            
            if result.returncode != 0:
                logger.debug(f"FFmpeg stderr: {result.stderr.decode()[:200]}")
                return None
            
            return result.stdout
            
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg audio extraction timed out")
            return None
        except FileNotFoundError:
            logger.warning("FFmpeg not found")
            return None
        except Exception as e:
            logger.warning(f"Audio extraction error: {e}")
            return None
    
    def _analyze_audio(self, audio_data: bytes, duration: float) -> AudioFeatures:
        """
        Analyze raw audio data and extract features.
        """
        if self._has_numpy:
            return self._analyze_with_numpy(audio_data, duration)
        else:
            return self._analyze_simple(audio_data, duration)
    
    def _analyze_with_numpy(self, audio_data: bytes, duration: float) -> AudioFeatures:
        """
        Analyze audio using NumPy for full feature extraction.
        """
        np = self.np
        
        # Convert bytes to numpy array (16-bit signed integers)
        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        
        if len(samples) == 0:
            return self._default_features()
        
        # Normalize to -1 to 1 range
        samples = samples / 32768.0
        
        # Energy features (RMS energy)
        energy = np.sqrt(np.mean(samples ** 2))
        
        # Calculate energy in windows for std and dynamics
        window_size = int(self.sample_rate * 0.1)  # 100ms windows
        if len(samples) > window_size:
            n_windows = len(samples) // window_size
            windowed = samples[:n_windows * window_size].reshape(n_windows, window_size)
            window_energies = np.sqrt(np.mean(windowed ** 2, axis=1))
            energy_std = np.std(window_energies)
            energy_mean = np.mean(window_energies)
            
            # Volume dynamics (ratio of max to mean energy)
            volume_dynamics = np.max(window_energies) / max(energy_mean, 0.001)
        else:
            energy_std = 0.0
            energy_mean = energy
            volume_dynamics = 1.0
        
        # Silence ratio (percentage of near-silence samples)
        silence_threshold = 0.02  # -34 dB
        silence_ratio = np.mean(np.abs(samples) < silence_threshold)
        
        # Simple spectral centroid estimation
        # Using zero-crossing rate as a proxy for spectral content
        zero_crossings = np.sum(np.abs(np.diff(np.sign(samples)))) / 2
        zcr = zero_crossings / len(samples) if len(samples) > 0 else 0
        
        # Map ZCR to approximate spectral centroid (rough estimation)
        spectral_centroid = zcr * self.sample_rate / 2
        
        # Speech rate is calculated from text features, set to 0 here
        speech_rate = 0.0
        
        return AudioFeatures(
            energy_mean=float(energy_mean),
            energy_std=float(energy_std),
            pitch_mean=0.0,  # Would need pitch detection
            pitch_std=0.0,
            silence_ratio=float(silence_ratio),
            speech_rate=speech_rate,
            volume_dynamics=float(min(volume_dynamics, 10.0)),  # Cap at 10
            spectral_centroid=float(spectral_centroid)
        )
    
    def _analyze_simple(self, audio_data: bytes, duration: float) -> AudioFeatures:
        """
        Simple audio analysis without NumPy.
        """
        # Unpack 16-bit samples
        n_samples = len(audio_data) // 2
        
        if n_samples == 0:
            return self._default_features()
        
        samples = struct.unpack(f'{n_samples}h', audio_data)
        
        # Calculate basic statistics
        sum_sq = sum(s * s for s in samples)
        mean_sq = sum_sq / n_samples
        rms = (mean_sq ** 0.5) / 32768.0  # Normalize
        
        # Silence ratio (samples below threshold)
        silence_threshold = 1000  # About -30 dB
        silent_samples = sum(1 for s in samples if abs(s) < silence_threshold)
        silence_ratio = silent_samples / n_samples
        
        return AudioFeatures(
            energy_mean=rms,
            energy_std=0.0,  # Need windowing for this
            silence_ratio=silence_ratio,
            volume_dynamics=1.0,
            spectral_centroid=0.0
        )
    
    def _default_features(self) -> AudioFeatures:
        """Return default audio features when extraction fails."""
        return AudioFeatures(
            energy_mean=0.5,
            energy_std=0.1,
            pitch_mean=0.0,
            pitch_std=0.0,
            silence_ratio=0.2,
            speech_rate=0.0,
            volume_dynamics=1.0,
            spectral_centroid=0.0
        )
    
    def get_audio_stats(
        self,
        video_path: str,
        start_seconds: float,
        end_seconds: float
    ) -> dict:
        """
        Get detailed audio statistics for a segment.
        
        Returns a dictionary with all computed metrics.
        """
        features = self.extract(video_path, start_seconds, end_seconds)
        
        return {
            'energy': {
                'mean': features.energy_mean,
                'std': features.energy_std,
            },
            'silence_ratio': features.silence_ratio,
            'volume_dynamics': features.volume_dynamics,
            'spectral_centroid': features.spectral_centroid,
            'pitch': {
                'mean': features.pitch_mean,
                'std': features.pitch_std,
            }
        }


def extract_audio_features(
    video_path: str,
    start_seconds: float,
    end_seconds: float
) -> AudioFeatures:
    """
    Convenience function to extract audio features.
    
    Args:
        video_path: Path to video file
        start_seconds: Segment start
        end_seconds: Segment end
        
    Returns:
        AudioFeatures object
    """
    extractor = AudioFeatureExtractor()
    return extractor.extract(video_path, start_seconds, end_seconds)

