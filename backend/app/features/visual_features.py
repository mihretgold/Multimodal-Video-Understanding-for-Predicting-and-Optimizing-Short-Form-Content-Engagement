"""
Visual Feature Extraction Module
================================
Extracts visual features from video frames.

Features extracted:
- Motion intensity (frame differencing)
- Scene change detection
- Brightness statistics
- Color variance

Note: Uses FFmpeg for frame extraction and basic image analysis.
For more advanced features, consider adding OpenCV as a dependency.
"""

import os
import subprocess
import tempfile
import logging
from typing import List, Optional, Tuple
from pathlib import Path

from ..models import VisualFeatures
from ..config import get_config

logger = logging.getLogger(__name__)


class VisualFeatureExtractor:
    """
    Extracts visual features from video segments.
    
    Uses FFmpeg to extract frames and analyzes them for:
    - Motion (difference between consecutive frames)
    - Scene changes (large differences)
    - Brightness
    - Color distribution
    """
    
    def __init__(self):
        """Initialize the visual feature extractor."""
        self.config = get_config().features
        self.sample_fps = self.config.visual_sample_fps
        self.histogram_bins = self.config.histogram_bins
        
        # Check for numpy/PIL availability
        try:
            import numpy as np
            self.np = np
            self._has_numpy = True
        except ImportError:
            self._has_numpy = False
            logger.warning("NumPy not available, visual features will use simplified analysis")
        
        try:
            from PIL import Image
            self.Image = Image
            self._has_pil = True
        except ImportError:
            self._has_pil = False
            logger.warning("PIL not available, using raw frame data")
    
    def extract(
        self,
        video_path: str,
        start_seconds: float,
        end_seconds: float
    ) -> VisualFeatures:
        """
        Extract visual features from a video segment.
        
        Args:
            video_path: Path to video file
            start_seconds: Segment start time
            end_seconds: Segment end time
            
        Returns:
            VisualFeatures object with computed features
        """
        duration = end_seconds - start_seconds
        
        if not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            return self._default_features()
        
        try:
            # Extract frames
            frames = self._extract_frames(video_path, start_seconds, end_seconds)
            
            if not frames:
                logger.warning("No frames extracted")
                return self._default_features()
            
            # Analyze frames
            return self._analyze_frames(frames, duration)
            
        except Exception as e:
            logger.warning(f"Visual feature extraction failed: {e}")
            return self._default_features()
    
    def _extract_frames(
        self,
        video_path: str,
        start_seconds: float,
        end_seconds: float
    ) -> List[bytes]:
        """
        Extract frames from video using FFmpeg.
        
        Returns list of raw frame data (RGB).
        """
        duration = end_seconds - start_seconds
        
        # Calculate number of frames to extract
        n_frames = max(2, int(duration * self.sample_fps))
        n_frames = min(n_frames, 30)  # Cap at 30 frames
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract frames as PPM (simple raw format)
                frame_pattern = os.path.join(temp_dir, 'frame_%03d.ppm')
                
                cmd = [
                    'ffmpeg',
                    '-y',
                    '-ss', str(start_seconds),
                    '-t', str(duration),
                    '-i', video_path,
                    '-vf', f'fps={self.sample_fps},scale=160:90',  # Small for efficiency
                    '-frames:v', str(n_frames),
                    frame_pattern
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=60
                )
                
                if result.returncode != 0:
                    logger.debug(f"FFmpeg stderr: {result.stderr.decode()[:200]}")
                    return []
                
                # Load extracted frames
                frames = []
                for i in range(1, n_frames + 1):
                    frame_path = os.path.join(temp_dir, f'frame_{i:03d}.ppm')
                    if os.path.exists(frame_path):
                        with open(frame_path, 'rb') as f:
                            frames.append(f.read())
                
                return frames
                
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg frame extraction timed out")
            return []
        except Exception as e:
            logger.warning(f"Frame extraction error: {e}")
            return []
    
    def _analyze_frames(self, frames: List[bytes], duration: float) -> VisualFeatures:
        """
        Analyze extracted frames for visual features.
        """
        if self._has_numpy and self._has_pil:
            return self._analyze_with_numpy(frames, duration)
        else:
            return self._analyze_simple(frames, duration)
    
    def _analyze_with_numpy(self, frames: List[bytes], duration: float) -> VisualFeatures:
        """
        Full analysis using NumPy and PIL.
        """
        np = self.np
        Image = self.Image
        
        # Convert PPM frames to numpy arrays
        frame_arrays = []
        for frame_data in frames:
            try:
                from io import BytesIO
                img = Image.open(BytesIO(frame_data))
                arr = np.array(img, dtype=np.float32)
                frame_arrays.append(arr)
            except Exception as e:
                logger.debug(f"Failed to parse frame: {e}")
                continue
        
        if len(frame_arrays) < 2:
            return self._default_features()
        
        # Stack frames for analysis
        frames_np = np.stack(frame_arrays)
        n_frames = len(frames_np)
        
        # Motion intensity (mean absolute difference between consecutive frames)
        motion_values = []
        scene_change_threshold = 50.0  # Pixel difference threshold
        scene_changes = 0
        
        for i in range(1, n_frames):
            diff = np.abs(frames_np[i] - frames_np[i-1])
            mean_diff = np.mean(diff)
            motion_values.append(mean_diff)
            
            # Count scene changes
            if mean_diff > scene_change_threshold:
                scene_changes += 1
        
        motion_intensity = np.mean(motion_values) / 255.0 if motion_values else 0.0
        
        # Scene change rate
        scene_change_rate = scene_changes / max(duration, 0.1)
        
        # Brightness analysis (convert to grayscale)
        gray_values = []
        for frame in frames_np:
            if len(frame.shape) == 3 and frame.shape[2] >= 3:
                gray = 0.299 * frame[:,:,0] + 0.587 * frame[:,:,1] + 0.114 * frame[:,:,2]
            else:
                gray = frame
            gray_values.append(np.mean(gray))
        
        brightness_mean = np.mean(gray_values) / 255.0
        brightness_std = np.std(gray_values) / 255.0
        
        # Color variance (variance across RGB channels)
        color_variances = []
        for frame in frames_np:
            if len(frame.shape) == 3 and frame.shape[2] >= 3:
                r_var = np.var(frame[:,:,0])
                g_var = np.var(frame[:,:,1])
                b_var = np.var(frame[:,:,2])
                color_variances.append((r_var + g_var + b_var) / 3)
        
        color_variance = np.mean(color_variances) / (255.0 ** 2) if color_variances else 0.0
        
        return VisualFeatures(
            motion_intensity=float(min(motion_intensity, 1.0)),
            scene_change_count=scene_changes,
            scene_change_rate=float(scene_change_rate),
            brightness_mean=float(brightness_mean),
            brightness_std=float(brightness_std),
            color_variance=float(min(color_variance, 1.0)),
            face_presence_ratio=0.0,  # Would need face detection
            text_overlay_ratio=0.0  # Would need OCR
        )
    
    def _analyze_simple(self, frames: List[bytes], duration: float) -> VisualFeatures:
        """
        Simple frame analysis without NumPy/PIL.
        
        Parses PPM format directly.
        """
        brightness_values = []
        motion_values = []
        prev_pixels = None
        scene_changes = 0
        scene_threshold = 1000  # Simple threshold
        
        for frame_data in frames:
            try:
                # Parse PPM (P6 format)
                pixels = self._parse_ppm_simple(frame_data)
                
                if pixels:
                    # Calculate average brightness
                    total = sum(pixels)
                    avg_brightness = total / len(pixels) if pixels else 0
                    brightness_values.append(avg_brightness / 255.0)
                    
                    # Calculate motion (difference from previous frame)
                    if prev_pixels and len(pixels) == len(prev_pixels):
                        diff = sum(abs(a - b) for a, b in zip(pixels, prev_pixels))
                        avg_diff = diff / len(pixels)
                        motion_values.append(avg_diff / 255.0)
                        
                        if avg_diff > scene_threshold:
                            scene_changes += 1
                    
                    prev_pixels = pixels
                    
            except Exception as e:
                logger.debug(f"Simple frame parse failed: {e}")
                continue
        
        # Calculate statistics
        brightness_mean = sum(brightness_values) / len(brightness_values) if brightness_values else 0.5
        motion_intensity = sum(motion_values) / len(motion_values) if motion_values else 0.0
        
        return VisualFeatures(
            motion_intensity=motion_intensity,
            scene_change_count=scene_changes,
            scene_change_rate=scene_changes / max(duration, 0.1),
            brightness_mean=brightness_mean,
            brightness_std=0.0,
            color_variance=0.0
        )
    
    def _parse_ppm_simple(self, data: bytes) -> Optional[List[int]]:
        """
        Simple PPM parser for brightness calculation.
        
        Returns flattened list of pixel values.
        """
        try:
            # PPM format: P6\n<width> <height>\n<maxval>\n<binary data>
            lines = data.split(b'\n', 3)
            if len(lines) < 4:
                return None
            
            if not lines[0].startswith(b'P6'):
                return None
            
            # Parse dimensions
            dims = lines[1].split()
            if len(dims) < 2:
                return None
            
            width = int(dims[0])
            height = int(dims[1])
            
            # Get pixel data (RGB, 3 bytes per pixel)
            pixel_data = lines[3]
            
            # Sample pixels (not all for efficiency)
            sample_stride = max(1, len(pixel_data) // 1000)
            pixels = [pixel_data[i] for i in range(0, len(pixel_data), sample_stride)]
            
            return pixels
            
        except Exception:
            return None
    
    def _default_features(self) -> VisualFeatures:
        """Return default visual features when extraction fails."""
        return VisualFeatures(
            motion_intensity=0.3,
            scene_change_count=0,
            scene_change_rate=0.0,
            brightness_mean=0.5,
            brightness_std=0.1,
            color_variance=0.1,
            face_presence_ratio=0.0,
            text_overlay_ratio=0.0
        )


def extract_visual_features(
    video_path: str,
    start_seconds: float,
    end_seconds: float
) -> VisualFeatures:
    """
    Convenience function to extract visual features.
    
    Args:
        video_path: Path to video file
        start_seconds: Segment start
        end_seconds: Segment end
        
    Returns:
        VisualFeatures object
    """
    extractor = VisualFeatureExtractor()
    return extractor.extract(video_path, start_seconds, end_seconds)

