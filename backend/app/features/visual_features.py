"""
Visual Feature Extraction Module
================================
Extracts visual features from video frames.

This module implements two categories of features:

SIGNAL-LEVEL FEATURES (basic statistics):
- Motion intensity (raw pixel differencing)
- Brightness statistics
- Color variance

CLASSICAL COMPUTER VISION FEATURES:
- Contrast: Standard deviation of grayscale intensities (image texture measure)
- Edge density: Ratio of edge pixels using Canny edge detection
- Motion magnitude: Temporal derivative via frame differencing
- Scene boundaries: Histogram-based scene cut detection

COMPUTER VISION CONCEPTS DEMONSTRATED:
1. Color space conversion (RGB → Grayscale using luminance weights)
2. Pixel-level statistics (mean, std for brightness and contrast)
3. Gradient-based edge detection (Canny algorithm)
4. Temporal differencing for motion estimation
5. Histogram comparison for scene change detection

Dependencies: OpenCV (cv2), NumPy, PIL
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

# Try to import OpenCV for computer vision features
try:
    import cv2
    CV2_AVAILABLE = True
    logger.info("OpenCV available - classical CV features enabled")
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available - using fallback methods for visual features")


class VisualFeatureExtractor:
    """
    Extracts visual features from video segments.
    
    Uses FFmpeg to extract frames and analyzes them for:
    
    SIGNAL-LEVEL FEATURES:
    - Motion intensity (difference between consecutive frames)
    - Scene changes (large differences)
    - Brightness (mean luminance)
    - Color distribution
    
    CLASSICAL COMPUTER VISION FEATURES (requires OpenCV):
    - Contrast: Pixel-level texture measure via grayscale std
    - Edge density: Canny edge detection ratio
    - Motion magnitude: Proper temporal differencing
    - Scene boundaries: Histogram-based detection
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
        
        # Check OpenCV availability for CV features
        self._has_cv2 = CV2_AVAILABLE
        if self._has_cv2:
            self.cv2 = cv2
            logger.debug("OpenCV initialized for classical CV feature extraction")
    
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
        
        Computes both signal-level and classical CV features.
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
        
        # =====================================================================
        # SIGNAL-LEVEL FEATURES (existing)
        # =====================================================================
        
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
        # Using standard luminance weights: Y = 0.299*R + 0.587*G + 0.114*B
        gray_frames = []
        gray_values = []
        for frame in frames_np:
            if len(frame.shape) == 3 and frame.shape[2] >= 3:
                # COMPUTER VISION CONCEPT: Color space conversion
                # The luminance formula is derived from human visual perception
                # Green contributes most (0.587) because human eyes are most
                # sensitive to green light
                gray = 0.299 * frame[:,:,0] + 0.587 * frame[:,:,1] + 0.114 * frame[:,:,2]
            else:
                gray = frame if len(frame.shape) == 2 else frame[:,:,0]
            gray_frames.append(gray)
            gray_values.append(np.mean(gray))
        
        brightness_mean = np.mean(gray_values) / 255.0
        brightness_std = np.std(gray_values) / 255.0  # Temporal variation
        
        # Color variance (variance across RGB channels)
        color_variances = []
        for frame in frames_np:
            if len(frame.shape) == 3 and frame.shape[2] >= 3:
                r_var = np.var(frame[:,:,0])
                g_var = np.var(frame[:,:,1])
                b_var = np.var(frame[:,:,2])
                color_variances.append((r_var + g_var + b_var) / 3)
        
        color_variance = np.mean(color_variances) / (255.0 ** 2) if color_variances else 0.0
        
        # =====================================================================
        # CLASSICAL COMPUTER VISION FEATURES (Step 1)
        # =====================================================================
        
        # CONTRAST: Standard deviation of grayscale pixel intensities
        # -----------------------------------------------------------------
        # COMPUTER VISION CONCEPT: Contrast as a texture measure
        # 
        # In classical CV, contrast is defined as the variation in pixel
        # intensities within an image. High contrast indicates rich texture,
        # sharp edges, and visual complexity. Low contrast suggests flat,
        # uniform regions.
        #
        # This is different from brightness_std (temporal variation across frames).
        # Contrast measures INTRA-FRAME variation; brightness_std measures
        # INTER-FRAME variation.
        #
        # Formula: contrast = mean(std(I_grayscale) for each frame)
        # Normalized to [0, 1] by dividing by 127.5 (half of 255)
        # -----------------------------------------------------------------
        contrast_values = []
        for gray in gray_frames:
            # Standard deviation of grayscale pixels within this frame
            frame_contrast = np.std(gray)
            contrast_values.append(frame_contrast)
        
        # Average contrast across all frames, normalized
        contrast = np.mean(contrast_values) / 127.5 if contrast_values else 0.0
        contrast = float(min(contrast, 1.0))  # Clamp to [0, 1]
        
        # =====================================================================
        # CLASSICAL COMPUTER VISION FEATURES (Step 2): Edge Detection
        # =====================================================================
        
        # EDGE DETECTION using Canny Algorithm
        # -----------------------------------------------------------------
        # COMPUTER VISION CONCEPT: Gradient-based edge detection
        #
        # The Canny edge detector is a multi-stage algorithm:
        # 1. Gaussian smoothing to reduce noise
        # 2. Gradient computation (Sobel operators for x and y directions)
        # 3. Non-maximum suppression (thin edges to 1-pixel width)
        # 4. Hysteresis thresholding (connect strong edges through weak ones)
        #
        # Parameters:
        # - low_threshold (50): Edges below this are discarded
        # - high_threshold (150): Edges above this are kept
        # - Ratio typically 1:2 or 1:3 as recommended by Canny
        #
        # Why edge detection matters for engagement:
        # - High edge density = visually complex, detailed scenes
        # - Action sequences typically have more edges (motion blur creates fewer)
        # - Text overlays and graphics increase edge density
        # -----------------------------------------------------------------
        
        edge_density = 0.0
        edge_intensity = 0.0
        
        if self._has_cv2:
            edge_densities = []
            edge_intensities = []
            
            for gray in gray_frames:
                # Convert to uint8 for OpenCV (Canny requires 8-bit input)
                gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
                
                # Apply Canny edge detection
                # Thresholds: 50 (low) and 150 (high) - standard ratio of 1:3
                edges = self.cv2.Canny(gray_uint8, 50, 150)
                
                # EDGE DENSITY: Ratio of edge pixels to total pixels
                # This measures how "busy" or complex the image is
                total_pixels = edges.shape[0] * edges.shape[1]
                edge_pixels = np.count_nonzero(edges)
                frame_edge_density = edge_pixels / total_pixels
                edge_densities.append(frame_edge_density)
                
                # EDGE INTENSITY: Average gradient magnitude at edge locations
                # Compute Sobel gradients for magnitude
                sobel_x = self.cv2.Sobel(gray_uint8, self.cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = self.cv2.Sobel(gray_uint8, self.cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
                
                # Average gradient magnitude (normalized by max possible ~360)
                frame_edge_intensity = np.mean(gradient_magnitude) / 360.0
                edge_intensities.append(frame_edge_intensity)
            
            edge_density = float(np.mean(edge_densities)) if edge_densities else 0.0
            edge_intensity = float(min(np.mean(edge_intensities), 1.0)) if edge_intensities else 0.0
            
            logger.debug(f"CV Features - Edge Density: {edge_density:.4f}, Edge Intensity: {edge_intensity:.4f}")
        else:
            logger.debug("OpenCV not available - edge detection skipped")
        
        # =====================================================================
        # CLASSICAL COMPUTER VISION FEATURES (Step 3): Motion Estimation
        # =====================================================================
        
        # MOTION MAGNITUDE via Frame Differencing
        # -----------------------------------------------------------------
        # COMPUTER VISION CONCEPT: Temporal derivative for motion detection
        #
        # Frame differencing is a foundational technique for motion detection:
        # - Compute absolute difference between consecutive frames
        # - Threshold to remove noise (small intensity changes from compression)
        # - Aggregate motion across the frame
        #
        # Formula: motion = mean(|I(t) - I(t-1)| > threshold)
        #
        # This is simpler than optical flow but works well for:
        # - Detecting scene activity level
        # - Identifying static vs dynamic content
        # - Measuring overall movement without direction
        #
        # LIMITATIONS (important for research honesty):
        # - Cannot distinguish object motion from camera motion
        # - Sensitive to lighting changes (can cause false motion)
        # - Does not capture motion direction or velocity vectors
        # - For proper motion vectors, optical flow (Lucas-Kanade, Farneback) is needed
        #
        # Why frame differencing over optical flow here:
        # - Computationally efficient (no iterative optimization)
        # - Sufficient for engagement scoring (we need activity level, not trajectories)
        # - Demonstrates core CV concept without overengineering
        # -----------------------------------------------------------------
        
        motion_magnitude = 0.0
        motion_values_cv = []
        
        # Use grayscale frames for motion (already computed)
        for i in range(1, len(gray_frames)):
            prev_gray = gray_frames[i-1]
            curr_gray = gray_frames[i]
            
            # Absolute difference between consecutive frames
            frame_diff = np.abs(curr_gray - prev_gray)
            
            # THRESHOLDING: Remove noise (compression artifacts, sensor noise)
            # Pixels with difference < 10 are considered static
            # This is a key CV concept: noise filtering before analysis
            motion_threshold = 10.0
            significant_motion = frame_diff > motion_threshold
            
            # Two complementary motion metrics:
            # 1. Motion coverage: What fraction of pixels are moving?
            motion_coverage = np.mean(significant_motion)
            
            # 2. Motion intensity: How much are moving pixels changing?
            if np.any(significant_motion):
                motion_strength = np.mean(frame_diff[significant_motion]) / 255.0
            else:
                motion_strength = 0.0
            
            # Combine coverage and strength (geometric mean for balance)
            frame_motion = np.sqrt(motion_coverage * motion_strength)
            motion_values_cv.append(frame_motion)
        
        if motion_values_cv:
            # Normalize by duration to get motion rate (motion per second)
            # This makes the metric comparable across segments of different lengths
            motion_magnitude = float(np.mean(motion_values_cv))
            # Clamp to [0, 1]
            motion_magnitude = min(motion_magnitude, 1.0)
        
        logger.debug(f"CV Features - Motion Magnitude: {motion_magnitude:.4f} "
                    f"(vs signal-level motion_intensity: {motion_intensity:.4f})")
        
        # =====================================================================
        # CLASSICAL COMPUTER VISION FEATURES (Step 4): Histogram-Based Scene Detection
        # =====================================================================
        
        # SCENE CHANGE DETECTION using Color Histograms
        # -----------------------------------------------------------------
        # COMPUTER VISION CONCEPT: Histogram comparison for shot boundary detection
        #
        # This is a textbook technique for video segmentation:
        # 1. Compute color histogram for each frame
        # 2. Compare histograms of consecutive frames using a distance metric
        # 3. Large distances indicate scene changes (cuts, transitions)
        #
        # Why histograms over pixel differencing?
        # - More robust to camera motion (global histogram doesn't change much)
        # - Captures color distribution changes, not just pixel positions
        # - Less sensitive to small object movements within a scene
        #
        # Distance metrics commonly used:
        # - Chi-Square: sum((h1-h2)^2 / (h1+h2))
        # - Bhattacharyya: measures overlap between distributions
        # - Correlation: measures linear relationship
        # - Intersection: sum(min(h1, h2))
        #
        # We use Chi-Square as it's the most common in academic literature
        # and provides good sensitivity to distribution changes.
        #
        # Threshold selection:
        # - Lower threshold = more scene changes detected (higher recall)
        # - Higher threshold = fewer false positives (higher precision)
        # - We use 0.5 as a balanced default (can be tuned per use case)
        # -----------------------------------------------------------------
        
        histogram_diff_mean = 0.0
        scene_boundaries = []
        histogram_diffs = []
        
        if self._has_cv2 and len(gray_frames) >= 2:
            # Compute histograms for all frames
            histograms = []
            for i, gray in enumerate(gray_frames):
                gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
                
                # Compute grayscale histogram (256 bins)
                # Using grayscale for simplicity; color histograms (H-S in HSV)
                # would be more robust but require color frames
                hist = self.cv2.calcHist([gray_uint8], [0], None, [64], [0, 256])
                
                # Normalize histogram to sum to 1 (probability distribution)
                hist = hist / (hist.sum() + 1e-7)
                histograms.append(hist)
            
            # Compare consecutive histograms
            scene_change_threshold = 0.5  # Chi-square threshold for scene cut
            
            # Calculate frame interval for timestamp estimation
            # Assuming uniform sampling across segment duration
            frame_interval = duration / max(len(gray_frames) - 1, 1)
            
            for i in range(1, len(histograms)):
                # Chi-Square distance between consecutive histograms
                # cv2.compareHist returns distance (0 = identical, higher = different)
                chi_square_dist = self.cv2.compareHist(
                    histograms[i-1], 
                    histograms[i], 
                    self.cv2.HISTCMP_CHISQR
                )
                
                histogram_diffs.append(chi_square_dist)
                
                # Detect scene boundary if distance exceeds threshold
                if chi_square_dist > scene_change_threshold:
                    # Estimate timestamp of this frame
                    timestamp = i * frame_interval
                    scene_boundaries.append(float(timestamp))
                    logger.debug(f"Scene boundary detected at {timestamp:.2f}s "
                               f"(chi-square={chi_square_dist:.4f})")
            
            # Mean histogram difference across the segment
            if histogram_diffs:
                histogram_diff_mean = float(np.mean(histogram_diffs))
            
            logger.debug(f"CV Features - Histogram Diff Mean: {histogram_diff_mean:.4f}, "
                        f"Scene Boundaries: {len(scene_boundaries)}")
        
        # Log all CV features for research observability
        logger.debug(f"CV Features Summary - Brightness: {brightness_mean:.3f}, Contrast: {contrast:.3f}, "
                    f"Edge Density: {edge_density:.4f}, Motion: {motion_magnitude:.4f}, "
                    f"Histogram Diff: {histogram_diff_mean:.4f}")
        
        return VisualFeatures(
            # Signal-level features (existing, kept for backward compatibility)
            motion_intensity=float(min(motion_intensity, 1.0)),
            scene_change_count=scene_changes,
            scene_change_rate=float(scene_change_rate),
            brightness_mean=float(brightness_mean),
            brightness_std=float(brightness_std),
            color_variance=float(min(color_variance, 1.0)),
            face_presence_ratio=0.0,  # Would need face detection
            text_overlay_ratio=0.0,   # Would need OCR
            
            # Classical CV features (Step 1: Contrast)
            contrast=contrast,
            
            # Classical CV features (Step 2: Edge Detection)
            edge_density=edge_density,
            edge_intensity=edge_intensity,
            
            # Classical CV features (Step 3: Motion Estimation)
            motion_magnitude=motion_magnitude,
            
            # Classical CV features (Step 4: Histogram-Based Scene Detection)
            histogram_diff_mean=histogram_diff_mean,
            scene_boundaries=scene_boundaries
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
            # Signal-level features
            motion_intensity=motion_intensity,
            scene_change_count=scene_changes,
            scene_change_rate=scene_changes / max(duration, 0.1),
            brightness_mean=brightness_mean,
            brightness_std=0.0,
            color_variance=0.0,
            face_presence_ratio=0.0,
            text_overlay_ratio=0.0,
            # CV features (not available in simple mode)
            contrast=0.0,
            edge_density=0.0,
            edge_intensity=0.0,
            motion_magnitude=0.0,
            histogram_diff_mean=0.0,
            scene_boundaries=[]
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
            # Signal-level defaults
            motion_intensity=0.3,
            scene_change_count=0,
            scene_change_rate=0.0,
            brightness_mean=0.5,
            brightness_std=0.1,
            color_variance=0.1,
            face_presence_ratio=0.0,
            text_overlay_ratio=0.0,
            # CV feature defaults
            contrast=0.3,
            edge_density=0.0,
            edge_intensity=0.0,
            motion_magnitude=0.0,
            histogram_diff_mean=0.0,
            scene_boundaries=[]
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

