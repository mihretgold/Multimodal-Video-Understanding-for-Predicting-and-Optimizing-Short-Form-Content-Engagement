"""
Segmentation Strategies Module
==============================
Implements different temporal segmentation algorithms.

RESEARCH CONTEXT:
-----------------
Temporal segmentation is crucial for short-form content because:
1. Arbitrary cuts create incoherent clips
2. Natural boundaries preserve narrative flow
3. Platform constraints (60-70s) must be respected

STRATEGY COMPARISON:
-------------------

FIXED-WINDOW (baseline):
  - Pro: Simple, deterministic, full coverage
  - Con: Cuts mid-sentence, ignores content structure
  - Use: When speed matters more than quality

PAUSE-BASED (our default):
  - Pro: Respects natural speech patterns
  - Con: Pauses don't always align with topic changes
  - Use: Dialogue-heavy content (interviews, vlogs)
  
SEMANTIC-BOUNDARY:
  - Pro: Groups coherent topics together
  - Con: Requires text understanding, computationally heavier
  - Use: Educational/informative content

HYBRID:
  - Pro: Combines multiple signals for robustness
  - Con: More complex, harder to tune
  - Use: Production systems with quality requirements

RESEARCH DECISION: We default to PAUSE-BASED because:
1. Speech pauses correlate with natural breakpoints
2. Computationally efficient (uses subtitle timestamps)
3. Works well without semantic understanding
4. Produces coherent clips for most content types

Strategies:
1. PauseBasedStrategy - Uses speech pauses as segment boundaries
2. FixedWindowStrategy - Regular intervals with boundary optimization
3. SemanticBoundaryStrategy - Groups semantically coherent content
4. HybridStrategy - Combines multiple strategies with voting
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import logging

from .boundaries import BoundaryDetector, SpeechBoundary, BoundaryType
from ..models import SubtitleData, Segment, generate_segment_id

logger = logging.getLogger(__name__)


@dataclass
class SegmentationParams:
    """Parameters for segmentation algorithms."""
    
    # Target duration constraints
    min_duration_seconds: float = 60.0
    target_duration_seconds: float = 65.0
    max_duration_seconds: float = 70.0
    
    # Segment count constraints
    min_segments: int = 1
    max_segments: int = 20
    
    # Strategy-specific parameters
    overlap_allowed: bool = False
    prefer_natural_boundaries: bool = True
    boundary_tolerance_seconds: float = 5.0  # How far from target to look for boundary
    
    @classmethod
    def from_config(cls, config) -> "SegmentationParams":
        """Create parameters from SegmentationConfig."""
        return cls(
            min_duration_seconds=config.min_duration_seconds,
            target_duration_seconds=config.target_duration_seconds,
            max_duration_seconds=config.max_duration_seconds,
            min_segments=config.min_segments,
            max_segments=config.max_segments,
        )


class SegmentationStrategy(ABC):
    """
    Abstract base class for segmentation strategies.
    
    Strategies take subtitle data and parameters, returning a list of segments.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging."""
        pass
    
    @property
    def description(self) -> str:
        """Strategy description."""
        return f"Segmentation strategy: {self.name}"
    
    @abstractmethod
    def segment(
        self,
        subtitle_data: SubtitleData,
        video_duration: float,
        params: SegmentationParams
    ) -> List[Segment]:
        """
        Generate segments from subtitle data.
        
        Args:
            subtitle_data: Transcription data with timestamps
            video_duration: Total video duration in seconds
            params: Segmentation parameters
            
        Returns:
            List of Segment objects
        """
        pass
    
    def _create_segment(
        self,
        start: float,
        end: float,
        subtitle_data: SubtitleData,
        segment_type: str = "auto",
        source: str = None
    ) -> Segment:
        """Helper to create a segment with proper ID and text preview."""
        segment = Segment(
            segment_id=generate_segment_id(subtitle_data.video_filename, start, end),
            start_seconds=start,
            end_seconds=end,
            segment_type=segment_type,
            source=source or f"strategy:{self.name}"
        )
        
        # Add text preview
        segment.text_preview = subtitle_data.get_text_in_range(start, end)[:200]
        
        return segment
    
    def _estimate_segments_needed(
        self,
        video_duration: float,
        params: SegmentationParams
    ) -> int:
        """Estimate how many segments should be created."""
        # Calculate based on target duration
        ideal_count = int(video_duration / params.target_duration_seconds)
        return max(params.min_segments, min(params.max_segments, ideal_count))


class PauseBasedStrategy(SegmentationStrategy):
    """
    Segmentation based on natural speech pauses.
    
    This strategy:
    1. Detects pause boundaries in the transcript
    2. Selects boundaries that create segments closest to target duration
    3. Adjusts to respect min/max duration constraints
    """
    
    @property
    def name(self) -> str:
        return "pause_based"
    
    def __init__(
        self,
        min_gap_seconds: float = 0.5,
        strong_gap_seconds: float = 1.5
    ):
        self.boundary_detector = BoundaryDetector(
            min_gap_seconds=min_gap_seconds,
            strong_gap_seconds=strong_gap_seconds
        )
    
    def segment(
        self,
        subtitle_data: SubtitleData,
        video_duration: float,
        params: SegmentationParams
    ) -> List[Segment]:
        """Generate segments using speech pause boundaries."""
        
        logger.info(f"Running pause-based segmentation on {video_duration:.1f}s video")
        
        # Detect all boundaries
        boundaries = self.boundary_detector.detect_from_subtitles(
            subtitle_data.entries,
            video_duration
        )
        
        logger.info(f"Detected {len(boundaries)} potential boundaries")
        
        if not boundaries:
            # Fallback to single segment or fixed windows
            logger.warning("No boundaries detected, using full video as single segment")
            return [self._create_segment(0, min(video_duration, params.max_duration_seconds), subtitle_data)]
        
        # Use dynamic programming to find optimal segment boundaries
        segments = self._find_optimal_segments(
            boundaries,
            video_duration,
            params
        )
        
        # Create segment objects
        result = []
        for start, end in segments:
            segment = self._create_segment(
                start, end, subtitle_data,
                segment_type="auto",
                source=f"strategy:{self.name}"
            )
            result.append(segment)
        
        logger.info(f"Created {len(result)} segments")
        return result
    
    def _find_optimal_segments(
        self,
        boundaries: List[SpeechBoundary],
        video_duration: float,
        params: SegmentationParams
    ) -> List[Tuple[float, float]]:
        """
        Find optimal segment boundaries using greedy approach.
        
        For each position, find the best boundary that creates a segment
        closest to the target duration.
        """
        segments = []
        current_start = 0.0
        boundary_times = [0.0] + [b.timestamp for b in boundaries] + [video_duration]
        boundary_strengths = {b.timestamp: b.strength for b in boundaries}
        
        segment_count = 0
        max_segments = self._estimate_segments_needed(video_duration, params)
        
        while current_start < video_duration - params.min_duration_seconds:
            if segment_count >= max_segments:
                break
            
            # Find target end time
            target_end = current_start + params.target_duration_seconds
            
            # Find boundaries near the target
            candidates = []
            for bt in boundary_times:
                if bt <= current_start:
                    continue
                    
                duration = bt - current_start
                
                # Check if duration is within acceptable range
                if params.min_duration_seconds <= duration <= params.max_duration_seconds:
                    # Score based on proximity to target and boundary strength
                    distance_penalty = abs(duration - params.target_duration_seconds) / params.target_duration_seconds
                    strength_bonus = boundary_strengths.get(bt, 0.5)
                    score = (1 - distance_penalty) * 0.7 + strength_bonus * 0.3
                    candidates.append((bt, score, duration))
                
                # Stop looking if we're past max duration
                if duration > params.max_duration_seconds:
                    break
            
            if not candidates:
                # No valid boundary found, try extending the search
                # or use a fixed end point
                end = min(current_start + params.target_duration_seconds, video_duration)
                if end - current_start >= params.min_duration_seconds:
                    segments.append((current_start, end))
                    segment_count += 1
                current_start = end
                continue
            
            # Select best candidate
            best = max(candidates, key=lambda x: x[1])
            end = best[0]
            
            segments.append((current_start, end))
            segment_count += 1
            current_start = end
        
        return segments


class FixedWindowStrategy(SegmentationStrategy):
    """
    Fixed-window segmentation with boundary optimization.
    
    This strategy:
    1. Creates regular intervals of target duration
    2. Snaps segment boundaries to nearby natural boundaries
    """
    
    @property
    def name(self) -> str:
        return "fixed_window"
    
    def __init__(self, snap_to_boundaries: bool = True):
        self.snap_to_boundaries = snap_to_boundaries
        if snap_to_boundaries:
            self.boundary_detector = BoundaryDetector()
    
    def segment(
        self,
        subtitle_data: SubtitleData,
        video_duration: float,
        params: SegmentationParams
    ) -> List[Segment]:
        """Generate fixed-duration segments with optional boundary snapping."""
        
        logger.info(f"Running fixed-window segmentation on {video_duration:.1f}s video")
        
        # Detect boundaries for snapping
        boundaries = []
        if self.snap_to_boundaries:
            boundaries = self.boundary_detector.detect_from_subtitles(
                subtitle_data.entries,
                video_duration
            )
        
        # Create fixed intervals
        segments = []
        current_start = 0.0
        segment_count = 0
        max_segments = self._estimate_segments_needed(video_duration, params)
        
        while current_start < video_duration - params.min_duration_seconds:
            if segment_count >= max_segments:
                break
            
            target_end = current_start + params.target_duration_seconds
            
            # Snap to nearby boundary if enabled
            if boundaries and self.snap_to_boundaries:
                best_boundary = self._find_nearest_boundary(
                    target_end,
                    boundaries,
                    params.boundary_tolerance_seconds
                )
                if best_boundary:
                    target_end = best_boundary.timestamp
            
            # Clamp to valid range
            end = min(target_end, video_duration)
            end = max(end, current_start + params.min_duration_seconds)
            
            segment = self._create_segment(
                current_start, end, subtitle_data,
                segment_type="auto",
                source=f"strategy:{self.name}"
            )
            segments.append(segment)
            segment_count += 1
            
            current_start = end
        
        logger.info(f"Created {len(segments)} segments")
        return segments
    
    def _find_nearest_boundary(
        self,
        target_time: float,
        boundaries: List[SpeechBoundary],
        tolerance: float
    ) -> Optional[SpeechBoundary]:
        """Find the best boundary within tolerance of target time."""
        candidates = [
            b for b in boundaries
            if abs(b.timestamp - target_time) <= tolerance
        ]
        
        if not candidates:
            return None
        
        # Prefer stronger boundaries that are closer
        def score(b):
            distance = abs(b.timestamp - target_time)
            return b.strength * (1 - distance / tolerance)
        
        return max(candidates, key=score)


class SemanticBoundaryStrategy(SegmentationStrategy):
    """
    Segmentation based on semantic coherence.
    
    This strategy groups subtitle entries that belong together semantically,
    creating segments around topic or scene changes.
    
    Note: This is a simplified version. Full implementation would use
    sentence embeddings to detect semantic shifts.
    """
    
    @property
    def name(self) -> str:
        return "semantic_boundary"
    
    def __init__(self):
        self.boundary_detector = BoundaryDetector()
    
    def segment(
        self,
        subtitle_data: SubtitleData,
        video_duration: float,
        params: SegmentationParams
    ) -> List[Segment]:
        """Generate segments based on semantic coherence."""
        
        logger.info(f"Running semantic-boundary segmentation on {video_duration:.1f}s video")
        
        # Detect all boundaries
        boundaries = self.boundary_detector.detect_from_subtitles(
            subtitle_data.entries,
            video_duration
        )
        
        # Enhance boundary detection with semantic features
        enhanced_boundaries = self._enhance_with_semantic_features(
            boundaries,
            subtitle_data
        )
        
        # Use enhanced boundaries for segmentation
        # For now, use pause-based approach with enhanced boundaries
        segments = []
        current_start = 0.0
        segment_count = 0
        max_segments = self._estimate_segments_needed(video_duration, params)
        
        # Get top boundaries based on strength
        top_boundaries = self.boundary_detector.get_top_boundaries(
            enhanced_boundaries,
            count=max_segments * 2,
            min_separation=params.min_duration_seconds * 0.8
        )
        
        # Create segments between top boundaries
        boundary_times = [0.0] + [b.timestamp for b in top_boundaries]
        
        for i in range(len(boundary_times) - 1):
            if segment_count >= max_segments:
                break
            
            start = boundary_times[i]
            end = boundary_times[i + 1] if i + 1 < len(boundary_times) else video_duration
            
            # Check duration constraints
            duration = end - start
            if duration < params.min_duration_seconds:
                # Merge with next segment
                continue
            elif duration > params.max_duration_seconds:
                # Split the segment
                while start < end and segment_count < max_segments:
                    segment_end = min(start + params.target_duration_seconds, end)
                    if segment_end - start >= params.min_duration_seconds:
                        segment = self._create_segment(
                            start, segment_end, subtitle_data,
                            segment_type="auto",
                            source=f"strategy:{self.name}"
                        )
                        segments.append(segment)
                        segment_count += 1
                    start = segment_end
            else:
                segment = self._create_segment(
                    start, end, subtitle_data,
                    segment_type="auto",
                    source=f"strategy:{self.name}"
                )
                segments.append(segment)
                segment_count += 1
        
        logger.info(f"Created {len(segments)} segments")
        return segments
    
    def _enhance_with_semantic_features(
        self,
        boundaries: List[SpeechBoundary],
        subtitle_data: SubtitleData
    ) -> List[SpeechBoundary]:
        """
        Enhance boundary strength using semantic features.
        
        Currently uses heuristics:
        - Question marks indicate topic shifts
        - Exclamation marks indicate emphasis/climax
        - Significant vocabulary changes between entries
        """
        enhanced = []
        
        for boundary in boundaries:
            # Find context around this boundary
            before_text = subtitle_data.get_text_in_range(
                max(0, boundary.timestamp - 10),
                boundary.timestamp
            )
            after_text = subtitle_data.get_text_in_range(
                boundary.timestamp,
                boundary.timestamp + 10
            )
            
            # Calculate semantic shift indicators
            bonus = 0.0
            
            # Questions often mark topic transitions
            if before_text.count('?') > 0:
                bonus += 0.1
            
            # Exclamations mark emotional peaks (good clip points)
            if before_text.count('!') > 0:
                bonus += 0.05
            
            # Vocabulary diversity change (simplified)
            before_words = set(before_text.lower().split())
            after_words = set(after_text.lower().split())
            overlap = len(before_words & after_words) / max(1, len(before_words | after_words))
            
            # Lower overlap suggests topic change
            if overlap < 0.3:
                bonus += 0.15
            
            # Create enhanced boundary
            enhanced_boundary = SpeechBoundary(
                timestamp=boundary.timestamp,
                boundary_type=boundary.boundary_type,
                strength=min(1.0, boundary.strength + bonus),
                context=boundary.context
            )
            enhanced.append(enhanced_boundary)
        
        return enhanced


class HybridStrategy(SegmentationStrategy):
    """
    Combines multiple segmentation strategies with voting.
    
    This strategy runs multiple approaches and combines their results
    to find consensus segment boundaries.
    """
    
    @property
    def name(self) -> str:
        return "hybrid"
    
    def __init__(
        self,
        strategies: List[SegmentationStrategy] = None,
        weights: List[float] = None
    ):
        """
        Initialize hybrid strategy.
        
        Args:
            strategies: List of strategies to combine
            weights: Optional weights for each strategy
        """
        self.strategies = strategies or [
            PauseBasedStrategy(),
            FixedWindowStrategy(snap_to_boundaries=True),
        ]
        
        if weights:
            if len(weights) != len(self.strategies):
                raise ValueError("Weights must match number of strategies")
            self.weights = weights
        else:
            self.weights = [1.0 / len(self.strategies)] * len(self.strategies)
    
    def segment(
        self,
        subtitle_data: SubtitleData,
        video_duration: float,
        params: SegmentationParams
    ) -> List[Segment]:
        """Generate segments using combined strategies."""
        
        logger.info(f"Running hybrid segmentation with {len(self.strategies)} strategies")
        
        # Run all strategies
        all_results = []
        for i, strategy in enumerate(self.strategies):
            try:
                segments = strategy.segment(subtitle_data, video_duration, params)
                all_results.append((segments, self.weights[i]))
                logger.info(f"Strategy '{strategy.name}' produced {len(segments)} segments")
            except Exception as e:
                logger.warning(f"Strategy '{strategy.name}' failed: {e}")
        
        if not all_results:
            logger.error("All strategies failed")
            return []
        
        # Find consensus boundaries
        consensus_segments = self._find_consensus(all_results, params)
        
        # Create segment objects
        result = []
        for start, end in consensus_segments:
            segment = self._create_segment(
                start, end, subtitle_data,
                segment_type="auto",
                source=f"strategy:{self.name}"
            )
            result.append(segment)
        
        logger.info(f"Hybrid strategy created {len(result)} consensus segments")
        return result
    
    def _find_consensus(
        self,
        results: List[Tuple[List[Segment], float]],
        params: SegmentationParams
    ) -> List[Tuple[float, float]]:
        """
        Find consensus segment boundaries from multiple strategies.
        
        Uses boundary voting: boundaries that appear in multiple strategies
        are preferred.
        """
        # Collect all boundaries with weights
        boundary_votes: Dict[float, float] = {}
        tolerance = 3.0  # Seconds
        
        for segments, weight in results:
            for segment in segments:
                # Vote for start boundary
                self._add_vote(boundary_votes, segment.start_seconds, weight, tolerance)
                # Vote for end boundary
                self._add_vote(boundary_votes, segment.end_seconds, weight, tolerance)
        
        # Sort boundaries by vote strength
        sorted_boundaries = sorted(
            boundary_votes.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select top boundaries with minimum separation
        selected = [0.0]  # Always start at 0
        for time, votes in sorted_boundaries:
            if all(abs(time - s) >= params.min_duration_seconds for s in selected):
                selected.append(time)
        selected.sort()
        
        # Create segments from consecutive boundaries
        segments = []
        for i in range(len(selected) - 1):
            start, end = selected[i], selected[i + 1]
            if params.min_duration_seconds <= end - start <= params.max_duration_seconds:
                segments.append((start, end))
        
        return segments
    
    def _add_vote(
        self,
        votes: Dict[float, float],
        time: float,
        weight: float,
        tolerance: float
    ) -> None:
        """Add a vote for a boundary, merging nearby times."""
        # Find existing nearby boundary
        for existing in list(votes.keys()):
            if abs(existing - time) < tolerance:
                # Merge: increase vote and adjust time to weighted average
                total_weight = votes[existing] + weight
                new_time = (existing * votes[existing] + time * weight) / total_weight
                del votes[existing]
                votes[new_time] = total_weight
                return
        
        # New boundary
        votes[time] = weight

