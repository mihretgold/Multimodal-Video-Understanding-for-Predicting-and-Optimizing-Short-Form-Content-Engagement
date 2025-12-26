"""
Boundary Detection Module
=========================
Detects natural segment boundaries in video content.

Boundary Types:
- Speech pauses (gaps between subtitle entries)
- Sentence endings (periods, question marks)
- Topic shifts (semantic changes)
- Silence regions (audio-based)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import re
from enum import Enum


class BoundaryType(Enum):
    """Types of segment boundaries."""
    SPEECH_PAUSE = "speech_pause"       # Gap between spoken words
    SENTENCE_END = "sentence_end"       # End of sentence punctuation
    TOPIC_SHIFT = "topic_shift"         # Semantic topic change
    SILENCE = "silence"                 # Audio silence region
    MANUAL = "manual"                   # Manually specified


@dataclass
class SpeechBoundary:
    """
    Represents a potential segment boundary.
    
    Attributes:
        timestamp: Time in seconds where boundary occurs
        boundary_type: Type of boundary detected
        strength: Confidence/strength of boundary (0-1)
        context: Optional text context around the boundary
    """
    timestamp: float
    boundary_type: BoundaryType
    strength: float = 0.5
    context: str = ""
    
    def __lt__(self, other):
        """Allow sorting by timestamp."""
        return self.timestamp < other.timestamp


class BoundaryDetector:
    """
    Detects potential segment boundaries from subtitle data.
    
    Uses multiple heuristics:
    1. Gap duration between subtitle entries
    2. Sentence-ending punctuation
    3. Question marks (natural pause points)
    4. Paragraph breaks in text
    """
    
    def __init__(
        self,
        min_gap_seconds: float = 0.5,
        strong_gap_seconds: float = 1.5,
        sentence_end_bonus: float = 0.2,
        question_bonus: float = 0.15
    ):
        """
        Initialize boundary detector.
        
        Args:
            min_gap_seconds: Minimum gap to consider a boundary
            strong_gap_seconds: Gap length that indicates strong boundary
            sentence_end_bonus: Extra strength for sentence-ending boundaries
            question_bonus: Extra strength for question-ending boundaries
        """
        self.min_gap_seconds = min_gap_seconds
        self.strong_gap_seconds = strong_gap_seconds
        self.sentence_end_bonus = sentence_end_bonus
        self.question_bonus = question_bonus
        
        # Sentence-ending patterns
        self.sentence_end_pattern = re.compile(r'[.!?][\s]*$')
        self.question_pattern = re.compile(r'\?[\s]*$')
    
    def detect_from_subtitles(
        self,
        subtitle_entries: List,
        video_duration: float
    ) -> List[SpeechBoundary]:
        """
        Detect boundaries from subtitle entries.
        
        Args:
            subtitle_entries: List of SubtitleEntry objects
            video_duration: Total video duration in seconds
            
        Returns:
            List of SpeechBoundary objects, sorted by timestamp
        """
        boundaries = []
        
        if not subtitle_entries:
            return boundaries
        
        # Analyze gaps between consecutive entries
        for i in range(len(subtitle_entries) - 1):
            current = subtitle_entries[i]
            next_entry = subtitle_entries[i + 1]
            
            # Calculate gap
            gap = next_entry.start_seconds - current.end_seconds
            
            if gap >= self.min_gap_seconds:
                # Calculate boundary strength based on gap duration
                strength = self._calculate_gap_strength(gap)
                
                # Check for sentence endings
                text = current.text.strip()
                if self.sentence_end_pattern.search(text):
                    strength = min(1.0, strength + self.sentence_end_bonus)
                    
                if self.question_pattern.search(text):
                    strength = min(1.0, strength + self.question_bonus)
                
                # Boundary at the end of the current entry
                boundary = SpeechBoundary(
                    timestamp=current.end_seconds,
                    boundary_type=BoundaryType.SPEECH_PAUSE,
                    strength=strength,
                    context=text[-50:] if len(text) > 50 else text
                )
                boundaries.append(boundary)
        
        # Add sentence boundaries within entries
        for entry in subtitle_entries:
            inner_boundaries = self._detect_inner_boundaries(entry)
            boundaries.extend(inner_boundaries)
        
        # Sort by timestamp and merge nearby boundaries
        boundaries.sort()
        boundaries = self._merge_nearby_boundaries(boundaries)
        
        return boundaries
    
    def _calculate_gap_strength(self, gap_seconds: float) -> float:
        """
        Calculate boundary strength from gap duration.
        
        Uses a sigmoid-like function centered at strong_gap_seconds.
        """
        if gap_seconds <= self.min_gap_seconds:
            return 0.3
        elif gap_seconds >= self.strong_gap_seconds:
            return 0.9
        else:
            # Linear interpolation between min and strong
            t = (gap_seconds - self.min_gap_seconds) / (self.strong_gap_seconds - self.min_gap_seconds)
            return 0.3 + 0.6 * t
    
    def _detect_inner_boundaries(self, entry) -> List[SpeechBoundary]:
        """Detect sentence boundaries within a subtitle entry."""
        boundaries = []
        text = entry.text
        
        # Find all sentence-ending positions
        for match in re.finditer(r'[.!?][\s]+', text):
            # Estimate timestamp within entry
            position_ratio = match.end() / len(text)
            timestamp = entry.start_seconds + position_ratio * (entry.end_seconds - entry.start_seconds)
            
            strength = 0.4  # Base strength for inner boundaries
            if '?' in match.group():
                strength += self.question_bonus
            
            boundaries.append(SpeechBoundary(
                timestamp=timestamp,
                boundary_type=BoundaryType.SENTENCE_END,
                strength=strength,
                context=text[max(0, match.start()-20):min(len(text), match.end()+20)]
            ))
        
        return boundaries
    
    def _merge_nearby_boundaries(
        self,
        boundaries: List[SpeechBoundary],
        merge_threshold: float = 0.5
    ) -> List[SpeechBoundary]:
        """
        Merge boundaries that are very close together.
        
        Keeps the stronger boundary when merging.
        """
        if not boundaries:
            return []
        
        merged = [boundaries[0]]
        
        for boundary in boundaries[1:]:
            prev = merged[-1]
            
            if boundary.timestamp - prev.timestamp < merge_threshold:
                # Keep the stronger one
                if boundary.strength > prev.strength:
                    merged[-1] = boundary
            else:
                merged.append(boundary)
        
        return merged
    
    def get_top_boundaries(
        self,
        boundaries: List[SpeechBoundary],
        count: int = 10,
        min_separation: float = 30.0
    ) -> List[SpeechBoundary]:
        """
        Get the top N strongest boundaries with minimum separation.
        
        Args:
            boundaries: List of detected boundaries
            count: Maximum number of boundaries to return
            min_separation: Minimum seconds between selected boundaries
            
        Returns:
            List of top boundaries, sorted by timestamp
        """
        # Sort by strength (descending)
        sorted_by_strength = sorted(boundaries, key=lambda b: b.strength, reverse=True)
        
        selected = []
        for boundary in sorted_by_strength:
            if len(selected) >= count:
                break
            
            # Check separation from already selected
            too_close = False
            for s in selected:
                if abs(boundary.timestamp - s.timestamp) < min_separation:
                    too_close = True
                    break
            
            if not too_close:
                selected.append(boundary)
        
        # Sort by timestamp for output
        selected.sort()
        return selected


def detect_silence_regions(
    audio_path: str,
    threshold_db: float = -40.0,
    min_duration: float = 0.5
) -> List[Tuple[float, float]]:
    """
    Detect silence regions in audio file.
    
    Note: This is a placeholder that returns empty list.
    Full implementation requires audio processing libraries.
    
    Args:
        audio_path: Path to audio file
        threshold_db: Silence threshold in decibels
        min_duration: Minimum silence duration to detect
        
    Returns:
        List of (start_seconds, end_seconds) tuples for silence regions
    """
    # Placeholder - will be implemented in Step 3 with audio feature extraction
    return []

