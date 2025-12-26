"""
Data Models and Schemas Module
==============================
Defines structured data representations for all pipeline stages.

This module provides:
- Type-safe dataclasses for all data entities
- Serialization/deserialization methods
- Validation utilities
- Schema documentation

These models form the contract between pipeline stages and enable:
- Reproducibility: All outputs are structured and serializable
- Inspectability: Features and scores can be examined
- Extensibility: New fields can be added without breaking existing code

Usage:
    from app.models.schemas import VideoMetadata, Segment, AnalysisResult
    
    metadata = VideoMetadata.from_file("video.mp4")
    segment = Segment(start=10.0, end=70.0, ...)
    result = AnalysisResult(segments=[segment], ...)
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal
from enum import Enum
import json
import hashlib


# =============================================================================
# ENUMS
# =============================================================================

class SegmentType(str, Enum):
    """Classification types for video segments."""
    FUNNY = "funny"
    EMOTIONAL = "emotional"
    INFORMATIVE = "informative"
    DRAMATIC = "dramatic"
    ACTION = "action"
    UNKNOWN = "unknown"


class ModalityType(str, Enum):
    """Types of modalities for feature extraction."""
    TEXT = "text"
    AUDIO = "audio"
    VISUAL = "visual"


class ProcessingStatus(str, Enum):
    """Status of processing operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# BASE CLASSES
# =============================================================================

@dataclass
class BaseModel:
    """Base class for all data models with common serialization methods."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary, handling nested objects."""
        def convert(obj):
            if isinstance(obj, BaseModel):
                return obj.to_dict()
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, (list, tuple)):
                return [convert(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        
        return {k: convert(v) for k, v in asdict(self).items()}
    
    def to_json(self, indent: int = 2) -> str:
        """Convert model to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseModel":
        """Create model from dictionary. Override in subclasses for nested objects."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> "BaseModel":
        """Create model from JSON string."""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# VIDEO METADATA
# =============================================================================

@dataclass
class VideoMetadata(BaseModel):
    """
    Metadata for a video file.
    
    Attributes:
        filename: Original filename of the video
        filepath: Full path to the video file
        duration_seconds: Total duration in seconds
        fps: Frames per second
        width: Video width in pixels
        height: Video height in pixels
        file_size_bytes: Size of the file in bytes
        codec: Video codec (e.g., "h264")
        has_audio: Whether the video has an audio track
        has_subtitles: Whether embedded subtitles were detected
        file_hash: SHA256 hash of the file for integrity verification
        created_at: Timestamp when metadata was extracted
    """
    filename: str
    filepath: str
    duration_seconds: float
    fps: float
    width: int
    height: int
    file_size_bytes: int = 0
    codec: str = "unknown"
    has_audio: bool = True
    has_subtitles: bool = False
    file_hash: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def duration_formatted(self) -> str:
        """Get duration as HH:MM:SS string."""
        hours = int(self.duration_seconds // 3600)
        minutes = int((self.duration_seconds % 3600) // 60)
        seconds = int(self.duration_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    @property
    def resolution(self) -> str:
        """Get resolution as WxH string."""
        return f"{self.width}x{self.height}"
    
    @classmethod
    def from_video_info(cls, filename: str, filepath: str, video_info: Dict) -> "VideoMetadata":
        """Create VideoMetadata from video_utils.get_video_info() output."""
        return cls(
            filename=filename,
            filepath=filepath,
            duration_seconds=video_info['duration'],
            fps=video_info['fps'],
            width=video_info['size'][0],
            height=video_info['size'][1]
        )


# =============================================================================
# SUBTITLE DATA
# =============================================================================

@dataclass
class SubtitleEntry(BaseModel):
    """
    A single subtitle entry with timing and text.
    
    Attributes:
        index: Sequential index of the subtitle
        start_seconds: Start time in seconds
        end_seconds: End time in seconds
        text: The subtitle text content
        confidence: Transcription confidence (0-1) if from Whisper
        language: Detected language code
    """
    index: int
    start_seconds: float
    end_seconds: float
    text: str
    confidence: Optional[float] = None
    language: Optional[str] = None
    
    @property
    def duration_seconds(self) -> float:
        """Calculate duration of this subtitle."""
        return self.end_seconds - self.start_seconds
    
    @property
    def words(self) -> List[str]:
        """Split text into words."""
        return self.text.split()
    
    @property
    def word_count(self) -> int:
        """Count words in the subtitle."""
        return len(self.words)


@dataclass
class SubtitleData(BaseModel):
    """
    Complete subtitle data for a video.
    
    Attributes:
        video_filename: Associated video filename
        entries: List of subtitle entries
        source: Where subtitles came from ("embedded", "whisper")
        language: Primary detected language
        total_duration_seconds: Duration covered by subtitles
        word_count: Total word count across all subtitles
        extracted_at: Timestamp of extraction
    """
    video_filename: str
    entries: List[SubtitleEntry]
    source: str  # "embedded" or "whisper"
    language: str = "unknown"
    total_duration_seconds: float = 0.0
    word_count: int = 0
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        if self.entries and self.total_duration_seconds == 0:
            self.total_duration_seconds = max(e.end_seconds for e in self.entries)
        if self.entries and self.word_count == 0:
            self.word_count = sum(e.word_count for e in self.entries)
    
    @classmethod
    def from_subtitle_list(
        cls, 
        video_filename: str,
        subtitles: List[Dict], 
        source: str,
        language: str = "unknown"
    ) -> "SubtitleData":
        """Create SubtitleData from raw subtitle list (API format)."""
        entries = [
            SubtitleEntry(
                index=i,
                start_seconds=s['start'],
                end_seconds=s['end'],
                text=s['text']
            )
            for i, s in enumerate(subtitles)
        ]
        return cls(
            video_filename=video_filename,
            entries=entries,
            source=source,
            language=language
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Override to handle nested SubtitleEntry objects."""
        return {
            'video_filename': self.video_filename,
            'entries': [e.to_dict() for e in self.entries],
            'source': self.source,
            'language': self.language,
            'total_duration_seconds': self.total_duration_seconds,
            'word_count': self.word_count,
            'extracted_at': self.extracted_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubtitleData":
        """Create from dictionary, reconstructing nested objects."""
        entries = [SubtitleEntry(**e) for e in data.get('entries', [])]
        return cls(
            video_filename=data['video_filename'],
            entries=entries,
            source=data['source'],
            language=data.get('language', 'unknown'),
            total_duration_seconds=data.get('total_duration_seconds', 0.0),
            word_count=data.get('word_count', 0),
            extracted_at=data.get('extracted_at', datetime.now().isoformat())
        )
    
    def get_text_in_range(self, start: float, end: float) -> str:
        """Get concatenated text for subtitles overlapping a time range."""
        texts = []
        for entry in self.entries:
            # Check for overlap
            if entry.start_seconds < end and entry.end_seconds > start:
                texts.append(entry.text)
        return " ".join(texts)


# =============================================================================
# FEATURE VECTORS
# =============================================================================

@dataclass
class TextFeatures(BaseModel):
    """
    Text-based features extracted from subtitles.
    
    Attributes:
        embedding: Semantic embedding vector (if computed)
        word_count: Number of words
        sentence_count: Number of sentences
        avg_word_length: Average word length
        sentiment_score: Sentiment polarity (-1 to 1)
        keyword_density: Density of important keywords
        question_count: Number of questions
        exclamation_count: Number of exclamations
    """
    embedding: Optional[List[float]] = None
    word_count: int = 0
    sentence_count: int = 0
    avg_word_length: float = 0.0
    sentiment_score: float = 0.0
    keyword_density: float = 0.0
    question_count: int = 0
    exclamation_count: int = 0


@dataclass
class AudioFeatures(BaseModel):
    """
    Audio-based features extracted from video.
    
    Attributes:
        energy_mean: Mean audio energy
        energy_std: Standard deviation of energy
        pitch_mean: Mean pitch frequency
        pitch_std: Standard deviation of pitch
        silence_ratio: Ratio of silence to total duration
        speech_rate: Words per second (if speech detected)
        volume_dynamics: Dynamic range of volume
        spectral_centroid: Center of mass of spectrum
    """
    energy_mean: float = 0.0
    energy_std: float = 0.0
    pitch_mean: float = 0.0
    pitch_std: float = 0.0
    silence_ratio: float = 0.0
    speech_rate: float = 0.0
    volume_dynamics: float = 0.0
    spectral_centroid: float = 0.0


@dataclass
class VisualFeatures(BaseModel):
    """
    Visual features extracted from video frames.
    
    Attributes:
        motion_intensity: Average motion between frames
        scene_change_count: Number of scene changes
        scene_change_rate: Scene changes per second
        brightness_mean: Average frame brightness
        brightness_std: Variation in brightness
        color_variance: Variance in color distribution
        face_presence_ratio: Ratio of frames with faces detected
        text_overlay_ratio: Ratio of frames with detected text
    """
    motion_intensity: float = 0.0
    scene_change_count: int = 0
    scene_change_rate: float = 0.0
    brightness_mean: float = 0.0
    brightness_std: float = 0.0
    color_variance: float = 0.0
    face_presence_ratio: float = 0.0
    text_overlay_ratio: float = 0.0


@dataclass 
class SegmentFeatures(BaseModel):
    """
    Complete multimodal features for a video segment.
    
    Attributes:
        segment_id: Unique identifier for this segment
        start_seconds: Segment start time
        end_seconds: Segment end time
        text_features: Text-based features (or None if disabled)
        audio_features: Audio-based features (or None if disabled)
        visual_features: Visual features (or None if disabled)
        computed_at: Timestamp when features were computed
        ablation_mode: Which ablation mode was used
    """
    segment_id: str
    start_seconds: float
    end_seconds: float
    text_features: Optional[TextFeatures] = None
    audio_features: Optional[AudioFeatures] = None
    visual_features: Optional[VisualFeatures] = None
    computed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    ablation_mode: str = "full_multimodal"
    
    @property
    def duration_seconds(self) -> float:
        """Calculate segment duration."""
        return self.end_seconds - self.start_seconds
    
    @property
    def has_text(self) -> bool:
        """Check if text features are present."""
        return self.text_features is not None
    
    @property
    def has_audio(self) -> bool:
        """Check if audio features are present."""
        return self.audio_features is not None
    
    @property
    def has_visual(self) -> bool:
        """Check if visual features are present."""
        return self.visual_features is not None
    
    @property
    def modalities_present(self) -> List[str]:
        """List which modalities have features."""
        modalities = []
        if self.has_text:
            modalities.append("text")
        if self.has_audio:
            modalities.append("audio")
        if self.has_visual:
            modalities.append("visual")
        return modalities
    
    def to_dict(self) -> Dict[str, Any]:
        """Override for nested feature objects."""
        return {
            'segment_id': self.segment_id,
            'start_seconds': self.start_seconds,
            'end_seconds': self.end_seconds,
            'text_features': self.text_features.to_dict() if self.text_features else None,
            'audio_features': self.audio_features.to_dict() if self.audio_features else None,
            'visual_features': self.visual_features.to_dict() if self.visual_features else None,
            'computed_at': self.computed_at,
            'ablation_mode': self.ablation_mode
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SegmentFeatures":
        """Create from dictionary with nested objects."""
        return cls(
            segment_id=data['segment_id'],
            start_seconds=data['start_seconds'],
            end_seconds=data['end_seconds'],
            text_features=TextFeatures(**data['text_features']) if data.get('text_features') else None,
            audio_features=AudioFeatures(**data['audio_features']) if data.get('audio_features') else None,
            visual_features=VisualFeatures(**data['visual_features']) if data.get('visual_features') else None,
            computed_at=data.get('computed_at', datetime.now().isoformat()),
            ablation_mode=data.get('ablation_mode', 'full_multimodal')
        )


# =============================================================================
# SCORING AND ANALYSIS
# =============================================================================

@dataclass
class ScoreBreakdown(BaseModel):
    """
    Breakdown of engagement score by component.
    
    Attributes:
        total_score: Final combined engagement score
        text_score: Contribution from text features
        audio_score: Contribution from audio features
        visual_score: Contribution from visual features
        text_weight: Weight applied to text score
        audio_weight: Weight applied to audio score
        visual_weight: Weight applied to visual score
        scoring_method: Method used ("rule_based", "learned", "hybrid")
    """
    total_score: float
    text_score: float = 0.0
    audio_score: float = 0.0
    visual_score: float = 0.0
    text_weight: float = 0.0
    audio_weight: float = 0.0
    visual_weight: float = 0.0
    scoring_method: str = "rule_based"
    
    @property
    def weighted_text(self) -> float:
        """Get weighted text contribution."""
        return self.text_score * self.text_weight
    
    @property
    def weighted_audio(self) -> float:
        """Get weighted audio contribution."""
        return self.audio_score * self.audio_weight
    
    @property
    def weighted_visual(self) -> float:
        """Get weighted visual contribution."""
        return self.visual_score * self.visual_weight


@dataclass
class Segment(BaseModel):
    """
    A video segment identified for potential extraction.
    
    This is the core output unit of the analysis pipeline.
    
    Attributes:
        segment_id: Unique identifier
        start_seconds: Start time in seconds
        end_seconds: End time in seconds
        segment_type: Classification (funny, emotional, etc.)
        confidence: Confidence in the classification (0-1)
        features: Extracted multimodal features
        score: Engagement score breakdown
        rank: Ranking among all segments (1 = best)
        text_preview: Preview of subtitle text in this segment
        source: How segment was identified ("gemini", "rule_based", "learned")
    """
    segment_id: str
    start_seconds: float
    end_seconds: float
    segment_type: str = "unknown"
    confidence: float = 0.0
    features: Optional[SegmentFeatures] = None
    score: Optional[ScoreBreakdown] = None
    rank: int = 0
    text_preview: str = ""
    source: str = "gemini"
    
    @property
    def duration_seconds(self) -> float:
        """Calculate segment duration."""
        return self.end_seconds - self.start_seconds
    
    @property
    def duration_formatted(self) -> str:
        """Get duration as MM:SS string."""
        minutes = int(self.duration_seconds // 60)
        seconds = int(self.duration_seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    @property
    def time_range_formatted(self) -> str:
        """Get time range as 'MM:SS - MM:SS' string."""
        start_min = int(self.start_seconds // 60)
        start_sec = int(self.start_seconds % 60)
        end_min = int(self.end_seconds // 60)
        end_sec = int(self.end_seconds % 60)
        return f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Override for nested objects."""
        return {
            'segment_id': self.segment_id,
            'start_seconds': self.start_seconds,
            'end_seconds': self.end_seconds,
            'segment_type': self.segment_type,
            'confidence': self.confidence,
            'features': self.features.to_dict() if self.features else None,
            'score': self.score.to_dict() if self.score else None,
            'rank': self.rank,
            'text_preview': self.text_preview,
            'source': self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Segment":
        """Create from dictionary."""
        return cls(
            segment_id=data['segment_id'],
            start_seconds=data['start_seconds'],
            end_seconds=data['end_seconds'],
            segment_type=data.get('segment_type', 'unknown'),
            confidence=data.get('confidence', 0.0),
            features=SegmentFeatures.from_dict(data['features']) if data.get('features') else None,
            score=ScoreBreakdown(**data['score']) if data.get('score') else None,
            rank=data.get('rank', 0),
            text_preview=data.get('text_preview', ''),
            source=data.get('source', 'gemini')
        )
    
    @classmethod
    def from_gemini_response(cls, response: Dict, segment_id: str) -> "Segment":
        """Create Segment from Gemini API response format."""
        return cls(
            segment_id=segment_id,
            start_seconds=float(response['start']),
            end_seconds=float(response['end']),
            segment_type=response.get('type', 'unknown'),
            source='gemini'
        )


# =============================================================================
# PIPELINE OUTPUTS
# =============================================================================

@dataclass
class AnalysisResult(BaseModel):
    """
    Complete result of video analysis pipeline.
    
    This is the primary output format for the entire pipeline.
    
    Attributes:
        result_id: Unique identifier for this analysis run
        video_metadata: Metadata about the analyzed video
        subtitle_data: Extracted subtitle information
        segments: List of identified segments, ranked by score
        config_snapshot: Copy of configuration used
        processing_time_seconds: Total processing time
        ablation_mode: Which modalities were used
        created_at: Timestamp of analysis
        version: Schema version for compatibility
    """
    result_id: str
    video_metadata: VideoMetadata
    subtitle_data: Optional[SubtitleData] = None
    segments: List[Segment] = field(default_factory=list)
    config_snapshot: Optional[Dict] = None
    processing_time_seconds: float = 0.0
    ablation_mode: str = "full_multimodal"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0.0"
    
    @property
    def segment_count(self) -> int:
        """Get number of segments identified."""
        return len(self.segments)
    
    @property
    def top_segment(self) -> Optional[Segment]:
        """Get the highest-ranked segment."""
        if not self.segments:
            return None
        return min(self.segments, key=lambda s: s.rank if s.rank > 0 else float('inf'))
    
    def get_segments_by_type(self, segment_type: str) -> List[Segment]:
        """Filter segments by type."""
        return [s for s in self.segments if s.segment_type == segment_type]
    
    def get_top_n_segments(self, n: int) -> List[Segment]:
        """Get top N segments by rank."""
        ranked = sorted(self.segments, key=lambda s: s.rank if s.rank > 0 else float('inf'))
        return ranked[:n]
    
    def to_dict(self) -> Dict[str, Any]:
        """Override for nested objects."""
        return {
            'result_id': self.result_id,
            'video_metadata': self.video_metadata.to_dict(),
            'subtitle_data': self.subtitle_data.to_dict() if self.subtitle_data else None,
            'segments': [s.to_dict() for s in self.segments],
            'config_snapshot': self.config_snapshot,
            'processing_time_seconds': self.processing_time_seconds,
            'ablation_mode': self.ablation_mode,
            'created_at': self.created_at,
            'version': self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisResult":
        """Create from dictionary."""
        return cls(
            result_id=data['result_id'],
            video_metadata=VideoMetadata(**data['video_metadata']),
            subtitle_data=SubtitleData.from_dict(data['subtitle_data']) if data.get('subtitle_data') else None,
            segments=[Segment.from_dict(s) for s in data.get('segments', [])],
            config_snapshot=data.get('config_snapshot'),
            processing_time_seconds=data.get('processing_time_seconds', 0.0),
            ablation_mode=data.get('ablation_mode', 'full_multimodal'),
            created_at=data.get('created_at', datetime.now().isoformat()),
            version=data.get('version', '1.0.0')
        )
    
    def save(self, filepath: str) -> None:
        """Save analysis result to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "AnalysisResult":
        """Load analysis result from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


# =============================================================================
# EVALUATION DATA
# =============================================================================

@dataclass
class GroundTruthSegment(BaseModel):
    """
    Human-annotated ground truth segment for evaluation.
    
    Attributes:
        segment_id: Unique identifier
        start_seconds: Start time
        end_seconds: End time
        annotator_id: ID of the human annotator
        engagement_rating: Human rating (1-5 or 0-1)
        segment_type: Human-assigned type
        notes: Optional annotator notes
    """
    segment_id: str
    start_seconds: float
    end_seconds: float
    annotator_id: str = "unknown"
    engagement_rating: float = 0.0
    segment_type: str = "unknown"
    notes: str = ""


@dataclass
class EvaluationResult(BaseModel):
    """
    Evaluation metrics comparing predicted vs ground truth segments.
    
    Attributes:
        result_id: Links to AnalysisResult
        ground_truth_segments: Human-annotated segments
        predicted_segments: System-predicted segments
        iou_scores: Intersection over Union for each prediction
        precision: Precision metric
        recall: Recall metric
        f1_score: F1 score
        rank_correlation: Correlation between predicted and actual rankings
        evaluated_at: Timestamp of evaluation
    """
    result_id: str
    ground_truth_segments: List[GroundTruthSegment] = field(default_factory=list)
    predicted_segments: List[Segment] = field(default_factory=list)
    iou_scores: List[float] = field(default_factory=list)
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    rank_correlation: float = 0.0
    evaluated_at: str = field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_segment_id(video_filename: str, start: float, end: float) -> str:
    """Generate a unique segment ID based on video and time range."""
    content = f"{video_filename}_{start:.2f}_{end:.2f}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def generate_result_id(video_filename: str) -> str:
    """Generate a unique result ID."""
    content = f"{video_filename}_{datetime.now().isoformat()}"
    return hashlib.md5(content.encode()).hexdigest()[:16]

