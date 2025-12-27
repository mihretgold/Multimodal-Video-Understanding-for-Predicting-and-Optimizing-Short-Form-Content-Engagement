"""
Configuration Management Module
===============================
Centralized configuration system for the Movie Shorts research pipeline.

This module provides:
- Type-safe configuration via dataclasses
- Environment variable overrides
- Experiment configuration support
- Default values with documentation

Usage:
    from app.config import get_config
    config = get_config()
    
    # Access configuration
    model_size = config.whisper.model_size
    target_duration = config.segmentation.target_duration_seconds

For experiments, create a config file and load it:
    config = load_experiment_config("experiments/ablation_text_only.yaml")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal
import os
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class PathConfig:
    """Configuration for file system paths."""
    
    # Base directory (defaults to app directory)
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    
    # Runtime directories
    upload_dir: str = "uploads"
    cuts_dir: str = "cuts"
    subtitles_dir: str = "subtitles"
    
    # Research output directories
    features_dir: str = "features"
    logs_dir: str = "logs"
    experiments_dir: str = "experiments"
    
    @property
    def uploads(self) -> Path:
        return self.base_dir / self.upload_dir
    
    @property
    def cuts(self) -> Path:
        return self.base_dir / self.cuts_dir
    
    @property
    def subtitles(self) -> Path:
        return self.base_dir / self.subtitles_dir
    
    @property
    def features(self) -> Path:
        return self.base_dir / self.features_dir
    
    @property
    def logs(self) -> Path:
        return self.base_dir / self.logs_dir
    
    @property
    def experiments(self) -> Path:
        return self.base_dir / self.experiments_dir
    
    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for dir_path in [self.uploads, self.cuts, self.subtitles, 
                         self.features, self.logs, self.experiments]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class WhisperConfig:
    """
    Configuration for Whisper speech-to-text model.
    
    RESEARCH DECISION: We use 'small' as default for balance of speed/accuracy.
    - tiny/base: Fast but miss nuances important for engagement detection
    - small: Good trade-off, ~2x realtime on CPU (our choice)
    - medium/large: Better accuracy but 10x+ slower
    
    The transcription quality directly impacts text feature extraction,
    especially for sentiment analysis and keyword detection.
    """
    
    # Model size: tiny, base, small, medium, large, large-v2, large-v3
    # RESEARCH NOTE: 'small' achieves ~10% WER while being practical for iteration
    model_size: str = "small"
    
    # Device configuration
    device: str = "auto"  # "auto", "cpu", "cuda"
    compute_type: str = "auto"  # "auto", "int8", "float16", "float32"
    
    # Transcription parameters
    language: Optional[str] = None  # None for auto-detect
    task: str = "transcribe"  # "transcribe" or "translate"
    
    # VAD (Voice Activity Detection) parameters
    vad_filter: bool = True
    vad_parameters: dict = field(default_factory=lambda: {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 100
    })


@dataclass
class GeminiConfig:
    """Configuration for Google Gemini LLM."""
    
    # Model selection
    model_name: str = "gemini-1.5-flash"
    
    # Generation parameters
    temperature: float = 0.7
    top_k: int = 1
    top_p: float = 0.8
    max_output_tokens: int = 2048
    
    # API configuration (loaded from environment)
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    
    # Request settings
    timeout_seconds: int = 120
    max_retries: int = 3


@dataclass
class SegmentationConfig:
    """
    Configuration for video segmentation parameters.
    
    RESEARCH DECISION: Pause-based segmentation as default.
    
    WHY NOT fixed-window? Fixed windows cut mid-sentence, creating incoherent clips.
    
    WHY NOT semantic-only? Semantic boundaries require full text understanding and
    may split visual action sequences.
    
    WHY pause-based? Speech pauses naturally align with:
    - Thought/topic transitions
    - Dramatic beats
    - Scene changes
    This preserves narrative coherence while being computationally efficient.
    
    DURATION CONSTRAINTS: 60-70 seconds based on:
    - YouTube Shorts max: 60s (now 3min, but 60s optimal for engagement)
    - TikTok sweet spot: 30-60s
    - Instagram Reels: 60s (now 90s)
    - Target 65s to avoid edge cases.
    """
    
    # Segmentation strategy
    # Options: 'pause_based', 'fixed_window', 'semantic_boundary', 'hybrid', 'llm'
    # RESEARCH NOTE: 'pause_based' respects natural speech patterns
    strategy: str = "pause_based"
    
    # Target segment duration for short-form content
    # RESEARCH NOTE: Platform analysis suggests 60-70s maximizes completion rate
    target_duration_seconds: float = 65.0  # Center of 60-70 range
    min_duration_seconds: float = 60.0
    max_duration_seconds: float = 70.0
    
    # Number of segments to extract
    min_segments: int = 3
    max_segments: int = 10
    segments_per_5min: float = 2.5  # Scaling factor
    
    # Segment boundary detection
    silence_threshold_db: float = -40.0
    min_silence_duration_seconds: float = 0.3
    min_gap_seconds: float = 0.5  # Min speech pause to consider
    strong_gap_seconds: float = 1.5  # Strong boundary gap threshold
    scene_change_threshold: float = 0.4
    boundary_tolerance_seconds: float = 5.0  # How far to look for natural boundary
    
    # Segment types for classification
    segment_types: list = field(default_factory=lambda: [
        "funny", "emotional", "informative", "dramatic", "action"
    ])
    
    # Whether to use LLM for segment classification (type detection)
    use_llm_classification: bool = True


@dataclass  
class FeatureExtractionConfig:
    """
    Configuration for multimodal feature extraction.
    
    RESEARCH DECISION: Feature selection rationale.
    
    TEXT FEATURES:
    - Sentiment: Emotional content drives engagement
    - Questions/exclamations: Indicate hooks and emphasis
    - Speech rate: Correlates with energy and pacing
    
    AUDIO FEATURES:
    - Energy (RMS): Loud moments often engaging
    - Silence ratio: Pauses can be dramatic or indicate problems
    - Dynamics: Variation keeps attention
    - Spectral centroid: Distinguishes speech vs music
    
    VISUAL FEATURES:
    - Motion: Action and movement capture attention
    - Scene changes: Editing pace affects engagement
    - Brightness: Dark scenes may be less engaging on mobile
    - Color variance: Visual interest indicator
    
    WHY FFmpeg-based? Faster than Python-native and already required for video.
    Future work: Add CLIP embeddings for semantic visual understanding.
    """
    
    # Text features
    # RESEARCH NOTE: MiniLM-L6-v2 balances speed and quality for embeddings
    text_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    extract_text_features: bool = True
    
    # Audio features
    # RESEARCH NOTE: 16kHz is standard for speech processing (Whisper uses this)
    extract_audio_features: bool = True
    audio_sample_rate: int = 16000
    audio_hop_length: int = 512
    audio_n_mels: int = 128
    
    # Visual features
    # RESEARCH NOTE: 1 FPS is sufficient for motion detection, minimizes I/O
    extract_visual_features: bool = True
    visual_sample_fps: float = 1.0  # Frames per second to sample
    histogram_bins: int = 64
    
    # Feature aggregation
    aggregation_method: str = "mean"  # "mean", "max", "concat"


@dataclass
class ScoringConfig:
    """
    Configuration for engagement scoring function.
    
    RESEARCH DECISION: Modality weight rationale.
    
    WHY text_weight=0.4 (highest)?
    - Dialogue content most directly indicates narrative engagement
    - Subtitles are most reliably extracted
    - Sentiment/questions are strong engagement predictors
    
    WHY audio_weight=0.3?
    - Energy levels correlate with exciting moments
    - But audio can mislead (music vs speech)
    
    WHY visual_weight=0.3?
    - Motion/scenes are important but less reliable
    - Simple visual features miss semantic content
    - Future: increase when adding CLIP embeddings
    
    The scoring function:
        E(S) = w_t * f_t(S) + w_a * f_a(S) + w_v * f_v(S)
    
    This linear combination enables:
    1. Interpretability: each weight's contribution is clear
    2. Ablation studies: set weights to 0 to disable modalities
    3. Easy extension: add more modalities with more terms
    """
    
    # Scoring mode
    # RESEARCH NOTE: 'rule_based' for interpretability, 'learned' for performance
    mode: Literal["rule_based", "learned", "hybrid"] = "rule_based"
    
    # Rule-based weights (must sum to 1.0 for interpretability)
    # RESEARCH NOTE: Weights based on preliminary experiments and domain knowledge
    text_weight: float = 0.4   # Highest: dialogue is most reliable signal
    audio_weight: float = 0.3  # Energy/dynamics important but can mislead
    visual_weight: float = 0.3 # Motion useful but misses semantics
    
    # Learned model configuration
    model_path: Optional[str] = None
    
    # Score normalization
    normalize_scores: bool = True
    score_min: float = 0.0
    score_max: float = 1.0


@dataclass
class AblationConfig:
    """
    Configuration for ablation studies.
    
    RESEARCH DECISION: Ablation study design.
    
    Ablation studies systematically remove modalities to answer:
    - RQ1: How much does each modality contribute?
    - RQ2: Are multimodal features better than unimodal?
    - RQ3: Which combinations work best?
    - RQ4 (NEW): How much do classical CV features add beyond signal-level?
    
    METRICS for comparison:
    - Spearman's ρ: Rank correlation with full system
    - Kendall's τ: Concordance measure
    - Top-K Agreement: Overlap in top selections
    
    HYPOTHESIS: full_multimodal > text_audio > text_only > audio_only ≈ visual_only
    
    CV ABLATION HYPOTHESIS:
    - visual_with_cv > visual_signal_only
    - Because CV features (edge detection, histogram analysis) capture
      structural information that raw pixel statistics miss.
    
    This ordering is expected because:
    1. Text carries most semantic information
    2. Audio adds energy/dynamics not in text
    3. Visual features (currently) are low-level
    4. CV features add edge/motion/scene structure beyond raw pixels
    """
    
    # Which modalities to enable
    use_text: bool = True
    use_audio: bool = True
    use_visual: bool = True
    
    # Classical Computer Vision feature control
    # When False, only signal-level visual features are used
    use_cv_features: bool = True
    
    # Ablation mode name (for logging and reproducibility)
    mode_name: str = "full_multimodal"
    
    # =========================================================================
    # Factory methods for common ablation configurations
    # =========================================================================
    
    @classmethod
    def text_only(cls) -> "AblationConfig":
        """Text modality only - baseline for subtitle-based systems."""
        return cls(use_text=True, use_audio=False, use_visual=False, 
                   use_cv_features=False, mode_name="text_only")
    
    @classmethod
    def audio_only(cls) -> "AblationConfig":
        """Audio modality only - tests audio-based engagement signals."""
        return cls(use_text=False, use_audio=True, use_visual=False,
                   use_cv_features=False, mode_name="audio_only")
    
    @classmethod
    def visual_only(cls) -> "AblationConfig":
        """Visual modality only (with CV) - tests motion/scene-based signals."""
        return cls(use_text=False, use_audio=False, use_visual=True,
                   use_cv_features=True, mode_name="visual_only")
    
    @classmethod
    def visual_signal_only(cls) -> "AblationConfig":
        """
        Visual signal-level only (NO CV features).
        
        This is the key ablation for measuring CV contribution.
        Uses: brightness, color_variance, motion_intensity (raw pixel diff)
        Excludes: contrast, edge_density, motion_magnitude, histogram_diff
        """
        return cls(use_text=False, use_audio=False, use_visual=True,
                   use_cv_features=False, mode_name="visual_signal_only")
    
    @classmethod
    def text_audio(cls) -> "AblationConfig":
        """Text + Audio - tests value of visual features."""
        return cls(use_text=True, use_audio=True, use_visual=False,
                   use_cv_features=False, mode_name="text_audio")
    
    @classmethod
    def full_no_cv(cls) -> "AblationConfig":
        """
        Full multimodal WITHOUT CV features.
        
        Uses all modalities but only signal-level visual features.
        Compare with full_multimodal to measure CV contribution.
        """
        return cls(use_text=True, use_audio=True, use_visual=True,
                   use_cv_features=False, mode_name="full_no_cv")
    
    @classmethod
    def full_multimodal(cls) -> "AblationConfig":
        """Full multimodal - all modalities + CV features enabled (reference)."""
        return cls(use_text=True, use_audio=True, use_visual=True,
                   use_cv_features=True, mode_name="full_multimodal")


@dataclass
class VideoConfig:
    """Configuration for video processing."""
    
    # Allowed formats
    allowed_extensions: set = field(default_factory=lambda: {'mp4', 'avi', 'mov', 'mkv', 'webm'})
    
    # Upload limits
    max_file_size_bytes: int = 2 * 1024 * 1024 * 1024  # 2GB
    
    # Encoding settings for output
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    encoding_preset: str = "ultrafast"  # ultrafast, fast, medium, slow
    encoding_threads: int = 4
    
    # Quality settings
    video_bitrate: Optional[str] = None  # None for auto
    audio_bitrate: str = "128k"


@dataclass
class FlaskConfig:
    """Configuration for Flask web server."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = True
    
    # Security
    secret_key: str = field(default_factory=lambda: os.getenv("FLASK_SECRET_KEY", "dev-secret-key"))
    
    # CORS
    cors_origins: list = field(default_factory=lambda: ["*"])


@dataclass
class ResearchConfig:
    """Configuration for research and experiment tracking."""
    
    # Experiment identification
    experiment_name: str = "default"
    experiment_version: str = "1.0.0"
    
    # Reproducibility
    random_seed: int = 42
    
    # Logging
    log_level: str = "INFO"
    log_features: bool = True
    log_scores: bool = True
    log_decisions: bool = True
    
    # Output formats
    save_intermediate_results: bool = True
    output_format: str = "json"  # "json" or "pickle"


@dataclass
class AppConfig:
    """
    Master configuration class that aggregates all configuration sections.
    
    This is the main configuration object used throughout the application.
    """
    
    paths: PathConfig = field(default_factory=PathConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    features: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    flask: FlaskConfig = field(default_factory=FlaskConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    
    def __post_init__(self):
        """Ensure all directories exist after initialization."""
        self.paths.ensure_directories()
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization."""
        def convert(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: convert(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, set):
                return list(obj)
            return obj
        return convert(self)
    
    def save(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        """Create configuration from dictionary."""
        # Handle PathConfig specially to convert base_dir back to Path
        paths_data = data.get('paths', {})
        if 'base_dir' in paths_data and isinstance(paths_data['base_dir'], str):
            paths_data['base_dir'] = Path(paths_data['base_dir'])
        
        # Handle VideoConfig specially to convert allowed_extensions list back to set
        video_data = data.get('video', {})
        if 'allowed_extensions' in video_data and isinstance(video_data['allowed_extensions'], list):
            video_data['allowed_extensions'] = set(video_data['allowed_extensions'])
        
        return cls(
            paths=PathConfig(**paths_data),
            whisper=WhisperConfig(**data.get('whisper', {})),
            gemini=GeminiConfig(**data.get('gemini', {})),
            segmentation=SegmentationConfig(**data.get('segmentation', {})),
            features=FeatureExtractionConfig(**data.get('features', {})),
            scoring=ScoringConfig(**data.get('scoring', {})),
            ablation=AblationConfig(**data.get('ablation', {})),
            video=VideoConfig(**video_data),
            flask=FlaskConfig(**data.get('flask', {})),
            research=ResearchConfig(**data.get('research', {}))
        )
    
    @classmethod
    def load(cls, filepath: str) -> "AppConfig":
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Configuration loaded from {filepath}")
        return cls.from_dict(data)


# =============================================================================
# GLOBAL CONFIGURATION SINGLETON
# =============================================================================

_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """
    Get the global application configuration.
    
    Creates a default configuration on first access.
    Thread-safe for typical usage patterns.
    
    Returns:
        The global AppConfig instance
    """
    global _config
    if _config is None:
        _config = AppConfig()
        logger.info("Initialized default application configuration")
    return _config


def set_config(config: AppConfig) -> None:
    """
    Set the global application configuration.
    
    Use this to load a custom configuration for experiments.
    
    Args:
        config: The AppConfig instance to use globally
    """
    global _config
    _config = config
    logger.info(f"Set global configuration (experiment: {config.research.experiment_name})")


def reset_config() -> None:
    """Reset the global configuration to None (forces reload on next get_config)."""
    global _config
    _config = None
    logger.info("Reset global configuration")


def load_experiment_config(filepath: str) -> AppConfig:
    """
    Load an experiment configuration and set it as global.
    
    Convenience function that combines load() and set_config().
    
    Args:
        filepath: Path to the configuration JSON file
        
    Returns:
        The loaded AppConfig instance
    """
    config = AppConfig.load(filepath)
    set_config(config)
    return config


# =============================================================================
# ENVIRONMENT VARIABLE OVERRIDES
# =============================================================================

def apply_environment_overrides(config: AppConfig) -> AppConfig:
    """
    Apply environment variable overrides to configuration.
    
    Environment variables follow the pattern:
    MOVIE_SHORTS_{SECTION}_{KEY}
    
    Examples:
        MOVIE_SHORTS_WHISPER_MODEL_SIZE=large
        MOVIE_SHORTS_FLASK_PORT=8080
        MOVIE_SHORTS_RESEARCH_LOG_LEVEL=DEBUG
    
    Also supports common simplified environment variables:
        WHISPER_MODEL=tiny (maps to whisper.model_size)
        PORT=8080 (maps to flask.port)
    
    Args:
        config: Base configuration to override
        
    Returns:
        Configuration with environment overrides applied
    """
    # Handle common simplified environment variables first
    if os.getenv("WHISPER_MODEL"):
        config.whisper.model_size = os.getenv("WHISPER_MODEL")
        logger.info(f"Environment override: whisper.model_size = {config.whisper.model_size}")
    
    if os.getenv("PORT"):
        try:
            config.flask.port = int(os.getenv("PORT"))
            logger.info(f"Environment override: flask.port = {config.flask.port}")
        except ValueError:
            pass
    
    prefix = "MOVIE_SHORTS_"
    
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
            
        parts = key[len(prefix):].lower().split('_', 1)
        if len(parts) != 2:
            continue
            
        section, attr = parts
        
        # Map section names to config attributes
        section_map = {
            'whisper': 'whisper',
            'gemini': 'gemini',
            'segmentation': 'segmentation',
            'features': 'features',
            'scoring': 'scoring',
            'ablation': 'ablation',
            'video': 'video',
            'flask': 'flask',
            'research': 'research'
        }
        
        if section not in section_map:
            continue
            
        section_config = getattr(config, section_map[section], None)
        if section_config is None or not hasattr(section_config, attr):
            continue
        
        # Convert value to appropriate type
        current_value = getattr(section_config, attr)
        try:
            if isinstance(current_value, bool):
                typed_value = value.lower() in ('true', '1', 'yes')
            elif isinstance(current_value, int):
                typed_value = int(value)
            elif isinstance(current_value, float):
                typed_value = float(value)
            else:
                typed_value = value
                
            setattr(section_config, attr, typed_value)
            logger.info(f"Environment override: {section}.{attr} = {typed_value}")
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to apply environment override {key}: {e}")
    
    return config


# =============================================================================
# PRESET CONFIGURATIONS FOR COMMON SCENARIOS
# =============================================================================

def get_development_config() -> AppConfig:
    """Get configuration optimized for development."""
    config = AppConfig()
    config.flask.debug = True
    config.whisper.model_size = "tiny"  # Faster for development
    config.research.log_level = "DEBUG"
    return config


def get_production_config() -> AppConfig:
    """Get configuration optimized for production."""
    config = AppConfig()
    config.flask.debug = False
    config.whisper.model_size = "medium"  # Better accuracy
    config.video.encoding_preset = "medium"  # Better quality
    config.research.log_level = "INFO"
    return config


def get_research_config(experiment_name: str) -> AppConfig:
    """Get configuration optimized for research experiments."""
    config = AppConfig()
    config.research.experiment_name = experiment_name
    config.research.save_intermediate_results = True
    config.research.log_features = True
    config.research.log_scores = True
    config.research.log_decisions = True
    config.research.log_level = "DEBUG"
    return config

