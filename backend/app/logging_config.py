"""
Research Logging System
=======================
Structured logging for research-quality observability.

This module provides:
- Structured JSON logging for machine-parseable outputs
- Research-specific loggers for features, scores, and decisions
- Log file rotation and organization by experiment
- Integration with the pipeline system

Usage:
    from app.logging_config import get_research_logger, log_segment_features
    
    # Get specialized loggers
    logger = get_research_logger("features")
    logger.info("Extracted features", extra={"segment_id": "...", "features": {...}})
    
    # Convenience functions
    log_segment_features(segment_id, features_dict)
    log_segment_score(segment_id, score_breakdown)
    log_pipeline_decision(decision_type, details)
"""

import os
import sys
import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import asdict

from .config import get_config


# =============================================================================
# CUSTOM FORMATTERS
# =============================================================================

class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    Outputs log records as JSON objects with consistent structure:
    {
        "timestamp": "2024-01-15T10:30:00.123456",
        "level": "INFO",
        "logger": "research.features",
        "message": "Extracted features",
        "context": {...}
    }
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra context if present
        if hasattr(record, 'context') and record.context:
            log_data['context'] = record.context
        
        # Add any extra attributes added via extra={}
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'created', 'filename', 
                          'funcName', 'levelname', 'levelno', 'lineno',
                          'module', 'msecs', 'pathname', 'process',
                          'processName', 'relativeCreated', 'stack_info',
                          'exc_info', 'exc_text', 'thread', 'threadName',
                          'message', 'context'):
                try:
                    # Try to serialize, skip if not serializable
                    json.dumps(value)
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """
    Human-readable formatter for console output.
    
    Format: [LEVEL] logger: message (key=value, ...)
    """
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        # Color the level
        level = record.levelname
        if self.use_colors and level in self.COLORS:
            level = f"{self.COLORS[level]}{level}{self.RESET}"
        
        # Build the message
        msg = f"[{level}] {record.name}: {record.getMessage()}"
        
        # Add extra context in parentheses
        extras = []
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'created', 'filename', 
                          'funcName', 'levelname', 'levelno', 'lineno',
                          'module', 'msecs', 'pathname', 'process',
                          'processName', 'relativeCreated', 'stack_info',
                          'exc_info', 'exc_text', 'thread', 'threadName',
                          'message', 'context'):
                if isinstance(value, (str, int, float, bool)):
                    extras.append(f"{key}={value}")
                elif isinstance(value, dict) and len(value) < 3:
                    extras.append(f"{key}={value}")
        
        if extras:
            msg += f" ({', '.join(extras)})"
        
        return msg


# =============================================================================
# LOGGER FACTORY
# =============================================================================

_loggers: Dict[str, logging.Logger] = {}
_initialized: bool = False


def _ensure_log_directories():
    """Ensure log directories exist."""
    config = get_config()
    log_dir = config.paths.logs
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different log types
    (log_dir / "pipeline").mkdir(exist_ok=True)
    (log_dir / "features").mkdir(exist_ok=True)
    (log_dir / "scores").mkdir(exist_ok=True)
    (log_dir / "decisions").mkdir(exist_ok=True)
    (log_dir / "experiments").mkdir(exist_ok=True)


def _setup_root_logger():
    """Configure the root logger with console handler."""
    global _initialized
    if _initialized:
        return
    
    config = get_config()
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.research.log_level, logging.INFO))
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ConsoleFormatter())
    root_logger.addHandler(console_handler)
    
    _initialized = True


def get_research_logger(
    name: str,
    log_to_file: bool = True,
    experiment_name: Optional[str] = None
) -> logging.Logger:
    """
    Get or create a research logger.
    
    Args:
        name: Logger name (e.g., "features", "scores", "decisions", "pipeline")
        log_to_file: Whether to write logs to file
        experiment_name: Optional experiment name for file organization
        
    Returns:
        Configured logger instance
    """
    _setup_root_logger()
    _ensure_log_directories()
    
    full_name = f"research.{name}"
    
    if full_name in _loggers:
        return _loggers[full_name]
    
    config = get_config()
    logger = logging.getLogger(full_name)
    logger.setLevel(getattr(logging, config.research.log_level, logging.INFO))
    logger.propagate = False  # Don't propagate to root
    
    # Console handler with human-readable format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ConsoleFormatter())
    logger.addHandler(console_handler)
    
    # File handler with JSON format
    if log_to_file:
        exp_name = experiment_name or config.research.experiment_name
        log_dir = config.paths.logs / name
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"{exp_name}_{timestamp}.jsonl"
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(StructuredFormatter())
        logger.addHandler(file_handler)
    
    _loggers[full_name] = logger
    return logger


def get_pipeline_logger(result_id: Optional[str] = None) -> logging.Logger:
    """Get a logger for pipeline execution."""
    logger = get_research_logger("pipeline")
    if result_id:
        # Add result_id to all log records from this logger
        logger = logging.LoggerAdapter(logger, {'result_id': result_id})
    return logger


def get_feature_logger() -> logging.Logger:
    """Get a logger for feature extraction."""
    return get_research_logger("features")


def get_score_logger() -> logging.Logger:
    """Get a logger for scoring decisions."""
    return get_research_logger("scores")


def get_decision_logger() -> logging.Logger:
    """Get a logger for high-level decisions."""
    return get_research_logger("decisions")


# =============================================================================
# CONVENIENCE LOGGING FUNCTIONS
# =============================================================================

def log_segment_features(
    segment_id: str,
    features: Dict[str, Any],
    modalities: list = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log extracted features for a segment.
    
    Args:
        segment_id: Unique segment identifier
        features: Dictionary of feature values
        modalities: List of modalities present
        logger: Optional logger override
    """
    config = get_config()
    if not config.research.log_features:
        return
    
    log = logger or get_feature_logger()
    log.info(
        f"Features extracted for segment {segment_id}",
        extra={
            'segment_id': segment_id,
            'modalities': modalities or [],
            'features': features
        }
    )


def log_segment_score(
    segment_id: str,
    total_score: float,
    text_score: float = 0.0,
    audio_score: float = 0.0,
    visual_score: float = 0.0,
    weights: Dict[str, float] = None,
    method: str = "rule_based",
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log scoring decision for a segment.
    
    Args:
        segment_id: Unique segment identifier
        total_score: Final combined score
        text_score: Text component score
        audio_score: Audio component score
        visual_score: Visual component score
        weights: Weight configuration used
        method: Scoring method used
        logger: Optional logger override
    """
    config = get_config()
    if not config.research.log_scores:
        return
    
    log = logger or get_score_logger()
    log.info(
        f"Scored segment {segment_id}: {total_score:.4f}",
        extra={
            'segment_id': segment_id,
            'total_score': total_score,
            'text_score': text_score,
            'audio_score': audio_score,
            'visual_score': visual_score,
            'weights': weights or {},
            'method': method
        }
    )


def log_pipeline_decision(
    decision_type: str,
    details: Dict[str, Any],
    result_id: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log a high-level pipeline decision.
    
    Args:
        decision_type: Type of decision (e.g., "segment_selection", "ablation_mode")
        details: Decision details
        result_id: Pipeline result ID
        logger: Optional logger override
    """
    config = get_config()
    if not config.research.log_decisions:
        return
    
    log = logger or get_decision_logger()
    extra = {
        'decision_type': decision_type,
        'details': details
    }
    if result_id:
        extra['result_id'] = result_id
    
    log.info(f"Decision: {decision_type}", extra=extra)


def log_stage_start(
    stage_name: str,
    result_id: str,
    input_summary: Dict[str, Any] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """Log the start of a pipeline stage."""
    log = logger or get_pipeline_logger(result_id)
    log.info(
        f"Starting stage: {stage_name}",
        extra={
            'stage_name': stage_name,
            'result_id': result_id,
            'event': 'stage_start',
            'input_summary': input_summary or {}
        }
    )


def log_stage_complete(
    stage_name: str,
    result_id: str,
    duration_seconds: float,
    output_summary: Dict[str, Any] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """Log the completion of a pipeline stage."""
    log = logger or get_pipeline_logger(result_id)
    log.info(
        f"Completed stage: {stage_name} ({duration_seconds:.2f}s)",
        extra={
            'stage_name': stage_name,
            'result_id': result_id,
            'event': 'stage_complete',
            'duration_seconds': duration_seconds,
            'output_summary': output_summary or {}
        }
    )


def log_stage_error(
    stage_name: str,
    result_id: str,
    error: str,
    duration_seconds: float,
    logger: Optional[logging.Logger] = None
) -> None:
    """Log a pipeline stage error."""
    log = logger or get_pipeline_logger(result_id)
    log.error(
        f"Stage failed: {stage_name}",
        extra={
            'stage_name': stage_name,
            'result_id': result_id,
            'event': 'stage_error',
            'error': error,
            'duration_seconds': duration_seconds
        }
    )


def log_ablation_result(
    mode_name: str,
    segment_count: int,
    top_score: float,
    processing_time: float,
    logger: Optional[logging.Logger] = None
) -> None:
    """Log ablation experiment result."""
    log = logger or get_decision_logger()
    log.info(
        f"Ablation result: {mode_name}",
        extra={
            'ablation_mode': mode_name,
            'segment_count': segment_count,
            'top_score': top_score,
            'processing_time_seconds': processing_time,
            'event': 'ablation_complete'
        }
    )


# =============================================================================
# LOG FILE UTILITIES
# =============================================================================

def get_experiment_log_path(
    experiment_name: str,
    log_type: str = "pipeline"
) -> Path:
    """Get the log file path for an experiment."""
    config = get_config()
    log_dir = config.paths.logs / log_type
    timestamp = datetime.now().strftime("%Y%m%d")
    return log_dir / f"{experiment_name}_{timestamp}.jsonl"


def read_log_file(log_path: Union[str, Path]) -> list:
    """
    Read a JSONL log file and return list of log entries.
    
    Args:
        log_path: Path to the log file
        
    Returns:
        List of parsed log entry dictionaries
    """
    entries = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def get_feature_logs_for_result(result_id: str) -> list:
    """Get all feature log entries for a specific result."""
    config = get_config()
    log_dir = config.paths.logs / "features"
    
    entries = []
    for log_file in log_dir.glob("*.jsonl"):
        for entry in read_log_file(log_file):
            if entry.get('result_id') == result_id:
                entries.append(entry)
    
    return entries


def get_score_logs_for_result(result_id: str) -> list:
    """Get all score log entries for a specific result."""
    config = get_config()
    log_dir = config.paths.logs / "scores"
    
    entries = []
    for log_file in log_dir.glob("*.jsonl"):
        for entry in read_log_file(log_file):
            if entry.get('result_id') == result_id:
                entries.append(entry)
    
    return entries

