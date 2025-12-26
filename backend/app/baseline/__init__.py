"""
Baseline System Package
=======================
Formalizes the video analysis pipeline as a reproducible research baseline.

This package provides:
- Formal specification of inputs, outputs, and intermediate representations
- Baseline runner that produces consistent, reproducible outputs
- Verification utilities to check output validity

Usage:
    from app.baseline import BaselineRunner, BaselineSpec
    
    # Run baseline on a video
    runner = BaselineRunner()
    result = runner.run("path/to/video.mp4")
    
    # Verify outputs
    is_valid = runner.verify_result(result)
"""

from .runner import BaselineRunner
from .specification import BaselineSpec, BaselineOutput

__all__ = [
    'BaselineRunner',
    'BaselineSpec',
    'BaselineOutput',
]

