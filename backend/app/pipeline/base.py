"""
Pipeline Base Module
====================
Defines the base class for all pipeline stages.

Each stage:
- Has a name and description
- Takes a PipelineContext and modifies it
- Can be cached for expensive operations
- Tracks execution time and status
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Any
import json

from .context import PipelineContext

logger = logging.getLogger(__name__)


class PipelineStage(ABC):
    """
    Abstract base class for pipeline stages.
    
    Subclasses must implement:
    - name: Stage identifier
    - description: Human-readable description
    - _execute(): The actual stage logic
    
    The base class handles:
    - Timing and logging
    - Error handling
    - Caching (optional)
    - Progress reporting
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this stage."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this stage does."""
        pass
    
    @property
    def cacheable(self) -> bool:
        """Whether this stage's output can be cached. Override to enable."""
        return False
    
    @abstractmethod
    def _execute(self, context: PipelineContext) -> None:
        """
        Execute the stage logic.
        
        This method should modify the context in place, adding its outputs
        to the appropriate context fields.
        
        Args:
            context: The pipeline context to read from and write to
            
        Raises:
            Any exception on failure (will be caught by run())
        """
        pass
    
    def _get_output_summary(self, context: PipelineContext) -> str:
        """
        Get a summary of what this stage produced.
        
        Override in subclasses for meaningful summaries.
        """
        return "completed"
    
    def _load_from_cache(self, context: PipelineContext) -> bool:
        """
        Try to load this stage's output from cache.
        
        Returns True if cache was loaded successfully.
        Override in subclasses to implement caching.
        """
        if not self.cacheable:
            return False
        
        cache_path = context.get_cache_path(self.name)
        if not cache_path.exists():
            return False
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            self._restore_from_cache(context, cached_data)
            logger.info(f"Loaded {self.name} from cache: {cache_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load cache for {self.name}: {e}")
            return False
    
    def _save_to_cache(self, context: PipelineContext) -> None:
        """
        Save this stage's output to cache.
        
        Override in subclasses to implement caching.
        """
        if not self.cacheable:
            return
        
        try:
            cache_data = self._get_cache_data(context)
            cache_path = context.get_cache_path(self.name)
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"Cached {self.name} to: {cache_path}")
            
        except Exception as e:
            logger.warning(f"Failed to cache {self.name}: {e}")
    
    def _get_cache_data(self, context: PipelineContext) -> dict:
        """
        Get the data to cache for this stage.
        
        Override in subclasses that enable caching.
        """
        return {}
    
    def _restore_from_cache(self, context: PipelineContext, cached_data: dict) -> None:
        """
        Restore stage output from cached data.
        
        Override in subclasses that enable caching.
        """
        pass
    
    def run(
        self, 
        context: PipelineContext, 
        use_cache: bool = True
    ) -> bool:
        """
        Run this pipeline stage.
        
        Handles timing, logging, caching, and error handling.
        
        Args:
            context: The pipeline context
            use_cache: Whether to try loading from cache first
            
        Returns:
            True if stage completed successfully, False otherwise
        """
        logger.info(f"Starting stage: {self.name}")
        start_time = time.time()
        
        try:
            # Try cache first
            if use_cache and self._load_from_cache(context):
                duration = time.time() - start_time
                context.record_stage(
                    self.name, 
                    success=True, 
                    duration=duration,
                    summary=f"loaded from cache"
                )
                return True
            
            # Execute the stage
            self._execute(context)
            
            # Save to cache if enabled
            if use_cache:
                self._save_to_cache(context)
            
            duration = time.time() - start_time
            summary = self._get_output_summary(context)
            context.record_stage(
                self.name,
                success=True,
                duration=duration,
                summary=summary
            )
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            logger.exception(f"Stage {self.name} failed: {error_msg}")
            context.record_stage(
                self.name,
                success=False,
                duration=duration,
                error=error_msg
            )
            return False
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"


class ConditionalStage(PipelineStage):
    """
    A pipeline stage that only runs if a condition is met.
    
    Useful for ablation studies where certain stages should be skipped.
    """
    
    @abstractmethod
    def should_run(self, context: PipelineContext) -> bool:
        """
        Determine if this stage should run.
        
        Args:
            context: The pipeline context
            
        Returns:
            True if the stage should execute, False to skip
        """
        pass
    
    def run(self, context: PipelineContext, use_cache: bool = True) -> bool:
        """Run the stage only if condition is met."""
        if not self.should_run(context):
            logger.info(f"Skipping stage {self.name}: condition not met")
            context.record_stage(
                self.name,
                success=True,
                duration=0.0,
                summary="skipped (condition not met)"
            )
            return True
        
        return super().run(context, use_cache)

