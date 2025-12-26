"""
Analysis Service Module
=======================
Provides AI-powered analysis of video content using Google Gemini.

This service handles:
- Subtitle analysis for engagement detection
- Segment recommendation generation
- LLM-based content reasoning
"""

import json
import logging
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from dotenv import load_dotenv

# Import configuration
from ..config import get_config

# Import data models
from ..models import Segment, generate_segment_id

# Load environment variables (for API key fallback)
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class AnalysisService:
    """
    Service for AI-powered video content analysis.
    
    Uses Google Gemini to analyze subtitle content and identify
    potentially engaging segments for short-form video extraction.
    """
    
    def __init__(self):
        """
        Initialize the analysis service with Gemini configuration.
        
        Loads configuration from the centralized config system.
        """
        config = get_config()
        gemini_config = config.gemini
        segmentation_config = config.segmentation
        
        # Configure Gemini API
        if gemini_config.api_key:
            genai.configure(api_key=gemini_config.api_key)
        else:
            logger.warning("No Gemini API key found in configuration")
        
        # Store segmentation config for use in analysis
        self.segmentation_config = segmentation_config
        
        # Initialize Gemini model with configuration
        self.model = genai.GenerativeModel(
            gemini_config.model_name,
            generation_config={
                'temperature': gemini_config.temperature,
                'top_k': gemini_config.top_k,
                'top_p': gemini_config.top_p,
                'max_output_tokens': gemini_config.max_output_tokens,
            }
        )
        
        logger.info(
            f"Initialized AnalysisService with model: {gemini_config.model_name}, "
            f"temperature: {gemini_config.temperature}"
        )

    def _calculate_num_sections(self, video_duration: float) -> int:
        """
        Calculate the number of sections to extract based on video duration.
        
        Uses configurable parameters from SegmentationConfig.
        
        Args:
            video_duration: Total video duration in seconds
            
        Returns:
            Number of sections to extract
        """
        config = self.segmentation_config
        
        # Formula: For every 5 minutes of video, extract segments_per_5min sections
        raw_sections = (video_duration / 300) * config.segments_per_5min
        
        # Clamp to configured min/max range
        num_sections = int(min(max(config.min_segments, raw_sections), config.max_segments))
        
        logger.debug(
            f"Calculated {num_sections} sections for {video_duration:.1f}s video "
            f"(range: {config.min_segments}-{config.max_segments})"
        )
        
        return num_sections

    def _build_analysis_prompt(self, subtitles: List[Dict], num_sections: int) -> str:
        """
        Build the prompt for Gemini analysis.
        
        Args:
            subtitles: List of subtitle dictionaries with start, end, text
            num_sections: Number of sections to request
            
        Returns:
            Formatted prompt string
        """
        config = self.segmentation_config
        
        # Build segment type string from configuration
        segment_types = "/".join(config.segment_types[:3])  # Use first 3 types
        
        prompt = f"""You are a Shorts Editor AI. I will give you subtitles from a video with start and end timestamps in seconds. Find {num_sections} engaging sections that each last **{config.min_duration_seconds:.0f} to {config.max_duration_seconds:.0f} seconds** (can cross subtitle boundaries). Each section should be either {segment_types}.

IMPORTANT: Your response must be a valid JSON array ONLY, with no markdown formatting, no code blocks, and no additional text.
The response must be in this exact format:
[
  {{"start": <start_time_in_seconds>, "end": <end_time_in_seconds>, "type": "{segment_types.split('/')[0]}"}},
  {{"start": <start_time_in_seconds>, "end": <end_time_in_seconds>, "type": "{segment_types.split('/')[1]}"}},
  ...
]

Subtitles:
{json.dumps(subtitles, indent=2)}
"""
        return prompt

    def analyze_subtitles(
        self, 
        subtitles: List[Dict[str, Any]],
        video_filename: Optional[str] = None,
        return_models: bool = False
    ) -> List[Dict[str, Any]] | List[Segment]:
        """
        Analyze subtitles and find engaging sections using Gemini.
        
        Args:
            subtitles: List of subtitle dictionaries, each containing:
                - start: Start time in seconds
                - end: End time in seconds
                - text: Subtitle text
            video_filename: Optional filename for segment ID generation
            return_models: If True, returns List[Segment] instead of dicts
                
        Returns:
            List of section dictionaries (or Segment models if return_models=True),
            each containing:
                - start: Start time in seconds
                - end: End time in seconds
                - type: Section type (e.g., "funny", "emotional", "informative")
                
        Raises:
            ValueError: If no subtitles provided or Gemini returns invalid response
        """
        if not subtitles:
            raise ValueError('No subtitles provided')
        
        # Calculate video duration and number of sections
        video_duration = max(subtitle['end'] for subtitle in subtitles)
        num_sections = self._calculate_num_sections(video_duration)
        
        # Build and send prompt
        prompt = self._build_analysis_prompt(subtitles, num_sections)
        
        logger.info(
            f"Analyzing {len(subtitles)} subtitles ({video_duration:.1f}s), "
            f"requesting {num_sections} sections"
        )
        
        # Call Gemini API
        try:
            response = self.model.generate_content(prompt)
            result = response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")
            raise ValueError(f"Failed to get response from Gemini: {str(e)}")
        
        # Clean response (remove markdown formatting if present)
        result = result.replace('```json', '').replace('```', '').strip()
        
        # Parse JSON response
        try:
            sections = json.loads(result)
            
            if not isinstance(sections, list):
                raise json.JSONDecodeError("Response is not an array", result, 0)
            
            # Validate and create sections
            validated_sections = []
            segment_models = []
            
            for i, section in enumerate(sections):
                if not all(key in section for key in ['start', 'end', 'type']):
                    logger.warning(f"Skipping invalid section: {section}")
                    continue
                
                start = float(section['start'])
                end = float(section['end'])
                seg_type = str(section['type'])
                
                validated_sections.append({
                    'start': start,
                    'end': end,
                    'type': seg_type
                })
                
                # Create Segment model if requested
                if return_models:
                    segment_id = generate_segment_id(
                        video_filename or "unknown",
                        start,
                        end
                    )
                    segment_models.append(Segment(
                        segment_id=segment_id,
                        start_seconds=start,
                        end_seconds=end,
                        segment_type=seg_type,
                        rank=i + 1,  # Initial rank based on Gemini order
                        source='gemini'
                    ))
            
            logger.info(f"Successfully extracted {len(validated_sections)} sections")
            
            return segment_models if return_models else validated_sections
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing Gemini response: {result}")
            logger.error(f"JSON decode error: {str(e)}")
            raise ValueError('Invalid response from Gemini')
