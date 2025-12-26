"""
Text Feature Extraction Module
==============================
Extracts linguistic and semantic features from subtitle text.

Features extracted:
- Word/sentence counts and statistics
- Sentiment analysis (positive/negative/neutral)
- Emotion keyword detection
- Speech rate estimation
- Question/exclamation patterns
- Keyword density
"""

import re
import logging
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

from ..models import TextFeatures
from ..config import get_config

logger = logging.getLogger(__name__)


# =============================================================================
# SENTIMENT AND EMOTION LEXICONS
# =============================================================================

# Positive sentiment words (simplified lexicon)
POSITIVE_WORDS = {
    'amazing', 'awesome', 'beautiful', 'best', 'brilliant', 'excellent',
    'fantastic', 'good', 'great', 'happy', 'incredible', 'love', 'lovely',
    'nice', 'outstanding', 'perfect', 'positive', 'super', 'wonderful',
    'wow', 'yes', 'excited', 'exciting', 'fun', 'funny', 'hilarious',
    'impressive', 'inspiring', 'interesting', 'recommend', 'success',
    'thank', 'thanks', 'cool', 'delicious', 'enjoy', 'favorite', 'glad',
    'grateful', 'helpful', 'kind', 'pleased', 'proud', 'satisfied', 'smile'
}

# Negative sentiment words
NEGATIVE_WORDS = {
    'awful', 'bad', 'boring', 'broken', 'confusing', 'dead', 'difficult',
    'disappointing', 'disgusting', 'fail', 'failed', 'hate', 'horrible',
    'hurt', 'lonely', 'negative', 'never', 'no', 'painful', 'poor', 'sad',
    'scary', 'terrible', 'ugly', 'unfortunately', 'unhappy', 'upset',
    'waste', 'weak', 'worried', 'wrong', 'worst', 'angry', 'annoyed',
    'anxious', 'disappointed', 'frustrated', 'scared', 'stressed', 'tired'
}

# Emotion-specific keywords
EMOTION_KEYWORDS = {
    'funny': {'laugh', 'lol', 'haha', 'hilarious', 'funny', 'joke', 'comedy', 'laughing'},
    'emotional': {'cry', 'crying', 'sad', 'tears', 'emotional', 'love', 'heart', 'feel', 'feeling'},
    'dramatic': {'shock', 'shocked', 'reveal', 'twist', 'dramatic', 'intense', 'tension', 'suspense'},
    'action': {'run', 'fight', 'chase', 'action', 'fast', 'quick', 'attack', 'battle', 'explosion'},
    'informative': {'learn', 'explain', 'how', 'why', 'what', 'understand', 'tip', 'guide', 'tutorial'},
}

# Intensifiers that modify sentiment
INTENSIFIERS = {'very', 'really', 'so', 'extremely', 'incredibly', 'absolutely', 'totally'}
NEGATORS = {'not', "n't", 'never', 'no', 'neither', 'nobody', 'nothing', 'nowhere'}


class TextFeatureExtractor:
    """
    Extracts text-based features from subtitle content.
    
    Features:
    - Basic statistics (word count, sentence count)
    - Sentiment score (-1 to 1)
    - Emotion keyword density
    - Speech rate (if duration provided)
    - Punctuation patterns (questions, exclamations)
    """
    
    def __init__(self):
        """Initialize the text feature extractor."""
        self.config = get_config().features
        
        # Compile regex patterns for efficiency
        self.sentence_pattern = re.compile(r'[.!?]+')
        self.word_pattern = re.compile(r'\b\w+\b')
        self.question_pattern = re.compile(r'\?')
        self.exclamation_pattern = re.compile(r'!')
    
    def extract(
        self,
        text: str,
        duration_seconds: Optional[float] = None
    ) -> TextFeatures:
        """
        Extract all text features from the given text.
        
        Args:
            text: Subtitle text content
            duration_seconds: Segment duration for speech rate calculation
            
        Returns:
            TextFeatures object with all computed features
        """
        if not text or not text.strip():
            return TextFeatures()
        
        text = text.strip()
        
        # Basic statistics
        words = self.word_pattern.findall(text.lower())
        word_count = len(words)
        sentences = self._count_sentences(text)
        avg_word_length = sum(len(w) for w in words) / max(1, word_count)
        
        # Punctuation counts
        question_count = len(self.question_pattern.findall(text))
        exclamation_count = len(self.exclamation_pattern.findall(text))
        
        # Sentiment analysis
        sentiment_score = self._analyze_sentiment(words)
        
        # Keyword density for engagement keywords
        keyword_density = self._calculate_keyword_density(words)
        
        # Speech rate (words per second)
        speech_rate = 0.0
        if duration_seconds and duration_seconds > 0:
            speech_rate = word_count / duration_seconds
        
        return TextFeatures(
            word_count=word_count,
            sentence_count=sentences,
            avg_word_length=avg_word_length,
            sentiment_score=sentiment_score,
            keyword_density=keyword_density,
            question_count=question_count,
            exclamation_count=exclamation_count
        )
    
    def _count_sentences(self, text: str) -> int:
        """Count sentences in text."""
        # Split by sentence-ending punctuation
        parts = self.sentence_pattern.split(text)
        # Filter empty parts
        return len([p for p in parts if p.strip()])
    
    def _analyze_sentiment(self, words: List[str]) -> float:
        """
        Analyze sentiment of words.
        
        Returns score from -1 (negative) to 1 (positive).
        Uses a simple lexicon-based approach with negation handling.
        """
        if not words:
            return 0.0
        
        positive_count = 0
        negative_count = 0
        negate_next = False
        intensity = 1.0
        
        for i, word in enumerate(words):
            # Check for negators
            if word in NEGATORS:
                negate_next = True
                continue
            
            # Check for intensifiers
            if word in INTENSIFIERS:
                intensity = 1.5
                continue
            
            # Score the word
            if word in POSITIVE_WORDS:
                if negate_next:
                    negative_count += intensity
                else:
                    positive_count += intensity
            elif word in NEGATIVE_WORDS:
                if negate_next:
                    positive_count += intensity
                else:
                    negative_count += intensity
            
            # Reset modifiers
            negate_next = False
            intensity = 1.0
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        # Normalize to -1 to 1 range
        score = (positive_count - negative_count) / total
        return max(-1.0, min(1.0, score))
    
    def _calculate_keyword_density(self, words: List[str]) -> float:
        """
        Calculate density of engagement-related keywords.
        
        Returns ratio of engagement keywords to total words.
        """
        if not words:
            return 0.0
        
        # Collect all engagement keywords
        all_keywords = set()
        for keywords in EMOTION_KEYWORDS.values():
            all_keywords.update(keywords)
        
        # Count matches
        matches = sum(1 for w in words if w in all_keywords)
        
        return matches / len(words)
    
    def detect_dominant_emotion(self, text: str) -> Tuple[str, float]:
        """
        Detect the dominant emotion category in text.
        
        Returns:
            Tuple of (emotion_name, confidence)
        """
        words = set(self.word_pattern.findall(text.lower()))
        
        if not words:
            return ('neutral', 0.0)
        
        # Score each emotion category
        scores = {}
        for emotion, keywords in EMOTION_KEYWORDS.items():
            matches = len(words & keywords)
            scores[emotion] = matches
        
        # Find dominant emotion
        if max(scores.values()) == 0:
            return ('neutral', 0.0)
        
        dominant = max(scores, key=scores.get)
        confidence = scores[dominant] / max(1, sum(scores.values()))
        
        return (dominant, confidence)
    
    def extract_engagement_signals(self, text: str) -> Dict[str, int]:
        """
        Extract specific engagement signals from text.
        
        Returns dict of signal types to counts.
        """
        return {
            'questions': len(self.question_pattern.findall(text)),
            'exclamations': len(self.exclamation_pattern.findall(text)),
            'emphasis_words': len([w for w in text.lower().split() if w in INTENSIFIERS]),
            'emotional_words': sum(
                1 for w in text.lower().split() 
                if w in POSITIVE_WORDS or w in NEGATIVE_WORDS
            )
        }


def extract_text_features(
    text: str,
    duration_seconds: Optional[float] = None
) -> TextFeatures:
    """
    Convenience function to extract text features.
    
    Args:
        text: Subtitle text
        duration_seconds: Segment duration
        
    Returns:
        TextFeatures object
    """
    extractor = TextFeatureExtractor()
    return extractor.extract(text, duration_seconds)

