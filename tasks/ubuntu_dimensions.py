"""
Ubuntu Sentiment Dimensions - Beyond Positive/Negative

This module introduces multi-dimensional sentiment analysis based on Ubuntu philosophy,
capturing communal and relational aspects of emotions in African languages.

Research Contribution: First computational framework for collectivist sentiment dimensions
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

class SentimentDimension(Enum):
    """Ubuntu-inspired sentiment dimensions."""
    
    # Traditional
    VALENCE = "valence"  # positive/negative (keep for comparison)
    
    # Ubuntu Dimensions
    INDIVIDUAL_COMMUNAL = "individual_communal"  # -1 (individual) to +1 (communal)
    ACTIVE_PASSIVE = "active_passive"  # -1 (passive) to +1 (active)
    IMMEDIATE_ANCESTRAL = "immediate_ancestral"  # -1 (immediate) to +1 (ancestral/eternal)
    HARMONIZING_DIVIDING = "harmonizing_dividing"  # -1 (dividing) to +1 (harmonizing)
    INTERNAL_RELATIONAL = "internal_relational"  # -1 (internal feeling) to +1 (relational)


@dataclass
class UbuntuSentiment:
    """Multi-dimensional sentiment score."""
    
    word: str
    translation: str
    language: str
    
    # Traditional dimension
    valence: float  # -1 to +1 (negative to positive)
    valence_confidence: float  # 0 to 1
    
    # Ubuntu dimensions (all -1 to +1)
    individual_communal: float  # How collective is this emotion?
    active_passive: float  # Does it require action or is it experienced?
    immediate_ancestral: float  # Time dimension - present vs eternal
    harmonizing_dividing: float  # Does it unite or divide?
    internal_relational: float  # Personal feeling vs social bond?
    
    # Metadata
    reasoning: str
    cultural_notes: Optional[str] = None
    example_usage: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'word': self.word,
            'translation': self.translation,
            'language': self.language,
            'valence': self.valence,
            'valence_confidence': self.valence_confidence,
            'dimensions': {
                'individual_communal': self.individual_communal,
                'active_passive': self.active_passive,
                'immediate_ancestral': self.immediate_ancestral,
                'harmonizing_dividing': self.harmonizing_dividing,
                'internal_relational': self.internal_relational
            },
            'reasoning': self.reasoning,
            'cultural_notes': self.cultural_notes,
            'example_usage': self.example_usage
        }
    
    def get_dimension_vector(self) -> np.ndarray:
        """Get all dimensions as vector for clustering/visualization."""
        return np.array([
            self.valence,
            self.individual_communal,
            self.active_passive,
            self.immediate_ancestral,
            self.harmonizing_dividing,
            self.internal_relational
        ])
    
    def get_ubuntu_profile(self) -> Dict[str, str]:
        """Get human-readable Ubuntu profile."""
        profile = {}
        
        # Individual vs Communal
        if self.individual_communal > 0.5:
            profile['communal'] = "Strongly communal - shared experience"
        elif self.individual_communal > 0:
            profile['communal'] = "Leans communal - involves others"
        elif self.individual_communal > -0.5:
            profile['communal'] = "Leans individual - personal experience"
        else:
            profile['communal'] = "Strongly individual - private feeling"
        
        # Active vs Passive
        if self.active_passive > 0.5:
            profile['agency'] = "Requires active participation"
        elif self.active_passive > 0:
            profile['agency'] = "Some agency involved"
        elif self.active_passive > -0.5:
            profile['agency'] = "Mostly experienced passively"
        else:
            profile['agency'] = "Fully passive state"
        
        # Immediate vs Ancestral
        if self.immediate_ancestral > 0.5:
            profile['temporal'] = "Connects to ancestors/eternal values"
        elif self.immediate_ancestral > 0:
            profile['temporal'] = "Long-term perspective"
        elif self.immediate_ancestral > -0.5:
            profile['temporal'] = "Present-focused"
        else:
            profile['temporal'] = "Immediate/momentary"
        
        # Harmonizing vs Dividing
        if self.harmonizing_dividing > 0.5:
            profile['social'] = "Strongly unifying - builds community"
        elif self.harmonizing_dividing > 0:
            profile['social'] = "Promotes harmony"
        elif self.harmonizing_dividing > -0.5:
            profile['social'] = "Potentially divisive"
        else:
            profile['social'] = "Strongly divisive - breaks bonds"
        
        # Internal vs Relational
        if self.internal_relational > 0.5:
            profile['nature'] = "Fundamentally relational"
        elif self.internal_relational > 0:
            profile['nature'] = "Involves relationships"
        elif self.internal_relational > -0.5:
            profile['nature'] = "Personal with relational aspects"
        else:
            profile['nature'] = "Purely internal state"
        
        return profile


class UbuntuSentimentAnalyzer:
    """Analyzes sentiment using Ubuntu dimensions via LLM."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def analyze_with_ubuntu_dimensions(
        self,
        word: str,
        translation: str,
        language: str,
        temperature: float = 0.3
    ) -> UbuntuSentiment:
        """
        Analyze word across all Ubuntu dimensions.
        
        Returns complete multi-dimensional sentiment profile.
        """
        
        prompt = f"""You are an expert in {language} language and Ubuntu philosophy.

Analyze this {language} word across multiple sentiment dimensions:

WORD: {word}
TRANSLATION: {translation}

Rate on these scales (-1 to +1):

1. VALENCE: negative (-1) to positive (+1)
   Traditional sentiment polarity

2. INDIVIDUAL-COMMUNAL: individual (-1) to communal (+1)
   Is this a personal feeling or shared community experience?
   Example: "loneliness" = -0.8 (individual), "ubuntu" = +0.9 (communal)

3. ACTIVE-PASSIVE: passive (-1) to active (+1)
   Is this experienced passively or does it require action?
   Example: "sadness" = -0.6 (passive), "celebration" = +0.8 (active)

4. IMMEDIATE-ANCESTRAL: immediate (-1) to ancestral/eternal (+1)
   Present moment or connected to ancestors/timeless values?
   Example: "hunger" = -0.9 (immediate), "wisdom" = +0.8 (ancestral)

5. HARMONIZING-DIVIDING: dividing (-1) to harmonizing (+1)
   Does this unite or divide people?
   Example: "jealousy" = -0.7 (dividing), "peace" = +0.9 (harmonizing)

6. INTERNAL-RELATIONAL: internal (-1) to relational (+1)
   Personal feeling or fundamentally about relationships?
   Example: "boredom" = -0.8 (internal), "love" = +0.9 (relational)

Respond in this EXACT format:
VALENCE: [score]
CONFIDENCE: [0.0-1.0]
INDIVIDUAL_COMMUNAL: [score]
ACTIVE_PASSIVE: [score]
IMMEDIATE_ANCESTRAL: [score]
HARMONIZING_DIVIDING: [score]
INTERNAL_RELATIONAL: [score]
REASONING: [2-3 sentences explaining the scores]
CULTURAL_NOTE: [Optional: any Ubuntu/cultural context]
EXAMPLE: [Optional: example sentence showing usage]
"""
        
        response = self.llm_client.generate(prompt, temperature=temperature, max_tokens=500)
        
        # Parse response
        return self._parse_ubuntu_response(word, translation, language, response)
    
    def _parse_ubuntu_response(
        self,
        word: str,
        translation: str,
        language: str,
        response: str
    ) -> UbuntuSentiment:
        """Parse LLM response into UbuntuSentiment object."""
        
        lines = response.strip().split('\n')
        scores = {}
        reasoning = ""
        cultural_notes = None
        example = None
        
        for line in lines:
            line = line.strip()
            if ':' not in line:
                continue
            
            key, value = line.split(':', 1)
            key = key.strip().upper()
            value = value.strip()
            
            if key == 'VALENCE':
                scores['valence'] = self._parse_score(value)
            elif key == 'CONFIDENCE':
                scores['valence_confidence'] = self._parse_score(value)
            elif key == 'INDIVIDUAL_COMMUNAL':
                scores['individual_communal'] = self._parse_score(value)
            elif key == 'ACTIVE_PASSIVE':
                scores['active_passive'] = self._parse_score(value)
            elif key == 'IMMEDIATE_ANCESTRAL':
                scores['immediate_ancestral'] = self._parse_score(value)
            elif key == 'HARMONIZING_DIVIDING':
                scores['harmonizing_dividing'] = self._parse_score(value)
            elif key == 'INTERNAL_RELATIONAL':
                scores['internal_relational'] = self._parse_score(value)
            elif key == 'REASONING':
                reasoning = value
            elif key == 'CULTURAL_NOTE':
                cultural_notes = value if value and value.lower() != 'none' else None
            elif key == 'EXAMPLE':
                example = value if value and value.lower() != 'none' else None
        
        # Create UbuntuSentiment with defaults if parsing failed
        return UbuntuSentiment(
            word=word,
            translation=translation,
            language=language,
            valence=scores.get('valence', 0.0),
            valence_confidence=scores.get('valence_confidence', 0.5),
            individual_communal=scores.get('individual_communal', 0.0),
            active_passive=scores.get('active_passive', 0.0),
            immediate_ancestral=scores.get('immediate_ancestral', 0.0),
            harmonizing_dividing=scores.get('harmonizing_dividing', 0.0),
            internal_relational=scores.get('internal_relational', 0.0),
            reasoning=reasoning or "Analysis incomplete",
            cultural_notes=cultural_notes,
            example_usage=example
        )
    
    def _parse_score(self, value: str) -> float:
        """Parse score from string, handling various formats."""
        try:
            # Extract first number found
            import re
            match = re.search(r'-?\d+\.?\d*', value)
            if match:
                score = float(match.group())
                # Clamp to valid range
                return max(-1.0, min(1.0, score))
            return 0.0
        except:
            return 0.0
    
    def batch_analyze(
        self,
        words: List[Dict[str, str]],
        language: str,
        progress_callback: Optional[callable] = None
    ) -> List[UbuntuSentiment]:
        """Analyze multiple words with Ubuntu dimensions."""
        
        results = []
        total = len(words)
        
        for i, word_dict in enumerate(words):
            word = word_dict.get('word', '')
            translation = word_dict.get('translation', '')
            
            result = self.analyze_with_ubuntu_dimensions(word, translation, language)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results
    
    def compare_cultural_profiles(
        self,
        sentiment1: UbuntuSentiment,
        sentiment2: UbuntuSentiment
    ) -> Dict[str, float]:
        """Compare two sentiments across dimensions."""
        
        return {
            'valence_diff': abs(sentiment1.valence - sentiment2.valence),
            'communal_diff': abs(sentiment1.individual_communal - sentiment2.individual_communal),
            'active_diff': abs(sentiment1.active_passive - sentiment2.active_passive),
            'temporal_diff': abs(sentiment1.immediate_ancestral - sentiment2.immediate_ancestral),
            'social_diff': abs(sentiment1.harmonizing_dividing - sentiment2.harmonizing_dividing),
            'relational_diff': abs(sentiment1.internal_relational - sentiment2.internal_relational),
            'overall_similarity': np.linalg.norm(
                sentiment1.get_dimension_vector() - sentiment2.get_dimension_vector()
            )
        }


def calculate_ubuntu_score(sentiment: UbuntuSentiment) -> float:
    """
    Calculate overall 'Ubuntu-ness' score.
    Higher score = more aligned with Ubuntu values (communal, harmonizing, relational).
    """
    ubuntu_score = (
        sentiment.individual_communal * 0.3 +  # Communal is core
        sentiment.harmonizing_dividing * 0.3 +  # Harmony is core
        sentiment.internal_relational * 0.25 +  # Relationships are core
        sentiment.immediate_ancestral * 0.15  # Connection to tradition
    )
    
    return ubuntu_score

