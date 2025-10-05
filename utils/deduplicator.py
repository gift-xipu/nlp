"""Utilities for deduplicating word lists."""

from typing import List, Dict, Set, Tuple
from difflib import SequenceMatcher
import re

class WordDeduplicator:
    """Handles deduplication of word lists."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.seen_words: Set[str] = set()
        self.seen_normalized: Dict[str, str] = {}
    
    def normalize_word(self, word: str) -> str:
        """Normalize word for comparison."""
        normalized = word.lower().strip()
        normalized = re.sub(r'[^\w\s-]', '', normalized)
        normalized = ' '.join(normalized.split())
        return normalized
    
    def is_duplicate(self, word: str, check_similarity: bool = True) -> Tuple[bool, str]:
        """Check if word is a duplicate."""
        normalized = self.normalize_word(word)
        
        # Exact match
        if normalized in self.seen_normalized:
            return True, f"Exact duplicate of '{self.seen_normalized[normalized]}'"
        
        # Fuzzy similarity
        if check_similarity:
            for seen_norm, seen_orig in self.seen_normalized.items():
                similarity = self.calculate_similarity(normalized, seen_norm)
                if similarity >= self.similarity_threshold:
                    return True, f"Similar to '{seen_orig}' ({similarity:.2f})"
        
        return False, ""
    
    def calculate_similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity between two words."""
        return SequenceMatcher(None, word1, word2).ratio()
    
    def add_word(self, word: str) -> bool:
        """Add word to seen set."""
        is_dup, _ = self.is_duplicate(word)
        
        if not is_dup:
            normalized = self.normalize_word(word)
            self.seen_words.add(word)
            self.seen_normalized[normalized] = word
            return True
        
        return False
    
    def add_words(self, existing_words: List[Dict[str, str]], new_words: List[Dict[str, str]], 
                  key: str = 'word') -> List[Dict[str, str]]:
        """Add new words to existing list, removing duplicates."""
        # Initialize with existing
        for word_dict in existing_words:
            word = word_dict.get(key, '')
            if word:
                normalized = self.normalize_word(word)
                self.seen_words.add(word)
                self.seen_normalized[normalized] = word
        
        # Add new words
        result = existing_words.copy()
        
        for word_dict in new_words:
            word = word_dict.get(key, '')
            if word and self.add_word(word):
                result.append(word_dict)
        
        return result
    
    def reset(self):
        """Reset the deduplicator state."""
        self.seen_words.clear()
        self.seen_normalized.clear()
