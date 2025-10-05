"""Validation utilities for words and lexicons."""

import re
from typing import List, Dict, Tuple
from config.languages import ORTHOGRAPHY_RULES, STOP_WORDS, COMMON_PREFIXES
from config.settings import MIN_WORD_LENGTH, MAX_WORD_LENGTH

class WordValidator:
    """Validates words based on linguistic rules."""
    
    def __init__(self, language: str):
        self.language = language.lower()
        self.orthography = ORTHOGRAPHY_RULES.get(self.language, {})
        self.stop_words = STOP_WORDS.get(self.language, [])
        self.prefixes = COMMON_PREFIXES.get(self.language, [])
    
    def is_valid_word(self, word: str, strict: bool = False) -> bool:
        """Check if word is valid."""
        if not word:
            return False
        
        word = word.strip()
        
        # Length check
        if len(word) < MIN_WORD_LENGTH or len(word) > MAX_WORD_LENGTH:
            return False
        
        # Must contain at least one letter
        if not re.search(r'[a-zA-Z]', word):
            return False
        
        # No excessive punctuation
        if re.search(r'[!@#$%^&*()+=\[\]{}|\\:;"<>?,]{2,}', word):
            return False
        
        return True
    
    def validate_translation(self, translation: str) -> bool:
        """Validate English translation."""
        if not translation or len(translation) < 3:
            return False
        return True
    
    def validate_word_entry(self, word_dict: Dict[str, str]) -> Tuple[bool, List[str]]:
        """Validate complete word entry."""
        errors = []
        
        word = word_dict.get('word', '')
        translation = word_dict.get('translation', '')
        
        if not self.is_valid_word(word):
            errors.append(f"Invalid word: '{word}'")
        
        if not self.validate_translation(translation):
            errors.append(f"Invalid translation: '{translation}'")
        
        return len(errors) == 0, errors

class LexiconValidator:
    """Validates complete lexicons."""
    
    def __init__(self):
        self.word_validators = {}
    
    def get_validator(self, language: str):
        """Get or create validator."""
        if language not in self.word_validators:
            self.word_validators[language] = WordValidator(language)
        return self.word_validators[language]
    
    def validate_lexicon(self, lexicon: List[Dict], language: str) -> Dict:
        """Validate entire lexicon."""
        validator = self.get_validator(language)
        
        total = len(lexicon)
        valid = 0
        invalid = []
        
        for idx, word_entry in enumerate(lexicon):
            is_valid, errors = validator.validate_word_entry(word_entry)
            if is_valid:
                valid += 1
            else:
                invalid.append({'index': idx, 'word': word_entry.get('word', ''), 'errors': errors})
        
        return {
            'total_words': total,
            'valid_words': valid,
            'invalid_words_count': len(invalid),
            'validity_rate': round(valid / total * 100, 2) if total > 0 else 0,
            'invalid_entries': invalid[:10]
        }
