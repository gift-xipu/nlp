#!/usr/bin/env python3
"""
Create all utils and config modules.
Run: python create_utils_config.py
"""

from pathlib import Path

def create_all():
    print("🔧 Creating utils and config modules...")
    print()
    
    # ==================== UTILS ====================
    
    utils_dir = Path("utils")
    utils_dir.mkdir(exist_ok=True)
    
    # utils/__init__.py
    Path("utils/__init__.py").write_text('"""Utils package."""\n')
    print("✅ Created: utils/__init__.py")
    
    # utils/validators.py
    Path("utils/validators.py").write_text('''"""Validation utilities for words and lexicons."""

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
        if re.search(r'[!@#$%^&*()+=\\[\\]{}|\\\\:;"<>?,]{2,}', word):
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
''')
    print("✅ Created: utils/validators.py")
    
    # utils/deduplicator.py
    Path("utils/deduplicator.py").write_text('''"""Utilities for deduplicating word lists."""

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
        normalized = re.sub(r'[^\\w\\s-]', '', normalized)
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
''')
    print("✅ Created: utils/deduplicator.py")
    
    # utils/file_handlers.py
    Path("utils/file_handlers.py").write_text('''"""File handling utilities for various formats."""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

class FileHandler:
    """Handles file I/O operations."""
    
    @staticmethod
    def read_csv(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Read CSV file."""
        return pd.read_csv(file_path, encoding='utf-8', **kwargs)
    
    @staticmethod
    def write_csv(file_path: Union[str, Path], data: Union[pd.DataFrame, List[Dict]], **kwargs):
        """Write data to CSV."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=False, encoding='utf-8', **kwargs)
        else:
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False, encoding='utf-8', **kwargs)
    
    @staticmethod
    def read_json(file_path: Union[str, Path]) -> Any:
        """Read JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def write_json(file_path: Union[str, Path], data: Any, indent: int = 2):
        """Write data to JSON."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    
    @staticmethod
    def export_lexicon(file_path: Union[str, Path], words: List[Dict], 
                      format: str = 'csv', metadata: Optional[Dict] = None):
        """Export lexicon in specified format."""
        file_path = Path(file_path)
        
        if format == 'csv':
            FileHandler.write_csv(file_path, words)
        
        elif format == 'txt':
            lines = []
            if metadata:
                lines.append(f"# Generated: {metadata.get('timestamp', datetime.now().isoformat())}")
                lines.append(f"# Language: {metadata.get('language', 'Unknown')}")
                lines.append(f"# Sentiment: {metadata.get('sentiment', 'Unknown')}")
                lines.append("")
            
            for word in words:
                if isinstance(word, dict):
                    lines.append(f"{word.get('word', '')}: {word.get('translation', '')}")
                else:
                    lines.append(str(word))
            
            file_path.write_text('\\n'.join(lines), encoding='utf-8')
        
        elif format == 'json':
            export_data = {
                'metadata': metadata or {},
                'words': words,
                'count': len(words)
            }
            FileHandler.write_json(file_path, export_data)
        
        elif format == 'xlsx':
            df = pd.DataFrame(words)
            df.to_excel(file_path, index=False)
    
    @staticmethod
    def generate_filename(prefix: str, language: str, sentiment: Optional[str] = None, 
                         extension: str = 'csv') -> str:
        """Generate standardized filename."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        parts = [prefix, language.lower()]
        
        if sentiment:
            parts.append(sentiment.lower())
        
        parts.append(timestamp)
        filename = '_'.join(parts)
        return f"{filename}.{extension.lstrip('.')}"
''')
    print("✅ Created: utils/file_handlers.py")
    
    # ==================== CONFIG ====================
    
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # config/__init__.py
    Path("config/__init__.py").write_text('"""Config package."""\n')
    print("✅ Created: config/__init__.py")
    
    # config/settings.py
    Path("config/settings.py").write_text('''"""Application configuration and settings."""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
LEXICONS_DIR = DATA_DIR / 'lexicons'
CORPORA_DIR = DATA_DIR / 'corpora'
EXPORTS_DIR = DATA_DIR / 'exports'
CACHE_DIR = DATA_DIR / 'cache'

# Create directories
for directory in [DATA_DIR, LEXICONS_DIR, CORPORA_DIR, EXPORTS_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model configurations
MODELS_CONFIG = {
    'openai': {
        'models': ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo'],
        'default': 'gpt-4o',
        'max_tokens': 4000
    },
    'claude': {
        'models': [
            'claude-sonnet-4-20250514',
            'claude-3-5-sonnet-20241022',
            'claude-3-opus-20240229'
        ],
        'default': 'claude-sonnet-4-20250514',
        'max_tokens': 4096
    },
    'gemini': {
        'models': ['gemini-1.5-pro', 'gemini-1.5-flash'],
        'default': 'gemini-1.5-pro',
        'max_tokens': 8192
    }
}

# Generation parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 500
WORDS_PER_BATCH = 20
MAX_BATCHES = 50

# Sentiment categories
SENTIMENT_CATEGORIES = ['positive', 'negative', 'neutral']

# Prompt strategies
PROMPT_STRATEGIES = ['zero-shot', 'few-shot', 'in-context']

# Export formats
EXPORT_FORMATS = ['csv', 'txt', 'json', 'xlsx']

# File size limits (MB)
MAX_CORPUS_SIZE_MB = 100
MAX_LEXICON_SIZE_MB = 50

# Validation thresholds
MIN_WORD_LENGTH = 2
MAX_WORD_LENGTH = 50
MIN_CONFIDENCE_SCORE = 0.0
MAX_CONFIDENCE_SCORE = 1.0

# Bootstrapping parameters
BOOTSTRAP_MIN_SEED_SIZE = 5
BOOTSTRAP_MAX_ITERATIONS = 10
BOOTSTRAP_SIMILARITY_THRESHOLD = 0.7
''')
    print("✅ Created: config/settings.py")
    
    # config/languages.py
    Path("config/languages.py").write_text('''"""Language-specific configuration for African languages."""

from typing import Dict, List

# Supported languages
LANGUAGES = {
    'sepedi': {
        'name': 'Sepedi',
        'native_name': 'Sepedi',
        'iso_code': 'nso',
        'family': 'Bantu',
        'speakers': '4.7 million'
    },
    'sesotho': {
        'name': 'Sesotho',
        'native_name': 'Sesotho',
        'iso_code': 'sot',
        'family': 'Bantu',
        'speakers': '5.6 million'
    },
    'setswana': {
        'name': 'Setswana',
        'native_name': 'Setswana',
        'iso_code': 'tsn',
        'family': 'Bantu',
        'speakers': '8.2 million'
    }
}

# Common prefixes
COMMON_PREFIXES = {
    'sepedi': ['mo', 'ba', 'le', 'di', 'se', 'bo', 'go', 'ma'],
    'sesotho': ['mo', 'ba', 'le', 'di', 'se', 'bo', 'ho', 'ma'],
    'setswana': ['mo', 'ba', 'le', 'di', 'se', 'bo', 'go', 'ma']
}

# Orthography rules
ORTHOGRAPHY_RULES = {
    'sepedi': {
        'uses_diacritics': False,
        'special_characters': ['š', 'ž'],
        'vowels': ['a', 'e', 'i', 'o', 'u']
    },
    'sesotho': {
        'uses_diacritics': False,
        'special_characters': ['š'],
        'vowels': ['a', 'e', 'i', 'o', 'u']
    },
    'setswana': {
        'uses_diacritics': False,
        'special_characters': ['š'],
        'vowels': ['a', 'e', 'i', 'o', 'u']
    }
}

# Seed words for bootstrapping
SEED_WORDS = {
    'sepedi': {
        'positive': ['thabo', 'katlego', 'tshepo', 'lerato', 'kgotso'],
        'negative': ['lehloeo', 'bothata', 'bohale', 'masoha', 'pherekano'],
        'neutral': ['motse', 'nako', 'pula', 'lefika', 'sediba']
    },
    'sesotho': {
        'positive': ['thabo', 'katleho', 'khotso', 'lerato', 'tshepo'],
        'negative': ['lehloeo', 'bohale', 'bothata', 'masoha', 'pherekano'],
        'neutral': ['motse', 'nako', 'pula', 'sediba', 'lefika']
    },
    'setswana': {
        'positive': ['thabo', 'boikokobetso', 'katlego', 'tshepo', 'lorato'],
        'negative': ['lehloeo', 'bothata', 'bohale', 'pherekano', 'masoha'],
        'neutral': ['motse', 'nako', 'pula', 'lefika', 'sediba']
    }
}

# Stop words
STOP_WORDS = {
    'sepedi': ['le', 'ka', 'go', 'ya', 'ke', 'wa', 'mo', 'di', 'se', 'ba'],
    'sesotho': ['le', 'ka', 'ho', 'ya', 'ke', 'wa', 'mo', 'di', 'se', 'ba'],
    'setswana': ['le', 'ka', 'go', 'ya', 'ke', 'wa', 'mo', 'di', 'se', 'ba']
}

def get_language_info(language_code: str) -> Dict:
    """Get language information."""
    return LANGUAGES.get(language_code.lower(), {})

def get_seed_words(language: str, sentiment: str) -> List[str]:
    """Get seed words for language and sentiment."""
    lang_key = language.lower()
    if lang_key in SEED_WORDS and sentiment in SEED_WORDS[lang_key]:
        return SEED_WORDS[lang_key][sentiment]
    return []

def get_supported_languages() -> List[str]:
    """Get list of supported language names."""
    return [lang['name'] for lang in LANGUAGES.values()]
''')
    print("✅ Created: config/languages.py")
    
    print()
    print("=" * 60)
    print("✅ All utils and config modules created!")
    print("=" * 60)
    print()
    print("Created:")
    print("  📁 utils/")
    print("     - validators.py")
    print("     - deduplicator.py")
    print("     - file_handlers.py")
    print()
    print("  📁 config/")
    print("     - settings.py")
    print("     - languages.py")
    print()
    print("🚀 Ready to run: streamlit run app.py")

if __name__ == "__main__":
    create_all()