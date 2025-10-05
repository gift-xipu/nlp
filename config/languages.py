"""Language-specific configuration for African languages."""

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
