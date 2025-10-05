"""Application configuration and settings."""

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
