"""
Task for generating word lists using LLMs.
"""

import re
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from models.llm_client import LLMClient
from prompts.loader import load_prompt
from utils.deduplicator import WordDeduplicator
from utils.validators import WordValidator
from config.settings import WORDS_PER_BATCH

class ListWordsTask:
    """Generates sentiment word lists using LLMs."""
    
    def __init__(self, llm_client: LLMClient, language: str, sentiment: str, prompt_strategy: str = 'few-shot'):
        self.llm_client = llm_client
        self.language = language
        self.sentiment = sentiment
        self.prompt_strategy = prompt_strategy
        self.deduplicator = WordDeduplicator()
        self.validator = WordValidator(language)
        
        # Load prompt template
        self.prompt_template = load_prompt(
            strategy=prompt_strategy,
            task='list-words',
            language=language,
            sentiment=sentiment
        )
    
    def generate_batch(self, temperature: float = 0.7, max_tokens: int = 500) -> List[Dict[str, str]]:
        """Generate one batch of words."""
        response = self.llm_client.generate(
            prompt=self.prompt_template,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return self._parse_response(response)
    
    def generate_words(self, target_count: int = 1000, temperature: float = 0.7, 
                      max_tokens: int = 500, progress_callback: Optional[callable] = None,
                      validate: bool = True) -> Tuple[List[Dict[str, str]], Dict[str, any]]:
        """Generate target number of words with deduplication."""
        all_words = []
        batch_count = 0
        max_batches = (target_count // WORDS_PER_BATCH) + 10
        
        stats = {
            'total_generated': 0,
            'duplicates_removed': 0,
            'invalid_removed': 0,
            'batches_processed': 0,
            'unique_words': 0
        }
        
        while len(all_words) < target_count and batch_count < max_batches:
            batch_count += 1
            
            # Generate batch
            batch_words = self.generate_batch(temperature=temperature, max_tokens=max_tokens)
            
            stats['total_generated'] += len(batch_words)
            stats['batches_processed'] = batch_count
            
            # Validate if enabled
            if validate:
                valid_words = []
                for word in batch_words:
                    if self.validator.is_valid_word(word['word']):
                        valid_words.append(word)
                    else:
                        stats['invalid_removed'] += 1
                batch_words = valid_words
            
            # Deduplicate
            before_dedup = len(all_words)
            all_words = self.deduplicator.add_words(all_words, batch_words)
            stats['duplicates_removed'] += (before_dedup + len(batch_words) - len(all_words))
            
            # Update progress
            if progress_callback:
                progress_callback(len(all_words), target_count, batch_count)
            
            # Add variation
            temperature = min(0.9, temperature + 0.02)
        
        stats['unique_words'] = len(all_words)
        
        if len(all_words) > target_count:
            all_words = all_words[:target_count]
        
        return all_words, stats
    
    def _parse_response(self, response: str) -> List[Dict[str, str]]:
        """Parse LLM response into structured word list."""
        words = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            if not line or line.startswith('#') or line.upper() == 'DONE':
                continue
            
            if re.match(r'^\d+\.?\s', line):
                line = re.sub(r'^\d+\.?\s*', '', line)
            
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    word = parts[0].strip().strip('*_`"\'')
                    translation = parts[1].strip().strip('*_`"\'')
                    
                    if word and translation:
                        words.append({
                            'word': word,
                            'translation': translation,
                            'language': self.language,
                            'sentiment': self.sentiment,
                            'prompt_strategy': self.prompt_strategy
                        })
        
        return words
