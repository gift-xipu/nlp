"""
Task for labeling words with sentiment scores and reasoning.
"""

import re
from typing import List, Dict, Optional
from tqdm import tqdm
from models.llm_client import LLMClient
from prompts.loader import load_prompt

class SentimentBearingTask:
    """Labels words with sentiment, scores, and reasoning."""
    
    def __init__(self, llm_client: LLMClient, language: str, prompt_strategy: str = 'few-shot'):
        self.llm_client = llm_client
        self.language = language
        self.prompt_strategy = prompt_strategy
        
        self.prompt_template = load_prompt(
            strategy=prompt_strategy,
            task='sentiment-bearing',
            language=language
        )
    
    def analyze_batch(self, words: List[Dict[str, str]], batch_size: int = 10,
                     temperature: float = 0.3, progress_callback: Optional[callable] = None) -> List[Dict[str, any]]:
        """Analyze sentiment for a batch of words."""
        analyzed_words = []
        total = len(words)
        
        for i in range(0, total, batch_size):
            batch = words[i:i + batch_size]
            batch_prompt = self._create_batch_prompt(batch)
            
            response = self.llm_client.generate(
                prompt=batch_prompt,
                temperature=temperature,
                max_tokens=1000
            )
            
            batch_results = self._parse_batch_response(batch, response)
            analyzed_words.extend(batch_results)
            
            if progress_callback:
                progress_callback(len(analyzed_words), total)
        
        return analyzed_words
    
    def _create_batch_prompt(self, words: List[Dict[str, str]]) -> str:
        """Create a prompt for batch processing."""
        words_list = []
        for idx, word_dict in enumerate(words, 1):
            word = word_dict.get('word', '')
            translation = word_dict.get('translation', '')
            words_list.append(f"{idx}. {word} ({translation})")
        
        batch_prompt = f"""You are analyzing {self.language} words for sentiment.

Analyze each word and provide:
1. Sentiment: positive, negative, or neutral
2. Confidence: 0.0 to 1.0
3. Reason: brief explanation (max 15 words)

Words:
{chr(10).join(words_list)}

Format: word|sentiment|score|reason

Example:
thabo|positive|0.95|expresses joy and happiness

Now analyze:"""
        
        return batch_prompt
    
    def _parse_batch_response(self, words: List[Dict[str, str]], response: str) -> List[Dict[str, any]]:
        """Parse batch sentiment response."""
        results = []
        lines = response.strip().split('\n')
        
        parsed_count = 0
        for line in lines:
            line = line.strip()
            if not line or ':' in line[:20]:
                continue
            
            if '|' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 4:
                    try:
                        score = float(parts[2])
                        if score > 1.0:
                            score = score / 100.0
                        score = max(0.0, min(1.0, score))
                    except:
                        score = 0.5
                    
                    sentiment = parts[1].lower()
                    if sentiment not in ['positive', 'negative', 'neutral']:
                        sentiment = 'neutral'
                    
                    if parsed_count < len(words):
                        original = words[parsed_count]
                        results.append({
                            'word': original.get('word', ''),
                            'translation': original.get('translation', ''),
                            'language': self.language,
                            'sentiment': sentiment,
                            'confidence_score': score,
                            'reasoning': parts[3][:200] if len(parts) > 3 else ''
                        })
                        parsed_count += 1
        
        # Fill missing
        while parsed_count < len(words):
            original = words[parsed_count]
            results.append({
                'word': original.get('word', ''),
                'translation': original.get('translation', ''),
                'language': self.language,
                'sentiment': 'neutral',
                'confidence_score': 0.5,
                'reasoning': 'Analysis incomplete'
            })
            parsed_count += 1
        
        return results
    
    def validate_sentiment_consistency(self, analyzed_words: List[Dict[str, any]], 
                                      original_sentiment: Optional[str] = None) -> Dict[str, any]:
        """Validate consistency of sentiment labels."""
        if not analyzed_words:
            return {'error': 'No words'}
        
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_score = 0
        high_confidence_count = 0
        
        for word in analyzed_words:
            sentiment = word.get('sentiment', 'neutral')
            score = word.get('confidence_score', 0)
            
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            total_score += score
            
            if score >= 0.7:
                high_confidence_count += 1
        
        total = len(analyzed_words)
        
        return {
            'total_words': total,
            'sentiment_distribution': sentiment_counts,
            'sentiment_percentages': {k: round(v / total * 100, 2) for k, v in sentiment_counts.items()},
            'average_confidence': round(total_score / total, 3),
            'high_confidence_count': high_confidence_count,
            'high_confidence_percentage': round(high_confidence_count / total * 100, 2)
        }
