"""
Lexicon bootstrapping using semantic similarity.
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from tqdm import tqdm
from models.llm_client import LLMClient
from config.languages import get_seed_words
from config.settings import BOOTSTRAP_MIN_SEED_SIZE, BOOTSTRAP_MAX_ITERATIONS, BOOTSTRAP_SIMILARITY_THRESHOLD

class LexiconBootstrapping:
    """Implements bootstrapping for semantic lexicon learning."""
    
    def __init__(self, llm_client: LLMClient, language: str, sentiment: str, seed_words: Optional[List[str]] = None):
        self.llm_client = llm_client
        self.language = language.lower()
        self.sentiment = sentiment
        
        if seed_words:
            self.seed_words = seed_words
        else:
            self.seed_words = get_seed_words(self.language, sentiment)
        
        if len(self.seed_words) < BOOTSTRAP_MIN_SEED_SIZE:
            raise ValueError(f"Need at least {BOOTSTRAP_MIN_SEED_SIZE} seed words")
        
        self.lexicon: Set[str] = set(self.seed_words)
        self.word_scores: Dict[str, float] = {word: 1.0 for word in self.seed_words}
        self.iteration_history: List[Dict] = []
    
    def bootstrap(self, candidate_words: List[Dict[str, str]], max_iterations: int = BOOTSTRAP_MAX_ITERATIONS,
                 similarity_threshold: float = BOOTSTRAP_SIMILARITY_THRESHOLD, min_confidence: float = 0.6,
                 expansion_rate: int = 10) -> Tuple[List[Dict[str, any]], Dict[str, any]]:
        """Run bootstrapping algorithm."""
        print(f"🌱 Bootstrapping with {len(self.seed_words)} seeds...")
        
        candidates = self._prepare_candidates(candidate_words, min_confidence)
        
        for iteration in range(max_iterations):
            print(f"\n📊 Iteration {iteration + 1}/{max_iterations}")
            
            similar_words = self._find_similar_words(candidates, similarity_threshold, expansion_rate)
            
            if not similar_words:
                print("✅ No more similar words")
                break
            
            added_count = 0
            for word_info in similar_words:
                word = word_info['word']
                if word not in self.lexicon:
                    self.lexicon.add(word)
                    self.word_scores[word] = word_info['similarity_score']
                    added_count += 1
            
            self.iteration_history.append({
                'iteration': iteration + 1,
                'lexicon_size': len(self.lexicon),
                'words_added': added_count,
                'avg_similarity': np.mean([w['similarity_score'] for w in similar_words])
            })
            
            print(f"   Added {added_count} words. Size: {len(self.lexicon)}")
            
            added_words_set = {w['word'] for w in similar_words}
            candidates = [c for c in candidates if c['word'] not in added_words_set]
            
            if added_count == 0:
                break
        
        final_lexicon = self._build_final_lexicon()
        stats = self._generate_statistics()
        
        return final_lexicon, stats
    
    def _prepare_candidates(self, candidate_words: List[Dict[str, str]], min_confidence: float) -> List[Dict[str, str]]:
        """Filter candidates."""
        candidates = []
        for word_dict in candidate_words:
            word = word_dict.get('word', '')
            if word in self.lexicon:
                continue
            confidence = word_dict.get('confidence_score', 1.0)
            if confidence < min_confidence:
                continue
            candidates.append(word_dict)
        return candidates
    
    def _find_similar_words(self, candidates: List[Dict[str, str]], threshold: float, top_k: int = 10) -> List[Dict[str, any]]:
        """Find similar words using LLM."""
        similar_words = []
        lexicon_sample = list(self.lexicon)[:20]
        
        for candidate in candidates[:100]:  # Limit for efficiency
            word = candidate.get('word', '')
            translation = candidate.get('translation', '')
            
            similarity = self._calculate_semantic_similarity(word, translation, lexicon_sample)
            
            if similarity >= threshold:
                similar_words.append({
                    'word': word,
                    'translation': translation,
                    'similarity_score': similarity,
                    'language': self.language,
                    'sentiment': self.sentiment,
                    'bootstrap_iteration': len(self.iteration_history) + 1
                })
        
        similar_words.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similar_words[:top_k]
    
    def _calculate_semantic_similarity(self, word: str, translation: str, lexicon_words: List[str]) -> float:
        """Calculate similarity using LLM."""
        prompt = f"""Rate semantic similarity (0.0-1.0):

Word: {word} ({translation})
Sentiment: {self.sentiment}
Reference words: {', '.join(lexicon_words[:10])}

Is "{word}" similar in {self.sentiment} sentiment?
Score (0.0-1.0):"""
        
        try:
            response = self.llm_client.generate(prompt=prompt, temperature=0.1, max_tokens=10)
            import re
            score_match = re.search(r'(\d*\.?\d+)', response)
            if score_match:
                score = float(score_match.group(1))
                if score > 1.0:
                    score = score / 10.0 if score <= 10.0 else score / 100.0
                return max(0.0, min(1.0, score))
        except:
            pass
        
        return 0.5
    
    def _build_final_lexicon(self) -> List[Dict[str, any]]:
        """Build final lexicon with metadata."""
        lexicon_list = []
        for word in self.lexicon:
            lexicon_list.append({
                'word': word,
                'language': self.language,
                'sentiment': self.sentiment,
                'confidence_score': self.word_scores.get(word, 0.5),
                'is_seed': word in self.seed_words,
                'source': 'seed' if word in self.seed_words else 'bootstrap'
            })
        
        lexicon_list.sort(key=lambda x: x['confidence_score'], reverse=True)
        return lexicon_list
    
    def _generate_statistics(self) -> Dict[str, any]:
        """Generate statistics."""
        seed_count = len([w for w in self.lexicon if w in self.seed_words])
        
        return {
            'total_words': len(self.lexicon),
            'seed_words': seed_count,
            'bootstrapped_words': len(self.lexicon) - seed_count,
            'expansion_ratio': round(len(self.lexicon) / len(self.seed_words), 2),
            'iterations_completed': len(self.iteration_history),
            'iteration_history': self.iteration_history,
            'average_confidence': round(np.mean(list(self.word_scores.values())), 3)
        }
    
    def visualize_growth(self) -> Dict[str, List]:
        """Get data for visualization."""
        iterations = [0] + [h['iteration'] for h in self.iteration_history]
        sizes = [len(self.seed_words)] + [h['lexicon_size'] for h in self.iteration_history]
        
        return {
            'iterations': iterations,
            'lexicon_sizes': sizes,
            'words_added_per_iteration': [h['words_added'] for h in self.iteration_history]
        }
