"""
Fine-tuning system without torch dependency.
Only uses API-based fine-tuning (OpenAI).
Local fine-tuning can be added later if needed.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import random

class CorpusPreparator:
    """Prepares parallel corpora for fine-tuning."""
    
    def __init__(self, data_dir: str = 'data/translated'):
        self.data_dir = Path(data_dir)
    
    def load_parallel_corpus(self, language: str) -> Tuple[List[str], List[str]]:
        """
        Load parallel corpus for a language.
        
        Expected structure:
        data/translated/sepedi/english.txt
        data/translated/sepedi/sepedi.txt
        
        Returns:
            (english_sentences, target_language_sentences)
        """
        lang_dir = self.data_dir / language.lower()
        
        if not lang_dir.exists():
            raise FileNotFoundError(f"Directory not found: {lang_dir}")
        
        # Try .txt files
        eng_txt = lang_dir / 'english.txt'
        lang_txt = lang_dir / f'{language.lower()}.txt'
        
        if eng_txt.exists() and lang_txt.exists():
            with open(eng_txt, 'r', encoding='utf-8') as f:
                english_texts = [line.strip() for line in f if line.strip()]
            
            with open(lang_txt, 'r', encoding='utf-8') as f:
                target_texts = [line.strip() for line in f if line.strip()]
        
        # Try CSV
        elif (lang_dir / 'corpus.csv').exists():
            df = pd.read_csv(lang_dir / 'corpus.csv')
            english_texts = df['english'].tolist()
            target_texts = df[language.lower()].tolist()
        
        else:
            raise FileNotFoundError(
                f"No corpus found in {lang_dir}.\n"
                f"Expected: {eng_txt} + {lang_txt}\n"
                f"Or: {lang_dir / 'corpus.csv'}"
            )
        
        if len(english_texts) != len(target_texts):
            raise ValueError(
                f"Misaligned corpus: {len(english_texts)} English vs "
                f"{len(target_texts)} {language} sentences"
            )
        
        print(f"✅ Loaded {len(english_texts)} parallel sentences for {language}")
        return english_texts, target_texts
    
    def create_training_data_for_all_tasks(
        self,
        language: str,
        english_texts: List[str],
        target_texts: List[str],
        sentiment_lexicon: Optional[List[Dict]] = None
    ) -> Dict[str, List[Dict]]:
        """Create training data for all tasks."""
        
        # 1. LIST WORDS
        list_words_data = self._create_list_words_data(language, sentiment_lexicon)
        
        # 2. SENTIMENT BEARING
        sentiment_data = self._create_sentiment_analysis_data(
            language, english_texts, target_texts, sentiment_lexicon
        )
        
        # 3. TRANSLATION
        translation_data = self._create_translation_data(
            language, english_texts, target_texts
        )
        
        return {
            'list_words': list_words_data,
            'sentiment_bearing': sentiment_data,
            'translation': translation_data,
            'all_combined': list_words_data + sentiment_data + translation_data
        }
    
    def _create_list_words_data(
        self,
        language: str,
        lexicon: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """Training data for word generation."""
        if not lexicon or len(lexicon) < 20:
            return []
        
        training_data = []
        by_sentiment = {}
        
        for word in lexicon:
            sent = word.get('sentiment', 'neutral')
            if sent not in by_sentiment:
                by_sentiment[sent] = []
            by_sentiment[sent].append(word)
        
        for sentiment, words in by_sentiment.items():
            if len(words) < 20:
                continue
            
            for i in range(0, len(words), 20):
                batch = words[i:i+20]
                if len(batch) < 20:
                    continue
                
                word_list = "\n".join([
                    f"{w['word']}: {w['translation']}"
                    for w in batch
                ])
                
                example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": f"You are an expert linguist fluent in {language}."
                        },
                        {
                            "role": "user",
                            "content": f"Generate exactly 20 {sentiment} words in {language}.\n"
                                       f"Format: word: English translation"
                        },
                        {
                            "role": "assistant",
                            "content": word_list
                        }
                    ]
                }
                training_data.append(example)
        
        return training_data
    
    def _create_sentiment_analysis_data(
        self,
        language: str,
        english_texts: List[str],
        target_texts: List[str],
        lexicon: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """Training data for sentiment analysis."""
        training_data = []
        
        for eng, target in zip(english_texts, target_texts):
            sentiment = self._infer_sentiment(eng, lexicon)
            
            example = {
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are an expert linguist analyzing sentiment in {language}."
                    },
                    {
                        "role": "user",
                        "content": f"Analyze sentiment:\n{target}"
                    },
                    {
                        "role": "assistant",
                        "content": f"{target}|{sentiment['label']}|{sentiment['score']:.2f}|{sentiment['reasoning']}"
                    }
                ]
            }
            training_data.append(example)
        
        return training_data
    
    def _create_translation_data(
        self,
        language: str,
        english_texts: List[str],
        target_texts: List[str]
    ) -> List[Dict]:
        """Training data for translation."""
        training_data = []
        
        for eng, target in zip(english_texts, target_texts):
            # English → Target
            training_data.append({
                "messages": [
                    {"role": "system", "content": f"You are an expert English to {language} translator."},
                    {"role": "user", "content": f"Translate to {language}:\n{eng}"},
                    {"role": "assistant", "content": target}
                ]
            })
            
            # Target → English
            training_data.append({
                "messages": [
                    {"role": "system", "content": f"You are an expert {language} to English translator."},
                    {"role": "user", "content": f"Translate to English:\n{target}"},
                    {"role": "assistant", "content": eng}
                ]
            })
        
        return training_data
    
    def _infer_sentiment(self, text: str, lexicon: Optional[List[Dict]] = None) -> Dict:
        """Simple sentiment inference."""
        text_lower = text.lower()
        
        positive_kw = ['good', 'great', 'happy', 'joy', 'love', 'success', 'peace']
        negative_kw = ['bad', 'sad', 'pain', 'hate', 'fail', 'angry', 'poor']
        
        pos_count = sum(1 for kw in positive_kw if kw in text_lower)
        neg_count = sum(1 for kw in negative_kw if kw in text_lower)
        
        if pos_count > neg_count:
            return {'label': 'positive', 'score': 0.8, 'reasoning': 'Positive indicators'}
        elif neg_count > pos_count:
            return {'label': 'negative', 'score': 0.8, 'reasoning': 'Negative indicators'}
        else:
            return {'label': 'neutral', 'score': 0.75, 'reasoning': 'No clear sentiment'}
    
    def save_training_data(self, data: List[Dict], output_path: str):
        """Save as JSONL."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✅ Saved {len(data)} examples to {output_path}")
    
    def split_train_val(self, data: List[Dict], val_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict]]:
        """Split into train and validation."""
        random.shuffle(data)
        split_idx = int(len(data) * (1 - val_ratio))
        return data[:split_idx], data[split_idx:]


class FineTuningManager:
    """Manages fine-tuned models registry."""
    
    def __init__(self, models_dir: str = 'data/finetuned_models'):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.models_dir / 'registry.json'
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load registry."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Save registry."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(
        self,
        provider: str,
        language: str,
        model_id: str,
        base_model: str,
        task_type: str = 'all',
        metadata: Optional[Dict] = None
    ):
        """Register a fine-tuned model."""
        key = f"{provider}_{language.lower()}_{task_type}"
        
        self.registry[key] = {
            'provider': provider,
            'language': language,
            'model_id': model_id,
            'base_model': base_model,
            'task_type': task_type,
            'created_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self._save_registry()
        print(f"✅ Registered {provider} model for {language}: {model_id}")
    
    def get_model_for_language(
        self,
        provider: str,
        language: str,
        task_type: str = 'all'
    ) -> Optional[str]:
        """Get fine-tuned model ID."""
        key = f"{provider}_{language.lower()}_{task_type}"
        
        if key in self.registry:
            return self.registry[key]['model_id']
        
        # Try 'all' if specific task not found
        all_key = f"{provider}_{language.lower()}_all"
        if all_key in self.registry:
            return self.registry[all_key]['model_id']
        
        return None
    
    def list_models(self, language: Optional[str] = None) -> List[Dict]:
        """List all models."""
        if language:
            return [
                info for key, info in self.registry.items()
                if info['language'].lower() == language.lower()
            ]
        return list(self.registry.values())
    
    def has_finetuned_model(self, provider: str, language: str) -> bool:
        """Check if model exists."""
        return self.get_model_for_language(provider, language) is not None


class OpenAIFineTuner:
    """Fine-tune OpenAI models."""
    
    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
    
    def upload_file(self, file_path: str) -> str:
        """Upload training file."""
        print(f"📤 Uploading {file_path}...")
        with open(file_path, 'rb') as f:
            response = self.client.files.create(file=f, purpose='fine-tune')
        print(f"✅ Uploaded: {response.id}")
        return response.id
    
    def start_finetuning(
        self,
        train_file_id: str,
        val_file_id: Optional[str] = None,
        model: str = 'gpt-4o-mini-2024-07-18',
        suffix: Optional[str] = None
    ) -> str:
        """Start fine-tuning."""
        print(f"🚀 Starting fine-tuning...")
        
        params = {'training_file': train_file_id, 'model': model}
        if val_file_id:
            params['validation_file'] = val_file_id
        if suffix:
            params['suffix'] = suffix
        
        job = self.client.fine_tuning.jobs.create(**params)
        print(f"✅ Job created: {job.id}")
        return job.id
    
    def check_status(self, job_id: str) -> Dict:
        """Check job status."""
        job = self.client.fine_tuning.jobs.retrieve(job_id)
        return {
            'id': job.id,
            'status': job.status,
            'model': job.fine_tuned_model,
            'created_at': job.created_at
        }