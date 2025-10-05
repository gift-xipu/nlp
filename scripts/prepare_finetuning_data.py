#!/usr/bin/env python3
"""Prepare fine-tuning data from parallel corpora."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.fine_tuning import CorpusPreparator

def prepare_all_languages():
    """Prepare training data for all languages."""
    
    languages = ['Sepedi', 'Sesotho', 'Setswana']
    preparator = CorpusPreparator('data/translated')
    
    for language in languages:
        print(f"\n{'='*60}")
        print(f"Processing {language}")
        print(f"{'='*60}")
        
        try:
            # Load corpus
            english_texts, target_texts = preparator.load_parallel_corpus(language)
            
            # Create sentiment training data
            sentiment_data = preparator.create_sentiment_training_data(
                language=language,
                english_texts=english_texts,
                target_texts=target_texts
            )
            
            # Create translation training data
            translation_data = preparator.create_translation_training_data(
                language=language,
                english_texts=english_texts,
                target_texts=target_texts
            )
            
            # Combine
            all_data = sentiment_data + translation_data
            
            # Split
            train_data, val_data = preparator.split_train_val(
                all_data, 
                val_ratio=0.1
            )
            
            # Save
            output_dir = Path('data/finetuning') / language.lower()
            output_dir.mkdir(parents=True, exist_ok=True)
            
            preparator.save_training_data(
                train_data,
                output_dir / 'train.jsonl'
            )
            preparator.save_training_data(
                val_data,
                output_dir / 'val.jsonl'
            )
            
            print(f"✅ {language} complete!")
            
        except Exception as e:
            print(f"❌ Error with {language}: {e}")
            continue
    
    print("\n" + "="*60)
    print("✅ All languages processed!")
    print("="*60)

if __name__ == '__main__':
    prepare_all_languages()