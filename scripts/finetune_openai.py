#!/usr/bin/env python3
"""Fine-tune OpenAI models."""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.fine_tuning import OpenAIFineTuner

def finetune_language(language: str, api_key: str):
    """Fine-tune for a specific language."""
    
    finetuner = OpenAIFineTuner(api_key)
    
    # File paths
    data_dir = Path('data/finetuning') / language.lower()
    train_file = data_dir / 'train.jsonl'
    val_file = data_dir / 'val.jsonl'
    
    if not train_file.exists():
        print(f"❌ Training file not found: {train_file}")
        return
    
    print(f"\n{'='*60}")
    print(f"Fine-tuning for {language}")
    print(f"{'='*60}\n")
    
    # Upload files
    print("📤 Uploading training file...")
    train_file_id = finetuner.upload_training_file(str(train_file))
    
    val_file_id = None
    if val_file.exists():
        print("📤 Uploading validation file...")
        val_file_id = finetuner.upload_training_file(str(val_file))
    
    # Start fine-tuning
    print("\n🚀 Starting fine-tuning...")
    job_id = finetuner.start_fine_tuning(
        training_file_id=train_file_id,
        validation_file_id=val_file_id,
        model='gpt-4o-mini-2024-07-18',
        suffix=f'{language.lower()}-sentiment',
        hyperparameters={
            'n_epochs': 3,
            'batch_size': 4,
            'learning_rate_multiplier': 1.0
        }
    )
    
    print(f"\n✅ Fine-tuning job created: {job_id}")
    print(f"💡 Track at: https://platform.openai.com/finetune/{job_id}")
    
    # Wait for completion (optional)
    wait = input("\nWait for completion? (y/n): ").lower()
    if wait == 'y':
        model_id = finetuner.wait_for_completion(job_id, check_interval=60)
        if model_id:
            print(f"\n🎉 Fine-tuning complete!")
            print(f"📦 Model ID: {model_id}")
            
            # Save model ID
            model_file = data_dir / 'model_id.txt'
            model_file.write_text(model_id)
            print(f"💾 Saved to: {model_file}")

def main():
    """Main function."""
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        api_key = input("Enter OpenAI API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided")
        return
    
    # Choose language
    print("\nSelect language:")
    print("1. Sepedi")
    print("2. Sesotho")
    print("3. Setswana")
    print("4. All languages")
    
    choice = input("\nChoice (1-4): ").strip()
    
    languages = {
        '1': ['Sepedi'],
        '2': ['Sesotho'],
        '3': ['Setswana'],
        '4': ['Sepedi', 'Sesotho', 'Setswana']
    }
    
    selected = languages.get(choice, [])
    
    if not selected:
        print("❌ Invalid choice")
        return
    
    for language in selected:
        finetune_language(language, api_key)

if __name__ == '__main__':
    main()