#!/usr/bin/env python3
"""Evaluate fine-tuned models."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.llm_factory import LLMFactory
from tasks.sentiment_bearing import SentimentBearingTask
import json

def evaluate_model(
    provider: str,
    model_id: str,
    api_key: str,
    language: str,
    test_words: list
):
    """Evaluate model on test words."""
    
    print(f"\n{'='*60}")
    print(f"Evaluating {provider} model: {model_id}")
    print(f"Language: {language}")
    print(f"{'='*60}\n")
    
    # Create client
    client = LLMFactory.create_client(
        provider=provider,
        api_key=api_key,
        model=model_id
    )
    
    # Create task
    task = SentimentBearingTask(
        llm_client=client,
        language=language,
        prompt_strategy='few-shot'
    )
    
    # Analyze test words
    results = task.analyze_batch(test_words, batch_size=10)
    
    # Calculate metrics
    correct = 0
    total = len(results)
    
    for result, expected in zip(results, test_words):
        if result['sentiment'] == expected.get('expected_sentiment'):
            correct += 1
    
    accuracy = correct / total * 100 if total > 0 else 0
    
    print(f"\n📊 Results:")
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Correct: {correct}/{total}")
    
    return results, accuracy

def load_test_set(language: str):
    """Load test set for language."""
    test_file = Path('data/test') / language.lower() / 'test_words.json'
    
    if test_file.exists():
        with open(test_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Default test words if no test file
    return [
        {'word': 'thabo', 'translation': 'joy', 'expected_sentiment': 'positive'},
        {'word': 'bohloko', 'translation': 'pain', 'expected_sentiment': 'negative'},
        {'word': 'motse', 'translation': 'village', 'expected_sentiment': 'neutral'}
    ]

def main():
    """Main evaluation function."""
    
    # Load test sets
    test_words = load_test_set('Sepedi')
    
    # Test base model
    print("\n" + "="*60)
    print("BASELINE: Base Model")
    print("="*60)
    
    base_results, base_acc = evaluate_model(
        provider='openai',
        model_id='gpt-4o-mini',
        api_key='your-api-key',
        language='Sepedi',
        test_words=test_words
    )
    
    # Test fine-tuned model
    print("\n" + "="*60)
    print("FINE-TUNED: Your Model")
    print("="*60)
    
    ft_results, ft_acc = evaluate_model(
        provider='openai',
        model_id='ft:gpt-4o-mini:org:sepedi:id',  # Your fine-tuned model ID
        api_key='your-api-key',
        language='Sepedi',
        test_words=test_words
    )
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Base Model:       {base_acc:.2f}%")
    print(f"Fine-Tuned Model: {ft_acc:.2f}%")
    print(f"Improvement:      {ft_acc - base_acc:+.2f}%")

if __name__ == '__main__':
    main()