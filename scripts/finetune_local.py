#!/usr/bin/env python3
"""Fine-tune local models with LoRA/QLoRA."""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
from pathlib import Path

def load_training_data(file_path: str):
    """Load JSONL training data."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_training_data(examples, tokenizer):
    """Format data for training."""
    formatted = []
    
    for example in examples:
        messages = example['messages']
        
        # Convert to chat format
        conversation = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            conversation += f"<|{role}|>\n{content}\n"
        
        formatted.append(conversation)
    
    # Tokenize
    tokenized = tokenizer(
        formatted,
        truncation=True,
        max_length=512,
        padding='max_length'
    )
    
    return tokenized

def finetune_local_model(
    language: str,
    base_model: str = "meta-llama/Llama-2-7b-hf",  # or "mistralai/Mistral-7B-v0.1"
    output_dir: str = "models/finetuned/local"
):
    """Fine-tune a local model with LoRA."""
    
    print(f"\n{'='*60}")
    print(f"Fine-tuning local model for {language}")
    print(f"Base model: {base_model}")
    print(f"{'='*60}\n")
    
    # Load data
    data_dir = Path('data/finetuning') / language.lower()
    train_file = data_dir / 'train.jsonl'
    
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    
    print("📚 Loading training data...")
    train_data = load_training_data(str(train_file))
    print(f"✅ Loaded {len(train_data)} training examples")
    
    # Load tokenizer and model
    print(f"\n🔧 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True  # Use 8-bit quantization
    )
    
    print("✅ Model loaded")
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    print("\n🔧 Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # Which layers to adapt
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare dataset
    print("\n📊 Preparing dataset...")
    
    # Convert to Hugging Face dataset format
    dataset_dict = {
        'text': [
            '\n'.join([f"{m['role']}: {m['content']}" for m in ex['messages']])
            for ex in train_data
        ]
    }
    
    from datasets import Dataset
    dataset = Dataset.from_dict(dataset_dict)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=512)
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Training arguments
    output_path = Path(output_dir) / language.lower()
    
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        report_to="none"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    # Train
    print(f"\n🚀 Starting training...")
    trainer.train()
    
    # Save
    print(f"\n💾 Saving model...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"\n✅ Training complete!")
    print(f"📦 Model saved to: {output_path}")
    
    return str(output_path)

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fine-tune local models')
    parser.add_argument('--language', required=True, choices=['Sepedi', 'Sesotho', 'Setswana'])
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--output-dir', default='models/finetuned/local')
    
    args = parser.parse_args()
    
    try:
        model_path = finetune_local_model(
            language=args.language,
            base_model=args.model,
            output_dir=args.output_dir
        )
        
        print(f"\n🎉 Success! Model saved to: {model_path}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()