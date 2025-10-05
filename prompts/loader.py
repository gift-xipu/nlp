"""
Prompt loading utilities.
"""

from pathlib import Path
from typing import Optional

class PromptLoader:
    """Handles loading and formatting of prompt templates."""
    
    def __init__(self, base_dir: str = "prompts"):
        self.base_dir = Path(base_dir)
        self._cache = {}
    
    def load_prompt(self, strategy: str, task: str, language: Optional[str] = None, 
                   sentiment: Optional[str] = None) -> str:
        """Load a prompt template."""
        prompt_path = self.base_dir / strategy / task / "prompt.txt"
        
        cache_key = str(prompt_path)
        if cache_key not in self._cache:
            if not prompt_path.exists():
                raise FileNotFoundError(f"Prompt not found: {prompt_path}")
            
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self._cache[cache_key] = f.read()
        
        prompt = self._cache[cache_key]
        
        if language:
            prompt = prompt.replace("{language}", language)
        if sentiment:
            prompt = prompt.replace("{sentiment}", sentiment)
        
        return prompt

def load_prompt(strategy: str, task: str, **kwargs) -> str:
    """Convenience function to load prompt."""
    loader = PromptLoader()
    return loader.load_prompt(strategy, task, **kwargs)
