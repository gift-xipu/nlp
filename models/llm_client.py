"""Abstract base class for LLM clients."""
from abc import ABC, abstractmethod

class LLMClient(ABC):
    """Abstract base class for all LLM clients."""
    
    def __init__(self, model: str):
        self.model = model
    
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 200) -> str:
        """Generate text from prompt."""
        pass
