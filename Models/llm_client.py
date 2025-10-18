from abc import ABC, abstractmethod

class LLMClient(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 200) -> str:

        pass