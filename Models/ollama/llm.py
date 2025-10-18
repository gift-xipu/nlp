import ollama
from models.llm_client import LLMClient

class OllamaLLM(LLMClient):
    def __init__(self, model: str = "llama2"):
        super().__init__(model)

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 200) -> str:
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response['message']['content']
        except Exception as e:
            raise RuntimeError(f"Ollama error: {str(e)}")
