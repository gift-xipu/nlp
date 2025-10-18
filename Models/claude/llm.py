import anthropic
from models.llm_client import LLMClient

class ClaudeLLM(LLMClient):
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        super().__init__(model)
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 200) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Claude API error: {str(e)}")
