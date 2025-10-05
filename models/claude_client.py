"""Anthropic Claude client implementation."""
from models.llm_client import LLMClient
import anthropic

class ClaudeClient(LLMClient):
    """Client for Anthropic Claude models."""
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        super().__init__(model)
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 200) -> str:
        """Generate text using Claude."""
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
