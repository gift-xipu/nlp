from models.llm_client import LLMClient
from openai import OpenAI

class OpenAILLM(LLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        super().__init__(model)
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 200) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")