from llm_client import LLMClient
import openai

class OpenAIClient(LLMClient):
    def __init__(self, api_key, model="gpt 4o"):
        super().__init__(model)
        openai.api_key = api_key

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 200) -> str:
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response["choices"][0]["message"]["content"]
    
