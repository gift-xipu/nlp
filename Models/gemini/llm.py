import google.generativeai as genai
from models.llm_client import LLMClient

class GeminiLLM(LLMClient):
    def __init__(self, api_key: str, model: str = "gemini-pro"):
        super().__init__(model)
        genai.configure(api_key=api_key)
        self.model_client = genai.GenerativeModel(model)

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 200) -> str:
        try:
            response = self.model_client.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
                }
            )
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}")
