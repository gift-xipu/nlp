import google.generativeai as genai

class GeminiClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        super().__init__(model)
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 200) -> str:
        response = self.client.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }
        )
        return response.text
