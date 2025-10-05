"""Local model client for fine-tuned models."""
from models.llm_client import LLMClient
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LocalClient(LLMClient):
    """Client for local fine-tuned models."""
    
    def __init__(self, model_path: str):
        super().__init__(model_path)
        
        print(f"🔧 Loading local model from {model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print("✅ Local model loaded")
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 200) -> str:
        """Generate text using local model."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            raise RuntimeError(f"Local model error: {str(e)}")