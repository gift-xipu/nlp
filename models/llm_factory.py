"""Factory pattern for creating LLM clients."""
from typing import Optional
from models.llm_client import LLMClient
from models.openai_client import OpenAIClient
from models.claude_client import ClaudeClient
from models.gemini_client import GeminiClient

class LLMFactory:
    """Factory for creating LLM clients."""
    
    _clients = {
        'openai': OpenAIClient,
        'claude': ClaudeClient,
        'gemini': GeminiClient
    }
    
    @staticmethod
    def create_client(provider: str, api_key: str, model: Optional[str] = None) -> LLMClient:
        """Create an LLM client."""
        provider_lower = provider.lower()
        
        if provider_lower not in LLMFactory._clients:
            raise ValueError(f"Unsupported provider: {provider}")
        
        client_class = LLMFactory._clients[provider_lower]
        
        if model:
            return client_class(api_key=api_key, model=model)
        else:
            return client_class(api_key=api_key)
    
    @staticmethod
    def get_supported_providers() -> list:
        """Get list of supported providers."""
        return list(LLMFactory._clients.keys())
