"""Factory pattern for creating LLM clients with optional local model support."""
from typing import Optional
from models.llm_client import LLMClient
from models.openai_client import OpenAIClient
from models.claude_client import ClaudeClient
from models.gemini_client import GeminiClient

# Try to import local client (requires torch)
try:
    from models.local_client import LocalClient
    LOCAL_CLIENT_AVAILABLE = True
except ImportError:
    LOCAL_CLIENT_AVAILABLE = False
    LocalClient = None


class LLMFactory:
    """Factory for creating LLM clients."""
    
    _clients = {
        'openai': OpenAIClient,
        'claude': ClaudeClient,
        'gemini': GeminiClient
    }
    
    # Add local client only if available
    if LOCAL_CLIENT_AVAILABLE:
        _clients['local'] = LocalClient
    
    @staticmethod
    def create_client(
        provider: str,
        api_key: str = None,
        model: Optional[str] = None,
        use_finetuned: bool = False,
        finetuned_model_id: Optional[str] = None
    ) -> LLMClient:
        """
        Create an LLM client.
        
        Args:
            provider: 'openai', 'claude', 'gemini', or 'local'
            api_key: API key (not needed for local)
            model: Model name or path
            use_finetuned: Use fine-tuned model
            finetuned_model_id: Fine-tuned model ID
        
        Returns:
            LLMClient instance
        """
        provider_lower = provider.lower()
        
        if provider_lower not in LLMFactory._clients:
            available = ', '.join(LLMFactory._clients.keys())
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Available: {available}"
            )
        
        client_class = LLMFactory._clients[provider_lower]
        
        # Local models don't need API key
        if provider_lower == 'local':
            if not LOCAL_CLIENT_AVAILABLE:
                raise ImportError(
                    "Local client requires torch. Install with: "
                    "pip install torch transformers"
                )
            if not model and not finetuned_model_id:
                raise ValueError("Local provider requires model_path")
            model_path = finetuned_model_id if use_finetuned else model
            return client_class(model_path=model_path)
        
        # API-based clients
        # Use fine-tuned model if specified
        if use_finetuned and finetuned_model_id:
            return client_class(api_key=api_key, model=finetuned_model_id)
        
        # Use base model
        if model:
            return client_class(api_key=api_key, model=model)
        else:
            return client_class(api_key=api_key)
    
    @staticmethod
    def get_supported_providers() -> list:
        """Get list of supported providers."""
        return list(LLMFactory._clients.keys())
    
    @staticmethod
    def is_local_available() -> bool:
        """Check if local model support is available."""
        return LOCAL_CLIENT_AVAILABLE