#!/bin/bash
# Script to create all model files

echo "Creating all model files..."

# Create models directory
mkdir -p models

# Create __init__.py
cat > models/__init__.py << 'EOF'
"""Models package for LLM clients."""
EOF

# Create llm_client.py
cat > models/llm_client.py << 'EOF'
"""Abstract base class for LLM clients."""
from abc import ABC, abstractmethod

class LLMClient(ABC):
    """Abstract base class for all LLM clients."""
    
    def __init__(self, model: str):
        self.model = model
    
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 200) -> str:
        """Generate text from prompt."""
        pass
EOF

# Create openai_client.py
cat > models/openai_client.py << 'EOF'
"""OpenAI GPT client implementation."""
from models.llm_client import LLMClient
from openai import OpenAI

class OpenAIClient(LLMClient):
    """Client for OpenAI GPT models."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        super().__init__(model)
        self.client = OpenAI(api_key=api_key)
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 200) -> str:
        """Generate text using OpenAI."""
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
EOF

# Create claude_client.py
cat > models/claude_client.py << 'EOF'
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
EOF

# Create gemini_client.py
cat > models/gemini_client.py << 'EOF'
"""Google Gemini client implementation."""
from models.llm_client import LLMClient
import google.generativeai as genai

class GeminiClient(LLMClient):
    """Client for Google Gemini models."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        super().__init__(model)
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 200) -> str:
        """Generate text using Gemini."""
        try:
            response = self.client.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
                }
            )
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}")
EOF

# Create llm_factory.py
cat > models/llm_factory.py << 'EOF'
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
EOF

echo "✅ All model files created!"
echo ""
echo "Files created:"
echo "  - models/__init__.py"
echo "  - models/llm_client.py"
echo "  - models/openai_client.py"
echo "  - models/claude_client.py"
echo "  - models/gemini_client.py"
echo "  - models/llm_factory.py"