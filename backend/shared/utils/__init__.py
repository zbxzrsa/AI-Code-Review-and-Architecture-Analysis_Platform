"""
Shared utilities for AI Code Review Platform.
"""
from .ai_client import (
    AIProvider,
    AIResponse,
    OpenAIProvider,
    AnthropicProvider,
    AIClientRouter,
    create_ai_client,
)

from .ai_provider_factory import (
    AIProviderFactory,
    AIProviderBase,
    OllamaProvider,
    HuggingFaceLocalProvider,
    ProviderType,
    ProviderTier,
    ProviderConfig,
    create_default_factory,
)

from .ollama_provider import (
    OllamaConfig,
    OllamaResponse,
    OllamaProvider as OllamaProviderDirect,
    CODE_REVIEW_PROMPTS,
    create_ollama_provider,
)

__all__ = [
    # Legacy AI client
    "AIProvider",
    "AIResponse", 
    "OpenAIProvider",
    "AnthropicProvider",
    "AIClientRouter",
    "create_ai_client",
    # New provider factory
    "AIProviderFactory",
    "AIProviderBase",
    "OllamaProvider",
    "HuggingFaceLocalProvider",
    "ProviderType",
    "ProviderTier",
    "ProviderConfig",
    "create_default_factory",
    # Ollama direct
    "OllamaConfig",
    "OllamaResponse",
    "OllamaProviderDirect",
    "CODE_REVIEW_PROMPTS",
    "create_ollama_provider",
]
