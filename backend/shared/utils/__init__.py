"""
AI 代码审查平台共享工具模块 (Shared utilities for AI Code Review Platform)

模块功能描述:
    提供 AI 提供者客户端和工具函数。

主要功能:
    - AI 提供者抽象和路由
    - 多提供者支持（OpenAI、Anthropic、Ollama、HuggingFace）
    - 提供者工厂模式
    - 本地模型集成

主要组件:
    - AIClientRouter: AI 客户端路由器
    - AIProviderFactory: AI 提供者工厂
    - OllamaProvider: Ollama 本地模型提供者
    - HuggingFaceLocalProvider: HuggingFace 本地模型提供者

最后修改日期: 2024-12-07
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
