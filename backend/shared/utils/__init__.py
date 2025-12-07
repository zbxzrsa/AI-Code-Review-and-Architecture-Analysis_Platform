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

# Common utilities (consolidated from multiple modules)
from .common import (
    # String utilities
    generate_id,
    hash_string,
    truncate,
    slugify,
    camel_to_snake,
    snake_to_camel,
    # DateTime utilities
    utc_now,
    iso_format,
    parse_iso,
    # Dictionary utilities
    deep_merge,
    flatten_dict,
    get_nested,
    set_nested,
    # List utilities
    chunk_list,
    unique,
    first,
    last,
    # Async utilities
    run_with_timeout,
    gather_with_concurrency,
    # Decorators
    retry,
    log_execution,
    # Validation
    is_valid_email,
    is_valid_uuid,
    is_valid_url,
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
    # Common utilities
    "generate_id",
    "hash_string",
    "truncate",
    "slugify",
    "camel_to_snake",
    "snake_to_camel",
    "utc_now",
    "iso_format",
    "parse_iso",
    "deep_merge",
    "flatten_dict",
    "get_nested",
    "set_nested",
    "chunk_list",
    "unique",
    "first",
    "last",
    "run_with_timeout",
    "gather_with_concurrency",
    "retry",
    "log_execution",
    "is_valid_email",
    "is_valid_uuid",
    "is_valid_url",
]
