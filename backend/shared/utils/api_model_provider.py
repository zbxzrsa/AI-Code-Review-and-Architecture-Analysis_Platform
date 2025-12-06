"""
API Model Provider - External model integration via API.

This provider allows importing external AI models through their APIs,
including Ollama, OpenAI, Anthropic, HuggingFace, and custom endpoints.

Imported models are restricted to Code Review AI Chat functionality only.
"""
import asyncio
import json
import logging
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
import httpx

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class ModelScope(Enum):
    """Scope of model usage."""
    CODE_REVIEW_CHAT = "code_review_chat"
    ALL = "all"


@dataclass
class ImportedModelConfig:
    """Configuration for an imported API model."""
    id: str
    name: str
    provider: ModelProvider
    api_endpoint: str
    api_key: Optional[str] = None
    model_id: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: float = 60.0
    scope: ModelScope = ModelScope.CODE_REVIEW_CHAT
    headers: Dict[str, str] = field(default_factory=dict)
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    """Response from an API model."""
    content: str
    model: str
    tokens_used: int
    latency_ms: float
    provider: str
    finish_reason: str = "stop"


class APIModelProvider:
    """
    API-based model provider for external model integration.
    
    Supports importing models from:
    - Ollama (local models via API)
    - OpenAI (GPT models)
    - Anthropic (Claude models)
    - HuggingFace (open models)
    - Custom API endpoints
    
    Note: Imported models are restricted to Code Review AI Chat only.
    """
    
    def __init__(self):
        self._imported_models: Dict[str, ImportedModelConfig] = {}
        self._client = httpx.AsyncClient(timeout=120.0)
    
    async def import_model(
        self,
        model_id: str,
        name: str,
        provider: str,
        api_endpoint: str,
        api_key: Optional[str] = None,
        model_name: str = "",
        **kwargs
    ) -> ImportedModelConfig:
        """
        Import an external model via API.
        
        Args:
            model_id: Unique identifier for the imported model
            name: Display name for the model
            provider: One of 'openai', 'anthropic', 'ollama', 'huggingface', 'custom'
            api_endpoint: API endpoint URL
            api_key: Optional API key for authentication
            model_name: The model identifier used by the API
            **kwargs: Additional configuration options
        
        Returns:
            ImportedModelConfig for the imported model
        """
        provider_enum = ModelProvider(provider.lower())
        
        config = ImportedModelConfig(
            id=model_id,
            name=name,
            provider=provider_enum,
            api_endpoint=api_endpoint,
            api_key=api_key,
            model_id=model_name or model_id,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 4096),
            timeout=kwargs.get("timeout", 60.0),
            scope=ModelScope.CODE_REVIEW_CHAT,
            headers=kwargs.get("headers", {}),
            extra_params=kwargs.get("extra_params", {}),
        )
        
        self._imported_models[model_id] = config
        logger.info(f"Imported model '{name}' ({model_id}) from {provider}")
        
        return config
    
    async def remove_model(self, model_id: str) -> bool:
        """Remove an imported model."""
        if model_id in self._imported_models:
            del self._imported_models[model_id]
            logger.info(f"Removed imported model: {model_id}")
            return True
        return False
    
    def get_model(self, model_id: str) -> Optional[ImportedModelConfig]:
        """Get an imported model configuration."""
        return self._imported_models.get(model_id)
    
    def list_models(self) -> List[ImportedModelConfig]:
        """List all imported models."""
        return list(self._imported_models.values())
    
    async def health_check(self, model_id: str) -> Dict[str, Any]:
        """Check if an imported model is available."""
        config = self._imported_models.get(model_id)
        if not config:
            return {"healthy": False, "error": "Model not found"}
        
        try:
            start = time.time()
            
            if config.provider == ModelProvider.OLLAMA:
                response = await self._client.get(
                    f"{config.api_endpoint.rstrip('/api/chat')}/api/tags",
                    timeout=10.0
                )
            else:
                # For other providers, try a minimal request
                response = await self._client.get(
                    config.api_endpoint,
                    headers=self._build_headers(config),
                    timeout=10.0
                )
            
            latency = (time.time() - start) * 1000
            
            return {
                "healthy": response.status_code < 400,
                "latency_ms": latency,
                "status_code": response.status_code,
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def chat(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> ModelResponse:
        """
        Send a chat request to an imported model.
        
        Args:
            model_id: ID of the imported model
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters for the request
        
        Returns:
            ModelResponse with the generated content
        """
        config = self._imported_models.get(model_id)
        if not config:
            raise ValueError(f"Model '{model_id}' not found in imported models")
        
        if config.scope != ModelScope.CODE_REVIEW_CHAT:
            raise PermissionError(
                f"Model '{model_id}' is not available for this operation"
            )
        
        start = time.time()
        
        try:
            if config.provider == ModelProvider.OLLAMA:
                response = await self._chat_ollama(config, messages, **kwargs)
            elif config.provider == ModelProvider.OPENAI:
                response = await self._chat_openai(config, messages, **kwargs)
            elif config.provider == ModelProvider.ANTHROPIC:
                response = await self._chat_anthropic(config, messages, **kwargs)
            else:
                response = await self._chat_custom(config, messages, **kwargs)
            
            latency_ms = (time.time() - start) * 1000
            response.latency_ms = latency_ms
            
            return response
            
        except Exception as e:
            logger.error(f"Chat error with model {model_id}: {e}")
            raise
    
    async def _chat_ollama(
        self,
        config: ImportedModelConfig,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> ModelResponse:
        """Chat with Ollama API."""
        payload = {
            "model": config.model_id,
            "messages": messages,
            "options": {
                "temperature": kwargs.get("temperature", config.temperature),
                "num_predict": kwargs.get("max_tokens", config.max_tokens),
            },
            "stream": False,
        }
        
        response = await self._client.post(
            config.api_endpoint,
            json=payload,
            headers=self._build_headers(config),
            timeout=config.timeout,
        )
        response.raise_for_status()
        
        data = response.json()
        
        return ModelResponse(
            content=data.get("message", {}).get("content", ""),
            model=config.model_id,
            tokens_used=data.get("eval_count", 0),
            latency_ms=0,
            provider="ollama",
            finish_reason=data.get("done_reason", "stop"),
        )
    
    async def _chat_openai(
        self,
        config: ImportedModelConfig,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> ModelResponse:
        """Chat with OpenAI-compatible API."""
        payload = {
            "model": config.model_id,
            "messages": messages,
            "temperature": kwargs.get("temperature", config.temperature),
            "max_tokens": kwargs.get("max_tokens", config.max_tokens),
        }
        
        response = await self._client.post(
            config.api_endpoint,
            json=payload,
            headers=self._build_headers(config),
            timeout=config.timeout,
        )
        response.raise_for_status()
        
        data = response.json()
        choice = data.get("choices", [{}])[0]
        
        return ModelResponse(
            content=choice.get("message", {}).get("content", ""),
            model=config.model_id,
            tokens_used=data.get("usage", {}).get("total_tokens", 0),
            latency_ms=0,
            provider="openai",
            finish_reason=choice.get("finish_reason", "stop"),
        )
    
    async def _chat_anthropic(
        self,
        config: ImportedModelConfig,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> ModelResponse:
        """Chat with Anthropic API."""
        # Convert messages to Anthropic format
        system_msg = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                chat_messages.append(msg)
        
        payload = {
            "model": config.model_id,
            "messages": chat_messages,
            "max_tokens": kwargs.get("max_tokens", config.max_tokens),
            "temperature": kwargs.get("temperature", config.temperature),
        }
        if system_msg:
            payload["system"] = system_msg
        
        response = await self._client.post(
            config.api_endpoint,
            json=payload,
            headers=self._build_headers(config),
            timeout=config.timeout,
        )
        response.raise_for_status()
        
        data = response.json()
        content_blocks = data.get("content", [])
        content = "".join(
            block.get("text", "") for block in content_blocks
            if block.get("type") == "text"
        )
        
        return ModelResponse(
            content=content,
            model=config.model_id,
            tokens_used=data.get("usage", {}).get("input_tokens", 0) + 
                       data.get("usage", {}).get("output_tokens", 0),
            latency_ms=0,
            provider="anthropic",
            finish_reason=data.get("stop_reason", "stop"),
        )
    
    async def _chat_custom(
        self,
        config: ImportedModelConfig,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> ModelResponse:
        """Chat with a custom API endpoint."""
        payload = {
            "model": config.model_id,
            "messages": messages,
            "temperature": kwargs.get("temperature", config.temperature),
            "max_tokens": kwargs.get("max_tokens", config.max_tokens),
            **config.extra_params,
        }
        
        response = await self._client.post(
            config.api_endpoint,
            json=payload,
            headers=self._build_headers(config),
            timeout=config.timeout,
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Try to extract content from various response formats
        content = ""
        if "choices" in data:
            content = data["choices"][0].get("message", {}).get("content", "")
        elif "content" in data:
            content = data["content"]
        elif "response" in data:
            content = data["response"]
        elif "message" in data:
            msg = data["message"]
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        
        return ModelResponse(
            content=content,
            model=config.model_id,
            tokens_used=data.get("usage", {}).get("total_tokens", 0),
            latency_ms=0,
            provider="custom",
            finish_reason="stop",
        )
    
    def _build_headers(self, config: ImportedModelConfig) -> Dict[str, str]:
        """Build request headers for the API call."""
        headers = {
            "Content-Type": "application/json",
            **config.headers,
        }
        
        if config.api_key:
            if config.provider == ModelProvider.OPENAI:
                headers["Authorization"] = f"Bearer {config.api_key}"
            elif config.provider == ModelProvider.ANTHROPIC:
                headers["x-api-key"] = config.api_key
                headers["anthropic-version"] = "2024-01-01"
            elif config.provider == ModelProvider.HUGGINGFACE:
                headers["Authorization"] = f"Bearer {config.api_key}"
            else:
                headers["Authorization"] = f"Bearer {config.api_key}"
        
        return headers
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()


# Global instance
_api_model_provider: Optional[APIModelProvider] = None


def get_api_model_provider() -> APIModelProvider:
    """Get the global API model provider instance."""
    global _api_model_provider
    if _api_model_provider is None:
        _api_model_provider = APIModelProvider()
    return _api_model_provider


async def import_ollama_model(
    name: str,
    model_name: str = "codellama:34b",
    base_url: str = "http://localhost:11434",
) -> ImportedModelConfig:
    """
    Helper function to import an Ollama model.
    
    Args:
        name: Display name for the model
        model_name: Ollama model name (e.g., 'codellama:34b')
        base_url: Ollama server URL
    
    Returns:
        ImportedModelConfig for the imported model
    """
    provider = get_api_model_provider()
    
    return await provider.import_model(
        model_id=f"ollama_{model_name.replace(':', '_')}",
        name=name,
        provider="ollama",
        api_endpoint=f"{base_url}/api/chat",
        model_name=model_name,
    )
