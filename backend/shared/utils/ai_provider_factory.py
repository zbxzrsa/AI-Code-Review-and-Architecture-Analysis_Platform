"""
AI Provider Factory - Unified interface for multiple AI providers.

Supports:
- Ollama (Local, Open Source) - Primary
- vLLM (Local, Open Source) - Alternative
- HuggingFace (Local/Cloud) - Fallback
- OpenAI (Cloud, Paid) - Optional user-provided
- Anthropic (Cloud, Paid) - Optional user-provided

Priority Order: Local/Free -> User-provided Cloud -> Platform Cloud
"""
import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """AI provider types."""
    OLLAMA = "ollama"
    VLLM = "vllm"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ProviderTier(Enum):
    """Provider pricing tier."""
    FREE = "free"           # Open source, self-hosted
    FREEMIUM = "freemium"   # Free tier available
    PAID = "paid"           # Paid only


@dataclass
class AIResponse:
    """Standardized AI response."""
    content: str
    model: str
    provider: str
    tokens_used: int
    latency_ms: float
    cost: float = 0.0  # 0 for open source
    confidence: float = 0.85
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderConfig:
    """Provider configuration."""
    type: ProviderType
    tier: ProviderTier
    endpoint: str
    model: str
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: float = 120.0
    priority: int = 0  # Lower = higher priority
    enabled: bool = True
    

class AIProviderBase(ABC):
    """Base class for AI providers."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self._healthy = True
        self._last_health_check = 0
        self._consecutive_failures = 0
        
    @property
    def name(self) -> str:
        return f"{self.config.type.value}:{self.config.model}"
    
    @property
    def is_free(self) -> bool:
        return self.config.tier == ProviderTier.FREE
    
    @abstractmethod
    async def analyze(
        self,
        code: str,
        language: str,
        prompt_template: str,
    ) -> AIResponse:
        """Analyze code and return response."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check provider health."""
        pass
    
    async def mark_healthy(self):
        """Mark provider as healthy."""
        self._healthy = True
        self._consecutive_failures = 0
        
    async def mark_unhealthy(self):
        """Mark provider as unhealthy after failure."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= 3:
            self._healthy = False
            logger.warning(f"Provider {self.name} marked unhealthy after {self._consecutive_failures} failures")


class OllamaProvider(AIProviderBase):
    """Ollama provider - Open source local inference."""
    
    async def analyze(
        self,
        code: str,
        language: str,
        prompt_template: str,
    ) -> AIResponse:
        """Analyze code using Ollama."""
        import httpx
        
        prompt = prompt_template.format(code=code, language=language)
        
        system_prompt = """You are an expert code reviewer. Provide detailed, actionable feedback with specific line references."""
        
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                f"{self.config.endpoint}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            
        latency_ms = (time.time() - start_time) * 1000
        tokens = data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
        
        await self.mark_healthy()
        
        return AIResponse(
            content=data.get("response", ""),
            model=self.config.model,
            provider=self.config.type.value,
            tokens_used=tokens,
            latency_ms=latency_ms,
            cost=0.0,  # Free
            confidence=0.85,
        )
    
    async def health_check(self) -> bool:
        """Check Ollama health."""
        import httpx
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.config.endpoint}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama health check failed: {e}")
            return False


class HuggingFaceLocalProvider(AIProviderBase):
    """HuggingFace Transformers local inference."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._pipeline = None
        
    async def _get_pipeline(self):
        """Lazy load pipeline."""
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-generation",
                model=self.config.model,
                device_map="auto",
            )
        return self._pipeline
    
    async def analyze(
        self,
        code: str,
        language: str,
        prompt_template: str,
    ) -> AIResponse:
        """Analyze code using HuggingFace."""
        prompt = prompt_template.format(code=code, language=language)
        
        start_time = time.time()
        
        pipe = await self._get_pipeline()
        result = await asyncio.to_thread(
            pipe,
            prompt,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            do_sample=True,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        content = result[0]["generated_text"]
        
        await self.mark_healthy()
        
        return AIResponse(
            content=content,
            model=self.config.model,
            provider=self.config.type.value,
            tokens_used=len(content.split()),  # Approximate
            latency_ms=latency_ms,
            cost=0.0,
            confidence=0.80,
        )
    
    async def health_check(self) -> bool:
        """Check HuggingFace availability."""
        try:
            await self._get_pipeline()
            return True
        except Exception as e:
            logger.debug(f"HuggingFace health check failed: {e}")
            return False


class OpenAIProvider(AIProviderBase):
    """OpenAI provider - Cloud paid service."""
    
    # Cost per 1K tokens (approximate)
    COST_PER_1K = {
        "gpt-4": 0.03,
        "gpt-4-turbo": 0.01,
        "gpt-3.5-turbo": 0.002,
    }
    
    async def analyze(
        self,
        code: str,
        language: str,
        prompt_template: str,
    ) -> AIResponse:
        """Analyze code using OpenAI."""
        import openai
        
        openai.api_key = self.config.api_key
        
        prompt = prompt_template.format(code=code, language=language)
        
        start_time = time.time()
        
        response = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model=self.config.model,
            messages=[
                {"role": "system", "content": "You are an expert code reviewer."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens
        
        # Calculate cost
        cost_rate = self.COST_PER_1K.get(self.config.model, 0.03)
        cost = (tokens / 1000) * cost_rate
        
        await self.mark_healthy()
        
        return AIResponse(
            content=content,
            model=self.config.model,
            provider=self.config.type.value,
            tokens_used=tokens,
            latency_ms=latency_ms,
            cost=cost,
            confidence=0.95,
        )
    
    async def health_check(self) -> bool:
        """Check OpenAI availability."""
        if not self.config.api_key:
            return False
        try:
            import openai
            openai.api_key = self.config.api_key
            await asyncio.to_thread(openai.Model.list)
            return True
        except Exception as e:
            logger.debug(f"OpenAI health check failed: {e}")
            return False


class AnthropicProvider(AIProviderBase):
    """Anthropic Claude provider - Cloud paid service."""
    
    COST_PER_1K = {
        "claude-3-opus": 0.015,
        "claude-3-sonnet": 0.003,
        "claude-3-haiku": 0.00025,
    }
    
    async def analyze(
        self,
        code: str,
        language: str,
        prompt_template: str,
    ) -> AIResponse:
        """Analyze code using Anthropic."""
        import anthropic
        
        client = anthropic.Anthropic(api_key=self.config.api_key)
        
        prompt = prompt_template.format(code=code, language=language)
        
        start_time = time.time()
        
        response = await asyncio.to_thread(
            client.messages.create,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        
        latency_ms = (time.time() - start_time) * 1000
        content = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        
        # Calculate cost
        cost_rate = self.COST_PER_1K.get(self.config.model.split("-")[0] + "-" + self.config.model.split("-")[1], 0.015)
        cost = (tokens / 1000) * cost_rate
        
        await self.mark_healthy()
        
        return AIResponse(
            content=content,
            model=self.config.model,
            provider=self.config.type.value,
            tokens_used=tokens,
            latency_ms=latency_ms,
            cost=cost,
            confidence=0.93,
        )
    
    async def health_check(self) -> bool:
        """Check Anthropic availability."""
        if not self.config.api_key:
            return False
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.config.api_key)
            await asyncio.to_thread(
                client.messages.create,
                model=self.config.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except Exception as e:
            logger.debug(f"Anthropic health check failed: {e}")
            return False


class AIProviderFactory:
    """
    Factory for creating and managing AI providers.
    
    Priority Order:
    1. Ollama (Free, Local)
    2. HuggingFace Local (Free, Local)
    3. User-provided Cloud Keys
    4. Platform Cloud Keys (if configured)
    """
    
    def __init__(self):
        self._providers: Dict[str, AIProviderBase] = {}
        self._priority_chain: List[AIProviderBase] = []
        
    def register_provider(self, provider: AIProviderBase):
        """Register a provider."""
        self._providers[provider.name] = provider
        self._rebuild_priority_chain()
        
    def _rebuild_priority_chain(self):
        """Rebuild priority chain based on tier and priority."""
        # Sort by: tier (FREE first), then priority
        def sort_key(p: AIProviderBase):
            tier_order = {ProviderTier.FREE: 0, ProviderTier.FREEMIUM: 1, ProviderTier.PAID: 2}
            return (tier_order[p.config.tier], p.config.priority)
        
        enabled = [p for p in self._providers.values() if p.config.enabled]
        self._priority_chain = sorted(enabled, key=sort_key)
        
    async def get_healthy_provider(self) -> Optional[AIProviderBase]:
        """Get first healthy provider from priority chain."""
        for provider in self._priority_chain:
            if provider._healthy and await provider.health_check():
                return provider
        return None
    
    async def analyze(
        self,
        code: str,
        language: str,
        prompt_template: str,
        preferred_provider: Optional[str] = None,
    ) -> AIResponse:
        """
        Analyze code using best available provider.
        
        Args:
            code: Source code
            language: Programming language
            prompt_template: Prompt template
            preferred_provider: Optional provider preference
            
        Returns:
            AIResponse from successful provider
        """
        # Try preferred provider first
        if preferred_provider and preferred_provider in self._providers:
            provider = self._providers[preferred_provider]
            if await provider.health_check():
                try:
                    return await provider.analyze(code, language, prompt_template)
                except Exception as e:
                    logger.warning(f"Preferred provider {preferred_provider} failed: {e}")
                    await provider.mark_unhealthy()
        
        # Fallback to priority chain
        for provider in self._priority_chain:
            if not provider._healthy:
                continue
                
            try:
                return await provider.analyze(code, language, prompt_template)
            except Exception as e:
                logger.warning(f"Provider {provider.name} failed: {e}")
                await provider.mark_unhealthy()
                continue
        
        raise RuntimeError("No healthy AI providers available")
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all providers."""
        results = {}
        for name, provider in self._providers.items():
            results[name] = await provider.health_check()
        return results


def create_default_factory(
    ollama_endpoint: str = "http://localhost:11434",
    ollama_model: str = "codellama:34b",
    openai_key: Optional[str] = None,
    anthropic_key: Optional[str] = None,
) -> AIProviderFactory:
    """
    Create factory with default configuration.
    
    Priority:
    1. Ollama (always enabled, free)
    2. OpenAI (if key provided)
    3. Anthropic (if key provided)
    """
    factory = AIProviderFactory()
    
    # Always register Ollama (free, local)
    ollama_config = ProviderConfig(
        type=ProviderType.OLLAMA,
        tier=ProviderTier.FREE,
        endpoint=ollama_endpoint,
        model=ollama_model,
        priority=0,
    )
    factory.register_provider(OllamaProvider(ollama_config))
    
    # Register OpenAI if key provided
    if openai_key:
        openai_config = ProviderConfig(
            type=ProviderType.OPENAI,
            tier=ProviderTier.PAID,
            endpoint="https://api.openai.com",
            model="gpt-4",
            api_key=openai_key,
            priority=10,
        )
        factory.register_provider(OpenAIProvider(openai_config))
    
    # Register Anthropic if key provided
    if anthropic_key:
        anthropic_config = ProviderConfig(
            type=ProviderType.ANTHROPIC,
            tier=ProviderTier.PAID,
            endpoint="https://api.anthropic.com",
            model="claude-3-opus-20240229",
            api_key=anthropic_key,
            priority=11,
        )
        factory.register_provider(AnthropicProvider(anthropic_config))
    
    return factory
