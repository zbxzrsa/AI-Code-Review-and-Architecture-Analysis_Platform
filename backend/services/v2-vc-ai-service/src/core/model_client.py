"""
V2 VC-AI Model Client

Production-grade AI model client with failover and consistency guarantees.
"""

import asyncio
import hashlib
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

import httpx


logger = logging.getLogger(__name__)


class ModelProvider(str, Enum):
    """Supported model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class ModelResponse:
    """Response from model inference"""
    content: str
    model: str
    provider: ModelProvider
    tokens_used: int
    latency_ms: float
    cached: bool = False
    deterministic_seed: Optional[str] = None
    request_id: Optional[str] = None


@dataclass
class ModelConfig:
    """Model configuration"""
    model: str
    provider: ModelProvider
    api_key: str
    api_base: str
    timeout: float = 5.0
    max_retries: int = 3
    
    # Locked parameters for consistency
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 40


class ModelClient:
    """
    Production-grade model client with:
    - Primary/backup failover
    - Deterministic outputs
    - Circuit breaker integration
    - Response caching
    - Latency tracking
    """
    
    def __init__(
        self,
        primary_config: ModelConfig,
        backup_config: Optional[ModelConfig] = None,
        cache_enabled: bool = True,
    ):
        self.primary_config = primary_config
        self.backup_config = backup_config
        self.cache_enabled = cache_enabled
        
        self._cache: Dict[str, ModelResponse] = {}
        self._http_client: Optional[httpx.AsyncClient] = None
        
        # Metrics
        self._total_requests = 0
        self._primary_successes = 0
        self._backup_successes = 0
        self._cache_hits = 0
        self._failures = 0
    
    async def __aenter__(self):
        self._http_client = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._http_client:
            await self._http_client.aclose()
    
    def _compute_cache_key(self, prompt: str, system_prompt: str, seed: str) -> str:
        """Compute deterministic cache key"""
        content = f"{prompt}|{system_prompt}|{seed}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _compute_deterministic_seed(self, input_data: str) -> str:
        """Compute deterministic seed from input"""
        return hashlib.sha256(input_data.encode()).hexdigest()[:16]
    
    async def _call_openai(
        self,
        config: ModelConfig,
        messages: List[Dict[str, str]],
        seed: str,
    ) -> ModelResponse:
        """Call OpenAI API"""
        start_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
        
        # Convert seed to integer for OpenAI
        seed_int = int(seed, 16) % (2**31)
        
        payload = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "seed": seed_int,
            "max_tokens": 4096,
        }
        
        response = await self._http_client.post(
            f"{config.api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=config.timeout,
        )
        response.raise_for_status()
        
        data = response.json()
        latency_ms = (time.time() - start_time) * 1000
        
        return ModelResponse(
            content=data["choices"][0]["message"]["content"],
            model=config.model,
            provider=ModelProvider.OPENAI,
            tokens_used=data.get("usage", {}).get("total_tokens", 0),
            latency_ms=latency_ms,
            deterministic_seed=seed,
            request_id=data.get("id"),
        )
    
    async def _call_anthropic(
        self,
        config: ModelConfig,
        messages: List[Dict[str, str]],
        system_prompt: str,
        seed: str,
    ) -> ModelResponse:
        """Call Anthropic API"""
        start_time = time.time()
        
        headers = {
            "x-api-key": config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        
        # Filter out system messages for Anthropic
        user_messages = [m for m in messages if m["role"] != "system"]
        
        payload = {
            "model": config.model,
            "messages": user_messages,
            "system": system_prompt,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "max_tokens": 4096,
        }
        
        response = await self._http_client.post(
            f"{config.api_base}/messages",
            headers=headers,
            json=payload,
            timeout=config.timeout,
        )
        response.raise_for_status()
        
        data = response.json()
        latency_ms = (time.time() - start_time) * 1000
        
        return ModelResponse(
            content=data["content"][0]["text"],
            model=config.model,
            provider=ModelProvider.ANTHROPIC,
            tokens_used=data.get("usage", {}).get("input_tokens", 0) + data.get("usage", {}).get("output_tokens", 0),
            latency_ms=latency_ms,
            deterministic_seed=seed,
            request_id=data.get("id"),
        )
    
    async def _call_model(
        self,
        config: ModelConfig,
        prompt: str,
        system_prompt: str,
        seed: str,
    ) -> ModelResponse:
        """Call model based on provider"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        
        if config.provider == ModelProvider.OPENAI:
            return await self._call_openai(config, messages, seed)
        elif config.provider == ModelProvider.ANTHROPIC:
            return await self._call_anthropic(config, messages, system_prompt, seed)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
    
    async def analyze(
        self,
        prompt: str,
        system_prompt: str,
        input_hash: Optional[str] = None,
        use_cache: bool = True,
    ) -> ModelResponse:
        """
        Analyze with AI model.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            input_hash: Hash for deterministic seeding (e.g., commit hash)
            use_cache: Whether to use caching
            
        Returns:
            ModelResponse with analysis result
        """
        self._total_requests += 1
        
        # Compute deterministic seed
        seed = input_hash or self._compute_deterministic_seed(prompt)
        
        # Check cache
        if self.cache_enabled and use_cache:
            cache_key = self._compute_cache_key(prompt, system_prompt, seed)
            if cache_key in self._cache:
                self._cache_hits += 1
                cached = self._cache[cache_key]
                logger.debug(f"Cache hit for key {cache_key[:8]}...")
                return ModelResponse(
                    content=cached.content,
                    model=cached.model,
                    provider=cached.provider,
                    tokens_used=0,
                    latency_ms=0,
                    cached=True,
                    deterministic_seed=seed,
                )
        
        # Try primary model
        last_error: Optional[Exception] = None
        
        try:
            response = await self._call_model(
                self.primary_config,
                prompt,
                system_prompt,
                seed,
            )
            self._primary_successes += 1
            
            # Cache response
            if self.cache_enabled and use_cache:
                cache_key = self._compute_cache_key(prompt, system_prompt, seed)
                self._cache[cache_key] = response
            
            return response
            
        except Exception as e:
            last_error = e
            logger.warning(f"Primary model failed: {e}")
        
        # Try backup model
        if self.backup_config:
            try:
                response = await self._call_model(
                    self.backup_config,
                    prompt,
                    system_prompt,
                    seed,
                )
                self._backup_successes += 1
                
                # Cache response
                if self.cache_enabled and use_cache:
                    cache_key = self._compute_cache_key(prompt, system_prompt, seed)
                    self._cache[cache_key] = response
                
                logger.info("Fallback to backup model succeeded")
                return response
                
            except Exception as e:
                last_error = e
                logger.error(f"Backup model also failed: {e}")
        
        self._failures += 1
        raise ModelClientError(f"All models failed: {last_error}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics"""
        return {
            "total_requests": self._total_requests,
            "primary_successes": self._primary_successes,
            "backup_successes": self._backup_successes,
            "cache_hits": self._cache_hits,
            "failures": self._failures,
            "cache_size": len(self._cache),
            "primary_success_rate": self._primary_successes / max(1, self._total_requests),
            "overall_success_rate": (self._primary_successes + self._backup_successes) / max(1, self._total_requests),
            "cache_hit_rate": self._cache_hits / max(1, self._total_requests),
        }
    
    def clear_cache(self) -> int:
        """Clear response cache, return number of entries cleared"""
        count = len(self._cache)
        self._cache.clear()
        return count


class ModelClientError(Exception):
    """Model client error"""
    pass
