"""
Unified AI client for interacting with multiple AI providers.
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class AIResponse:
    """Standardized response from AI models."""
    content: str
    model: str
    tokens_used: int
    latency_ms: float
    cost: float
    confidence: float = 0.0


class AIProvider(ABC):
    """Abstract base class for AI providers."""

    @abstractmethod
    async def analyze_code(
        self,
        code: str,
        language: str,
        prompt_template: str,
    ) -> AIResponse:
        """Analyze code and return insights."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is available."""
        pass


class OpenAIProvider(AIProvider):
    """OpenAI provider implementation."""

    def __init__(self, api_key: str, model: str = "gpt-4", temperature: float = 0.7):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self._client = None

    async def analyze_code(
        self,
        code: str,
        language: str,
        prompt_template: str,
    ) -> AIResponse:
        """Analyze code using OpenAI."""
        try:
            import openai
            openai.api_key = self.api_key

            start_time = time.time()

            prompt = prompt_template.format(
                code=code,
                language=language,
            )

            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert code reviewer and software architect.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=2000,
            )

            latency_ms = (time.time() - start_time) * 1000
            content = response.choices[0].message.content

            # Estimate cost (rough approximation)
            tokens_used = response.usage.total_tokens
            cost = (tokens_used / 1000) * 0.03  # Approximate cost per 1K tokens

            return AIResponse(
                content=content,
                model=self.model,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                cost=cost,
                confidence=0.95,
            )

        except Exception as e:
            logger.error(f"OpenAI analysis failed: {str(e)}")
            raise

    async def health_check(self) -> bool:
        """Check OpenAI API availability."""
        try:
            import openai
            openai.api_key = self.api_key
            await asyncio.to_thread(openai.Model.list)
            return True
        except Exception as e:
            logger.error(f"OpenAI health check failed: {str(e)}")
            return False


class AnthropicProvider(AIProvider):
    """Anthropic Claude provider implementation."""

    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229", temperature: float = 0.7):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    async def analyze_code(
        self,
        code: str,
        language: str,
        prompt_template: str,
    ) -> AIResponse:
        """Analyze code using Anthropic Claude."""
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)

            start_time = time.time()

            prompt = prompt_template.format(
                code=code,
                language=language,
            )

            response = await asyncio.to_thread(
                client.messages.create,
                model=self.model,
                max_tokens=2000,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            latency_ms = (time.time() - start_time) * 1000
            content = response.content[0].text

            # Estimate tokens and cost
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            cost = (tokens_used / 1000) * 0.015  # Approximate cost

            return AIResponse(
                content=content,
                model=self.model,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                cost=cost,
                confidence=0.93,
            )

        except Exception as e:
            logger.error(f"Anthropic analysis failed: {str(e)}")
            raise

    async def health_check(self) -> bool:
        """Check Anthropic API availability."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            await asyncio.to_thread(
                client.messages.create,
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except Exception as e:
            logger.error(f"Anthropic health check failed: {str(e)}")
            return False


class AIClientRouter:
    """Routes requests to appropriate AI providers."""

    def __init__(
        self,
        primary_provider: AIProvider,
        secondary_provider: Optional[AIProvider] = None,
    ):
        self.primary = primary_provider
        self.secondary = secondary_provider

    async def analyze_code(
        self,
        code: str,
        language: str,
        prompt_template: str,
        strategy: str = "primary",
    ) -> AIResponse:
        """
        Analyze code using specified routing strategy.

        Strategies:
        - "primary": Use primary provider only
        - "secondary": Use secondary provider only
        - "ensemble": Get responses from both and combine
        - "adaptive": Try primary, fallback to secondary
        """
        if strategy == "primary":
            return await self.primary.analyze_code(code, language, prompt_template)

        elif strategy == "secondary":
            if not self.secondary:
                raise ValueError("Secondary provider not configured")
            return await self.secondary.analyze_code(code, language, prompt_template)

        elif strategy == "ensemble":
            primary_response = await self.primary.analyze_code(code, language, prompt_template)
            if self.secondary:
                secondary_response = await self.secondary.analyze_code(code, language, prompt_template)
                # Combine responses (simple concatenation for now)
                combined_content = f"Primary:\n{primary_response.content}\n\nSecondary:\n{secondary_response.content}"
                return AIResponse(
                    content=combined_content,
                    model=f"{primary_response.model}+{secondary_response.model}",
                    tokens_used=primary_response.tokens_used + secondary_response.tokens_used,
                    latency_ms=max(primary_response.latency_ms, secondary_response.latency_ms),
                    cost=primary_response.cost + secondary_response.cost,
                    confidence=min(primary_response.confidence, secondary_response.confidence),
                )
            return primary_response

        elif strategy == "adaptive":
            try:
                return await self.primary.analyze_code(code, language, prompt_template)
            except Exception as e:
                logger.warning(f"Primary provider failed, falling back to secondary: {str(e)}")
                if self.secondary:
                    return await self.secondary.analyze_code(code, language, prompt_template)
                raise

        else:
            raise ValueError(f"Unknown routing strategy: {strategy}")

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all providers."""
        results = {
            "primary": await self.primary.health_check(),
        }
        if self.secondary:
            results["secondary"] = await self.secondary.health_check()
        return results


def create_ai_client(
    primary_provider: str,
    primary_api_key: str,
    primary_model: str,
    secondary_provider: Optional[str] = None,
    secondary_api_key: Optional[str] = None,
    secondary_model: Optional[str] = None,
) -> AIClientRouter:
    """Factory function to create AI client with configured providers."""

    primary = _create_provider(primary_provider, primary_api_key, primary_model)

    secondary = None
    if secondary_provider and secondary_api_key:
        secondary = _create_provider(secondary_provider, secondary_api_key, secondary_model or "")

    return AIClientRouter(primary, secondary)


def _create_provider(provider_name: str, api_key: str, model: str) -> AIProvider:
    """Create provider instance."""
    if provider_name.lower() == "openai":
        return OpenAIProvider(api_key, model)
    elif provider_name.lower() == "anthropic":
        return AnthropicProvider(api_key, model)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
