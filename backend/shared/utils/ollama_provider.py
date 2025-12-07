"""
Ollama 提供者 - 开源本地 LLM 推理 (Ollama Provider - Open-source local LLM inference)

模块功能描述:
    与 Ollama 集成实现本地 LLM 推理。

支持的模型:
    - CodeLlama (7B/13B/34B)
    - DeepSeek Coder (6.7B/33B)
    - Llama 3 (8B/70B)
    - Mistral (7B)
    - Mixtral (8x7B)

主要组件:
    - OllamaProvider: Ollama 提供者主类
    - OllamaConfig: 配置数据类
    - OllamaResponse: 响应数据类

GitHub: https://github.com/ollama/ollama (95k+ stars)
License: MIT

最后修改日期: 2024-12-07
"""
import asyncio
import json
import logging
import time
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import httpx

logger = logging.getLogger(__name__)

# Model name constants
MODEL_CODELLAMA_34B = "codellama:34b"
MODEL_CODELLAMA_13B = "codellama:13b"
MODEL_CODELLAMA_7B = "codellama:7b"
MODEL_DEEPSEEK_33B = "deepseek-coder:33b"
MODEL_DEEPSEEK_6B = "deepseek-coder:6.7b"
MODEL_LLAMA3_70B = "llama3:70b"
MODEL_LLAMA3_8B = "llama3:8b"
MODEL_MISTRAL_7B = "mistral:7b"
MODEL_MIXTRAL_8X7B = "mixtral:8x7b"


@dataclass
class OllamaConfig:
    """
    Ollama 提供者配置数据类
    
    功能描述:
        Ollama 服务的配置参数。
    
    属性说明:
        - base_url: Ollama 服务器 URL
        - model: 模型名称
        - temperature: 温度参数
        - max_tokens: 最大令牌数
        - timeout: 超时时间（秒）
    """
    base_url: str = "http://localhost:11434"
    model: str = MODEL_CODELLAMA_34B
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: float = 120.0  # seconds
    num_ctx: int = 4096  # context window
    num_predict: int = 2000  # max tokens to predict
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1


@dataclass
class OllamaResponse:
    """
    Ollama API 响应数据类
    
    功能描述:
        存储 Ollama API 的响应结果。
    
    属性说明:
        - content: 生成的内容
        - model: 使用的模型
        - tokens_used: 使用的令牌数
        - latency_ms: 延迟（毫秒）
    """
    content: str
    model: str
    tokens_used: int
    latency_ms: float
    eval_count: int
    eval_duration_ns: int


class OllamaProvider:
    """
    Ollama provider for local LLM inference.
    
    Supports:
    - codellama:7b, codellama:13b, codellama:34b
    - deepseek-coder:6.7b, deepseek-coder:33b
    - llama3:8b, llama3:70b
    - mistral:7b, mixtral:8x7b
    - starcoder2:3b, starcoder2:7b, starcoder2:15b
    """
    
    # Model recommendations for code review
    RECOMMENDED_MODELS = {
        "code_review": [
            MODEL_CODELLAMA_34B,   # Best for code review
            MODEL_DEEPSEEK_33B,    # Alternative high quality
            MODEL_CODELLAMA_13B,   # Good balance
            MODEL_CODELLAMA_7B,    # Fast, lower quality
        ],
        "security_analysis": [
            MODEL_CODELLAMA_34B,
            MODEL_LLAMA3_70B,
            MODEL_MIXTRAL_8X7B,
        ],
        "architecture": [
            MODEL_LLAMA3_70B,
            MODEL_CODELLAMA_34B,
            MODEL_MIXTRAL_8X7B,
        ],
        "fast": [
            MODEL_CODELLAMA_7B,
            MODEL_MISTRAL_7B,
            MODEL_DEEPSEEK_6B,
        ],
    }

    def __init__(self, config: Optional[OllamaConfig] = None):
        """Initialize Ollama provider."""
        self.config = config or OllamaConfig()
        self._client: Optional[httpx.AsyncClient] = None
        self._available_models: List[str] = []
        
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=httpx.Timeout(self.config.timeout),
            )
        return self._client
    
    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if Ollama is available."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    async def list_models(self) -> List[str]:
        """List available models."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            self._available_models = [m["name"] for m in data.get("models", [])]
            return self._available_models
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []

    async def ensure_model(self, model: Optional[str] = None) -> bool:
        """Ensure model is available, pull if not."""
        model = model or self.config.model
        
        # Check if already available
        available = await self.list_models()
        if model in available:
            return True
        
        # Pull the model
        logger.info(f"Pulling Ollama model: {model}")
        try:
            client = await self._get_client()
            response = await client.post(
                "/api/pull",
                json={"name": model},
                timeout=None,  # Pulling can take a long time
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to pull model {model}: {e}")
            return False

    async def analyze_code(
        self,
        code: str,
        language: str,
        prompt_template: str,
        model: Optional[str] = None,
    ) -> OllamaResponse:
        """
        Analyze code using Ollama.
        
        Args:
            code: Source code to analyze
            language: Programming language
            prompt_template: Template with {code} and {language} placeholders
            model: Optional model override
            
        Returns:
            OllamaResponse with analysis results
        """
        model = model or self.config.model
        
        # Format prompt
        prompt = prompt_template.format(code=code, language=language)
        
        # System prompt for code review
        system_prompt = """You are an expert code reviewer and software architect with deep knowledge of:
- Security vulnerabilities (OWASP Top 10, CWE)
- Code quality and best practices
- Performance optimization
- Software architecture patterns
- Testing strategies

Provide detailed, actionable feedback with specific line references and code examples for fixes."""

        start_time = time.time()
        
        try:
            client = await self._get_client()
            
            response = await client.post(
                "/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_ctx": self.config.num_ctx,
                        "num_predict": self.config.num_predict,
                        "top_p": self.config.top_p,
                        "top_k": self.config.top_k,
                        "repeat_penalty": self.config.repeat_penalty,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            
            latency_ms = (time.time() - start_time) * 1000
            
            return OllamaResponse(
                content=data.get("response", ""),
                model=model,
                tokens_used=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                latency_ms=latency_ms,
                eval_count=data.get("eval_count", 0),
                eval_duration_ns=data.get("eval_duration", 0),
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Ollama analysis failed: {e}")
            raise

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
    ) -> OllamaResponse:
        """
        Chat completion with Ollama.
        
        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": "..."}
            model: Optional model override
            
        Returns:
            OllamaResponse with chat completion
        """
        model = model or self.config.model
        start_time = time.time()
        
        try:
            client = await self._get_client()
            
            response = await client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_ctx": self.config.num_ctx,
                        "num_predict": self.config.num_predict,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            
            latency_ms = (time.time() - start_time) * 1000
            
            return OllamaResponse(
                content=data.get("message", {}).get("content", ""),
                model=model,
                tokens_used=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                latency_ms=latency_ms,
                eval_count=data.get("eval_count", 0),
                eval_duration_ns=data.get("eval_duration", 0),
            )
            
        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
            raise

    async def embed(self, text: str, model: str = "nomic-embed-text") -> List[float]:
        """
        Generate embeddings using Ollama.
        
        Args:
            text: Text to embed
            model: Embedding model (default: nomic-embed-text)
            
        Returns:
            List of embedding values
        """
        try:
            client = await self._get_client()
            
            response = await client.post(
                "/api/embeddings",
                json={
                    "model": model,
                    "prompt": text,
                },
            )
            response.raise_for_status()
            data = response.json()
            
            return data.get("embedding", [])
            
        except Exception as e:
            logger.error(f"Ollama embedding failed: {e}")
            raise

    async def stream_analyze(
        self,
        code: str,
        language: str,
        prompt_template: str,
        model: Optional[str] = None,
    ):
        """
        Stream code analysis with real-time token output.
        
        Args:
            code: Source code to analyze
            language: Programming language
            prompt_template: Template with {code} and {language} placeholders
            model: Optional model override
            
        Yields:
            Tokens as they are generated
        """
        model = model or self.config.model
        prompt = prompt_template.format(code=code, language=language)
        
        system_prompt = """You are an expert code reviewer and software architect with deep knowledge of:
- Security vulnerabilities (OWASP Top 10, CWE)
- Code quality and best practices
- Performance optimization
- Software architecture patterns

Provide detailed, actionable feedback with specific line references."""

        try:
            client = await self._get_client()
            
            async with client.stream(
                "POST",
                "/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": True,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_ctx": self.config.num_ctx,
                        "num_predict": self.config.num_predict,
                    },
                },
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            token = data.get("response", "")
                            if token:
                                yield token
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"Streaming analysis failed: {e}")
            raise

    async def batch_analyze(
        self,
        code_items: List[Dict[str, str]],
        prompt_template: str,
        max_concurrent: int = 3,
    ) -> List[OllamaResponse]:
        """
        Batch analyze multiple code snippets with concurrency control.
        
        Args:
            code_items: List of {"code": str, "language": str} dicts
            prompt_template: Template with {code} and {language} placeholders
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of OllamaResponse objects (may include exceptions)
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_one(item: Dict[str, str]) -> OllamaResponse:
            async with semaphore:
                return await self.analyze_code(
                    item["code"],
                    item["language"],
                    prompt_template
                )
        
        tasks = [analyze_one(item) for item in code_items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results

    def get_recommended_model(
        self,
        task_type: str,
        available_only: bool = True,
    ) -> str:
        """
        Get recommended model for a specific task type.
        
        Args:
            task_type: One of 'code_review', 'security_analysis', 'architecture', 'fast'
            available_only: Only return models that are available
            
        Returns:
            Recommended model name
        """
        recommendations = self.RECOMMENDED_MODELS.get(task_type, [])
        
        if not available_only or not self._available_models:
            return recommendations[0] if recommendations else self.config.model
        
        for model in recommendations:
            if model in self._available_models:
                return model
        
        # Return first available model or default
        if self._available_models:
            return self._available_models[0]
        return self.config.model

    def get_model_stats(self) -> Dict[str, Any]:
        """
        Get statistics about available models and configuration.
        
        Returns:
            Dictionary with model information
        """
        return {
            "current_model": self.config.model,
            "available_models": self._available_models,
            "base_url": self.config.base_url,
            "config": {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "num_ctx": self.config.num_ctx,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
            },
            "recommended_models": self.RECOMMENDED_MODELS,
        }


# Code review specific prompts
CODE_REVIEW_PROMPTS = {
    "comprehensive": """Analyze the following {language} code and provide a comprehensive code review:

```{language}
{code}
```

Please provide:
1. **Security Issues**: Identify any security vulnerabilities (SQL injection, XSS, etc.)
2. **Code Quality**: Point out code smells, anti-patterns, and maintainability issues
3. **Performance**: Identify performance bottlenecks and optimization opportunities
4. **Best Practices**: Suggest improvements based on language-specific best practices
5. **Testing**: Recommend test cases that should be written

Format your response as structured JSON.""",

    "security": """Perform a security audit on the following {language} code:

```{language}
{code}
```

Focus on:
- OWASP Top 10 vulnerabilities
- Input validation issues
- Authentication/authorization flaws
- Sensitive data exposure
- Injection vulnerabilities

Provide severity ratings (Critical/High/Medium/Low) and remediation steps.""",

    "performance": """Analyze the performance of the following {language} code:

```{language}
{code}
```

Identify:
- Time complexity issues (O(n²) loops, etc.)
- Memory leaks or excessive allocation
- Database query optimization opportunities
- Caching opportunities
- Async/parallel processing improvements""",

    "architecture": """Review the architecture and design of the following {language} code:

```{language}
{code}
```

Evaluate:
- SOLID principle adherence
- Design pattern usage
- Coupling and cohesion
- Dependency management
- Testability""",
}


async def create_ollama_provider(
    base_url: str = "http://localhost:11434",
    model: str = MODEL_CODELLAMA_34B,
    ensure_model: bool = True,
) -> OllamaProvider:
    """
    Factory function to create and initialize Ollama provider.
    
    Args:
        base_url: Ollama server URL
        model: Model to use
        ensure_model: Whether to pull model if not available
        
    Returns:
        Initialized OllamaProvider
    """
    config = OllamaConfig(base_url=base_url, model=model)
    provider = OllamaProvider(config)
    
    # Check health
    if not await provider.health_check():
        raise RuntimeError(f"Ollama server not available at {base_url}")
    
    # Ensure model is available
    if ensure_model:
        if not await provider.ensure_model(model):
            logger.warning(f"Model {model} not available, using fallback")
            # Try fallback models
            for fallback in ["codellama:13b", "codellama:7b", "mistral:7b"]:
                if await provider.ensure_model(fallback):
                    provider.config.model = fallback
                    break
    
    return provider
