"""
Base AI Model Configuration and Interface
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Supported AI model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    CUSTOM = "custom"


class VersionType(Enum):
    """Version types in the three-version system"""
    V1_EXPERIMENTAL = "v1"  # New technologies, trial and error
    V2_PRODUCTION = "v2"    # Stable, user-facing
    V3_QUARANTINE = "v3"    # Deprecated, poor reviews


@dataclass
class AIConfig:
    """Configuration for an AI model"""
    model_id: str
    provider: ModelProvider
    model_name: str
    version: str
    api_endpoint: Optional[str] = None
    api_key_env: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    context_window: int = 8192
    
    # Model capabilities
    supports_streaming: bool = True
    supports_function_calling: bool = True
    supports_vision: bool = False
    
    # Performance settings
    timeout_seconds: int = 60
    retry_count: int = 3
    rate_limit_rpm: int = 60
    
    # Cost tracking
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'provider': self.provider.value,
            'model_name': self.model_name,
            'version': self.version,
            'api_endpoint': self.api_endpoint,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'supports_streaming': self.supports_streaming,
            'supports_function_calling': self.supports_function_calling,
            'description': self.description
        }


class BaseAI(ABC):
    """
    Abstract base class for AI models
    
    All AI models in the three-version system inherit from this class
    """
    
    def __init__(self, config: AIConfig, version_type: VersionType):
        self.config = config
        self.version_type = version_type
        self.is_active = True
        self.request_count = 0
        self.error_count = 0
        self.total_tokens_used = 0
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate a response from the AI model"""
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """Stream a response from the AI model"""
        pass
    
    @abstractmethod
    async def analyze_code(
        self,
        code: str,
        language: str,
        analysis_type: str = "review"
    ) -> Dict[str, Any]:
        """Analyze code and return structured results"""
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get model usage metrics"""
        error_rate = self.error_count / max(1, self.request_count)
        return {
            'model_id': self.config.model_id,
            'version_type': self.version_type.value,
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': error_rate,
            'total_tokens_used': self.total_tokens_used,
            'is_active': self.is_active
        }
    
    def record_request(self, tokens_used: int = 0, success: bool = True) -> None:
        """Record a request for metrics"""
        self.request_count += 1
        self.total_tokens_used += tokens_used
        if not success:
            self.error_count += 1
    
    def deactivate(self) -> None:
        """Deactivate the model"""
        self.is_active = False
        logger.info(f"Model {self.config.model_id} deactivated")
    
    def activate(self) -> None:
        """Activate the model"""
        self.is_active = True
        logger.info(f"Model {self.config.model_id} activated")


# Default model configurations for each version
DEFAULT_CONFIGS = {
    VersionType.V1_EXPERIMENTAL: {
        'version_control': AIConfig(
            model_id='v1-vc-ai',
            provider=ModelProvider.ANTHROPIC,
            model_name='claude-3-5-sonnet-20241022',
            version='1.0.0',
            description='V1 Version Control AI - Experimental'
        ),
        'code_ai': AIConfig(
            model_id='v1-code-ai',
            provider=ModelProvider.OPENAI,
            model_name='gpt-4-turbo-preview',
            version='1.0.0',
            description='V1 Code AI - Experimental, testing new capabilities'
        )
    },
    VersionType.V2_PRODUCTION: {
        'version_control': AIConfig(
            model_id='v2-vc-ai',
            provider=ModelProvider.ANTHROPIC,
            model_name='claude-3-5-sonnet-20241022',
            version='2.0.0',
            description='V2 Version Control AI - Production stable'
        ),
        'code_ai': AIConfig(
            model_id='v2-code-ai',
            provider=ModelProvider.ANTHROPIC,
            model_name='claude-3-5-sonnet-20241022',
            version='2.0.0',
            description='V2 Code AI - Production stable, user-facing'
        )
    },
    VersionType.V3_QUARANTINE: {
        'version_control': AIConfig(
            model_id='v3-vc-ai',
            provider=ModelProvider.OPENAI,
            model_name='gpt-3.5-turbo',
            version='3.0.0',
            description='V3 Version Control AI - Quarantine/Archive'
        ),
        'code_ai': AIConfig(
            model_id='v3-code-ai',
            provider=ModelProvider.OPENAI,
            model_name='gpt-3.5-turbo',
            version='3.0.0',
            description='V3 Code AI - Quarantine, deprecated'
        )
    }
}
