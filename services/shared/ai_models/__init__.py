"""
AI Models Configuration for Three-Version System

Each version contains:
- Version Control AI: Internal model for version management and evolution
- Code AI: User-facing model for code review and analysis

Version Lifecycle:
- V1 (Experimental): New technologies, trial and error
- V2 (Production): Stable, user-facing, unchanged
- V3 (Quarantine): Deprecated technologies, poor reviews
"""

from .base_ai import BaseAI, AIConfig
from .version_control_ai import VersionControlAI
from .code_ai import CodeAI
from .model_registry import UserModelRegistry

__all__ = [
    'BaseAI',
    'AIConfig',
    'VersionControlAI',
    'CodeAI',
    'UserModelRegistry'
]
