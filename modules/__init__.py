"""
Versioned Function Modules

Central module loader supporting three-version architecture.
"""

import importlib
from typing import Any, Optional

__version__ = "1.0.0"

# Available modules
MODULES = [
    "CodeReviewAI",
    "Authentication",
    "SelfHealing",
    "AIOrchestration",
    "Caching",
    "Monitoring",
]

# Default versions
DEFAULT_VERSIONS = {
    "CodeReviewAI": "V2",
    "Authentication": "V2",
    "SelfHealing": "V2",
    "AIOrchestration": "V2",
    "Caching": "V2",
    "Monitoring": "V2",
}


def get_module(name: str, version: Optional[str] = None) -> Any:
    """
    Get module by name and version.

    Args:
        name: Module name (e.g., "CodeReviewAI")
        version: Version string ("V1", "V2", "V3") or None for default

    Returns:
        Imported module

    Example:
        >>> auth = get_module("Authentication", "V2")
        >>> from modules import get_module
        >>> cache = get_module("Caching")  # Uses default V2
    """
    if name not in MODULES:
        raise ValueError(f"Unknown module: {name}. Available: {MODULES}")

    version = version or DEFAULT_VERSIONS.get(name, "V2")
    module_path = f"modules.{name}_{version}"

    try:
        return importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Failed to import {module_path}: {e}")


def get_production_module(name: str) -> Any:
    """Get production (V2) module."""
    return get_module(name, "V2")


def get_experimental_module(name: str) -> Any:
    """Get experimental (V1) module."""
    return get_module(name, "V1")


def get_legacy_module(name: str) -> Any:
    """Get legacy/quarantine (V3) module."""
    return get_module(name, "V3")


def list_modules() -> dict:
    """List all available modules with their versions."""
    result = {}
    for name in MODULES:
        result[name] = {
            "versions": ["V1", "V2", "V3"],
            "default": DEFAULT_VERSIONS.get(name, "V2"),
        }
    return result


# Convenience exports for production modules
def get_code_reviewer():
    """Get CodeReviewAI V2 CodeReviewer."""
    module = get_module("CodeReviewAI", "V2")
    return module.CodeReviewer


def get_auth_manager():
    """Get Authentication V2 AuthManager."""
    module = get_module("Authentication", "V2")
    return module.AuthManager


def get_health_monitor():
    """Get SelfHealing V2 HealthMonitor."""
    module = get_module("SelfHealing", "V2")
    return module.HealthMonitor


def get_orchestrator():
    """Get AIOrchestration V2 Orchestrator."""
    module = get_module("AIOrchestration", "V2")
    return module.Orchestrator


def get_cache_manager():
    """Get Caching V2 CacheManager."""
    module = get_module("Caching", "V2")
    return module.CacheManager


def get_metrics_collector():
    """Get Monitoring V2 MetricsCollector."""
    module = get_module("Monitoring", "V2")
    return module.MetricsCollector
