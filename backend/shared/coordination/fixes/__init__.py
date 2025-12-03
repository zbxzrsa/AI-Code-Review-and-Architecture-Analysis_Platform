"""
Critical Fixes Module

Contains all bug fixes and optimizations for the three-version system.
"""

from .critical_fixes import (
    # Fixed Components
    ThreadSafePromotionManager,
    SecureAccessControl,
    PersistentVersionState,
    RobustEvolutionLoop,
    PromotionCleanup,
    VersionTransitionRateLimiter,
    MetricsValidator,
    CachedAccessControl,
    FixedVersionSystem,
    
    # Exceptions
    PromotionLimitError,
    DuplicatePromotionError,
    TransitionCooldownError,
    MetricsValidationError,
)

__all__ = [
    "ThreadSafePromotionManager",
    "SecureAccessControl",
    "PersistentVersionState",
    "RobustEvolutionLoop",
    "PromotionCleanup",
    "VersionTransitionRateLimiter",
    "MetricsValidator",
    "CachedAccessControl",
    "FixedVersionSystem",
    "PromotionLimitError",
    "DuplicatePromotionError",
    "TransitionCooldownError",
    "MetricsValidationError",
]
