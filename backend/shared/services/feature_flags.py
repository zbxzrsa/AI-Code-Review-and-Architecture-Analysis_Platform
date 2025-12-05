"""
Feature Flag Service - Request-level feature flags for gradual rollouts.

Enables gradual rollout of new features and A/B testing capabilities.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from enum import Enum
import json

logger = logging.getLogger(__name__)


class RolloutStrategy(str, Enum):
    """Rollout strategies for feature flags."""
    ALL_USERS = "all_users"
    PERCENTAGE = "percentage"
    USER_LIST = "user_list"
    GRADUAL = "gradual"
    CANARY = "canary"


@dataclass
class FeatureFlag:
    """Feature flag definition."""
    name: str
    enabled: bool
    description: str = ""
    rollout_strategy: RolloutStrategy = RolloutStrategy.ALL_USERS
    rollout_percentage: float = 0.0  # 0-100
    allowed_users: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_enabled_for_user(self, user_id: str) -> bool:
        """Check if flag is enabled for specific user."""
        if not self.enabled:
            return False

        if self.rollout_strategy == RolloutStrategy.ALL_USERS:
            return True

        if self.rollout_strategy == RolloutStrategy.USER_LIST:
            return user_id in self.allowed_users

        if self.rollout_strategy == RolloutStrategy.PERCENTAGE:
            # Deterministic hash-based rollout
            user_hash = hash(user_id) % 100
            return user_hash < self.rollout_percentage

        if self.rollout_strategy == RolloutStrategy.GRADUAL:
            # Gradual rollout based on time
            elapsed_hours = (datetime.now(timezone.utc) - self.created_at).total_seconds() / 3600
            current_percentage = min(100, elapsed_hours * 10)  # 10% per hour
            user_hash = hash(user_id) % 100
            return user_hash < current_percentage

        if self.rollout_strategy == RolloutStrategy.CANARY:
            # Canary deployment - only for specific users
            return user_id in self.allowed_users

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "description": self.description,
            "rollout_strategy": self.rollout_strategy.value,
            "rollout_percentage": self.rollout_percentage,
            "allowed_users": self.allowed_users,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class FeatureFlagService:
    """Service for managing feature flags."""

    def __init__(self):
        """Initialize feature flag service."""
        self.flags: Dict[str, FeatureFlag] = {}
        self._initialize_default_flags()

    def _initialize_default_flags(self) -> None:
        """Initialize default feature flags."""
        # Code Review AI features
        self.register_flag(
            FeatureFlag(
                name="code-review-ai-sast",
                enabled=True,
                description="SAST (Static Application Security Testing) scanning",
                rollout_strategy=RolloutStrategy.ALL_USERS,
            )
        )

        self.register_flag(
            FeatureFlag(
                name="code-review-ai-performance-analysis",
                enabled=True,
                description="Performance bottleneck detection",
                rollout_strategy=RolloutStrategy.PERCENTAGE,
                rollout_percentage=80.0,
            )
        )

        self.register_flag(
            FeatureFlag(
                name="code-review-ai-patch-generation",
                enabled=False,
                description="Intelligent patch suggestion generation",
                rollout_strategy=RolloutStrategy.GRADUAL,
            )
        )

        self.register_flag(
            FeatureFlag(
                name="code-review-ai-test-generation",
                enabled=False,
                description="Automatic test generation",
                rollout_strategy=RolloutStrategy.CANARY,
                allowed_users=["admin@example.com", "beta-tester@example.com"],
            )
        )

        # Version Control AI features
        self.register_flag(
            FeatureFlag(
                name="version-control-ai-auto-promotion",
                enabled=True,
                description="Automatic experiment promotion to V2",
                rollout_strategy=RolloutStrategy.ALL_USERS,
            )
        )

        self.register_flag(
            FeatureFlag(
                name="version-control-ai-regression-detection",
                enabled=True,
                description="Regression detection against baselines",
                rollout_strategy=RolloutStrategy.PERCENTAGE,
                rollout_percentage=100.0,
            )
        )

        self.register_flag(
            FeatureFlag(
                name="version-control-ai-cost-analysis",
                enabled=False,
                description="Cost-benefit analysis for promotions",
                rollout_strategy=RolloutStrategy.GRADUAL,
            )
        )

        logger.info(
            "Default feature flags initialized",
            flag_count=len(self.flags),
        )

    def register_flag(self, flag: FeatureFlag) -> None:
        """Register a feature flag."""
        self.flags[flag.name] = flag
        logger.info(
            "Feature flag registered",
            flag_name=flag.name,
            enabled=flag.enabled,
        )

    def is_enabled(
        self,
        flag_name: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Check if feature flag is enabled.

        Args:
            flag_name: Name of the flag
            user_id: Optional user ID for user-specific rollouts

        Returns:
            True if flag is enabled for the user
        """
        flag = self.flags.get(flag_name)
        if not flag:
            logger.warning(f"Feature flag not found: {flag_name}")
            return False

        if user_id:
            return flag.is_enabled_for_user(user_id)
        else:
            return flag.enabled

    def get_enabled_flags(self, user_id: Optional[str] = None) -> List[str]:
        """Get all enabled flags for a user."""
        enabled = []
        for flag_name, flag in self.flags.items():
            if self.is_enabled(flag_name, user_id):
                enabled.append(flag_name)
        return enabled

    def update_flag(
        self,
        flag_name: str,
        enabled: Optional[bool] = None,
        rollout_percentage: Optional[float] = None,
        allowed_users: Optional[List[str]] = None,
    ) -> bool:
        """Update a feature flag."""
        flag = self.flags.get(flag_name)
        if not flag:
            logger.warning(f"Feature flag not found: {flag_name}")
            return False

        if enabled is not None:
            flag.enabled = enabled

        if rollout_percentage is not None:
            flag.rollout_percentage = min(100, max(0, rollout_percentage))

        if allowed_users is not None:
            flag.allowed_users = allowed_users

        flag.updated_at = datetime.now(timezone.utc)

        logger.info(
            "Feature flag updated",
            flag_name=flag_name,
            enabled=flag.enabled,
            rollout_percentage=flag.rollout_percentage,
        )

        return True

    def get_flag(self, flag_name: str) -> Optional[FeatureFlag]:
        """Get feature flag details."""
        return self.flags.get(flag_name)

    def list_flags(self) -> List[Dict[str, Any]]:
        """List all feature flags."""
        return [flag.to_dict() for flag in self.flags.values()]

    def get_flag_stats(self) -> Dict[str, Any]:
        """Get statistics about feature flags."""
        enabled_count = sum(1 for flag in self.flags.values() if flag.enabled)
        return {
            "total_flags": len(self.flags),
            "enabled_flags": enabled_count,
            "disabled_flags": len(self.flags) - enabled_count,
            "flags": {
                name: {
                    "enabled": flag.enabled,
                    "strategy": flag.rollout_strategy.value,
                    "percentage": flag.rollout_percentage,
                }
                for name, flag in self.flags.items()
            },
        }


# Global feature flag service instance
_feature_flag_service: Optional[FeatureFlagService] = None


def get_feature_flag_service() -> FeatureFlagService:
    """Get or create global feature flag service."""
    global _feature_flag_service
    if _feature_flag_service is None:
        _feature_flag_service = FeatureFlagService()
    return _feature_flag_service


def is_feature_enabled(
    flag_name: str,
    user_id: Optional[str] = None,
) -> bool:
    """Check if feature is enabled."""
    service = get_feature_flag_service()
    return service.is_enabled(flag_name, user_id)


def get_enabled_features(user_id: Optional[str] = None) -> List[str]:
    """Get all enabled features for user."""
    service = get_feature_flag_service()
    return service.get_enabled_flags(user_id)
