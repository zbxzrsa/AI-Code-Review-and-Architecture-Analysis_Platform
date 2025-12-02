"""
Database models for provider-service.
"""
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Float, Integer, Text, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()


class Provider(Base):
    """AI provider model."""
    __tablename__ = "providers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), unique=True, nullable=False, index=True)
    provider_type = Column(String(50), nullable=False)  # openai, anthropic, huggingface, local
    model_name = Column(String(100), nullable=False)
    api_endpoint = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    is_platform_provided = Column(Boolean, default=True)
    cost_per_1k_tokens = Column(Float, nullable=False)
    max_tokens = Column(Integer, nullable=False)
    timeout_seconds = Column(Integer, default=30)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "provider_type": self.provider_type,
            "model_name": self.model_name,
            "is_active": self.is_active,
            "is_platform_provided": self.is_platform_provided,
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
            "max_tokens": self.max_tokens,
        }


class UserProvider(Base):
    """User-provided provider model."""
    __tablename__ = "user_providers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    provider_name = Column(String(100), nullable=False)
    provider_type = Column(String(50), nullable=False)  # openai, anthropic, huggingface
    model_name = Column(String(100), nullable=False)
    encrypted_api_key = Column(Text, nullable=False)
    encrypted_dek = Column(Text, nullable=False)  # Data Encryption Key
    key_last_4_chars = Column(String(4), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "provider_name": self.provider_name,
            "provider_type": self.provider_type,
            "model_name": self.model_name,
            "key_last_4_chars": f"****{self.key_last_4_chars}",
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
        }


class ProviderHealth(Base):
    """Provider health status model."""
    __tablename__ = "provider_health"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    provider_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    is_healthy = Column(Boolean, default=True)
    last_check_at = Column(DateTime, nullable=True)
    last_error = Column(Text, nullable=True)
    consecutive_failures = Column(Integer, default=0)
    response_time_ms = Column(Float, nullable=True)
    success_rate = Column(Float, default=1.0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "provider_id": str(self.provider_id),
            "is_healthy": self.is_healthy,
            "last_check_at": self.last_check_at.isoformat() if self.last_check_at else None,
            "response_time_ms": self.response_time_ms,
            "success_rate": self.success_rate,
        }


class UserQuota(Base):
    """User quota model."""
    __tablename__ = "user_quotas"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), unique=True, nullable=False, index=True)
    daily_limit = Column(Integer, nullable=False)  # Requests per day
    monthly_limit = Column(Integer, nullable=False)  # Requests per month
    daily_cost_limit = Column(Float, nullable=False)  # USD per day
    monthly_cost_limit = Column(Float, nullable=False)  # USD per month
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "daily_limit": self.daily_limit,
            "monthly_limit": self.monthly_limit,
            "daily_cost_limit": self.daily_cost_limit,
            "monthly_cost_limit": self.monthly_cost_limit,
        }


class UsageTracking(Base):
    """Usage tracking model."""
    __tablename__ = "usage_tracking"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    date = Column(String(10), nullable=False)  # YYYY-MM-DD
    requests_count = Column(Integer, default=0)
    tokens_used = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "date": self.date,
            "requests_count": self.requests_count,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
        }


class CostAlert(Base):
    """Cost alert model."""
    __tablename__ = "cost_alerts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    alert_type = Column(String(50), nullable=False)  # daily_80, daily_90, daily_100, monthly_80, etc.
    threshold_percentage = Column(Integer, nullable=False)  # 80, 90, 100
    triggered_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime, nullable=True)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "alert_type": self.alert_type,
            "threshold_percentage": self.threshold_percentage,
            "triggered_at": self.triggered_at.isoformat(),
            "acknowledged": self.acknowledged,
        }
