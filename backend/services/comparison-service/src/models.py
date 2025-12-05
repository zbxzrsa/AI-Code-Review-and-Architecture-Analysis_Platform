"""
Database models for comparison-service.
"""
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Float, Integer, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()


class Comparison(Base):
    """Comparison model."""
    __tablename__ = "comparisons"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    comparison_id = Column(String(255), unique=True, nullable=False, index=True)
    v1_version_id = Column(UUID(as_uuid=True), nullable=False)
    v2_version_id = Column(UUID(as_uuid=True), nullable=False)
    dataset_id = Column(UUID(as_uuid=True), nullable=False)
    dataset_size = Column(Integer, nullable=False)
    
    # Functional metrics
    accuracy_v1 = Column(Float, nullable=True)
    accuracy_v2 = Column(Float, nullable=True)
    precision_v1 = Column(Float, nullable=True)
    precision_v2 = Column(Float, nullable=True)
    recall_v1 = Column(Float, nullable=True)
    recall_v2 = Column(Float, nullable=True)
    f1_score_v1 = Column(Float, nullable=True)
    f1_score_v2 = Column(Float, nullable=True)
    false_positive_rate_v1 = Column(Float, nullable=True)
    false_positive_rate_v2 = Column(Float, nullable=True)
    false_negative_rate_v1 = Column(Float, nullable=True)
    false_negative_rate_v2 = Column(Float, nullable=True)
    
    # Non-functional metrics
    latency_p50_v1 = Column(Float, nullable=True)
    latency_p50_v2 = Column(Float, nullable=True)
    latency_p95_v1 = Column(Float, nullable=True)
    latency_p95_v2 = Column(Float, nullable=True)
    latency_p99_v1 = Column(Float, nullable=True)
    latency_p99_v2 = Column(Float, nullable=True)
    throughput_v1 = Column(Float, nullable=True)
    throughput_v2 = Column(Float, nullable=True)
    error_rate_v1 = Column(Float, nullable=True)
    error_rate_v2 = Column(Float, nullable=True)
    
    # Resource metrics
    cpu_usage_v1 = Column(Float, nullable=True)
    cpu_usage_v2 = Column(Float, nullable=True)
    memory_usage_v1 = Column(Float, nullable=True)
    memory_usage_v2 = Column(Float, nullable=True)
    
    # Economic metrics
    cost_per_analysis_v1 = Column(Float, nullable=True)
    cost_per_analysis_v2 = Column(Float, nullable=True)
    token_usage_v1 = Column(Integer, nullable=True)
    token_usage_v2 = Column(Integer, nullable=True)
    
    # Statistical significance
    statistical_tests = Column(JSON, nullable=True)  # t-test, Mann-Whitney U results
    
    # Recommendation
    recommendation = Column(String(255), nullable=False)
    confidence = Column(Float, nullable=False)
    reasoning = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "comparison_id": self.comparison_id,
            "versions": [str(self.v1_version_id), str(self.v2_version_id)],
            "dataset": str(self.dataset_id),
            "dataset_size": self.dataset_size,
            "metrics": {
                "accuracy": {
                    "v1": self.accuracy_v1,
                    "v2": self.accuracy_v2,
                    "change": f"{((self.accuracy_v2 - self.accuracy_v1) / self.accuracy_v1 * 100):.1f}%" if self.accuracy_v1 else None,
                },
                "precision": {
                    "v1": self.precision_v1,
                    "v2": self.precision_v2,
                },
                "recall": {
                    "v1": self.recall_v1,
                    "v2": self.recall_v2,
                },
                "f1_score": {
                    "v1": self.f1_score_v1,
                    "v2": self.f1_score_v2,
                },
                "latency_p95": {
                    "v1": self.latency_p95_v1,
                    "v2": self.latency_p95_v2,
                    "change": f"{((self.latency_p95_v2 - self.latency_p95_v1) / self.latency_p95_v1 * 100):.1f}%" if self.latency_p95_v1 else None,
                },
                "cost_per_analysis": {
                    "v1": self.cost_per_analysis_v1,
                    "v2": self.cost_per_analysis_v2,
                    "change": f"{((self.cost_per_analysis_v2 - self.cost_per_analysis_v1) / self.cost_per_analysis_v1 * 100):.1f}%" if self.cost_per_analysis_v1 else None,
                },
            },
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "created_at": self.created_at.isoformat(),
        }


class StatisticalTest(Base):
    """Statistical test result model."""
    __tablename__ = "statistical_tests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    comparison_id = Column(UUID(as_uuid=True), ForeignKey("comparisons.id"), nullable=False, index=True)
    test_name = Column(String(100), nullable=False)  # t-test, Mann-Whitney U, etc.
    metric = Column(String(100), nullable=False)  # accuracy, latency, etc.
    p_value = Column(Float, nullable=False)
    is_significant = Column(String(50), nullable=False)  # yes, no, borderline
    effect_size = Column(Float, nullable=True)
    confidence_level = Column(Float, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "test_name": self.test_name,
            "metric": self.metric,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "effect_size": self.effect_size,
            "confidence_level": self.confidence_level,
        }
