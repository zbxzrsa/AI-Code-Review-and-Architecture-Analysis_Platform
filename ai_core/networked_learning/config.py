"""
Configuration for Networked Learning System

Defines all configuration classes for:
- Data source priorities and collection schedules
- Quality thresholds for data cleaning
- Retention policies for data management
- Deprecation criteria for technology elimination
"""

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Any, Dict, List, Optional


class DataSourcePriority(int, Enum):
    """
    Priority levels for data sources.
    Lower number = higher priority.
    """
    GITHUB = 1      # Technical documents, open-source code
    ARXIV = 2       # Academic papers
    TECH_BLOGS = 3  # Selected technical articles
    CUSTOM = 4      # User-defined sources


class CollectionProtocol(str, Enum):
    """Supported collection protocols."""
    HTTPS = "https"
    API = "api"
    RSS = "rss"
    WEBHOOK = "webhook"


@dataclass
class CollectionSchedule:
    """
    Schedule configuration for data collection.
    
    Attributes:
        interval_seconds: Time between collection cycles (default: 1 hour)
        max_items_per_cycle: Maximum items to collect per source per cycle
        timeout_seconds: Request timeout
        retry_attempts: Number of retry attempts on failure
        backoff_multiplier: Exponential backoff multiplier
    """
    interval_seconds: int = 3600  # 1 hour
    max_items_per_cycle: int = 100
    timeout_seconds: int = 30
    retry_attempts: int = 3
    backoff_multiplier: float = 2.0
    
    # Rate limiting per source
    github_rate_limit: int = 5000  # requests per hour
    arxiv_rate_limit: int = 100   # requests per 5 seconds
    blog_rate_limit: int = 60     # requests per minute


@dataclass
class QualityThresholds:
    """
    Quality thresholds for data cleaning pipeline.
    
    Attributes:
        min_quality_score: Minimum quality score (0.0-1.0) for acceptance
        content_integrity_weight: Weight for content integrity assessment
        technical_relevance_weight: Weight for technical relevance
        timeliness_weight: Weight for content freshness
        duplicate_similarity_threshold: Similarity threshold for duplicate detection
    """
    min_quality_score: float = 0.8
    content_integrity_weight: float = 0.4
    technical_relevance_weight: float = 0.4
    timeliness_weight: float = 0.2
    duplicate_similarity_threshold: float = 0.85
    
    # Additional quality filters
    min_content_length: int = 100  # characters
    max_content_length: int = 1_000_000  # 1MB text
    min_code_ratio: float = 0.1  # For code-focused content
    max_ad_ratio: float = 0.1  # Maximum advertisement content
    
    def calculate_score(
        self,
        integrity: float,
        relevance: float,
        timeliness: float,
    ) -> float:
        """Calculate weighted quality score."""
        return (
            integrity * self.content_integrity_weight +
            relevance * self.technical_relevance_weight +
            timeliness * self.timeliness_weight
        )


@dataclass
class RetentionPolicy:
    """
    Data retention policy configuration.
    
    Attributes:
        raw_data_retention_days: Days to retain raw/original data
        processed_data_permanent: Whether processed data is permanent
        cleanup_schedule_hour: Hour of day for cleanup tasks (0-23)
        archive_before_delete: Whether to archive before deletion
    """
    raw_data_retention_days: int = 90
    processed_data_permanent: bool = True
    cleanup_schedule_hour: int = 0  # Midnight
    archive_before_delete: bool = True
    
    # Storage thresholds
    archive_compression: str = "gzip"
    max_archive_size_gb: float = 100.0
    auto_delete_deprecated: bool = True


@dataclass
class DeprecationCriteria:
    """
    Criteria for technology deprecation/elimination.
    
    Technologies are deprecated when:
    - Accuracy falls below threshold
    - Consecutive evaluation failures exceed limit
    
    Attributes:
        min_accuracy: Minimum accuracy threshold (default: 0.75)
        max_consecutive_failures: Max consecutive failures before deprecation
        evaluation_window_days: Window for evaluation history
        require_user_confirmation: Require manual confirmation for deprecation
    """
    min_accuracy: float = 0.75
    max_consecutive_failures: int = 3
    evaluation_window_days: int = 30
    require_user_confirmation: bool = True
    
    # Actions on deprecation
    stop_learning_tasks: bool = True
    trigger_data_cleanup: bool = True
    notify_administrators: bool = True
    grace_period_days: int = 7  # Days before final cleanup


@dataclass
class MemoryConfig:
    """
    Memory management configuration.
    
    Attributes:
        max_memory_percent: Maximum system memory usage (default: 70%)
        cache_strategy: Caching strategy (LRU, LFU, ARC)
        cache_size_mb: Maximum cache size in MB
        gc_threshold_percent: Threshold to trigger garbage collection
    """
    max_memory_percent: float = 0.70
    cache_strategy: str = "lru"
    cache_size_mb: int = 1024  # 1GB
    gc_threshold_percent: float = 0.80
    
    # Eviction settings
    eviction_batch_size: int = 100
    min_item_age_seconds: int = 300  # Don't evict items < 5 min old


@dataclass
class StorageConfig:
    """
    Storage management configuration.
    
    Attributes:
        enable_sharding: Enable automatic sharding
        shard_size_gb: Target size per shard
        enable_horizontal_scaling: Enable horizontal scaling
        replication_factor: Data replication factor
    """
    enable_sharding: bool = True
    shard_size_gb: float = 10.0
    enable_horizontal_scaling: bool = True
    replication_factor: int = 2
    
    # Storage backends
    primary_backend: str = "postgresql"  # postgresql, mongodb, s3
    archive_backend: str = "s3"
    cache_backend: str = "redis"


@dataclass
class MonitoringConfig:
    """
    System monitoring configuration.
    
    Attributes:
        enable_real_time_monitoring: Enable real-time status monitoring
        metrics_interval_seconds: Interval for metrics collection
        log_level: Logging level
        alert_channels: Channels for alert notifications
    """
    enable_real_time_monitoring: bool = True
    metrics_interval_seconds: int = 10
    log_level: str = "INFO"
    alert_channels: List[str] = field(default_factory=lambda: ["slack", "email"])
    
    # Performance thresholds for alerts
    max_processing_latency_ms: int = 500
    min_availability_percent: float = 99.9
    max_daily_data_tb: float = 1.0


@dataclass
class UserReviewConfig:
    """
    User review/approval workflow configuration.
    
    Defines which operations require manual confirmation.
    """
    require_new_technology_approval: bool = True
    require_deprecation_approval: bool = True
    require_parameter_change_approval: bool = True
    
    # Approval settings
    approval_timeout_hours: int = 72
    auto_approve_minor_changes: bool = True
    notify_on_pending_approval: bool = True
    escalation_after_hours: int = 24


@dataclass
class DataSourceConfig:
    """Configuration for a specific data source."""
    name: str
    priority: DataSourcePriority
    enabled: bool = True
    protocol: CollectionProtocol = CollectionProtocol.API
    base_url: str = ""
    api_key_env: str = ""  # Environment variable for API key
    rate_limit: int = 100  # Requests per hour
    custom_headers: Dict[str, str] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NetworkedLearningConfig:
    """
    Master configuration for the Networked Learning System.
    
    Aggregates all sub-configurations for easy management.
    """
    # Version target (v1 = experiment, v3 = quarantine)
    target_version: str = "v1"
    
    # Sub-configurations
    collection: CollectionSchedule = field(default_factory=CollectionSchedule)
    quality: QualityThresholds = field(default_factory=QualityThresholds)
    retention: RetentionPolicy = field(default_factory=RetentionPolicy)
    deprecation: DeprecationCriteria = field(default_factory=DeprecationCriteria)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    user_review: UserReviewConfig = field(default_factory=UserReviewConfig)
    
    # Data sources (ordered by priority)
    data_sources: List[DataSourceConfig] = field(default_factory=lambda: [
        DataSourceConfig(
            name="github",
            priority=DataSourcePriority.GITHUB,
            base_url="https://api.github.com",
            api_key_env="GITHUB_TOKEN",
            rate_limit=5000,
            filters={"language": ["python", "javascript", "go", "rust"]},
        ),
        DataSourceConfig(
            name="arxiv",
            priority=DataSourcePriority.ARXIV,
            base_url="https://export.arxiv.org/api",
            rate_limit=100,
            filters={"categories": ["cs.SE", "cs.AI", "cs.LG", "cs.PL"]},
        ),
        DataSourceConfig(
            name="tech_blogs",
            priority=DataSourcePriority.TECH_BLOGS,
            base_url="",  # Multiple sources
            rate_limit=60,
            filters={"domains": [
                "engineering.*.com",
                "blog.*.io",
                "dev.to",
                "medium.com/tag/programming",
            ]},
        ),
    ])
    
    # V2 integration
    v2_push_endpoint: str = "/api/v2/learning/ingest"
    v2_batch_size: int = 100
    
    def get_source_by_name(self, name: str) -> Optional[DataSourceConfig]:
        """Get data source configuration by name."""
        for source in self.data_sources:
            if source.name == name:
                return source
        return None
    
    def get_sources_by_priority(self) -> List[DataSourceConfig]:
        """Get enabled data sources sorted by priority."""
        return sorted(
            [s for s in self.data_sources if s.enabled],
            key=lambda s: s.priority.value
        )
