"""
数据清洗管道 (Data Cleansing Pipeline)

模块功能描述:
    从 V1/V3 接收原始学习数据，执行多阶段清洗和验证，
    将高质量数据输送给 V2 生产系统。

清洗阶段:
    1. 去重（Deduplication）: 移除重复内容
    2. 规范化（Normalization）: 标准化格式和编码
    3. 验证（Validation）: 检查必填字段和约束
    4. 富化（Enrichment）: 添加元数据和质量标记
    5. 输出（Output）: 达到质量阈值后发送到 V2

主要特性:
    - 异步处理支持高吞吐量
    - 可配置的质量阈值
    - 详细的统计和跟踪
    - 可插拔的验证规则
    - 基于内容 hash 的去重

最后修改日期: 2024-12-07
"""

import asyncio
import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from collections import OrderedDict
from enum import Enum
from abc import ABC, abstractmethod
import logging
import aiohttp

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Models
# =============================================================================

class CleansingStage(Enum):
    """
    清洗阶段枚举
    
    定义数据清洗的各个阶段。
    
    阶段说明:
        - RAW: 原始数据
        - DEDUPLICATION: 去重阶段
        - NORMALIZATION: 规范化阶段
        - VALIDATION: 验证阶段
        - ENRICHMENT: 富化阶段
        - READY: 已就绪
        - REJECTED: 已拒绝
    """
    RAW = "raw"
    DEDUPLICATION = "deduplication"
    NORMALIZATION = "normalization"
    VALIDATION = "validation"
    ENRICHMENT = "enrichment"
    READY = "ready"
    REJECTED = "rejected"


class RejectionReason(Enum):
    """
    拒绝原因枚举
    
    定义数据被拒绝的各种原因。
    
    原因说明:
        - DUPLICATE: 重复内容
        - TOO_SHORT: 内容过短
        - TOO_LONG: 内容过长
        - MISSING_FIELD: 缺少必填字段
        - INVALID_FORMAT: 格式无效
        - LOW_QUALITY: 质量过低
        - BLOCKED_CONTENT: 内容被屏蔽
        - PROCESSING_ERROR: 处理错误
    """
    DUPLICATE = "duplicate"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    MISSING_FIELD = "missing_field"
    INVALID_FORMAT = "invalid_format"
    LOW_QUALITY = "low_quality"
    BLOCKED_CONTENT = "blocked_content"
    PROCESSING_ERROR = "processing_error"


@dataclass
class CleansingResult:
    """
    清洗结果数据类
    
    功能描述:
        表示处理单个数据项的结果。
    
    属性说明:
        - data_id: 数据标识符
        - stage: 当前阶段
        - original_quality: 原始质量分
        - final_quality: 最终质量分
        - passed: 是否通过
        - rejection_reason: 拒绝原因
        - issues: 问题列表
        - transformations: 已执行的转换
    """
    data_id: str
    stage: CleansingStage
    original_quality: float
    final_quality: float
    passed: bool
    rejection_reason: Optional[RejectionReason] = None
    issues: List[str] = field(default_factory=list)
    transformations: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_id": self.data_id,
            "stage": self.stage.value,
            "original_quality": round(self.original_quality, 3),
            "final_quality": round(self.final_quality, 3),
            "passed": self.passed,
            "rejection_reason": self.rejection_reason.value if self.rejection_reason else None,
            "issues": self.issues,
            "transformations": self.transformations,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


@dataclass
class CleansingConfig:
    """
    清洗配置 / Cleansing Configuration
    
    Comprehensive configuration for the cleansing pipeline.
    """
    # Content length constraints
    min_content_length: int = 50
    max_content_length: int = 100000
    
    # Quality thresholds
    min_final_quality: float = 0.7
    quality_boost_per_transform: float = 0.02
    quality_penalty_per_issue: float = 0.05
    
    # Stage toggles
    enable_dedup: bool = True
    enable_normalization: bool = True
    enable_validation: bool = True
    enable_enrichment: bool = True
    
    # Deduplication settings
    similarity_threshold: float = 0.85
    dedup_cache_max_size: int = 100000
    use_content_hash: bool = True
    use_simhash: bool = False
    
    # Validation rules
    required_fields: List[str] = field(default_factory=lambda: [
        "title", "content", "source"
    ])
    
    # Blocked patterns (spam, ads, etc.)
    blocked_patterns: List[str] = field(default_factory=lambda: [
        r"buy now",
        r"click here",
        r"free trial",
        r"limited time offer",
        r"subscribe now",
    ])
    
    # Normalization settings
    normalize_unicode: bool = True
    remove_control_chars: bool = True
    collapse_whitespace: bool = True
    strip_html_tags: bool = True
    
    # V2 integration
    v2_push_enabled: bool = True
    v2_push_endpoint: str = "/api/v2/learning/ingest"
    v2_batch_size: int = 50


@dataclass
class PipelineStats:
    """
    管道统计 / Pipeline Statistics
    """
    total_received: int = 0
    total_passed: int = 0
    total_rejected: int = 0
    rejections_by_reason: Dict[str, int] = field(default_factory=dict)
    rejections_by_stage: Dict[str, int] = field(default_factory=dict)
    avg_processing_time_ms: float = 0.0
    avg_quality_improvement: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        pass_rate = self.total_passed / self.total_received if self.total_received > 0 else 0
        return {
            "total_received": self.total_received,
            "total_passed": self.total_passed,
            "total_rejected": self.total_rejected,
            "pass_rate": round(pass_rate * 100, 2),
            "rejections_by_reason": self.rejections_by_reason,
            "rejections_by_stage": self.rejections_by_stage,
            "avg_processing_time_ms": round(self.avg_processing_time_ms, 2),
            "avg_quality_improvement": round(self.avg_quality_improvement, 3),
        }


# =============================================================================
# Validation Rules
# =============================================================================

class ValidationRule(ABC):
    """Abstract base class for validation rules."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def validate(self, item: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate an item.
        
        Returns:
            Tuple of (passed, error_message)
        """
        pass


class RequiredFieldRule(ValidationRule):
    """Check for required fields."""
    
    def __init__(self, fields: List[str]):
        self.fields = fields
    
    @property
    def name(self) -> str:
        return "required_fields"
    
    def validate(self, item: Any) -> Tuple[bool, Optional[str]]:
        for field in self.fields:
            value = getattr(item, field, None)
            if not value:
                return False, f"Missing required field: {field}"
        return True, None


class ContentLengthRule(ValidationRule):
    """Check content length constraints."""
    
    def __init__(self, min_length: int, max_length: int):
        self.min_length = min_length
        self.max_length = max_length
    
    @property
    def name(self) -> str:
        return "content_length"
    
    def validate(self, item: Any) -> Tuple[bool, Optional[str]]:
        content = getattr(item, "content", "")
        length = len(content)
        
        if length < self.min_length:
            return False, f"Content too short: {length} < {self.min_length}"
        
        # Note: We don't fail on too long, just truncate in normalization
        return True, None


class BlockedPatternRule(ValidationRule):
    """Check for blocked content patterns."""
    
    def __init__(self, patterns: List[str]):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
    
    @property
    def name(self) -> str:
        return "blocked_patterns"
    
    def validate(self, item: Any) -> Tuple[bool, Optional[str]]:
        content = getattr(item, "content", "").lower()
        title = getattr(item, "title", "").lower()
        text = f"{title} {content}"
        
        for pattern in self.patterns:
            if pattern.search(text):
                return False, f"Blocked pattern found: {pattern.pattern}"
        
        return True, None


# =============================================================================
# Deduplication
# =============================================================================

class DeduplicationCache:
    """
    去重缓存 / Deduplication Cache
    
    Maintains a cache of content hashes for fast duplicate detection.
    """
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self._hashes: Dict[str, str] = {}  # hash -> data_id
        self._access_order: OrderedDict[str, None] = OrderedDict()  # LRU tracking - O(1)
    
    def get_hash(self, content: str) -> str:
        """Generate content hash."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def is_duplicate(self, content: str) -> Tuple[bool, Optional[str]]:
        """
        Check if content is a duplicate.
        
        Returns:
            Tuple of (is_duplicate, original_data_id)
        """
        content_hash = self.get_hash(content)
        
        if content_hash in self._hashes:
            return True, self._hashes[content_hash]
        
        return False, None
    
    def add(self, content: str, data_id: str):
        """Add content to cache."""
        content_hash = self.get_hash(content)
        
        # Check size limit
        if len(self._hashes) >= self.max_size:
            self._evict_oldest()
        
        self._hashes[content_hash] = data_id
        self._access_order[content_hash] = None  # O(1) insert
    
    def _evict_oldest(self):
        """Evict oldest entries (LRU)."""
        if len(self._access_order) > 0:
            oldest_hash, _ = self._access_order.popitem(last=False)  # O(1) pop oldest
            self._hashes.pop(oldest_hash, None)
    
    def clear(self):
        """Clear the cache."""
        self._hashes.clear()
        self._access_order.clear()
    
    @property
    def size(self) -> int:
        return len(self._hashes)


# =============================================================================
# Normalizers
# =============================================================================

class ContentNormalizer:
    """
    内容规范化器 / Content Normalizer
    
    Normalizes content format and encoding.
    """
    
    def __init__(self, config: CleansingConfig):
        self.config = config
    
    def normalize(self, text: str) -> Tuple[str, List[str]]:
        """
        Normalize text content.
        
        Returns:
            Tuple of (normalized_text, transformations_applied)
        """
        transformations = []
        
        if not text:
            return text, transformations
        
        # Unicode normalization
        if self.config.normalize_unicode:
            text = unicodedata.normalize("NFC", text)
            transformations.append("unicode_normalized")
        
        # Remove control characters
        if self.config.remove_control_chars:
            original_len = len(text)
            text = "".join(
                char for char in text
                if not unicodedata.category(char).startswith("C")
                or char in "\n\t"
            )
            if len(text) != original_len:
                transformations.append("control_chars_removed")
        
        # Strip HTML tags
        if self.config.strip_html_tags:
            original = text
            text = re.sub(r'<[^>]+>', ' ', text)
            if text != original:
                transformations.append("html_stripped")
        
        # Collapse whitespace
        if self.config.collapse_whitespace:
            original = text
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            if text != original:
                transformations.append("whitespace_collapsed")
        
        # Truncate if too long
        if len(text) > self.config.max_content_length:
            text = text[:self.config.max_content_length]
            transformations.append("content_truncated")
        
        return text, transformations


# =============================================================================
# Main Pipeline
# =============================================================================

class DataCleansingPipeline:
    """
    数据清洗管道 / Data Cleansing Pipeline
    
    Multi-stage pipeline for cleaning and validating learning data.
    
    Stages:
    1. Deduplication - Remove duplicate content
    2. Normalization - Standardize format
    3. Validation - Check constraints
    4. Enrichment - Add metadata
    5. Output - Send to V2
    
    Usage:
        config = CleansingConfig(min_final_quality=0.7)
        
        async def on_ready(item, result):
            print(f"Ready for V2: {item.title}")
        
        pipeline = DataCleansingPipeline(config, on_data_ready_for_v2=on_ready)
        
        results = await pipeline.process_batch(data_items)
    """
    
    def __init__(
        self,
        config: Optional[CleansingConfig] = None,
        on_data_ready_for_v2: Optional[Callable[[Any, CleansingResult], Any]] = None,
    ):
        """
        Initialize the cleansing pipeline.
        
        Args:
            config: Pipeline configuration
            on_data_ready_for_v2: Callback when data passes all stages
        """
        self.config = config or CleansingConfig()
        self.on_data_ready_for_v2 = on_data_ready_for_v2
        
        # Components
        self._dedup_cache = DeduplicationCache(self.config.dedup_cache_max_size)
        self._normalizer = ContentNormalizer(self.config)
        self._validation_rules: List[ValidationRule] = self._build_validation_rules()
        
        # Statistics
        self.stats = PipelineStats()
        
        # V2 integration
        self._v2_session: Optional[aiohttp.ClientSession] = None
        self._v2_buffer: List[Tuple[Any, CleansingResult]] = []
    
    def _build_validation_rules(self) -> List[ValidationRule]:
        """Build validation rules from config."""
        rules = []
        
        rules.append(RequiredFieldRule(self.config.required_fields))
        rules.append(ContentLengthRule(
            self.config.min_content_length,
            self.config.max_content_length
        ))
        
        if self.config.blocked_patterns:
            rules.append(BlockedPatternRule(self.config.blocked_patterns))
        
        return rules
    
    async def start(self):
        """Start the pipeline (initialize resources)."""
        if self.config.v2_push_enabled:
            self._v2_session = aiohttp.ClientSession()
        logger.info("Data cleansing pipeline started")
    
    async def stop(self):
        """Stop the pipeline (cleanup resources)."""
        # Flush remaining buffer
        if self._v2_buffer:
            await self._flush_v2_buffer()
        
        if self._v2_session:
            await self._v2_session.close()
            self._v2_session = None
        
        logger.info("Data cleansing pipeline stopped")
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
        return False
    
    async def process_batch(self, data_items: List[Any]) -> List[CleansingResult]:
        """
        批量处理数据 / Process a batch of data items.
        
        Args:
            data_items: List of data items to process
            
        Returns:
            List of CleansingResult for each item
        """
        results = []
        
        for item in data_items:
            result = await self.process_item(item)
            results.append(result)
            
            if result.passed:
                # Callback
                if self.on_data_ready_for_v2:
                    try:
                        await self.on_data_ready_for_v2(item, result)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
                # Buffer for V2
                if self.config.v2_push_enabled:
                    self._v2_buffer.append((item, result))
                    if len(self._v2_buffer) >= self.config.v2_batch_size:
                        await self._flush_v2_buffer()
        
        return results
    
    async def process_item(self, item: Any) -> CleansingResult:
        """
        处理单个数据项 / Process a single data item.
        
        Args:
            item: Data item to process
            
        Returns:
            CleansingResult with processing details
        """
        import time
        start_time = time.time()
        
        self.stats.total_received += 1
        
        # Initialize result
        result = CleansingResult(
            data_id=getattr(item, "data_id", str(id(item))),
            stage=CleansingStage.RAW,
            original_quality=getattr(item, "quality_score", 0.5),
            final_quality=0.0,
            passed=False,
        )
        
        try:
            # Stage 1: Deduplication
            if self.config.enable_dedup:
                if not await self._deduplicate(item, result):
                    return self._finalize_result(result, start_time)
                result.stage = CleansingStage.DEDUPLICATION
            
            # Stage 2: Normalization
            if self.config.enable_normalization:
                await self._normalize(item, result)
                result.stage = CleansingStage.NORMALIZATION
            
            # Stage 3: Validation
            if self.config.enable_validation:
                if not await self._validate(item, result):
                    return self._finalize_result(result, start_time)
                result.stage = CleansingStage.VALIDATION
            
            # Stage 4: Enrichment
            if self.config.enable_enrichment:
                await self._enrich(item, result)
                result.stage = CleansingStage.ENRICHMENT
            
            # Calculate final quality
            result.final_quality = self._calculate_final_quality(item, result)
            
            # Check quality threshold
            if result.final_quality >= self.config.min_final_quality:
                result.passed = True
                result.stage = CleansingStage.READY
                self.stats.total_passed += 1
                
                # Update quality improvement stats
                improvement = result.final_quality - result.original_quality
                self.stats.avg_quality_improvement = (
                    self.stats.avg_quality_improvement * 0.9 + improvement * 0.1
                )
            else:
                result.rejection_reason = RejectionReason.LOW_QUALITY
                result.issues.append(
                    f"Quality {result.final_quality:.2f} below threshold "
                    f"{self.config.min_final_quality}"
                )
                result.stage = CleansingStage.REJECTED
                self._record_rejection(result, "quality")
            
        except Exception as e:
            result.rejection_reason = RejectionReason.PROCESSING_ERROR
            result.issues.append(f"Processing error: {str(e)}")
            result.stage = CleansingStage.REJECTED
            self._record_rejection(result, "error")
            logger.error(f"Pipeline error for {result.data_id}: {e}")
        
        return self._finalize_result(result, start_time)
    
    def _finalize_result(self, result: CleansingResult, start_time: float) -> CleansingResult:
        """Finalize result with timing."""
        import time
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        # Update average processing time
        self.stats.avg_processing_time_ms = (
            self.stats.avg_processing_time_ms * 0.9 + result.processing_time_ms * 0.1
        )
        
        return result
    
    async def _deduplicate(self, item: Any, result: CleansingResult) -> bool:
        """
        去重检查 / Deduplication check.
        """
        content = getattr(item, "content", "")
        
        is_dup, original_id = self._dedup_cache.is_duplicate(content)
        
        if is_dup:
            result.rejection_reason = RejectionReason.DUPLICATE
            result.issues.append(f"Duplicate of {original_id}")
            result.stage = CleansingStage.REJECTED
            self._record_rejection(result, "deduplication")
            return False
        
        # Add to cache
        data_id = getattr(item, "data_id", str(id(item)))
        self._dedup_cache.add(content, data_id)
        
        result.transformations.append("dedup_passed")
        return True
    
    async def _normalize(self, item: Any, result: CleansingResult):
        """
        规范化处理 / Normalization.
        """
        # Normalize content
        content = getattr(item, "content", "")
        normalized_content, transforms = self._normalizer.normalize(content)
        
        if hasattr(item, "content"):
            item.content = normalized_content
        
        result.transformations.extend(transforms)
        
        # Normalize title
        title = getattr(item, "title", "")
        if title:
            normalized_title, title_transforms = self._normalizer.normalize(title)
            if hasattr(item, "title"):
                item.title = normalized_title
            if title_transforms:
                result.transformations.append("title_normalized")
        
        result.transformations.append("normalization_complete")
    
    async def _validate(self, item: Any, result: CleansingResult) -> bool:
        """
        验证检查 / Validation.
        """
        for rule in self._validation_rules:
            passed, error = rule.validate(item)
            
            if not passed:
                # Map rule to rejection reason
                reason_map = {
                    "required_fields": RejectionReason.MISSING_FIELD,
                    "content_length": RejectionReason.TOO_SHORT,
                    "blocked_patterns": RejectionReason.BLOCKED_CONTENT,
                }
                
                result.rejection_reason = reason_map.get(
                    rule.name,
                    RejectionReason.INVALID_FORMAT
                )
                result.issues.append(error or f"Validation failed: {rule.name}")
                result.stage = CleansingStage.REJECTED
                self._record_rejection(result, "validation")
                return False
        
        result.transformations.append("validation_passed")
        return True
    
    async def _enrich(self, item: Any, result: CleansingResult):
        """
        数据增强 / Enrichment.
        """
        now = datetime.now(timezone.utc)
        
        # Add cleansing metadata
        if hasattr(item, "metadata"):
            if not isinstance(item.metadata, dict):
                item.metadata = {}
            
            item.metadata["cleaned_at"] = now.isoformat()
            item.metadata["pipeline_version"] = "2.0"
            item.metadata["cleansing_stages"] = [
                t for t in result.transformations
            ]
        
        # Mark as cleaned
        if hasattr(item, "is_cleaned"):
            item.is_cleaned = True
        
        if hasattr(item, "is_validated"):
            item.is_validated = True
        
        result.transformations.append("metadata_enriched")
    
    def _calculate_final_quality(self, item: Any, result: CleansingResult) -> float:
        """
        计算最终质量分数 / Calculate final quality score.
        """
        base_quality = result.original_quality
        
        # Bonus for successful transformations
        bonus = len(result.transformations) * self.config.quality_boost_per_transform
        
        # Penalty for issues
        penalty = len(result.issues) * self.config.quality_penalty_per_issue
        
        # Content quality factors
        content = getattr(item, "content", "")
        
        # Length bonus
        if len(content) >= 500:
            bonus += 0.05
        elif len(content) >= 200:
            bonus += 0.02
        
        # Title bonus
        title = getattr(item, "title", "")
        if title and len(title) >= 20:
            bonus += 0.02
        
        final = base_quality + bonus - penalty
        return min(1.0, max(0.0, final))
    
    def _record_rejection(self, result: CleansingResult, stage: str):
        """Record rejection statistics."""
        self.stats.total_rejected += 1
        
        # By stage
        if stage not in self.stats.rejections_by_stage:
            self.stats.rejections_by_stage[stage] = 0
        self.stats.rejections_by_stage[stage] += 1
        
        # By reason
        if result.rejection_reason:
            reason = result.rejection_reason.value
            if reason not in self.stats.rejections_by_reason:
                self.stats.rejections_by_reason[reason] = 0
            self.stats.rejections_by_reason[reason] += 1
    
    async def _flush_v2_buffer(self):
        """Flush buffer to V2 system."""
        if not self._v2_buffer or not self._v2_session:
            return
        
        items_to_send = self._v2_buffer[:self.config.v2_batch_size]
        self._v2_buffer = self._v2_buffer[self.config.v2_batch_size:]
        
        payload = {
            "source": "cleansing_pipeline",
            "items": [
                {
                    **(item.to_dict() if hasattr(item, "to_dict") else {"content": str(item)}),
                    "cleansing_result": result.to_dict(),
                }
                for item, result in items_to_send
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        try:
            async with self._v2_session.post(
                self.config.v2_push_endpoint,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status == 200:
                    logger.info(f"Pushed {len(items_to_send)} items to V2")
                else:
                    logger.warning(f"V2 push failed: {response.status}")
                    # Re-add to buffer for retry
                    self._v2_buffer.extend(items_to_send)
                    
        except Exception as e:
            logger.error(f"V2 push error: {e}")
            # Re-add to buffer for retry
            self._v2_buffer.extend(items_to_send)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息 / Get statistics."""
        stats_dict = self.stats.to_dict()
        stats_dict["cache_size"] = self._dedup_cache.size
        stats_dict["v2_buffer_size"] = len(self._v2_buffer)
        return stats_dict
    
    def clear_cache(self):
        """清除去重缓存 / Clear deduplication cache."""
        self._dedup_cache.clear()
        logger.info("Deduplication cache cleared")
    
    def add_validation_rule(self, rule: ValidationRule):
        """Add a custom validation rule."""
        self._validation_rules.append(rule)
    
    def remove_validation_rule(self, rule_name: str) -> bool:
        """Remove a validation rule by name."""
        original_len = len(self._validation_rules)
        self._validation_rules = [
            r for r in self._validation_rules if r.name != rule_name
        ]
        return len(self._validation_rules) < original_len


# =============================================================================
# Integration with Auto Learning System
# =============================================================================

async def create_integrated_pipeline(
    config: Optional[CleansingConfig] = None,
    v2_callback: Optional[Callable] = None,
) -> DataCleansingPipeline:
    """
    Factory function to create an integrated cleansing pipeline.
    
    Args:
        config: Pipeline configuration
        v2_callback: Callback when data is ready for V2
        
    Returns:
        Configured and started DataCleansingPipeline
    """
    pipeline = DataCleansingPipeline(
        config=config,
        on_data_ready_for_v2=v2_callback,
    )
    await pipeline.start()
    return pipeline
