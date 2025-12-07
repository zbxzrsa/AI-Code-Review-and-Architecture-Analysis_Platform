"""
Base Collector Interface

Provides the abstract base class for all data source collectors.
"""

import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional

import aiohttp

from ..config import CollectionSchedule, DataSourceConfig, DataSourcePriority

logger = logging.getLogger(__name__)


class ContentType(str, Enum):
    """Type of collected content."""
    CODE = "code"
    DOCUMENTATION = "documentation"
    PAPER = "paper"
    ARTICLE = "article"
    TUTORIAL = "tutorial"
    API_REFERENCE = "api_reference"
    UNKNOWN = "unknown"


@dataclass
class CollectedItem:
    """
    Represents a single collected item from a data source.
    
    Attributes:
        source: Data source name (github, arxiv, etc.)
        source_id: Unique identifier within the source
        url: Original URL of the content
        title: Title of the content
        content: Raw content text
        content_type: Type of content
        language: Programming language (if applicable)
        tags: Associated tags/topics
        author: Content author
        created_at: Original creation time
        updated_at: Last update time
        collected_at: Time of collection
        metadata: Additional source-specific metadata
    """
    source: str
    source_id: str
    url: str
    title: str
    content: str
    content_type: ContentType = ContentType.UNKNOWN
    language: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    collected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def content_hash(self) -> str:
        """Generate hash of content for deduplication."""
        return hashlib.sha256(self.content.encode()).hexdigest()
    
    @property
    def unique_id(self) -> str:
        """Generate unique identifier across all sources."""
        return f"{self.source}:{self.source_id}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source": self.source,
            "source_id": self.source_id,
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "content_type": self.content_type.value,
            "language": self.language,
            "tags": self.tags,
            "author": self.author,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "collected_at": self.collected_at.isoformat(),
            "content_hash": self.content_hash,
            "metadata": self.metadata,
        }


@dataclass
class CollectionResult:
    """
    Result of a collection cycle.
    
    Attributes:
        source: Data source name
        success: Whether collection was successful
        items_collected: Number of items collected
        items_filtered: Number of items filtered out
        errors: List of error messages
        duration_seconds: Time taken for collection
        rate_limit_remaining: Remaining rate limit
    """
    source: str
    success: bool
    items_collected: int = 0
    items_filtered: int = 0
    items_deduplicated: int = 0
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    rate_limit_remaining: Optional[int] = None
    next_collection_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/metrics."""
        return {
            "source": self.source,
            "success": self.success,
            "items_collected": self.items_collected,
            "items_filtered": self.items_filtered,
            "items_deduplicated": self.items_deduplicated,
            "errors": self.errors,
            "duration_seconds": self.duration_seconds,
            "rate_limit_remaining": self.rate_limit_remaining,
        }


class BaseCollector(ABC):
    """
    Abstract base class for data source collectors.
    
    Subclasses must implement:
    - collect(): Perform collection from the source
    - _parse_item(): Parse raw response into CollectedItem
    
    Features:
    - Async HTTP client with connection pooling
    - Automatic rate limiting
    - Retry with exponential backoff
    - Progress tracking
    """
    
    def __init__(
        self,
        config: DataSourceConfig,
        schedule: CollectionSchedule,
    ):
        self.config = config
        self.schedule = schedule
        self.priority = config.priority
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_remaining = config.rate_limit
        self._last_request_time: Optional[datetime] = None
        self._consecutive_errors = 0
        
    @property
    def name(self) -> str:
        """Collector name."""
        return self.config.name
    
    @property
    def is_rate_limited(self) -> bool:
        """Check if rate limit is exhausted."""
        return self._rate_limit_remaining <= 0
    
    async def start(self):
        """Initialize the collector."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.schedule.timeout_seconds)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self._get_default_headers(),
            )
        logger.info(f"Started collector: {self.name}")
    
    async def stop(self):
        """Cleanup collector resources."""
        if self._session:
            await self._session.close()
            self._session = None
        logger.info(f"Stopped collector: {self.name}")
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for requests."""
        headers = {
            "User-Agent": "AI-Code-Review-Platform/1.0",
            "Accept": "application/json",
        }
        headers.update(self.config.custom_headers)
        return headers
    
    async def _make_request(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """
        Make HTTP request with retry and rate limiting.
        
        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Additional request arguments
            
        Returns:
            Response JSON or None on failure
        """
        if self._session is None:
            await self.start()
        
        for attempt in range(self.schedule.retry_attempts):
            try:
                # Rate limiting
                await self._wait_for_rate_limit()
                
                async with self._session.request(method, url, **kwargs) as response:
                    # Update rate limit from headers
                    self._update_rate_limit(response.headers)
                    
                    if response.status == 200:
                        self._consecutive_errors = 0
                        return await response.json()
                    elif response.status == 429:
                        # Rate limited - wait and retry
                        retry_after = int(response.headers.get("Retry-After", 60))
                        logger.warning(f"{self.name}: Rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                    elif response.status >= 500:
                        # Server error - retry with backoff
                        wait_time = (self.schedule.backoff_multiplier ** attempt) * 1
                        logger.warning(f"{self.name}: Server error {response.status}, retry in {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"{self.name}: Request failed with status {response.status}")
                        return None
                        
            except asyncio.TimeoutError:
                logger.warning(f"{self.name}: Request timeout (attempt {attempt + 1})")
            except aiohttp.ClientError as e:
                logger.error(f"{self.name}: Client error: {e}")
            except Exception as e:
                logger.error(f"{self.name}: Unexpected error: {e}")
            
            self._consecutive_errors += 1
        
        return None
    
    async def _wait_for_rate_limit(self):
        """Wait if rate limit would be exceeded."""
        if self._rate_limit_remaining <= 0:
            # Wait until rate limit resets (simplified)
            await asyncio.sleep(60)
            self._rate_limit_remaining = self.config.rate_limit
    
    def _update_rate_limit(self, headers: Dict[str, str]):
        """Update rate limit from response headers."""
        if "X-RateLimit-Remaining" in headers:
            self._rate_limit_remaining = int(headers["X-RateLimit-Remaining"])
        elif "RateLimit-Remaining" in headers:
            self._rate_limit_remaining = int(headers["RateLimit-Remaining"])
    
    @abstractmethod
    async def collect(
        self,
        since: Optional[datetime] = None,
        max_items: Optional[int] = None,
    ) -> AsyncIterator[CollectedItem]:
        """
        Collect items from the data source.
        
        Args:
            since: Only collect items updated after this time
            max_items: Maximum number of items to collect
            
        Yields:
            CollectedItem instances
        """
        pass
    
    @abstractmethod
    def _parse_item(self, raw_data: Dict[str, Any]) -> Optional[CollectedItem]:
        """
        Parse raw API response into CollectedItem.
        
        Args:
            raw_data: Raw data from API response
            
        Returns:
            Parsed CollectedItem or None if parsing fails
        """
        pass
    
    async def run_collection_cycle(
        self,
        since: Optional[datetime] = None,
    ) -> CollectionResult:
        """
        Run a complete collection cycle.
        
        Args:
            since: Only collect items updated after this time
            
        Returns:
            CollectionResult with statistics
        """
        import time
        
        start_time = time.time()
        result = CollectionResult(source=self.name, success=True)
        max_items = self.schedule.max_items_per_cycle
        
        try:
            async for item in self.collect(since=since, max_items=max_items):
                result.items_collected += 1
                
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            logger.error(f"{self.name}: Collection failed: {e}")
        
        result.duration_seconds = time.time() - start_time
        result.rate_limit_remaining = self._rate_limit_remaining
        
        logger.info(
            f"{self.name}: Collected {result.items_collected} items "
            f"in {result.duration_seconds:.2f}s"
        )
        
        return result
