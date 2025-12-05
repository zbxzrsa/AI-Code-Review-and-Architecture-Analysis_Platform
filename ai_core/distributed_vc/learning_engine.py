"""
Online Learning Engine

7×24 continuous learning with:
- Multiple learning channels (GitHub, blogs, papers)
- Incremental learning without service interruption
- Learning delay < 5 minutes
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import logging
import hashlib
import aiohttp
import json

logger = logging.getLogger(__name__)


class ChannelType(Enum):
    """Learning channel types"""
    GITHUB_TRENDING = "github_trending"
    GITHUB_RELEASES = "github_releases"
    TECH_BLOGS = "tech_blogs"
    PAPER_DATABASE = "paper_database"
    STACKOVERFLOW = "stackoverflow"
    DOCUMENTATION = "documentation"
    CODE_SAMPLES = "code_samples"
    CUSTOM_API = "custom_api"


class LearningStatus(Enum):
    """Learning task status"""
    PENDING = "pending"
    FETCHING = "fetching"
    PROCESSING = "processing"
    INTEGRATING = "integrating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class LearningSource:
    """A learning data source"""
    source_id: str
    channel_type: ChannelType
    name: str
    url: str
    api_key: Optional[str] = None
    fetch_interval_seconds: int = 300  # 5 minutes default
    priority: int = 1
    enabled: bool = True
    last_fetch: Optional[str] = None
    fetch_count: int = 0
    error_count: int = 0


@dataclass
class LearningItem:
    """A piece of learned knowledge"""
    item_id: str
    source_id: str
    channel_type: ChannelType
    title: str
    content: str
    url: Optional[str] = None
    timestamp: str = ""
    relevance_score: float = 0.0
    processed: bool = False
    integrated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningMetrics:
    """Metrics for learning performance"""
    total_items_fetched: int = 0
    total_items_processed: int = 0
    total_items_integrated: int = 0
    average_fetch_time_ms: float = 0.0
    average_process_time_ms: float = 0.0
    average_integration_time_ms: float = 0.0
    last_learning_delay_seconds: float = 0.0
    learning_delay_target_seconds: float = 300.0  # 5 minutes


class LearningChannel(ABC):
    """Abstract base class for learning channels"""
    
    def __init__(self, source: LearningSource):
        self.source = source
        self.session: Optional[aiohttp.ClientSession] = None
    
    def initialize(self) -> None:
        """Initialize the channel"""
        self.session = aiohttp.ClientSession()
    
    async def close(self) -> None:
        """Close the channel"""
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def fetch(self) -> List[LearningItem]:
        """Fetch learning items from the source"""
        pass
    
    @abstractmethod
    async def process(self, item: LearningItem) -> Dict[str, Any]:
        """Process a learning item"""
        pass


class GitHubTrendingChannel(LearningChannel):
    """GitHub Trending Repositories Channel"""
    
    async def fetch(self) -> List[LearningItem]:
        """Fetch trending repositories"""
        items = []
        
        try:
            # GitHub trending API (unofficial)
            url = "https://api.github.com/search/repositories"
            params = {
                "q": "created:>2024-01-01",
                "sort": "stars",
                "order": "desc",
                "per_page": 30
            }
            
            headers = {"Accept": "application/vnd.github.v3+json"}
            if self.source.api_key:
                headers["Authorization"] = f"token {self.source.api_key}"
            
            async with self.session.get(url, params=params, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    for repo in data.get("items", []):
                        item = LearningItem(
                            item_id=f"gh_{repo['id']}",
                            source_id=self.source.source_id,
                            channel_type=ChannelType.GITHUB_TRENDING,
                            title=repo["full_name"],
                            content=repo.get("description", ""),
                            url=repo["html_url"],
                            timestamp=datetime.now().isoformat(),
                            metadata={
                                "stars": repo["stargazers_count"],
                                "language": repo.get("language"),
                                "topics": repo.get("topics", [])
                            }
                        )
                        items.append(item)
                        
        except Exception as e:
            logger.error(f"GitHub fetch error: {e}")
            self.source.error_count += 1
        
        self.source.fetch_count += 1
        self.source.last_fetch = datetime.now().isoformat()
        
        return items
    
    async def process(self, item: LearningItem) -> Dict[str, Any]:
        """Process GitHub repository data"""
        # Extract patterns, technologies, best practices
        processed = {
            "repository": item.title,
            "technologies": [],
            "patterns": [],
            "quality_indicators": {}
        }
        
        # Analyze language
        if item.metadata.get("language"):
            processed["technologies"].append(item.metadata["language"])
        
        # Analyze topics
        for topic in item.metadata.get("topics", []):
            processed["technologies"].append(topic)
        
        # Quality indicators
        processed["quality_indicators"] = {
            "popularity": item.metadata.get("stars", 0),
            "relevance": item.relevance_score
        }
        
        item.processed = True
        return processed


class TechBlogChannel(LearningChannel):
    """Technical Blog Aggregation Channel"""
    
    BLOG_SOURCES = [
        {"name": "Dev.to", "url": "https://dev.to/api/articles"},
        {"name": "Hashnode", "url": "https://api.hashnode.com"},
        {"name": "Medium", "url": "https://medium.com/feed"}
    ]
    
    async def fetch(self) -> List[LearningItem]:
        """Fetch articles from tech blogs"""
        items = []
        
        try:
            # Fetch from Dev.to API
            url = "https://dev.to/api/articles"
            params = {"per_page": 20, "top": 7}  # Top articles from past week
            
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    articles = await resp.json()
                    
                    for article in articles:
                        item = LearningItem(
                            item_id=f"devto_{article['id']}",
                            source_id=self.source.source_id,
                            channel_type=ChannelType.TECH_BLOGS,
                            title=article["title"],
                            content=article.get("description", ""),
                            url=article["url"],
                            timestamp=datetime.now().isoformat(),
                            metadata={
                                "author": article.get("user", {}).get("username"),
                                "tags": article.get("tag_list", []),
                                "reactions": article.get("positive_reactions_count", 0)
                            }
                        )
                        items.append(item)
                        
        except Exception as e:
            logger.error(f"Blog fetch error: {e}")
            self.source.error_count += 1
        
        return items
    
    async def process(self, item: LearningItem) -> Dict[str, Any]:
        """Process blog article"""
        processed = {
            "article": item.title,
            "topics": item.metadata.get("tags", []),
            "author": item.metadata.get("author"),
            "engagement": item.metadata.get("reactions", 0)
        }
        
        item.processed = True
        return processed


class PaperDatabaseChannel(LearningChannel):
    """Academic Paper Database Channel (ArXiv, etc.)"""
    
    async def fetch(self) -> List[LearningItem]:
        """Fetch recent papers"""
        items = []
        
        try:
            # ArXiv API for CS papers
            url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": "cat:cs.SE+OR+cat:cs.AI+OR+cat:cs.LG",
                "start": 0,
                "max_results": 20,
                "sortBy": "submittedDate",
                "sortOrder": "descending"
            }
            
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    # Parse XML response (simplified)
                    text = await resp.text()
                    # In production, use proper XML parsing
                    
                    item = LearningItem(
                        item_id=f"arxiv_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        source_id=self.source.source_id,
                        channel_type=ChannelType.PAPER_DATABASE,
                        title="ArXiv CS Papers",
                        content=text[:1000],  # Sample
                        url="https://arxiv.org",
                        timestamp=datetime.now().isoformat()
                    )
                    items.append(item)
                    
        except Exception as e:
            logger.error(f"Paper fetch error: {e}")
            self.source.error_count += 1
        
        return items
    
    async def process(self, item: LearningItem) -> Dict[str, Any]:
        """Process academic paper"""
        processed = {
            "paper": item.title,
            "abstract": item.content[:500],
            "source": "arxiv"
        }
        
        item.processed = True
        return processed


class OnlineLearningEngine:
    """
    Online Learning Engine
    
    Features:
    - 7×24 continuous learning
    - Multiple channel support
    - Incremental updates without service interruption
    - Learning delay tracking (target: < 5 minutes)
    """
    
    def __init__(
        self,
        learning_delay_target: float = 300.0,  # 5 minutes
        max_concurrent: int = 5,
        batch_size: int = 100
    ):
        self.learning_delay_target = learning_delay_target
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        
        self.sources: Dict[str, LearningSource] = {}
        self.channels: Dict[str, LearningChannel] = {}
        self.learning_queue: asyncio.Queue = asyncio.Queue()
        self.processed_items: List[LearningItem] = []
        
        self.metrics = LearningMetrics(
            learning_delay_target_seconds=learning_delay_target
        )
        
        self.is_running = False
        self._fetch_tasks: List[asyncio.Task] = []
        self._process_task: Optional[asyncio.Task] = None
        self._integration_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.on_item_learned: Optional[Callable] = None
        self.on_batch_completed: Optional[Callable] = None
    
    def register_source(self, source: LearningSource) -> None:
        """Register a learning source"""
        self.sources[source.source_id] = source
        
        # Create appropriate channel
        channel = self._create_channel(source)
        if channel:
            self.channels[source.source_id] = channel
            logger.info(f"Registered learning source: {source.name}")
    
    def _create_channel(self, source: LearningSource) -> Optional[LearningChannel]:
        """Create channel based on type"""
        channel_map = {
            ChannelType.GITHUB_TRENDING: GitHubTrendingChannel,
            ChannelType.TECH_BLOGS: TechBlogChannel,
            ChannelType.PAPER_DATABASE: PaperDatabaseChannel,
        }
        
        channel_class = channel_map.get(source.channel_type)
        if channel_class:
            return channel_class(source)
        return None
    
    async def start(self) -> None:
        """Start the learning engine (7×24 operation)"""
        logger.info("Starting Online Learning Engine...")
        self.is_running = True
        
        # Initialize channels
        for channel in self.channels.values():
            await channel.initialize()
        
        # Start fetch tasks for each source
        for source_id, source in self.sources.items():
            if source.enabled:
                task = asyncio.create_task(
                    self._fetch_loop(source_id)
                )
                self._fetch_tasks.append(task)
        
        # Start processing task
        self._process_task = asyncio.create_task(self._process_loop())
        
        # Start integration task
        self._integration_task = asyncio.create_task(self._integration_loop())
        
        logger.info(f"Learning engine started with {len(self._fetch_tasks)} fetch tasks")
    
    async def stop(self) -> None:
        """Stop the learning engine"""
        logger.info("Stopping Online Learning Engine...")
        self.is_running = False
        
        # Cancel tasks
        for task in self._fetch_tasks:
            task.cancel()
        
        if self._process_task:
            self._process_task.cancel()
        
        if self._integration_task:
            self._integration_task.cancel()
        
        # Close channels
        for channel in self.channels.values():
            await channel.close()
        
        logger.info("Learning engine stopped")
    
    async def _fetch_loop(self, source_id: str) -> None:
        """Continuous fetch loop for a source"""
        source = self.sources[source_id]
        channel = self.channels.get(source_id)
        
        if not channel:
            return
        
        while self.is_running:
            try:
                fetch_start = datetime.now()
                
                # Fetch items
                items = await channel.fetch()
                
                fetch_time = (datetime.now() - fetch_start).total_seconds() * 1000
                self.metrics.average_fetch_time_ms = (
                    self.metrics.average_fetch_time_ms * 0.9 + fetch_time * 0.1
                )
                
                # Add to queue
                for item in items:
                    await self.learning_queue.put(item)
                    self.metrics.total_items_fetched += 1
                
                logger.info(f"Fetched {len(items)} items from {source.name}")
                
            except Exception as e:
                logger.error(f"Fetch error for {source_id}: {e}")
            
            # Wait for next fetch
            await asyncio.sleep(source.fetch_interval_seconds)
    
    async def _process_loop(self) -> None:
        """Continuous processing loop"""
        while self.is_running:
            try:
                # Get item from queue with timeout
                item = await asyncio.wait_for(
                    self.learning_queue.get(),
                    timeout=5.0
                )
                
                process_start = datetime.now()
                
                # Process item
                channel = self.channels.get(item.source_id)
                if channel:
                    processed_data = await channel.process(item)
                    item.relevance_score = self._calculate_relevance(item, processed_data)
                    self.processed_items.append(item)
                    self.metrics.total_items_processed += 1
                
                process_time = (datetime.now() - process_start).total_seconds() * 1000
                self.metrics.average_process_time_ms = (
                    self.metrics.average_process_time_ms * 0.9 + process_time * 0.1
                )
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Process error: {e}")
    
    async def _integration_loop(self) -> None:
        """Incremental integration loop"""
        while self.is_running:
            try:
                # Wait for batch
                await asyncio.sleep(60)  # Check every minute
                
                # Get unintegrated items
                to_integrate = [
                    item for item in self.processed_items
                    if item.processed and not item.integrated
                ][:self.batch_size]
                
                if to_integrate:
                    integration_start = datetime.now()
                    
                    # Perform incremental integration
                    await self._integrate_batch(to_integrate)
                    
                    integration_time = (datetime.now() - integration_start).total_seconds() * 1000
                    self.metrics.average_integration_time_ms = (
                        self.metrics.average_integration_time_ms * 0.9 + integration_time * 0.1
                    )
                    
                    # Calculate learning delay
                    oldest_item = min(to_integrate, key=lambda x: x.timestamp)
                    delay = (datetime.now() - datetime.fromisoformat(oldest_item.timestamp)).total_seconds()
                    self.metrics.last_learning_delay_seconds = delay
                    
                    logger.info(
                        f"Integrated {len(to_integrate)} items, "
                        f"delay: {delay:.1f}s (target: {self.learning_delay_target}s)"
                    )
                    
                    # Callback
                    if self.on_batch_completed:
                        await self.on_batch_completed(to_integrate)
                
            except Exception as e:
                logger.error(f"Integration error: {e}")
    
    async def _integrate_batch(self, items: List[LearningItem]) -> None:
        """Integrate a batch of learned items (incremental learning)"""
        # Sort by relevance
        sorted_items = sorted(items, key=lambda x: x.relevance_score, reverse=True)
        
        for item in sorted_items:
            # Mark as integrated
            item.integrated = True
            self.metrics.total_items_integrated += 1
            
            # Callback
            if self.on_item_learned:
                await self.on_item_learned(item)
    
    def _calculate_relevance(
        self,
        item: LearningItem,
        processed_data: Dict[str, Any]
    ) -> float:
        """Calculate relevance score for an item"""
        score = 0.5  # Base score
        
        # Boost for popularity
        if "popularity" in processed_data.get("quality_indicators", {}):
            pop = processed_data["quality_indicators"]["popularity"]
            score += min(0.3, pop / 10000)
        
        # Boost for relevant technologies
        tech_boost = 0
        for tech in processed_data.get("technologies", []):
            if tech.lower() in ["python", "ai", "machine-learning", "deep-learning"]:
                tech_boost += 0.05
        score += min(0.2, tech_boost)
        
        return min(1.0, score)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get learning metrics"""
        return {
            "total_items_fetched": self.metrics.total_items_fetched,
            "total_items_processed": self.metrics.total_items_processed,
            "total_items_integrated": self.metrics.total_items_integrated,
            "queue_size": self.learning_queue.qsize(),
            "average_fetch_time_ms": self.metrics.average_fetch_time_ms,
            "average_process_time_ms": self.metrics.average_process_time_ms,
            "average_integration_time_ms": self.metrics.average_integration_time_ms,
            "last_learning_delay_seconds": self.metrics.last_learning_delay_seconds,
            "learning_delay_target_seconds": self.metrics.learning_delay_target_seconds,
            "meets_delay_target": (
                self.metrics.last_learning_delay_seconds <= self.metrics.learning_delay_target_seconds
            ),
            "sources": {
                source_id: {
                    "name": source.name,
                    "enabled": source.enabled,
                    "fetch_count": source.fetch_count,
                    "error_count": source.error_count,
                    "last_fetch": source.last_fetch
                }
                for source_id, source in self.sources.items()
            }
        }
    
    def get_recent_learnings(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recently learned items"""
        recent = sorted(
            self.processed_items,
            key=lambda x: x.timestamp,
            reverse=True
        )[:limit]
        
        return [
            {
                "item_id": item.item_id,
                "title": item.title,
                "channel": item.channel_type.value,
                "relevance": item.relevance_score,
                "integrated": item.integrated,
                "timestamp": item.timestamp
            }
            for item in recent
        ]
