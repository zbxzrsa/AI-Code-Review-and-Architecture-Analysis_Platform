"""
V1/V3 自动网络学习系统 (V1/V3 Auto Network Learning System)

模块功能描述:
    实现 7×24 小时持续自动学习，从多个数据源并行获取知识，
    并将高质量数据集成到 V2 生产系统。

主要功能:
    - 7×24 小时持续学习
    - 多数据源并行获取
    - 智能速率限制
    - 自动重试和容错
    - 质量过滤和清洗
    - V2 系统集成

支持的数据源:
    - GitHub: 趋势仓库、版本发布、代码示例
    - ArXiv: 计算机科学/AI/机器学习研究论文
    - Dev.to: 技术文章
    - Hashnode: 开发者博客
    - StackOverflow: 问答知识
    - HuggingFace: 机器学习模型和数据集

最后修改日期: 2024-12-07
"""

import asyncio
import aiohttp
import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import logging
import json

# Import centralized configuration for secure API key management
try:
    from backend.shared.config.config_manager import get_config
    _config = get_config()
except ImportError:
    # Fallback for standalone usage - will use environment variables directly
    _config = None

logger = logging.getLogger(__name__)


def _get_data_source_api_key(source_name: str) -> str:
    """
    Securely retrieve API key for a data source.
    
    Uses centralized configuration when available, falls back to
    environment variables for standalone usage.
    
    Args:
        source_name: Name of the data source (e.g., 'github', 'huggingface')
        
    Returns:
        API key string or empty string if not configured
    """
    import os  # Local import for fallback case
    
    if _config is not None:
        # Use centralized secure configuration
        key = _config.data_sources.get_api_key(source_name)
        if key:
            logger.debug(f"Retrieved {source_name} API key from secure config")
            return key
        logger.debug(f"No {source_name} API key configured in secure config")
        return ""
    
    # Fallback to environment variables for standalone usage
    env_var_map = {
        "github": "GITHUB_TOKEN",
        "huggingface": "HUGGINGFACE_TOKEN",
        "stackoverflow": "STACKOVERFLOW_KEY",
        "devto": "DEVTO_API_KEY",
    }
    
    env_var = env_var_map.get(source_name.lower())
    if env_var:
        key = os.getenv(env_var, "")
        if key:
            logger.debug(f"Retrieved {source_name} API key from environment variable {env_var}")
        else:
            logger.debug(f"No {source_name} API key found in environment variable {env_var}")
        return key
    
    logger.warning(f"Unknown data source for API key retrieval: {source_name}")
    return ""


# =============================================================================
# Data Models
# =============================================================================

class DataSource(Enum):
    """
    学习数据源枚举
    
    定义系统支持的所有学习数据源类型。
    
    数据源类型:
        - GITHUB_TRENDING: GitHub 趋势仓库
        - GITHUB_RELEASES: GitHub 版本发布
        - GITHUB_DOCS: GitHub 文档
        - ARXIV_CS/AI/SE: ArXiv 论文
        - DEV_TO: Dev.to 文章
        - HASHNODE: Hashnode 博客
        - STACKOVERFLOW: Stack Overflow 问答
        - HUGGINGFACE: HuggingFace 模型
        - MEDIUM: Medium 文章
        - HACKERNEWS: Hacker News
    """
    GITHUB_TRENDING = "github_trending"
    GITHUB_RELEASES = "github_releases"
    GITHUB_DOCS = "github_docs"
    ARXIV_CS = "arxiv_cs"
    ARXIV_AI = "arxiv_ai"
    ARXIV_SE = "arxiv_se"
    DEV_TO = "dev_to"
    HASHNODE = "hashnode"
    STACKOVERFLOW = "stackoverflow"
    HUGGINGFACE = "huggingface"
    MEDIUM = "medium"
    HACKERNEWS = "hackernews"


class LearningStatus(Enum):
    """
    学习状态枚举
    
    表示学习数据项的当前处理状态。
    
    状态说明:
        - PENDING: 等待处理
        - FETCHING: 正在获取
        - CLEANING: 正在清洗
        - VALIDATING: 正在验证
        - READY: 已就绪
        - INTEGRATED: 已集成
        - FAILED: 失败
    """
    PENDING = "pending"
    FETCHING = "fetching"
    CLEANING = "cleaning"
    VALIDATING = "validating"
    READY = "ready"
    INTEGRATED = "integrated"
    FAILED = "failed"


@dataclass
class LearningData:
    """
    学习数据项数据类
    
    功能描述:
        表示从任意数据源学习到的单条内容。
    
    属性说明:
        - data_id: 数据唯一标识符
        - source: 数据源类型
        - title: 标题
        - content: 内容
        - url: 原始链接
        - quality_score: 质量评分
        - is_cleaned: 是否已清洗
        - is_validated: 是否已验证
        - status: 处理状态
    """
    data_id: str
    source: DataSource
    title: str
    content: str
    url: str
    fetched_at: datetime
    quality_score: float = 0.0
    is_cleaned: bool = False
    is_validated: bool = False
    status: LearningStatus = LearningStatus.PENDING
    language: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def content_hash(self) -> str:
        """Generate hash for deduplication."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "data_id": self.data_id,
            "source": self.source.value,
            "title": self.title,
            "content": self.content[:1000],  # Truncate for serialization
            "url": self.url,
            "fetched_at": self.fetched_at.isoformat(),
            "quality_score": self.quality_score,
            "status": self.status.value,
            "language": self.language,
            "tags": self.tags,
            "metadata": self.metadata,
        }


@dataclass
class NetworkLearningConfig:
    """
    联网学习配置 / Network Learning Configuration
    
    Comprehensive configuration for the auto-learning system.
    """
    # 学习间隔 / Learning Intervals
    v1_learning_interval_minutes: int = 30
    v3_learning_interval_minutes: int = 60
    
    # 质量阈值 / Quality Thresholds
    min_quality_for_v2: float = 0.7  # Minimum quality to push to V2
    min_quality_for_retention: float = 0.5  # Minimum quality to keep
    
    # 速率限制 / Rate Limiting
    max_requests_per_hour: int = 100
    max_concurrent_sources: int = 3
    request_timeout_seconds: int = 30
    
    # 重试配置 / Retry Configuration
    max_retries: int = 3
    retry_backoff_seconds: float = 30.0
    exponential_backoff: bool = True
    
    # 内容过滤 / Content Filtering
    min_content_length: int = 100
    max_content_length: int = 100000
    required_languages: List[str] = field(default_factory=lambda: [
        "python", "javascript", "typescript", "go", "rust", "java"
    ])
    
    # 数据源配置 / Data Source Configuration
    enabled_sources: List[DataSource] = field(default_factory=lambda: [
        DataSource.GITHUB_TRENDING,
        DataSource.ARXIV_CS,
        DataSource.ARXIV_AI,
        DataSource.DEV_TO,
        DataSource.HACKERNEWS,
    ])
    
    # 数据源优先级 / Source Priorities (1 = highest)
    source_priorities: Dict[str, int] = field(default_factory=lambda: {
        "github_trending": 1,
        "arxiv_cs": 2,
        "arxiv_ai": 2,
        "dev_to": 3,
        "hackernews": 3,
        "stackoverflow": 4,
        "huggingface": 4,
    })
    
    # V2 集成 / V2 Integration
    v2_push_enabled: bool = True
    v2_push_endpoint: str = "/api/v2/learning/ingest"
    v2_push_batch_size: int = 50
    
    # 存储配置 / Storage Configuration
    max_items_in_memory: int = 10000
    persist_to_disk: bool = True
    storage_path: str = "/data/learning"


# =============================================================================
# Rate Limiting
# =============================================================================

class AsyncRateLimiter:
    """
    异步速率限制器 / Async Rate Limiter
    
    Implements sliding window rate limiting with async support.
    """
    
    def __init__(self, max_requests: int, period_seconds: int = 3600):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed in period
            period_seconds: Time window in seconds (default: 1 hour)
        """
        self.max_requests = max_requests
        self.period = period_seconds
        self.requests: List[datetime] = []
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """
        Try to acquire a request slot.
        
        Returns:
            True if slot acquired, False if rate limited
        """
        async with self._lock:
            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(seconds=self.period)
            
            # Remove expired requests
            self.requests = [r for r in self.requests if r > cutoff]
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False
    
    async def wait_and_acquire(self, timeout: float = 60.0) -> bool:
        """
        Wait until a slot is available.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if acquired, False if timeout
        """
        start = datetime.now(timezone.utc)
        while (datetime.now(timezone.utc) - start).total_seconds() < timeout:
            if await self.acquire():
                return True
            await asyncio.sleep(1)
        return False
    
    def get_remaining(self) -> int:
        """Get remaining request slots."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self.period)
        active = len([r for r in self.requests if r > cutoff])
        return max(0, self.max_requests - active)
    
    def get_reset_time(self) -> Optional[datetime]:
        """Get when the oldest request will expire."""
        if not self.requests:
            return None
        oldest = min(self.requests)
        return oldest + timedelta(seconds=self.period)


# =============================================================================
# Data Source Connectors
# =============================================================================

class DataSourceConnector(ABC):
    """
    数据源连接器基类 / Base Data Source Connector
    
    Abstract base class for all data source connectors.
    """
    
    def __init__(
        self,
        source: DataSource,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        self.source = source
        self.api_key = api_key
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.fetch_count = 0
        self.error_count = 0
        self.last_fetch: Optional[datetime] = None
    
    async def connect(self):
        """Initialize HTTP session."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def disconnect(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    @abstractmethod
    async def fetch(self, limit: int = 50) -> List[LearningData]:
        """
        Fetch data from the source.
        
        Args:
            limit: Maximum items to fetch
            
        Returns:
            List of LearningData items
        """
        pass
    
    def _get_headers(self) -> Dict[str, str]:
        """Get default headers."""
        return {
            "User-Agent": "AI-Code-Review-Platform/1.0",
            "Accept": "application/json",
        }
    
    def _record_fetch(self, success: bool):
        """Record fetch attempt."""
        self.fetch_count += 1
        self.last_fetch = datetime.now(timezone.utc)
        if not success:
            self.error_count += 1


class GitHubConnector(DataSourceConnector):
    """
    GitHub数据源连接器 / GitHub Data Source Connector
    
    Fetches trending repositories, releases, and documentation.
    """
    
    API_BASE = "https://api.github.com"
    
    def _get_headers(self) -> Dict[str, str]:
        headers = super()._get_headers()
        headers["Accept"] = "application/vnd.github.v3+json"
        
        # Get auth token from secure configuration
        # Priority: instance api_key > centralized config > environment variable
        token = self.api_key or _get_data_source_api_key("github")
        if token:
            headers["Authorization"] = f"token {token}"
            logger.debug("GitHub API request with authentication")
        else:
            logger.debug("GitHub API request without authentication (rate limits may apply)")
        
        return headers
    
    async def fetch(self, limit: int = 50) -> List[LearningData]:
        """Fetch trending repositories from GitHub."""
        items = []
        
        try:
            url = f"{self.API_BASE}/search/repositories"
            params = {
                "q": "stars:>1000 pushed:>2024-01-01",
                "sort": "updated",
                "order": "desc",
                "per_page": min(limit, 100)
            }
            
            async with self.session.get(
                url,
                params=params,
                headers=self._get_headers()
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    for repo in data.get("items", []):
                        items.append(LearningData(
                            data_id=f"gh_{repo['id']}",
                            source=self.source,
                            title=repo["full_name"],
                            content=repo.get("description", "") or "",
                            url=repo["html_url"],
                            fetched_at=datetime.now(timezone.utc),
                            language=repo.get("language"),
                            tags=repo.get("topics", []),
                            metadata={
                                "stars": repo["stargazers_count"],
                                "forks": repo["forks_count"],
                                "watchers": repo["watchers_count"],
                                "open_issues": repo["open_issues_count"],
                                "license": repo.get("license", {}).get("name") if repo.get("license") else None,
                            }
                        ))
                    
                    self._record_fetch(True)
                else:
                    logger.warning(f"GitHub API error: {resp.status}")
                    self._record_fetch(False)
                    
        except Exception as e:
            logger.error(f"GitHub fetch error: {e}")
            self._record_fetch(False)
        
        return items


class ArxivConnector(DataSourceConnector):
    """
    ArXiv数据源连接器 / ArXiv Data Source Connector
    
    Fetches academic papers from ArXiv.
    """
    
    API_BASE = "https://export.arxiv.org/api/query"
    NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}
    
    CATEGORY_MAP = {
        DataSource.ARXIV_CS: "cat:cs.SE OR cat:cs.PL",
        DataSource.ARXIV_AI: "cat:cs.AI OR cat:cs.LG OR cat:cs.CL",
        DataSource.ARXIV_SE: "cat:cs.SE",
    }
    
    async def fetch(self, limit: int = 50) -> List[LearningData]:
        """Fetch papers from ArXiv."""
        items = []
        
        try:
            category = self.CATEGORY_MAP.get(self.source, "cat:cs.AI")
            params = {
                "search_query": category,
                "start": 0,
                "max_results": min(limit, 100),
                "sortBy": "submittedDate",
                "sortOrder": "descending"
            }
            
            async with self.session.get(self.API_BASE, params=params) as resp:
                if resp.status == 200:
                    xml_text = await resp.text()
                    items = self._parse_arxiv_response(xml_text)
                    self._record_fetch(True)
                else:
                    logger.warning(f"ArXiv API error: {resp.status}")
                    self._record_fetch(False)
                    
        except Exception as e:
            logger.error(f"ArXiv fetch error: {e}")
            self._record_fetch(False)
        
        return items
    
    def _parse_arxiv_response(self, xml_text: str) -> List[LearningData]:
        """Parse ArXiv XML response."""
        items = []
        
        try:
            root = ET.fromstring(xml_text)
            
            for entry in root.findall("atom:entry", self.NAMESPACE):
                arxiv_id = entry.find("atom:id", self.NAMESPACE)
                title = entry.find("atom:title", self.NAMESPACE)
                summary = entry.find("atom:summary", self.NAMESPACE)
                published = entry.find("atom:published", self.NAMESPACE)
                
                if arxiv_id is None or title is None:
                    continue
                
                # Extract ID from URL
                id_text = arxiv_id.text.split("/")[-1] if arxiv_id.text else ""
                
                # Get authors
                authors = []
                for author in entry.findall("atom:author", self.NAMESPACE):
                    name = author.find("atom:name", self.NAMESPACE)
                    if name is not None and name.text:
                        authors.append(name.text)
                
                # Get categories
                categories = []
                for cat in entry.findall("atom:category", self.NAMESPACE):
                    term = cat.get("term")
                    if term:
                        categories.append(term)
                
                items.append(LearningData(
                    data_id=f"arxiv_{id_text}",
                    source=self.source,
                    title=" ".join((title.text or "").split()),
                    content=" ".join((summary.text or "").split()),
                    url=f"https://arxiv.org/abs/{id_text}",
                    fetched_at=datetime.now(timezone.utc),
                    tags=categories,
                    metadata={
                        "authors": authors[:5],
                        "author_count": len(authors),
                        "primary_category": categories[0] if categories else None,
                    }
                ))
                
        except ET.ParseError as e:
            logger.error(f"ArXiv XML parse error: {e}")
        
        return items


class DevToConnector(DataSourceConnector):
    """
    Dev.to数据源连接器 / Dev.to Data Source Connector
    
    Fetches technical articles from Dev.to.
    """
    
    API_BASE = "https://dev.to/api/articles"
    
    async def fetch(self, limit: int = 50) -> List[LearningData]:
        """Fetch articles from Dev.to."""
        items = []
        
        try:
            params = {
                "per_page": min(limit, 100),
                "top": 7,  # Last 7 days
            }
            
            async with self.session.get(
                self.API_BASE,
                params=params,
                headers=self._get_headers()
            ) as resp:
                if resp.status == 200:
                    articles = await resp.json()
                    
                    for article in articles:
                        items.append(LearningData(
                            data_id=f"devto_{article['id']}",
                            source=self.source,
                            title=article.get("title", ""),
                            content=article.get("description", "") or article.get("body_markdown", "")[:2000],
                            url=article.get("url", ""),
                            fetched_at=datetime.now(timezone.utc),
                            tags=article.get("tag_list", []),
                            metadata={
                                "reactions": article.get("public_reactions_count", 0),
                                "comments": article.get("comments_count", 0),
                                "reading_time": article.get("reading_time_minutes"),
                                "author": article.get("user", {}).get("username"),
                            }
                        ))
                    
                    self._record_fetch(True)
                else:
                    logger.warning(f"Dev.to API error: {resp.status}")
                    self._record_fetch(False)
                    
        except Exception as e:
            logger.error(f"Dev.to fetch error: {e}")
            self._record_fetch(False)
        
        return items


class HackerNewsConnector(DataSourceConnector):
    """
    HackerNews数据源连接器 / HackerNews Data Source Connector
    
    Fetches top stories from Hacker News.
    """
    
    API_BASE = "https://hacker-news.firebaseio.com/v0"
    
    async def fetch(self, limit: int = 50) -> List[LearningData]:
        """
        Fetch top stories from Hacker News.
        
        Optimized: Uses asyncio.gather for parallel fetching (~5x faster).
        """
        items = []
        
        try:
            # Get top story IDs
            async with self.session.get(f"{self.API_BASE}/topstories.json") as resp:
                if resp.status != 200:
                    self._record_fetch(False)
                    return items
                
                story_ids = await resp.json()
            
            # Parallel fetch with asyncio.gather (P0 optimization)
            async def fetch_story(story_id: int) -> Optional[LearningData]:
                """Fetch a single story."""
                try:
                    async with self.session.get(
                        f"{self.API_BASE}/item/{story_id}.json"
                    ) as resp:
                        if resp.status == 200:
                            story = await resp.json()
                            if story and self._is_tech_related(story):
                                return LearningData(
                                    data_id=f"hn_{story_id}",
                                    source=self.source,
                                    title=story.get("title", ""),
                                    content=story.get("text", "") or story.get("title", ""),
                                    url=story.get("url", f"https://news.ycombinator.com/item?id={story_id}"),
                                    fetched_at=datetime.now(timezone.utc),
                                    tags=["hackernews"],
                                    metadata={
                                        "score": story.get("score", 0),
                                        "comments": story.get("descendants", 0),
                                        "author": story.get("by"),
                                        "type": story.get("type"),
                                    }
                                )
                except Exception as e:
                    logger.debug(f"Skipping HackerNews story {story_id}: {type(e).__name__}")
                return None
            
            # Fetch all stories in parallel with concurrency limit
            batch_size = 10  # Limit concurrent requests to avoid rate limiting
            for i in range(0, min(len(story_ids), limit), batch_size):
                batch_ids = story_ids[i:i + batch_size]
                results = await asyncio.gather(
                    *[fetch_story(sid) for sid in batch_ids],
                    return_exceptions=True
                )
                for result in results:
                    if isinstance(result, LearningData):
                        items.append(result)
            
            self._record_fetch(True)
            
        except Exception as e:
            logger.error(f"HackerNews fetch error: {e}")
            self._record_fetch(False)
        
        return items
    
    def _is_tech_related(self, story: Dict[str, Any]) -> bool:
        """Check if story is tech-related."""
        title = (story.get("title") or "").lower()
        url = story.get("url") or ""
        
        tech_keywords = [
            "python", "javascript", "rust", "go", "java",
            "programming", "coding", "developer", "software",
            "api", "database", "algorithm", "machine learning",
            "ai", "open source", "github", "framework",
            "kubernetes", "docker", "cloud", "security",
        ]
        
        for keyword in tech_keywords:
            if keyword in title:
                return True
        
        tech_domains = ["github.com", "gitlab.com", "dev.to", "medium.com"]
        for domain in tech_domains:
            if domain in url:
                return True
        
        return False


class HuggingFaceConnector(DataSourceConnector):
    """
    HuggingFace数据源连接器 / HuggingFace Data Source Connector
    
    Fetches trending models and datasets.
    """
    
    API_BASE = "https://huggingface.co/api"
    
    async def fetch(self, limit: int = 50) -> List[LearningData]:
        """Fetch trending models from HuggingFace."""
        items = []
        
        try:
            params = {
                "sort": "downloads",
                "direction": -1,
                "limit": min(limit, 100),
            }
            
            async with self.session.get(
                f"{self.API_BASE}/models",
                params=params,
                headers=self._get_headers()
            ) as resp:
                if resp.status == 200:
                    models = await resp.json()
                    
                    for model in models:
                        model_id = model.get("modelId", model.get("id", ""))
                        items.append(LearningData(
                            data_id=f"hf_{model_id.replace('/', '_')}",
                            source=self.source,
                            title=model_id,
                            content=model.get("description", "") or f"HuggingFace model: {model_id}",
                            url=f"https://huggingface.co/{model_id}",
                            fetched_at=datetime.now(timezone.utc),
                            tags=model.get("tags", []),
                            metadata={
                                "downloads": model.get("downloads", 0),
                                "likes": model.get("likes", 0),
                                "pipeline_tag": model.get("pipeline_tag"),
                                "library_name": model.get("library_name"),
                            }
                        ))
                    
                    self._record_fetch(True)
                else:
                    logger.warning(f"HuggingFace API error: {resp.status}")
                    self._record_fetch(False)
                    
        except Exception as e:
            logger.error(f"HuggingFace fetch error: {e}")
            self._record_fetch(False)
        
        return items


# =============================================================================
# Quality Assessment
# =============================================================================

class QualityAssessor:
    """
    质量评估器 / Quality Assessor
    
    Evaluates and scores learning data quality.
    """
    
    TECH_KEYWORDS = {
        "algorithm", "api", "architecture", "async", "backend",
        "cache", "class", "code", "compile", "concurrency",
        "container", "database", "debug", "deploy", "docker",
        "framework", "function", "git", "http", "interface",
        "kubernetes", "library", "linux", "microservice", "module",
        "network", "optimization", "package", "parallel", "pattern",
        "performance", "protocol", "python", "query", "refactor",
        "rest", "security", "server", "sql", "syntax", "test",
        "thread", "type", "variable", "version", "virtual",
    }
    
    SOURCE_TRUST = {
        DataSource.GITHUB_TRENDING: 0.2,
        DataSource.GITHUB_RELEASES: 0.2,
        DataSource.ARXIV_CS: 0.25,
        DataSource.ARXIV_AI: 0.25,
        DataSource.DEV_TO: 0.15,
        DataSource.HACKERNEWS: 0.15,
        DataSource.HUGGINGFACE: 0.2,
        DataSource.STACKOVERFLOW: 0.15,
    }
    
    def __init__(self, config: NetworkLearningConfig):
        self.config = config
    
    def assess(self, item: LearningData) -> float:
        """
        Assess quality of a learning data item.
        
        Args:
            item: LearningData to assess
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.0
        
        # Content length score (0-0.25)
        score += self._score_content_length(item)
        
        # Technical relevance (0-0.3)
        score += self._score_technical_relevance(item)
        
        # Title quality (0-0.15)
        score += self._score_title(item)
        
        # Source trust (0-0.25)
        score += self.SOURCE_TRUST.get(item.source, 0.1)
        
        # Metadata bonus (0-0.05)
        score += self._score_metadata(item)
        
        return min(1.0, max(0.0, score))
    
    def _score_content_length(self, item: LearningData) -> float:
        """Score based on content length."""
        length = len(item.content)
        
        if length < self.config.min_content_length:
            return 0.0
        elif length >= 1000:
            return 0.25
        elif length >= 500:
            return 0.2
        elif length >= 200:
            return 0.15
        else:
            return 0.1
    
    def _score_technical_relevance(self, item: LearningData) -> float:
        """Score based on technical keyword presence."""
        content_lower = (item.content + " " + item.title).lower()
        
        keyword_count = sum(1 for kw in self.TECH_KEYWORDS if kw in content_lower)
        
        # Normalize: 10+ keywords = max score
        return min(0.3, keyword_count * 0.03)
    
    def _score_title(self, item: LearningData) -> float:
        """Score based on title quality."""
        if not item.title:
            return 0.0
        
        title_len = len(item.title)
        
        if title_len > 50:
            return 0.15
        elif title_len > 20:
            return 0.1
        elif title_len > 10:
            return 0.05
        else:
            return 0.0
    
    def _score_metadata(self, item: LearningData) -> float:
        """Score based on metadata quality."""
        score = 0.0
        
        # Has URL
        if item.url:
            score += 0.01
        
        # Has tags
        if item.tags:
            score += 0.02
        
        # Has language
        if item.language:
            score += 0.01
        
        # Has rich metadata
        if item.metadata:
            score += 0.01
        
        return score
    
    def batch_assess(self, items: List[LearningData]) -> List[Tuple[LearningData, float]]:
        """Assess multiple items."""
        return [(item, self.assess(item)) for item in items]


# =============================================================================
# Data Cleaning
# =============================================================================

class DataCleaner:
    """
    数据清洗器 / Data Cleaner
    
    Cleans and normalizes learning data.
    """
    
    def __init__(self, config: NetworkLearningConfig):
        self.config = config
        self._seen_hashes: set = set()
    
    def clean(self, item: LearningData) -> Optional[LearningData]:
        """
        Clean a learning data item.
        
        Args:
            item: Item to clean
            
        Returns:
            Cleaned item or None if filtered out
        """
        # Check content length
        if len(item.content) < self.config.min_content_length:
            return None
        
        if len(item.content) > self.config.max_content_length:
            item.content = item.content[:self.config.max_content_length]
        
        # Deduplicate
        if item.content_hash in self._seen_hashes:
            return None
        self._seen_hashes.add(item.content_hash)
        
        # Normalize content
        item.content = self._normalize_text(item.content)
        item.title = self._normalize_text(item.title)
        
        # Normalize tags
        item.tags = [tag.lower().strip() for tag in item.tags if tag]
        
        item.is_cleaned = True
        item.status = LearningStatus.CLEANING
        
        return item
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text content."""
        if not text:
            return ""
        
        # Collapse whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Strip
        text = text.strip()
        
        return text
    
    def batch_clean(self, items: List[LearningData]) -> List[LearningData]:
        """Clean multiple items."""
        cleaned = []
        for item in items:
            result = self.clean(item)
            if result:
                cleaned.append(result)
        
        logger.debug(f"Cleaned {len(cleaned)}/{len(items)} items")
        return cleaned
    
    def reset_dedup_cache(self):
        """Reset deduplication cache."""
        self._seen_hashes.clear()


# =============================================================================
# Main Learning System
# =============================================================================

class V1V3AutoLearningSystem:
    """
    V1/V3自动联网学习系统 / V1/V3 Auto Network Learning System
    
    Comprehensive automatic learning system for V1 (experimentation)
    and V3 (quarantine) versions.
    
    Features:
    - 7x24 continuous learning
    - Multi-source parallel fetching
    - Smart rate limiting
    - Automatic retry with exponential backoff
    - Quality filtering and cleaning
    - V2 system integration
    
    Usage:
        config = NetworkLearningConfig(
            v1_learning_interval_minutes=30,
            min_quality_for_v2=0.7,
        )
        
        system = V1V3AutoLearningSystem(
            version="v1",
            config=config,
            on_data_ready=my_callback,
        )
        
        await system.start()
        # System runs automatically
        await system.stop()
    """
    
    def __init__(
        self,
        version: str,  # "v1" or "v3"
        config: Optional[NetworkLearningConfig] = None,
        on_data_ready: Optional[Callable[[List[LearningData]], Any]] = None,
    ):
        """
        Initialize the auto learning system.
        
        Args:
            version: Target version ("v1" or "v3")
            config: Learning configuration
            on_data_ready: Callback when quality data is ready
        """
        if version not in ("v1", "v3"):
            raise ValueError("Version must be 'v1' or 'v3'")
        
        self.version = version
        self.config = config or NetworkLearningConfig()
        self.on_data_ready = on_data_ready
        
        # Components
        self.connectors: Dict[DataSource, DataSourceConnector] = {}
        self.rate_limiter = AsyncRateLimiter(self.config.max_requests_per_hour)
        self.quality_assessor = QualityAssessor(self.config)
        self.data_cleaner = DataCleaner(self.config)
        
        # State
        self.learned_data: List[LearningData] = []
        self._running = False
        self._learning_task: Optional[asyncio.Task] = None
        self._v2_session: Optional[aiohttp.ClientSession] = None
        
        # Metrics
        self._total_fetched = 0
        self._total_cleaned = 0
        self._total_integrated = 0
        self._cycle_count = 0
        
        # Initialize connectors
        self._init_connectors()
    
    def _init_connectors(self):
        """Initialize data source connectors."""
        connector_map = {
            DataSource.GITHUB_TRENDING: GitHubConnector,
            DataSource.GITHUB_RELEASES: GitHubConnector,
            DataSource.GITHUB_DOCS: GitHubConnector,
            DataSource.ARXIV_CS: ArxivConnector,
            DataSource.ARXIV_AI: ArxivConnector,
            DataSource.ARXIV_SE: ArxivConnector,
            DataSource.DEV_TO: DevToConnector,
            DataSource.HACKERNEWS: HackerNewsConnector,
            DataSource.HUGGINGFACE: HuggingFaceConnector,
        }
        
        for source in self.config.enabled_sources:
            connector_class = connector_map.get(source)
            if connector_class:
                self.connectors[source] = connector_class(
                    source,
                    timeout=self.config.request_timeout_seconds,
                )
        
        logger.info(f"Initialized {len(self.connectors)} connectors for {self.version}")
    
    async def start(self):
        """启动自动学习 / Start automatic learning."""
        if self._running:
            return
        
        self._running = True
        
        # Connect all data sources
        for connector in self.connectors.values():
            await connector.connect()
        
        # Initialize V2 session
        if self.config.v2_push_enabled:
            self._v2_session = aiohttp.ClientSession()
        
        # Start learning loop
        interval = (
            self.config.v1_learning_interval_minutes
            if self.version == "v1"
            else self.config.v3_learning_interval_minutes
        )
        self._learning_task = asyncio.create_task(self._learning_loop(interval))
        
        logger.info(f"{self.version.upper()} Auto Learning System started (interval: {interval}min)")
    
    async def stop(self):
        """停止自动学习 / Stop automatic learning."""
        self._running = False
        
        # Cancel learning task
        if self._learning_task:
            self._learning_task.cancel()
            try:
                await self._learning_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect connectors
        for connector in self.connectors.values():
            await connector.disconnect()
        
        # Close V2 session
        if self._v2_session:
            await self._v2_session.close()
            self._v2_session = None
        
        logger.info(f"{self.version.upper()} Auto Learning System stopped")
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
        return False
    
    async def _learning_loop(self, interval_minutes: int):
        """学习主循环 / Main learning loop."""
        while self._running:
            try:
                await self._execute_learning_cycle()
                self._cycle_count += 1
            except Exception as e:
                logger.error(f"Learning cycle error: {e}")
            
            await asyncio.sleep(interval_minutes * 60)
    
    async def _execute_learning_cycle(self):
        """执行一次学习循环 / Execute one learning cycle."""
        all_items: List[LearningData] = []
        
        # Sort sources by priority
        sorted_sources = sorted(
            self.connectors.items(),
            key=lambda x: self.config.source_priorities.get(x[0].value, 99)
        )
        
        # Concurrent fetch with rate limiting
        semaphore = asyncio.Semaphore(self.config.max_concurrent_sources)
        
        async def fetch_with_limit(source: DataSource, connector: DataSourceConnector):
            async with semaphore:
                if not await self.rate_limiter.acquire():
                    logger.debug(f"Rate limited: {source.value}")
                    return []
                
                try:
                    items = await connector.fetch(limit=50)
                    return items
                except Exception as e:
                    logger.error(f"Fetch error for {source.value}: {e}")
                    return []
        
        # Execute fetches
        tasks = [
            fetch_with_limit(source, connector)
            for source, connector in sorted_sources
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_items.extend(result)
        
        self._total_fetched += len(all_items)
        
        # Clean data
        cleaned_items = self.data_cleaner.batch_clean(all_items)
        self._total_cleaned += len(cleaned_items)
        
        # Quality assessment
        for item in cleaned_items:
            item.quality_score = self.quality_assessor.assess(item)
            item.status = LearningStatus.READY
        
        # Filter by quality
        quality_items = [
            item for item in cleaned_items
            if item.quality_score >= self.config.min_quality_for_v2
        ]
        
        # Store
        self.learned_data.extend(quality_items)
        
        # Trim if too many
        if len(self.learned_data) > self.config.max_items_in_memory:
            self.learned_data = self.learned_data[-self.config.max_items_in_memory:]
        
        # Callback
        if self.on_data_ready and quality_items:
            try:
                await self.on_data_ready(quality_items)
            except Exception as e:
                logger.error(f"Callback error: {e}")
        
        # Push to V2
        if self.config.v2_push_enabled and quality_items:
            await self._push_to_v2(quality_items)
        
        logger.info(
            f"{self.version.upper()} cycle #{self._cycle_count}: "
            f"fetched={len(all_items)}, cleaned={len(cleaned_items)}, "
            f"quality={len(quality_items)}"
        )
    
    async def _push_to_v2(self, items: List[LearningData]) -> bool:
        """推送数据到V2 / Push data to V2 system."""
        if not self._v2_session or not items:
            return False
        
        # Batch processing
        for i in range(0, len(items), self.config.v2_push_batch_size):
            batch = items[i:i + self.config.v2_push_batch_size]
            
            payload = {
                "source": f"auto_learn_{self.version}",
                "items": [item.to_dict() for item in batch],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cycle": self._cycle_count,
            }
            
            try:
                async with self._v2_session.post(
                    self.config.v2_push_endpoint,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status == 200:
                        self._total_integrated += len(batch)
                        for item in batch:
                            item.status = LearningStatus.INTEGRATED
                    else:
                        logger.warning(f"V2 push failed: {response.status}")
                        
            except Exception as e:
                logger.error(f"V2 push error: {e}")
                return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """获取学习统计 / Get learning statistics."""
        return {
            "version": self.version,
            "running": self._running,
            "cycle_count": self._cycle_count,
            "total_fetched": self._total_fetched,
            "total_cleaned": self._total_cleaned,
            "total_integrated": self._total_integrated,
            "items_in_memory": len(self.learned_data),
            "connectors": {
                source.value: {
                    "fetch_count": connector.fetch_count,
                    "error_count": connector.error_count,
                    "last_fetch": connector.last_fetch.isoformat() if connector.last_fetch else None,
                }
                for source, connector in self.connectors.items()
            },
            "rate_limit": {
                "remaining": self.rate_limiter.get_remaining(),
                "max": self.config.max_requests_per_hour,
                "reset_at": self.rate_limiter.get_reset_time().isoformat() if self.rate_limiter.get_reset_time() else None,
            },
        }
    
    def get_recent_data(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取最近学习的数据 / Get recently learned data."""
        recent = sorted(
            self.learned_data,
            key=lambda x: x.fetched_at,
            reverse=True,
        )[:limit]
        
        return [item.to_dict() for item in recent]
    
    def get_data_by_source(self, source: DataSource) -> List[LearningData]:
        """获取指定来源的数据 / Get data by source."""
        return [item for item in self.learned_data if item.source == source]


# =============================================================================
# Factory Function
# =============================================================================

def create_learning_system(
    version: str,
    config: Optional[NetworkLearningConfig] = None,
    on_data_ready: Optional[Callable] = None,
) -> V1V3AutoLearningSystem:
    """
    Factory function to create a learning system.
    
    Args:
        version: "v1" or "v3"
        config: Optional configuration
        on_data_ready: Optional callback
        
    Returns:
        Configured V1V3AutoLearningSystem
    """
    return V1V3AutoLearningSystem(
        version=version,
        config=config,
        on_data_ready=on_data_ready,
    )
