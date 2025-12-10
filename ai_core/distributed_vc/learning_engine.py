"""
在线学习引擎 (Online Learning Engine)

模块功能描述:
    实现 7×24 小时持续学习功能，支持从多个渠道收集和处理知识。
    采用增量学习模式，在不中断服务的情况下持续更新知识库。

主要组件:
    - LearningChannel: 学习渠道抽象基类
    - GitHubTrendingChannel: GitHub 趋势仓库渠道
    - TechBlogChannel: 技术博客聚合渠道
    - OnlineLearningEngine: 在线学习主引擎

性能指标:
    - 学习延迟: < 5 分钟
    - 支持增量学习
    - 不中断服务

最后修改日期: 2024-12-07
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
from collections import deque, defaultdict
import logging
import hashlib
import aiohttp
import json

logger = logging.getLogger(__name__)


class ChannelType(Enum):
    """
    学习渠道类型枚举
    
    定义系统支持的各类学习数据源类型。
    
    渠道类型:
        - GITHUB_TRENDING: GitHub 趋势仓库
        - GITHUB_RELEASES: GitHub 版本发布
        - TECH_BLOGS: 技术博客
        - PAPER_DATABASE: 论文数据库
        - STACKOVERFLOW: Stack Overflow 问答
        - DOCUMENTATION: 技术文档
        - CODE_SAMPLES: 代码示例
        - CUSTOM_API: 自定义 API
    """
    GITHUB_TRENDING = "github_trending"
    GITHUB_RELEASES = "github_releases"
    TECH_BLOGS = "tech_blogs"
    PAPER_DATABASE = "paper_database"
    STACKOVERFLOW = "stackoverflow"
    DOCUMENTATION = "documentation"
    CODE_SAMPLES = "code_samples"
    CUSTOM_API = "custom_api"


class LearningStatus(Enum):
    """
    学习任务状态枚举
    
    表示学习任务的当前执行状态。
    
    状态说明:
        - PENDING: 等待执行
        - FETCHING: 正在获取数据
        - PROCESSING: 正在处理数据
        - INTEGRATING: 正在整合知识
        - COMPLETED: 完成
        - FAILED: 失败
    """
    PENDING = "pending"
    FETCHING = "fetching"
    PROCESSING = "processing"
    INTEGRATING = "integrating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class LearningSource:
    """
    学习数据源配置类
    
    定义单个学习数据源的配置信息。
    
    属性说明:
        - source_id: 数据源唯一标识符
        - channel_type: 渠道类型
        - name: 数据源名称
        - url: 数据源 URL
        - api_key: API 密钥（可选）
        - fetch_interval_seconds: 获取间隔（秒）
        - priority: 优先级
        - enabled: 是否启用
        - last_fetch: 最后获取时间
        - fetch_count: 获取次数
        - error_count: 错误次数
    """
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
    """
    学习项目数据类
    
    表示单条学习到的知识内容。
    
    属性说明:
        - item_id: 项目唯一标识符
        - source_id: 来源标识符
        - channel_type: 渠道类型
        - title: 标题
        - content: 内容
        - url: 原始链接
        - timestamp: 时间戳
        - relevance_score: 相关性评分
        - processed: 是否已处理
        - integrated: 是否已整合
        - metadata: 元数据
    """
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
    """
    学习性能指标类
    
    记录学习引擎的各项性能指标。
    
    指标说明:
        - total_items_fetched: 已获取项目总数
        - total_items_processed: 已处理项目总数
        - total_items_integrated: 已整合项目总数
        - average_fetch_time_ms: 平均获取时间（毫秒）
        - average_process_time_ms: 平均处理时间（毫秒）
        - average_integration_time_ms: 平均整合时间（毫秒）
        - last_learning_delay_seconds: 最近学习延迟（秒）
        - learning_delay_target_seconds: 目标学习延迟（秒）
    """
    total_items_fetched: int = 0
    total_items_processed: int = 0
    total_items_integrated: int = 0
    average_fetch_time_ms: float = 0.0
    average_process_time_ms: float = 0.0
    average_integration_time_ms: float = 0.0
    last_learning_delay_seconds: float = 0.0
    learning_delay_target_seconds: float = 300.0  # 5 minutes


class LearningChannel(ABC):
    """
    学习渠道抽象基类
    
    功能描述:
        定义学习渠道的基本接口，所有具体渠道实现必须继承此类。
    
    抽象方法:
        - fetch(): 从数据源获取学习项目
        - process(): 处理单个学习项目
    """
    
    def __init__(self, source: LearningSource):
        self.source = source
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self) -> None:
        """
        初始化渠道
        
        创建 HTTP 会话用于网络请求。
        Must be called within an async context.
        """
        # Create session with timeout and connection limits for reliability
        timeout = aiohttp.ClientTimeout(total=60, connect=10, sock_read=30)
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            raise_for_status=False  # Handle errors explicitly
        )
    
    async def close(self) -> None:
        """
        关闭渠道
        
        释放 HTTP 会话资源。
        """
        if self.session and not self.session.closed:
            try:
                await self.session.close()
            except Exception as e:
                logger.warning(f"Error closing session: {e}")
    
    @abstractmethod
    async def fetch(self) -> List[LearningItem]:
        """
        从数据源获取学习项目
        
        返回值:
            List[LearningItem]: 获取到的学习项目列表
        """
        pass
    
    @abstractmethod
    async def process(self, item: LearningItem) -> Dict[str, Any]:
        """
        处理学习项目
        
        参数:
            item: 要处理的学习项目
        
        返回值:
            Dict[str, Any]: 处理结果字典
        """
        pass


class GitHubTrendingChannel(LearningChannel):
    """
    GitHub 趋势仓库学习渠道
    
    功能描述:
        从 GitHub 获取趋势仓库信息，提取技术栈、设计模式和最佳实践。
    """
    
    async def fetch(self) -> List[LearningItem]:
        """
        获取趋势仓库
        
        调用 GitHub API 获取最近的热门仓库列表。
        
        返回值:
            List[LearningItem]: 仓库信息列表
        """
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
        """
        处理 GitHub 仓库数据
        
        提取仓库的技术栈、设计模式和质量指标。
        
        参数:
            item: GitHub 仓库学习项目
        
        返回值:
            Dict[str, Any]: 包含技术、模式和质量指标的字典
        """
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
    """
    技术博客聚合学习渠道
    
    功能描述:
        从多个技术博客源聚合内容，提取技术知识和最佳实践。
    
    支持的博客源:
        - dev.to
        - hackernoon.com
        - medium.com
        - dzone.com
    """
    
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
    """
    学术论文数据库学习渠道
    
    功能描述:
        从学术论文数据库（如 ArXiv）获取最新论文，
        提取前沿技术和研究成果。
    
    支持的数据库:
        - ArXiv (cs.SE, cs.AI, cs.LG 类别)
    """
    
    async def fetch(self) -> List[LearningItem]:
        """
        获取最新论文
        
        调用 ArXiv API 获取计算机科学相关论文。
        
        返回值:
            List[LearningItem]: 论文信息列表
        """
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
        """
        处理学术论文
        
        提取论文标题、摘要和来源信息。
        
        参数:
            item: 论文学习项目
        
        返回值:
            Dict[str, Any]: 处理后的论文信息
        """
        processed = {
            "paper": item.title,
            "abstract": item.content[:500],
            "source": "arxiv"
        }
        
        item.processed = True
        return processed


class OnlineLearningEngine:
    """
    在线学习引擎主类
    
    功能描述:
        实现 7×24 小时持续学习，支持多渠道数据源，
        在不中断服务的情况下进行增量更新。
    
    核心特性:
        - 7×24 持续学习
        - 多渠道支持
        - 增量更新，不中断服务
        - 学习延迟跟踪（目标: < 5 分钟）
    
    属性说明:
        - learning_delay_target: 目标学习延迟（秒）
        - max_concurrent: 最大并发数
        - batch_size: 批处理大小
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
        
        # Use bounded deque to prevent unbounded memory growth
        self.processed_items: deque = deque(maxlen=10000)
        
        # Track statistics separately
        self.stats = {
            "total_processed": 0,
            "total_integrated": 0,
            "by_channel": defaultdict(int),
            "by_date": defaultdict(int)
        }
        
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
        """
        注册学习数据源
        
        参数:
            source: 学习数据源配置对象
            
        Raises:
            ValueError: If validation fails
        """
        import re
        import urllib.parse
        
        # Validate source_id
        if not source.source_id or not source.source_id.strip():
            raise ValueError("source_id cannot be empty")
        
        if not re.match(r'^[a-zA-Z0-9_-]+$', source.source_id):
            raise ValueError(
                f"source_id must contain only alphanumeric characters, "
                f"underscores, and hyphens: {source.source_id}"
            )
        
        if source.source_id in self.sources:
            raise ValueError(f"Source {source.source_id} already registered")
        
        # Validate name
        if not source.name or len(source.name) > 200:
            raise ValueError("name must be 1-200 characters")
        
        # Validate URL
        if source.url:
            if not source.url.startswith(('http://', 'https://')):
                raise ValueError(f"Invalid URL format: {source.url}")
            
            try:
                parsed = urllib.parse.urlparse(source.url)
                if not parsed.netloc:
                    raise ValueError(f"Invalid URL: {source.url}")
            except Exception as e:
                raise ValueError(f"Invalid URL: {source.url}") from e
        
        # Validate fetch interval
        if source.fetch_interval_seconds < 60:
            raise ValueError(
                f"fetch_interval_seconds must be >= 60 (got {source.fetch_interval_seconds})"
            )
        
        if source.fetch_interval_seconds > 86400:
            logger.warning(
                f"Large fetch interval: {source.fetch_interval_seconds}s (24h+)"
            )
        
        # Validate priority
        if not 1 <= source.priority <= 5:
            raise ValueError(f"priority must be 1-5 (got {source.priority})")
        
        # Validate channel type
        if source.channel_type not in ChannelType:
            raise ValueError(f"Invalid channel_type: {source.channel_type}")
        
        # Register source
        self.sources[source.source_id] = source
        
        # Create appropriate channel
        channel = self._create_channel(source)
        if channel:
            self.channels[source.source_id] = channel
            logger.info(
                f"Registered learning source: {source.name} "
                f"(type={source.channel_type.value}, interval={source.fetch_interval_seconds}s)"
            )
        else:
            logger.warning(
                f"No channel implementation for {source.channel_type.value}"
            )
    
    def _create_channel(self, source: LearningSource) -> Optional[LearningChannel]:
        """
        根据类型创建渠道
        
        参数:
            source: 学习数据源配置
        
        返回值:
            Optional[LearningChannel]: 创建的渠道对象，不支持的类型返回 None
        """
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
        """
        启动学习引擎（7×24 运行）
        
        初始化所有渠道，启动获取、处理和整合任务。
        """
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
    
    async def _reenable_source(self, source_id: str, backoff: int) -> None:
        """
        Re-enable a disabled source after backoff period.
        
        Args:
            source_id: Source identifier
            backoff: Backoff period in seconds
        """
        await asyncio.sleep(backoff)
        
        if source_id in self.sources:
            self.sources[source_id].enabled = True
            self.sources[source_id].error_count = 0
            logger.info(
                f"Re-enabled source {source_id} after {backoff}s backoff"
            )
    
    async def stop(self) -> None:
        """
        停止学习引擎
        
        取消所有任务并关闭渠道资源。
        """
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
        """
        持续获取循环
        
        为指定数据源运行持续的数据获取循环。
        
        参数:
            source_id: 数据源标识符
        """
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
                
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.error(f"Network error for {source_id}: {e}")
                source.error_count += 1
                
                # Circuit breaker logic
                if source.error_count >= 5:
                    logger.warning(
                        f"Source {source_id} circuit breaker opened after "
                        f"{source.error_count} consecutive failures"
                    )
                    source.enabled = False
                    
                    # Schedule re-enable after backoff
                    asyncio.create_task(self._reenable_source(source_id, backoff=300))
                    
            except KeyError as e:
                logger.error(f"Configuration error for {source_id}: {e}")
                source.enabled = False
                
            except Exception as e:
                logger.critical(
                    f"Unexpected error in fetch loop for {source_id}: {e}",
                    exc_info=True
                )
                source.error_count += 1
            
            # Wait for next fetch
            await asyncio.sleep(source.fetch_interval_seconds)
    
    async def _process_loop(self) -> None:
        """
        持续处理循环
        
        从队列中获取项目并进行处理。
        """
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
                    
                    # Add to bounded deque
                    self.processed_items.append(item)
                    
                    # Update statistics
                    self.stats["total_processed"] += 1
                    self.stats["by_channel"][item.channel_type.value] += 1
                    self.stats["by_date"][datetime.now().date().isoformat()] += 1
                    
                    self.metrics.total_items_processed += 1
                
                process_time = (datetime.now() - process_start).total_seconds() * 1000
                self.metrics.average_process_time_ms = (
                    self.metrics.average_process_time_ms * 0.9 + process_time * 0.1
                )
                
            except asyncio.TimeoutError:
                continue
                
            except (KeyError, AttributeError) as e:
                logger.error(f"Data processing error: {e}")
                
            except Exception as e:
                logger.error(f"Unexpected process error: {e}", exc_info=True)
    
    async def _integration_loop(self) -> None:
        """
        增量整合循环
        
        定期将已处理的项目批量整合到知识库中。
        """
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
                
            except (ValueError, TypeError) as e:
                logger.error(f"Data integration error: {e}")
                
            except Exception as e:
                logger.error(f"Unexpected integration error: {e}", exc_info=True)
    
    async def _integrate_batch(self, items: List[LearningItem]) -> None:
        """Integrate a batch of learned items (incremental learning)"""
        # Sort by relevance
        sorted_items = sorted(items, key=lambda x: x.relevance_score, reverse=True)
        
        for item in sorted_items:
            # Mark as integrated
            item.integrated = True
            self.metrics.total_items_integrated += 1
            self.stats["total_integrated"] += 1
            
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


# =============================================================================
# V1/V3 Automatic Networked Learning Enhancement
# =============================================================================

@dataclass
class AutoNetworkLearningConfig:
    """
    V1/V3 Automatic Network Learning Configuration
    
    Controls automatic learning behavior for experimentation (V1)
    and quarantine (V3) versions.
    """
    enabled: bool = True
    learning_interval_minutes: int = 30
    max_items_per_fetch: int = 100
    quality_threshold: float = 0.7
    rate_limit_per_hour: int = 100
    retry_on_failure: bool = True
    max_retries: int = 3
    backoff_seconds: float = 60.0
    
    # Data Source Priority (1 = Highest)
    source_priorities: Dict[str, int] = field(default_factory=lambda: {
        "github": 1,
        "arxiv": 2,
        "tech_blogs": 3,
        "stackoverflow": 4,
    })
    
    # Content filters
    min_content_length: int = 100
    max_content_length: int = 100000
    allowed_languages: List[str] = field(default_factory=lambda: [
        "python", "javascript", "go", "rust", "java", "typescript"
    ])
    
    # Integration settings
    auto_integrate: bool = True
    integration_batch_size: int = 50
    v2_push_enabled: bool = True
    v2_push_endpoint: str = "/api/v2/learning/ingest"


class DataQualityFilter:
    """
    Filters data based on quality criteria.
    
    Quality scoring based on:
    - Content completeness
    - Technical relevance
    - Freshness
    """
    
    def __init__(self, quality_threshold: float = 0.7):
        self.quality_threshold = quality_threshold
        self._tech_keywords = {
            "algorithm", "api", "architecture", "async", "backend",
            "cache", "class", "code", "database", "debug", "deploy",
            "docker", "framework", "function", "git", "http",
            "kubernetes", "library", "microservice", "module",
            "optimization", "pattern", "performance", "python",
            "security", "server", "test", "variable",
        }
    
    def calculate_quality(self, item: LearningItem) -> float:
        """Calculate quality score for an item."""
        score = 0.0
        
        # Content length score (0-0.3)
        content_len = len(item.content)
        if content_len >= 500:
            score += 0.3
        elif content_len >= 200:
            score += 0.2
        elif content_len >= 100:
            score += 0.1
        
        # Technical relevance (0-0.4)
        content_lower = item.content.lower()
        keyword_count = sum(1 for kw in self._tech_keywords if kw in content_lower)
        score += min(0.4, keyword_count * 0.04)
        
        # Title quality (0-0.2)
        if item.title and len(item.title) > 10:
            score += 0.2
        elif item.title:
            score += 0.1
        
        # Source bonus (0-0.1)
        if item.url:
            score += 0.1
        
        return min(1.0, score)
    
    def filter(self, items: List[LearningItem]) -> List[LearningItem]:
        """Filter items by quality threshold."""
        filtered = []
        for item in items:
            quality = self.calculate_quality(item)
            if quality >= self.quality_threshold:
                item.relevance_score = quality
                filtered.append(item)
        
        logger.debug(f"Quality filter: {len(filtered)}/{len(items)} passed (threshold={self.quality_threshold})")
        return filtered


class AsyncRateLimiter:
    """
    Async rate limiter using token bucket algorithm.
    
    Controls request rate to avoid API throttling.
    """
    
    def __init__(self, requests_per_hour: int = 100):
        self.requests_per_hour = requests_per_hour
        self.tokens = requests_per_hour
        self.max_tokens = requests_per_hour
        self.refill_rate = requests_per_hour / 3600  # tokens per second
        self.last_refill = datetime.now()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens acquired, False otherwise
        """
        async with self._lock:
            # Refill tokens based on time elapsed
            now = datetime.now()
            elapsed = (now - self.last_refill).total_seconds()
            refill_amount = elapsed * self.refill_rate
            
            self.tokens = min(self.max_tokens, self.tokens + refill_amount)
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    async def wait_for_token(self, timeout: float = 60.0) -> bool:
        """
        Wait until a token is available.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if token acquired, False if timeout
        """
        start = datetime.now()
        
        while (datetime.now() - start).total_seconds() < timeout:
            if await self.acquire():
                return True
            await asyncio.sleep(1)
        
        return False
    
    def get_remaining(self) -> int:
        """Get remaining tokens."""
        return int(self.tokens)


class EnhancedLearningEngine(OnlineLearningEngine):
    """
    Enhanced learning engine supporting automatic network learning for V1/V3.
    
    Features:
    - Priority-based source fetching
    - Quality filtering
    - Rate limiting per source
    - Automatic retry with backoff
    - V2 integration for cleaned data
    
    Usage:
        config = AutoNetworkLearningConfig(
            learning_interval_minutes=30,
            quality_threshold=0.7,
        )
        
        engine = EnhancedLearningEngine(version="v1", config=config)
        await engine.start()
        await engine.auto_learn_loop()
    """
    
    def __init__(
        self,
        version: str,
        config: Optional[AutoNetworkLearningConfig] = None,
    ):
        """
        Initialize enhanced learning engine.
        
        Args:
            version: Target version ("v1" or "v3")
            config: Learning configuration
        """
        super().__init__()
        self.version = version
        self.config = config or AutoNetworkLearningConfig()
        
        # Quality filter
        self.quality_filter = DataQualityFilter(self.config.quality_threshold)
        
        # Rate limiters per source
        self._rate_limiters: Dict[str, AsyncRateLimiter] = {}
        for source in self.config.source_priorities:
            self._rate_limiters[source] = AsyncRateLimiter(
                self.config.rate_limit_per_hour
            )
        
        # Retry tracking
        self._source_errors: Dict[str, int] = {}
        
        # Knowledge base
        self._knowledge_base: List[Dict[str, Any]] = []
        
        # V2 integration session
        self._v2_session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"Enhanced learning engine initialized for {version}")
    
    async def start(self) -> None:
        """Start the enhanced learning engine."""
        await super().start()
        
        if self.config.v2_push_enabled:
            self._v2_session = aiohttp.ClientSession()
        
        logger.info(f"Enhanced learning engine started for {self.version}")
    
    async def stop(self) -> None:
        """Stop the enhanced learning engine."""
        await super().stop()
        
        if self._v2_session:
            await self._v2_session.close()
        
        logger.info(f"Enhanced learning engine stopped for {self.version}")
    
    async def auto_learn_loop(self) -> None:
        """
        Automatic learning main loop.
        
        Continuously fetches, filters, and integrates data from
        prioritized sources.
        """
        logger.info(f"Starting auto-learn loop for {self.version}")
        
        while self.is_running:
            try:
                # Fetch from sources in priority order
                for source, priority in sorted(
                    self.config.source_priorities.items(),
                    key=lambda x: x[1]
                ):
                    if not self.config.enabled:
                        break
                    
                    # Check rate limit
                    rate_limiter = self._rate_limiters.get(source)
                    if rate_limiter and not await rate_limiter.acquire():
                        logger.debug(f"Rate limited for source: {source}")
                        continue
                    
                    # Fetch and process
                    try:
                        items = await self._fetch_from_source(source)
                        clean_items = await self._clean_and_filter(items)
                        await self._integrate_to_knowledge_base(clean_items)
                        
                        # Reset error count on success
                        self._source_errors[source] = 0
                        
                    except Exception as e:
                        await self._handle_fetch_error(source, e)
                
                logger.info(
                    f"Auto-learn cycle complete. "
                    f"Knowledge base size: {len(self._knowledge_base)}"
                )
                
            except Exception as e:
                logger.error(f"Auto-learn loop error: {e}")
                await asyncio.sleep(self.config.backoff_seconds)
            
            # Wait for next cycle
            await asyncio.sleep(self.config.learning_interval_minutes * 60)
    
    async def _fetch_from_source(self, source: str) -> List[LearningItem]:
        """
        Fetch items from a specific source.
        
        Args:
            source: Source name (github, arxiv, etc.)
            
        Returns:
            List of fetched items
        """
        items = []
        
        # Find matching channel
        for source_id, channel in self.channels.items():
            if source.lower() in source_id.lower():
                try:
                    fetched = await channel.fetch()
                    items.extend(fetched[:self.config.max_items_per_fetch])
                    
                    logger.info(f"Fetched {len(items)} items from {source}")
                except Exception as e:
                    logger.warning(f"Fetch error from {source}: {e}")
        
        return items
    
    async def _clean_and_filter(self, items: List[LearningItem]) -> List[LearningItem]:
        """
        Clean and filter items based on quality.
        
        Args:
            items: Raw items to clean
            
        Returns:
            Cleaned and filtered items
        """
        if not items:
            return []
        
        # Apply quality filter
        quality_filtered = self.quality_filter.filter(items)
        
        # Apply content length filter
        length_filtered = [
            item for item in quality_filtered
            if self.config.min_content_length <= len(item.content) <= self.config.max_content_length
        ]
        
        logger.info(
            f"Cleaning: {len(items)} -> {len(quality_filtered)} (quality) -> "
            f"{len(length_filtered)} (length)"
        )
        
        return length_filtered
    
    async def _integrate_to_knowledge_base(self, items: List[LearningItem]) -> None:
        """
        Integrate cleaned items to knowledge base.
        
        Args:
            items: Cleaned items to integrate
        """
        if not items:
            return
        
        for item in items:
            knowledge_entry = {
                "id": item.item_id,
                "source": item.source_id,
                "title": item.title,
                "content": item.content,
                "relevance": item.relevance_score,
                "timestamp": datetime.now().isoformat(),
                "version": self.version,
            }
            
            self._knowledge_base.append(knowledge_entry)
            item.integrated = True
            self.metrics.total_items_integrated += 1
        
        # Push to V2 if enabled
        if self.config.auto_integrate and self.config.v2_push_enabled:
            await self._push_to_v2(items)
        
        logger.info(f"Integrated {len(items)} items to knowledge base")
    
    async def _push_to_v2(self, items: List[LearningItem]) -> bool:
        """
        Push cleaned data to V2 system.
        
        Args:
            items: Items to push
            
        Returns:
            True if successful
        """
        if not self._v2_session or not items:
            return False
        
        payload = {
            "source": f"auto_learn_{self.version}",
            "items": [
                {
                    "id": item.item_id,
                    "title": item.title,
                    "content": item.content,
                    "relevance": item.relevance_score,
                    "channel": item.channel_type.value,
                }
                for item in items
            ],
            "timestamp": datetime.now().isoformat(),
        }
        
        try:
            async with self._v2_session.post(
                self.config.v2_push_endpoint,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status == 200:
                    logger.info(f"Pushed {len(items)} items to V2")
                    return True
                else:
                    logger.warning(f"V2 push failed: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"V2 push error: {e}")
            return False
    
    async def _handle_fetch_error(self, source: str, error: Exception) -> None:
        """
        Handle fetch error with retry logic.
        
        Args:
            source: Source that failed
            error: The exception
        """
        self._source_errors[source] = self._source_errors.get(source, 0) + 1
        error_count = self._source_errors[source]
        
        logger.error(f"Fetch error for {source} (attempt {error_count}): {error}")
        
        if self.config.retry_on_failure and error_count < self.config.max_retries:
            # Exponential backoff
            backoff = self.config.backoff_seconds * (2 ** (error_count - 1))
            logger.info(f"Retrying {source} in {backoff}s")
            await asyncio.sleep(backoff)
        else:
            logger.warning(f"Max retries reached for {source}, skipping")
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        by_source = {}
        for entry in self._knowledge_base:
            source = entry.get("source", "unknown")
            by_source[source] = by_source.get(source, 0) + 1
        
        return {
            "total_entries": len(self._knowledge_base),
            "by_source": by_source,
            "version": self.version,
            "rate_limits": {
                source: limiter.get_remaining()
                for source, limiter in self._rate_limiters.items()
            },
            "source_errors": dict(self._source_errors),
        }
    
    def get_recent_knowledge(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent knowledge entries."""
        return sorted(
            self._knowledge_base,
            key=lambda x: x.get("timestamp", ""),
            reverse=True,
        )[:limit]
