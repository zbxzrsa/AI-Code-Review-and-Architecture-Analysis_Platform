"""
Technical Blog Collector

Priority 3 data source for:
- Engineering blogs from tech companies
- Developer community articles
- Tutorial content
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from .base import BaseCollector, CollectedItem, ContentType

logger = logging.getLogger(__name__)


class TechBlogCollector(BaseCollector):
    """
    Collector for technical blogs and articles.
    
    Sources:
    - Dev.to
    - Medium (programming tags)
    - Company engineering blogs
    - Hacker News (top stories)
    
    Uses RSS feeds and APIs where available.
    """
    
    # Known blog sources with their APIs/RSS feeds
    BLOG_SOURCES = {
        "dev.to": {
            "api_url": "https://dev.to/api/articles",
            "type": "api",
            "params": {"tag": "programming", "top": 7},
        },
        "hackernews": {
            "api_url": "https://hacker-news.firebaseio.com/v0/topstories.json",
            "item_url": "https://hacker-news.firebaseio.com/v0/item/{}.json",
            "type": "api",
        },
    }
    
    async def collect(
        self,
        since: Optional[datetime] = None,
        max_items: Optional[int] = None,
    ) -> AsyncIterator[CollectedItem]:
        """
        Collect articles from technical blogs.
        
        Collection strategy:
        1. Query each blog source API
        2. Filter by date and relevance
        3. Extract article content
        """
        max_items = max_items or self.schedule.max_items_per_cycle
        collected = 0
        
        # Collect from Dev.to
        async for item in self._collect_devto(since, max_items - collected):
            yield item
            collected += 1
            if collected >= max_items:
                return
        
        # Collect from Hacker News
        async for item in self._collect_hackernews(since, max_items - collected):
            yield item
            collected += 1
            if collected >= max_items:
                return
    
    async def _collect_devto(
        self,
        since: Optional[datetime] = None,
        max_items: int = 30,
    ) -> AsyncIterator[CollectedItem]:
        """Collect articles from Dev.to API."""
        source = self.BLOG_SOURCES["dev.to"]
        
        params = {**source["params"], "per_page": min(max_items, 30)}
        
        response = await self._make_request("GET", source["api_url"], params=params)
        
        if not response or not isinstance(response, list):
            return
        
        for article in response:
            item = self._parse_devto_article(article)
            if item:
                # Filter by date if specified
                if since and item.created_at and item.created_at < since:
                    continue
                yield item
    
    def _parse_devto_article(
        self,
        article: Dict[str, Any],
    ) -> Optional[CollectedItem]:
        """Parse Dev.to article."""
        try:
            return CollectedItem(
                source="dev.to",
                source_id=str(article.get("id")),
                url=article.get("url", ""),
                title=article.get("title", ""),
                content=article.get("description", "") or article.get("body_markdown", ""),
                content_type=ContentType.ARTICLE,
                tags=article.get("tag_list", []),
                author=article.get("user", {}).get("username"),
                created_at=self._parse_datetime(article.get("created_at")),
                updated_at=self._parse_datetime(article.get("edited_at")),
                metadata={
                    "reactions": article.get("public_reactions_count", 0),
                    "comments": article.get("comments_count", 0),
                    "reading_time": article.get("reading_time_minutes"),
                    "cover_image": article.get("cover_image"),
                },
            )
        except Exception as e:
            logger.debug(f"Failed to parse Dev.to article: {e}")
            return None
    
    async def _collect_hackernews(
        self,
        since: Optional[datetime] = None,
        max_items: int = 20,
    ) -> AsyncIterator[CollectedItem]:
        """Collect top stories from Hacker News."""
        source = self.BLOG_SOURCES["hackernews"]
        
        # Get top story IDs
        response = await self._make_request("GET", source["api_url"])
        
        if not response or not isinstance(response, list):
            return
        
        # Fetch top N stories
        collected = 0
        for story_id in response[:max_items * 2]:  # Fetch extra for filtering
            if collected >= max_items:
                break
            
            item_url = source["item_url"].format(story_id)
            story = await self._make_request("GET", item_url)
            
            if not story:
                continue
            
            # Filter to programming-related stories
            if not self._is_programming_related(story):
                continue
            
            item = self._parse_hn_story(story)
            if item:
                # Filter by date if specified
                if since and item.created_at and item.created_at < since:
                    continue
                yield item
                collected += 1
    
    def _is_programming_related(self, story: Dict[str, Any]) -> bool:
        """Check if HN story is programming-related."""
        title = (story.get("title") or "").lower()
        url = story.get("url") or ""
        
        programming_keywords = [
            "python", "javascript", "rust", "go", "java",
            "programming", "coding", "developer", "software",
            "api", "database", "algorithm", "machine learning",
            "ai", "open source", "github", "framework",
            "kubernetes", "docker", "cloud", "aws", "gcp",
            "security", "performance", "optimization",
        ]
        
        # Check title for keywords
        for keyword in programming_keywords:
            if keyword in title:
                return True
        
        # Check URL domain
        programming_domains = [
            "github.com", "gitlab.com", "dev.to",
            "medium.com", "stackoverflow.com",
        ]
        
        for domain in programming_domains:
            if domain in url:
                return True
        
        return False
    
    def _parse_hn_story(
        self,
        story: Dict[str, Any],
    ) -> Optional[CollectedItem]:
        """Parse Hacker News story."""
        try:
            story_id = story.get("id")
            timestamp = story.get("time")
            
            return CollectedItem(
                source="hackernews",
                source_id=str(story_id),
                url=story.get("url") or f"https://news.ycombinator.com/item?id={story_id}",
                title=story.get("title", ""),
                content=story.get("text", "") or story.get("title", ""),
                content_type=ContentType.ARTICLE,
                tags=["hackernews"],
                author=story.get("by"),
                created_at=datetime.fromtimestamp(timestamp, tz=timezone.utc) if timestamp else None,
                metadata={
                    "score": story.get("score", 0),
                    "comments": story.get("descendants", 0),
                    "type": story.get("type"),
                },
            )
        except Exception as e:
            logger.debug(f"Failed to parse HN story: {e}")
            return None
    
    def _parse_item(self, raw_data: Dict[str, Any]) -> Optional[CollectedItem]:
        """Parse raw data - implemented in specific parsers."""
        return None
    
    @staticmethod
    def _parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
        """Parse datetime string."""
        if not dt_str:
            return None
        try:
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except Exception:
            return None
