"""
ArXiv Academic Paper Collector

Priority 2 data source for:
- Academic papers on software engineering
- AI/ML research papers
- Programming language theory
"""

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from .base import BaseCollector, CollectedItem, ContentType

logger = logging.getLogger(__name__)


class ArXivCollector(BaseCollector):
    """
    Collector for ArXiv academic papers.
    
    Collects papers from:
    - cs.SE (Software Engineering)
    - cs.AI (Artificial Intelligence)
    - cs.LG (Machine Learning)
    - cs.PL (Programming Languages)
    
    Rate Limit: ~3 requests per second
    """
    
    API_BASE = "https://export.arxiv.org/api/query"
    NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get headers for ArXiv API."""
        headers = super()._get_default_headers()
        headers["Accept"] = "application/xml"
        return headers
    
    async def collect(
        self,
        since: Optional[datetime] = None,
        max_items: Optional[int] = None,
    ) -> AsyncIterator[CollectedItem]:
        """
        Collect papers from ArXiv.
        
        Collection strategy:
        1. Query each category
        2. Parse XML response
        3. Extract paper metadata and abstracts
        """
        max_items = max_items or self.schedule.max_items_per_cycle
        collected = 0
        
        # Get categories from filters
        categories = self.config.filters.get("categories", ["cs.SE", "cs.AI"])
        
        for category in categories:
            if collected >= max_items:
                break
            
            async for item in self._collect_category(category, since, max_items - collected):
                yield item
                collected += 1
                if collected >= max_items:
                    break
    
    async def _collect_category(
        self,
        category: str,
        since: Optional[datetime] = None,
        max_results: int = 50,
    ) -> AsyncIterator[CollectedItem]:
        """Collect papers from a specific category."""
        # Build query
        query = f"cat:{category}"
        
        params = {
            "search_query": query,
            "start": 0,
            "max_results": min(max_results, 100),
            "sortBy": "lastUpdatedDate",
            "sortOrder": "descending",
        }
        
        # Make request
        async with self._session.get(self.API_BASE, params=params) as response:
            if response.status != 200:
                logger.error(f"ArXiv API error: {response.status}")
                return
            
            xml_content = await response.text()
        
        # Parse XML
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"Failed to parse ArXiv XML: {e}")
            return
        
        # Extract entries
        for entry in root.findall("atom:entry", self.NAMESPACE):
            item = self._parse_entry(entry, category)
            if item:
                # Filter by date if specified
                if since and item.updated_at and item.updated_at < since:
                    continue
                yield item
    
    def _parse_entry(
        self,
        entry: ET.Element,
        category: str,
    ) -> Optional[CollectedItem]:
        """Parse an ArXiv entry element."""
        try:
            # Extract basic info
            arxiv_id = self._get_text(entry, "atom:id")
            if arxiv_id:
                arxiv_id = arxiv_id.split("/")[-1]  # Extract ID from URL
            
            title = self._get_text(entry, "atom:title")
            abstract = self._get_text(entry, "atom:summary")
            
            if not title or not abstract:
                return None
            
            # Clean whitespace
            title = " ".join(title.split())
            abstract = " ".join(abstract.split())
            
            # Get authors
            authors = []
            for author in entry.findall("atom:author", self.NAMESPACE):
                name = self._get_text(author, "atom:name")
                if name:
                    authors.append(name)
            
            # Get dates
            published = self._get_text(entry, "atom:published")
            updated = self._get_text(entry, "atom:updated")
            
            # Get PDF link
            pdf_url = None
            for link in entry.findall("atom:link", self.NAMESPACE):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href")
                    break
            
            # Get categories
            categories = []
            for cat in entry.findall("atom:category", self.NAMESPACE):
                term = cat.get("term")
                if term:
                    categories.append(term)
            
            return CollectedItem(
                source="arxiv",
                source_id=arxiv_id,
                url=f"https://arxiv.org/abs/{arxiv_id}",
                title=title,
                content=abstract,
                content_type=ContentType.PAPER,
                tags=categories,
                author=", ".join(authors[:3]),  # First 3 authors
                created_at=self._parse_datetime(published),
                updated_at=self._parse_datetime(updated),
                metadata={
                    "pdf_url": pdf_url,
                    "primary_category": category,
                    "all_authors": authors,
                    "author_count": len(authors),
                },
            )
            
        except Exception as e:
            logger.debug(f"Failed to parse ArXiv entry: {e}")
            return None
    
    def _get_text(self, element: ET.Element, path: str) -> Optional[str]:
        """Get text content from element by path."""
        child = element.find(path, self.NAMESPACE)
        return child.text if child is not None else None
    
    def _parse_item(self, raw_data: Dict[str, Any]) -> Optional[CollectedItem]:
        """Parse raw data - implemented in _parse_entry."""
        return None
    
    @staticmethod
    def _parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
        """Parse ArXiv datetime string."""
        if not dt_str:
            return None
        try:
            # ArXiv uses ISO 8601 format
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except Exception:
            return None
