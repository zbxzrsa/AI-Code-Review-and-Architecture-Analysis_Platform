"""
Data Source Collectors

Provides collectors for various data sources:
- GitHub: Technical documents, open-source code
- ArXiv: Academic papers
- Tech Blogs: Selected technical articles
"""

from .base import BaseCollector, CollectedItem, CollectionResult
from .github_collector import GitHubCollector
from .arxiv_collector import ArXivCollector
from .blog_collector import TechBlogCollector

__all__ = [
    "BaseCollector",
    "CollectedItem",
    "CollectionResult",
    "GitHubCollector",
    "ArXivCollector",
    "TechBlogCollector",
]
