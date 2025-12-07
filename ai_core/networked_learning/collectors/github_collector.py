"""
GitHub Data Collector

Priority 1 data source for:
- Technical documentation
- Open-source code
- README files
- Code examples
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from .base import BaseCollector, CollectedItem, ContentType

logger = logging.getLogger(__name__)


class GitHubCollector(BaseCollector):
    """
    Collector for GitHub repositories and content.
    
    Collects:
    - Repository README files
    - Documentation files
    - Code samples from popular repositories
    - Technical discussions from issues/PRs
    
    Rate Limit: 5000 requests/hour (authenticated)
    """
    
    API_BASE = "https://api.github.com"
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get headers with GitHub token if available."""
        headers = super()._get_default_headers()
        headers["Accept"] = "application/vnd.github.v3+json"
        
        # Add auth token from environment
        token = os.getenv(self.config.api_key_env, "")
        if token:
            headers["Authorization"] = f"token {token}"
        
        return headers
    
    async def collect(
        self,
        since: Optional[datetime] = None,
        max_items: Optional[int] = None,
    ) -> AsyncIterator[CollectedItem]:
        """
        Collect content from GitHub.
        
        Collection strategy:
        1. Search for trending repositories by language
        2. Fetch README and documentation
        3. Collect code samples from relevant files
        """
        max_items = max_items or self.schedule.max_items_per_cycle
        collected = 0
        
        # Get languages from filters
        languages = self.config.filters.get("language", ["python"])
        
        for language in languages:
            if collected >= max_items:
                break
            
            # Search for trending repositories
            async for item in self._collect_trending_repos(language, since):
                yield item
                collected += 1
                if collected >= max_items:
                    break
            
            # Search for documentation repositories
            async for item in self._collect_documentation(language, since):
                yield item
                collected += 1
                if collected >= max_items:
                    break
    
    async def _collect_trending_repos(
        self,
        language: str,
        since: Optional[datetime] = None,
    ) -> AsyncIterator[CollectedItem]:
        """Collect content from trending repositories."""
        # Build search query
        query = f"language:{language} stars:>100"
        if since:
            query += f" pushed:>{since.strftime('%Y-%m-%d')}"
        
        params = {
            "q": query,
            "sort": "updated",
            "order": "desc",
            "per_page": 30,
        }
        
        url = f"{self.API_BASE}/search/repositories"
        response = await self._make_request("GET", url, params=params)
        
        if not response or "items" not in response:
            return
        
        for repo in response["items"]:
            # Fetch README content
            readme_item = await self._fetch_readme(repo)
            if readme_item:
                yield readme_item
            
            # Fetch documentation files
            async for doc_item in self._fetch_docs(repo):
                yield doc_item
    
    async def _collect_documentation(
        self,
        language: str,
        since: Optional[datetime] = None,
    ) -> AsyncIterator[CollectedItem]:
        """Collect from documentation-focused repositories."""
        # Search for awesome lists and documentation repos
        query = f"awesome-{language} OR {language}-guide OR {language}-tutorial"
        
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": 10,
        }
        
        url = f"{self.API_BASE}/search/repositories"
        response = await self._make_request("GET", url, params=params)
        
        if not response or "items" not in response:
            return
        
        for repo in response["items"]:
            readme_item = await self._fetch_readme(repo)
            if readme_item:
                yield readme_item
    
    async def _fetch_readme(
        self,
        repo: Dict[str, Any],
    ) -> Optional[CollectedItem]:
        """Fetch README content from a repository."""
        owner = repo["owner"]["login"]
        repo_name = repo["name"]
        
        url = f"{self.API_BASE}/repos/{owner}/{repo_name}/readme"
        response = await self._make_request("GET", url)
        
        if not response:
            return None
        
        # Decode content (base64)
        import base64
        try:
            content = base64.b64decode(response.get("content", "")).decode("utf-8")
        except Exception:
            return None
        
        return CollectedItem(
            source="github",
            source_id=f"{owner}/{repo_name}/README",
            url=repo["html_url"],
            title=f"{repo_name} - README",
            content=content,
            content_type=ContentType.DOCUMENTATION,
            language=repo.get("language"),
            tags=repo.get("topics", []),
            author=owner,
            created_at=self._parse_datetime(repo.get("created_at")),
            updated_at=self._parse_datetime(repo.get("updated_at")),
            metadata={
                "stars": repo.get("stargazers_count", 0),
                "forks": repo.get("forks_count", 0),
                "license": repo.get("license", {}).get("name") if repo.get("license") else None,
                "description": repo.get("description"),
            },
        )
    
    async def _fetch_docs(
        self,
        repo: Dict[str, Any],
    ) -> AsyncIterator[CollectedItem]:
        """Fetch documentation files from a repository."""
        owner = repo["owner"]["login"]
        repo_name = repo["name"]
        
        # Check for docs directory
        url = f"{self.API_BASE}/repos/{owner}/{repo_name}/contents/docs"
        response = await self._make_request("GET", url)
        
        if not response or not isinstance(response, list):
            return
        
        for file_info in response[:5]:  # Limit files per repo
            if file_info.get("type") != "file":
                continue
            
            name = file_info.get("name", "")
            if not name.endswith((".md", ".rst", ".txt")):
                continue
            
            # Fetch file content
            file_url = file_info.get("download_url")
            if not file_url:
                continue
            
            try:
                async with self._session.get(file_url) as resp:
                    if resp.status == 200:
                        content = await resp.text()
                        yield CollectedItem(
                            source="github",
                            source_id=f"{owner}/{repo_name}/docs/{name}",
                            url=file_info.get("html_url", file_url),
                            title=f"{repo_name} - {name}",
                            content=content,
                            content_type=ContentType.DOCUMENTATION,
                            language=repo.get("language"),
                            tags=repo.get("topics", []),
                            author=owner,
                            updated_at=self._parse_datetime(repo.get("updated_at")),
                            metadata={
                                "path": file_info.get("path"),
                                "size": file_info.get("size"),
                            },
                        )
            except Exception as e:
                logger.debug(f"Failed to fetch {file_url}: {e}")
    
    def _parse_item(self, raw_data: Dict[str, Any]) -> Optional[CollectedItem]:
        """Parse raw GitHub API response."""
        # Generic parser - specific parsing in collection methods
        return None
    
    @staticmethod
    def _parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
        """Parse GitHub datetime string."""
        if not dt_str:
            return None
        try:
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except Exception:
            return None
