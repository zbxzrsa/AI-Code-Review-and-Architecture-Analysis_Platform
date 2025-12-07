"""
Duplicate Detection for Collected Data

Uses content hashing and similarity scoring to identify duplicates.
"""

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ..collectors.base import CollectedItem

logger = logging.getLogger(__name__)


@dataclass
class DuplicateGroup:
    """Group of duplicate items."""
    canonical_id: str  # ID of the item to keep
    duplicate_ids: List[str]  # IDs of duplicates
    similarity: float  # Similarity score


class DuplicateDetector:
    """
    Detects and handles duplicate content.
    
    Strategies:
    1. Exact hash matching (fast)
    2. Shingle-based similarity (approximate)
    3. URL-based deduplication
    
    Threshold: 0.85 similarity for duplicate detection
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        shingle_size: int = 5,
    ):
        self.similarity_threshold = similarity_threshold
        self.shingle_size = shingle_size
        
        # Index structures
        self._hash_index: Dict[str, str] = {}  # content_hash -> unique_id
        self._url_index: Dict[str, str] = {}   # normalized_url -> unique_id
        self._shingle_index: Dict[str, Set[str]] = defaultdict(set)  # shingle -> unique_ids
    
    def add_to_index(self, item: CollectedItem):
        """Add item to deduplication index."""
        unique_id = item.unique_id
        
        # Add to hash index
        self._hash_index[item.content_hash] = unique_id
        
        # Add to URL index
        normalized_url = self._normalize_url(item.url)
        self._url_index[normalized_url] = unique_id
        
        # Add to shingle index
        shingles = self._compute_shingles(item.content)
        for shingle in shingles:
            self._shingle_index[shingle].add(unique_id)
    
    def find_duplicate(self, item: CollectedItem) -> Optional[str]:
        """
        Find if item is a duplicate of existing content.
        
        Args:
            item: Item to check
            
        Returns:
            unique_id of existing duplicate, or None
        """
        # Check exact hash match
        if item.content_hash in self._hash_index:
            return self._hash_index[item.content_hash]
        
        # Check URL match
        normalized_url = self._normalize_url(item.url)
        if normalized_url in self._url_index:
            return self._url_index[normalized_url]
        
        # Check shingle similarity
        similar_id = self._find_similar_by_shingles(item)
        if similar_id:
            return similar_id
        
        return None
    
    def _find_similar_by_shingles(self, item: CollectedItem) -> Optional[str]:
        """Find similar items using shingle comparison."""
        item_shingles = self._compute_shingles(item.content)
        
        if not item_shingles:
            return None
        
        # Count shingle overlaps with indexed items
        candidate_scores: Dict[str, int] = defaultdict(int)
        
        for shingle in item_shingles:
            for candidate_id in self._shingle_index.get(shingle, set()):
                candidate_scores[candidate_id] += 1
        
        if not candidate_scores:
            return None
        
        # Find best match
        best_id = max(candidate_scores.keys(), key=lambda k: candidate_scores[k])
        overlap = candidate_scores[best_id]
        
        # Estimate Jaccard similarity
        # This is approximate - assumes similar shingle counts
        similarity = overlap / len(item_shingles)
        
        if similarity >= self.similarity_threshold:
            return best_id
        
        return None
    
    def _compute_shingles(self, text: str) -> Set[str]:
        """Compute shingles (n-grams) from text."""
        # Normalize text
        text = text.lower()
        words = text.split()
        
        if len(words) < self.shingle_size:
            return {" ".join(words)} if words else set()
        
        shingles = set()
        for i in range(len(words) - self.shingle_size + 1):
            shingle = " ".join(words[i:i + self.shingle_size])
            # Hash shingle for compact storage
            shingle_hash = hashlib.md5(shingle.encode()).hexdigest()[:8]
            shingles.add(shingle_hash)
        
        return shingles
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for comparison."""
        # Remove trailing slashes
        url = url.rstrip("/")
        
        # Remove common tracking parameters
        tracking_params = ["utm_source", "utm_medium", "utm_campaign", "ref", "source"]
        
        from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
        
        try:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            
            # Remove tracking params
            for param in tracking_params:
                params.pop(param, None)
            
            # Rebuild URL
            clean_query = urlencode(params, doseq=True)
            cleaned = urlunparse((
                parsed.scheme,
                parsed.netloc.lower(),
                parsed.path,
                parsed.params,
                clean_query,
                "",  # Remove fragment
            ))
            
            return cleaned
        except Exception:
            return url.lower()
    
    def deduplicate_batch(
        self,
        items: List[CollectedItem],
    ) -> Tuple[List[CollectedItem], List[DuplicateGroup]]:
        """
        Deduplicate a batch of items.
        
        Args:
            items: Items to deduplicate
            
        Returns:
            Tuple of (unique_items, duplicate_groups)
        """
        unique_items = []
        duplicate_groups: Dict[str, List[str]] = defaultdict(list)
        
        for item in items:
            existing_id = self.find_duplicate(item)
            
            if existing_id:
                duplicate_groups[existing_id].append(item.unique_id)
            else:
                self.add_to_index(item)
                unique_items.append(item)
        
        # Convert to DuplicateGroup objects
        groups = [
            DuplicateGroup(
                canonical_id=canonical,
                duplicate_ids=duplicates,
                similarity=self.similarity_threshold,
            )
            for canonical, duplicates in duplicate_groups.items()
            if duplicates
        ]
        
        logger.info(
            f"Deduplication: {len(unique_items)} unique, "
            f"{sum(len(g.duplicate_ids) for g in groups)} duplicates"
        )
        
        return unique_items, groups
    
    def clear_index(self):
        """Clear all indexes."""
        self._hash_index.clear()
        self._url_index.clear()
        self._shingle_index.clear()
    
    @property
    def index_size(self) -> int:
        """Number of items in index."""
        return len(self._hash_index)
