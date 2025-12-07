"""
Caching_V1 - Semantic Cache

Code-aware caching with semantic deduplication.
"""

import logging
import hashlib
import re
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


@dataclass
class SemanticEntry:
    """Semantic cache entry"""
    code_hash: str
    normalized_hash: str
    result: Any
    created_at: datetime
    expires_at: datetime
    language: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticCache:
    """
    Semantic code caching.

    Features:
    - Code normalization (ignores whitespace, comments)
    - Hash-based deduplication
    - Language-aware processing
    """

    def __init__(self, ttl_seconds: int = 3600, max_entries: int = 5000):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries

        # Cache by normalized hash
        self._cache: Dict[str, SemanticEntry] = {}

        # Index: code_hash -> normalized_hash
        self._hash_index: Dict[str, str] = {}

        # Stats
        self._hits = 0
        self._misses = 0
        self._dedup_hits = 0

    def get(self, code: str, language: str = "python") -> Optional[Any]:
        """Get cached result for code"""
        code_hash = self._hash_code(code)

        # Check direct hash
        if code_hash in self._hash_index:
            normalized_hash = self._hash_index[code_hash]
            entry = self._cache.get(normalized_hash)

            if entry and not self._is_expired(entry):
                self._hits += 1
                return entry.result

        # Check normalized hash
        normalized = self._normalize_code(code, language)
        normalized_hash = self._hash_code(normalized)

        entry = self._cache.get(normalized_hash)

        if entry and not self._is_expired(entry):
            # Add index for this code variant
            self._hash_index[code_hash] = normalized_hash
            self._hits += 1
            self._dedup_hits += 1
            return entry.result

        self._misses += 1
        return None

    def set(
        self,
        code: str,
        result: Any,
        language: str = "python",
        metadata: Optional[Dict] = None,
    ):
        """Cache result for code"""
        # Evict if necessary
        if len(self._cache) >= self.max_entries:
            self._evict_oldest()

        code_hash = self._hash_code(code)
        normalized = self._normalize_code(code, language)
        normalized_hash = self._hash_code(normalized)

        now = datetime.now(timezone.utc)

        entry = SemanticEntry(
            code_hash=code_hash,
            normalized_hash=normalized_hash,
            result=result,
            created_at=now,
            expires_at=now + timedelta(seconds=self.ttl_seconds),
            language=language,
            metadata=metadata or {},
        )

        self._cache[normalized_hash] = entry
        self._hash_index[code_hash] = normalized_hash

    def invalidate(self, code: str):
        """Invalidate cache for code"""
        code_hash = self._hash_code(code)

        if code_hash in self._hash_index:
            normalized_hash = self._hash_index.pop(code_hash)
            self._cache.pop(normalized_hash, None)

    def _normalize_code(self, code: str, language: str) -> str:
        """Normalize code for semantic comparison"""
        normalized = code

        # Remove comments
        if language == "python":
            # Remove # comments
            normalized = re.sub(r'#.*$', '', normalized, flags=re.MULTILINE)
            # Remove docstrings (simplified)
            normalized = re.sub(r'""".*?"""', '', normalized, flags=re.DOTALL)
            normalized = re.sub(r"'''.*?'''", '', normalized, flags=re.DOTALL)
        elif language in ["javascript", "typescript", "java", "go"]:
            # Remove // comments
            normalized = re.sub(r'//.*$', '', normalized, flags=re.MULTILINE)
            # Remove /* */ comments
            normalized = re.sub(r'/\*.*?\*/', '', normalized, flags=re.DOTALL)

        # Normalize whitespace
        lines = [line.strip() for line in normalized.split('\n')]
        lines = [line for line in lines if line]  # Remove empty lines
        normalized = '\n'.join(lines)

        # Normalize multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized)

        return normalized.strip()

    def _hash_code(self, code: str) -> str:
        """Generate hash for code"""
        return hashlib.sha256(code.encode()).hexdigest()

    def _is_expired(self, entry: SemanticEntry) -> bool:
        """Check if entry is expired"""
        return datetime.now(timezone.utc) > entry.expires_at

    def _evict_oldest(self):
        """Evict oldest entry"""
        if not self._cache:
            return

        oldest_hash = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].created_at
        )

        entry = self._cache.pop(oldest_hash)

        # Clean up index
        self._hash_index = {
            k: v for k, v in self._hash_index.items()
            if v != oldest_hash
        }

    def cleanup_expired(self):
        """Remove expired entries"""
        now = datetime.now(timezone.utc)

        expired = [
            h for h, entry in self._cache.items()
            if entry.expires_at < now
        ]

        for h in expired:
            self._cache.pop(h)

        # Clean up index
        self._hash_index = {
            k: v for k, v in self._hash_index.items()
            if v in self._cache
        }

        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired semantic cache entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "entries": len(self._cache),
            "index_size": len(self._hash_index),
            "hits": self._hits,
            "misses": self._misses,
            "dedup_hits": self._dedup_hits,
            "hit_rate": self._hits / max(1, self._hits + self._misses),
            "dedup_rate": self._dedup_hits / max(1, self._hits),
        }
