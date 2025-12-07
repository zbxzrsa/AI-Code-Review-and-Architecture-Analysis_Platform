"""
Caching_V2 - Semantic Cache

Production semantic cache for code analysis results.
"""

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional


@dataclass
class SemanticCacheEntry:
    """Semantic cache entry."""
    semantic_key: str
    original_code: str
    normalized_code: str
    language: str
    result: Any
    created_at: datetime
    expires_at: datetime
    similarity_score: float = 1.0


class SemanticCache:
    """
    Production Semantic Cache.

    Features:
    - Code normalization
    - Semantic key generation
    - Similarity-based lookup
    - Deduplication
    """

    def __init__(self, ttl: int = 3600, similarity_threshold: float = 0.9):
        self.ttl = ttl
        self.similarity_threshold = similarity_threshold

        self._cache: Dict[str, SemanticCacheEntry] = {}
        self._duplicates_found = 0

    def get(self, code: str, language: str = "python") -> Optional[Any]:
        """Get cached result for code."""
        normalized = self._normalize_code(code, language)
        semantic_key = self._generate_semantic_key(normalized, language)

        if semantic_key in self._cache:
            entry = self._cache[semantic_key]

            # Check expiration
            if entry.expires_at > datetime.now(timezone.utc):
                return entry.result
            else:
                del self._cache[semantic_key]

        return None

    def set(
        self,
        code: str,
        result: Any,
        language: str = "python",
        ttl: Optional[int] = None,
    ):
        """Cache result for code."""
        normalized = self._normalize_code(code, language)
        semantic_key = self._generate_semantic_key(normalized, language)

        now = datetime.now(timezone.utc)
        effective_ttl = ttl or self.ttl

        self._cache[semantic_key] = SemanticCacheEntry(
            semantic_key=semantic_key,
            original_code=code,
            normalized_code=normalized,
            language=language,
            result=result,
            created_at=now,
            expires_at=now + timedelta(seconds=effective_ttl),
        )

    def check_duplicate(self, code: str, language: str = "python") -> bool:
        """Check if code is semantically duplicate."""
        existing = self.get(code, language)
        if existing is not None:
            self._duplicates_found += 1
            return True
        return False

    def _normalize_code(self, code: str, language: str) -> str:
        """Normalize code for semantic comparison."""
        normalized = code

        # Remove comments
        if language == "python":
            # Remove single-line comments
            normalized = re.sub(r'#.*$', '', normalized, flags=re.MULTILINE)
            # Remove docstrings
            normalized = re.sub(r'""".*?"""', '', normalized, flags=re.DOTALL)
            normalized = re.sub(r"'''.*?'''", '', normalized, flags=re.DOTALL)

        elif language in ["javascript", "typescript", "java", "c", "cpp"]:
            # Remove single-line comments
            normalized = re.sub(r'//.*$', '', normalized, flags=re.MULTILINE)
            # Remove multi-line comments
            normalized = re.sub(r'/\*.*?\*/', '', normalized, flags=re.DOTALL)

        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = normalized.strip()

        # Lowercase for case-insensitive comparison
        normalized = normalized.lower()

        return normalized

    def _generate_semantic_key(self, normalized_code: str, language: str) -> str:
        """Generate semantic key from normalized code."""
        content = f"{language}:{normalized_code}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def find_similar(self, code: str, language: str = "python") -> List[SemanticCacheEntry]:
        """Find similar cached entries."""
        normalized = self._normalize_code(code, language)
        similar = []

        for entry in self._cache.values():
            if entry.language != language:
                continue

            # Simple similarity based on length ratio
            len_ratio = min(len(normalized), len(entry.normalized_code)) / max(len(normalized), len(entry.normalized_code), 1)

            if len_ratio >= self.similarity_threshold:
                similar.append(entry)

        return similar

    def clear_expired(self):
        """Clear expired entries."""
        now = datetime.now(timezone.utc)
        expired = [k for k, v in self._cache.items() if v.expires_at <= now]

        for key in expired:
            del self._cache[key]

        return len(expired)

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "entries": len(self._cache),
            "duplicates_found": self._duplicates_found,
            "similarity_threshold": self.similarity_threshold,
            "ttl_seconds": self.ttl,
        }


__all__ = ["SemanticCacheEntry", "SemanticCache"]
