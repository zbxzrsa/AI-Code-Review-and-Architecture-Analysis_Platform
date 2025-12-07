"""Tests for Caching_V1"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cache_manager import CacheManager
from src.semantic_cache import SemanticCache


class TestCacheManager:
    @pytest.fixture
    def cache(self):
        return CacheManager()

    def test_set_and_get(self, cache):
        cache.set("test_key", "test_value", level="l2")
        result = cache.get("test_key", level="l2")
        assert result == "test_value"

    def test_get_cascading(self, cache):
        cache.set("cascade_key", "cascade_value", level="l3")
        result = cache.get_cascading("cascade_key")
        assert result == "cascade_value"

    def test_delete(self, cache):
        cache.set("delete_key", "value", level="l2")
        cache.delete("delete_key", level="l2")
        assert cache.get("delete_key", level="l2") is None

    def test_clear(self, cache):
        cache.set("key1", "value1", level="l2")
        cache.set("key2", "value2", level="l2")
        cache.clear(level="l2")
        assert cache.get("key1", level="l2") is None

    def test_stats(self, cache):
        cache.set("stat_key", "value", level="l2")
        cache.get("stat_key", level="l2")

        stats = cache.get_stats()
        assert "l2" in stats
        assert stats["l2"]["hits"] >= 1


class TestSemanticCache:
    @pytest.fixture
    def cache(self):
        return SemanticCache()

    def test_set_and_get(self, cache):
        code = "def hello():\n    print('hello')"
        cache.set(code, {"result": "ok"}, language="python")
        result = cache.get(code, language="python")
        assert result == {"result": "ok"}

    def test_normalized_match(self, cache):
        code1 = "def hello():\n    print('hello')  # comment"
        code2 = "def hello():\n    print('hello')"

        cache.set(code1, {"result": "ok"}, language="python")
        result = cache.get(code2, language="python")

        # Should find due to normalization
        assert result == {"result": "ok"}

    def test_invalidate(self, cache):
        code = "def test(): pass"
        cache.set(code, "result", language="python")
        cache.invalidate(code)
        assert cache.get(code, language="python") is None

    def test_stats(self, cache):
        code = "def test(): pass"
        cache.set(code, "result", language="python")
        cache.get(code, language="python")

        stats = cache.get_stats()
        assert stats["hits"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
