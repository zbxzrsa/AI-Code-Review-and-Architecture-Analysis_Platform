"""
Cache strategies and patterns for different use cases.
"""
import logging
from typing import Any, Optional, Dict, Callable
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class CacheLevel(str, Enum):
    """Cache levels."""
    L1_SESSION = "session"  # 5 minutes
    L2_PROJECT = "project"  # 1 hour
    L3_GLOBAL = "global"  # 24 hours


class CacheStrategy:
    """Cache strategy patterns."""

    # ============================================
    # Cache Key Patterns
    # ============================================

    @staticmethod
    def session_key(session_id: str, key: str) -> str:
        """Session cache key pattern."""
        return f"session:{session_id}:{key}"

    @staticmethod
    def project_key(project_id: str, key: str) -> str:
        """Project cache key pattern."""
        return f"project:{project_id}:{key}"

    @staticmethod
    def analysis_key(project_id: str, code_hash: str) -> str:
        """Analysis result cache key."""
        return f"project:{project_id}:analysis:{code_hash}"

    @staticmethod
    def file_tree_key(project_id: str) -> str:
        """File tree cache key."""
        return f"project:{project_id}:file_tree"

    @staticmethod
    def recent_issues_key(project_id: str) -> str:
        """Recent issues cache key."""
        return f"project:{project_id}:recent_issues"

    @staticmethod
    def model_response_key(model: str, version: str, prompt_hash: str) -> str:
        """Model response cache key."""
        return f"model:{model}:{version}:{prompt_hash}:response"

    @staticmethod
    def provider_health_key(provider: str) -> str:
        """Provider health cache key."""
        return f"provider:{provider}:health_status"

    @staticmethod
    def ratelimit_key(user_id: str, date: str = None) -> str:
        """Rate limit cache key."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        return f"ratelimit:{user_id}:requests:{date}"

    @staticmethod
    def quota_key(user_id: str, quota_type: str, date: str = None) -> str:
        """Quota cache key."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        return f"quota:{user_id}:{quota_type}:{date}"

    @staticmethod
    def cost_tracking_key(user_id: str, date: str = None) -> str:
        """Cost tracking cache key."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        return f"cost:{user_id}:{date}"

    # ============================================
    # Cache TTL Configuration
    # ============================================

    @staticmethod
    def get_ttl(level: CacheLevel) -> int:
        """Get TTL for cache level."""
        ttl_map = {
            CacheLevel.L1_SESSION: 300,  # 5 minutes
            CacheLevel.L2_PROJECT: 3600,  # 1 hour
            CacheLevel.L3_GLOBAL: 86400,  # 24 hours
        }
        return ttl_map.get(level, 3600)

    # ============================================
    # Cache Invalidation Patterns
    # ============================================

    @staticmethod
    def invalidate_session_on_logout(redis_client, session_id: str) -> bool:
        """Invalidate all session caches on logout."""
        try:
            pattern = f"session:{session_id}:*"
            redis_client.delete_by_pattern(pattern)
            return True
        except Exception as e:
            logger.error(f"Session invalidation failed: {e}")
            return False

    @staticmethod
    def invalidate_project_on_change(redis_client, project_id: str) -> bool:
        """Invalidate project caches on project change."""
        try:
            pattern = f"project:{project_id}:*"
            redis_client.delete_by_pattern(pattern)
            return True
        except Exception as e:
            logger.error(f"Project invalidation failed: {e}")
            return False

    @staticmethod
    def invalidate_analysis_on_new_version(
        redis_client,
        project_id: str
    ) -> bool:
        """Invalidate analysis cache when new version is promoted."""
        try:
            pattern = f"project:{project_id}:analysis:*"
            redis_client.delete_by_pattern(pattern)
            return True
        except Exception as e:
            logger.error(f"Analysis invalidation failed: {e}")
            return False

    # ============================================
    # Cache Warming Patterns
    # ============================================

    @staticmethod
    def warm_project_cache(
        redis_client,
        project_id: str,
        file_tree: Dict[str, Any],
        recent_issues: list
    ) -> bool:
        """Pre-populate project cache."""
        try:
            redis_client.set_project_cache(
                project_id,
                "file_tree",
                file_tree,
                ttl=3600
            )
            redis_client.set_project_cache(
                project_id,
                "recent_issues",
                recent_issues,
                ttl=3600
            )
            return True
        except Exception as e:
            logger.error(f"Cache warming failed: {e}")
            return False

    # ============================================
    # Cache Hit/Miss Tracking
    # ============================================

    @staticmethod
    def track_cache_hit(redis_client, cache_key: str) -> bool:
        """Track cache hit."""
        try:
            metric_key = f"metrics:cache_hits:{cache_key}"
            redis_client.client.incr(metric_key)
            redis_client.client.expire(metric_key, 86400)
            return True
        except Exception as e:
            logger.error(f"Cache hit tracking failed: {e}")
            return False

    @staticmethod
    def track_cache_miss(redis_client, cache_key: str) -> bool:
        """Track cache miss."""
        try:
            metric_key = f"metrics:cache_misses:{cache_key}"
            redis_client.client.incr(metric_key)
            redis_client.client.expire(metric_key, 86400)
            return True
        except Exception as e:
            logger.error(f"Cache miss tracking failed: {e}")
            return False

    @staticmethod
    def get_cache_stats(redis_client) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            hits = redis_client.client.keys("metrics:cache_hits:*")
            misses = redis_client.client.keys("metrics:cache_misses:*")

            total_hits = sum(
                int(redis_client.client.get(key) or 0) for key in hits
            )
            total_misses = sum(
                int(redis_client.client.get(key) or 0) for key in misses
            )

            hit_rate = (
                total_hits / (total_hits + total_misses)
                if (total_hits + total_misses) > 0
                else 0
            )

            return {
                "total_hits": total_hits,
                "total_misses": total_misses,
                "hit_rate": hit_rate,
                "total_requests": total_hits + total_misses
            }
        except Exception as e:
            logger.error(f"Cache stats retrieval failed: {e}")
            return {}


class CachePatterns:
    """Common cache patterns."""

    @staticmethod
    def cache_aside(
        redis_client,
        cache_key: str,
        fetch_func: Callable,
        ttl: int = 3600
    ) -> Any:
        """
        Cache-aside pattern (lazy loading).

        1. Check cache
        2. If miss, fetch from source
        3. Store in cache
        4. Return value
        """
        try:
            # Try cache
            cached = redis_client.get_global_cache(cache_key)
            if cached is not None:
                CacheStrategy.track_cache_hit(redis_client, cache_key)
                return cached

            # Cache miss - fetch from source
            CacheStrategy.track_cache_miss(redis_client, cache_key)
            value = fetch_func()

            # Store in cache
            redis_client.set_global_cache(cache_key, value, ttl)

            return value
        except Exception as e:
            logger.error(f"Cache-aside pattern failed: {e}")
            return fetch_func()

    @staticmethod
    def write_through(
        redis_client,
        cache_key: str,
        value: Any,
        persist_func: Callable,
        ttl: int = 3600
    ) -> bool:
        """
        Write-through pattern.

        1. Write to cache
        2. Write to persistent storage
        3. Return success
        """
        try:
            # Write to cache
            redis_client.set_global_cache(cache_key, value, ttl)

            # Write to persistent storage
            persist_func(value)

            return True
        except Exception as e:
            logger.error(f"Write-through pattern failed: {e}")
            return False

    @staticmethod
    def write_behind(
        redis_client,
        cache_key: str,
        value: Any,
        queue_func: Callable,
        ttl: int = 3600
    ) -> bool:
        """
        Write-behind pattern (write-back).

        1. Write to cache immediately
        2. Queue write to persistent storage
        3. Return success
        """
        try:
            # Write to cache immediately
            redis_client.set_global_cache(cache_key, value, ttl)

            # Queue write to persistent storage
            queue_func(cache_key, value)

            return True
        except Exception as e:
            logger.error(f"Write-behind pattern failed: {e}")
            return False

    @staticmethod
    def refresh_ahead(
        redis_client,
        cache_key: str,
        fetch_func: Callable,
        ttl: int = 3600,
        refresh_threshold: float = 0.8
    ) -> Any:
        """
        Refresh-ahead pattern.

        Proactively refresh cache before expiration.
        """
        try:
            # Get cache with TTL
            value = redis_client.get_global_cache(cache_key)
            ttl_remaining = redis_client.client.ttl(cache_key)

            # If cache exists and TTL is above threshold, return
            if value and ttl_remaining > (ttl * refresh_threshold):
                return value

            # Refresh cache
            new_value = fetch_func()
            redis_client.set_global_cache(cache_key, new_value, ttl)

            return new_value
        except Exception as e:
            logger.error(f"Refresh-ahead pattern failed: {e}")
            return fetch_func()


class RateLimitStrategy:
    """Rate limiting strategies."""

    @staticmethod
    def token_bucket(
        redis_client,
        user_id: str,
        capacity: int = 100,
        refill_rate: int = 10,
        window: int = 60
    ) -> bool:
        """
        Token bucket rate limiting.

        Tokens are added at a fixed rate.
        Each request consumes one token.
        """
        try:
            key = f"token_bucket:{user_id}"
            bucket = redis_client.client.get(key)

            if bucket is None:
                # Initialize bucket
                redis_client.client.setex(key, window, capacity)
                return True

            tokens = int(bucket)
            if tokens > 0:
                redis_client.client.decr(key)
                return True

            return False
        except Exception as e:
            logger.error(f"Token bucket rate limiting failed: {e}")
            return True  # Allow on error

    @staticmethod
    def sliding_window(
        redis_client,
        user_id: str,
        limit: int = 100,
        window: int = 60
    ) -> bool:
        """
        Sliding window rate limiting.

        Counts requests in a rolling time window.
        """
        try:
            now = datetime.now().timestamp()
            window_start = now - window

            key = f"sliding_window:{user_id}"

            # Remove old entries
            redis_client.client.zremrangebyscore(key, 0, window_start)

            # Count requests in window
            count = redis_client.client.zcard(key)

            if count < limit:
                # Add new request
                redis_client.client.zadd(key, {str(now): now})
                redis_client.client.expire(key, window)
                return True

            return False
        except Exception as e:
            logger.error(f"Sliding window rate limiting failed: {e}")
            return True  # Allow on error

    @staticmethod
    def leaky_bucket(
        redis_client,
        user_id: str,
        capacity: int = 100,
        leak_rate: float = 1.0
    ) -> bool:
        """
        Leaky bucket rate limiting.

        Requests are processed at a fixed rate.
        Excess requests are dropped.
        """
        try:
            key = f"leaky_bucket:{user_id}"
            last_request = redis_client.client.get(f"{key}:last")

            now = datetime.now().timestamp()

            if last_request is None:
                # First request
                redis_client.client.setex(key, 3600, 1)
                redis_client.client.setex(f"{key}:last", 3600, now)
                return True

            # Calculate leaked tokens
            last_time = float(last_request)
            leaked = (now - last_time) * leak_rate

            current = int(redis_client.client.get(key) or 0)
            current = max(0, current - leaked)

            if current < capacity:
                redis_client.client.setex(key, 3600, current + 1)
                redis_client.client.setex(f"{key}:last", 3600, now)
                return True

            return False
        except Exception as e:
            logger.error(f"Leaky bucket rate limiting failed: {e}")
            return True  # Allow on error
