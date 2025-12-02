"""
Redis client with multi-level caching strategy.

Cache hierarchy:
- L1: Session cache (5 minutes) - User-specific, frequently accessed
- L2: Project cache (1 hour) - Project-specific analysis results
- L3: Global cache (24 hours) - Model responses, provider health
"""
import redis
import json
import hashlib
import logging
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
from functools import wraps

logger = logging.getLogger(__name__)


class RedisClient:
    """Redis client with multi-level caching."""

    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        """Initialize Redis client."""
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True,
            health_check_interval=30
        )
        self.pipeline = None

    def health_check(self) -> bool:
        """Check Redis connection health."""
        try:
            self.client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    # ============================================
    # L1: Session Cache (5 minutes)
    # ============================================

    def set_session_cache(
        self,
        session_id: str,
        key: str,
        value: Any,
        ttl: int = 300
    ) -> bool:
        """Set session-level cache (5 minutes default)."""
        try:
            cache_key = f"session:{session_id}:{key}"
            self.client.setex(
                cache_key,
                ttl,
                json.dumps(value) if not isinstance(value, str) else value
            )
            return True
        except Exception as e:
            logger.error(f"Session cache set failed: {e}")
            return False

    def get_session_cache(self, session_id: str, key: str) -> Optional[Any]:
        """Get session-level cache."""
        try:
            cache_key = f"session:{session_id}:{key}"
            value = self.client.get(cache_key)
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return None
        except Exception as e:
            logger.error(f"Session cache get failed: {e}")
            return None

    def delete_session_cache(self, session_id: str, key: str) -> bool:
        """Delete session-level cache."""
        try:
            cache_key = f"session:{session_id}:{key}"
            self.client.delete(cache_key)
            return True
        except Exception as e:
            logger.error(f"Session cache delete failed: {e}")
            return False

    def clear_session(self, session_id: str) -> bool:
        """Clear all session caches."""
        try:
            pattern = f"session:{session_id}:*"
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Session clear failed: {e}")
            return False

    # ============================================
    # L2: Project Cache (1 hour)
    # ============================================

    def set_project_cache(
        self,
        project_id: str,
        key: str,
        value: Any,
        ttl: int = 3600
    ) -> bool:
        """Set project-level cache (1 hour default)."""
        try:
            cache_key = f"project:{project_id}:{key}"
            self.client.setex(
                cache_key,
                ttl,
                json.dumps(value) if not isinstance(value, str) else value
            )
            return True
        except Exception as e:
            logger.error(f"Project cache set failed: {e}")
            return False

    def get_project_cache(self, project_id: str, key: str) -> Optional[Any]:
        """Get project-level cache."""
        try:
            cache_key = f"project:{project_id}:{key}"
            value = self.client.get(cache_key)
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return None
        except Exception as e:
            logger.error(f"Project cache get failed: {e}")
            return None

    def delete_project_cache(self, project_id: str, key: str) -> bool:
        """Delete project-level cache."""
        try:
            cache_key = f"project:{project_id}:{key}"
            self.client.delete(cache_key)
            return True
        except Exception as e:
            logger.error(f"Project cache delete failed: {e}")
            return False

    # ============================================
    # L3: Global Cache (24 hours)
    # ============================================

    def set_global_cache(
        self,
        key: str,
        value: Any,
        ttl: int = 86400
    ) -> bool:
        """Set global cache (24 hours default)."""
        try:
            self.client.setex(
                key,
                ttl,
                json.dumps(value) if not isinstance(value, str) else value
            )
            return True
        except Exception as e:
            logger.error(f"Global cache set failed: {e}")
            return False

    def get_global_cache(self, key: str) -> Optional[Any]:
        """Get global cache."""
        try:
            value = self.client.get(key)
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return None
        except Exception as e:
            logger.error(f"Global cache get failed: {e}")
            return None

    # ============================================
    # Semantic Deduplication
    # ============================================

    def cache_analysis_result(
        self,
        code: str,
        result: Dict[str, Any],
        project_id: str,
        ttl: int = 3600
    ) -> str:
        """Cache analysis result with semantic deduplication."""
        try:
            # Create hash of code for deduplication
            code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]
            cache_key = f"project:{project_id}:analysis:{code_hash}"

            self.client.setex(cache_key, ttl, json.dumps(result))
            return code_hash
        except Exception as e:
            logger.error(f"Analysis result caching failed: {e}")
            return ""

    def get_cached_analysis(
        self,
        code: str,
        project_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached analysis result by code hash."""
        try:
            code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]
            cache_key = f"project:{project_id}:analysis:{code_hash}"
            value = self.client.get(cache_key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Analysis result retrieval failed: {e}")
            return None

    # ============================================
    # Rate Limiting
    # ============================================

    def check_rate_limit(
        self,
        user_id: str,
        limit: int = 100,
        window: int = 86400
    ) -> bool:
        """Check if user is within rate limit."""
        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            key = f"ratelimit:{user_id}:requests:{date_str}"

            current = self.client.get(key)
            if current and int(current) >= limit:
                logger.warning(f"Rate limit exceeded for user {user_id}")
                return False

            # Increment counter
            pipe = self.client.pipeline()
            pipe.incr(key)
            pipe.expire(key, window)
            pipe.execute()

            return True
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True  # Allow on error

    def get_rate_limit_status(self, user_id: str) -> Dict[str, Any]:
        """Get rate limit status for user."""
        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            key = f"ratelimit:{user_id}:requests:{date_str}"
            current = self.client.get(key)
            ttl = self.client.ttl(key)

            return {
                "requests_used": int(current) if current else 0,
                "reset_in_seconds": ttl if ttl > 0 else 0
            }
        except Exception as e:
            logger.error(f"Rate limit status check failed: {e}")
            return {"requests_used": 0, "reset_in_seconds": 0}

    # ============================================
    # Provider Health Monitoring
    # ============================================

    def set_provider_health(
        self,
        provider: str,
        health_status: Dict[str, Any],
        ttl: int = 300
    ) -> bool:
        """Set provider health status."""
        try:
            key = f"provider:{provider}:health_status"
            self.client.setex(key, ttl, json.dumps(health_status))
            return True
        except Exception as e:
            logger.error(f"Provider health set failed: {e}")
            return False

    def get_provider_health(self, provider: str) -> Optional[Dict[str, Any]]:
        """Get provider health status."""
        try:
            key = f"provider:{provider}:health_status"
            value = self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Provider health get failed: {e}")
            return None

    # ============================================
    # Pub/Sub
    # ============================================

    def publish_event(self, channel: str, message: Dict[str, Any]) -> int:
        """Publish event to channel."""
        try:
            return self.client.publish(channel, json.dumps(message))
        except Exception as e:
            logger.error(f"Event publish failed: {e}")
            return 0

    def subscribe(self, channels: list) -> Any:
        """Subscribe to channels."""
        try:
            pubsub = self.client.pubsub()
            pubsub.subscribe(channels)
            return pubsub
        except Exception as e:
            logger.error(f"Subscription failed: {e}")
            return None

    # ============================================
    # Quota Management
    # ============================================

    def increment_quota(
        self,
        user_id: str,
        quota_type: str,
        amount: int = 1
    ) -> int:
        """Increment user quota."""
        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            key = f"quota:{user_id}:{quota_type}:{date_str}"

            pipe = self.client.pipeline()
            pipe.incrby(key, amount)
            pipe.expire(key, 86400)  # 24 hours
            result = pipe.execute()

            return result[0]
        except Exception as e:
            logger.error(f"Quota increment failed: {e}")
            return 0

    def get_quota_usage(self, user_id: str, quota_type: str) -> int:
        """Get quota usage for user."""
        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            key = f"quota:{user_id}:{quota_type}:{date_str}"
            value = self.client.get(key)
            return int(value) if value else 0
        except Exception as e:
            logger.error(f"Quota usage get failed: {e}")
            return 0

    # ============================================
    # Distributed Locks
    # ============================================

    def acquire_lock(
        self,
        lock_name: str,
        timeout: int = 30,
        blocking: bool = True
    ) -> bool:
        """Acquire distributed lock."""
        try:
            lock_key = f"lock:{lock_name}"
            lock_value = str(datetime.now().timestamp())

            if blocking:
                # Blocking acquire
                while True:
                    if self.client.set(lock_key, lock_value, nx=True, ex=timeout):
                        return True
                    import time
                    time.sleep(0.1)
            else:
                # Non-blocking acquire
                return self.client.set(lock_key, lock_value, nx=True, ex=timeout)
        except Exception as e:
            logger.error(f"Lock acquire failed: {e}")
            return False

    def release_lock(self, lock_name: str) -> bool:
        """Release distributed lock."""
        try:
            lock_key = f"lock:{lock_name}"
            self.client.delete(lock_key)
            return True
        except Exception as e:
            logger.error(f"Lock release failed: {e}")
            return False

    # ============================================
    # Batch Operations
    # ============================================

    def mset_session_cache(
        self,
        session_id: str,
        data: Dict[str, Any],
        ttl: int = 300
    ) -> bool:
        """Set multiple session cache values."""
        try:
            pipe = self.client.pipeline()
            for key, value in data.items():
                cache_key = f"session:{session_id}:{key}"
                pipe.setex(
                    cache_key,
                    ttl,
                    json.dumps(value) if not isinstance(value, str) else value
                )
            pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Batch session cache set failed: {e}")
            return False

    def mget_session_cache(
        self,
        session_id: str,
        keys: list
    ) -> Dict[str, Any]:
        """Get multiple session cache values."""
        try:
            cache_keys = [f"session:{session_id}:{key}" for key in keys]
            values = self.client.mget(cache_keys)

            result = {}
            for key, value in zip(keys, values):
                if value:
                    try:
                        result[key] = json.loads(value)
                    except json.JSONDecodeError:
                        result[key] = value
            return result
        except Exception as e:
            logger.error(f"Batch session cache get failed: {e}")
            return {}

    # ============================================
    # Utility Methods
    # ============================================

    def delete_by_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Pattern delete failed: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get Redis statistics."""
        try:
            info = self.client.info()
            return {
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands": info.get("total_commands_processed"),
                "uptime_seconds": info.get("uptime_in_seconds")
            }
        except Exception as e:
            logger.error(f"Stats retrieval failed: {e}")
            return {}

    def flush_all(self) -> bool:
        """Flush all Redis data (use with caution)."""
        try:
            self.client.flushall()
            logger.warning("Redis flushed completely")
            return True
        except Exception as e:
            logger.error(f"Flush failed: {e}")
            return False


# Decorator for caching function results
def cache_result(ttl: int = 3600, key_prefix: str = ""):
    """Decorator to cache function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            redis_client = RedisClient()

            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{str(args)}:{str(kwargs)}"
            cache_key = hashlib.md5(cache_key.encode()).hexdigest()

            # Try to get from cache
            cached = redis_client.get_global_cache(cache_key)
            if cached is not None:
                return cached

            # Execute function and cache result
            result = func(*args, **kwargs)
            redis_client.set_global_cache(cache_key, result, ttl)

            return result
        return wrapper
    return decorator
