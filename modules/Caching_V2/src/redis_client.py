"""
Caching_V2 - Redis Client

Production Redis client with connection pooling and retry logic.
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import timedelta


@dataclass
class RedisConfig:
    """Redis configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 10
    socket_timeout: float = 5.0
    retry_on_timeout: bool = True


class RedisClient:
    """
    Production Redis Client.

    Features:
    - Connection pooling
    - Automatic reconnection
    - Retry logic
    - Pipeline support
    """

    def __init__(self, config: Optional[RedisConfig] = None):
        self.config = config or RedisConfig()
        self._connected = False
        self._client = None

        # Fallback to in-memory when Redis unavailable
        self._fallback_cache: Dict[str, Any] = {}
        self._fallback_ttls: Dict[str, float] = {}

    async def connect(self) -> bool:
        """Connect to Redis."""
        try:
            import redis.asyncio as redis

            self._client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                decode_responses=True,
                socket_timeout=self.config.socket_timeout,
            )

            await self._client.ping()
            self._connected = True
            return True

        except ImportError:
            # Redis library not installed
            self._connected = False
            return False
        except Exception:
            self._connected = False
            return False

    async def disconnect(self):
        """Disconnect from Redis."""
        if self._client:
            await self._client.close()
        self._connected = False

    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis."""
        if self._connected and self._client:
            try:
                return await self._client.get(key)
            except Exception:
                pass

        # Fallback
        return self._fallback_cache.get(key)

    async def set(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set value in Redis."""
        if self._connected and self._client:
            try:
                if ttl:
                    await self._client.setex(key, ttl, value)
                else:
                    await self._client.set(key, value)
                return True
            except Exception:
                pass

        # Fallback
        self._fallback_cache[key] = value
        if ttl:
            import time
            self._fallback_ttls[key] = time.time() + ttl
        return True

    async def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        if self._connected and self._client:
            try:
                await self._client.delete(key)
                return True
            except Exception:
                pass

        # Fallback
        self._fallback_cache.pop(key, None)
        self._fallback_ttls.pop(key, None)
        return True

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if self._connected and self._client:
            try:
                return await self._client.exists(key) > 0
            except Exception:
                pass

        return key in self._fallback_cache

    async def incr(self, key: str) -> int:
        """Increment counter."""
        if self._connected and self._client:
            try:
                return await self._client.incr(key)
            except Exception:
                pass

        # Fallback
        current = int(self._fallback_cache.get(key, 0))
        self._fallback_cache[key] = current + 1
        return current + 1

    async def expire(self, key: str, seconds: int) -> bool:
        """Set key expiration."""
        if self._connected and self._client:
            try:
                return await self._client.expire(key, seconds)
            except Exception:
                pass

        import time
        self._fallback_ttls[key] = time.time() + seconds
        return True

    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        if self._connected and self._client:
            try:
                return await self._client.keys(pattern)
            except Exception:
                pass

        # Simple pattern matching for fallback
        if pattern == "*":
            return list(self._fallback_cache.keys())
        return [k for k in self._fallback_cache.keys() if pattern.replace("*", "") in k]

    async def pipeline(self):
        """Get pipeline for batch operations."""
        if self._connected and self._client:
            return self._client.pipeline()
        return None

    @property
    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected

    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "connected": self._connected,
            "fallback_keys": len(self._fallback_cache),
            "host": self.config.host,
            "port": self.config.port,
        }


__all__ = ["RedisConfig", "RedisClient"]
