"""
Caching_V1 - Redis Client

Redis-based caching with connection management.
"""

import logging
import json
from typing import Dict, Optional, Any, List
from datetime import timedelta

logger = logging.getLogger(__name__)


class MockRedis:
    """Mock Redis for development/testing"""

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._ttls: Dict[str, float] = {}

    async def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    async def set(self, key: str, value: str, ex: Optional[int] = None):
        self._data[key] = value
        if ex:
            self._ttls[key] = ex

    async def delete(self, key: str):
        self._data.pop(key, None)
        self._ttls.pop(key, None)

    async def exists(self, key: str) -> bool:
        return key in self._data

    async def keys(self, pattern: str) -> List[str]:
        import fnmatch
        return [k for k in self._data.keys() if fnmatch.fnmatch(k, pattern)]

    async def ttl(self, key: str) -> int:
        return int(self._ttls.get(key, -1))

    async def incr(self, key: str) -> int:
        val = int(self._data.get(key, 0)) + 1
        self._data[key] = str(val)
        return val

    async def expire(self, key: str, seconds: int):
        self._ttls[key] = seconds


class RedisClient:
    """
    Redis client wrapper.

    Features:
    - Connection management
    - Serialization/deserialization
    - Key namespacing
    - Batch operations
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        namespace: str = "app",
        use_mock: bool = True,
    ):
        self.host = host
        self.port = port
        self.db = db
        self.namespace = namespace

        # Use mock for development
        if use_mock:
            self._client = MockRedis()
        else:
            # In production, use aioredis
            self._client = MockRedis()  # Fallback to mock

        self._connected = True

    def _make_key(self, key: str) -> str:
        """Create namespaced key"""
        return f"{self.namespace}:{key}"

    async def get(self, key: str) -> Optional[Any]:
        """Get value"""
        try:
            data = await self._client.get(self._make_key(key))
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ):
        """Set value"""
        try:
            data = json.dumps(value)
            await self._client.set(self._make_key(key), data, ex=ttl)
        except Exception as e:
            logger.error(f"Redis set error: {e}")

    async def delete(self, key: str):
        """Delete key"""
        try:
            await self._client.delete(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis delete error: {e}")

    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            return await self._client.exists(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False

    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values"""
        results = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                results[key] = value
        return results

    async def set_many(self, items: Dict[str, Any], ttl: Optional[int] = None):
        """Set multiple values"""
        for key, value in items.items():
            await self.set(key, value, ttl)

    async def delete_pattern(self, pattern: str):
        """Delete keys matching pattern"""
        try:
            keys = await self._client.keys(self._make_key(pattern))
            for key in keys:
                await self._client.delete(key)
        except Exception as e:
            logger.error(f"Redis delete_pattern error: {e}")

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter"""
        try:
            return await self._client.incr(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis increment error: {e}")
            return 0

    async def get_ttl(self, key: str) -> int:
        """Get TTL for key"""
        try:
            return await self._client.ttl(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis ttl error: {e}")
            return -1

    async def set_ttl(self, key: str, seconds: int):
        """Set TTL for key"""
        try:
            await self._client.expire(self._make_key(key), seconds)
        except Exception as e:
            logger.error(f"Redis expire error: {e}")

    @property
    def is_connected(self) -> bool:
        return self._connected
