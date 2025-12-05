"""
Redis Caching Decorator

Provides easy-to-use decorators for caching function results in Redis.
Supports async functions with automatic serialization.
"""

import json
import hashlib
import functools
import logging
from typing import Callable, Optional, Any, Union
from datetime import timedelta
import redis.asyncio as redis

from app.core.config import settings

logger = logging.getLogger(__name__)


# Global Redis connection
_redis_client: Optional[redis.Redis] = None


async def get_redis() -> redis.Redis:
    """Get or create Redis connection"""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
    return _redis_client


async def close_redis():
    """Close Redis connection"""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None


def _generate_cache_key(
    prefix: str,
    func: Callable,
    args: tuple,
    kwargs: dict,
    key_builder: Optional[Callable] = None,
) -> str:
    """Generate a unique cache key"""
    if key_builder:
        custom_key = key_builder(*args, **kwargs)
        return f"{prefix}:{func.__name__}:{custom_key}"
    
    # Default key generation
    key_parts = [prefix, func.__module__, func.__name__]
    
    # Add args
    for arg in args:
        if hasattr(arg, "id"):
            key_parts.append(str(arg.id))
        elif isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        else:
            key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:8])
    
    # Add kwargs
    for k, v in sorted(kwargs.items()):
        if isinstance(v, (str, int, float, bool)):
            key_parts.append(f"{k}={v}")
        else:
            key_parts.append(f"{k}={hashlib.md5(str(v).encode()).hexdigest()[:8]}")
    
    return ":".join(key_parts)


def cached(
    ttl: Union[int, timedelta] = 300,
    prefix: str = "cache",
    key_builder: Optional[Callable] = None,
    skip_cache_if: Optional[Callable] = None,
):
    """
    Cache decorator for async functions.
    
    Args:
        ttl: Time to live in seconds or timedelta
        prefix: Cache key prefix
        key_builder: Custom function to build cache key
        skip_cache_if: Function that returns True to skip caching
    
    Usage:
        @cached(ttl=300, prefix="projects")
        async def get_project(project_id: str):
            return await db.query(Project).get(project_id)
        
        @cached(ttl=timedelta(hours=1), key_builder=lambda user_id: f"user:{user_id}")
        async def get_user_stats(user_id: str):
            return await calculate_stats(user_id)
    """
    if isinstance(ttl, timedelta):
        ttl_seconds = int(ttl.total_seconds())
    else:
        ttl_seconds = ttl
    
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Check if we should skip caching
            if skip_cache_if and skip_cache_if(*args, **kwargs):
                return await func(*args, **kwargs)
            
            # Generate cache key
            cache_key = _generate_cache_key(prefix, func, args, kwargs, key_builder)
            
            try:
                # Try to get from cache
                r = await get_redis()
                cached_value = await r.get(cache_key)
                
                if cached_value is not None:
                    return json.loads(cached_value)
            except Exception as e:
                # If Redis fails, just execute the function
                logger.debug(f"Cache read failed: {type(e).__name__}")
            
            # Execute function
            result = await func(*args, **kwargs)
            
            try:
                # Store in cache
                r = await get_redis()
                await r.setex(
                    cache_key,
                    ttl_seconds,
                    json.dumps(result, default=str)
                )
            except Exception as e:
                # If Redis fails, just return the result
                logger.debug(f"Cache write failed: {type(e).__name__}")
            
            return result
        
        # Add cache invalidation helper
        async def invalidate(*args, **kwargs):
            """Invalidate cache for specific arguments"""
            cache_key = _generate_cache_key(prefix, func, args, kwargs, key_builder)
            try:
                r = await get_redis()
                await r.delete(cache_key)
            except Exception as e:
                logger.debug(f"Cache invalidation failed: {type(e).__name__}")
        
        wrapper.invalidate = invalidate
        return wrapper
    
    return decorator


def cached_property(ttl: int = 3600, prefix: str = "prop"):
    """
    Cache decorator for property-like methods.
    
    Usage:
        class User:
            @cached_property(ttl=3600)
            async def stats(self):
                return await expensive_calculation()
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Use object id as part of key
            obj_id = getattr(self, "id", id(self))
            cache_key = f"{prefix}:{func.__name__}:{obj_id}"
            
            try:
                r = await get_redis()
                cached_value = await r.get(cache_key)
                
                if cached_value is not None:
                    return json.loads(cached_value)
            except Exception as e:
                logger.debug(f"Property cache read failed: {type(e).__name__}")
            
            result = await func(self, *args, **kwargs)
            
            try:
                r = await get_redis()
                await r.setex(cache_key, ttl, json.dumps(result, default=str))
            except Exception as e:
                logger.debug(f"Property cache write failed: {type(e).__name__}")
            
            return result
        
        return wrapper
    
    return decorator


async def invalidate_pattern(pattern: str):
    """
    Invalidate all cache keys matching a pattern.
    
    Usage:
        await invalidate_pattern("projects:*")
        await invalidate_pattern("user:123:*")
    """
    try:
        r = await get_redis()
        cursor = 0
        while True:
            cursor, keys = await r.scan(cursor=cursor, match=pattern, count=100)
            if keys:
                await r.delete(*keys)
            if cursor == 0:
                break
    except Exception:
        pass


# Usage examples:
"""
from shared.utils.cache_decorator import cached, invalidate_pattern

# Basic caching
@cached(ttl=300)
async def get_project(project_id: str):
    return await db.get(Project, project_id)

# Custom key builder
@cached(
    ttl=3600,
    prefix="user_projects",
    key_builder=lambda user_id, status=None: f"{user_id}:{status or 'all'}"
)
async def get_user_projects(user_id: str, status: str = None):
    return await db.query(Project).filter_by(user_id=user_id, status=status)

# Skip caching for certain conditions
@cached(
    ttl=60,
    skip_cache_if=lambda request: request.headers.get("Cache-Control") == "no-cache"
)
async def get_data(request):
    return await fetch_data()

# Invalidate cache
await get_project.invalidate("project-123")
await invalidate_pattern("user:123:*")
"""
