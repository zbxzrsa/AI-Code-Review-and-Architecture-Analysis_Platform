# Caching_V2 API Reference

## Overview

Production caching with semantic deduplication and cache warming.

## Classes

### CacheManager

Multi-level cache management (inherited from V1).

```python
from modules.Caching_V2.src.cache_manager import CacheManager

cache = CacheManager(
    l1_ttl=300,   # 5 min
    l2_ttl=3600,  # 1 hour
    l3_ttl=86400  # 24 hours
)

# Set with level
cache.set("user:123", user_data, level="l2")

# Cascading get (checks L1 -> L2 -> L3)
data = cache.get_cascading("user:123")

# Get stats
stats = cache.get_stats()
```

### SemanticCache

Code-aware caching with normalization (inherited from V1).

```python
from modules.Caching_V2.src.semantic_cache import SemanticCache

semantic = SemanticCache(ttl_seconds=3600)

# Cache code analysis result
code = "def hello(): print('hi')"
semantic.set(code, analysis_result, language="python")

# Get (matches normalized code)
result = semantic.get(similar_code, language="python")
```

### CacheWarmer (V2 Feature)

Proactive cache warming and refresh-ahead.

```python
from modules.Caching_V2.src.cache_warmer import CacheWarmer

async def cache_setter(key, value, ttl):
    await redis.set(key, value, ex=ttl)

warmer = CacheWarmer(cache_setter, max_concurrent_warmers=5)

# Register warming task
async def load_user_data():
    return await db.get_user_stats()

warmer.register_task(
    key="user:stats",
    loader=load_user_data,
    ttl_seconds=3600,
    priority=1,
    refresh_before_seconds=60
)

# Start background warming
await warmer.start_background_warming(interval_seconds=30)

# Manual warm
await warmer.warm_all()
await warmer.warm_expired()
```

## Configuration

See `config/caching_config.yaml`
