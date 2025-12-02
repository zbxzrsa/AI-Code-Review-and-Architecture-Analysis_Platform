# Redis Caching Strategy Documentation

## Overview

Multi-level caching architecture with Redis supporting session, project, and global caches with different TTLs and invalidation strategies.

---

## Cache Hierarchy

### L1: Session Cache (5 minutes)

User-specific, frequently accessed data.

**Use Cases**:

- Current code snippet being edited
- Current file being viewed
- User preferences
- Session state

**Key Pattern**: `session:{session_id}:{key}`

**Examples**:

```
session:abc123:code_snippet
session:abc123:current_file
session:abc123:user_preferences
```

**TTL**: 300 seconds (5 minutes)

### L2: Project Cache (1 hour)

Project-specific analysis results and metadata.

**Use Cases**:

- Analysis results (semantic deduplication by code hash)
- File tree structure
- Recent issues
- Baseline metrics

**Key Pattern**: `project:{project_id}:{key}`

**Examples**:

```
project:proj456:analysis:a1b2c3d4
project:proj456:file_tree
project:proj456:recent_issues
```

**TTL**: 3600 seconds (1 hour)

### L3: Global Cache (24 hours)

Shared, infrequently changing data.

**Use Cases**:

- Model responses (by prompt hash)
- Provider health status
- Rate limit counters
- Quota tracking

**Key Pattern**: `{type}:{identifier}:{metric}`

**Examples**:

```
model:gpt-4:v2.1.0:a1b2c3d4:response
provider:openai:health_status
ratelimit:user123:requests:2024-12-02
quota:user123:api_calls:2024-12-02
```

**TTL**: 86400 seconds (24 hours)

---

## Cache Key Patterns

### Session Keys

```
session:{session_id}:code_snippet
session:{session_id}:current_file
session:{session_id}:user_preferences
```

### Project Keys

```
project:{project_id}:analysis:{code_hash}
project:{project_id}:file_tree
project:{project_id}:recent_issues
project:{project_id}:baseline_metrics
```

### Model Keys

```
model:{model_name}:{version}:{prompt_hash}:response
```

### Provider Keys

```
provider:{provider_name}:health_status
provider:{provider_name}:rate_limit_status
```

### Rate Limit Keys

```
ratelimit:{user_id}:requests:{date}
ratelimit:{user_id}:daily_limit
ratelimit:{user_id}:monthly_limit
```

### Quota Keys

```
quota:{user_id}:api_calls:{date}
quota:{user_id}:analysis_runs:{date}
quota:{user_id}:storage_bytes:{date}
```

### Cost Tracking Keys

```
cost:{user_id}:{date}
cost:{user_id}:monthly:{year}-{month}
```

---

## Pub/Sub Channels

### Event Channels

```
events:version_promoted
events:baseline_violated
events:ai_analysis_completed
events:experiment_promoted
events:experiment_quarantined
```

### Notification Channels

```
notifications:user:{user_id}
notifications:admin
notifications:alerts
```

### System Channels

```
system:health_check
system:cache_invalidation
system:quota_warning
```

---

## Cache Operations

### Setting Cache Values

```python
from backend.shared.cache.redis_client import RedisClient

redis = RedisClient()

# L1: Session cache
redis.set_session_cache(
    session_id="abc123",
    key="code_snippet",
    value={"code": "...", "language": "python"},
    ttl=300
)

# L2: Project cache
redis.set_project_cache(
    project_id="proj456",
    key="analysis:a1b2c3d4",
    value={"findings": [...], "score": 0.85},
    ttl=3600
)

# L3: Global cache
redis.set_global_cache(
    key="model:gpt-4:v2.1.0:a1b2c3d4:response",
    value={"response": "...", "tokens": 150},
    ttl=86400
)
```

### Getting Cache Values

```python
# L1: Session cache
code_snippet = redis.get_session_cache(
    session_id="abc123",
    key="code_snippet"
)

# L2: Project cache
analysis = redis.get_project_cache(
    project_id="proj456",
    key="analysis:a1b2c3d4"
)

# L3: Global cache
response = redis.get_global_cache(
    key="model:gpt-4:v2.1.0:a1b2c3d4:response"
)
```

### Deleting Cache Values

```python
# Delete single value
redis.delete_session_cache(session_id="abc123", key="code_snippet")

# Clear entire session
redis.clear_session(session_id="abc123")

# Delete by pattern
redis.delete_by_pattern("project:proj456:*")
```

---

## Semantic Deduplication

Cache analysis results by code hash to avoid duplicate analysis.

```python
import hashlib

# Cache analysis result
code = "def hello(): pass"
result = {"findings": [], "score": 1.0}

code_hash = redis.cache_analysis_result(
    code=code,
    result=result,
    project_id="proj456",
    ttl=3600
)

# Later, retrieve cached result for same code
cached = redis.get_cached_analysis(
    code=code,
    project_id="proj456"
)
```

**Benefits**:

- Avoid redundant analysis
- Reduce API costs
- Improve response time
- Semantic caching by code content

---

## Rate Limiting

### Fixed Window Rate Limiting

```python
# Check rate limit
if redis.check_rate_limit(
    user_id="user123",
    limit=100,
    window=86400  # 24 hours
):
    # Process request
    pass
else:
    # Rate limit exceeded
    return 429

# Get rate limit status
status = redis.get_rate_limit_status(user_id="user123")
# Returns: {"requests_used": 45, "reset_in_seconds": 3600}
```

### Token Bucket Rate Limiting

```python
from backend.shared.cache.cache_strategies import RateLimitStrategy

# Check token bucket
if RateLimitStrategy.token_bucket(
    redis_client=redis,
    user_id="user123",
    capacity=100,
    refill_rate=10,
    window=60
):
    # Process request
    pass
else:
    # Rate limit exceeded
    return 429
```

### Sliding Window Rate Limiting

```python
# Check sliding window
if RateLimitStrategy.sliding_window(
    redis_client=redis,
    user_id="user123",
    limit=100,
    window=60
):
    # Process request
    pass
else:
    # Rate limit exceeded
    return 429
```

---

## Cache Patterns

### Cache-Aside (Lazy Loading)

```python
from backend.shared.cache.cache_strategies import CachePatterns

# Fetch data with cache-aside pattern
def fetch_file_tree():
    # Fetch from database
    return db.get_file_tree(project_id)

file_tree = CachePatterns.cache_aside(
    redis_client=redis,
    cache_key="project:proj456:file_tree",
    fetch_func=fetch_file_tree,
    ttl=3600
)
```

**Flow**:

1. Check cache
2. If miss, fetch from source
3. Store in cache
4. Return value

### Write-Through

```python
# Write with cache-through pattern
def persist_file_tree(data):
    # Write to database
    db.save_file_tree(project_id, data)

success = CachePatterns.write_through(
    redis_client=redis,
    cache_key="project:proj456:file_tree",
    value=file_tree,
    persist_func=persist_file_tree,
    ttl=3600
)
```

**Flow**:

1. Write to cache
2. Write to persistent storage
3. Return success

### Write-Behind (Write-Back)

```python
# Write with write-behind pattern
def queue_file_tree_update(key, data):
    # Queue write to message broker
    queue.enqueue("update_file_tree", key, data)

success = CachePatterns.write_behind(
    redis_client=redis,
    cache_key="project:proj456:file_tree",
    value=file_tree,
    queue_func=queue_file_tree_update,
    ttl=3600
)
```

**Flow**:

1. Write to cache immediately
2. Queue write to persistent storage
3. Return success

### Refresh-Ahead

```python
# Refresh cache before expiration
def fetch_fresh_data():
    return db.get_fresh_data(project_id)

data = CachePatterns.refresh_ahead(
    redis_client=redis,
    cache_key="project:proj456:data",
    fetch_func=fetch_fresh_data,
    ttl=3600,
    refresh_threshold=0.8  # Refresh when 80% expired
)
```

---

## Cache Invalidation

### Session Invalidation

```python
from backend.shared.cache.cache_strategies import CacheStrategy

# Invalidate all session caches on logout
CacheStrategy.invalidate_session_on_logout(
    redis_client=redis,
    session_id="abc123"
)
```

### Project Invalidation

```python
# Invalidate project caches on project change
CacheStrategy.invalidate_project_on_change(
    redis_client=redis,
    project_id="proj456"
)
```

### Analysis Invalidation

```python
# Invalidate analysis cache when new version is promoted
CacheStrategy.invalidate_analysis_on_new_version(
    redis_client=redis,
    project_id="proj456"
)
```

---

## Cache Warming

Pre-populate cache with frequently accessed data.

```python
# Warm project cache
CacheStrategy.warm_project_cache(
    redis_client=redis,
    project_id="proj456",
    file_tree={"files": [...], "dirs": [...]},
    recent_issues=[{"id": "1", "severity": "high"}, ...]
)
```

---

## Monitoring & Metrics

### Cache Statistics

```python
# Get cache statistics
stats = redis.get_stats()
# Returns: {
#     "used_memory": "2.5M",
#     "connected_clients": 5,
#     "total_commands": 150000,
#     "uptime_seconds": 86400
# }

# Get cache hit/miss rate
cache_stats = CacheStrategy.get_cache_stats(redis_client=redis)
# Returns: {
#     "total_hits": 1000,
#     "total_misses": 100,
#     "hit_rate": 0.909,
#     "total_requests": 1100
# }
```

### Cache Hit Tracking

```python
# Track cache hit
CacheStrategy.track_cache_hit(redis, "project:proj456:file_tree")

# Track cache miss
CacheStrategy.track_cache_miss(redis, "project:proj456:file_tree")
```

---

## Quota Management

### Increment Quota

```python
# Increment API call quota
calls_used = redis.increment_quota(
    user_id="user123",
    quota_type="api_calls",
    amount=1
)

# Get quota usage
usage = redis.get_quota_usage(
    user_id="user123",
    quota_type="api_calls"
)
```

---

## Distributed Locks

### Acquire Lock

```python
# Acquire lock (blocking)
if redis.acquire_lock(
    lock_name="analysis:proj456",
    timeout=30,
    blocking=True
):
    # Critical section
    perform_analysis()
    redis.release_lock("analysis:proj456")
```

### Release Lock

```python
# Release lock
redis.release_lock("analysis:proj456")
```

---

## Pub/Sub

### Publishing Events

```python
# Publish event
redis.publish_event(
    channel="events:version_promoted",
    message={
        "version_id": "v123",
        "promoted_at": "2024-12-02T10:00:00Z",
        "promoted_by": "admin"
    }
)
```

### Subscribing to Events

```python
# Subscribe to channels
pubsub = redis.subscribe([
    "events:version_promoted",
    "events:baseline_violated"
])

# Listen for messages
for message in pubsub.listen():
    if message['type'] == 'message':
        print(f"Received: {message['data']}")
```

---

## Best Practices

1. **Key Naming**: Use consistent, hierarchical key patterns
2. **TTL Management**: Set appropriate TTLs for each cache level
3. **Error Handling**: Gracefully handle cache misses
4. **Invalidation**: Invalidate cache on data changes
5. **Monitoring**: Track cache hit rates and memory usage
6. **Serialization**: Use JSON for complex objects
7. **Compression**: Consider compression for large values
8. **Expiration**: Use EXPIRE to prevent memory bloat
9. **Batch Operations**: Use pipelines for multiple operations
10. **Connection Pooling**: Reuse Redis connections

---

## Performance Tuning

### Memory Management

- Monitor memory usage with `redis.get_stats()`
- Set appropriate eviction policies
- Use compression for large values
- Implement TTL for all keys

### Query Optimization

- Use pipelines for batch operations
- Minimize key lookups
- Use appropriate data structures
- Cache frequently accessed data

### Scaling

- Use Redis Cluster for horizontal scaling
- Implement sharding by user_id or project_id
- Use Redis Sentinel for high availability
- Monitor performance metrics

---

## Troubleshooting

### High Memory Usage

- Check key expiration
- Review cache TTLs
- Implement eviction policies
- Monitor cache size

### Low Hit Rate

- Increase TTL for frequently accessed data
- Implement cache warming
- Review cache key patterns
- Monitor access patterns

### Connection Issues

- Check Redis server status
- Verify network connectivity
- Review connection pool settings
- Check firewall rules

---

## Future Enhancements

- [ ] Redis Cluster support
- [ ] Automatic cache warming
- [ ] Advanced compression
- [ ] Cache analytics dashboard
- [ ] Distributed tracing
- [ ] Multi-region replication
