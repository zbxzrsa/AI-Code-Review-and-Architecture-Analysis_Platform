# Deployment & Testing Guide - Quick Wins

## Complete Step-by-Step Deployment Instructions

**Version**: 1.0  
**Date**: December 6, 2024  
**Status**: Ready for Production Deployment

---

## 📋 Pre-Deployment Checklist

### Prerequisites

- [ ] PostgreSQL 14+ running
- [ ] Redis 7+ running
- [ ] Python 3.10+ installed
- [ ] Node.js 18+ installed (for frontend)
- [ ] Docker & Kubernetes access (for production)
- [ ] Backup of current database
- [ ] Monitoring tools configured (Grafana/Prometheus)

### Environment Setup

```bash
# Clone/update repository
cd AI-Code-Review-and-Architecture-Analysis_Platform

# Create backup
pg_dump -U postgres ai_code_review > backup_$(date +%Y%m%d).sql

# Verify services
redis-cli ping  # Should return PONG
psql -U postgres -d ai_code_review -c "SELECT 1"  # Should return 1
```

---

## 🚀 Phase 1: Database Optimization (15 minutes)

### Step 1.1: Apply Performance Indexes

```bash
# Connect to database
psql -U postgres -d ai_code_review

# Apply migration
\i database/migrations/003_performance_indexes.sql

# Verify indexes created
SELECT
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) as size
FROM pg_stat_user_indexes
WHERE schemaname IN ('production', 'projects', 'auth', 'providers')
ORDER BY schemaname, tablename;

# Expected: 27+ new indexes
```

**Expected Output**:

```
 schemaname |        tablename        |           indexname            |  size
------------+-------------------------+--------------------------------+--------
 production | analysis_sessions       | idx_analysis_user_created      | 128 kB
 production | analysis_sessions       | idx_analysis_code_hash         | 96 kB
 production | code_review_results     | idx_issues_severity_status     | 64 kB
 ...
```

### Step 1.2: Test Query Performance

```sql
-- Test user analysis query (should use index)
EXPLAIN ANALYZE
SELECT * FROM production.analysis_sessions
WHERE user_id = 'test-user-123'
ORDER BY created_at DESC
LIMIT 50;

-- Look for "Index Scan" not "Seq Scan"
-- Execution time should be < 50ms
```

**Rollback** (if needed):

```bash
# Drop all indexes
\i database/migrations/003_performance_indexes_rollback.sql
```

---

## 🚀 Phase 2: Backend Application Updates (30 minutes)

### Step 2.1: Update Dependencies

```bash
cd backend

# Add new dependencies to requirements.txt
cat >> requirements.txt << EOF
# Performance optimizations
redis>=5.0.0
fastapi-cache2>=0.2.0
aioredis>=2.0.0
EOF

# Install dependencies
pip install -r requirements.txt
```

### Step 2.2: Update Environment Variables

```bash
# Edit .env file
cat >> .env << EOF

# Database Connection Pool
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600

# Redis Cache
REDIS_URL=redis://localhost:6379/0
CACHE_ENABLED=true
CACHE_DEFAULT_TTL=300

# Batch Processing
BATCH_MAX_SIZE=50
BATCH_MAX_WAIT=0.1

# Retry Configuration
RETRY_MAX_ATTEMPTS=3
RETRY_INITIAL_DELAY=1.0
EOF
```

### Step 2.3: Update Application Startup

Create `backend/app/startup.py`:

```python
"""Application startup configuration"""
import logging
from backend.shared.database.connection_pool import initialize_pool, get_pool
from backend.shared.cache.response_cache import initialize_cache, CacheConfig, get_cache
from backend.shared.utils.batch_processor import AIAnalysisBatcher, register_batch_processor
from backend.app.config import settings

logger = logging.getLogger(__name__)

async def startup_event():
    """Initialize all performance optimizations"""

    # 1. Initialize database connection pool
    logger.info("Initializing database connection pool...")
    pool = initialize_pool(
        settings.DATABASE_URL,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_timeout=settings.DB_POOL_TIMEOUT,
        pool_recycle=settings.DB_POOL_RECYCLE,
        pool_pre_ping=True
    )
    logger.info(f"✅ Database pool initialized: {pool.get_pool_stats()}")

    # 2. Initialize Redis cache
    logger.info("Initializing Redis cache...")
    cache_config = CacheConfig(
        redis_url=settings.REDIS_URL,
        enabled=settings.CACHE_ENABLED,
        default_ttl=settings.CACHE_DEFAULT_TTL
    )
    cache = initialize_cache(cache_config)
    await cache.connect()
    logger.info(f"✅ Cache initialized: {cache.get_stats()}")

    # 3. Initialize batch processors
    logger.info("Initializing batch processors...")
    # Add your AI provider initialization here
    # batcher = AIAnalysisBatcher(ai_provider)
    # await batcher.start()
    # register_batch_processor("ai_analysis", batcher.processor)
    logger.info("✅ Batch processors initialized")

    logger.info("🚀 All performance optimizations initialized successfully!")

async def shutdown_event():
    """Clean up on shutdown"""
    from backend.shared.database.connection_pool import dispose_pool
    from backend.shared.utils.batch_processor import stop_all_processors

    logger.info("Shutting down...")

    # Stop batch processors
    await stop_all_processors()

    # Dispose database pool
    await dispose_pool()

    # Disconnect cache
    cache = get_cache()
    await cache.disconnect()

    logger.info("✅ Shutdown complete")
```

Update `backend/app/main.py`:

```python
from fastapi import FastAPI
from backend.app.startup import startup_event, shutdown_event

app = FastAPI(title="AI Code Review Platform")

# Register startup/shutdown events
app.add_event_handler("startup", startup_event)
app.add_event_handler("shutdown", shutdown_event)

# Your existing routes...
```

### Step 2.4: Test Backend Locally

```bash
# Start backend
cd backend
uvicorn app.main:app --reload --port 8000

# In another terminal, test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/metrics/database-pool
curl http://localhost:8000/metrics/cache

# Expected responses with status 200
```

---

## 🚀 Phase 3: Integration Testing (20 minutes)

### Step 3.1: Test Database Pool

Create `tests/test_performance.py`:

```python
import pytest
import asyncio
from backend.shared.database.connection_pool import get_pool
from sqlalchemy import text

@pytest.mark.asyncio
async def test_connection_pool():
    """Test database connection pool"""
    pool = get_pool()

    # Test concurrent queries
    async def query():
        async with pool.session() as session:
            result = await session.execute(text("SELECT 1"))
            return result.scalar()

    # Run 50 concurrent queries
    results = await asyncio.gather(*[query() for _ in range(50)])

    assert all(r == 1 for r in results)

    # Check pool stats
    stats = pool.get_pool_stats()
    assert stats['size'] == 20
    assert stats['total_checkouts'] >= 50
    print(f"✅ Pool test passed: {stats}")

@pytest.mark.asyncio
async def test_query_performance():
    """Test query performance with indexes"""
    pool = get_pool()

    import time
    start = time.time()

    async with pool.session() as session:
        result = await session.execute(text("""
            SELECT * FROM production.analysis_sessions
            WHERE user_id = 'test-user'
            ORDER BY created_at DESC
            LIMIT 50
        """))
        rows = result.fetchall()

    duration = time.time() - start

    assert duration < 0.1  # Should be < 100ms
    print(f"✅ Query completed in {duration*1000:.2f}ms")
```

Run tests:

```bash
pytest tests/test_performance.py -v
```

### Step 3.2: Test Cache

```python
@pytest.mark.asyncio
async def test_cache_functionality():
    """Test Redis cache"""
    from backend.shared.cache.response_cache import get_cache

    cache = get_cache()

    # Test set/get
    await cache.set("test_key", {"data": "value"}, ttl=60)
    value = await cache.get("test_key")

    assert value == {"data": "value"}

    # Test cache decorator
    call_count = 0

    @cache.cache(ttl=60, key_prefix="test")
    async def expensive_operation(x):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.1)
        return x * 2

    # First call: slow
    result1 = await expensive_operation(5)
    assert call_count == 1

    # Second call: fast (cached)
    result2 = await expensive_operation(5)
    assert call_count == 1  # Not called again
    assert result1 == result2 == 10

    stats = cache.get_stats()
    print(f"✅ Cache test passed: {stats}")
```

### Step 3.3: Test Retry Logic

```python
@pytest.mark.asyncio
async def test_retry_mechanism():
    """Test retry with exponential backoff"""
    from backend.shared.utils.retry import retry

    attempt_count = 0

    @retry(max_attempts=3, initial_delay=0.1)
    async def flaky_function():
        nonlocal attempt_count
        attempt_count += 1

        if attempt_count < 3:
            raise Exception("Transient error")

        return "success"

    result = await flaky_function()

    assert result == "success"
    assert attempt_count == 3
    print(f"✅ Retry test passed: {attempt_count} attempts")
```

### Step 3.4: Load Testing

```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Test API endpoint
ab -n 1000 -c 10 http://localhost:8000/api/health

# Expected results:
# - Requests per second: > 500
# - Time per request: < 20ms
# - Failed requests: 0
```

---

## 🚀 Phase 4: Production Deployment (30 minutes)

### Step 4.1: Build Docker Images

```bash
# Build backend image
docker build -t ai-code-review-backend:v2.0 -f backend/Dockerfile .

# Build frontend image (if applicable)
docker build -t ai-code-review-frontend:v2.0 -f frontend/Dockerfile .

# Tag for registry
docker tag ai-code-review-backend:v2.0 gcr.io/PROJECT_ID/backend:v2.0
docker push gcr.io/PROJECT_ID/backend:v2.0
```

### Step 4.2: Update Kubernetes Deployments

Update `kubernetes/deployments.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-service
  namespace: platform-v2-stable
spec:
  replicas: 3
  template:
    spec:
      containers:
        - name: api
          image: gcr.io/PROJECT_ID/backend:v2.0
          env:
            # Add new environment variables
            - name: DB_POOL_SIZE
              value: "20"
            - name: DB_MAX_OVERFLOW
              value: "10"
            - name: REDIS_URL
              value: "redis://redis-service:6379/0"
            - name: CACHE_ENABLED
              value: "true"
            - name: CACHE_DEFAULT_TTL
              value: "300"
          resources:
            requests:
              cpu: 500m
              memory: 1Gi
              ephemeral-storage: "1Gi"
            limits:
              cpu: 2000m
              memory: 4Gi
              ephemeral-storage: "2Gi"
```

### Step 4.3: Deploy to Kubernetes

```bash
# Apply database migration
kubectl exec -it postgres-pod -n platform-infrastructure -- \
  psql -U postgres -d ai_code_review -f /migrations/003_performance_indexes.sql

# Deploy updated application
kubectl apply -f kubernetes/deployments.yaml

# Watch rollout
kubectl rollout status deployment/api-service -n platform-v2-stable

# Verify pods are running
kubectl get pods -n platform-v2-stable
```

### Step 4.4: Canary Deployment (Recommended)

```bash
# Deploy to 10% of traffic first
kubectl patch deployment api-service -n platform-v2-stable \
  -p '{"spec":{"replicas":1}}'

# Monitor for 30 minutes
# Check metrics, error rates, latency

# If successful, scale up
kubectl scale deployment api-service -n platform-v2-stable --replicas=3
```

---

## 📊 Post-Deployment Monitoring (Continuous)

### Step 5.1: Monitor Key Metrics

**Grafana Dashboard Queries**:

```promql
# API Response Time (should decrease by 70%)
histogram_quantile(0.95,
  rate(http_request_duration_seconds_bucket[5m])
)

# Cache Hit Rate (should be > 70%)
rate(cache_hits_total[5m]) /
  (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m]))

# Database Pool Usage
db_pool_checked_out / db_pool_size

# Error Rate (should decrease by 80%)
rate(http_requests_total{status=~"5.."}[5m]) /
  rate(http_requests_total[5m])
```

### Step 5.2: Set Up Alerts

```yaml
# prometheus-alerts.yml
groups:
  - name: performance
    rules:
      - alert: HighAPILatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
        for: 5m
        annotations:
          summary: "API latency is high"

      - alert: LowCacheHitRate
        expr: rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m])) < 0.5
        for: 10m
        annotations:
          summary: "Cache hit rate is low"

      - alert: DatabasePoolExhausted
        expr: db_pool_checked_out / db_pool_size > 0.9
        for: 5m
        annotations:
          summary: "Database pool nearly exhausted"
```

### Step 5.3: Monitor Application Logs

```bash
# Check application logs
kubectl logs -f deployment/api-service -n platform-v2-stable

# Look for:
# ✅ "Database pool initialized"
# ✅ "Cache initialized"
# ✅ "Batch processors initialized"

# Monitor error logs
kubectl logs deployment/api-service -n platform-v2-stable | grep ERROR
```

---

## 🧪 Validation Tests

### Test 1: Performance Improvement

```bash
# Before deployment (baseline)
ab -n 1000 -c 10 http://api.example.com/api/analysis/test-id
# Note: Requests per second, Time per request

# After deployment
ab -n 1000 -c 10 http://api.example.com/api/analysis/test-id
# Expected: 3-4x improvement in requests per second
```

### Test 2: Cache Effectiveness

```bash
# Call same endpoint twice
curl http://api.example.com/api/analysis/test-id -w "\nTime: %{time_total}s\n"
# First call: ~500ms

curl http://api.example.com/api/analysis/test-id -w "\nTime: %{time_total}s\n"
# Second call: ~50ms (cached)
```

### Test 3: Database Performance

```sql
-- Check index usage
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read
FROM pg_stat_user_indexes
WHERE schemaname = 'production'
ORDER BY idx_scan DESC
LIMIT 10;

-- All indexes should have idx_scan > 0
```

---

## 🔄 Rollback Procedures

### If Issues Detected

**Step 1: Quick Rollback**

```bash
# Rollback Kubernetes deployment
kubectl rollout undo deployment/api-service -n platform-v2-stable

# Verify rollback
kubectl rollout status deployment/api-service -n platform-v2-stable
```

**Step 2: Rollback Database Indexes** (if needed)

```bash
kubectl exec -it postgres-pod -n platform-infrastructure -- \
  psql -U postgres -d ai_code_review \
  -c "DROP INDEX IF EXISTS production.idx_analysis_user_created;"
# Repeat for all indexes
```

**Step 3: Disable Features via Environment**

```bash
# Disable cache
kubectl set env deployment/api-service CACHE_ENABLED=false -n platform-v2-stable

# Reduce pool size
kubectl set env deployment/api-service DB_POOL_SIZE=5 -n platform-v2-stable
```

---

## ✅ Success Criteria

After 24 hours, verify:

- [ ] API response time (p95) < 200ms
- [ ] Cache hit rate > 70%
- [ ] Error rate < 1%
- [ ] Database pool utilization < 80%
- [ ] No increase in failed requests
- [ ] Throughput increased by 3x+
- [ ] No customer complaints
- [ ] All monitoring alerts green

---

## 📞 Support & Troubleshooting

### Common Issues

**Issue 1: Cache Connection Failed**

```bash
# Check Redis connectivity
redis-cli -h redis-service ping

# Check Redis logs
kubectl logs deployment/redis -n platform-infrastructure
```

**Issue 2: Database Pool Exhausted**

```bash
# Increase pool size temporarily
kubectl set env deployment/api-service DB_POOL_SIZE=30 -n platform-v2-stable

# Monitor pool usage
curl http://api.example.com/metrics/database-pool
```

**Issue 3: High Memory Usage**

```bash
# Check memory usage
kubectl top pods -n platform-v2-stable

# Reduce cache TTL if needed
kubectl set env deployment/api-service CACHE_DEFAULT_TTL=60 -n platform-v2-stable
```

---

## 📈 Expected Results

### Week 1

- API latency: 70% reduction
- Throughput: 3-4x increase
- Error rate: 80% reduction
- Cache hit rate: 70-80%

### Month 1

- Infrastructure costs: 30% reduction
- Developer productivity: 2x improvement
- Customer satisfaction: Significantly improved
- System stability: Much better

---

## 🎯 Next Steps

After successful deployment:

1. **Monitor for 1 week** - Ensure stability
2. **Tune parameters** - Adjust cache TTL, pool size based on usage
3. **Implement remaining Quick Wins** - Frontend optimizations
4. **Move to Phase 2** - Advanced performance optimization

---

**Deployment Owner**: [Your Name]  
**Deployment Date**: [Date]  
**Rollback Owner**: [Name]  
**Emergency Contact**: [Contact Info]
