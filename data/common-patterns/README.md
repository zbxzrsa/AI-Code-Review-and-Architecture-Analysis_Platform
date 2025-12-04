# Common Code Patterns for Cache Warming

This directory contains common code patterns used to warm the semantic cache in offline deployments.

## Purpose

When deploying in air-gapped or offline environments, these patterns help:

1. **Pre-compute embeddings** for common vulnerability patterns
2. **Reduce cold-start latency** by having responses ready
3. **Improve cache hit rate** for similar code patterns

## Pattern Categories

### Security Vulnerabilities

| File                   | Pattern              | Language   |
| ---------------------- | -------------------- | ---------- |
| `sql-injection.py`     | SQL Injection        | Python     |
| `xss-vulnerability.js` | Cross-Site Scripting | JavaScript |
| `hardcoded-secrets.ts` | Hardcoded Secrets    | TypeScript |

### More patterns to add:

- `command-injection.py` - OS command injection
- `path-traversal.js` - Path traversal attacks
- `insecure-deserialization.java` - Insecure deserialization
- `weak-crypto.py` - Weak cryptographic algorithms
- `race-condition.go` - Race condition vulnerabilities

## Usage

### Warm the cache on startup:

```python
from services.semantic_cache import SemanticCacheService

cache = SemanticCacheService()
await cache.warm_cache('/data/common-patterns')
```

### Docker Compose:

```yaml
semantic-cache:
  environment:
    - WARM_CACHE_ON_STARTUP=true
    - PATTERNS_DIR=/data/common-patterns
  volumes:
    - ./data/common-patterns:/data/common-patterns:ro
```

### Kubernetes:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cache-patterns
data:
  sql-injection.py: |
    # Pattern content...
```

## Adding New Patterns

1. Create a file with the vulnerability pattern
2. Include both **vulnerable** and **safe** versions
3. Add comments explaining the issue
4. Update this README

## Cache Statistics

After warming, check cache stats:

```bash
curl http://localhost:8103/stats
```

Expected output:

```json
{
  "total_entries": 50,
  "index_size_mb": 2.5,
  "similarity_threshold": 0.92,
  "max_cache_size": 100000
}
```
