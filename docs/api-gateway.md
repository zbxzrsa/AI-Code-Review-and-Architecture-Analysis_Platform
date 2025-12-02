# API Gateway Configuration

## Overview

Production-grade API gateway with rate limiting, circuit breaking, TLS termination, and comprehensive security headers.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    API Gateway                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Entry Points                                              │
│  ├─→ HTTP (80) → HTTPS redirect                            │
│  └─→ HTTPS (443) → TLS 1.3                                 │
│                                                             │
│  Middlewares                                               │
│  ├─→ Rate Limiting (per user tier)                         │
│  ├─→ Circuit Breaker (50% failure threshold)               │
│  ├─→ Security Headers (HSTS, CSP, etc.)                    │
│  ├─→ CORS (origin-based)                                   │
│  ├─→ Compression (gzip)                                    │
│  └─→ Authentication (basic auth for admin)                 │
│                                                             │
│  Routers & Services                                        │
│  ├─→ V2 Production API                                     │
│  ├─→ V1 Experimentation API                                │
│  ├─→ V3 Quarantine API (admin only)                        │
│  ├─→ Code Review AI Service                                │
│  ├─→ Version Control AI Service (admin only)               │
│  └─→ WebSocket (collaborative editing)                     │
│                                                             │
│  Health Checks                                             │
│  ├─→ Liveness probes                                       │
│  ├─→ Readiness probes                                      │
│  └─→ Service availability monitoring                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Deployment Options

### Production: Traefik v2

**Advantages**:

- Cloud-native (Kubernetes-ready)
- Automatic certificate management
- Dynamic configuration
- Built-in metrics
- Excellent performance

**Configuration**:

- `gateway/traefik.yml` - Static configuration
- `gateway/dynamic.yml` - Dynamic configuration

### Local Development: Nginx

**Advantages**:

- Lightweight
- Widely available
- Simple configuration
- Good performance

**Configuration**:

- `gateway/nginx.conf` - Complete configuration

---

## Rate Limiting

### Three-Tier System

#### Anonymous Users

- **Limit**: 10 requests/minute
- **Burst**: 20 requests
- **Applied to**: Unauthenticated requests

#### Authenticated Users

- **Limit**: 100 requests/minute
- **Burst**: 200 requests
- **Applied to**: Requests with valid JWT

#### Admin Users

- **Limit**: 1000 requests/minute
- **Burst**: 2000 requests
- **Applied to**: Admin endpoints

### Implementation

**Traefik**:

```yaml
middlewares:
  rate-limit-anonymous:
    rateLimit:
      average: 10
      period: 60s
      burst: 20
```

**Nginx**:

```nginx
limit_req_zone $binary_remote_addr zone=anonymous:10m rate=10r/m;
limit_req zone=anonymous burst=20 nodelay;
```

### Handling Rate Limit Exceeded

**Response**:

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 60
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1701505260
```

---

## Circuit Breaker

### Configuration

```yaml
circuitBreaker:
  expression: "NetworkErrorRatio() > 0.5"
  checkInterval: 100ms
  fallbackDuration: 60s
  responseCode: 503
```

### States

1. **Closed** - Normal operation

   - Requests pass through
   - Errors are counted
   - If error ratio > 50%, transition to Open

2. **Open** - Circuit is broken

   - All requests return 503
   - After 60s, transition to Half-Open

3. **Half-Open** - Testing recovery
   - Limited requests allowed
   - If successful, transition to Closed
   - If failed, transition to Open

### Monitoring

```promql
# Circuit breaker state
traefik_service_circuit_breaker_open{service="v2_api_service"}

# Error ratio
rate(traefik_service_request_errors_total[5m]) / rate(traefik_service_requests_total[5m])
```

---

## TLS/HTTPS

### Certificate Management

**Traefik**:

- Automatic certificate provisioning via Let's Encrypt
- ACME HTTP challenge
- Certificate renewal 30 days before expiration
- Automatic redirect from HTTP to HTTPS

**Configuration**:

```yaml
certificatesResolvers:
  letsencrypt:
    acme:
      email: admin@example.com
      storage: /etc/traefik/acme.json
      httpChallenge:
        entryPoint: web
      certificatesDuration: 2160h # 90 days
```

### TLS Configuration

- **Minimum Version**: TLS 1.3
- **Ciphers**: Modern ciphers only
- **HSTS**: max-age=31536000 (1 year)
- **HSTS Preload**: Enabled

### Certificate Pinning

For production, implement certificate pinning:

```typescript
// Frontend
const certificatePins = [
  "sha256/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
  "sha256/BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB=",
];
```

---

## Security Headers

### Implemented Headers

#### Strict-Transport-Security (HSTS)

```
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
```

- Forces HTTPS for 1 year
- Includes subdomains
- Preload list eligible

#### Content-Security-Policy (CSP)

```
Content-Security-Policy: default-src 'self';
  script-src 'self' 'unsafe-inline' 'unsafe-eval';
  style-src 'self' 'unsafe-inline';
  img-src 'self' data: https:;
  font-src 'self' data:;
  connect-src 'self' https:;
  frame-ancestors 'none'
```

#### X-Frame-Options

```
X-Frame-Options: DENY
```

- Prevents clickjacking
- Disallows framing

#### X-Content-Type-Options

```
X-Content-Type-Options: nosniff
```

- Prevents MIME type sniffing

#### X-XSS-Protection

```
X-XSS-Protection: 1; mode=block
```

- Enables XSS filter
- Blocks page if XSS detected

#### Referrer-Policy

```
Referrer-Policy: strict-origin-when-cross-origin
```

- Sends referrer only for same-origin requests

#### Permissions-Policy

```
Permissions-Policy: geolocation=(), microphone=(), camera=()
```

- Disables sensitive APIs

---

## CORS Configuration

### Allowed Origins

- `https://app.example.com`
- `https://admin.example.com`

### Allowed Methods

- GET, POST, PUT, DELETE, PATCH, OPTIONS

### Allowed Headers

- Content-Type
- Authorization
- X-Requested-With

### Configuration

**Traefik**:

```yaml
middlewares:
  cors:
    headers:
      accessControlAllowOriginList:
        - "https://app.example.com"
      accessControlAllowMethods:
        - GET
        - POST
        - PUT
        - DELETE
        - PATCH
        - OPTIONS
      accessControlMaxAge: 3600
      accessControlAllowCredentials: true
```

---

## Routing Rules

### V2 Production API

- **Path**: `/api/v2`
- **Rate Limit**: Authenticated (100 req/min)
- **Authentication**: JWT required
- **Health Check**: `/health/ready`

### V1 Experimentation API

- **Path**: `/api/v1`
- **Rate Limit**: Authenticated (100 req/min)
- **Authentication**: JWT required
- **Health Check**: `/health/ready`

### V3 Quarantine API

- **Path**: `/api/v3`
- **Rate Limit**: Admin (1000 req/min)
- **Authentication**: Basic auth required
- **Health Check**: `/health/ready`

### Code Review AI Service

- **Path**: `/code-review-ai`
- **Rate Limit**: Authenticated (100 req/min)
- **Authentication**: JWT required
- **Timeout**: 60s (for streaming responses)

### Version Control AI Service

- **Path**: `/version-control-ai`
- **Rate Limit**: Admin (1000 req/min)
- **Authentication**: Basic auth required
- **Timeout**: 60s (for intensive evaluation)

### WebSocket (Collaborative Editing)

- **Path**: `/ws`
- **Protocol**: WebSocket
- **Upgrade**: Connection: Upgrade
- **Timeout**: 86400s (24 hours)

---

## Health Checks

### Endpoint Configuration

```yaml
services:
  v2-api-service:
    loadBalancer:
      servers:
        - url: "http://platform-v2-api:8000"
      healthCheck:
        path: /health/ready
        interval: 10s
        timeout: 5s
        scheme: http
```

### Health Check Responses

**Healthy (200)**:

```json
{
  "status": "ready",
  "checks": {
    "database": "ok",
    "ai_models": "ok"
  }
}
```

**Unhealthy (503)**:

```json
{
  "status": "not_ready",
  "error": "database connection failed"
}
```

---

## Compression

### Gzip Configuration

**Traefik**:

```yaml
middlewares:
  compression:
    compress:
      minResponseBodyBytes: 1000
      excludedContentTypes:
        - text/event-stream
```

**Nginx**:

```nginx
gzip on;
gzip_min_length 1000;
gzip_types text/plain text/css application/json application/javascript;
```

### Excluded Content Types

- `text/event-stream` (SSE)
- `application/octet-stream` (binary)

---

## Monitoring & Metrics

### Prometheus Integration

**Traefik Metrics**:

- `traefik_requests_total` - Total requests
- `traefik_request_duration_seconds` - Request duration
- `traefik_service_requests_total` - Per-service requests
- `traefik_service_request_errors_total` - Per-service errors
- `traefik_service_circuit_breaker_open` - Circuit breaker state

### Grafana Dashboards

Create dashboards for:

- Request rate per service
- Error rate and types
- Response time (p50, p95, p99)
- Circuit breaker state
- Rate limit violations
- Certificate expiration

### Alerts

```yaml
alerts:
  - name: HighErrorRate
    condition: error_rate > 0.05
    severity: critical

  - name: CircuitBreakerOpen
    condition: circuit_breaker_open == 1
    severity: warning

  - name: CertificateExpiringSoon
    condition: cert_expiration_days < 30
    severity: warning
```

---

## Troubleshooting

### Circuit Breaker Stuck Open

```bash
# Check service health
curl http://service:8000/health/ready

# Check error logs
docker logs traefik

# Manually reset (if needed)
# Restart the service
docker restart service-container
```

### Rate Limit Issues

```bash
# Check current rate limit
curl -i http://api.example.com/api/v2/health

# Headers show:
# X-RateLimit-Limit: 100
# X-RateLimit-Remaining: 95
# X-RateLimit-Reset: 1701505260
```

### Certificate Issues

```bash
# Check certificate status
curl -I https://api.example.com

# View certificate details
openssl s_client -connect api.example.com:443

# Check ACME logs
docker logs traefik | grep acme
```

### CORS Issues

```bash
# Check CORS headers
curl -H "Origin: https://app.example.com" \
     -H "Access-Control-Request-Method: POST" \
     -i https://api.example.com/api/v2/health
```

---

## Performance Tuning

### Connection Pooling

**Nginx**:

```nginx
upstream v2_api {
  server platform-v2-api:8000;
  keepalive 32;
}
```

### Buffer Sizes

**Nginx**:

```nginx
proxy_buffer_size 4k;
proxy_buffers 8 4k;
proxy_busy_buffers_size 8k;
```

### Timeouts

**Traefik**:

```yaml
forwardauth:
  address: "http://auth-service:8080/auth"
  trustForwardHeader: true
```

---

## Security Best Practices

1. **Keep TLS Updated**: Regularly update TLS version and ciphers
2. **Monitor Certificates**: Set up alerts for expiration
3. **Rotate Secrets**: Regularly rotate API keys and credentials
4. **Audit Logs**: Enable and monitor access logs
5. **DDoS Protection**: Consider WAF (Web Application Firewall)
6. **Rate Limiting**: Adjust limits based on usage patterns
7. **Circuit Breaking**: Monitor and tune thresholds

---

## Migration Guide

### From Nginx to Traefik

1. Export current configuration
2. Create Traefik configuration
3. Test in staging environment
4. Gradually shift traffic
5. Monitor metrics
6. Rollback if needed

### From HTTP to HTTPS

1. Obtain SSL certificate
2. Configure TLS in gateway
3. Set up HTTP → HTTPS redirect
4. Update frontend URLs
5. Monitor for mixed content warnings
6. Enable HSTS after verification

---

## Future Enhancements

- [ ] WAF (Web Application Firewall)
- [ ] DDoS protection
- [ ] Advanced rate limiting (per endpoint)
- [ ] Request signing
- [ ] API versioning management
- [ ] GraphQL support
- [ ] gRPC support
- [ ] Service mesh integration (Istio)
