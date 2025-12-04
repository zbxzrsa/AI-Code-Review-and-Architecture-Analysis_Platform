# Optimization Features Guide

This guide covers the optimization features implemented in the AI Code Review Platform frontend.

## Table of Contents

1. [Performance Monitoring](#performance-monitoring)
2. [Caching Service](#caching-service)
3. [Security Service](#security-service)
4. [Feedback System](#feedback-system)
5. [ML Auto-Promotion](#ml-auto-promotion)

---

## Performance Monitoring

The `performanceMonitor` service provides comprehensive performance tracking for the frontend application.

### Basic Usage

```typescript
import { performanceMonitor } from '@/services';

// Track API calls
performanceMonitor.trackAPICall('/api/users', 'GET', 150, 200, false);

// Track component renders
performanceMonitor.trackComponentRender('Dashboard', 25);

// Record custom metrics
performanceMonitor.recordMetric('custom.action', 42, 'count');
```

### React Hook

```typescript
import { usePerformanceTracking } from '@/services';

function MyComponent() {
  const { trackRender, trackAPICall } = usePerformanceTracking('MyComponent');
  
  useEffect(() => {
    trackRender();
  }, []);
  
  const fetchData = async () => {
    const start = Date.now();
    const response = await api.get('/data');
    trackAPICall('/data', 'GET', Date.now() - start, response.status);
  };
}
```

### Getting Statistics

```typescript
// API statistics
const apiStats = performanceMonitor.getAPIStats();
console.log(`Avg Response Time: ${apiStats.avgResponseTime}ms`);
console.log(`P95 Latency: ${apiStats.p95Latency}ms`);
console.log(`Error Rate: ${apiStats.errorRate * 100}%`);

// Component statistics
const componentStats = performanceMonitor.getComponentStats();
componentStats.forEach(comp => {
  console.log(`${comp.name}: ${comp.avgRenderTime}ms avg`);
});

// Full report
const report = performanceMonitor.getReport();
```

### Event Listeners

```typescript
import { eventBus } from '@/services';

// Listen for slow API calls
eventBus.on('performance:api:slow', ({ endpoint, duration }) => {
  console.warn(`Slow API: ${endpoint} took ${duration}ms`);
});

// Listen for critical performance issues
eventBus.on('performance:api:critical', ({ endpoint, duration }) => {
  alert(`Critical: ${endpoint} took ${duration}ms!`);
});
```

---

## Caching Service

The `cacheService` provides multi-layer caching with memory, session, and local storage.

### Basic Usage

```typescript
import { cacheService } from '@/services';

// Set cache with default options
cacheService.set('user:123', userData);

// Get cached value
const cached = cacheService.get('user:123');
if (cached) {
  console.log('From cache:', cached.data);
}

// Check if exists
if (cacheService.has('user:123')) {
  // Use cached value
}
```

### Cache Options

```typescript
// Custom TTL (time-to-live)
cacheService.set('short-lived', data, { ttl: 30000 }); // 30 seconds

// Specific cache layers
cacheService.set('session-only', data, { 
  layers: ['session'] // Only store in sessionStorage
});

// Multiple layers with tags
cacheService.set('user:123', userData, {
  ttl: 300000, // 5 minutes
  layers: ['memory', 'session', 'local'],
  tags: ['users', 'profile']
});
```

### Cache Invalidation

```typescript
// Invalidate single key
cacheService.invalidate('user:123');

// Invalidate by tag
cacheService.invalidateByTag('users'); // Removes all items tagged 'users'

// Invalidate by pattern
cacheService.invalidateByPattern('api:/users'); // Removes matching keys

// Clear all caches
cacheService.clearAll();
```

### Wrap Pattern

```typescript
// Automatically cache function results
const userData = await cacheService.wrap(
  'user:123',
  async () => {
    return await api.get('/users/123');
  },
  { ttl: 300000 }
);
```

### Decorator Usage

```typescript
import { cached } from '@/services';

class UserService {
  @cached((id: string) => `user:${id}`, { ttl: 300000 })
  async getUser(id: string) {
    return await api.get(`/users/${id}`);
  }
}
```

### Statistics

```typescript
const stats = cacheService.getStats();
console.log(`Hit Rate: ${(stats.hitRate * 100).toFixed(1)}%`);
console.log(`Memory Size: ${stats.memorySize} items`);
console.log(`Hits: ${stats.hits}, Misses: ${stats.misses}`);
```

---

## Security Service

The `securityService` provides frontend security utilities.

### JWT Validation

```typescript
import { securityService } from '@/services';

const result = securityService.validateJWT(token);
if (!result.valid) {
  console.error('Invalid token:', result.error);
  // Redirect to login
}
```

### CSRF Protection

```typescript
// Generate token (usually done once per session)
const csrfToken = securityService.generateCSRFToken();

// Include in requests
fetch('/api/data', {
  headers: {
    'X-CSRF-Token': csrfToken
  }
});

// Validate on form submission
if (!securityService.validateCSRFToken(submittedToken)) {
  throw new Error('CSRF validation failed');
}
```

### XSS Prevention

```typescript
// Sanitize user input before rendering
const safeHTML = securityService.sanitizeHTML(userInput);

// Validate URLs before using
if (securityService.validateURL(url)) {
  window.location.href = url;
}
```

### Input Validation

```typescript
// Create validators
const emailValidator = securityService.createValidator('email');
const passwordValidator = securityService.createValidator('minLength', { min: 8 });

// Validate inputs
const emailResult = emailValidator(email);
if (!emailResult.valid) {
  setError('email', emailResult.error);
}

// Validate multiple fields
const errors = securityService.validateForm({
  email: { value: email, rules: ['required', 'email'] },
  password: { value: password, rules: ['required', { type: 'minLength', min: 8 }] }
});
```

### Rate Limiting

```typescript
// Check rate limit before action
const result = securityService.checkRateLimit('api-calls', {
  maxRequests: 100,
  windowMs: 60000 // 1 minute
});

if (!result.allowed) {
  console.log(`Rate limited. Retry after ${result.retryAfter}ms`);
  return;
}
```

### Security Event Logging

```typescript
// Log security events
securityService.logSecurityEvent({
  type: 'authentication',
  severity: 'low',
  message: 'User logged in successfully',
  details: { userId, method: 'password' }
});

// Get security events
const events = securityService.getSecurityEvents({ 
  severity: 'high',
  since: Date.now() - 3600000 // Last hour
});

// Get summary
const summary = securityService.getSecuritySummary();
```

---

## Feedback System

The feedback system collects user satisfaction and feature requests.

### Using the Widget

The `FeedbackWidget` is automatically included in the main layout. Users can:

1. Click the feedback button (bottom-right)
2. Choose feedback type:
   - **Rating**: 5-star satisfaction rating
   - **NPS**: Net Promoter Score (0-10)
   - **Feature Request**: Suggest new features
   - **Bug Report**: Report issues
   - **General**: General feedback

### Quick Feedback Component

```tsx
import { QuickFeedback } from '@/components/feedback';

function ArticlePage() {
  const handleFeedback = (type: 'positive' | 'negative') => {
    // Track feedback
    analytics.track('article_feedback', { articleId, type });
  };

  return (
    <div>
      <h1>Article Title</h1>
      <p>Article content...</p>
      
      <QuickFeedback 
        question="Was this article helpful?"
        onFeedback={handleFeedback}
      />
    </div>
  );
}
```

### Programmatic Feedback

```typescript
// Submit feedback programmatically
const feedbackData = {
  type: 'rating',
  rating: 5,
  description: 'Great experience!',
  category: 'usability'
};

// Store locally or send to API
localStorage.setItem(
  `feedback:${Date.now()}`,
  JSON.stringify(feedbackData)
);
```

---

## ML Auto-Promotion

The ML Auto-Promotion dashboard manages AI model version promotion.

### Accessing the Dashboard

Navigate to: **Administration → ML Auto-Promotion** (Admin only)

### Features

1. **Model Versions Table**
   - View all model versions
   - See stage (development, staging, production, quarantine)
   - Monitor metrics (accuracy, error rate, latency)
   - Promote or rollback versions

2. **Promotion Rules**
   - Configure automatic promotion thresholds
   - Enable/disable individual rules
   - Adjust threshold values

3. **Auto-Promotion Toggle**
   - Enable/disable automatic promotion
   - System monitors metrics continuously
   - Promotes versions meeting all criteria

4. **Promotion History**
   - View past promotions
   - See rollback events
   - Track promotion reasons

### API Integration

```typescript
// Get model versions
const versions = await api.get('/admin/ml/versions');

// Promote a version
await api.post('/admin/ml/promote', {
  versionId: 'v3.3.0-beta',
  targetStage: 'production'
});

// Configure promotion rules
await api.put('/admin/ml/rules', {
  rules: [
    { metric: 'accuracy', operator: 'gte', threshold: 0.90 },
    { metric: 'errorRate', operator: 'lte', threshold: 0.02 }
  ]
});
```

---

## Best Practices

### Performance

1. **Use caching wisely**
   - Cache expensive API calls
   - Use appropriate TTL values
   - Invalidate on data changes

2. **Monitor performance**
   - Track API response times
   - Monitor component render times
   - Set up alerts for slow operations

3. **Optimize renders**
   - Use React.memo for expensive components
   - Track render counts
   - Identify re-render causes

### Security

1. **Validate all inputs**
   - Use security service validators
   - Sanitize HTML before rendering
   - Validate URLs before navigation

2. **Protect against CSRF**
   - Include CSRF tokens in state-changing requests
   - Validate tokens server-side

3. **Monitor security events**
   - Log authentication events
   - Track authorization failures
   - Review security summary regularly

### User Feedback

1. **Make it easy**
   - Keep feedback widget visible
   - Use quick feedback for simple questions
   - Minimize required fields

2. **Act on feedback**
   - Review feedback regularly
   - Prioritize based on frequency
   - Close the loop with users

---

## Troubleshooting

### Common Issues

**Cache not invalidating:**
```typescript
// Make sure to use the correct key
cacheService.invalidate('exact-key'); // Not pattern matching

// Use pattern for partial matches
cacheService.invalidateByPattern('api:/users');
```

**Performance events not firing:**
```typescript
// Ensure eventBus subscription is set up before tracking
eventBus.on('performance:api:slow', handler);
performanceMonitor.trackAPICall(...); // Now event will fire
```

**Security validation failing:**
```typescript
// Check validator configuration
const validator = securityService.createValidator('minLength', { min: 8 });
// Not: { minLength: 8 }
```

---

## Support

For issues or questions:
1. Check the documentation
2. Review source code in `src/services/`
3. Submit a feedback request through the widget
4. Contact the development team
