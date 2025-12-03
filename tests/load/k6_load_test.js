/**
 * K6 Load Testing Configuration
 * 
 * Phase 5: Testing & Validation
 * - 1000 concurrent users
 * - 5000 requests/minute target
 * - Multiple scenarios (smoke, load, stress, spike)
 */

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const apiLatency = new Trend('api_latency');
const codeReviewLatency = new Trend('code_review_latency');
const requestCount = new Counter('request_count');

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const API_KEY = __ENV.API_KEY || 'test-api-key';

// Test scenarios
export const options = {
  scenarios: {
    // Smoke test: Verify system works with minimal load
    smoke: {
      executor: 'constant-vus',
      vus: 5,
      duration: '1m',
      tags: { scenario: 'smoke' },
      exec: 'smokeTest',
    },
    
    // Load test: Normal expected load
    load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 100 },   // Ramp up
        { duration: '5m', target: 100 },   // Stay at 100 users
        { duration: '2m', target: 200 },   // Ramp up more
        { duration: '5m', target: 200 },   // Stay at 200 users
        { duration: '2m', target: 0 },     // Ramp down
      ],
      tags: { scenario: 'load' },
      exec: 'loadTest',
      startTime: '2m', // Start after smoke test
    },
    
    // Stress test: Beyond normal capacity
    stress: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 200 },
        { duration: '5m', target: 500 },
        { duration: '5m', target: 1000 },  // 1000 concurrent users
        { duration: '5m', target: 1000 },
        { duration: '5m', target: 0 },
      ],
      tags: { scenario: 'stress' },
      exec: 'stressTest',
      startTime: '20m', // Start after load test
    },
    
    // Spike test: Sudden traffic spike
    spike: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '30s', target: 100 },
        { duration: '30s', target: 1000 }, // Sudden spike
        { duration: '2m', target: 1000 },
        { duration: '30s', target: 100 },  // Quick drop
        { duration: '1m', target: 0 },
      ],
      tags: { scenario: 'spike' },
      exec: 'spikeTest',
      startTime: '45m', // Start after stress test
    },
  },
  
  // Thresholds (SLO validation)
  thresholds: {
    'http_req_duration': ['p(95)<3000', 'p(99)<5000'], // p95 < 3s, p99 < 5s
    'http_req_failed': ['rate<0.02'],                   // Error rate < 2%
    'errors': ['rate<0.02'],
    'api_latency': ['p(95)<2000'],
    'code_review_latency': ['p(95)<5000'],
  },
};

// Helper functions
function getHeaders() {
  return {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${API_KEY}`,
    'X-Request-ID': `k6-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
  };
}

function checkResponse(res, name) {
  const success = check(res, {
    [`${name}: status is 200`]: (r) => r.status === 200,
    [`${name}: response time < 3s`]: (r) => r.timings.duration < 3000,
    [`${name}: has body`]: (r) => r.body && r.body.length > 0,
  });
  
  errorRate.add(!success);
  requestCount.add(1);
  
  return success;
}

// Test scenarios
export function smokeTest() {
  group('Smoke Test', () => {
    // Health check
    let res = http.get(`${BASE_URL}/health`, { headers: getHeaders() });
    checkResponse(res, 'health');
    apiLatency.add(res.timings.duration);
    
    // Readiness check
    res = http.get(`${BASE_URL}/ready`, { headers: getHeaders() });
    checkResponse(res, 'ready');
    
    sleep(1);
  });
}

export function loadTest() {
  group('Load Test - Code Review Flow', () => {
    // 1. Get user projects
    let res = http.get(`${BASE_URL}/api/v2/projects`, { headers: getHeaders() });
    checkResponse(res, 'list_projects');
    apiLatency.add(res.timings.duration);
    
    // 2. Submit code for review
    const codePayload = JSON.stringify({
      code: `
def calculate_sum(a, b):
    return a + b

def main():
    result = calculate_sum(5, 3)
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
      `,
      language: 'python',
      analysis_types: ['quality', 'security'],
    });
    
    res = http.post(`${BASE_URL}/api/v2/cr-ai/review`, codePayload, { 
      headers: getHeaders(),
      timeout: '30s',
    });
    checkResponse(res, 'code_review');
    codeReviewLatency.add(res.timings.duration);
    
    // 3. Get review results
    if (res.status === 200) {
      const reviewId = JSON.parse(res.body).review_id;
      res = http.get(`${BASE_URL}/api/v2/cr-ai/review/${reviewId}`, { 
        headers: getHeaders() 
      });
      checkResponse(res, 'get_review');
    }
    
    sleep(Math.random() * 2 + 1); // Random 1-3s think time
  });
}

export function stressTest() {
  group('Stress Test - High Volume', () => {
    // Rapid fire requests
    for (let i = 0; i < 5; i++) {
      const res = http.get(`${BASE_URL}/api/v2/projects`, { headers: getHeaders() });
      checkResponse(res, 'stress_projects');
      apiLatency.add(res.timings.duration);
    }
    
    // Code analysis under stress
    const codePayload = JSON.stringify({
      code: 'print("stress test")',
      language: 'python',
      analysis_types: ['quality'],
    });
    
    const res = http.post(`${BASE_URL}/api/v2/cr-ai/review`, codePayload, { 
      headers: getHeaders(),
      timeout: '60s',
    });
    checkResponse(res, 'stress_review');
    codeReviewLatency.add(res.timings.duration);
    
    sleep(0.5);
  });
}

export function spikeTest() {
  group('Spike Test - Sudden Load', () => {
    // Simulate sudden traffic surge
    const requests = [];
    
    for (let i = 0; i < 3; i++) {
      requests.push(['GET', `${BASE_URL}/health`, null, { headers: getHeaders() }]);
      requests.push(['GET', `${BASE_URL}/api/v2/projects`, null, { headers: getHeaders() }]);
    }
    
    const responses = http.batch(requests);
    
    responses.forEach((res, idx) => {
      checkResponse(res, `spike_batch_${idx}`);
      apiLatency.add(res.timings.duration);
    });
    
    sleep(0.2);
  });
}

// Lifecycle hooks
export function setup() {
  console.log('Load test starting...');
  console.log(`Target: ${BASE_URL}`);
  
  // Verify system is accessible
  const res = http.get(`${BASE_URL}/health`);
  if (res.status !== 200) {
    throw new Error(`System not healthy: ${res.status}`);
  }
  
  return { startTime: Date.now() };
}

export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  console.log(`Load test completed in ${duration}s`);
}

// Summary handler
export function handleSummary(data) {
  return {
    'stdout': textSummary(data, { indent: ' ', enableColors: true }),
    'results/load_test_summary.json': JSON.stringify(data, null, 2),
    'results/load_test_summary.html': htmlReport(data),
  };
}

function textSummary(data, options) {
  return `
========================================
        LOAD TEST SUMMARY
========================================
Duration: ${data.state.testRunDurationMs / 1000}s
Scenarios: ${Object.keys(options.scenarios || {}).length}

Requests:
  Total: ${data.metrics.http_reqs?.values?.count || 0}
  Rate: ${(data.metrics.http_reqs?.values?.rate || 0).toFixed(2)}/s
  Failed: ${(data.metrics.http_req_failed?.values?.rate * 100 || 0).toFixed(2)}%

Response Times:
  Avg: ${(data.metrics.http_req_duration?.values?.avg || 0).toFixed(2)}ms
  p95: ${(data.metrics.http_req_duration?.values?.['p(95)'] || 0).toFixed(2)}ms
  p99: ${(data.metrics.http_req_duration?.values?.['p(99)'] || 0).toFixed(2)}ms

Custom Metrics:
  Error Rate: ${((data.metrics.errors?.values?.rate || 0) * 100).toFixed(2)}%
  API Latency p95: ${(data.metrics.api_latency?.values?.['p(95)'] || 0).toFixed(2)}ms
  Code Review p95: ${(data.metrics.code_review_latency?.values?.['p(95)'] || 0).toFixed(2)}ms

Thresholds:
${Object.entries(data.metrics || {})
  .filter(([k, v]) => v.thresholds)
  .map(([k, v]) => `  ${k}: ${v.thresholds ? (Object.values(v.thresholds).every(t => t.ok) ? '✓ PASS' : '✗ FAIL') : 'N/A'}`)
  .join('\n')}
========================================
`;
}

function htmlReport(data) {
  return `<!DOCTYPE html>
<html>
<head><title>Load Test Report</title></head>
<body>
<h1>Load Test Report</h1>
<pre>${JSON.stringify(data, null, 2)}</pre>
</body>
</html>`;
}
