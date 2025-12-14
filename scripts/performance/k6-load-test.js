/**
 * k6 性能测试脚本
 * 
 * 测试目标：
 * - API网关路由性能
 * - 发布闸门决策性能
 * - 沙盒编排性能
 * - 技术监测选择性能
 * 
 * 运行方式：
 * k6 run scripts/performance/k6-load-test.js
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// 自定义指标
const errorRate = new Rate('errors');
const apiGatewayLatency = new Trend('api_gateway_latency');
const releaseGateLatency = new Trend('release_gate_latency');
const sandboxLatency = new Trend('sandbox_latency');
const techMonitorLatency = new Trend('tech_monitor_latency');

// 测试配置
export const options = {
  stages: [
    { duration: '30s', target: 10 },   // 预热：10用户/30秒
    { duration: '1m', target: 50 },    // 爬升：50用户/1分钟
    { duration: '2m', target: 100 },   // 稳定：100用户/2分钟
    { duration: '1m', target: 50 },    // 下降：50用户/1分钟
    { duration: '30s', target: 0 },     // 冷却：0用户/30秒
  ],
  thresholds: {
    'http_req_duration': ['p(95)<500', 'p(99)<1000'], // 95%请求<500ms, 99%<1000ms
    'errors': ['rate<0.01'],                          // 错误率<1%
    'api_gateway_latency': ['p(95)<200'],             // API网关95%<200ms
    'release_gate_latency': ['p(95)<100'],           // 发布闸门95%<100ms
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:3000';

export default function () {
  // 测试1: API网关路由
  const apiGatewayStart = Date.now();
  const apiGatewayRes = http.post(`${BASE_URL}/api/gateway/route`, JSON.stringify({
    path: '/api/test',
    isExperimental: Math.random() > 0.5,
    clientVersion: 'v2',
  }), {
    headers: { 'Content-Type': 'application/json' },
  });
  
  const apiGatewayDuration = Date.now() - apiGatewayStart;
  apiGatewayLatency.add(apiGatewayDuration);
  
  check(apiGatewayRes, {
    'API网关状态200': (r) => r.status === 200,
    'API网关响应时间<500ms': (r) => r.timings.duration < 500,
  }) || errorRate.add(1);

  sleep(0.1);

  // 测试2: 发布闸门决策
  const releaseGateStart = Date.now();
  const releaseGateRes = http.post(`${BASE_URL}/api/release-gate/approve`, JSON.stringify({
    sigs: { v1: true, v3: true, v2: true },
    report: {
      p99DeltaPct: -5,
      errorRateDeltaPct: -2,
      allKnownIssuesClosed: true,
      loadTestPassed: true,
      scenarioSimPassed: true,
    },
  }), {
    headers: { 'Content-Type': 'application/json' },
  });
  
  const releaseGateDuration = Date.now() - releaseGateStart;
  releaseGateLatency.add(releaseGateDuration);
  
  check(releaseGateRes, {
    '发布闸门状态200': (r) => r.status === 200,
    '发布闸门响应时间<200ms': (r) => r.timings.duration < 200,
  }) || errorRate.add(1);

  sleep(0.1);

  // 测试3: 沙盒编排
  const sandboxStart = Date.now();
  const sandboxRes = http.post(`${BASE_URL}/api/sandbox/experiment`, JSON.stringify({
    candidateId: `test-${Math.random()}`,
    cycles: 3,
    enableShadow: true,
    enableAB: true,
    knownIssuesClosed: true,
    perfDeltaPct: 20,
  }), {
    headers: { 'Content-Type': 'application/json' },
  });
  
  const sandboxDuration = Date.now() - sandboxStart;
  sandboxLatency.add(sandboxDuration);
  
  check(sandboxRes, {
    '沙盒编排状态200': (r) => r.status === 200,
    '沙盒编排响应时间<300ms': (r) => r.timings.duration < 300,
  }) || errorRate.add(1);

  sleep(0.1);

  // 测试4: 技术监测
  const techMonitorStart = Date.now();
  const techMonitorRes = http.post(`${BASE_URL}/api/tech-monitor/select`, JSON.stringify({
    candidates: [
      {
        id: 'candidate-1',
        name: 'Tech A',
        rationale: '高性能',
        latencyImprovementPct: 10,
        throughputImprovementPct: 15,
      },
      {
        id: 'candidate-2',
        name: 'Tech B',
        rationale: '更稳定',
        latencyImprovementPct: 20,
        throughputImprovementPct: 10,
      },
    ],
  }), {
    headers: { 'Content-Type': 'application/json' },
  });
  
  const techMonitorDuration = Date.now() - techMonitorStart;
  techMonitorLatency.add(techMonitorDuration);
  
  check(techMonitorRes, {
    '技术监测状态200': (r) => r.status === 200,
    '技术监测响应时间<150ms': (r) => r.timings.duration < 150,
  }) || errorRate.add(1);

  sleep(0.5); // 请求间隔
}

export function handleSummary(data) {
  return {
    'stdout': textSummary(data, { indent: ' ', enableColors: true }),
    'performance-report.json': JSON.stringify(data),
  };
}

function textSummary(data, options) {
  // 简化的文本摘要
  return `
性能测试摘要:
- 总请求数: ${data.metrics.http_reqs.values.count}
- 平均响应时间: ${data.metrics.http_req_duration.values.avg.toFixed(2)}ms
- P95响应时间: ${data.metrics.http_req_duration.values['p(95)'].toFixed(2)}ms
- P99响应时间: ${data.metrics.http_req_duration.values['p(99)'].toFixed(2)}ms
- 错误率: ${(data.metrics.errors.values.rate * 100).toFixed(2)}%
- API网关P95: ${data.metrics.api_gateway_latency.values['p(95)'].toFixed(2)}ms
- 发布闸门P95: ${data.metrics.release_gate_latency.values['p(95)'].toFixed(2)}ms
`;
}

