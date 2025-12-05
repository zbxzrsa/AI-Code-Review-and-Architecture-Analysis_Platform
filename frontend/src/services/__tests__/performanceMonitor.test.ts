/**
 * Performance Monitor Unit Tests
 */

import { describe, it, expect, beforeEach, vi } from "vitest";

// Mock eventBus
vi.mock("../eventBus", () => ({
  eventBus: {
    emit: vi.fn(),
    emitSync: vi.fn(),
  },
}));

// Mock PerformanceObserver
vi.stubGlobal(
  "PerformanceObserver",
  class {
    // Empty implementation - PerformanceObserver not available in test environment
    observe() {
      // No-op: Test environment doesn't support performance observation
    }
    disconnect() {
      // No-op: Nothing to disconnect in test environment
    }
  }
);

// Import after mocking
import { performanceMonitor } from "../performanceMonitor";

describe("PerformanceMonitor", () => {
  const monitor = performanceMonitor;

  beforeEach(() => {
    monitor.clearMetrics();
    vi.clearAllMocks();
  });

  describe("API Tracking", () => {
    it("should track API call metrics", () => {
      monitor.trackAPICall("/api/users", "GET", 150, 200, false);

      const stats = monitor.getAPIStats();
      expect(stats.totalCalls).toBe(1);
      expect(stats.avgResponseTime).toBe(150);
    });

    it("should calculate average response time correctly", () => {
      monitor.trackAPICall("/api/users", "GET", 100, 200);
      monitor.trackAPICall("/api/users", "GET", 200, 200);
      monitor.trackAPICall("/api/users", "GET", 300, 200);

      const stats = monitor.getAPIStats();
      expect(stats.avgResponseTime).toBe(200);
    });

    it("should track cached calls separately", () => {
      monitor.trackAPICall("/api/users", "GET", 5, 200, true);
      monitor.trackAPICall("/api/users", "GET", 100, 200, false);

      const stats = monitor.getAPIStats();
      expect(stats.cacheHitRate).toBe(0.5);
    });

    it("should track error rate", () => {
      monitor.trackAPICall("/api/users", "GET", 100, 200);
      monitor.trackAPICall("/api/users", "GET", 100, 500);
      monitor.trackAPICall("/api/users", "GET", 100, 404);

      const stats = monitor.getAPIStats();
      expect(stats.errorRate).toBeCloseTo(0.67, 1);
    });

    it("should calculate P95 latency", () => {
      // Add 100 calls with varying latencies
      for (let i = 1; i <= 100; i++) {
        monitor.trackAPICall("/api/test", "GET", i * 10, 200);
      }

      const stats = monitor.getAPIStats();
      expect(stats.p95ResponseTime).toBeGreaterThan(900);
      expect(stats.p95ResponseTime).toBeLessThanOrEqual(1000);
    });
  });

  describe("Component Tracking", () => {
    it("should track component render time", () => {
      monitor.trackComponentRender("Dashboard", 25);
      monitor.trackComponentRender("Dashboard", 35);

      const stats = monitor.getComponentStats();
      const dashboard = stats.find((c) => c.componentName === "Dashboard");

      expect(dashboard).toBeDefined();
      expect(dashboard?.updateCount).toBe(2);
      expect(dashboard?.renderTime).toBe(35); // Last render time
    });

    it("should track multiple components", () => {
      monitor.trackComponentRender("Dashboard", 20);
      monitor.trackComponentRender("Sidebar", 15);
      monitor.trackComponentRender("Header", 10);

      const stats = monitor.getComponentStats();
      expect(stats.length).toBe(3);
    });
  });

  describe("Custom Metrics", () => {
    it("should record custom metrics", () => {
      monitor.recordMetric("custom.metric", 42, "count");

      const report = monitor.getReport();
      const metric = report.customMetrics.find(
        (m) => m.name === "custom.metric"
      );

      expect(metric).toBeDefined();
      expect(metric?.value).toBe(42);
    });
  });

  describe("Memory Tracking", () => {
    it("should return memory stats structure", () => {
      // trackMemory requires performance.memory which isn't available in test env
      const stats = monitor.getMemoryStats();
      // Memory stats should be defined with default values
      expect(stats).toBeDefined();
      expect(stats.current).toBeNull(); // No memory API in test env
      expect(stats.trend).toBe("stable");
    });
  });

  describe("Report Generation", () => {
    it("should generate comprehensive report", () => {
      monitor.trackAPICall("/api/test", "GET", 100, 200);
      monitor.trackComponentRender("Test", 20);
      monitor.recordMetric("test.metric", 10, "ms");

      const report = monitor.getReport();

      expect(report.api).toBeDefined();
      expect(report.components).toBeDefined();
      expect(report.memory).toBeDefined();
      expect(report.customMetrics).toBeDefined();
      expect(report.timestamp).toBeDefined();
    });
  });

  describe("Clear Metrics", () => {
    it("should clear all metrics", () => {
      monitor.trackAPICall("/api/test", "GET", 100, 200);
      monitor.trackComponentRender("Test", 20);

      monitor.clearMetrics();

      const stats = monitor.getAPIStats();
      expect(stats.totalCalls).toBe(0);
    });
  });
});
