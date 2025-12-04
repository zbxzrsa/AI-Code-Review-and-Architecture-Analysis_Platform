/**
 * Performance Monitoring Service
 *
 * Tracks and reports performance metrics for optimization:
 * - API response times
 * - Component render times
 * - Memory usage
 * - Network latency
 * - User interaction metrics
 */

import { eventBus } from "./eventBus";

// Performance metric types
interface PerformanceMetric {
  name: string;
  value: number;
  unit: "ms" | "bytes" | "count" | "percent";
  timestamp: number;
  tags?: Record<string, string>;
}

interface APIMetric {
  endpoint: string;
  method: string;
  duration: number;
  status: number;
  timestamp: number;
  cached?: boolean;
}

interface ComponentMetric {
  componentName: string;
  renderTime: number;
  mountTime?: number;
  updateCount: number;
  timestamp: number;
}

interface MemoryMetric {
  usedJSHeapSize: number;
  totalJSHeapSize: number;
  jsHeapSizeLimit: number;
  timestamp: number;
}

// Performance thresholds
const THRESHOLDS = {
  API_SLOW: 1000, // 1 second
  API_CRITICAL: 3000, // 3 seconds
  RENDER_SLOW: 16, // 16ms (60fps threshold)
  RENDER_CRITICAL: 100, // 100ms
  MEMORY_WARNING: 0.7, // 70% of limit
  MEMORY_CRITICAL: 0.9, // 90% of limit
};

class PerformanceMonitor {
  private metrics: PerformanceMetric[] = [];
  private apiMetrics: APIMetric[] = [];
  private componentMetrics: Map<string, ComponentMetric> = new Map();
  private memoryMetrics: MemoryMetric[] = [];
  private isEnabled: boolean = true;
  private maxMetrics: number = 1000;
  private reportInterval: number = 60000; // 1 minute
  private reportTimer: NodeJS.Timeout | null = null;

  constructor() {
    this.startPeriodicReporting();
    this.setupPerformanceObserver();
  }

  /**
   * Enable or disable monitoring
   */
  setEnabled(enabled: boolean): void {
    this.isEnabled = enabled;
    if (enabled) {
      this.startPeriodicReporting();
    } else {
      this.stopPeriodicReporting();
    }
  }

  /**
   * Track API call performance
   */
  trackAPICall(
    endpoint: string,
    method: string,
    duration: number,
    status: number,
    cached: boolean = false
  ): void {
    if (!this.isEnabled) return;

    const metric: APIMetric = {
      endpoint,
      method,
      duration,
      status,
      timestamp: Date.now(),
      cached,
    };

    this.apiMetrics.push(metric);
    this.trimMetrics();

    // Check thresholds and emit warnings
    if (duration > THRESHOLDS.API_CRITICAL) {
      eventBus.emit("performance:api:critical", {
        endpoint,
        duration,
        threshold: THRESHOLDS.API_CRITICAL,
      });
      console.warn(
        `[Performance] Critical API latency: ${endpoint} took ${duration}ms`
      );
    } else if (duration > THRESHOLDS.API_SLOW) {
      eventBus.emit("performance:api:slow", {
        endpoint,
        duration,
        threshold: THRESHOLDS.API_SLOW,
      });
    }

    // Track cache hit rate
    if (cached) {
      this.recordMetric("api.cache.hit", 1, "count", { endpoint });
    } else {
      this.recordMetric("api.cache.miss", 1, "count", { endpoint });
    }
  }

  /**
   * Track component render performance
   */
  trackComponentRender(componentName: string, renderTime: number): void {
    if (!this.isEnabled) return;

    const existing = this.componentMetrics.get(componentName);
    const metric: ComponentMetric = {
      componentName,
      renderTime,
      mountTime: existing?.mountTime,
      updateCount: (existing?.updateCount || 0) + 1,
      timestamp: Date.now(),
    };

    this.componentMetrics.set(componentName, metric);

    // Check thresholds
    if (renderTime > THRESHOLDS.RENDER_CRITICAL) {
      eventBus.emit("performance:render:critical", {
        componentName,
        renderTime,
        threshold: THRESHOLDS.RENDER_CRITICAL,
      });
      console.warn(
        `[Performance] Critical render time: ${componentName} took ${renderTime}ms`
      );
    } else if (renderTime > THRESHOLDS.RENDER_SLOW) {
      eventBus.emit("performance:render:slow", {
        componentName,
        renderTime,
        threshold: THRESHOLDS.RENDER_SLOW,
      });
    }
  }

  /**
   * Track component mount time
   */
  trackComponentMount(componentName: string, mountTime: number): void {
    if (!this.isEnabled) return;

    const existing = this.componentMetrics.get(componentName);
    const metric: ComponentMetric = {
      componentName,
      renderTime: existing?.renderTime || 0,
      mountTime,
      updateCount: existing?.updateCount || 0,
      timestamp: Date.now(),
    };

    this.componentMetrics.set(componentName, metric);
  }

  /**
   * Track memory usage
   */
  trackMemory(): MemoryMetric | null {
    if (!this.isEnabled) return null;

    // Check if performance.memory is available (Chrome only)
    const perf = performance as Performance & {
      memory?: {
        usedJSHeapSize: number;
        totalJSHeapSize: number;
        jsHeapSizeLimit: number;
      };
    };

    if (!perf.memory) return null;

    const metric: MemoryMetric = {
      usedJSHeapSize: perf.memory.usedJSHeapSize,
      totalJSHeapSize: perf.memory.totalJSHeapSize,
      jsHeapSizeLimit: perf.memory.jsHeapSizeLimit,
      timestamp: Date.now(),
    };

    this.memoryMetrics.push(metric);

    // Check memory thresholds
    const usageRatio = metric.usedJSHeapSize / metric.jsHeapSizeLimit;
    if (usageRatio > THRESHOLDS.MEMORY_CRITICAL) {
      eventBus.emit("performance:memory:critical", {
        usage: usageRatio,
        threshold: THRESHOLDS.MEMORY_CRITICAL,
      });
      console.warn(
        `[Performance] Critical memory usage: ${(usageRatio * 100).toFixed(1)}%`
      );
    } else if (usageRatio > THRESHOLDS.MEMORY_WARNING) {
      eventBus.emit("performance:memory:warning", {
        usage: usageRatio,
        threshold: THRESHOLDS.MEMORY_WARNING,
      });
    }

    return metric;
  }

  /**
   * Record a custom metric
   */
  recordMetric(
    name: string,
    value: number,
    unit: "ms" | "bytes" | "count" | "percent",
    tags?: Record<string, string>
  ): void {
    if (!this.isEnabled) return;

    const metric: PerformanceMetric = {
      name,
      value,
      unit,
      timestamp: Date.now(),
      tags,
    };

    this.metrics.push(metric);
    this.trimMetrics();
  }

  /**
   * Start timing an operation
   */
  startTimer(operationName: string): () => number {
    const startTime = performance.now();
    return () => {
      const duration = performance.now() - startTime;
      this.recordMetric(operationName, duration, "ms");
      return duration;
    };
  }

  /**
   * Get API performance statistics
   */
  getAPIStats(): {
    avgResponseTime: number;
    p95ResponseTime: number;
    p99ResponseTime: number;
    errorRate: number;
    cacheHitRate: number;
    totalCalls: number;
    slowCalls: number;
  } {
    if (this.apiMetrics.length === 0) {
      return {
        avgResponseTime: 0,
        p95ResponseTime: 0,
        p99ResponseTime: 0,
        errorRate: 0,
        cacheHitRate: 0,
        totalCalls: 0,
        slowCalls: 0,
      };
    }

    const durations = this.apiMetrics
      .map((m) => m.duration)
      .sort((a, b) => a - b);
    const errors = this.apiMetrics.filter((m) => m.status >= 400).length;
    const cached = this.apiMetrics.filter((m) => m.cached).length;
    const slow = this.apiMetrics.filter(
      (m) => m.duration > THRESHOLDS.API_SLOW
    ).length;

    return {
      avgResponseTime: durations.reduce((a, b) => a + b, 0) / durations.length,
      p95ResponseTime: durations[Math.floor(durations.length * 0.95)] || 0,
      p99ResponseTime: durations[Math.floor(durations.length * 0.99)] || 0,
      errorRate: errors / this.apiMetrics.length,
      cacheHitRate: cached / this.apiMetrics.length,
      totalCalls: this.apiMetrics.length,
      slowCalls: slow,
    };
  }

  /**
   * Get component performance statistics
   */
  getComponentStats(): ComponentMetric[] {
    return Array.from(this.componentMetrics.values());
  }

  /**
   * Get slowest components
   */
  getSlowestComponents(limit: number = 10): ComponentMetric[] {
    return this.getComponentStats()
      .sort((a, b) => b.renderTime - a.renderTime)
      .slice(0, limit);
  }

  /**
   * Get memory statistics
   */
  getMemoryStats(): {
    current: MemoryMetric | null;
    avgUsage: number;
    peakUsage: number;
    trend: "stable" | "increasing" | "decreasing";
  } {
    if (this.memoryMetrics.length === 0) {
      return {
        current: null,
        avgUsage: 0,
        peakUsage: 0,
        trend: "stable",
      };
    }

    const usages = this.memoryMetrics.map((m) => m.usedJSHeapSize);
    const avgUsage = usages.reduce((a, b) => a + b, 0) / usages.length;
    const peakUsage = Math.max(...usages);

    // Calculate trend from last 10 samples
    const recentMetrics = this.memoryMetrics.slice(-10);
    let trend: "stable" | "increasing" | "decreasing" = "stable";
    if (recentMetrics.length >= 2) {
      const first = recentMetrics[0].usedJSHeapSize;
      const last = recentMetrics[recentMetrics.length - 1].usedJSHeapSize;
      const change = (last - first) / first;
      if (change > 0.1) trend = "increasing";
      else if (change < -0.1) trend = "decreasing";
    }

    return {
      current: this.memoryMetrics[this.memoryMetrics.length - 1] || null,
      avgUsage,
      peakUsage,
      trend,
    };
  }

  /**
   * Get full performance report
   */
  getReport() {
    return {
      api: this.getAPIStats(),
      components: this.getComponentStats(),
      memory: this.getMemoryStats(),
      customMetrics: this.metrics.slice(-100),
      timestamp: Date.now(),
    };
  }

  /**
   * Clear all metrics
   */
  clearMetrics(): void {
    this.metrics = [];
    this.apiMetrics = [];
    this.componentMetrics.clear();
    this.memoryMetrics = [];
  }

  /**
   * Export metrics for analysis
   */
  exportMetrics(): string {
    return JSON.stringify(this.getReport(), null, 2);
  }

  // Private methods

  private trimMetrics(): void {
    if (this.metrics.length > this.maxMetrics) {
      this.metrics = this.metrics.slice(-this.maxMetrics);
    }
    if (this.apiMetrics.length > this.maxMetrics) {
      this.apiMetrics = this.apiMetrics.slice(-this.maxMetrics);
    }
    if (this.memoryMetrics.length > this.maxMetrics) {
      this.memoryMetrics = this.memoryMetrics.slice(-this.maxMetrics);
    }
  }

  private startPeriodicReporting(): void {
    this.stopPeriodicReporting();
    this.reportTimer = setInterval(() => {
      this.trackMemory();
      eventBus.emit("performance:report", this.getReport());
    }, this.reportInterval);
  }

  private stopPeriodicReporting(): void {
    if (this.reportTimer) {
      clearInterval(this.reportTimer);
      this.reportTimer = null;
    }
  }

  private setupPerformanceObserver(): void {
    if (typeof PerformanceObserver === "undefined") return;

    // Observe long tasks
    try {
      const longTaskObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          this.recordMetric("longTask", entry.duration, "ms", {
            name: entry.name,
          });
          if (entry.duration > 50) {
            eventBus.emit("performance:longTask", {
              duration: entry.duration,
              name: entry.name,
            });
          }
        }
      });
      longTaskObserver.observe({ entryTypes: ["longtask"] });
    } catch {
      // Long task observer not supported
    }

    // Observe largest contentful paint
    try {
      const lcpObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        const lastEntry = entries[entries.length - 1];
        if (lastEntry) {
          this.recordMetric("lcp", lastEntry.startTime, "ms");
          eventBus.emit("performance:lcp", { value: lastEntry.startTime });
        }
      });
      lcpObserver.observe({ entryTypes: ["largest-contentful-paint"] });
    } catch {
      // LCP observer not supported
    }

    // Observe first input delay
    try {
      const fidObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        const firstEntry = entries[0];
        if (firstEntry) {
          const processingStart = (firstEntry as PerformanceEventTiming)
            .processingStart;
          const fid = processingStart - firstEntry.startTime;
          this.recordMetric("fid", fid, "ms");
          eventBus.emit("performance:fid", { value: fid });
        }
      });
      fidObserver.observe({ entryTypes: ["first-input"] });
    } catch {
      // FID observer not supported
    }
  }
}

// Singleton instance
export const performanceMonitor = new PerformanceMonitor();

// React hook for component performance tracking
export function usePerformanceTracking(componentName: string) {
  const mountTime = performance.now();

  // Track mount time on first render
  if (typeof window !== "undefined") {
    requestAnimationFrame(() => {
      performanceMonitor.trackComponentMount(
        componentName,
        performance.now() - mountTime
      );
    });
  }

  return {
    trackRender: (renderStart: number) => {
      const renderTime = performance.now() - renderStart;
      performanceMonitor.trackComponentRender(componentName, renderTime);
    },
    startTimer: () =>
      performanceMonitor.startTimer(`${componentName}.operation`),
  };
}

export default performanceMonitor;
