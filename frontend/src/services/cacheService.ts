/**
 * Frontend Cache Service
 *
 * Multi-layer caching for performance optimization:
 * - Memory cache (L1) - Fast, limited size
 * - Session storage (L2) - Persists during session
 * - Local storage (L3) - Persists across sessions
 * - IndexedDB (L4) - Large data storage
 */

import { performanceMonitor } from "./performanceMonitor";

// Cache entry with metadata
interface CacheEntry<T> {
  data: T;
  timestamp: number;
  ttl: number;
  hits: number;
  tags?: string[];
}

// Cache statistics
interface CacheStats {
  hits: number;
  misses: number;
  size: number;
  hitRate: number;
}

// Cache configuration
interface CacheConfig {
  maxMemoryItems: number;
  defaultTTL: number;
  enableCompression: boolean;
  enableStats: boolean;
}

const DEFAULT_CONFIG: CacheConfig = {
  maxMemoryItems: 500,
  defaultTTL: 5 * 60 * 1000, // 5 minutes
  enableCompression: true,
  enableStats: true,
};

class CacheService {
  private memoryCache: Map<string, CacheEntry<unknown>> = new Map();
  private config: CacheConfig;
  private stats = {
    memoryHits: 0,
    memoryMisses: 0,
    sessionHits: 0,
    sessionMisses: 0,
    localHits: 0,
    localMisses: 0,
  };

  constructor(config: Partial<CacheConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.startCleanupInterval();
  }

  // ==================== Memory Cache (L1) ====================

  /**
   * Get from memory cache
   */
  getMemory<T>(key: string): T | null {
    const entry = this.memoryCache.get(key) as CacheEntry<T> | undefined;

    if (!entry) {
      this.stats.memoryMisses++;
      return null;
    }

    if (this.isExpired(entry)) {
      this.memoryCache.delete(key);
      this.stats.memoryMisses++;
      return null;
    }

    entry.hits++;
    this.stats.memoryHits++;
    return entry.data;
  }

  /**
   * Set in memory cache
   */
  setMemory<T>(key: string, data: T, ttl?: number, tags?: string[]): void {
    // Ensure we don't exceed max items
    if (this.memoryCache.size >= this.config.maxMemoryItems) {
      this.evictLRU();
    }

    const entry: CacheEntry<T> = {
      data,
      timestamp: Date.now(),
      ttl: ttl || this.config.defaultTTL,
      hits: 0,
      tags,
    };

    this.memoryCache.set(key, entry);
  }

  /**
   * Delete from memory cache
   */
  deleteMemory(key: string): boolean {
    return this.memoryCache.delete(key);
  }

  // ==================== Session Storage (L2) ====================

  /**
   * Get from session storage
   */
  getSession<T>(key: string): T | null {
    try {
      const raw = sessionStorage.getItem(`cache:${key}`);
      if (!raw) {
        this.stats.sessionMisses++;
        return null;
      }

      const entry: CacheEntry<T> = JSON.parse(raw);

      if (this.isExpired(entry)) {
        sessionStorage.removeItem(`cache:${key}`);
        this.stats.sessionMisses++;
        return null;
      }

      this.stats.sessionHits++;
      return entry.data;
    } catch {
      this.stats.sessionMisses++;
      return null;
    }
  }

  /**
   * Set in session storage
   */
  setSession<T>(key: string, data: T, ttl?: number, tags?: string[]): void {
    try {
      const entry: CacheEntry<T> = {
        data,
        timestamp: Date.now(),
        ttl: ttl || this.config.defaultTTL,
        hits: 0,
        tags,
      };

      sessionStorage.setItem(`cache:${key}`, JSON.stringify(entry));
    } catch (error) {
      // Storage quota exceeded, clear old items
      this.clearExpiredSession();
      console.warn(
        "[Cache] Session storage quota exceeded, clearing expired items"
      );
    }
  }

  /**
   * Delete from session storage
   */
  deleteSession(key: string): void {
    sessionStorage.removeItem(`cache:${key}`);
  }

  // ==================== Local Storage (L3) ====================

  /**
   * Get from local storage
   */
  getLocal<T>(key: string): T | null {
    try {
      const raw = localStorage.getItem(`cache:${key}`);
      if (!raw) {
        this.stats.localMisses++;
        return null;
      }

      const entry: CacheEntry<T> = JSON.parse(raw);

      if (this.isExpired(entry)) {
        localStorage.removeItem(`cache:${key}`);
        this.stats.localMisses++;
        return null;
      }

      this.stats.localHits++;
      return entry.data;
    } catch {
      this.stats.localMisses++;
      return null;
    }
  }

  /**
   * Set in local storage
   */
  setLocal<T>(key: string, data: T, ttl?: number, tags?: string[]): void {
    try {
      const entry: CacheEntry<T> = {
        data,
        timestamp: Date.now(),
        ttl: ttl || 24 * 60 * 60 * 1000, // Default 24 hours for local
        hits: 0,
        tags,
      };

      localStorage.setItem(`cache:${key}`, JSON.stringify(entry));
    } catch (error) {
      // Storage quota exceeded, clear old items
      this.clearExpiredLocal();
      console.warn(
        "[Cache] Local storage quota exceeded, clearing expired items"
      );
    }
  }

  /**
   * Delete from local storage
   */
  deleteLocal(key: string): void {
    localStorage.removeItem(`cache:${key}`);
  }

  // ==================== Multi-Layer Cache ====================

  /**
   * Get from cache (checks all layers)
   */
  get<T>(
    key: string
  ): { data: T; layer: "memory" | "session" | "local" } | null {
    // Check L1 (Memory)
    let data = this.getMemory<T>(key);
    if (data !== null) {
      performanceMonitor.recordMetric("cache.hit", 1, "count", {
        layer: "memory",
      });
      return { data, layer: "memory" };
    }

    // Check L2 (Session)
    data = this.getSession<T>(key);
    if (data !== null) {
      // Promote to L1
      this.setMemory(key, data);
      performanceMonitor.recordMetric("cache.hit", 1, "count", {
        layer: "session",
      });
      return { data, layer: "session" };
    }

    // Check L3 (Local)
    data = this.getLocal<T>(key);
    if (data !== null) {
      // Promote to L1 and L2
      this.setMemory(key, data);
      this.setSession(key, data);
      performanceMonitor.recordMetric("cache.hit", 1, "count", {
        layer: "local",
      });
      return { data, layer: "local" };
    }

    performanceMonitor.recordMetric("cache.miss", 1, "count");
    return null;
  }

  /**
   * Set in cache (writes to specified layers)
   */
  set<T>(
    key: string,
    data: T,
    options: {
      ttl?: number;
      tags?: string[];
      layers?: ("memory" | "session" | "local")[];
    } = {}
  ): void {
    const { ttl, tags, layers = ["memory", "session"] } = options;

    if (layers.includes("memory")) {
      this.setMemory(key, data, ttl, tags);
    }
    if (layers.includes("session")) {
      this.setSession(key, data, ttl, tags);
    }
    if (layers.includes("local")) {
      this.setLocal(key, data, ttl, tags);
    }
  }

  /**
   * Delete from all cache layers
   */
  delete(key: string): void {
    this.deleteMemory(key);
    this.deleteSession(key);
    this.deleteLocal(key);
  }

  /**
   * Invalidate by tag
   */
  invalidateByTag(tag: string): number {
    let count = 0;

    // Clear from memory
    for (const [key, entry] of this.memoryCache.entries()) {
      if (entry.tags?.includes(tag)) {
        this.memoryCache.delete(key);
        count++;
      }
    }

    // Clear from session storage
    for (let i = sessionStorage.length - 1; i >= 0; i--) {
      const key = sessionStorage.key(i);
      if (key?.startsWith("cache:")) {
        try {
          const entry = JSON.parse(sessionStorage.getItem(key) || "{}");
          if (entry.tags?.includes(tag)) {
            sessionStorage.removeItem(key);
            count++;
          }
        } catch {
          // Ignore parse errors
        }
      }
    }

    // Clear from local storage
    for (let i = localStorage.length - 1; i >= 0; i--) {
      const key = localStorage.key(i);
      if (key?.startsWith("cache:")) {
        try {
          const entry = JSON.parse(localStorage.getItem(key) || "{}");
          if (entry.tags?.includes(tag)) {
            localStorage.removeItem(key);
            count++;
          }
        } catch {
          // Ignore parse errors
        }
      }
    }

    return count;
  }

  /**
   * Invalidate by pattern
   */
  invalidateByPattern(pattern: RegExp): number {
    let count = 0;

    // Clear from memory
    for (const key of this.memoryCache.keys()) {
      if (pattern.test(key)) {
        this.memoryCache.delete(key);
        count++;
      }
    }

    // Clear from session storage
    for (let i = sessionStorage.length - 1; i >= 0; i--) {
      const key = sessionStorage.key(i);
      if (key?.startsWith("cache:") && pattern.test(key.slice(6))) {
        sessionStorage.removeItem(key);
        count++;
      }
    }

    // Clear from local storage
    for (let i = localStorage.length - 1; i >= 0; i--) {
      const key = localStorage.key(i);
      if (key?.startsWith("cache:") && pattern.test(key.slice(6))) {
        localStorage.removeItem(key);
        count++;
      }
    }

    return count;
  }

  // ==================== Statistics ====================

  /**
   * Get cache statistics
   */
  getStats(): {
    memory: CacheStats;
    session: CacheStats;
    local: CacheStats;
    total: CacheStats;
  } {
    const memoryTotal = this.stats.memoryHits + this.stats.memoryMisses;
    const sessionTotal = this.stats.sessionHits + this.stats.sessionMisses;
    const localTotal = this.stats.localHits + this.stats.localMisses;
    const total = memoryTotal + sessionTotal + localTotal;

    return {
      memory: {
        hits: this.stats.memoryHits,
        misses: this.stats.memoryMisses,
        size: this.memoryCache.size,
        hitRate: memoryTotal > 0 ? this.stats.memoryHits / memoryTotal : 0,
      },
      session: {
        hits: this.stats.sessionHits,
        misses: this.stats.sessionMisses,
        size: this.getSessionCacheSize(),
        hitRate: sessionTotal > 0 ? this.stats.sessionHits / sessionTotal : 0,
      },
      local: {
        hits: this.stats.localHits,
        misses: this.stats.localMisses,
        size: this.getLocalCacheSize(),
        hitRate: localTotal > 0 ? this.stats.localHits / localTotal : 0,
      },
      total: {
        hits:
          this.stats.memoryHits + this.stats.sessionHits + this.stats.localHits,
        misses:
          this.stats.memoryMisses +
          this.stats.sessionMisses +
          this.stats.localMisses,
        size:
          this.memoryCache.size +
          this.getSessionCacheSize() +
          this.getLocalCacheSize(),
        hitRate:
          total > 0
            ? (this.stats.memoryHits +
                this.stats.sessionHits +
                this.stats.localHits) /
              total
            : 0,
      },
    };
  }

  /**
   * Reset statistics
   */
  resetStats(): void {
    this.stats = {
      memoryHits: 0,
      memoryMisses: 0,
      sessionHits: 0,
      sessionMisses: 0,
      localHits: 0,
      localMisses: 0,
    };
  }

  // ==================== Cleanup ====================

  /**
   * Clear all caches
   */
  clearAll(): void {
    this.memoryCache.clear();
    this.clearAllSession();
    this.clearAllLocal();
    this.resetStats();
  }

  /**
   * Clear expired items from all caches
   */
  clearExpired(): number {
    let count = 0;

    // Clear expired from memory
    for (const [key, entry] of this.memoryCache.entries()) {
      if (this.isExpired(entry)) {
        this.memoryCache.delete(key);
        count++;
      }
    }

    count += this.clearExpiredSession();
    count += this.clearExpiredLocal();

    return count;
  }

  // ==================== Private Helpers ====================

  private isExpired(entry: CacheEntry<unknown>): boolean {
    return Date.now() - entry.timestamp > entry.ttl;
  }

  private evictLRU(): void {
    // Find least recently used (lowest hits)
    let lruKey: string | null = null;
    let lruHits = Infinity;

    for (const [key, entry] of this.memoryCache.entries()) {
      if (entry.hits < lruHits) {
        lruHits = entry.hits;
        lruKey = key;
      }
    }

    if (lruKey) {
      this.memoryCache.delete(lruKey);
    }
  }

  private getSessionCacheSize(): number {
    let count = 0;
    for (let i = 0; i < sessionStorage.length; i++) {
      const key = sessionStorage.key(i);
      if (key?.startsWith("cache:")) count++;
    }
    return count;
  }

  private getLocalCacheSize(): number {
    let count = 0;
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key?.startsWith("cache:")) count++;
    }
    return count;
  }

  private clearExpiredSession(): number {
    let count = 0;
    for (let i = sessionStorage.length - 1; i >= 0; i--) {
      const key = sessionStorage.key(i);
      if (key?.startsWith("cache:")) {
        try {
          const entry = JSON.parse(sessionStorage.getItem(key) || "{}");
          if (this.isExpired(entry)) {
            sessionStorage.removeItem(key);
            count++;
          }
        } catch {
          // Remove invalid entries
          sessionStorage.removeItem(key);
          count++;
        }
      }
    }
    return count;
  }

  private clearExpiredLocal(): number {
    let count = 0;
    for (let i = localStorage.length - 1; i >= 0; i--) {
      const key = localStorage.key(i);
      if (key?.startsWith("cache:")) {
        try {
          const entry = JSON.parse(localStorage.getItem(key) || "{}");
          if (this.isExpired(entry)) {
            localStorage.removeItem(key);
            count++;
          }
        } catch {
          // Remove invalid entries
          localStorage.removeItem(key);
          count++;
        }
      }
    }
    return count;
  }

  private clearAllSession(): void {
    for (let i = sessionStorage.length - 1; i >= 0; i--) {
      const key = sessionStorage.key(i);
      if (key?.startsWith("cache:")) {
        sessionStorage.removeItem(key);
      }
    }
  }

  private clearAllLocal(): void {
    for (let i = localStorage.length - 1; i >= 0; i--) {
      const key = localStorage.key(i);
      if (key?.startsWith("cache:")) {
        localStorage.removeItem(key);
      }
    }
  }

  private startCleanupInterval(): void {
    // Clean expired items every 5 minutes
    setInterval(() => {
      const cleared = this.clearExpired();
      if (cleared > 0) {
        console.log(`[Cache] Cleared ${cleared} expired items`);
      }
    }, 5 * 60 * 1000);
  }
}

// Singleton instance
export const cacheService = new CacheService();

// Decorator for caching function results
export function cached<T>(
  keyGenerator: (...args: unknown[]) => string,
  options?: {
    ttl?: number;
    tags?: string[];
    layers?: ("memory" | "session" | "local")[];
  }
) {
  return function (
    _target: unknown,
    _propertyKey: string,
    descriptor: PropertyDescriptor
  ) {
    const originalMethod = descriptor.value;

    descriptor.value = async function (this: unknown, ...args: unknown[]) {
      const key = keyGenerator(...args);
      const cachedValue = cacheService.get<T>(key);

      if (cachedValue) {
        return cachedValue.data;
      }

      const result = await originalMethod.apply(this, args);
      cacheService.set(key, result, options);
      return result;
    };

    return descriptor;
  };
}

export default cacheService;
