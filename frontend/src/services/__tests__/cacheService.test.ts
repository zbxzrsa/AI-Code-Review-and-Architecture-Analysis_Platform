/**
 * Cache Service Unit Tests
 */

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

// Mock storage APIs
const mockSessionStorage: Record<string, string> = {};
const mockLocalStorage: Record<string, string> = {};

vi.stubGlobal("sessionStorage", {
  getItem: (key: string) => mockSessionStorage[key] || null,
  setItem: (key: string, value: string) => {
    mockSessionStorage[key] = value;
  },
  removeItem: (key: string) => {
    delete mockSessionStorage[key];
  },
  clear: () => {
    Object.keys(mockSessionStorage).forEach(
      (k) => delete mockSessionStorage[k]
    );
  },
  key: (index: number) => Object.keys(mockSessionStorage)[index] || null,
  get length() {
    return Object.keys(mockSessionStorage).length;
  },
});

vi.stubGlobal("localStorage", {
  getItem: (key: string) => mockLocalStorage[key] || null,
  setItem: (key: string, value: string) => {
    mockLocalStorage[key] = value;
  },
  removeItem: (key: string) => {
    delete mockLocalStorage[key];
  },
  clear: () => {
    Object.keys(mockLocalStorage).forEach((k) => delete mockLocalStorage[k]);
  },
  key: (index: number) => Object.keys(mockLocalStorage)[index] || null,
  get length() {
    return Object.keys(mockLocalStorage).length;
  },
});

import { cacheService } from "../cacheService";

describe("CacheService", () => {
  const cache = cacheService;

  beforeEach(() => {
    cache.clearAll();
    vi.useFakeTimers();
  });

  afterEach(() => {
    cache.clearAll();
    vi.useRealTimers();
    Object.keys(mockSessionStorage).forEach(
      (k) => delete mockSessionStorage[k]
    );
    Object.keys(mockLocalStorage).forEach((k) => delete mockLocalStorage[k]);
  });

  describe("Memory Cache (L1)", () => {
    it("should store and retrieve values", () => {
      cache.set("test-key", { data: "test" });
      const result = cache.get("test-key");

      expect(result).toBeDefined();
      expect(result?.data).toEqual({ data: "test" });
    });

    it("should return null for non-existent keys", () => {
      const result = cache.get("non-existent");
      expect(result).toBeNull();
    });

    it("should expire entries after TTL", () => {
      cache.set("test-key", { data: "test" }, { ttl: 1000 });

      // Before expiry
      expect(cache.get("test-key")).not.toBeNull();

      // After expiry
      vi.advanceTimersByTime(1001);
      expect(cache.get("test-key")).toBeNull();
    });

    it("should track hits and misses", () => {
      cache.set("hit-key", { data: "test" });

      cache.get("hit-key"); // Hit
      cache.get("hit-key"); // Hit
      cache.get("miss-key"); // Miss

      const stats = cache.getStats();
      expect(stats.memory.hits).toBe(2);
      expect(stats.memory.misses).toBe(1);
    });

    it("should calculate hit rate correctly", () => {
      cache.set("key", "value");

      cache.get("key"); // Hit
      cache.get("key"); // Hit
      cache.get("miss"); // Miss
      cache.get("miss2"); // Miss

      const stats = cache.getStats();
      expect(stats.memory.hitRate).toBe(0.5);
    });
  });

  describe("Session Cache (L2)", () => {
    it("should store in session storage", () => {
      cache.set("session-key", { data: "session" }, { layers: ["session"] });

      const result = cache.get("session-key");
      expect(result?.data).toEqual({ data: "session" });
    });
  });

  describe("Local Cache (L3)", () => {
    it("should store in local storage", () => {
      cache.set("local-key", { data: "local" }, { layers: ["local"] });

      const result = cache.get("local-key");
      expect(result?.data).toEqual({ data: "local" });
    });
  });

  describe("Multi-Layer Caching", () => {
    it("should store in multiple layers", () => {
      cache.set(
        "multi-key",
        { data: "multi" },
        {
          layers: ["memory", "session", "local"],
        }
      );

      // Clear memory cache only
      cache.deleteMemory("multi-key");

      // Should still find in session/local
      const result = cache.get("multi-key");
      expect(result?.data).toEqual({ data: "multi" });
    });
  });

  describe("Tags", () => {
    it("should invalidate by tag", () => {
      cache.set("user-1", { name: "John" }, { tags: ["users"] });
      cache.set("user-2", { name: "Jane" }, { tags: ["users"] });
      cache.set("product-1", { name: "Widget" }, { tags: ["products"] });

      cache.invalidateByTag("users");

      expect(cache.get("user-1")).toBeNull();
      expect(cache.get("user-2")).toBeNull();
      expect(cache.get("product-1")).not.toBeNull();
    });
  });

  describe("Pattern Invalidation", () => {
    it("should invalidate by pattern", () => {
      cache.set("api:/users/1", { id: 1 });
      cache.set("api:/users/2", { id: 2 });
      cache.set("api:/products/1", { id: 1 });

      cache.invalidateByPattern(/api:\/users/);

      expect(cache.get("api:/users/1")).toBeNull();
      expect(cache.get("api:/users/2")).toBeNull();
      expect(cache.get("api:/products/1")).not.toBeNull();
    });
  });

  describe("Clear All", () => {
    it("should clear all caches", () => {
      cache.set("key1", "value1");
      cache.set("key2", "value2");

      cache.clearAll();

      expect(cache.get("key1")).toBeNull();
      expect(cache.get("key2")).toBeNull();
    });
  });

  describe("Statistics", () => {
    it("should provide accurate statistics", () => {
      cache.set("key1", "value1");
      cache.set("key2", "value2");
      cache.get("key1");
      cache.get("key2");
      cache.get("miss");

      const stats = cache.getStats();

      expect(stats.memory.size).toBe(2);
      expect(stats.memory.hits).toBe(2);
      expect(stats.memory.misses).toBe(1);
      expect(stats.memory.hitRate).toBeCloseTo(0.67, 1);
    });
  });

  describe("Delete", () => {
    it("should delete from all layers", () => {
      cache.set("delete-me", "value", {
        layers: ["memory", "session", "local"],
      });

      expect(cache.get("delete-me")).not.toBeNull();

      cache.delete("delete-me");

      expect(cache.get("delete-me")).toBeNull();
    });
  });

  describe("Reset Stats", () => {
    it("should reset statistics", () => {
      cache.set("key", "value");
      cache.get("key"); // Hit
      cache.get("miss"); // Miss

      cache.resetStats();

      const stats = cache.getStats();
      expect(stats.memory.hits).toBe(0);
      expect(stats.memory.misses).toBe(0);
    });
  });
});
