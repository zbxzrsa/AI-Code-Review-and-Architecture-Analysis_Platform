/**
 * Jest Test Setup
 *
 * This file runs before each test file
 */

import "@testing-library/jest-dom";
import { cleanup } from "@testing-library/react";
import { afterEach, beforeAll, afterAll, vi } from "vitest";

// Cleanup after each test
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

// Mock window.matchMedia
beforeAll(() => {
  Object.defineProperty(globalThis, "matchMedia", {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  });

  // Mock IntersectionObserver
  class MockIntersectionObserver {
    observe = vi.fn();
    disconnect = vi.fn();
    unobserve = vi.fn();
  }
  Object.defineProperty(globalThis, "IntersectionObserver", {
    writable: true,
    value: MockIntersectionObserver,
  });

  // Mock ResizeObserver
  class MockResizeObserver {
    observe = vi.fn();
    disconnect = vi.fn();
    unobserve = vi.fn();
  }
  Object.defineProperty(globalThis, "ResizeObserver", {
    writable: true,
    value: MockResizeObserver,
  });

  // Mock scrollTo
  Object.defineProperty(globalThis, "scrollTo", {
    writable: true,
    value: vi.fn(),
  });

  // Mock localStorage
  const localStorageMock = {
    getItem: vi.fn(),
    setItem: vi.fn(),
    removeItem: vi.fn(),
    clear: vi.fn(),
    length: 0,
    key: vi.fn(),
  };
  Object.defineProperty(globalThis, "localStorage", {
    value: localStorageMock,
  });

  // Mock sessionStorage
  Object.defineProperty(globalThis, "sessionStorage", {
    value: localStorageMock,
  });

  // Suppress console errors during tests
  vi.spyOn(console, "error").mockImplementation(() => {});
  vi.spyOn(console, "warn").mockImplementation(() => {});
});

afterAll(() => {
  vi.restoreAllMocks();
});

// Mock fetch globally
globalThis.fetch = vi.fn() as typeof fetch;

// Mock crypto for UUID generation
Object.defineProperty(globalThis, "crypto", {
  value: {
    randomUUID: () => "test-uuid-" + Math.random().toString(36).substr(2, 9),
    getRandomValues: (arr: Uint8Array) => {
      for (let i = 0; i < arr.length; i++) {
        arr[i] = Math.floor(Math.random() * 256);
      }
      return arr;
    },
  },
});
