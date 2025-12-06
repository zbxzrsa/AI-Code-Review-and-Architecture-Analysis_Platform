/**
 * Test Setup File
 *
 * Configuration for Vitest with React Testing Library
 */

import "@testing-library/jest-dom";
import { vi, afterEach } from "vitest";

// Mock globalThis.matchMedia
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

// Mock ResizeObserver
class ResizeObserverMock {
  observe = vi.fn();
  unobserve = vi.fn();
  disconnect = vi.fn();
}

globalThis.ResizeObserver = ResizeObserverMock;

// Mock IntersectionObserver
class IntersectionObserverMock {
  root = null;
  rootMargin = "";
  thresholds = [];
  observe = vi.fn();
  unobserve = vi.fn();
  disconnect = vi.fn();
  takeRecords = vi.fn().mockReturnValue([]);
}

globalThis.IntersectionObserver =
  IntersectionObserverMock as unknown as typeof IntersectionObserver;

// Mock scrollTo
globalThis.scrollTo = vi.fn() as unknown as typeof globalThis.scrollTo;

// Mock clipboard API
Object.assign(navigator, {
  clipboard: {
    writeText: vi.fn().mockResolvedValue(undefined),
    readText: vi.fn().mockResolvedValue(""),
  },
});

// Suppress console errors during tests (optional)
const originalError = console.error;
console.error = (...args: any[]) => {
  // Ignore React's act() warnings in tests
  if (
    typeof args[0] === "string" &&
    args[0].includes("Warning: An update to")
  ) {
    return;
  }
  originalError.apply(console, args);
};

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
const sessionStorageMock = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
  length: 0,
  key: vi.fn(),
};

Object.defineProperty(globalThis, "sessionStorage", {
  value: sessionStorageMock,
});

// Reset all mocks after each test
afterEach(() => {
  vi.clearAllMocks();
});
