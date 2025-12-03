/**
 * useAsyncData Hook
 *
 * A reusable hook for fetching data with:
 * - Loading states
 * - Error handling
 * - Refresh functionality
 * - Retry logic
 * - Caching
 */

import { useState, useEffect, useCallback, useRef } from "react";

// ============================================
// Types
// ============================================

export interface AsyncDataState<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
  isRefreshing: boolean;
  lastUpdated: Date | null;
}

export interface AsyncDataOptions {
  /** Whether to fetch immediately on mount */
  immediate?: boolean;
  /** Retry count on failure */
  retryCount?: number;
  /** Retry delay in ms */
  retryDelay?: number;
  /** Cache duration in ms (0 = no cache) */
  cacheDuration?: number;
  /** Dependencies that trigger refetch */
  deps?: unknown[];
  /** Callback on success */
  onSuccess?: (data: unknown) => void;
  /** Callback on error */
  onError?: (error: Error) => void;
}

export interface AsyncDataResult<T> extends AsyncDataState<T> {
  /** Manually trigger fetch */
  fetch: () => Promise<void>;
  /** Refresh data (shows refreshing state instead of loading) */
  refresh: () => Promise<void>;
  /** Reset state to initial */
  reset: () => void;
  /** Manually set data */
  setData: (data: T | null) => void;
  /** Retry after error */
  retry: () => Promise<void>;
}

// ============================================
// Cache Implementation
// ============================================

interface CacheEntry<T> {
  data: T;
  timestamp: number;
}

const cache = new Map<string, CacheEntry<unknown>>();

const getCached = <T>(key: string, duration: number): T | null => {
  const entry = cache.get(key);
  if (!entry) return null;

  const now = Date.now();
  if (now - entry.timestamp > duration) {
    cache.delete(key);
    return null;
  }

  return entry.data as T;
};

const setCache = <T>(key: string, data: T): void => {
  cache.set(key, { data, timestamp: Date.now() });
};

// ============================================
// Hook Implementation
// ============================================

export function useAsyncData<T>(
  fetcher: () => Promise<T>,
  options: AsyncDataOptions = {}
): AsyncDataResult<T> {
  const {
    immediate = true,
    retryCount = 3,
    retryDelay = 1000,
    cacheDuration = 0,
    deps = [],
    onSuccess,
    onError,
  } = options;

  // State
  const [state, setState] = useState<AsyncDataState<T>>({
    data: null,
    loading: immediate,
    error: null,
    isRefreshing: false,
    lastUpdated: null,
  });

  // Refs
  const mountedRef = useRef(true);
  const retryCountRef = useRef(0);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  // Generate cache key from fetcher
  const cacheKey = fetcher.toString();

  // Core fetch function
  const executeFetch = useCallback(
    async (isRefresh = false) => {
      // Check cache first
      if (cacheDuration > 0 && !isRefresh) {
        const cached = getCached<T>(cacheKey, cacheDuration);
        if (cached) {
          setState((prev) => ({
            ...prev,
            data: cached,
            loading: false,
            error: null,
          }));
          return;
        }
      }

      setState((prev) => ({
        ...prev,
        loading: !isRefresh,
        isRefreshing: isRefresh,
        error: null,
      }));

      try {
        const data = await fetcherRef.current();

        if (!mountedRef.current) return;

        // Cache result
        if (cacheDuration > 0) {
          setCache(cacheKey, data);
        }

        setState({
          data,
          loading: false,
          error: null,
          isRefreshing: false,
          lastUpdated: new Date(),
        });

        retryCountRef.current = 0;
        onSuccess?.(data);
      } catch (err) {
        if (!mountedRef.current) return;

        const error = err instanceof Error ? err : new Error(String(err));

        // Retry logic
        if (retryCountRef.current < retryCount) {
          retryCountRef.current++;
          await new Promise((resolve) =>
            setTimeout(resolve, retryDelay * retryCountRef.current)
          );
          if (mountedRef.current) {
            return executeFetch(isRefresh);
          }
        }

        setState((prev) => ({
          ...prev,
          loading: false,
          isRefreshing: false,
          error,
        }));

        onError?.(error);
      }
    },
    [cacheKey, cacheDuration, retryCount, retryDelay, onSuccess, onError]
  );

  // Public methods
  const fetch = useCallback(async () => {
    retryCountRef.current = 0;
    await executeFetch(false);
  }, [executeFetch]);

  const refresh = useCallback(async () => {
    retryCountRef.current = 0;
    await executeFetch(true);
  }, [executeFetch]);

  const retry = useCallback(async () => {
    retryCountRef.current = 0;
    await executeFetch(false);
  }, [executeFetch]);

  const reset = useCallback(() => {
    setState({
      data: null,
      loading: false,
      error: null,
      isRefreshing: false,
      lastUpdated: null,
    });
    retryCountRef.current = 0;
  }, []);

  const setData = useCallback((data: T | null) => {
    setState((prev) => ({
      ...prev,
      data,
      lastUpdated: data ? new Date() : null,
    }));
  }, []);

  // Initial fetch and dependency changes
  useEffect(() => {
    if (immediate) {
      executeFetch(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [immediate, ...deps]);

  // Cleanup
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  return {
    ...state,
    fetch,
    refresh,
    reset,
    setData,
    retry,
  };
}

// ============================================
// Utility: Combine Multiple Async Data
// ============================================

export function useAsyncDataAll<T extends readonly unknown[]>(
  fetchers: { [K in keyof T]: () => Promise<T[K]> },
  options: AsyncDataOptions = {}
): {
  data: { [K in keyof T]: T[K] | null };
  loading: boolean;
  error: Error | null;
  fetch: () => Promise<void>;
  refresh: () => Promise<void>;
} {
  const [state, setState] = useState<{
    data: { [K in keyof T]: T[K] | null };
    loading: boolean;
    error: Error | null;
  }>({
    data: fetchers.map(() => null) as { [K in keyof T]: T[K] | null },
    loading: options.immediate !== false,
    error: null,
  });

  const executeFetch = useCallback(
    async (isRefresh = false) => {
      setState((prev) => ({ ...prev, loading: !isRefresh, error: null }));

      try {
        const results = await Promise.all(fetchers.map((f) => f()));
        setState({
          data: results as { [K in keyof T]: T[K] | null },
          loading: false,
          error: null,
        });
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        setState((prev) => ({ ...prev, loading: false, error }));
      }
    },
    [fetchers]
  );

  useEffect(() => {
    if (options.immediate !== false) {
      executeFetch(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...(options.deps || [])]);

  return {
    ...state,
    fetch: () => executeFetch(false),
    refresh: () => executeFetch(true),
  };
}

export default useAsyncData;
