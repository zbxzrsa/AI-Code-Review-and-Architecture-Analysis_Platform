/**
 * Rate Limiter Hook
 *
 * Provides:
 * - Cooldown timer display
 * - Form submission blocking when rate limited
 * - Exponential backoff for retries
 * - 429 response handling
 */

import { useState, useEffect, useCallback, useRef } from "react";
import { rateLimiter, RateLimitConfig } from "../services/security";

/**
 * Rate limit status
 */
export interface RateLimitStatus {
  isLimited: boolean;
  remaining: number;
  resetTime: number | null;
  cooldownSeconds: number;
  canRetry: boolean;
}

/**
 * Exponential backoff configuration
 */
export interface BackoffConfig {
  initialDelayMs: number;
  maxDelayMs: number;
  multiplier: number;
  maxRetries: number;
}

const defaultBackoffConfig: BackoffConfig = {
  initialDelayMs: 1000,
  maxDelayMs: 60000,
  multiplier: 2,
  maxRetries: 5,
};

/**
 * Rate Limiter Hook
 *
 * @param key - Unique key for the rate limit (e.g., endpoint path)
 * @param config - Rate limit configuration
 */
export function useRateLimiter(key: string, config?: RateLimitConfig) {
  const [status, setStatus] = useState<RateLimitStatus>({
    isLimited: false,
    remaining: config?.maxRequests || 100,
    resetTime: null,
    cooldownSeconds: 0,
    canRetry: true,
  });

  const [retryCount, setRetryCount] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  /**
   * Update cooldown timer every second
   */
  useEffect(() => {
    if (status.resetTime && status.isLimited) {
      const resetTime = status.resetTime; // Capture for closure
      timerRef.current = setInterval(() => {
        const now = Date.now();
        const remaining = Math.max(0, Math.ceil((resetTime - now) / 1000));

        if (remaining <= 0) {
          setStatus((prev) => ({
            ...prev,
            isLimited: false,
            cooldownSeconds: 0,
            remaining: config?.maxRequests || 100,
            canRetry: true,
          }));
          if (timerRef.current) {
            clearInterval(timerRef.current);
          }
        } else {
          setStatus((prev) => ({
            ...prev,
            cooldownSeconds: remaining,
          }));
        }
      }, 1000);

      return () => {
        if (timerRef.current) {
          clearInterval(timerRef.current);
        }
      };
    }
  }, [status.resetTime, status.isLimited, config?.maxRequests]);

  /**
   * Check if action is rate limited
   */
  const checkLimit = useCallback((): boolean => {
    const limited = rateLimiter.shouldLimit(key, config);
    const remaining = rateLimiter.getRemaining(key, config);
    const resetTime = rateLimiter.getResetTime(key);

    if (limited) {
      setStatus({
        isLimited: true,
        remaining: 0,
        resetTime,
        cooldownSeconds: resetTime
          ? Math.ceil((resetTime - Date.now()) / 1000)
          : 0,
        canRetry: false,
      });
    } else {
      setStatus((prev) => ({
        ...prev,
        isLimited: false,
        remaining,
        canRetry: true,
      }));
    }

    return limited;
  }, [key, config]);

  /**
   * Handle 429 response from server
   */
  const handleRateLimitResponse = useCallback(
    (retryAfterSeconds?: number, resetTime?: number) => {
      const actualResetTime =
        resetTime ||
        (retryAfterSeconds
          ? Date.now() + retryAfterSeconds * 1000
          : Date.now() + 60000);

      setStatus({
        isLimited: true,
        remaining: 0,
        resetTime: actualResetTime,
        cooldownSeconds: retryAfterSeconds || 60,
        canRetry: false,
      });

      setRetryCount((prev) => prev + 1);
    },
    []
  );

  /**
   * Calculate exponential backoff delay
   */
  const getBackoffDelay = useCallback(
    (
      attempt: number,
      backoffConfig: BackoffConfig = defaultBackoffConfig
    ): number => {
      const delay =
        backoffConfig.initialDelayMs *
        Math.pow(backoffConfig.multiplier, attempt);

      // Add jitter (±10%)
      const jitter = delay * 0.1 * (Math.random() * 2 - 1);

      return Math.min(delay + jitter, backoffConfig.maxDelayMs);
    },
    []
  );

  /**
   * Execute with exponential backoff
   */
  const executeWithBackoff = useCallback(
    async <T>(
      fn: () => Promise<T>,
      backoffConfig: BackoffConfig = defaultBackoffConfig
    ): Promise<T> => {
      let lastError: Error | null = null;

      for (let attempt = 0; attempt < backoffConfig.maxRetries; attempt++) {
        // Check client-side rate limit
        if (checkLimit()) {
          throw new Error("Rate limited");
        }

        try {
          return await fn();
        } catch (error: any) {
          lastError = error;

          // Check if rate limited (429)
          if (error.response?.status === 429) {
            const retryAfter = parseInt(
              error.response.headers["retry-after"] || "60"
            );
            handleRateLimitResponse(retryAfter);

            // Wait for backoff delay
            const delay = getBackoffDelay(attempt, backoffConfig);
            await new Promise((resolve) => setTimeout(resolve, delay));
            continue;
          }

          // For other errors, don't retry
          throw error;
        }
      }

      throw lastError || new Error("Max retries exceeded");
    },
    [checkLimit, handleRateLimitResponse, getBackoffDelay]
  );

  /**
   * Reset the rate limiter
   */
  const reset = useCallback(() => {
    rateLimiter.clear(key);
    setRetryCount(0);
    setStatus({
      isLimited: false,
      remaining: config?.maxRequests || 100,
      resetTime: null,
      cooldownSeconds: 0,
      canRetry: true,
    });
  }, [key, config?.maxRequests]);

  /**
   * Format cooldown time for display
   */
  const formatCooldown = useCallback((seconds: number): string => {
    if (seconds < 60) {
      return `${seconds}s`;
    }
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (minutes < 60) {
      return secs > 0 ? `${minutes}m ${secs}s` : `${minutes}m`;
    }
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
  }, []);

  return {
    status,
    retryCount,
    checkLimit,
    handleRateLimitResponse,
    executeWithBackoff,
    getBackoffDelay,
    reset,
    formatCooldown,

    // Convenience properties
    isLimited: status.isLimited,
    cooldownText: formatCooldown(status.cooldownSeconds),
    canSubmit: !status.isLimited && status.canRetry,
  };
}

/**
 * Pre-configured rate limiters for common endpoints
 */
export const rateLimitConfigs = {
  login: { maxRequests: 5, windowMs: 15 * 60 * 1000 }, // 5 per 15 min
  passwordChange: { maxRequests: 3, windowMs: 24 * 60 * 60 * 1000 }, // 3 per 24h
  apiKeyGeneration: { maxRequests: 10, windowMs: 60 * 60 * 1000 }, // 10 per hour
  generalApi: { maxRequests: 100, windowMs: 60 * 1000 }, // 100 per min
  register: { maxRequests: 3, windowMs: 60 * 1000 }, // 3 per min
  passwordReset: { maxRequests: 3, windowMs: 5 * 60 * 1000 }, // 3 per 5 min
  twoFactor: { maxRequests: 5, windowMs: 60 * 1000 }, // 5 per min
};

export default useRateLimiter;
