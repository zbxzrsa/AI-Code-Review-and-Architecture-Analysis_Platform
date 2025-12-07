/**
 * Enhanced API Service
 *
 * Improved API mechanisms with:
 * - Automatic retry with exponential backoff
 * - Request deduplication
 * - Response caching
 * - Request cancellation
 * - Error classification
 * - Offline queue
 * - Rate limiting awareness
 */

import axios, {
  AxiosInstance,
  AxiosRequestConfig,
  AxiosResponse,
  AxiosError,
  InternalAxiosRequestConfig,
} from "axios";

// ============================================
// Types
// ============================================

export interface ApiError {
  code: string;
  message: string;
  status: number;
  details?: Record<string, any>;
  retryable: boolean;
  timestamp: Date;
}

export interface RequestOptions extends AxiosRequestConfig {
  retry?: boolean;
  maxRetries?: number;
  retryDelay?: number;
  cache?: boolean;
  cacheTTL?: number;
  dedupe?: boolean;
  priority?: "low" | "normal" | "high";
  offline?: boolean;
}

interface CacheEntry<T = unknown> {
  data: T;
  timestamp: number;
  ttl: number;
}

interface PendingRequest<T = unknown> {
  id: string;
  config: RequestOptions;
  resolve: (value: T) => void;
  reject: (reason: unknown) => void;
  retryCount: number;
  timestamp: number;
}

// ============================================
// Error Classification
// ============================================

export const ErrorCodes = {
  NETWORK_ERROR: "NETWORK_ERROR",
  TIMEOUT: "TIMEOUT",
  UNAUTHORIZED: "UNAUTHORIZED",
  FORBIDDEN: "FORBIDDEN",
  NOT_FOUND: "NOT_FOUND",
  VALIDATION_ERROR: "VALIDATION_ERROR",
  RATE_LIMITED: "RATE_LIMITED",
  SERVER_ERROR: "SERVER_ERROR",
  SERVICE_UNAVAILABLE: "SERVICE_UNAVAILABLE",
  UNKNOWN: "UNKNOWN",
} as const;

function classifyError(error: AxiosError): ApiError {
  const status = error.response?.status || 0;
  const data = error.response?.data as any;

  let code: string = ErrorCodes.UNKNOWN;
  let message = "An unexpected error occurred";
  let retryable = false;

  if (error.response) {
    // Response error - use status code
    switch (status) {
      case 401:
        code = ErrorCodes.UNAUTHORIZED;
        message = "Authentication required";
        break;
      case 403:
        code = ErrorCodes.FORBIDDEN;
        message = "Access denied";
        break;
      case 404:
        code = ErrorCodes.NOT_FOUND;
        message = "Resource not found";
        break;
      case 422:
        code = ErrorCodes.VALIDATION_ERROR;
        message = data?.message || "Validation failed";
        break;
      case 429:
        code = ErrorCodes.RATE_LIMITED;
        message = "Too many requests. Please slow down.";
        retryable = true;
        break;
      case 500:
      case 502:
      case 504:
        code = ErrorCodes.SERVER_ERROR;
        message = "Server error. Please try again later.";
        retryable = true;
        break;
      case 503:
        code = ErrorCodes.SERVICE_UNAVAILABLE;
        message = "Service temporarily unavailable";
        retryable = true;
        break;
    }
  } else if (error.code === "ECONNABORTED") {
    code = ErrorCodes.TIMEOUT;
    message = "Request timed out";
    retryable = true;
  } else {
    code = ErrorCodes.NETWORK_ERROR;
    message = "Network error. Please check your connection.";
    retryable = true;
  }

  return {
    code,
    message: data?.message || message,
    status,
    details: data?.details,
    retryable,
    timestamp: new Date(),
  };
}

// ============================================
// Enhanced API Class
// ============================================

class EnhancedApi {
  private readonly client: AxiosInstance;
  private readonly cache: Map<string, CacheEntry> = new Map();
  private readonly pendingRequests: Map<string, Promise<unknown>> = new Map();
  private offlineQueue: PendingRequest[] = [];
  private isOnline: boolean = navigator.onLine;
  private readonly retryDelays: number[] = [1000, 2000, 4000, 8000, 16000]; // Exponential backoff

  constructor(baseURL: string = "/api") {
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        "Content-Type": "application/json",
      },
    });

    this.setupInterceptors();
    this.setupOnlineListener();
  }

  // ============================================
  // Interceptors
  // ============================================

  private setupInterceptors(): void {
    // Request interceptor
    this.client.interceptors.request.use(
      (config: InternalAxiosRequestConfig) => {
        // Add request ID for tracking
        config.headers["X-Request-ID"] = this.generateRequestId();

        // Add timestamp
        config.headers["X-Request-Time"] = Date.now().toString();

        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response: AxiosResponse) => {
        // Log response time
        const requestTime = response.config.headers["X-Request-Time"];
        if (requestTime) {
          const duration = Date.now() - Number.parseInt(requestTime as string);
          console.debug(
            `[API] ${response.config.method?.toUpperCase()} ${response.config.url} - ${duration}ms`
          );
        }
        return response;
      },
      (error: AxiosError) => {
        const apiError = classifyError(error);
        console.error(`[API Error] ${apiError.code}: ${apiError.message}`);
        return Promise.reject(apiError);
      }
    );
  }

  // ============================================
  // Online/Offline Handling
  // ============================================

  private setupOnlineListener(): void {
    globalThis.addEventListener("online", () => {
      this.isOnline = true;
      this.processOfflineQueue();
    });

    globalThis.addEventListener("offline", () => {
      this.isOnline = false;
    });
  }

  private async processOfflineQueue(): Promise<void> {
    // Debug: Processing offline queue
    if (import.meta.env.DEV) {
      console.debug(`[API] Processing ${this.offlineQueue.length} queued requests`);
    }

    const queue = [...this.offlineQueue];
    this.offlineQueue = [];

    for (const request of queue) {
      try {
        const response = await this.request(request.config);
        request.resolve(response);
      } catch (error) {
        request.reject(error);
      }
    }
  }

  // ============================================
  // Caching
  // ============================================

  private getCacheKey(config: RequestOptions): string {
    return `${config.method || "GET"}:${config.url}:${JSON.stringify(config.params || {})}`;
  }

  private getFromCache(key: string): any | null {
    const entry = this.cache.get(key);
    if (!entry) return null;

    const isExpired = Date.now() - entry.timestamp > entry.ttl;
    if (isExpired) {
      this.cache.delete(key);
      return null;
    }

    return entry.data;
  }

  private setCache(key: string, data: any, ttl: number): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl,
    });
  }

  public clearCache(pattern?: string): void {
    if (pattern) {
      for (const key of this.cache.keys()) {
        if (key.includes(pattern)) {
          this.cache.delete(key);
        }
      }
    } else {
      this.cache.clear();
    }
  }

  // ============================================
  // Request Deduplication
  // ============================================

  private getDedupeKey(config: RequestOptions): string {
    return this.getCacheKey(config);
  }

  // ============================================
  // Retry Logic
  // ============================================

  private async retryRequest(
    config: RequestOptions,
    error: ApiError,
    retryCount: number
  ): Promise<any> {
    const maxRetries = config.maxRetries ?? 3;

    if (!error.retryable || retryCount >= maxRetries) {
      throw error;
    }

    const delay = config.retryDelay ?? this.retryDelays[retryCount] ?? 16000;

    // Add jitter to prevent thundering herd
    const jitter = Math.random() * 1000;

    // Debug: Retry logging
    if (import.meta.env.DEV) {
      console.debug(
        `[API] Retrying request (${retryCount + 1}/${maxRetries}) in ${delay + jitter}ms`
      );
    }

    await this.sleep(delay + jitter);

    return this.executeRequest(config, retryCount + 1);
  }

  // ============================================
  // Core Request Method
  // ============================================

  private async executeRequest(config: RequestOptions, retryCount: number = 0): Promise<any> {
    try {
      const response = await this.client.request(config);
      return response.data;
    } catch (error) {
      if (config.retry !== false && (error as ApiError).retryable) {
        return this.retryRequest(config, error as ApiError, retryCount);
      }
      throw error;
    }
  }

  public async request<T = any>(config: RequestOptions): Promise<T> {
    // Check offline mode
    if (!this.isOnline && config.offline !== false) {
      return new Promise((resolve, reject) => {
        this.offlineQueue.push({
          id: this.generateRequestId(),
          config,
          resolve: resolve as (value: unknown) => void,
          reject,
          retryCount: 0,
          timestamp: Date.now(),
        });
      });
    }

    // Check cache for GET requests
    if (config.method?.toUpperCase() === "GET" && config.cache !== false) {
      const cacheKey = this.getCacheKey(config);
      const cached = this.getFromCache(cacheKey);
      if (cached) {
        console.debug(`[API] Cache hit: ${cacheKey}`);
        return cached;
      }
    }

    // Deduplicate concurrent identical requests
    if (config.dedupe !== false && config.method?.toUpperCase() === "GET") {
      const dedupeKey = this.getDedupeKey(config);
      const pending = this.pendingRequests.get(dedupeKey);
      if (pending) {
        console.debug(`[API] Deduplicating request: ${dedupeKey}`);
        return pending;
      }

      const promise = this.executeRequest(config);
      this.pendingRequests.set(dedupeKey, promise);

      try {
        const result = await promise;

        // Cache successful GET requests
        if (config.cache !== false) {
          const cacheKey = this.getCacheKey(config);
          this.setCache(cacheKey, result, config.cacheTTL ?? 60000);
        }

        return result;
      } finally {
        this.pendingRequests.delete(dedupeKey);
      }
    }

    return this.executeRequest(config);
  }

  // ============================================
  // Convenience Methods
  // ============================================

  public async get<T = any>(url: string, options?: RequestOptions): Promise<T> {
    return this.request<T>({ ...options, method: "GET", url });
  }

  public async post<T = any>(url: string, data?: any, options?: RequestOptions): Promise<T> {
    return this.request<T>({ ...options, method: "POST", url, data });
  }

  public async put<T = any>(url: string, data?: any, options?: RequestOptions): Promise<T> {
    return this.request<T>({ ...options, method: "PUT", url, data });
  }

  public async patch<T = any>(url: string, data?: any, options?: RequestOptions): Promise<T> {
    return this.request<T>({ ...options, method: "PATCH", url, data });
  }

  public async delete<T = any>(url: string, options?: RequestOptions): Promise<T> {
    return this.request<T>({ ...options, method: "DELETE", url });
  }

  // ============================================
  // Utility Methods
  // ============================================

  private generateRequestId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  public getQueueLength(): number {
    return this.offlineQueue.length;
  }

  public getCacheSize(): number {
    return this.cache.size;
  }

  public isNetworkOnline(): boolean {
    return this.isOnline;
  }
}

// ============================================
// Export Singleton Instance
// ============================================

export const enhancedApi = new EnhancedApi();

export default enhancedApi;
