/**
 * Error Logging Service
 *
 * Centralized error handling and logging with:
 * - Error categorization (client, server, network)
 * - Logging to external providers
 * - Error recovery strategies
 * - User-friendly error messages
 */

/**
 * Error categories for classification
 */
export enum ErrorCategory {
  CLIENT = "client",
  SERVER = "server",
  NETWORK = "network",
  AUTHENTICATION = "authentication",
  AUTHORIZATION = "authorization",
  VALIDATION = "validation",
  UNKNOWN = "unknown",
}

/**
 * Error severity levels
 */
export enum ErrorSeverity {
  LOW = "low",
  MEDIUM = "medium",
  HIGH = "high",
  CRITICAL = "critical",
}

/**
 * Structured error log entry
 */
export interface ErrorLogEntry {
  id: string;
  timestamp: string;
  category: ErrorCategory;
  severity: ErrorSeverity;
  message: string;
  stack?: string;
  componentStack?: string;
  url: string;
  userAgent: string;
  userId?: string;
  sessionId?: string;
  metadata?: Record<string, unknown>;
  recovered?: boolean;
}

/**
 * Recovery strategy definition
 */
export interface RecoveryStrategy {
  type: "retry" | "fallback" | "redirect" | "refresh" | "ignore";
  maxRetries?: number;
  fallbackValue?: unknown;
  redirectUrl?: string;
  delayMs?: number;
}

/**
 * User-friendly error messages
 */
const userFriendlyMessages: Record<ErrorCategory, string> = {
  [ErrorCategory.CLIENT]: "Something went wrong. Please try again.",
  [ErrorCategory.SERVER]: "Server error. Please try again later.",
  [ErrorCategory.NETWORK]: "Network error. Please check your connection.",
  [ErrorCategory.AUTHENTICATION]: "Please log in to continue.",
  [ErrorCategory.AUTHORIZATION]:
    "You don't have permission to perform this action.",
  [ErrorCategory.VALIDATION]: "Please check your input and try again.",
  [ErrorCategory.UNKNOWN]: "An unexpected error occurred.",
};

/**
 * Generate unique error ID
 */
const generateErrorId = (): string => {
  return `err_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
};

/**
 * Categorize error based on its properties
 */
const categorizeError = (error: Error | unknown): ErrorCategory => {
  if (
    error instanceof TypeError ||
    error instanceof SyntaxError ||
    error instanceof ReferenceError
  ) {
    return ErrorCategory.CLIENT;
  }

  if (error instanceof Error) {
    const message = error.message.toLowerCase();

    if (
      message.includes("network") ||
      message.includes("fetch") ||
      message.includes("connection")
    ) {
      return ErrorCategory.NETWORK;
    }

    if (
      message.includes("401") ||
      message.includes("unauthorized") ||
      message.includes("unauthenticated")
    ) {
      return ErrorCategory.AUTHENTICATION;
    }

    if (
      message.includes("403") ||
      message.includes("forbidden") ||
      message.includes("permission")
    ) {
      return ErrorCategory.AUTHORIZATION;
    }

    if (
      message.includes("400") ||
      message.includes("validation") ||
      message.includes("invalid")
    ) {
      return ErrorCategory.VALIDATION;
    }

    if (message.includes("500") || message.includes("server")) {
      return ErrorCategory.SERVER;
    }
  }

  return ErrorCategory.UNKNOWN;
};

/**
 * Determine error severity
 */
const determineSeverity = (
  category: ErrorCategory,
  _error: Error | unknown
): ErrorSeverity => {
  switch (category) {
    case ErrorCategory.AUTHENTICATION:
    case ErrorCategory.AUTHORIZATION:
      return ErrorSeverity.MEDIUM;
    case ErrorCategory.SERVER:
      return ErrorSeverity.HIGH;
    case ErrorCategory.CLIENT:
      return ErrorSeverity.HIGH;
    case ErrorCategory.NETWORK:
      return ErrorSeverity.MEDIUM;
    case ErrorCategory.VALIDATION:
      return ErrorSeverity.LOW;
    default:
      return ErrorSeverity.MEDIUM;
  }
};

/**
 * Error Logging Service Class
 */
class ErrorLoggingService {
  private readonly isProduction: boolean;
  private readonly logEndpoint: string;
  private errorQueue: ErrorLogEntry[] = [];
  private readonly flushInterval: number = 5000;
  private readonly maxQueueSize: number = 50;
  private flushTimer: ReturnType<typeof setTimeout> | null = null;

  constructor() {
    this.isProduction = process.env.NODE_ENV === "production";
    this.logEndpoint = "/api/errors";

    // Start flush timer
    this.startFlushTimer();

    // Flush on page unload
    if (typeof globalThis.window !== "undefined") {
      globalThis.addEventListener("beforeunload", () => this.flush());
    }
  }

  /**
   * Start periodic flush timer
   */
  private startFlushTimer(): void {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
    }
    this.flushTimer = setInterval(() => this.flush(), this.flushInterval);
  }

  /**
   * Create error log entry
   */
  private createLogEntry(
    error: Error | unknown,
    category: ErrorCategory,
    metadata?: Record<string, unknown>,
    componentStack?: string
  ): ErrorLogEntry {
    const errorObj = error instanceof Error ? error : new Error(String(error));

    return {
      id: generateErrorId(),
      timestamp: new Date().toISOString(),
      category,
      severity: determineSeverity(category, error),
      message: errorObj.message,
      stack: errorObj.stack,
      componentStack,
      url: typeof window !== "undefined" ? window.location.href : "",
      userAgent: typeof navigator !== "undefined" ? navigator.userAgent : "",
      userId: this.getCurrentUserId(),
      sessionId: this.getSessionId(),
      metadata,
    };
  }

  /**
   * Get current user ID from auth store
   */
  private getCurrentUserId(): string | undefined {
    try {
      const authData = localStorage.getItem("auth-storage");
      if (authData) {
        const parsed = JSON.parse(authData);
        return parsed?.state?.user?.id;
      }
    } catch {
      // Ignore
    }
    return undefined;
  }

  /**
   * Get or create session ID
   */
  private getSessionId(): string {
    let sessionId = sessionStorage.getItem("error-session-id");
    if (!sessionId) {
      sessionId = `session_${Date.now()}_${Math.random()
        .toString(36)
        .substring(2, 9)}`;
      sessionStorage.setItem("error-session-id", sessionId);
    }
    return sessionId;
  }

  /**
   * Log error to console and queue
   */
  public log(
    error: Error | unknown,
    category?: ErrorCategory,
    metadata?: Record<string, unknown>,
    componentStack?: string
  ): ErrorLogEntry {
    const resolvedCategory = category || categorizeError(error);
    const entry = this.createLogEntry(
      error,
      resolvedCategory,
      metadata,
      componentStack
    );

    // Always log to console in development
    if (!this.isProduction) {
      /* eslint-disable no-console */
      console.group(`🚨 Error [${entry.category}] - ${entry.id}`);
      console.error("Message:", entry.message);
      console.error("Stack:", entry.stack);
      if (entry.componentStack) {
        console.error("Component Stack:", entry.componentStack);
      }
      console.error("Metadata:", entry.metadata);
      console.groupEnd();
      /* eslint-enable no-console */
    }

    // Add to queue
    this.errorQueue.push(entry);

    // Flush if queue is full
    if (this.errorQueue.length >= this.maxQueueSize) {
      this.flush();
    }

    return entry;
  }

  /**
   * Log error from React Error Boundary
   */
  public logComponentError(
    error: Error,
    errorInfo: React.ErrorInfo,
    metadata?: Record<string, unknown>
  ): ErrorLogEntry {
    return this.log(
      error,
      ErrorCategory.CLIENT,
      { ...metadata, errorBoundary: true },
      errorInfo.componentStack || undefined
    );
  }

  /**
   * Log network/API error
   */
  public logNetworkError(
    error: Error | unknown,
    endpoint: string,
    method: string,
    statusCode?: number
  ): ErrorLogEntry {
    const category =
      statusCode && statusCode >= 500
        ? ErrorCategory.SERVER
        : statusCode === 401
        ? ErrorCategory.AUTHENTICATION
        : statusCode === 403
        ? ErrorCategory.AUTHORIZATION
        : ErrorCategory.NETWORK;

    return this.log(error, category, {
      endpoint,
      method,
      statusCode,
    });
  }

  /**
   * Flush error queue to backend
   */
  public async flush(): Promise<void> {
    if (this.errorQueue.length === 0) return;

    const errors = [...this.errorQueue];
    this.errorQueue = [];

    if (this.isProduction) {
      try {
        await fetch(this.logEndpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ errors }),
          keepalive: true,
        });
      } catch {
        // Re-add to queue if send fails
        this.errorQueue.unshift(...errors);
      }
    }
  }

  /**
   * Get user-friendly error message
   */
  public getUserMessage(
    error: Error | unknown,
    category?: ErrorCategory
  ): string {
    const resolvedCategory = category || categorizeError(error);
    return userFriendlyMessages[resolvedCategory];
  }

  /**
   * Get recovery strategy for error
   */
  public getRecoveryStrategy(
    error: Error | unknown,
    category?: ErrorCategory
  ): RecoveryStrategy {
    const resolvedCategory = category || categorizeError(error);

    switch (resolvedCategory) {
      case ErrorCategory.NETWORK:
        return { type: "retry", maxRetries: 3, delayMs: 1000 };
      case ErrorCategory.AUTHENTICATION:
        return { type: "redirect", redirectUrl: "/login" };
      case ErrorCategory.AUTHORIZATION:
        return { type: "redirect", redirectUrl: "/dashboard" };
      case ErrorCategory.SERVER:
        return { type: "retry", maxRetries: 2, delayMs: 2000 };
      case ErrorCategory.VALIDATION:
        return { type: "ignore" };
      default:
        return { type: "refresh" };
    }
  }

  /**
   * Execute retry recovery
   */
  private async executeRetry(
    strategy: RecoveryStrategy,
    retryFn?: () => Promise<unknown>
  ): Promise<{ success: boolean; result?: unknown }> {
    if (!retryFn || !strategy.maxRetries) {
      return { success: false };
    }

    for (let i = 0; i < strategy.maxRetries; i++) {
      try {
        await new Promise((resolve) =>
          setTimeout(resolve, strategy.delayMs || 1000)
        );
        const result = await retryFn();
        return { success: true, result };
      } catch {
        if (i === strategy.maxRetries - 1) {
          return { success: false };
        }
      }
    }
    return { success: false };
  }

  /**
   * Execute redirect recovery
   */
  private executeRedirect(strategy: RecoveryStrategy): {
    success: boolean;
    result?: unknown;
  } {
    if (strategy.redirectUrl && typeof window !== "undefined") {
      window.location.href = strategy.redirectUrl;
      return { success: true };
    }
    return { success: false };
  }

  /**
   * Execute refresh recovery
   */
  private executeRefresh(): { success: boolean; result?: unknown } {
    if (typeof window !== "undefined") {
      window.location.reload();
      return { success: true };
    }
    return { success: false };
  }

  /**
   * Execute recovery strategy
   */
  public async executeRecovery(
    strategy: RecoveryStrategy,
    retryFn?: () => Promise<unknown>
  ): Promise<{ success: boolean; result?: unknown }> {
    switch (strategy.type) {
      case "retry":
        return this.executeRetry(strategy, retryFn);
      case "redirect":
        return this.executeRedirect(strategy);
      case "refresh":
        return this.executeRefresh();
      case "fallback":
        return { success: true, result: strategy.fallbackValue };
      case "ignore":
      default:
        return { success: true };
    }
  }
}

// Singleton instance
export const errorLoggingService = new ErrorLoggingService();

/**
 * React hook for error logging
 */
export const useErrorLogging = () => {
  return {
    log: errorLoggingService.log.bind(errorLoggingService),
    logNetworkError:
      errorLoggingService.logNetworkError.bind(errorLoggingService),
    getUserMessage:
      errorLoggingService.getUserMessage.bind(errorLoggingService),
    getRecoveryStrategy:
      errorLoggingService.getRecoveryStrategy.bind(errorLoggingService),
    executeRecovery:
      errorLoggingService.executeRecovery.bind(errorLoggingService),
  };
};

export default errorLoggingService;
