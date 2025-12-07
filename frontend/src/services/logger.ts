/**
 * Standardized Logging Service (TD-004)
 *
 * Winston-style logging with:
 * - Standardized log levels
 * - Structured JSON logging
 * - Request tracking ID
 * - Hierarchical query support
 * - Context enrichment
 *
 * Log Levels (in order of severity):
 * - error: Critical errors requiring immediate attention
 * - warn: Warning conditions that should be investigated
 * - info: Informational messages about normal operation
 * - http: HTTP request/response logging
 * - debug: Debug information for development
 * - verbose: Detailed trace information
 *
 * Usage:
 *   import { logger } from '@/services/logger';
 *
 *   logger.info('User logged in', { userId: '123' });
 *   logger.error('Failed to load data', { error, endpoint: '/api/users' });
 *   logger.withContext({ requestId: 'req-123' }).info('Processing request');
 */

import { v4 as uuidv4 } from "uuid";

/**
 * Log levels in order of severity (lower number = higher severity)
 */
export enum LogLevel {
  ERROR = 0,
  WARN = 1,
  INFO = 2,
  HTTP = 3,
  DEBUG = 4,
  VERBOSE = 5,
}

/**
 * Log level names
 */
export const LOG_LEVEL_NAMES: Record<LogLevel, string> = {
  [LogLevel.ERROR]: "error",
  [LogLevel.WARN]: "warn",
  [LogLevel.INFO]: "info",
  [LogLevel.HTTP]: "http",
  [LogLevel.DEBUG]: "debug",
  [LogLevel.VERBOSE]: "verbose",
};

/**
 * Parse log level from string
 */
export function parseLogLevel(level: string): LogLevel {
  const normalizedLevel = level.toLowerCase();
  const entry = Object.entries(LOG_LEVEL_NAMES).find(([, name]) => name === normalizedLevel);
  return entry ? (Number.parseInt(entry[0], 10) as LogLevel) : LogLevel.INFO;
}

/**
 * Log entry structure
 */
export interface LogEntry {
  /** Timestamp in ISO format */
  timestamp: string;
  /** Log level */
  level: string;
  /** Log message */
  message: string;
  /** Additional context */
  context?: Record<string, unknown>;
  /** Request tracking ID */
  requestId?: string;
  /** Source component/module */
  source?: string;
  /** User ID if authenticated */
  userId?: string;
  /** Session ID */
  sessionId?: string;
  /** Error stack trace */
  stack?: string;
  /** Environment */
  env?: string;
}

/**
 * Logger configuration
 */
export interface LoggerConfig {
  /** Minimum log level to output */
  level: LogLevel;
  /** Whether to output to console */
  console: boolean;
  /** Whether to output JSON format */
  json: boolean;
  /** Whether to send logs to server */
  remote: boolean;
  /** Remote logging endpoint */
  remoteEndpoint?: string;
  /** Default context for all logs */
  defaultContext?: Record<string, unknown>;
  /** Include timestamps */
  timestamps: boolean;
  /** Include stack traces for errors */
  stackTraces: boolean;
}

/**
 * Default configuration
 */
const DEFAULT_CONFIG: LoggerConfig = {
  level: import.meta.env.MODE === "production" ? LogLevel.WARN : LogLevel.DEBUG,
  console: true,
  json: import.meta.env.MODE === "production",
  remote: import.meta.env.MODE === "production",
  remoteEndpoint: "/api/logs",
  timestamps: true,
  stackTraces: true,
};

/**
 * Log transport interface
 */
interface LogTransport {
  log(entry: LogEntry): void | Promise<void>;
}

/**
 * Console transport
 */
class ConsoleTransport implements LogTransport {
  private readonly useJson: boolean;

  constructor(useJson: boolean = false) {
    this.useJson = useJson;
  }

  log(entry: LogEntry): void {
    const { level, message, context, timestamp } = entry;

    if (this.useJson) {
      console.log(JSON.stringify(entry));
      return;
    }

    const prefix = `[${timestamp}] [${level.toUpperCase()}]`;
    const contextStr = context ? ` ${JSON.stringify(context)}` : "";
    const fullMessage = `${prefix} ${message}${contextStr}`;

    switch (level) {
      case "error":
        console.error(fullMessage);
        if (entry.stack) console.error(entry.stack);
        break;
      case "warn":
        console.warn(fullMessage);
        break;
      case "info":
        console.info(fullMessage);
        break;
      case "debug":
      case "verbose":
        console.debug(fullMessage);
        break;
      default:
        console.log(fullMessage);
    }
  }
}

/**
 * Remote transport - sends logs to server
 */
class RemoteTransport implements LogTransport {
  private readonly endpoint: string;
  private buffer: LogEntry[] = [];
  private flushTimeout: ReturnType<typeof setTimeout> | null = null;
  private readonly bufferSize = 10;
  private readonly flushInterval = 5000;

  constructor(endpoint: string) {
    this.endpoint = endpoint;
  }

  async log(entry: LogEntry): Promise<void> {
    this.buffer.push(entry);

    if (this.buffer.length >= this.bufferSize) {
      await this.flush();
    } else if (!this.flushTimeout) {
      this.flushTimeout = setTimeout(() => this.flush(), this.flushInterval);
    }
  }

  private async flush(): Promise<void> {
    if (this.buffer.length === 0) return;

    const entries = [...this.buffer];
    this.buffer = [];

    if (this.flushTimeout) {
      clearTimeout(this.flushTimeout);
      this.flushTimeout = null;
    }

    try {
      await fetch(this.endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ logs: entries }),
        keepalive: true,
      });
    } catch (error) {
      // Restore entries on failure (with limit)
      this.buffer = [...entries.slice(-50), ...this.buffer].slice(-100);
      console.warn("Failed to send logs to server:", error);
    }
  }
}

/**
 * Main Logger class
 */
class Logger {
  private config: LoggerConfig;
  private transports: LogTransport[] = [];
  private defaultContext: Record<string, unknown> = {};
  private requestId: string | null = null;
  private sessionId: string | null = null;
  private userId: string | null = null;

  constructor(config: Partial<LoggerConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.initTransports();
  }

  private initTransports(): void {
    if (this.config.console) {
      this.transports.push(new ConsoleTransport(this.config.json));
    }

    if (this.config.remote && this.config.remoteEndpoint) {
      this.transports.push(new RemoteTransport(this.config.remoteEndpoint));
    }
  }

  /**
   * Set the minimum log level
   */
  setLevel(level: LogLevel | string): void {
    this.config.level = typeof level === "string" ? parseLogLevel(level) : level;
  }

  /**
   * Set default context for all logs
   */
  setDefaultContext(context: Record<string, unknown>): void {
    this.defaultContext = context;
  }

  /**
   * Set request tracking ID
   */
  setRequestId(requestId: string): void {
    this.requestId = requestId;
  }

  /**
   * Set session ID
   */
  setSessionId(sessionId: string): void {
    this.sessionId = sessionId;
  }

  /**
   * Set user ID
   */
  setUserId(userId: string | null): void {
    this.userId = userId;
  }

  /**
   * Generate a new request ID
   */
  generateRequestId(): string {
    this.requestId = uuidv4();
    return this.requestId;
  }

  /**
   * Create a child logger with additional context
   */
  withContext(context: Record<string, unknown>): ContextLogger {
    return new ContextLogger(this, context);
  }

  /**
   * Create a child logger for a specific source/module
   */
  forSource(source: string): ContextLogger {
    return new ContextLogger(this, { source });
  }

  /**
   * Check if a log level is enabled
   */
  isLevelEnabled(level: LogLevel): boolean {
    return level <= this.config.level;
  }

  /**
   * Core logging method
   */
  private log(
    level: LogLevel,
    message: string,
    context?: Record<string, unknown>,
    error?: Error
  ): void {
    if (!this.isLevelEnabled(level)) return;

    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level: LOG_LEVEL_NAMES[level],
      message,
      context: { ...this.defaultContext, ...context },
      env: import.meta.env.MODE,
    };

    if (this.requestId) entry.requestId = this.requestId;
    if (this.sessionId) entry.sessionId = this.sessionId;
    if (this.userId) entry.userId = this.userId;
    if (error && this.config.stackTraces) entry.stack = error.stack;

    for (const transport of this.transports) {
      try {
        transport.log(entry);
      } catch (e) {
        console.error("Transport error:", e);
      }
    }
  }

  // Log level methods
  error(message: string, context?: Record<string, unknown> | Error): void {
    if (context instanceof Error) {
      this.log(LogLevel.ERROR, message, { error: context.message }, context);
    } else {
      this.log(LogLevel.ERROR, message, context);
    }
  }

  warn(message: string, context?: Record<string, unknown>): void {
    this.log(LogLevel.WARN, message, context);
  }

  info(message: string, context?: Record<string, unknown>): void {
    this.log(LogLevel.INFO, message, context);
  }

  http(message: string, context?: Record<string, unknown>): void {
    this.log(LogLevel.HTTP, message, context);
  }

  debug(message: string, context?: Record<string, unknown>): void {
    this.log(LogLevel.DEBUG, message, context);
  }

  verbose(message: string, context?: Record<string, unknown>): void {
    this.log(LogLevel.VERBOSE, message, context);
  }

  /**
   * Log HTTP request
   */
  httpRequest(
    method: string,
    url: string,
    status?: number,
    duration?: number,
    context?: Record<string, unknown>
  ): void {
    this.http(`${method} ${url}`, {
      method,
      url,
      status,
      duration,
      ...context,
    });
  }

  /**
   * Log HTTP response
   */
  httpResponse(
    method: string,
    url: string,
    status: number,
    duration: number,
    context?: Record<string, unknown>
  ): void {
    const level = status >= 500 ? LogLevel.ERROR : status >= 400 ? LogLevel.WARN : LogLevel.HTTP;

    this.log(level, `${method} ${url} ${status} ${duration}ms`, {
      method,
      url,
      status,
      duration,
      ...context,
    });
  }

  /**
   * Time an operation
   */
  time<T>(label: string, fn: () => T | Promise<T>): T | Promise<T> {
    const start = performance.now();

    try {
      const result = fn();

      if (result instanceof Promise) {
        return result
          .then((value) => {
            const duration = performance.now() - start;
            this.debug(`${label} completed`, {
              duration: `${duration.toFixed(2)}ms`,
            });
            return value;
          })
          .catch((error) => {
            const duration = performance.now() - start;
            this.error(`${label} failed`, {
              duration: `${duration.toFixed(2)}ms`,
              error: error.message,
            });
            throw error;
          });
      }

      const duration = performance.now() - start;
      this.debug(`${label} completed`, {
        duration: `${duration.toFixed(2)}ms`,
      });
      return result;
    } catch (error) {
      const duration = performance.now() - start;
      this.error(`${label} failed`, {
        duration: `${duration.toFixed(2)}ms`,
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }
}

/**
 * Context logger - logger with additional context
 */
class ContextLogger {
  private parent: Logger;
  private context: Record<string, unknown>;

  constructor(parent: Logger, context: Record<string, unknown>) {
    this.parent = parent;
    this.context = context;
  }

  error(message: string, context?: Record<string, unknown>): void {
    this.parent.error(message, { ...this.context, ...context });
  }

  warn(message: string, context?: Record<string, unknown>): void {
    this.parent.warn(message, { ...this.context, ...context });
  }

  info(message: string, context?: Record<string, unknown>): void {
    this.parent.info(message, { ...this.context, ...context });
  }

  http(message: string, context?: Record<string, unknown>): void {
    this.parent.http(message, { ...this.context, ...context });
  }

  debug(message: string, context?: Record<string, unknown>): void {
    this.parent.debug(message, { ...this.context, ...context });
  }

  verbose(message: string, context?: Record<string, unknown>): void {
    this.parent.verbose(message, { ...this.context, ...context });
  }
}

// Create and export default logger instance
export const logger = new Logger();

// Export for creating custom loggers
export { Logger, ContextLogger, ConsoleTransport, RemoteTransport };

// Convenience exports
export const createLogger = (config?: Partial<LoggerConfig>) => new Logger(config);
