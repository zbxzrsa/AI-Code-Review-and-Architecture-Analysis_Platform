/**
 * Frontend Configuration Service (TD-002)
 *
 * Centralizes all configuration with:
 * - Environment variable validation
 * - Type-safe access
 * - Default values
 * - Multi-environment support
 */

/**
 * Environment types
 */
type Environment = "development" | "staging" | "production" | "test";

/**
 * Configuration validation error
 */
class ConfigValidationError extends Error {
  constructor(key: string, message: string) {
    super(`Configuration error for ${key}: ${message}`);
    this.name = "ConfigValidationError";
  }
}

/**
 * Get environment variable with validation
 */
function getEnv(key: string, defaultValue?: string, required = false): string {
  const value = import.meta.env[key] ?? defaultValue;

  if (required && !value) {
    throw new ConfigValidationError(key, "Required but not set");
  }

  return value ?? "";
}

/**
 * Get boolean environment variable
 */
function getEnvBool(key: string, defaultValue = false): boolean {
  const value = import.meta.env[key];
  if (value === undefined) return defaultValue;
  return value === "true" || value === "1";
}

/**
 * Get integer environment variable
 */
function getEnvInt(key: string, defaultValue = 0): number {
  const value = import.meta.env[key];
  if (value === undefined) return defaultValue;
  const parsed = Number.parseInt(value, 10);
  if (Number.isNaN(parsed)) {
    throw new ConfigValidationError(key, `Invalid integer: ${value}`);
  }
  return parsed;
}

/**
 * API Configuration
 */
export const apiConfig = {
  /** Base URL for API requests */
  baseUrl: getEnv("VITE_API_URL", "/api"),

  /** WebSocket URL */
  wsUrl: getEnv("VITE_WS_URL", "ws://localhost:8000/ws"),

  /** Request timeout in milliseconds */
  timeout: getEnvInt("VITE_API_TIMEOUT", 30000),

  /** Enable request retry */
  enableRetry: getEnvBool("VITE_API_RETRY", true),

  /** Maximum retry attempts */
  maxRetries: getEnvInt("VITE_API_MAX_RETRIES", 3),
} as const;

/**
 * Authentication Configuration
 */
export const authConfig = {
  /** Session timeout in minutes */
  sessionTimeout: getEnvInt("VITE_SESSION_TIMEOUT", 60),

  /** Enable 2FA */
  enable2FA: getEnvBool("VITE_ENABLE_2FA", true),

  /** Inactivity timeout in minutes */
  inactivityTimeout: getEnvInt("VITE_INACTIVITY_TIMEOUT", 30),

  /** Remember me duration in days */
  rememberMeDays: getEnvInt("VITE_REMEMBER_ME_DAYS", 7),
} as const;

/**
 * Feature Flags
 */
export const featureFlags = {
  /** Enable experiments feature */
  enableExperiments: getEnvBool("VITE_FF_EXPERIMENTS", true),

  /** Enable three-version cycle */
  enableThreeVersion: getEnvBool("VITE_FF_THREE_VERSION", true),

  /** Enable AI chat */
  enableAIChat: getEnvBool("VITE_FF_AI_CHAT", true),

  /** Enable code fix suggestions */
  enableCodeFix: getEnvBool("VITE_FF_CODE_FIX", true),

  /** Enable dark mode */
  enableDarkMode: getEnvBool("VITE_FF_DARK_MODE", true),

  /** Enable offline mode */
  enableOffline: getEnvBool("VITE_FF_OFFLINE", false),

  /** Mock mode for development */
  mockMode: getEnvBool("VITE_MOCK_MODE", false),
} as const;

/**
 * UI Configuration
 */
export const uiConfig = {
  /** Default theme */
  defaultTheme: getEnv("VITE_DEFAULT_THEME", "system") as "light" | "dark" | "system",

  /** Default language */
  defaultLanguage: getEnv("VITE_DEFAULT_LANGUAGE", "en"),

  /** Supported languages */
  supportedLanguages: ["en", "zh-CN", "zh-TW"] as const,

  /** Items per page default */
  pageSize: getEnvInt("VITE_PAGE_SIZE", 20),

  /** Max file size for upload (MB) */
  maxFileSize: getEnvInt("VITE_MAX_FILE_SIZE", 10),

  /** Enable animations */
  enableAnimations: getEnvBool("VITE_ENABLE_ANIMATIONS", true),
} as const;

/**
 * Analytics Configuration
 */
export const analyticsConfig = {
  /** Enable analytics */
  enabled: getEnvBool("VITE_ANALYTICS_ENABLED", false),

  /** Google Analytics ID */
  gaId: getEnv("VITE_GA_ID", ""),

  /** Sentry DSN */
  sentryDsn: getEnv("VITE_SENTRY_DSN", ""),

  /** Enable error tracking */
  enableErrorTracking: getEnvBool("VITE_ERROR_TRACKING", true),
} as const;

/**
 * Development Configuration
 */
export const devConfig = {
  /** Enable debug logging */
  debugLogging: getEnvBool("VITE_DEBUG", false),

  /** Enable React DevTools */
  enableDevTools: getEnvBool("VITE_ENABLE_DEVTOOLS", true),

  /** Enable query devtools */
  enableQueryDevtools: getEnvBool("VITE_QUERY_DEVTOOLS", false),
} as const;

/**
 * Current environment
 */
export const environment: Environment = (import.meta.env.MODE || "development") as Environment;

/**
 * Environment checks
 */
export const isProduction = environment === "production";
export const isDevelopment = environment === "development";
export const isTest = environment === "test";

/**
 * Complete configuration object
 */
export const config = {
  environment,
  isProduction,
  isDevelopment,
  isTest,
  api: apiConfig,
  auth: authConfig,
  features: featureFlags,
  ui: uiConfig,
  analytics: analyticsConfig,
  dev: devConfig,
} as const;

/**
 * Validate configuration
 * Call this on app startup to catch configuration errors early
 */
export function validateConfig(): string[] {
  const errors: string[] = [];

  // Validate API URL
  if (!apiConfig.baseUrl) {
    errors.push("VITE_API_URL is required");
  }

  // Validate timeout
  if (apiConfig.timeout < 1000) {
    errors.push("VITE_API_TIMEOUT must be at least 1000ms");
  }

  // Production-specific validations
  if (isProduction) {
    if (featureFlags.mockMode) {
      errors.push("VITE_MOCK_MODE should be false in production");
    }

    if (devConfig.debugLogging) {
      errors.push("VITE_DEBUG should be false in production");
    }

    if (!analyticsConfig.enabled) {
      console.warn("Analytics is disabled in production");
    }
  }

  return errors;
}

/**
 * Log configuration (safe for logging, excludes secrets)
 */
export function logConfig(): void {
  console.log("Application Configuration:", {
    environment,
    api: {
      baseUrl: apiConfig.baseUrl,
      timeout: apiConfig.timeout,
    },
    features: featureFlags,
    ui: {
      defaultTheme: uiConfig.defaultTheme,
      defaultLanguage: uiConfig.defaultLanguage,
    },
  });
}

export default config;
