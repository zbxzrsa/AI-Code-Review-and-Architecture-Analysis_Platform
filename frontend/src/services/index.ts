export { api, apiService } from "./api";
export {
  storage,
  storageKeys,
  authStorage,
  preferencesStorage,
  projectStorage,
} from "./storage";
export {
  default as WebSocketService,
  mainWebSocket,
  collaborationWebSocket,
  notificationWebSocket,
  createDocumentWebSocket,
} from "./websocket";
export {
  errorLoggingService,
  useErrorLogging,
  ErrorCategory,
  ErrorSeverity,
} from "./errorLogging";
export type { ErrorLogEntry, RecoveryStrategy } from "./errorLogging";
export {
  csrfManager,
  rateLimiter,
  twoFactorAuth,
  sessionSecurity,
  secureStorage,
  validateSecurityHeaders,
  sanitizeInput,
  isValidEmail,
  validatePasswordStrength,
  getDeviceFingerprint,
  setupCSPReporting,
  TwoFactorState,
} from "./security";
export type {
  RateLimitConfig,
  SecurityHeaders,
  TwoFactorSetupData,
  PasswordStrength,
} from "./security";

// Performance & Caching
export {
  performanceMonitor,
  usePerformanceTracking,
} from "./performanceMonitor";
export { cacheService, cached } from "./cacheService";
export { securityService } from "./securityService";
export { eventBus, useEvent, useEventEmitter } from "./eventBus";
export { enhancedApi } from "./enhancedApi";

// Lifecycle Controller API
export { lifecycleApi, LifecycleApiClient } from "./lifecycleApi";
export type {
  ComparisonRequest,
  VersionOutput,
  Issue,
  RollbackRequest,
  RollbackResponse,
  VersionConfig,
  VersionMetrics,
  EvaluationStatus,
  ComparisonStats,
  RollbackStats,
} from "./lifecycleApi";
