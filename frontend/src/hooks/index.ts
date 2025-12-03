export { useAuth } from "./useAuth";
export { useSecureAuth } from "./useSecureAuth";
export { useRateLimiter, rateLimitConfigs } from "./useRateLimiter";
export type { RateLimitStatus, BackoffConfig } from "./useRateLimiter";
export { useSSE } from "./useSSE";
export { useWebSocket } from "./useWebSocket";

// Project hooks (exclude API key hooks to avoid conflict with useUser)
export {
  useProjects,
  useProject,
  useCreateProject,
  useUpdateProject,
  useDeleteProject,
  useArchiveProject,
  useRestoreProject,
} from "./useProjects";

// User profile and settings hooks (includes API key management)
export * from "./useUser";

// Admin hooks
export * from "./useAdmin";

export { useOfflineSync } from "./useOfflineSync";

// Self-Evolution hooks
export { useAutoFix } from "./useAutoFix";
export type { Vulnerability, Fix, CycleStatus } from "./useAutoFix";
export { useRealTimeNotifications } from "./useRealTimeNotifications";
export type { SystemNotification } from "./useRealTimeNotifications";
export { useLearning } from "./useLearning";
export type {
  LearningSource,
  KnowledgeUpdate,
  LearningStatus,
} from "./useLearning";

// Data fetching hooks
export { useAsyncData, useAsyncDataAll } from "./useAsyncData";
export type {
  AsyncDataState,
  AsyncDataOptions,
  AsyncDataResult,
} from "./useAsyncData";
