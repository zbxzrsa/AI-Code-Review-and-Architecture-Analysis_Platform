/**
 * Admin Hooks
 *
 * React Query hooks for admin dashboard - users, providers, and audit management.
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { message } from "antd";
import { AxiosError } from "axios";
import { apiService } from "../services/api";
import { ensureArray } from "../utils/safeData";
import {
  useAdminStore,
  type AdminUser,
  type UserStats,
  type UserFilters,
  type AIProvider,
  type ProviderModel,
  type ProviderHealth,
  type ProviderMetrics,
  type AuditLog,
  type AuditFilters,
  type AuditAnalytics,
  type SecurityAlert,
} from "../store/adminStore";

// Type-safe error extraction for API errors
interface ApiError {
  detail?: string;
  message?: string;
}

function getErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof AxiosError) {
    const data = error.response?.data as ApiError | undefined;
    return data?.detail || data?.message || fallback;
  }
  return fallback;
}

// ============================================
// Query Keys
// ============================================

export const adminKeys = {
  all: ["admin"] as const,

  // Users
  users: () => [...adminKeys.all, "users"] as const,
  usersList: (
    filters: UserFilters,
    page: number,
    pageSize: number,
    sort: { field: string; order: string }
  ) => [...adminKeys.users(), "list", filters, page, pageSize, sort] as const,
  user: (userId: string) => [...adminKeys.users(), userId] as const,
  userStats: () => [...adminKeys.users(), "stats"] as const,

  // Providers
  providers: () => [...adminKeys.all, "providers"] as const,
  providersList: () => [...adminKeys.providers(), "list"] as const,
  provider: (providerId: string) => [...adminKeys.providers(), providerId] as const,
  providerHealth: (providerId: string) => [...adminKeys.providers(), providerId, "health"] as const,
  providerMetrics: (providerId: string, period?: string) =>
    [...adminKeys.providers(), providerId, "metrics", period] as const,
  providerModels: (providerId: string) => [...adminKeys.providers(), providerId, "models"] as const,

  // Audit
  audit: () => [...adminKeys.all, "audit"] as const,
  auditLogs: (filters: AuditFilters, page: number, pageSize: number) =>
    [...adminKeys.audit(), "logs", filters, page, pageSize] as const,
  auditLog: (logId: string) => [...adminKeys.audit(), "log", logId] as const,
  auditAnalytics: (period?: string) => [...adminKeys.audit(), "analytics", period] as const,
  securityAlerts: (params?: { severity?: string; resolved?: boolean }) =>
    [...adminKeys.audit(), "alerts", params] as const,
  loginPatterns: (period?: string) => [...adminKeys.audit(), "login-patterns", period] as const,
};

// ============================================
// User Hooks
// ============================================

/**
 * Fetch users list with filters and pagination
 */
export function useAdminUsers() {
  const { userFilters, userPagination, userSort, setUsers } = useAdminStore();

  return useQuery({
    queryKey: adminKeys.usersList(
      userFilters,
      userPagination.page,
      userPagination.pageSize,
      userSort
    ),
    queryFn: async () => {
      const response = await apiService.admin.users.list({
        page: userPagination.page,
        limit: userPagination.pageSize,
        search: userFilters.search || undefined,
        role: userFilters.role === "all" ? undefined : userFilters.role,
        status: userFilters.status === "all" ? undefined : userFilters.status,
        sort_field: userSort.field,
        sort_order: userSort.order,
        start_date: userFilters.dateRange?.[0],
        end_date: userFilters.dateRange?.[1],
      });

      const data = response.data as { items: AdminUser[]; total: number };
      setUsers(data.items);
      return data;
    },
  });
}

/**
 * Fetch single user
 */
export function useAdminUser(userId: string) {
  return useQuery({
    queryKey: adminKeys.user(userId),
    queryFn: async () => {
      const response = await apiService.admin.users.get(userId);
      return response.data as AdminUser;
    },
    enabled: !!userId,
  });
}

/**
 * Fetch user statistics
 */
export function useUserStats() {
  const { setUserStats } = useAdminStore();

  return useQuery({
    queryKey: adminKeys.userStats(),
    queryFn: async () => {
      const response = await apiService.admin.users.getStats();
      const stats = response.data as UserStats;
      setUserStats(stats);
      return stats;
    },
  });
}

/**
 * Update user
 */
export function useUpdateUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      userId,
      data,
    }: {
      userId: string;
      data: Parameters<typeof apiService.admin.users.update>[1];
    }) => {
      const response = await apiService.admin.users.update(userId, data);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: adminKeys.users() });
      message.success("User updated successfully");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to update user"));
    },
  });
}

/**
 * Delete user
 */
export function useDeleteUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (userId: string) => {
      await apiService.admin.users.delete(userId);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: adminKeys.users() });
      message.success("User deleted successfully");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to delete user"));
    },
  });
}

/**
 * Suspend user
 */
export function useSuspendUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ userId, reason }: { userId: string; reason?: string }) => {
      await apiService.admin.users.suspend(userId, reason);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: adminKeys.users() });
      message.success("User suspended");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to suspend user"));
    },
  });
}

/**
 * Reactivate user
 */
export function useReactivateUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (userId: string) => {
      await apiService.admin.users.reactivate(userId);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: adminKeys.users() });
      message.success("User reactivated");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to reactivate user"));
    },
  });
}

/**
 * Reset user password
 */
export function useResetUserPassword() {
  return useMutation({
    mutationFn: async (userId: string) => {
      await apiService.admin.users.resetPassword(userId);
    },
    onSuccess: () => {
      message.success("Password reset email sent");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to reset password"));
    },
  });
}

/**
 * Resend welcome email
 */
export function useResendWelcome() {
  return useMutation({
    mutationFn: async (userId: string) => {
      await apiService.admin.users.resendWelcome(userId);
    },
    onSuccess: () => {
      message.success("Welcome email sent");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to send email"));
    },
  });
}

/**
 * Bulk user operations
 */
export function useBulkUserAction() {
  const queryClient = useQueryClient();
  const { clearUserSelection } = useAdminStore();

  return useMutation({
    mutationFn: async (data: Parameters<typeof apiService.admin.users.bulk>[0]) => {
      await apiService.admin.users.bulk(data);
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: adminKeys.users() });
      clearUserSelection();
      message.success(`Bulk ${variables.action} completed`);
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Bulk operation failed"));
    },
  });
}

/**
 * Import users from CSV
 */
export function useImportUsers() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (file: File) => {
      const response = await apiService.admin.users.import(file);
      return response.data;
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: adminKeys.users() });
      message.success(`Imported ${data.imported} users`);
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Import failed"));
    },
  });
}

/**
 * Export users
 */
export function useExportUsers() {
  return useMutation({
    mutationFn: async (params?: Parameters<typeof apiService.admin.users.export>[0]) => {
      const response = await apiService.admin.users.export(params);
      return response.data;
    },
    onSuccess: (data, variables) => {
      const format = variables?.format || "csv";
      const url = globalThis.URL.createObjectURL(new Blob([data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", `users.${format}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      globalThis.URL.revokeObjectURL(url);
      message.success("Export started");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Export failed"));
    },
  });
}

// ============================================
// Provider Hooks
// ============================================

/**
 * Fetch all providers
 */
export function useProviders() {
  const { setProviders } = useAdminStore();

  return useQuery({
    queryKey: adminKeys.providersList(),
    queryFn: async () => {
      const response = await apiService.admin.providers.list();
      const providers = response.data as AIProvider[];
      setProviders(providers);
      return providers;
    },
  });
}

/**
 * Fetch single provider
 */
export function useProvider(providerId: string) {
  return useQuery({
    queryKey: adminKeys.provider(providerId),
    queryFn: async () => {
      const response = await apiService.admin.providers.get(providerId);
      return response.data as AIProvider & { config: Record<string, unknown> };
    },
    enabled: !!providerId,
  });
}

/**
 * Update provider
 */
export function useUpdateProvider() {
  const queryClient = useQueryClient();
  const { updateProvider } = useAdminStore();

  return useMutation({
    mutationFn: async ({
      providerId,
      data,
    }: {
      providerId: string;
      data: Parameters<typeof apiService.admin.providers.update>[1];
    }) => {
      const response = await apiService.admin.providers.update(providerId, data);
      return response.data;
    },
    onSuccess: (data, { providerId }) => {
      updateProvider(providerId, data);
      queryClient.invalidateQueries({ queryKey: adminKeys.providers() });
      message.success("Provider updated");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to update provider"));
    },
  });
}

/**
 * Update provider API key
 */
export function useUpdateProviderApiKey() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ providerId, apiKey }: { providerId: string; apiKey: string }) => {
      await apiService.admin.providers.updateApiKey(providerId, apiKey);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: adminKeys.providers() });
      message.success("API key updated");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to update API key"));
    },
  });
}

/**
 * Test provider connectivity
 */
export function useTestProvider() {
  return useMutation({
    mutationFn: async (providerId: string) => {
      const response = await apiService.admin.providers.test(providerId);
      return response.data as {
        success: boolean;
        latency: number;
        message?: string;
      };
    },
    onSuccess: (data) => {
      if (data.success) {
        message.success(`Connection successful (${data.latency}ms)`);
      } else {
        message.warning(data.message || "Connection failed");
      }
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Connection test failed"));
    },
  });
}

/**
 * Fetch provider health
 */
export function useProviderHealth(providerId: string) {
  return useQuery({
    queryKey: adminKeys.providerHealth(providerId),
    queryFn: async () => {
      const response = await apiService.admin.providers.getHealth(providerId);
      return response.data as ProviderHealth;
    },
    enabled: !!providerId,
    refetchInterval: 30000, // Refresh every 30 seconds
  });
}

/**
 * Fetch provider metrics
 */
export function useProviderMetrics(providerId: string, period?: "hour" | "day" | "week" | "month") {
  return useQuery({
    queryKey: adminKeys.providerMetrics(providerId, period),
    queryFn: async () => {
      const response = await apiService.admin.providers.getMetrics(providerId, {
        period,
      });
      return response.data as ProviderMetrics;
    },
    enabled: !!providerId,
  });
}

/**
 * Fetch provider models
 */
export function useProviderModels(providerId: string) {
  const { setProviderModels } = useAdminStore();

  return useQuery({
    queryKey: adminKeys.providerModels(providerId),
    queryFn: async () => {
      const response = await apiService.admin.providers.getModels(providerId);
      const models = response.data as ProviderModel[];
      setProviderModels(models);
      return models;
    },
    enabled: !!providerId,
  });
}

/**
 * Update provider model
 */
export function useUpdateProviderModel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      providerId,
      modelId,
      data,
    }: {
      providerId: string;
      modelId: string;
      data: Parameters<typeof apiService.admin.providers.updateModel>[2];
    }) => {
      const response = await apiService.admin.providers.updateModel(providerId, modelId, data);
      return response.data;
    },
    onSuccess: (_, { providerId }) => {
      queryClient.invalidateQueries({
        queryKey: adminKeys.providerModels(providerId),
      });
      message.success("Model configuration updated");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to update model"));
    },
  });
}

/**
 * Set provider fallback order
 */
export function useSetFallbackOrder() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (providerIds: string[]) => {
      await apiService.admin.providers.setFallbackOrder(providerIds);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: adminKeys.providers() });
      message.success("Fallback order updated");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to update fallback order"));
    },
  });
}

// ============================================
// Audit Hooks
// ============================================

/**
 * Fetch audit logs
 */
export function useAuditLogs() {
  const { auditFilters, auditPagination, setAuditLogs } = useAdminStore();

  return useQuery({
    queryKey: adminKeys.auditLogs(auditFilters, auditPagination.page, auditPagination.pageSize),
    queryFn: async () => {
      const response = await apiService.admin.audit.list({
        page: auditPagination.page,
        limit: auditPagination.pageSize,
        search: auditFilters.search || undefined,
        action: auditFilters.action === "all" ? undefined : auditFilters.action,
        resource: auditFilters.resource === "all" ? undefined : auditFilters.resource,
        status: auditFilters.status === "all" ? undefined : auditFilters.status,
        user_id: auditFilters.userId,
        start_date: auditFilters.dateRange?.[0],
        end_date: auditFilters.dateRange?.[1],
      });

      const data = response.data as { items: AuditLog[]; total: number };
      setAuditLogs(data.items);
      return data;
    },
  });
}

/**
 * Fetch single audit log
 */
export function useAuditLog(logId: string) {
  return useQuery({
    queryKey: adminKeys.auditLog(logId),
    queryFn: async () => {
      const response = await apiService.admin.audit.get(logId);
      return response.data as AuditLog;
    },
    enabled: !!logId,
  });
}

/**
 * Export audit logs
 */
export function useExportAuditLogs() {
  const { auditFilters } = useAdminStore();

  return useMutation({
    mutationFn: async (format: "csv" | "json" | "pdf" = "csv") => {
      const response = await apiService.admin.audit.export({
        format,
        action: auditFilters.action === "all" ? undefined : auditFilters.action,
        resource: auditFilters.resource === "all" ? undefined : auditFilters.resource,
        status: auditFilters.status === "all" ? undefined : auditFilters.status,
        start_date: auditFilters.dateRange?.[0],
        end_date: auditFilters.dateRange?.[1],
      });
      return { data: response.data, format };
    },
    onSuccess: ({ data, format }) => {
      const url = globalThis.URL.createObjectURL(new Blob([data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", `audit-logs.${format}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      globalThis.URL.revokeObjectURL(url);
      message.success("Export completed");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Export failed"));
    },
  });
}

/**
 * Fetch audit analytics
 */
export function useAuditAnalytics(period?: "day" | "week" | "month") {
  const { setAuditAnalytics } = useAdminStore();

  return useQuery({
    queryKey: adminKeys.auditAnalytics(period),
    queryFn: async () => {
      const response = await apiService.admin.audit.getAnalytics({ period });
      const analytics = response.data as AuditAnalytics;
      setAuditAnalytics(analytics);
      return analytics;
    },
  });
}

/**
 * Fetch security alerts
 */
export function useSecurityAlerts(params?: { severity?: string; resolved?: boolean }) {
  const { setSecurityAlerts } = useAdminStore();

  return useQuery({
    queryKey: adminKeys.securityAlerts(params),
    queryFn: async () => {
      const response = await apiService.admin.audit.getAlerts(params);
      const data = response.data as { items: SecurityAlert[]; total: number };
      setSecurityAlerts(data.items);
      return data;
    },
  });
}

/**
 * Resolve security alert
 */
export function useResolveAlert() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ alertId, notes }: { alertId: string; notes?: string }) => {
      await apiService.admin.audit.resolveAlert(alertId, notes);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: adminKeys.securityAlerts() });
      message.success("Alert resolved");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to resolve alert"));
    },
  });
}

/**
 * Fetch login patterns
 */
export function useLoginPatterns(period?: "day" | "week" | "month") {
  return useQuery({
    queryKey: adminKeys.loginPatterns(period),
    queryFn: async () => {
      const response = await apiService.admin.audit.getLoginPatterns({
        period,
      });
      return ensureArray<{ hour: number; day: number; count: number }>(response.data);
    },
  });
}
