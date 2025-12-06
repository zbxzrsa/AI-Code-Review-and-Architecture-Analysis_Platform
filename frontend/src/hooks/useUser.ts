/**
 * User Profile & Settings Hooks
 *
 * React Query hooks for user profile, settings, OAuth, sessions,
 * 2FA, API keys, integrations, and webhooks.
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { message } from "antd";
import { AxiosError } from "axios";
import { apiService } from "../services/api";
import {
  useAuthStore,
  type User,
  type UserSettings,
  type OAuthConnection,
  type Session,
  type LoginHistory,
  type ApiActivity,
  type UserApiKey,
  type Integration,
  type UserWebhook,
  type TwoFactorSetup,
} from "../store/authStore";

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

/**
 * Safely extract array from API response
 * Handles both direct arrays and { items: [...] } / { connections: [...] } formats
 */
function extractArray<T>(data: unknown, key = "items"): T[] {
  if (Array.isArray(data)) {
    return data as T[];
  }
  if (data && typeof data === "object") {
    const obj = data as Record<string, unknown>;
    if (Array.isArray(obj[key])) {
      return obj[key] as T[];
    }
    // Also check common keys
    if (Array.isArray(obj.items)) {
      return obj.items as T[];
    }
    if (Array.isArray(obj.connections)) {
      return obj.connections as T[];
    }
    if (Array.isArray(obj.data)) {
      return obj.data as T[];
    }
  }
  return [];
}

// Query Keys
export const userKeys = {
  all: ["user"] as const,
  profile: () => [...userKeys.all, "profile"] as const,
  settings: () => [...userKeys.all, "settings"] as const,
  oauth: () => [...userKeys.all, "oauth"] as const,
  oauthConnections: () => [...userKeys.oauth(), "connections"] as const,
  oauthProviders: () => [...userKeys.oauth(), "providers"] as const,
  sessions: () => [...userKeys.all, "sessions"] as const,
  loginHistory: (params?: { page?: number; limit?: number }) =>
    [...userKeys.all, "loginHistory", params] as const,
  apiActivity: (params?: { page?: number; limit?: number }) =>
    [...userKeys.all, "apiActivity", params] as const,
  apiKeys: () => [...userKeys.all, "apiKeys"] as const,
  apiKeyUsage: (
    keyId: string,
    params?: { start_date?: string; end_date?: string }
  ) => [...userKeys.apiKeys(), keyId, "usage", params] as const,
  twoFactor: () => [...userKeys.all, "2fa"] as const,
  integrations: () => [...userKeys.all, "integrations"] as const,
  webhooks: () => [...userKeys.all, "webhooks"] as const,
  webhookLogs: (
    webhookId: string,
    params?: { page?: number; limit?: number }
  ) => [...userKeys.webhooks(), webhookId, "logs", params] as const,
  ipWhitelist: () => [...userKeys.all, "ipWhitelist"] as const,
  loginAlerts: () => [...userKeys.all, "loginAlerts"] as const,
};

// ============================================
// Profile Hooks
// ============================================

/**
 * Fetch user profile
 */
export function useUserProfile() {
  const { setUser } = useAuthStore();

  return useQuery({
    queryKey: userKeys.profile(),
    queryFn: async () => {
      const response = await apiService.user.getProfile();
      setUser(response.data);
      return response.data as User;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

/**
 * Update user profile
 */
export function useUpdateProfile() {
  const queryClient = useQueryClient();
  const { updateUser } = useAuthStore();

  return useMutation({
    mutationFn: async (data: {
      name?: string;
      username?: string;
      bio?: string;
    }) => {
      const response = await apiService.user.updateProfile(data);
      return response.data;
    },
    onSuccess: (data) => {
      updateUser(data);
      queryClient.invalidateQueries({ queryKey: userKeys.profile() });
      message.success("Profile updated successfully");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to update profile"));
    },
  });
}

/**
 * Upload avatar
 */
export function useUploadAvatar() {
  const queryClient = useQueryClient();
  const { updateUser } = useAuthStore();

  return useMutation({
    mutationFn: async (file: File) => {
      const response = await apiService.user.uploadAvatar(file);
      return response.data;
    },
    onSuccess: (data) => {
      updateUser({ avatar: data.avatar_url });
      queryClient.invalidateQueries({ queryKey: userKeys.profile() });
      message.success("Avatar updated successfully");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to upload avatar"));
    },
  });
}

/**
 * Delete avatar
 */
export function useDeleteAvatar() {
  const queryClient = useQueryClient();
  const { updateUser } = useAuthStore();

  return useMutation({
    mutationFn: async () => {
      await apiService.user.deleteAvatar();
    },
    onSuccess: () => {
      updateUser({ avatar: undefined });
      queryClient.invalidateQueries({ queryKey: userKeys.profile() });
      message.success("Avatar removed");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to remove avatar"));
    },
  });
}

// ============================================
// Settings Hooks
// ============================================

/**
 * Fetch user settings
 */
export function useUserSettings() {
  const { setSettings } = useAuthStore();

  return useQuery({
    queryKey: userKeys.settings(),
    queryFn: async () => {
      const response = await apiService.user.getSettings();
      setSettings(response.data);
      return response.data as UserSettings;
    },
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * Update user settings
 */
export function useUpdateSettings() {
  const queryClient = useQueryClient();
  const { updateSettings } = useAuthStore();

  return useMutation({
    mutationFn: async (data: Partial<UserSettings>) => {
      const response = await apiService.user.updateSettings(data);
      return response.data;
    },
    onSuccess: (data) => {
      updateSettings(data);
      queryClient.invalidateQueries({ queryKey: userKeys.settings() });
      message.success("Settings updated");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to update settings"));
    },
  });
}

/**
 * Update privacy settings
 */
export function useUpdatePrivacy() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (
      data: Parameters<typeof apiService.user.updatePrivacy>[0]
    ) => {
      const response = await apiService.user.updatePrivacy(data);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: userKeys.settings() });
      message.success("Privacy settings updated");
    },
    onError: (error: unknown) => {
      message.error(
        getErrorMessage(error, "Failed to update privacy settings")
      );
    },
  });
}

/**
 * Update notification preferences
 */
export function useUpdateNotifications() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (data: Record<string, unknown>) => {
      const response = await apiService.user.updateNotifications(data);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: userKeys.settings() });
      message.success("Notification preferences updated");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to update notifications"));
    },
  });
}

// ============================================
// OAuth Hooks
// ============================================

/**
 * Fetch OAuth providers
 */
export function useOAuthProviders() {
  return useQuery({
    queryKey: userKeys.oauthProviders(),
    queryFn: async () => {
      const response = await apiService.user.getOAuthProviders();
      return response.data as {
        provider: string;
        name: string;
        icon: string;
      }[];
    },
  });
}

/**
 * Fetch OAuth connections
 */
export function useOAuthConnections() {
  return useQuery({
    queryKey: userKeys.oauthConnections(),
    queryFn: async () => {
      const response = await apiService.user.getOAuthConnections();
      return extractArray<OAuthConnection>(response.data, "connections");
    },
  });
}

/**
 * Connect OAuth provider
 */
export function useConnectOAuth() {
  return useMutation({
    mutationFn: async (provider: string) => {
      const response = await apiService.user.connectOAuth(provider);
      // This typically returns a redirect URL
      return response.data as { redirect_url: string };
    },
    onSuccess: (data) => {
      // Redirect to OAuth provider
      window.location.href = data.redirect_url;
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to connect account"));
    },
  });
}

/**
 * Disconnect OAuth provider
 */
export function useDisconnectOAuth() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (provider: string) => {
      await apiService.user.disconnectOAuth(provider);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: userKeys.oauthConnections() });
      message.success("Account disconnected");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to disconnect account"));
    },
  });
}

// ============================================
// Password Hooks
// ============================================

/**
 * Change password
 */
export function useChangePassword() {
  return useMutation({
    mutationFn: async (data: {
      currentPassword: string;
      newPassword: string;
    }) => {
      await apiService.user.changePassword(data);
    },
    onSuccess: () => {
      message.success("Password changed successfully");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to change password"));
    },
  });
}

/**
 * Change email
 */
export function useChangeEmail() {
  return useMutation({
    mutationFn: async (data: { newEmail: string; password: string }) => {
      await apiService.user.changeEmail(data.newEmail, data.password);
    },
    onSuccess: () => {
      message.success("Verification email sent to your new address");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to change email"));
    },
  });
}

// ============================================
// Two-Factor Authentication Hooks
// ============================================

/**
 * Get 2FA status
 */
export function use2FAStatus() {
  return useQuery({
    queryKey: userKeys.twoFactor(),
    queryFn: async () => {
      const response = await apiService.user.get2FAStatus();
      return response.data as {
        enabled: boolean;
        backupCodesRemaining: number;
      };
    },
  });
}

/**
 * Setup 2FA (get QR code)
 */
export function useSetup2FA() {
  return useMutation({
    mutationFn: async () => {
      const response = await apiService.user.setup2FA();
      return response.data as TwoFactorSetup;
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to setup 2FA"));
    },
  });
}

/**
 * Enable 2FA
 */
export function useEnable2FA() {
  const queryClient = useQueryClient();
  const { updateUser } = useAuthStore();

  return useMutation({
    mutationFn: async (code: string) => {
      const response = await apiService.user.enable2FA(code);
      return response.data;
    },
    onSuccess: () => {
      updateUser({ twoFactorEnabled: true });
      queryClient.invalidateQueries({ queryKey: userKeys.twoFactor() });
      message.success("Two-factor authentication enabled");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Invalid verification code"));
    },
  });
}

/**
 * Disable 2FA
 */
export function useDisable2FA() {
  const queryClient = useQueryClient();
  const { updateUser } = useAuthStore();

  return useMutation({
    mutationFn: async (data: { code: string; password: string }) => {
      await apiService.user.disable2FA(data.code, data.password);
    },
    onSuccess: () => {
      updateUser({ twoFactorEnabled: false });
      queryClient.invalidateQueries({ queryKey: userKeys.twoFactor() });
      message.success("Two-factor authentication disabled");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to disable 2FA"));
    },
  });
}

/**
 * Regenerate backup codes
 */
export function useRegenerateBackupCodes() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (password: string) => {
      const response = await apiService.user.regenerateBackupCodes(password);
      return response.data as { backupCodes: string[] };
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: userKeys.twoFactor() });
      message.success("Backup codes regenerated");
    },
    onError: (error: unknown) => {
      message.error(
        getErrorMessage(error, "Failed to regenerate backup codes")
      );
    },
  });
}

// ============================================
// Sessions Hooks
// ============================================

/**
 * Fetch active sessions
 */
export function useSessions() {
  return useQuery({
    queryKey: userKeys.sessions(),
    queryFn: async () => {
      const response = await apiService.user.getSessions();
      return extractArray<Session>(response.data, "items");
    },
  });
}

/**
 * Revoke session
 */
export function useRevokeSession() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (sessionId: string) => {
      await apiService.user.revokeSession(sessionId);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: userKeys.sessions() });
      message.success("Session revoked");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to revoke session"));
    },
  });
}

/**
 * Revoke all sessions
 */
export function useRevokeAllSessions() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async () => {
      await apiService.user.revokeAllSessions();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: userKeys.sessions() });
      message.success("All other sessions revoked");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to revoke sessions"));
    },
  });
}

// ============================================
// Activity Hooks
// ============================================

/**
 * Fetch login history
 */
export function useLoginHistory(params?: { page?: number; limit?: number }) {
  return useQuery({
    queryKey: userKeys.loginHistory(params),
    queryFn: async () => {
      const response = await apiService.user.getLoginHistory(params);
      return response.data as { items: LoginHistory[]; total: number };
    },
  });
}

/**
 * Fetch API activity
 */
export function useApiActivity(params?: { page?: number; limit?: number }) {
  return useQuery({
    queryKey: userKeys.apiActivity(params),
    queryFn: async () => {
      const response = await apiService.user.getApiActivity(params);
      return response.data as { items: ApiActivity[]; total: number };
    },
  });
}

// ============================================
// API Keys Hooks
// ============================================

/**
 * Fetch API keys
 */
export function useApiKeys() {
  return useQuery({
    queryKey: userKeys.apiKeys(),
    queryFn: async () => {
      const response = await apiService.user.getApiKeys();
      return extractArray<UserApiKey>(response.data, "items");
    },
  });
}

/**
 * Create API key
 */
export function useCreateApiKey() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (data: {
      name: string;
      permissions: string[];
      expiresAt?: string;
    }) => {
      const response = await apiService.user.createApiKey(data);
      return response.data as { key: string; id: string };
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: userKeys.apiKeys() });
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to create API key"));
    },
  });
}

/**
 * Revoke API key
 */
export function useRevokeApiKey() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (keyId: string) => {
      await apiService.user.revokeApiKey(keyId);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: userKeys.apiKeys() });
      message.success("API key revoked");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to revoke API key"));
    },
  });
}

/**
 * Fetch API key usage
 */
export function useApiKeyUsage(
  keyId: string,
  params?: { start_date?: string; end_date?: string },
  enabled = true
) {
  return useQuery({
    queryKey: userKeys.apiKeyUsage(keyId, params),
    queryFn: async () => {
      const response = await apiService.user.getApiKeyUsage(keyId, params);
      return response.data as { usage: { date: string; count: number }[] };
    },
    enabled,
  });
}

// ============================================
// Integrations Hooks
// ============================================

/**
 * Fetch integrations
 */
export function useIntegrations() {
  return useQuery({
    queryKey: userKeys.integrations(),
    queryFn: async () => {
      const response = await apiService.user.getIntegrations();
      return extractArray<Integration>(response.data, "items");
    },
  });
}

/**
 * Connect Slack
 */
export function useConnectSlack() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (code: string) => {
      const response = await apiService.user.connectSlack(code);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: userKeys.integrations() });
      message.success("Slack connected");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to connect Slack"));
    },
  });
}

/**
 * Disconnect Slack
 */
export function useDisconnectSlack() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async () => {
      await apiService.user.disconnectSlack();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: userKeys.integrations() });
      message.success("Slack disconnected");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to disconnect Slack"));
    },
  });
}

/**
 * Connect Teams
 */
export function useConnectTeams() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (code: string) => {
      const response = await apiService.user.connectTeams(code);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: userKeys.integrations() });
      message.success("Microsoft Teams connected");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to connect Teams"));
    },
  });
}

/**
 * Disconnect Teams
 */
export function useDisconnectTeams() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async () => {
      await apiService.user.disconnectTeams();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: userKeys.integrations() });
      message.success("Microsoft Teams disconnected");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to disconnect Teams"));
    },
  });
}

// ============================================
// Webhooks Hooks
// ============================================

/**
 * Fetch webhooks
 */
export function useUserWebhooks() {
  return useQuery({
    queryKey: userKeys.webhooks(),
    queryFn: async () => {
      const response = await apiService.user.getWebhooks();
      return extractArray<UserWebhook>(response.data, "items");
    },
  });
}

/**
 * Create webhook
 */
export function useCreateUserWebhook() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (data: {
      name: string;
      url: string;
      events: string[];
      secret?: string;
    }) => {
      const response = await apiService.user.createWebhook(data);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: userKeys.webhooks() });
      message.success("Webhook created");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to create webhook"));
    },
  });
}

/**
 * Update webhook
 */
export function useUpdateUserWebhook() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      webhookId,
      data,
    }: {
      webhookId: string;
      data: {
        name?: string;
        url?: string;
        events?: string[];
        isActive?: boolean;
      };
    }) => {
      const response = await apiService.user.updateWebhook(webhookId, data);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: userKeys.webhooks() });
      message.success("Webhook updated");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to update webhook"));
    },
  });
}

/**
 * Delete webhook
 */
export function useDeleteUserWebhook() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (webhookId: string) => {
      await apiService.user.deleteWebhook(webhookId);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: userKeys.webhooks() });
      message.success("Webhook deleted");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to delete webhook"));
    },
  });
}

/**
 * Test webhook
 */
export function useTestUserWebhook() {
  return useMutation({
    mutationFn: async (webhookId: string) => {
      const response = await apiService.user.testWebhook(webhookId);
      return response.data;
    },
    onSuccess: () => {
      message.success("Webhook test sent");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Webhook test failed"));
    },
  });
}

/**
 * Fetch webhook logs
 */
export function useWebhookLogs(
  webhookId: string,
  params?: { page?: number; limit?: number }
) {
  return useQuery({
    queryKey: userKeys.webhookLogs(webhookId, params),
    queryFn: async () => {
      const response = await apiService.user.getWebhookLogs(webhookId, params);
      return response.data as { items: any[]; total: number };
    },
    enabled: !!webhookId,
  });
}

// ============================================
// Security Hooks
// ============================================

/**
 * Fetch IP whitelist
 */
export function useIpWhitelist() {
  return useQuery({
    queryKey: userKeys.ipWhitelist(),
    queryFn: async () => {
      const response = await apiService.user.getIpWhitelist();
      return response.data as {
        ip: string;
        description?: string;
        addedAt: string;
      }[];
    },
  });
}

/**
 * Add IP to whitelist
 */
export function useAddIpToWhitelist() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (data: { ip: string; description?: string }) => {
      await apiService.user.addIpToWhitelist(data.ip, data.description);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: userKeys.ipWhitelist() });
      message.success("IP added to whitelist");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to add IP"));
    },
  });
}

/**
 * Remove IP from whitelist
 */
export function useRemoveIpFromWhitelist() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (ip: string) => {
      await apiService.user.removeIpFromWhitelist(ip);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: userKeys.ipWhitelist() });
      message.success("IP removed from whitelist");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to remove IP"));
    },
  });
}

/**
 * Fetch login alert settings
 */
export function useLoginAlerts() {
  return useQuery({
    queryKey: userKeys.loginAlerts(),
    queryFn: async () => {
      const response = await apiService.user.getLoginAlerts();
      return response.data as {
        emailOnNewDevice: boolean;
        emailOnNewLocation: boolean;
        emailOnFailedAttempts: boolean;
      };
    },
  });
}

/**
 * Update login alert settings
 */
export function useUpdateLoginAlerts() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (
      data: Parameters<typeof apiService.user.updateLoginAlerts>[0]
    ) => {
      await apiService.user.updateLoginAlerts(data);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: userKeys.loginAlerts() });
      message.success("Login alerts updated");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to update login alerts"));
    },
  });
}

// ============================================
// Account Deletion Hooks (GDPR)
// ============================================

/**
 * Download personal data
 */
export function useDownloadPersonalData() {
  return useMutation({
    mutationFn: async () => {
      const response = await apiService.user.downloadPersonalData();
      return response.data;
    },
    onSuccess: (data) => {
      // Create download link
      const url = window.URL.createObjectURL(new Blob([data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "my-data.json");
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      message.success("Data download started");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to download data"));
    },
  });
}

/**
 * Request account deletion
 */
export function useRequestAccountDeletion() {
  return useMutation({
    mutationFn: async (password: string) => {
      await apiService.user.requestAccountDeletion(password);
    },
    onSuccess: () => {
      message.success(
        "Account deletion requested. Check your email to confirm."
      );
    },
    onError: (error: unknown) => {
      message.error(
        getErrorMessage(error, "Failed to request account deletion")
      );
    },
  });
}

/**
 * Cancel account deletion
 */
export function useCancelAccountDeletion() {
  return useMutation({
    mutationFn: async () => {
      await apiService.user.cancelAccountDeletion();
    },
    onSuccess: () => {
      message.success("Account deletion cancelled");
    },
    onError: (error: unknown) => {
      message.error(getErrorMessage(error, "Failed to cancel deletion"));
    },
  });
}
