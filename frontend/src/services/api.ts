/**
 * API Service / API服务
 *
 * Secure Axios instance with:
 * - httpOnly cookie-based authentication (no localStorage tokens)
 * - CSRF token protection for state-changing requests
 * - Automatic token refresh
 * - Rate limiting awareness
 *
 * 带有以下功能的安全Axios实例：
 * - 基于httpOnly cookie的认证（不使用localStorage存储令牌）
 * - 状态改变请求的CSRF令牌保护
 * - 自动令牌刷新
 * - 速率限制感知
 */

import axios, {
  AxiosInstance,
  AxiosError,
  InternalAxiosRequestConfig,
  AxiosResponse,
} from "axios";
import { useAuthStore } from "../store/authStore";
import { csrfManager, rateLimiter, sessionSecurity } from "./security";
import { errorLoggingService } from "./errorLogging";

/**
 * API Base URL / API基础URL
 *
 * Uses relative path '/api' so Vite proxy can route to backend.
 * 使用相对路径'/api'以便Vite代理可以路由到后端。
 */
const API_BASE_URL = import.meta.env.VITE_API_URL || "/api";

/**
 * State-changing HTTP methods that require CSRF protection
 */
const CSRF_PROTECTED_METHODS = ["post", "put", "patch", "delete"];

/**
 * Endpoints that are rate-limited with specific configs
 */
const RATE_LIMITED_ENDPOINTS: Record<
  string,
  { maxRequests: number; windowMs: number }
> = {
  "/auth/login": { maxRequests: 5, windowMs: 60000 },
  "/auth/register": { maxRequests: 3, windowMs: 60000 },
  "/auth/password/reset": { maxRequests: 3, windowMs: 300000 },
  "/auth/2fa/verify": { maxRequests: 5, windowMs: 60000 },
};

/**
 * Flag to prevent multiple simultaneous refresh attempts
 */
let isRefreshing = false;
let refreshSubscribers: Array<(success: boolean) => void> = [];

/**
 * Add subscriber to wait for token refresh
 */
function subscribeToRefresh(callback: (success: boolean) => void): void {
  refreshSubscribers.push(callback);
}

/**
 * Notify all subscribers about refresh result
 */
function notifyRefreshSubscribers(success: boolean): void {
  refreshSubscribers.forEach((callback) => callback(success));
  refreshSubscribers = [];
}

// Create axios instance with secure defaults
export const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
  },
  // CRITICAL: Enable cookies for httpOnly authentication
  withCredentials: true,
});

/**
 * Request Interceptor
 *
 * 1. Add CSRF token to state-changing requests
 * 2. Check client-side rate limits
 * 3. Update session activity
 */
api.interceptors.request.use(
  async (config: InternalAxiosRequestConfig) => {
    const method = config.method?.toLowerCase();
    const url = config.url || "";

    // Update session activity
    sessionSecurity.updateActivity();

    // Check rate limiting for specific endpoints
    for (const [endpoint, rateConfig] of Object.entries(
      RATE_LIMITED_ENDPOINTS
    )) {
      if (url.includes(endpoint)) {
        if (rateLimiter.shouldLimit(endpoint, rateConfig)) {
          const resetTime = rateLimiter.getResetTime(endpoint);
          const error = new Error("Too many requests. Please try again later.");
          (error as any).isRateLimited = true;
          (error as any).resetTime = resetTime;
          throw error;
        }
      }
    }

    // Add CSRF token to state-changing requests
    if (method && CSRF_PROTECTED_METHODS.includes(method)) {
      try {
        const csrfToken = await csrfManager.ensureToken();
        if (csrfToken && config.headers) {
          config.headers["X-CSRF-Token"] = csrfToken;
        }
      } catch (error) {
        // Continue without CSRF token - server will reject if required
        console.warn("Failed to get CSRF token:", error);
      }
    }

    // NOTE: No Authorization header needed - using httpOnly cookies
    // The browser automatically sends the cookie with each request

    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

/**
 * Response Interceptor
 *
 * 1. Extract CSRF token from response headers
 * 2. Handle 401 with automatic token refresh
 * 3. Handle 403 CSRF errors
 * 4. Handle 429 rate limit errors
 * 5. Log errors to error service
 */
api.interceptors.response.use(
  (response: AxiosResponse) => {
    // Extract CSRF token from response headers
    const newCsrfToken = response.headers["x-csrf-token"];
    if (newCsrfToken) {
      csrfManager.setToken(newCsrfToken);
    }

    return response;
  },
  async (error: AxiosError) => {
    const originalRequest = error.config as InternalAxiosRequestConfig & {
      _retry?: boolean;
      _retryCount?: number;
    };

    // Handle network errors
    if (!error.response) {
      errorLoggingService.logNetworkError(
        error,
        originalRequest?.url || "unknown",
        originalRequest?.method || "unknown"
      );
      throw error;
    }

    const status = error.response.status;
    const url = originalRequest?.url || "";

    // Handle 401 Unauthorized - Token expired
    if (status === 401 && !originalRequest._retry) {
      // Don't retry refresh or login endpoints
      if (url.includes("/auth/refresh") || url.includes("/auth/login")) {
        useAuthStore.getState().logout();
        csrfManager.clearToken();
        window.location.href = "/login";
        throw error;
      }

      originalRequest._retry = true;

      // If already refreshing, wait for it to complete
      if (isRefreshing) {
        return new Promise((resolve, reject) => {
          subscribeToRefresh((success) => {
            if (success) {
              resolve(api(originalRequest));
            } else {
              reject(error);
            }
          });
        });
      }

      isRefreshing = true;

      try {
        // Call refresh endpoint - server uses httpOnly cookie
        await api.post("/auth/refresh");

        isRefreshing = false;
        notifyRefreshSubscribers(true);

        // Retry original request
        return api(originalRequest);
      } catch (refreshError) {
        isRefreshing = false;
        notifyRefreshSubscribers(false);

        // Refresh failed - logout user
        useAuthStore.getState().logout();
        csrfManager.clearToken();
        sessionSecurity.stopInactivityTimer();

        // Redirect to login with return URL
        const returnUrl = encodeURIComponent(
          window.location.pathname + window.location.search
        );
        window.location.href = `/login?returnUrl=${returnUrl}`;

        throw refreshError;
      }
    }

    // Handle 403 Forbidden - Could be CSRF error
    if (status === 403) {
      const responseData = error.response.data as any;

      // If CSRF token invalid, fetch new one and retry
      if (
        responseData?.error?.includes("CSRF") ||
        responseData?.code === "CSRF_INVALID"
      ) {
        if (!originalRequest._retryCount || originalRequest._retryCount < 2) {
          originalRequest._retryCount = (originalRequest._retryCount || 0) + 1;

          try {
            await csrfManager.fetchToken();
            return api(originalRequest);
          } catch {
            // Continue to reject
          }
        }
      }
    }

    // Handle 429 Too Many Requests
    if (status === 429) {
      const retryAfter = error.response.headers["retry-after"];
      const resetTime = retryAfter
        ? Date.now() + parseInt(retryAfter) * 1000
        : Date.now() + 60000;

      (error as any).retryAfter = retryAfter;
      (error as any).resetTime = resetTime;
    }

    // Log error
    errorLoggingService.logNetworkError(
      error,
      url,
      originalRequest?.method || "unknown",
      status
    );

    throw error;
  }
);

// API methods
export const apiService = {
  // Auth
  auth: {
    /**
     * Login with email and password
     * Server returns JWT in httpOnly cookie
     */
    login: (email: string, password: string, invitation_code?: string) =>
      api.post("/auth/login", { email, password, invitation_code }),

    /**
     * Register new account
     */
    register: (data: {
      email: string;
      password: string;
      name: string;
      invitation_code: string;
    }) => api.post("/auth/register", data),

    /**
     * Logout - clears httpOnly cookies
     */
    logout: () => api.post("/auth/logout"),

    /**
     * Refresh token - uses httpOnly cookie
     * Server sets new JWT in httpOnly cookie
     */
    refresh: () => api.post("/auth/refresh"),

    /**
     * Get current user
     */
    me: () => api.get("/auth/me"),

    /**
     * Update user profile
     */
    updateProfile: (data: { name?: string; avatar?: string }) =>
      api.put("/auth/profile", data),

    /**
     * Change password
     */
    changePassword: (oldPassword: string, newPassword: string) =>
      api.post("/auth/change-password", {
        old_password: oldPassword,
        new_password: newPassword,
      }),

    /**
     * Verify 2FA code during login
     */
    verify2FA: (code: string, isBackupCode?: boolean) =>
      api.post("/auth/2fa/verify", { code, is_backup_code: isBackupCode }),

    /**
     * Resend 2FA code (if using SMS/email)
     */
    resend2FA: () => api.post("/auth/2fa/resend"),
  },

  // Projects
  projects: {
    /**
     * List all projects with pagination and filters
     */
    list: (params?: {
      page?: number;
      limit?: number;
      search?: string;
      status?: string;
      language?: string;
      sort_field?: string;
      sort_order?: string;
    }) => api.get("/projects", { params }),

    /**
     * Get project by ID
     */
    get: (id: string) => api.get(`/projects/${id}`),

    /**
     * Create a new project
     */
    create: (data: {
      name: string;
      description?: string;
      repository_url?: string;
      language: string;
      framework?: string;
      settings?: unknown;
    }) => api.post("/projects", data),

    /**
     * Update project settings
     */
    update: (
      id: string,
      data: Partial<{
        name: string;
        description: string;
        status: string;
        settings: unknown;
      }>
    ) => api.put(`/projects/${id}`, data),

    /**
     * Delete a project
     */
    delete: (id: string) => api.delete(`/projects/${id}`),

    /**
     * Archive a project
     */
    archive: (id: string) => api.post(`/projects/${id}/archive`),

    /**
     * Restore an archived project
     */
    restore: (id: string) => api.post(`/projects/${id}/restore`),

    /**
     * Get project statistics
     */
    getStats: (id: string) => api.get(`/projects/${id}/stats`),

    // Files
    getFiles: (id: string, path?: string) =>
      api.get(`/projects/${id}/files`, { params: { path } }),

    getFile: (id: string, path: string) =>
      api.get(`/projects/${id}/files/${encodeURIComponent(path)}`),

    updateFile: (id: string, path: string, content: string) =>
      api.put(`/projects/${id}/files/${encodeURIComponent(path)}`, { content }),

    // Activity
    /**
     * Get project activity log
     */
    getActivity: (id: string, params?: { page?: number; limit?: number }) =>
      api.get(`/projects/${id}/activity`, { params }),

    // Team Management
    /**
     * Get project team members
     */
    getTeam: (id: string) => api.get(`/projects/${id}/team`),

    /**
     * Invite a team member
     */
    inviteMember: (id: string, data: { email: string; role: string }) =>
      api.post(`/projects/${id}/team/invite`, data),

    /**
     * Update member role
     */
    updateMemberRole: (id: string, memberId: string, role: string) =>
      api.put(`/projects/${id}/team/${memberId}`, { role }),

    /**
     * Remove team member
     */
    removeMember: (id: string, memberId: string) =>
      api.delete(`/projects/${id}/team/${memberId}`),

    // Webhooks
    /**
     * Get project webhooks
     */
    getWebhooks: (id: string) => api.get(`/projects/${id}/webhooks`),

    /**
     * Create webhook
     */
    createWebhook: (
      id: string,
      data: { url: string; events: string[]; secret?: string }
    ) => api.post(`/projects/${id}/webhooks`, data),

    /**
     * Update webhook
     */
    updateWebhook: (
      id: string,
      webhookId: string,
      data: { url?: string; events?: string[]; is_active?: boolean }
    ) => api.put(`/projects/${id}/webhooks/${webhookId}`, data),

    /**
     * Delete webhook
     */
    deleteWebhook: (id: string, webhookId: string) =>
      api.delete(`/projects/${id}/webhooks/${webhookId}`),

    /**
     * Test webhook
     */
    testWebhook: (id: string, webhookId: string) =>
      api.post(`/projects/${id}/webhooks/${webhookId}/test`),

    // API Keys
    /**
     * Get project API keys
     */
    getApiKeys: (id: string) => api.get(`/projects/${id}/api-keys`),

    /**
     * Create API key
     */
    createApiKey: (
      id: string,
      data: { name: string; permissions: string[]; expires_at?: string }
    ) => api.post(`/projects/${id}/api-keys`, data),

    /**
     * Revoke API key
     */
    revokeApiKey: (id: string, keyId: string) =>
      api.delete(`/projects/${id}/api-keys/${keyId}`),

    /**
     * Get API key usage
     */
    getApiKeyUsage: (
      id: string,
      keyId: string,
      params?: { start_date?: string; end_date?: string }
    ) => api.get(`/projects/${id}/api-keys/${keyId}/usage`, { params }),
  },

  // ============================================
  // OAuth
  // ============================================
  oauth: {
    /**
     * Get available OAuth providers
     */
    getProviders: () => api.get("/auth/oauth/providers"),

    /**
     * Initiate OAuth flow
     */
    connect: (provider: string, returnUrl?: string) =>
      api.get(`/auth/oauth/connect/${provider}`, {
        params: { return_url: returnUrl },
      }),

    /**
     * Get connected OAuth accounts
     */
    getConnections: () => api.get("/auth/oauth/connections"),

    /**
     * Disconnect OAuth provider
     */
    disconnect: (provider: string) =>
      api.delete(`/auth/oauth/connections/${provider}`),

    /**
     * List repositories from OAuth provider
     */
    listRepositories: (provider: string) =>
      api.get(`/repositories/oauth/${provider}`),
  },

  // ============================================
  // Repositories
  // ============================================
  repositories: {
    /**
     * List all repositories
     */
    list: (params?: {
      page?: number;
      limit?: number;
      project_id?: string;
      provider?: string;
      status?: string;
    }) => api.get("/repositories", { params }),

    /**
     * Get repository by ID
     */
    get: (id: string) => api.get(`/repositories/${id}`),

    /**
     * Create repository from URL
     */
    create: (data: { url: string; name?: string; project_id?: string }) =>
      api.post("/repositories", data),

    /**
     * Connect repository from OAuth provider
     */
    connect: (data: {
      provider: string;
      repo_full_name: string;
      project_id?: string;
      default_branch?: string;
    }) => api.post("/repositories/connect", data),

    /**
     * Delete repository
     */
    delete: (id: string) => api.delete(`/repositories/${id}`),

    /**
     * Sync repository with remote
     */
    sync: (id: string) => api.post(`/repositories/${id}/sync`),

    /**
     * Get repository file tree
     */
    getTree: (id: string, path?: string) =>
      api.get(`/repositories/${id}/tree`, { params: { path } }),

    /**
     * Get file content from repository
     */
    getFile: (id: string, filePath: string) =>
      api.get(`/repositories/${id}/files/${encodeURIComponent(filePath)}`),

    /**
     * Create webhook for repository
     */
    createWebhook: (id: string) => api.post(`/repositories/${id}/webhook`),

    /**
     * Get repository branches
     */
    getBranches: (id: string) => api.get(`/repositories/${id}/branches`),

    /**
     * Get repository commits
     */
    getCommits: (
      id: string,
      params?: { branch?: string; page?: number; limit?: number }
    ) => api.get(`/repositories/${id}/commits`, { params }),
  },

  // Analysis
  analysis: {
    start: (
      projectId: string,
      options?: { files?: string[]; version?: string }
    ) => api.post(`/projects/${projectId}/analyze`, options),

    getSession: (sessionId: string) => api.get(`/analyze/${sessionId}`),

    getSessions: (
      projectId: string,
      params?: { page?: number; limit?: number }
    ) => api.get(`/projects/${projectId}/sessions`, { params }),

    getIssues: (
      sessionId: string,
      params?: { severity?: string; type?: string }
    ) => api.get(`/analyze/${sessionId}/issues`, { params }),

    applyFix: (sessionId: string, issueId: string) =>
      api.post(`/analyze/${sessionId}/issues/${issueId}/fix`),

    dismissIssue: (sessionId: string, issueId: string, reason?: string) =>
      api.post(`/analyze/${sessionId}/issues/${issueId}/dismiss`, { reason }),

    streamUrl: (sessionId: string) =>
      `${API_BASE_URL}/analyze/${sessionId}/stream`,

    // Security vulnerabilities
    getVulnerabilities: (params?: {
      severity?: string;
      status?: string;
      project?: string;
      page?: number;
      limit?: number;
    }) => api.get("/security/vulnerabilities", { params }),

    getSecurityMetrics: () => api.get("/security/metrics"),

    getComplianceStatus: () => api.get("/security/compliance"),
  },

  // Experiments (Admin)
  experiments: {
    list: (params?: { status?: string; page?: number; limit?: number }) =>
      api.get("/experiments", { params }),

    get: (id: string) => api.get(`/experiments/${id}`),

    create: (data: { name: string; config: unknown; dataset_id: string }) =>
      api.post("/experiments", data),

    start: (id: string) => api.post(`/experiments/${id}/start`),

    stop: (id: string) => api.post(`/experiments/${id}/stop`),

    evaluate: (id: string) => api.post(`/experiments/${id}/evaluate`),

    promote: (id: string) => api.post(`/experiments/${id}/promote`),

    quarantine: (id: string, reason: string) =>
      api.post(`/experiments/${id}/quarantine`, { reason }),

    getMetrics: (id: string) => api.get(`/experiments/${id}/metrics`),

    compare: (id1: string, id2: string) =>
      api.get("/experiments/compare", {
        params: { experiment_a: id1, experiment_b: id2 },
      }),
  },

  // Versions (Admin)
  versions: {
    list: () => api.get("/versions"),

    getCurrent: () => api.get("/versions/current"),

    getHistory: (params?: { page?: number; limit?: number }) =>
      api.get("/versions/history", { params }),

    rollback: (versionId: string) =>
      api.post(`/versions/${versionId}/rollback`),
  },

  // Audit (Admin)
  audit: {
    list: (params?: {
      entity?: string;
      action?: string;
      user_id?: string;
      start_date?: string;
      end_date?: string;
      page?: number;
      limit?: number;
    }) => api.get("/audit", { params }),

    get: (id: string) => api.get(`/audit/${id}`),
  },

  // Metrics
  metrics: {
    getDashboard: () => api.get("/metrics/dashboard"),

    getSystem: () => api.get("/metrics/system"),

    getProvider: (provider: string) =>
      api.get(`/metrics/providers/${provider}`),

    getUsage: (params?: { start_date?: string; end_date?: string }) =>
      api.get("/metrics/usage", { params }),
  },

  // User Profile & Settings
  user: {
    // Profile
    /**
     * Get current user profile
     */
    getProfile: () => api.get("/user/profile"),

    /**
     * Update user profile
     */
    updateProfile: (data: { name?: string; username?: string; bio?: string }) =>
      api.put("/user/profile", data),

    /**
     * Upload avatar
     */
    uploadAvatar: (file: File) => {
      const formData = new FormData();
      formData.append("avatar", file);
      return api.post("/user/avatar", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
    },

    /**
     * Delete avatar
     */
    deleteAvatar: () => api.delete("/user/avatar"),

    // Settings
    /**
     * Get user settings
     */
    getSettings: () => api.get("/user/settings"),

    /**
     * Update user settings
     */
    updateSettings: (data: Record<string, unknown>) =>
      api.put("/user/settings", data),

    /**
     * Update privacy settings
     */
    updatePrivacy: (data: {
      profileVisibility?: "public" | "private" | "connections";
      showEmail?: boolean;
      showActivity?: boolean;
      showProjects?: boolean;
      showStatistics?: boolean;
      allowDataSharing?: boolean;
      allowAnalytics?: boolean;
    }) => api.put("/user/settings/privacy", data),

    /**
     * Update notification preferences
     */
    updateNotifications: (data: Record<string, unknown>) =>
      api.put("/user/settings/notifications", data),

    // OAuth / Account Linking
    /**
     * Get available OAuth providers
     */
    getOAuthProviders: () => api.get("/user/oauth/providers"),

    /**
     * Get connected OAuth accounts
     */
    getOAuthConnections: () => api.get("/user/oauth/connections"),

    /**
     * Initiate OAuth connection
     */
    connectOAuth: (provider: string) =>
      api.post(`/user/oauth/connect/${provider}`),

    /**
     * Disconnect OAuth provider
     */
    disconnectOAuth: (provider: string) =>
      api.delete(`/user/oauth/${provider}`),

    // Password
    /**
     * Change password
     */
    changePassword: (data: { currentPassword: string; newPassword: string }) =>
      api.post("/user/password/change", data),

    /**
     * Request password reset email
     */
    requestPasswordReset: (email: string) =>
      api.post("/user/password/reset-request", { email }),

    /**
     * Reset password with token
     */
    resetPassword: (token: string, newPassword: string) =>
      api.post("/user/password/reset", { token, new_password: newPassword }),

    // Email
    /**
     * Change email address (initiates verification)
     */
    changeEmail: (newEmail: string, password: string) =>
      api.post("/user/email/change", { new_email: newEmail, password }),

    /**
     * Verify email with token
     */
    verifyEmail: (token: string) => api.post("/user/email/verify", { token }),

    /**
     * Resend verification email
     */
    resendVerification: () => api.post("/user/email/resend-verification"),

    // Two-Factor Authentication
    /**
     * Get 2FA status
     */
    get2FAStatus: () => api.get("/user/2fa/status"),

    /**
     * Begin 2FA setup (get QR code)
     */
    setup2FA: () => api.post("/user/2fa/setup"),

    /**
     * Verify and enable 2FA
     */
    enable2FA: (code: string) => api.post("/user/2fa/enable", { code }),

    /**
     * Disable 2FA
     */
    disable2FA: (code: string, password: string) =>
      api.post("/user/2fa/disable", { code, password }),

    /**
     * Get backup codes
     */
    getBackupCodes: () => api.get("/user/2fa/backup-codes"),

    /**
     * Regenerate backup codes
     */
    regenerateBackupCodes: (password: string) =>
      api.post("/user/2fa/backup-codes/regenerate", { password }),

    // Sessions
    /**
     * Get all active sessions
     */
    getSessions: () => api.get("/user/sessions"),

    /**
     * Revoke a specific session
     */
    revokeSession: (sessionId: string) =>
      api.delete(`/user/sessions/${sessionId}`),

    /**
     * Revoke all sessions except current
     */
    revokeAllSessions: () => api.post("/user/sessions/revoke-all"),

    // Login History
    /**
     * Get login history
     */
    getLoginHistory: (params?: { page?: number; limit?: number }) =>
      api.get("/user/login-history", { params }),

    // API Activity
    /**
     * Get API activity log
     */
    getApiActivity: (params?: { page?: number; limit?: number }) =>
      api.get("/user/api-activity", { params }),

    // API Keys
    /**
     * Get user API keys
     */
    getApiKeys: () => api.get("/user/api-keys"),

    /**
     * Create API key
     */
    createApiKey: (data: {
      name: string;
      permissions: string[];
      expiresAt?: string;
    }) => api.post("/user/api-keys", data),

    /**
     * Revoke API key
     */
    revokeApiKey: (keyId: string) => api.delete(`/user/api-keys/${keyId}`),

    /**
     * Get API key usage statistics
     */
    getApiKeyUsage: (
      keyId: string,
      params?: { start_date?: string; end_date?: string }
    ) => api.get(`/user/api-keys/${keyId}/usage`, { params }),

    // Integrations
    /**
     * Get connected integrations
     */
    getIntegrations: () => api.get("/user/integrations"),

    /**
     * Connect Slack workspace
     */
    connectSlack: (code: string) =>
      api.post("/user/integrations/slack/connect", { code }),

    /**
     * Disconnect Slack
     */
    disconnectSlack: () => api.delete("/user/integrations/slack"),

    /**
     * Connect Microsoft Teams
     */
    connectTeams: (code: string) =>
      api.post("/user/integrations/teams/connect", { code }),

    /**
     * Disconnect Teams
     */
    disconnectTeams: () => api.delete("/user/integrations/teams"),

    // Webhooks
    /**
     * Get user webhooks
     */
    getWebhooks: () => api.get("/user/webhooks"),

    /**
     * Create webhook
     */
    createWebhook: (data: {
      name: string;
      url: string;
      events: string[];
      secret?: string;
    }) => api.post("/user/webhooks", data),

    /**
     * Update webhook
     */
    updateWebhook: (
      webhookId: string,
      data: {
        name?: string;
        url?: string;
        events?: string[];
        isActive?: boolean;
      }
    ) => api.put(`/user/webhooks/${webhookId}`, data),

    /**
     * Delete webhook
     */
    deleteWebhook: (webhookId: string) =>
      api.delete(`/user/webhooks/${webhookId}`),

    /**
     * Test webhook
     */
    testWebhook: (webhookId: string) =>
      api.post(`/user/webhooks/${webhookId}/test`),

    /**
     * Get webhook delivery logs
     */
    getWebhookLogs: (
      webhookId: string,
      params?: { page?: number; limit?: number }
    ) => api.get(`/user/webhooks/${webhookId}/logs`, { params }),

    // Data & Privacy (GDPR)
    /**
     * Download personal data
     */
    downloadPersonalData: () =>
      api.get("/user/data/export", { responseType: "blob" }),

    /**
     * Request account deletion
     */
    requestAccountDeletion: (password: string) =>
      api.post("/user/account/delete-request", { password }),

    /**
     * Confirm account deletion
     */
    confirmAccountDeletion: (token: string) =>
      api.post("/user/account/delete-confirm", { token }),

    /**
     * Cancel account deletion
     */
    cancelAccountDeletion: () => api.post("/user/account/delete-cancel"),

    // IP Whitelist (Admin/sensitive users)
    /**
     * Get IP whitelist
     */
    getIpWhitelist: () => api.get("/user/security/ip-whitelist"),

    /**
     * Add IP to whitelist
     */
    addIpToWhitelist: (ip: string, description?: string) =>
      api.post("/user/security/ip-whitelist", { ip, description }),

    /**
     * Remove IP from whitelist
     */
    removeIpFromWhitelist: (ip: string) =>
      api.delete(`/user/security/ip-whitelist/${encodeURIComponent(ip)}`),

    // Login Alerts
    /**
     * Get login alert settings
     */
    getLoginAlerts: () => api.get("/user/security/login-alerts"),

    /**
     * Update login alert settings
     */
    updateLoginAlerts: (data: {
      emailOnNewDevice?: boolean;
      emailOnNewLocation?: boolean;
      emailOnFailedAttempts?: boolean;
    }) => api.put("/user/security/login-alerts", data),
  },

  // Admin APIs
  admin: {
    // ============================================
    // User Management
    // ============================================
    users: {
      /**
       * List all users with filters and pagination
       */
      list: (params?: {
        page?: number;
        limit?: number;
        search?: string;
        role?: string;
        status?: string;
        sort_field?: string;
        sort_order?: string;
        start_date?: string;
        end_date?: string;
      }) => api.get("/admin/users", { params }),

      /**
       * Get user by ID
       */
      get: (userId: string) => api.get(`/admin/users/${userId}`),

      /**
       * Update user information
       */
      update: (
        userId: string,
        data: {
          email?: string;
          name?: string;
          role?: string;
          status?: string;
        }
      ) => api.put(`/admin/users/${userId}`, data),

      /**
       * Delete user
       */
      delete: (userId: string) => api.delete(`/admin/users/${userId}`),

      /**
       * Suspend user
       */
      suspend: (userId: string, reason?: string) =>
        api.post(`/admin/users/${userId}/suspend`, { reason }),

      /**
       * Reactivate user
       */
      reactivate: (userId: string) =>
        api.post(`/admin/users/${userId}/reactivate`),

      /**
       * Reset user password
       */
      resetPassword: (userId: string) =>
        api.post(`/admin/users/${userId}/reset-password`),

      /**
       * Resend welcome email
       */
      resendWelcome: (userId: string) =>
        api.post(`/admin/users/${userId}/resend-welcome`),

      /**
       * Bulk user operations
       */
      bulk: (data: {
        userIds: string[];
        action: "change_role" | "suspend" | "reactivate" | "delete";
        role?: string;
      }) => api.post("/admin/users/bulk", data),

      /**
       * Import users from CSV
       */
      import: (file: File) => {
        const formData = new FormData();
        formData.append("file", file);
        return api.post("/admin/users/import", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
      },

      /**
       * Export users to CSV
       */
      export: (params?: {
        role?: string;
        status?: string;
        format?: "csv" | "json";
      }) => api.get("/admin/users/export", { params, responseType: "blob" }),

      /**
       * Get user statistics
       */
      getStats: () => api.get("/admin/users/stats"),
    },

    // ============================================
    // Provider Management
    // ============================================
    providers: {
      /**
       * List all providers
       */
      list: () => api.get("/admin/providers"),

      /**
       * Get provider by ID
       */
      get: (providerId: string) => api.get(`/admin/providers/${providerId}`),

      /**
       * Update provider configuration
       */
      update: (
        providerId: string,
        data: {
          name?: string;
          status?: string;
          priority?: number;
          isDefault?: boolean;
          config?: Record<string, unknown>;
        }
      ) => api.put(`/admin/providers/${providerId}`, data),

      /**
       * Update provider API key
       */
      updateApiKey: (providerId: string, apiKey: string) =>
        api.put(`/admin/providers/${providerId}/api-key`, { api_key: apiKey }),

      /**
       * Test provider connectivity
       */
      test: (providerId: string) =>
        api.post(`/admin/providers/${providerId}/test`),

      /**
       * Get provider health status
       */
      getHealth: (providerId: string) =>
        api.get(`/admin/providers/${providerId}/health`),

      /**
       * Get provider metrics
       */
      getMetrics: (
        providerId: string,
        params?: {
          period?: "hour" | "day" | "week" | "month";
        }
      ) => api.get(`/admin/providers/${providerId}/metrics`, { params }),

      /**
       * Get provider models
       */
      getModels: (providerId: string) =>
        api.get(`/admin/providers/${providerId}/models`),

      /**
       * Update model configuration
       */
      updateModel: (
        providerId: string,
        modelId: string,
        data: {
          isEnabled?: boolean;
          isDefault?: boolean;
          costPerInputToken?: number;
          costPerOutputToken?: number;
        }
      ) => api.put(`/admin/providers/${providerId}/models/${modelId}`, data),

      /**
       * Set fallback order
       */
      setFallbackOrder: (providerIds: string[]) =>
        api.put("/admin/providers/fallback-order", {
          provider_ids: providerIds,
        }),
    },

    // ============================================
    // Audit Logs
    // ============================================
    audit: {
      /**
       * Get audit logs with filters
       */
      list: (params?: {
        page?: number;
        limit?: number;
        search?: string;
        action?: string;
        resource?: string;
        status?: string;
        user_id?: string;
        start_date?: string;
        end_date?: string;
        sort_field?: string;
        sort_order?: string;
      }) => api.get("/admin/audit/logs", { params }),

      /**
       * Get single audit log entry
       */
      get: (logId: string) => api.get(`/admin/audit/logs/${logId}`),

      /**
       * Export audit logs
       */
      export: (params?: {
        format?: "csv" | "json" | "pdf";
        action?: string;
        resource?: string;
        status?: string;
        start_date?: string;
        end_date?: string;
      }) => api.get("/admin/audit/export", { params, responseType: "blob" }),

      /**
       * Get audit analytics
       */
      getAnalytics: (params?: { period?: "day" | "week" | "month" }) =>
        api.get("/admin/audit/analytics", { params }),

      /**
       * Get security alerts
       */
      getAlerts: (params?: {
        severity?: string;
        resolved?: boolean;
        page?: number;
        limit?: number;
      }) => api.get("/admin/audit/alerts", { params }),

      /**
       * Resolve security alert
       */
      resolveAlert: (alertId: string, notes?: string) =>
        api.post(`/admin/audit/alerts/${alertId}/resolve`, { notes }),

      /**
       * Get login patterns
       */
      getLoginPatterns: (params?: { period?: "day" | "week" | "month" }) =>
        api.get("/admin/audit/login-patterns", { params }),
    },

    // ============================================
    // System Settings
    // ============================================
    settings: {
      /**
       * Get system settings
       */
      get: () => api.get("/admin/settings"),

      /**
       * Update system settings
       */
      update: (settings: Record<string, unknown>) =>
        api.put("/admin/settings", settings),

      /**
       * Get system health
       */
      getHealth: () => api.get("/admin/health"),

      /**
       * Trigger system backup
       */
      backup: () => api.post("/admin/backup"),

      /**
       * Get backup history
       */
      getBackups: () => api.get("/admin/backups"),
    },
  },
};

export default api;
