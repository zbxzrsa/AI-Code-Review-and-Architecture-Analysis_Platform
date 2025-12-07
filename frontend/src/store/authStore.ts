/**
 * 认证状态存储 (Auth Store)
 *
 * 功能描述:
 *   使用 Zustand 管理用户认证状态。
 *
 * 安全说明:
 *   - 令牌不存储在 localStorage（易受 XSS 攻击）
 *   - 认证通过服务器设置的 httpOnly cookies 处理
 *   - 仅持久化非敏感用户数据以改善用户体验
 *   - CSRF 令牌通过安全服务存储在内存中
 *
 * 主要类型:
 *   - User: 用户资料
 *   - Session: 会话信息
 *   - OAuthConnection: OAuth 连接状态
 *
 * 最后修改日期: 2024-12-07
 */

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

/** User role types */
export type UserRole = "admin" | "user" | "viewer" | "guest";

/** OAuth provider types */
export type OAuthProvider = "github" | "gitlab" | "google" | "microsoft";

/**
 * User profile entity
 */
export interface User {
  id: string;
  email: string;
  name: string;
  username?: string;
  bio?: string;
  role: UserRole;
  avatar?: string;
  createdAt?: string;
  lastLoginAt?: string;
  emailVerified?: boolean;
  twoFactorEnabled?: boolean;
}

/**
 * OAuth connection status
 */
export interface OAuthConnection {
  provider: OAuthProvider;
  connected: boolean;
  connectedAt?: string;
  username?: string;
  email?: string;
  avatarUrl?: string;
  scopes?: string[];
}

/**
 * Active session information
 */
export interface Session {
  id: string;
  device: string;
  browser: string;
  os: string;
  ip: string;
  location?: string;
  lastActive: string;
  createdAt: string;
  isCurrent: boolean;
}

/**
 * Login history entry
 */
export interface LoginHistory {
  id: string;
  timestamp: string;
  ip: string;
  location?: string;
  device: string;
  browser: string;
  success: boolean;
  failureReason?: string;
}

/**
 * API activity entry
 */
export interface ApiActivity {
  id: string;
  endpoint: string;
  method: string;
  timestamp: string;
  statusCode: number;
  responseTime: number;
  apiKeyId?: string;
  apiKeyName?: string;
}

/**
 * User privacy settings
 */
export interface PrivacySettings {
  profileVisibility: "public" | "private" | "connections";
  showEmail: boolean;
  showActivity: boolean;
  showProjects: boolean;
  showStatistics: boolean;
  allowDataSharing: boolean;
  allowAnalytics: boolean;
}

/**
 * Notification preferences
 */
export interface NotificationPreferences {
  // Email notifications
  emailAnalysisComplete: boolean;
  emailNewIssues: boolean;
  emailWeeklyDigest: boolean;
  emailSecurityAlerts: boolean;
  emailProductUpdates: boolean;

  // Notification frequency
  digestFrequency: "immediate" | "daily" | "weekly" | "none";

  // Channels
  inAppNotifications: boolean;
  desktopNotifications: boolean;
  slackNotifications: boolean;
  teamsNotifications: boolean;

  // Do Not Disturb
  dndEnabled: boolean;
  dndStart?: string; // HH:MM format
  dndEnd?: string; // HH:MM format
  dndDays?: number[]; // 0-6, Sunday-Saturday
}

/**
 * User preferences/settings
 */
export interface UserSettings {
  // Appearance
  theme: "light" | "dark" | "system";
  language: string;

  // Editor
  editorFontSize: number;
  editorTabSize: number;
  editorLineNumbers: boolean;
  editorMinimap: boolean;
  editorWordWrap: boolean;

  // Analysis defaults
  defaultAiModel: string;
  defaultAnalysisDepth: "quick" | "standard" | "deep";
  autoAnalyzeOnPush: boolean;

  // Privacy
  privacy: PrivacySettings;

  // Notifications
  notifications: NotificationPreferences;
}

/**
 * 2FA setup data
 */
export interface TwoFactorSetup {
  secret: string;
  qrCode: string;
  backupCodes: string[];
}

/**
 * Integration connection
 */
export interface Integration {
  id: string;
  type: "slack" | "teams" | "webhook";
  name: string;
  connected: boolean;
  connectedAt?: string;
  config?: Record<string, unknown>;
  lastSync?: string;
  status: "active" | "error" | "disconnected";
}

/**
 * Webhook configuration
 */
export interface UserWebhook {
  id: string;
  name: string;
  url: string;
  events: string[];
  secret?: string;
  isActive: boolean;
  createdAt: string;
  lastTriggered?: string;
  lastStatus?: "success" | "failure";
}

/**
 * API key for user
 */
export interface UserApiKey {
  id: string;
  name: string;
  prefix: string;
  permissions: string[];
  createdAt: string;
  lastUsedAt?: string;
  expiresAt?: string;
  usageCount: number;
}

/**
 * Two-factor authentication state
 */
export interface TwoFactorState {
  required: boolean;
  verified: boolean;
  setupComplete: boolean;
}

/**
 * Auth State Interface
 *
 * NOTE: No token/refreshToken fields - these are stored in httpOnly cookies
 */
interface AuthState {
  // User data
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;

  // 2FA state
  twoFactor: TwoFactorState;

  // User settings (cached)
  settings: UserSettings | null;

  // Permissions (derived from user role)
  permissions: string[];

  // Actions
  setUser: (user: User | null) => void;
  updateUser: (updates: Partial<User>) => void;
  setAuthenticated: (authenticated: boolean) => void;
  setTwoFactorState: (state: Partial<TwoFactorState>) => void;
  setSettings: (settings: UserSettings | null) => void;
  updateSettings: (updates: Partial<UserSettings>) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  logout: () => void;

  // Deprecated - kept for backward compatibility, but tokens are in httpOnly cookies
  /** @deprecated Use httpOnly cookies instead */
  token: string | null;
  /** @deprecated Use httpOnly cookies instead */
  refreshToken: string | null;
  /** @deprecated Tokens are now in httpOnly cookies */
  setTokens: (token: string, refreshToken: string) => void;
}

/** Default notification preferences */
export const defaultNotificationPreferences: NotificationPreferences = {
  emailAnalysisComplete: true,
  emailNewIssues: true,
  emailWeeklyDigest: false,
  emailSecurityAlerts: true,
  emailProductUpdates: false,
  digestFrequency: "immediate",
  inAppNotifications: true,
  desktopNotifications: true,
  slackNotifications: false,
  teamsNotifications: false,
  dndEnabled: false,
};

/** Default privacy settings */
export const defaultPrivacySettings: PrivacySettings = {
  profileVisibility: "public",
  showEmail: false,
  showActivity: true,
  showProjects: true,
  showStatistics: true,
  allowDataSharing: false,
  allowAnalytics: true,
};

/** Default user settings */
export const defaultUserSettings: UserSettings = {
  theme: "system",
  language: "en",
  editorFontSize: 14,
  editorTabSize: 2,
  editorLineNumbers: true,
  editorMinimap: true,
  editorWordWrap: true,
  defaultAiModel: "gpt-4",
  defaultAnalysisDepth: "standard",
  autoAnalyzeOnPush: false,
  privacy: defaultPrivacySettings,
  notifications: defaultNotificationPreferences,
};

/**
 * Get permissions based on user role
 */
function getPermissionsForRole(role: UserRole): string[] {
  const permissions: Record<UserRole, string[]> = {
    admin: ["admin:all", "read:*", "write:*", "delete:*"],
    user: [
      "read:projects",
      "write:projects",
      "read:analyses",
      "write:analyses",
    ],
    viewer: ["read:projects", "read:analyses"],
    guest: [],
  };
  return permissions[role] || [];
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, _get) => ({
      // State
      user: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,
      settings: null,
      permissions: [],

      // 2FA state
      twoFactor: {
        required: false,
        verified: false,
        setupComplete: false,
      },

      // Deprecated token fields (kept for backward compatibility)
      token: null,
      refreshToken: null,

      // Actions
      setUser: (user) =>
        set({
          user,
          isAuthenticated: !!user,
          permissions: user ? getPermissionsForRole(user.role) : [],
        }),

      updateUser: (updates) =>
        set((state) => {
          const updatedUser = state.user ? { ...state.user, ...updates } : null;
          return {
            user: updatedUser,
            permissions: updatedUser
              ? getPermissionsForRole(updatedUser.role)
              : [],
          };
        }),

      setAuthenticated: (authenticated) =>
        set({ isAuthenticated: authenticated }),

      setTwoFactorState: (twoFactorUpdates) =>
        set((state) => ({
          twoFactor: { ...state.twoFactor, ...twoFactorUpdates },
        })),

      /** @deprecated - Tokens are now stored in httpOnly cookies */
      setTokens: (_token, _refreshToken) => {
        // No-op: Tokens are now handled via httpOnly cookies
        // This method is kept for backward compatibility
        console.warn(
          "setTokens is deprecated. Tokens are now stored in httpOnly cookies."
        );
        set({ isAuthenticated: true });
      },

      setSettings: (settings) => set({ settings }),

      updateSettings: (updates) =>
        set((state) => ({
          settings: state.settings
            ? { ...state.settings, ...updates }
            : { ...defaultUserSettings, ...updates },
        })),

      setLoading: (isLoading) => set({ isLoading }),

      setError: (error) => set({ error }),

      logout: () => {
        // Clear CSRF token from security service
        import("../services/security")
          .then(({ csrfManager, sessionSecurity }) => {
            csrfManager.clearToken();
            sessionSecurity.stopInactivityTimer();
          })
          .catch(() => {});

        set({
          user: null,
          isAuthenticated: false,
          error: null,
          settings: null,
          permissions: [],
          twoFactor: {
            required: false,
            verified: false,
            setupComplete: false,
          },
          // Deprecated
          token: null,
          refreshToken: null,
        });
      },
    }),
    {
      name: "auth-storage",
      storage: createJSONStorage(() => localStorage),
      // SECURITY: Only persist non-sensitive user data for UX
      // Tokens are NOT stored in localStorage - they're in httpOnly cookies
      partialize: (state) => ({
        // Only persist user info for displaying name/avatar on page load
        user: state.user
          ? {
              id: state.user.id,
              email: state.user.email,
              name: state.user.name,
              avatar: state.user.avatar,
              role: state.user.role,
            }
          : null,
        // Don't persist:
        // - tokens (httpOnly cookies)
        // - isAuthenticated (will be verified on mount)
        // - settings (fetched from server)
        // - permissions (derived from role)
      }),
    }
  )
);

export default useAuthStore;
