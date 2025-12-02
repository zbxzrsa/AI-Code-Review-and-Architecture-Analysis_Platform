/**
 * Admin Store
 * 
 * State management for admin dashboard including users, providers, and audit.
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

// ============================================
// User Management Types
// ============================================

export type UserRole = 'admin' | 'analyst' | 'viewer';
export type UserStatus = 'active' | 'inactive' | 'suspended';

export interface AdminUser {
  id: string;
  username: string;
  email: string;
  name: string;
  role: UserRole;
  status: UserStatus;
  avatar?: string;
  joinedAt: string;
  lastLoginAt?: string;
  emailVerified: boolean;
  twoFactorEnabled: boolean;
  projectCount: number;
  analysisCount: number;
}

export interface UserFilters {
  search: string;
  role: UserRole | 'all';
  status: UserStatus | 'all';
  dateRange?: [string, string];
}

export interface UserStats {
  total: number;
  active: number;
  inactive: number;
  suspended: number;
  recentlyJoined: number;
  byRole: { role: UserRole; count: number }[];
  activityTrend: { date: string; count: number }[];
}

// ============================================
// Provider Management Types
// ============================================

export type ProviderType = 'openai' | 'anthropic' | 'google' | 'azure' | 'huggingface' | 'local';
export type ProviderStatus = 'active' | 'inactive' | 'error' | 'rate_limited';

export interface AIProvider {
  id: string;
  type: ProviderType;
  name: string;
  status: ProviderStatus;
  isDefault: boolean;
  priority: number;
  apiKeyConfigured: boolean;
  lastChecked?: string;
  lastError?: string;
  errorRate: number;
  avgResponseTime: number;
  requestsToday: number;
  costToday: number;
  costThisMonth: number;
  quotaUsed: number;
  quotaLimit: number;
}

export interface ProviderModel {
  id: string;
  providerId: string;
  name: string;
  displayName: string;
  type: 'chat' | 'completion' | 'embedding' | 'image';
  isEnabled: boolean;
  isDefault: boolean;
  maxTokens: number;
  costPerInputToken: number;
  costPerOutputToken: number;
  requestsToday: number;
  avgResponseTime: number;
}

export interface ProviderConfig {
  apiKey?: string;
  apiKeyMasked?: string;
  baseUrl?: string;
  organizationId?: string;
  rateLimit: number;
  rateLimitPeriod: 'second' | 'minute' | 'hour';
  costLimitDaily: number;
  costLimitMonthly: number;
  alertThreshold: number;
  timeout: number;
  retryCount: number;
  retryDelay: number;
}

export interface ProviderHealth {
  status: ProviderStatus;
  latency: number;
  lastCheck: string;
  uptime: number;
  recentErrors: { timestamp: string; message: string; code?: string }[];
}

export interface ProviderMetrics {
  responseTimeTrend: { timestamp: string; value: number }[];
  errorRateTrend: { timestamp: string; value: number }[];
  requestsTrend: { timestamp: string; value: number }[];
  costTrend: { timestamp: string; value: number }[];
}

// ============================================
// Audit Log Types
// ============================================

export type AuditAction = 
  | 'CREATE' | 'READ' | 'UPDATE' | 'DELETE'
  | 'LOGIN' | 'LOGOUT' | 'LOGIN_FAILED'
  | 'PASSWORD_CHANGE' | 'PASSWORD_RESET'
  | 'ROLE_CHANGE' | 'STATUS_CHANGE'
  | 'ANALYZE' | 'EXPORT' | 'IMPORT'
  | 'PROVIDER_CONFIG' | 'SETTINGS_CHANGE'
  | 'BACKUP' | 'RESTORE' | 'SYSTEM';

export type AuditResource = 
  | 'user' | 'project' | 'analysis' | 'issue'
  | 'provider' | 'settings' | 'api_key'
  | 'webhook' | 'integration' | 'system';

export type AuditStatus = 'success' | 'failure' | 'warning';

export interface AuditLog {
  id: string;
  timestamp: string;
  userId: string;
  username: string;
  userEmail: string;
  action: AuditAction;
  resource: AuditResource;
  resourceId?: string;
  resourceName?: string;
  oldValue?: Record<string, unknown>;
  newValue?: Record<string, unknown>;
  ipAddress: string;
  userAgent?: string;
  location?: string;
  status: AuditStatus;
  details?: string;
  duration?: number;
}

export interface AuditFilters {
  search: string;
  action: AuditAction | 'all';
  resource: AuditResource | 'all';
  status: AuditStatus | 'all';
  userId?: string;
  dateRange?: [string, string];
}

export interface AuditAnalytics {
  totalLogs: number;
  successRate: number;
  failureRate: number;
  mostActiveUsers: { userId: string; username: string; count: number }[];
  actionDistribution: { action: AuditAction; count: number }[];
  failedActionsTimeline: { timestamp: string; count: number }[];
  loginPatterns: { hour: number; day: number; count: number }[];
  topResources: { resource: AuditResource; count: number }[];
}

export interface SecurityAlert {
  id: string;
  type: 'failed_login' | 'suspicious_activity' | 'bulk_delete' | 'rate_limit' | 'unusual_location';
  severity: 'low' | 'medium' | 'high' | 'critical';
  timestamp: string;
  userId?: string;
  username?: string;
  description: string;
  ipAddress?: string;
  location?: string;
  resolved: boolean;
  resolvedAt?: string;
  resolvedBy?: string;
}

// ============================================
// Pagination Types
// ============================================

export interface PaginationState {
  page: number;
  pageSize: number;
  total: number;
}

export interface SortState {
  field: string;
  order: 'asc' | 'desc';
}

// ============================================
// Admin Store State
// ============================================

interface AdminState {
  // Users
  users: AdminUser[];
  selectedUserIds: string[];
  userFilters: UserFilters;
  userPagination: PaginationState;
  userSort: SortState;
  userStats: UserStats | null;
  
  // Providers
  providers: AIProvider[];
  selectedProviderId: string | null;
  providerModels: ProviderModel[];
  
  // Audit
  auditLogs: AuditLog[];
  auditFilters: AuditFilters;
  auditPagination: PaginationState;
  auditAnalytics: AuditAnalytics | null;
  securityAlerts: SecurityAlert[];
  
  // UI State
  isLoading: boolean;
  error: string | null;
  
  // User Actions
  setUsers: (users: AdminUser[]) => void;
  selectUser: (userId: string) => void;
  deselectUser: (userId: string) => void;
  selectAllUsers: (userIds: string[]) => void;
  clearUserSelection: () => void;
  setUserFilters: (filters: Partial<UserFilters>) => void;
  resetUserFilters: () => void;
  setUserPagination: (pagination: Partial<PaginationState>) => void;
  setUserSort: (sort: SortState) => void;
  setUserStats: (stats: UserStats) => void;
  
  // Provider Actions
  setProviders: (providers: AIProvider[]) => void;
  selectProvider: (providerId: string | null) => void;
  setProviderModels: (models: ProviderModel[]) => void;
  updateProvider: (providerId: string, updates: Partial<AIProvider>) => void;
  
  // Audit Actions
  setAuditLogs: (logs: AuditLog[]) => void;
  setAuditFilters: (filters: Partial<AuditFilters>) => void;
  resetAuditFilters: () => void;
  setAuditPagination: (pagination: Partial<PaginationState>) => void;
  setAuditAnalytics: (analytics: AuditAnalytics) => void;
  setSecurityAlerts: (alerts: SecurityAlert[]) => void;
  
  // General Actions
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

// Default Values
const defaultUserFilters: UserFilters = {
  search: '',
  role: 'all',
  status: 'all',
};

const defaultAuditFilters: AuditFilters = {
  search: '',
  action: 'all',
  resource: 'all',
  status: 'all',
};

const defaultPagination: PaginationState = {
  page: 1,
  pageSize: 20,
  total: 0,
};

const defaultSort: SortState = {
  field: 'joinedAt',
  order: 'desc',
};

const initialState = {
  users: [],
  selectedUserIds: [],
  userFilters: defaultUserFilters,
  userPagination: defaultPagination,
  userSort: defaultSort,
  userStats: null,
  
  providers: [],
  selectedProviderId: null,
  providerModels: [],
  
  auditLogs: [],
  auditFilters: defaultAuditFilters,
  auditPagination: defaultPagination,
  auditAnalytics: null,
  securityAlerts: [],
  
  isLoading: false,
  error: null,
};

export const useAdminStore = create<AdminState>()(
  persist(
    (set, get) => ({
      ...initialState,

      // User Actions
      setUsers: (users) => set({ users }),
      
      selectUser: (userId) => set((state) => ({
        selectedUserIds: [...state.selectedUserIds, userId],
      })),
      
      deselectUser: (userId) => set((state) => ({
        selectedUserIds: state.selectedUserIds.filter((id) => id !== userId),
      })),
      
      selectAllUsers: (userIds) => set({ selectedUserIds: userIds }),
      
      clearUserSelection: () => set({ selectedUserIds: [] }),
      
      setUserFilters: (filters) => set((state) => ({
        userFilters: { ...state.userFilters, ...filters },
        userPagination: { ...state.userPagination, page: 1 },
      })),
      
      resetUserFilters: () => set({
        userFilters: defaultUserFilters,
        userPagination: { ...defaultPagination, pageSize: get().userPagination.pageSize },
      }),
      
      setUserPagination: (pagination) => set((state) => ({
        userPagination: { ...state.userPagination, ...pagination },
      })),
      
      setUserSort: (sort) => set({ userSort: sort }),
      
      setUserStats: (stats) => set({ userStats: stats }),

      // Provider Actions
      setProviders: (providers) => set({ providers }),
      
      selectProvider: (providerId) => set({ selectedProviderId: providerId }),
      
      setProviderModels: (models) => set({ providerModels: models }),
      
      updateProvider: (providerId, updates) => set((state) => ({
        providers: state.providers.map((p) =>
          p.id === providerId ? { ...p, ...updates } : p
        ),
      })),

      // Audit Actions
      setAuditLogs: (logs) => set({ auditLogs: logs }),
      
      setAuditFilters: (filters) => set((state) => ({
        auditFilters: { ...state.auditFilters, ...filters },
        auditPagination: { ...state.auditPagination, page: 1 },
      })),
      
      resetAuditFilters: () => set({
        auditFilters: defaultAuditFilters,
        auditPagination: { ...defaultPagination, pageSize: get().auditPagination.pageSize },
      }),
      
      setAuditPagination: (pagination) => set((state) => ({
        auditPagination: { ...state.auditPagination, ...pagination },
      })),
      
      setAuditAnalytics: (analytics) => set({ auditAnalytics: analytics }),
      
      setSecurityAlerts: (alerts) => set({ securityAlerts: alerts }),

      // General Actions
      setLoading: (isLoading) => set({ isLoading }),
      
      setError: (error) => set({ error }),
      
      reset: () => set(initialState),
    }),
    {
      name: 'admin-storage',
      storage: createJSONStorage(() => sessionStorage),
      partialize: (state) => ({
        userFilters: state.userFilters,
        userPagination: state.userPagination,
        userSort: state.userSort,
        auditFilters: state.auditFilters,
        auditPagination: state.auditPagination,
      }),
    }
  )
);

export default useAdminStore;
