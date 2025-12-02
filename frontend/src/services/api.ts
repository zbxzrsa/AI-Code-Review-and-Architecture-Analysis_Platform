/**
 * API Service / API服务
 * 
 * Axios instance with authentication and error handling.
 * 带有认证和错误处理的Axios实例。
 */

import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig } from 'axios';
import { useAuthStore } from '../store/authStore';

/**
 * API Base URL / API基础URL
 * 
 * Uses relative path '/api' so Vite proxy can route to backend.
 * 使用相对路径'/api'以便Vite代理可以路由到后端。
 */
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

// Create axios instance / 创建axios实例
export const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  },
  // Important for cookies / 对于Cookie很重要
  withCredentials: true
});

// Request interceptor - add auth token
api.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const { token } = useAuthStore.getState();
    
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor - handle errors and token refresh
api.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config as InternalAxiosRequestConfig & { _retry?: boolean };
    
    // Handle 401 Unauthorized
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      
      const { refreshToken, setTokens, logout } = useAuthStore.getState();
      
      if (refreshToken) {
        try {
          const response = await axios.post(`${API_BASE_URL}/auth/refresh`, {
            refresh_token: refreshToken
          });
          
          const { access_token, refresh_token } = response.data;
          setTokens(access_token, refresh_token);
          
          // Retry original request
          if (originalRequest.headers) {
            originalRequest.headers.Authorization = `Bearer ${access_token}`;
          }
          return api(originalRequest);
        } catch (refreshError) {
          logout();
          window.location.href = '/login';
          return Promise.reject(refreshError);
        }
      } else {
        logout();
        window.location.href = '/login';
      }
    }
    
    return Promise.reject(error);
  }
);

// API methods
export const apiService = {
  // Auth
  auth: {
    login: (email: string, password: string, invitation_code?: string) =>
      api.post('/auth/login', { email, password, invitation_code }),
    
    register: (data: { email: string; password: string; name: string; invitation_code: string }) =>
      api.post('/auth/register', data),
    
    logout: () => api.post('/auth/logout'),
    
    refresh: (refreshToken: string) =>
      api.post('/auth/refresh', { refresh_token: refreshToken }),
    
    me: () => api.get('/auth/me'),
    
    updateProfile: (data: { name?: string; avatar?: string }) =>
      api.put('/auth/profile', data),
    
    changePassword: (oldPassword: string, newPassword: string) =>
      api.post('/auth/change-password', { old_password: oldPassword, new_password: newPassword })
  },

  // Projects
  projects: {
    list: (params?: { page?: number; limit?: number; search?: string }) =>
      api.get('/projects', { params }),
    
    get: (id: string) => api.get(`/projects/${id}`),
    
    create: (data: { name: string; description?: string; repository_url?: string; language: string }) =>
      api.post('/projects', data),
    
    update: (id: string, data: Partial<{ name: string; description: string; settings: unknown }>) =>
      api.put(`/projects/${id}`, data),
    
    delete: (id: string) => api.delete(`/projects/${id}`),
    
    getFiles: (id: string, path?: string) =>
      api.get(`/projects/${id}/files`, { params: { path } }),
    
    getFile: (id: string, path: string) =>
      api.get(`/projects/${id}/files/${encodeURIComponent(path)}`),
    
    updateFile: (id: string, path: string, content: string) =>
      api.put(`/projects/${id}/files/${encodeURIComponent(path)}`, { content })
  },

  // Analysis
  analysis: {
    start: (projectId: string, options?: { files?: string[]; version?: string }) =>
      api.post(`/projects/${projectId}/analyze`, options),
    
    getSession: (sessionId: string) =>
      api.get(`/analyze/${sessionId}`),
    
    getSessions: (projectId: string, params?: { page?: number; limit?: number }) =>
      api.get(`/projects/${projectId}/sessions`, { params }),
    
    getIssues: (sessionId: string, params?: { severity?: string; type?: string }) =>
      api.get(`/analyze/${sessionId}/issues`, { params }),
    
    applyFix: (sessionId: string, issueId: string) =>
      api.post(`/analyze/${sessionId}/issues/${issueId}/fix`),
    
    dismissIssue: (sessionId: string, issueId: string, reason?: string) =>
      api.post(`/analyze/${sessionId}/issues/${issueId}/dismiss`, { reason }),
    
    streamUrl: (sessionId: string) =>
      `${API_BASE_URL}/analyze/${sessionId}/stream`
  },

  // Experiments (Admin)
  experiments: {
    list: (params?: { status?: string; page?: number; limit?: number }) =>
      api.get('/experiments', { params }),
    
    get: (id: string) => api.get(`/experiments/${id}`),
    
    create: (data: { name: string; config: unknown; dataset_id: string }) =>
      api.post('/experiments', data),
    
    start: (id: string) => api.post(`/experiments/${id}/start`),
    
    stop: (id: string) => api.post(`/experiments/${id}/stop`),
    
    evaluate: (id: string) => api.post(`/experiments/${id}/evaluate`),
    
    promote: (id: string) => api.post(`/experiments/${id}/promote`),
    
    quarantine: (id: string, reason: string) =>
      api.post(`/experiments/${id}/quarantine`, { reason }),
    
    getMetrics: (id: string) => api.get(`/experiments/${id}/metrics`),
    
    compare: (id1: string, id2: string) =>
      api.get('/experiments/compare', { params: { experiment_a: id1, experiment_b: id2 } })
  },

  // Versions (Admin)
  versions: {
    list: () => api.get('/versions'),
    
    getCurrent: () => api.get('/versions/current'),
    
    getHistory: (params?: { page?: number; limit?: number }) =>
      api.get('/versions/history', { params }),
    
    rollback: (versionId: string) =>
      api.post(`/versions/${versionId}/rollback`)
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
    }) => api.get('/audit', { params }),
    
    get: (id: string) => api.get(`/audit/${id}`)
  },

  // Metrics
  metrics: {
    getDashboard: () => api.get('/metrics/dashboard'),
    
    getSystem: () => api.get('/metrics/system'),
    
    getProvider: (provider: string) =>
      api.get(`/metrics/providers/${provider}`),
    
    getUsage: (params?: { start_date?: string; end_date?: string }) =>
      api.get('/metrics/usage', { params })
  }
};

export default api;
