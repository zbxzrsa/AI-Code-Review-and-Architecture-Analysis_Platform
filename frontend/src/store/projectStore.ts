import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

/** Project status types */
export type ProjectStatus = 'active' | 'inactive' | 'archived';

/** View mode for project list */
export type ViewMode = 'table' | 'card';

/** Sort field options */
export type SortField = 'name' | 'created_at' | 'updated_at' | 'last_analyzed_at';

/** Sort order */
export type SortOrder = 'asc' | 'desc';

/**
 * Project entity representing a code review project
 */
export interface Project {
  id: string;
  name: string;
  description?: string;
  repository_url?: string;
  language: string;
  framework?: string;
  created_at: string;
  updated_at: string;
  last_analyzed_at?: string;
  owner_id: string;
  owner_name?: string;
  is_public: boolean;
  status: ProjectStatus;
  settings: ProjectSettings;
  stats?: ProjectStats;
  team_members?: TeamMember[];
}

/**
 * Project statistics
 */
export interface ProjectStats {
  total_analyses: number;
  total_issues: number;
  resolved_issues: number;
  files_analyzed: number;
  last_analysis_score?: number;
}

/**
 * Team member with role
 */
export interface TeamMember {
  id: string;
  user_id: string;
  email: string;
  name: string;
  avatar?: string;
  role: 'owner' | 'admin' | 'member' | 'viewer';
  invited_at: string;
  accepted_at?: string;
}

/**
 * Webhook configuration
 */
export interface Webhook {
  id: string;
  url: string;
  events: WebhookEvent[];
  is_active: boolean;
  secret?: string;
  last_triggered_at?: string;
  last_status?: 'success' | 'failure';
  created_at: string;
}

export type WebhookEvent = 
  | 'analysis.started'
  | 'analysis.completed'
  | 'analysis.failed'
  | 'issue.created'
  | 'issue.resolved';

/**
 * API Key for project
 */
export interface APIKey {
  id: string;
  name: string;
  key_prefix: string;
  permissions: string[];
  last_used_at?: string;
  expires_at?: string;
  created_at: string;
  usage_count: number;
}

/**
 * Activity log entry
 */
export interface ActivityLog {
  id: string;
  action: string;
  description: string;
  actor: {
    id: string;
    name: string;
    email: string;
    avatar?: string;
  };
  metadata?: Record<string, unknown>;
  created_at: string;
}

/**
 * Notification settings for a project
 */
export interface NotificationSettings {
  email_on_analysis_complete: boolean;
  email_on_critical_issues: boolean;
  email_digest: 'none' | 'daily' | 'weekly';
  slack_webhook_url?: string;
  teams_webhook_url?: string;
}

/**
 * Analysis settings for a project
 */
export interface AnalysisSettings {
  ai_model: 'gpt-4' | 'gpt-3.5-turbo' | 'claude-3-opus' | 'claude-3-sonnet';
  analysis_frequency: 'manual' | 'on_push' | 'on_pr' | 'scheduled';
  schedule_cron?: string;
  priority: 'low' | 'medium' | 'high';
  max_files_per_analysis: number;
  excluded_patterns: string[];
}

export interface ProjectSettings {
  auto_review: boolean;
  review_on_push: boolean;
  review_on_pr: boolean;
  severity_threshold: 'error' | 'warning' | 'info' | 'hint';
  enabled_rules: string[];
  ignored_paths: string[];
  custom_rules?: Record<string, unknown>;
  analysis?: AnalysisSettings;
  notifications?: NotificationSettings;
}

/**
 * Draft project for wizard
 */
export interface ProjectDraft {
  id?: string;
  step: number;
  basic_info: {
    name: string;
    description: string;
    repository_url: string;
    language: string;
  };
  analysis_settings: Partial<AnalysisSettings>;
  notification_settings: Partial<NotificationSettings>;
  last_saved_at?: string;
}

/**
 * Filter state for project list
 */
export interface ProjectFilters {
  search: string;
  status: ProjectStatus | 'all';
  language: string;
  sortField: SortField;
  sortOrder: SortOrder;
}

/**
 * Pagination state
 */
export interface PaginationState {
  page: number;
  pageSize: number;
  total: number;
}

export interface AnalysisSession {
  id: string;
  project_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  started_at: string;
  completed_at?: string;
  issues_count: number;
  files_analyzed: number;
  version: 'v1' | 'v2' | 'v3';
}

export interface CodeFile {
  path: string;
  content: string;
  language: string;
  size: number;
  last_modified: string;
}

interface ProjectState {
  // Current project
  currentProject: Project | null;
  projects: Project[];
  
  // Analysis
  currentSession: AnalysisSession | null;
  sessions: AnalysisSession[];
  
  // Files
  files: CodeFile[];
  selectedFile: CodeFile | null;
  
  // UI State
  isLoading: boolean;
  error: string | null;
  
  // List UI State
  viewMode: ViewMode;
  filters: ProjectFilters;
  pagination: PaginationState;
  
  // Draft for wizard
  draft: ProjectDraft | null;
  
  // Settings page state
  unsavedChanges: boolean;
  
  // Actions - Projects
  setCurrentProject: (project: Project | null) => void;
  setProjects: (projects: Project[]) => void;
  addProject: (project: Project) => void;
  updateProject: (id: string, updates: Partial<Project>) => void;
  removeProject: (id: string) => void;
  
  // Actions - Sessions
  setCurrentSession: (session: AnalysisSession | null) => void;
  setSessions: (sessions: AnalysisSession[]) => void;
  addSession: (session: AnalysisSession) => void;
  updateSession: (id: string, updates: Partial<AnalysisSession>) => void;
  
  // Actions - Files
  setFiles: (files: CodeFile[]) => void;
  setSelectedFile: (file: CodeFile | null) => void;
  updateFile: (path: string, content: string) => void;
  
  // Actions - UI State
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setViewMode: (mode: ViewMode) => void;
  setFilters: (filters: Partial<ProjectFilters>) => void;
  resetFilters: () => void;
  setPagination: (pagination: Partial<PaginationState>) => void;
  
  // Actions - Draft
  setDraft: (draft: ProjectDraft | null) => void;
  updateDraft: (updates: Partial<ProjectDraft>) => void;
  saveDraftToStorage: () => void;
  loadDraftFromStorage: () => void;
  clearDraft: () => void;
  
  // Actions - Settings
  setUnsavedChanges: (hasChanges: boolean) => void;
  
  reset: () => void;
}

/** Default filter state */
const defaultFilters: ProjectFilters = {
  search: '',
  status: 'all',
  language: 'all',
  sortField: 'updated_at',
  sortOrder: 'desc',
};

/** Default pagination state */
const defaultPagination: PaginationState = {
  page: 1,
  pageSize: 10,
  total: 0,
};

/** Default draft state */
const defaultDraft: ProjectDraft = {
  step: 0,
  basic_info: {
    name: '',
    description: '',
    repository_url: '',
    language: 'python',
  },
  analysis_settings: {
    ai_model: 'gpt-4',
    analysis_frequency: 'on_pr',
    priority: 'medium',
    max_files_per_analysis: 100,
    excluded_patterns: ['node_modules/**', '.git/**', 'dist/**'],
  },
  notification_settings: {
    email_on_analysis_complete: true,
    email_on_critical_issues: true,
    email_digest: 'weekly',
  },
};

const initialState = {
  currentProject: null,
  projects: [],
  currentSession: null,
  sessions: [],
  files: [],
  selectedFile: null,
  isLoading: false,
  error: null,
  viewMode: 'table' as ViewMode,
  filters: defaultFilters,
  pagination: defaultPagination,
  draft: null,
  unsavedChanges: false,
};

export const useProjectStore = create<ProjectState>()(
  persist(
    (set, get) => ({
      ...initialState,

      setCurrentProject: (project) => set({ currentProject: project }),

      setProjects: (projects) => set({ projects }),

      addProject: (project) => set((state) => ({
        projects: [...state.projects, project]
      })),

      updateProject: (id, updates) => set((state) => ({
        projects: state.projects.map((p) =>
          p.id === id ? { ...p, ...updates } : p
        ),
        currentProject: state.currentProject?.id === id
          ? { ...state.currentProject, ...updates }
          : state.currentProject
      })),

      removeProject: (id) => set((state) => ({
        projects: state.projects.filter((p) => p.id !== id),
        currentProject: state.currentProject?.id === id
          ? null
          : state.currentProject
      })),

      setCurrentSession: (session) => set({ currentSession: session }),

      setSessions: (sessions) => set({ sessions }),

      addSession: (session) => set((state) => ({
        sessions: [session, ...state.sessions]
      })),

      updateSession: (id, updates) => set((state) => ({
        sessions: state.sessions.map((s) =>
          s.id === id ? { ...s, ...updates } : s
        ),
        currentSession: state.currentSession?.id === id
          ? { ...state.currentSession, ...updates }
          : state.currentSession
      })),

      setFiles: (files) => set({ files }),

      setSelectedFile: (file) => set({ selectedFile: file }),

      updateFile: (path, content) => set((state) => ({
        files: state.files.map((f) =>
          f.path === path ? { ...f, content } : f
        ),
        selectedFile: state.selectedFile?.path === path
          ? { ...state.selectedFile, content }
          : state.selectedFile
      })),

      setLoading: (isLoading) => set({ isLoading }),

      setError: (error) => set({ error }),
      
      // View mode
      setViewMode: (mode) => set({ viewMode: mode }),
      
      // Filters
      setFilters: (filters) => set((state) => ({
        filters: { ...state.filters, ...filters },
        pagination: { ...state.pagination, page: 1 }, // Reset page on filter change
      })),
      
      resetFilters: () => set({
        filters: defaultFilters,
        pagination: defaultPagination,
      }),
      
      // Pagination
      setPagination: (pagination) => set((state) => ({
        pagination: { ...state.pagination, ...pagination },
      })),
      
      // Draft management
      setDraft: (draft) => set({ draft }),
      
      updateDraft: (updates) => set((state) => ({
        draft: state.draft
          ? { ...state.draft, ...updates, last_saved_at: new Date().toISOString() }
          : { ...defaultDraft, ...updates, last_saved_at: new Date().toISOString() },
      })),
      
      saveDraftToStorage: () => {
        const { draft } = get();
        if (draft) {
          localStorage.setItem('project-draft', JSON.stringify(draft));
        }
      },
      
      loadDraftFromStorage: () => {
        const savedDraft = localStorage.getItem('project-draft');
        if (savedDraft) {
          try {
            const parsed = JSON.parse(savedDraft);
            set({ draft: parsed });
          } catch (e) {
            console.error('Failed to parse draft from storage', e);
          }
        }
      },
      
      clearDraft: () => {
        localStorage.removeItem('project-draft');
        set({ draft: null });
      },
      
      // Settings
      setUnsavedChanges: (hasChanges) => set({ unsavedChanges: hasChanges }),

      reset: () => set(initialState)
    }),
    {
      name: 'project-storage',
      storage: createJSONStorage(() => sessionStorage),
      partialize: (state) => ({
        currentProject: state.currentProject,
        selectedFile: state.selectedFile,
        viewMode: state.viewMode,
        filters: state.filters,
      })
    }
  )
);

export { defaultFilters, defaultPagination, defaultDraft };
export default useProjectStore;
