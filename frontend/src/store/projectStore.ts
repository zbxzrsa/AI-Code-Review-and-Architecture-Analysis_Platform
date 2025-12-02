import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

export interface Project {
  id: string;
  name: string;
  description?: string;
  repository_url?: string;
  language: string;
  framework?: string;
  created_at: string;
  updated_at: string;
  owner_id: string;
  is_public: boolean;
  settings: ProjectSettings;
}

export interface ProjectSettings {
  auto_review: boolean;
  review_on_push: boolean;
  review_on_pr: boolean;
  severity_threshold: 'error' | 'warning' | 'info' | 'hint';
  enabled_rules: string[];
  ignored_paths: string[];
  custom_rules?: Record<string, unknown>;
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
  
  // Actions
  setCurrentProject: (project: Project | null) => void;
  setProjects: (projects: Project[]) => void;
  addProject: (project: Project) => void;
  updateProject: (id: string, updates: Partial<Project>) => void;
  removeProject: (id: string) => void;
  
  setCurrentSession: (session: AnalysisSession | null) => void;
  setSessions: (sessions: AnalysisSession[]) => void;
  addSession: (session: AnalysisSession) => void;
  updateSession: (id: string, updates: Partial<AnalysisSession>) => void;
  
  setFiles: (files: CodeFile[]) => void;
  setSelectedFile: (file: CodeFile | null) => void;
  updateFile: (path: string, content: string) => void;
  
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

const initialState = {
  currentProject: null,
  projects: [],
  currentSession: null,
  sessions: [],
  files: [],
  selectedFile: null,
  isLoading: false,
  error: null
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

      reset: () => set(initialState)
    }),
    {
      name: 'project-storage',
      storage: createJSONStorage(() => sessionStorage),
      partialize: (state) => ({
        currentProject: state.currentProject,
        selectedFile: state.selectedFile
      })
    }
  )
);

export default useProjectStore;
