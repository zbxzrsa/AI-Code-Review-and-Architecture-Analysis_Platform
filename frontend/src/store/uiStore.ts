import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

export type Theme = 'light' | 'dark' | 'system';
export type Language = 'en' | 'zh-CN' | 'zh-TW';

interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message?: string;
  duration?: number;
  timestamp: number;
}

interface Modal {
  id: string;
  type: string;
  props?: Record<string, unknown>;
}

interface SidebarState {
  isOpen: boolean;
  width: number;
  activeSection?: string;
}

interface UIState {
  // Theme
  theme: Theme;
  resolvedTheme: 'light' | 'dark';
  
  // Language
  language: Language;
  
  // Sidebar
  sidebar: SidebarState;
  
  // Command palette
  isCommandPaletteOpen: boolean;
  
  // Notifications
  notifications: Notification[];
  
  // Modals
  modals: Modal[];
  
  // Loading states
  globalLoading: boolean;
  loadingMessage?: string;
  
  // Breadcrumbs
  breadcrumbs: Array<{ label: string; path?: string }>;
  
  // Actions
  setTheme: (theme: Theme) => void;
  setLanguage: (language: Language) => void;
  
  toggleSidebar: () => void;
  setSidebarWidth: (width: number) => void;
  setSidebarSection: (section: string) => void;
  
  toggleCommandPalette: () => void;
  openCommandPalette: () => void;
  closeCommandPalette: () => void;
  
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => string;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
  
  openModal: (type: string, props?: Record<string, unknown>) => string;
  closeModal: (id: string) => void;
  closeAllModals: () => void;
  
  setGlobalLoading: (loading: boolean, message?: string) => void;
  
  setBreadcrumbs: (breadcrumbs: Array<{ label: string; path?: string }>) => void;
}

// Generate unique ID
const generateId = () => Math.random().toString(36).substring(2, 9);

// Get system theme preference
const getSystemTheme = (): 'light' | 'dark' => {
  if (typeof window !== 'undefined') {
    return window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark'
      : 'light';
  }
  return 'light';
};

export const useUIStore = create<UIState>()(
  persist(
    (set, get) => ({
      // Initial state
      theme: 'system',
      resolvedTheme: getSystemTheme(),
      language: 'en',
      sidebar: {
        isOpen: true,
        width: 280,
        activeSection: undefined
      },
      isCommandPaletteOpen: false,
      notifications: [],
      modals: [],
      globalLoading: false,
      loadingMessage: undefined,
      breadcrumbs: [],

      // Theme actions
      setTheme: (theme) => {
        const resolvedTheme = theme === 'system' ? getSystemTheme() : theme;
        set({ theme, resolvedTheme });
        
        // Update document class
        if (typeof document !== 'undefined') {
          document.documentElement.classList.remove('light', 'dark');
          document.documentElement.classList.add(resolvedTheme);
        }
      },

      // Language actions
      setLanguage: (language) => set({ language }),

      // Sidebar actions
      toggleSidebar: () => set((state) => ({
        sidebar: { ...state.sidebar, isOpen: !state.sidebar.isOpen }
      })),

      setSidebarWidth: (width) => set((state) => ({
        sidebar: { ...state.sidebar, width }
      })),

      setSidebarSection: (section) => set((state) => ({
        sidebar: { ...state.sidebar, activeSection: section }
      })),

      // Command palette actions
      toggleCommandPalette: () => set((state) => ({
        isCommandPaletteOpen: !state.isCommandPaletteOpen
      })),

      openCommandPalette: () => set({ isCommandPaletteOpen: true }),

      closeCommandPalette: () => set({ isCommandPaletteOpen: false }),

      // Notification actions
      addNotification: (notification) => {
        const id = generateId();
        const newNotification: Notification = {
          ...notification,
          id,
          timestamp: Date.now()
        };

        set((state) => ({
          notifications: [...state.notifications, newNotification]
        }));

        // Auto-remove after duration
        const duration = notification.duration ?? 5000;
        if (duration > 0) {
          setTimeout(() => {
            get().removeNotification(id);
          }, duration);
        }

        return id;
      },

      removeNotification: (id) => set((state) => ({
        notifications: state.notifications.filter((n) => n.id !== id)
      })),

      clearNotifications: () => set({ notifications: [] }),

      // Modal actions
      openModal: (type, props) => {
        const id = generateId();
        set((state) => ({
          modals: [...state.modals, { id, type, props }]
        }));
        return id;
      },

      closeModal: (id) => set((state) => ({
        modals: state.modals.filter((m) => m.id !== id)
      })),

      closeAllModals: () => set({ modals: [] }),

      // Loading actions
      setGlobalLoading: (loading, message) => set({
        globalLoading: loading,
        loadingMessage: message
      }),

      // Breadcrumb actions
      setBreadcrumbs: (breadcrumbs) => set({ breadcrumbs })
    }),
    {
      name: 'ui-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        theme: state.theme,
        language: state.language,
        sidebar: {
          isOpen: state.sidebar.isOpen,
          width: state.sidebar.width
        }
      })
    }
  )
);

// Listen for system theme changes
if (typeof window !== 'undefined') {
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    const state = useUIStore.getState();
    if (state.theme === 'system') {
      state.setTheme('system');
    }
  });
}

export default useUIStore;
