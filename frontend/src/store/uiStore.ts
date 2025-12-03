import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

export type Theme = 'light' | 'dark' | 'system';
export type Language = 'en' | 'zh-CN' | 'zh-TW';
export type NotificationPosition = 'top-right' | 'top-center' | 'bottom-right' | 'bottom-center';

export interface NotificationAction {
  label: string;
  onClick: () => void;
  type?: 'primary' | 'default' | 'danger';
}

interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message?: string;
  duration?: number;
  timestamp: number;
  dismissible?: boolean;
  actions?: NotificationAction[];
  icon?: React.ReactNode;
}

export interface FavoriteItem {
  id: string;
  label: string;
  path: string;
  icon?: string;
  order: number;
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
  searchQuery: string;
  expandedKeys: string[];
  favorites: FavoriteItem[];
  mobileDrawerOpen: boolean;
}

interface NotificationSettings {
  position: NotificationPosition;
  maxVisible: number;
  defaultDuration: number;
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
  notificationSettings: NotificationSettings;
  
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
  setSidebarSearch: (query: string) => void;
  setSidebarExpandedKeys: (keys: string[]) => void;
  toggleMobileDrawer: () => void;
  closeMobileDrawer: () => void;
  
  // Favorites
  addFavorite: (item: Omit<FavoriteItem, 'id' | 'order'>) => void;
  removeFavorite: (id: string) => void;
  reorderFavorites: (favorites: FavoriteItem[]) => void;
  
  toggleCommandPalette: () => void;
  openCommandPalette: () => void;
  closeCommandPalette: () => void;
  
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => string;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
  setNotificationPosition: (position: NotificationPosition) => void;
  
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
        activeSection: undefined,
        searchQuery: '',
        expandedKeys: [],
        favorites: [],
        mobileDrawerOpen: false,
      },
      isCommandPaletteOpen: false,
      notifications: [],
      notificationSettings: {
        position: 'top-right',
        maxVisible: 3,
        defaultDuration: 5000,
      },
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

      setSidebarSearch: (query) => set((state) => ({
        sidebar: { ...state.sidebar, searchQuery: query }
      })),

      setSidebarExpandedKeys: (keys) => set((state) => ({
        sidebar: { ...state.sidebar, expandedKeys: keys }
      })),

      toggleMobileDrawer: () => set((state) => ({
        sidebar: { ...state.sidebar, mobileDrawerOpen: !state.sidebar.mobileDrawerOpen }
      })),

      closeMobileDrawer: () => set((state) => ({
        sidebar: { ...state.sidebar, mobileDrawerOpen: false }
      })),

      // Favorites actions
      addFavorite: (item) => set((state) => {
        const id = generateId();
        const order = state.sidebar.favorites.length;
        return {
          sidebar: {
            ...state.sidebar,
            favorites: [...state.sidebar.favorites, { ...item, id, order }]
          }
        };
      }),

      removeFavorite: (id) => set((state) => ({
        sidebar: {
          ...state.sidebar,
          favorites: state.sidebar.favorites.filter(f => f.id !== id)
        }
      })),

      reorderFavorites: (favorites) => set((state) => ({
        sidebar: { ...state.sidebar, favorites }
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

      setNotificationPosition: (position) => set((state) => ({
        notificationSettings: { ...state.notificationSettings, position }
      })),

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
          isOpen: state.sidebar?.isOpen ?? true,
          width: state.sidebar?.width ?? 280,
          favorites: state.sidebar?.favorites ?? [],
        }
      }),
      merge: (persistedState, currentState) => {
        const persisted = persistedState as Partial<UIState>;
        return {
          ...currentState,
          ...persisted,
          sidebar: {
            ...currentState.sidebar,
            ...persisted.sidebar,
            // Ensure arrays have defaults
            expandedKeys: persisted.sidebar?.expandedKeys ?? currentState.sidebar.expandedKeys ?? [],
            favorites: persisted.sidebar?.favorites ?? currentState.sidebar.favorites ?? [],
          }
        };
      }
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
