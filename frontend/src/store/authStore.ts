import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

export interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'user' | 'viewer' | 'guest';
  avatar?: string;
  createdAt?: string;
  lastLoginAt?: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  
  // Actions
  setUser: (user: User | null) => void;
  setTokens: (token: string, refreshToken: string) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      refreshToken: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,

      setUser: (user) => set({ 
        user, 
        isAuthenticated: !!user 
      }),

      setTokens: (token, refreshToken) => set({ 
        token, 
        refreshToken,
        isAuthenticated: true 
      }),

      setLoading: (isLoading) => set({ isLoading }),

      setError: (error) => set({ error }),

      logout: () => set({
        user: null,
        token: null,
        refreshToken: null,
        isAuthenticated: false,
        error: null
      })
    }),
    {
      name: 'auth-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        token: state.token,
        refreshToken: state.refreshToken,
        user: state.user
      })
    }
  )
);

export default useAuthStore;
