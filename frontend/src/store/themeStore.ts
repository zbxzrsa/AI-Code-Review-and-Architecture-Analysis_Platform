import { create } from "zustand";
import { persist } from "zustand/middleware";

export type ThemeMode = "light" | "dark" | "high-contrast";

interface ThemeStore {
  mode: ThemeMode;
  setMode: (mode: ThemeMode) => void;
  toggleMode: () => void;
}

export const useThemeStore = create<ThemeStore>()(
  persist(
    (set) => ({
      mode: "light",
      setMode: (mode) => set({ mode }),
      toggleMode: () =>
        set((state) => ({
          mode:
            state.mode === "light"
              ? "dark"
              : state.mode === "dark"
              ? "high-contrast"
              : "light",
        })),
    }),
    {
      name: "theme-storage",
    }
  )
);
