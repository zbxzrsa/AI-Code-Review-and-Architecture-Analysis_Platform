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
        set((state) => {
          const modeMap: Record<ThemeMode, ThemeMode> = {
            light: "dark",
            dark: "high-contrast",
            "high-contrast": "light",
          };
          return { mode: modeMap[state.mode] };
        }),
    }),
    {
      name: "theme-storage",
    }
  )
);
