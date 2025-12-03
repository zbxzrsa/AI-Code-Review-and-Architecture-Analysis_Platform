/**
 * Theme Context
 * 
 * Provides theme switching between:
 * - Default (Modern)
 * - Pixel (Retro 8-bit)
 */

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { ConfigProvider, theme as antdTheme } from 'antd';

export type ThemeStyle = 'default' | 'pixel';
export type ThemeMode = 'light' | 'dark' | 'system';

interface ThemeContextType {
  themeStyle: ThemeStyle;
  themeMode: ThemeMode;
  setThemeStyle: (style: ThemeStyle) => void;
  setThemeMode: (mode: ThemeMode) => void;
  isPixel: boolean;
  isDark: boolean;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

const THEME_STYLE_KEY = 'app-theme-style';
const THEME_MODE_KEY = 'app-theme-mode';

// Pixel theme token overrides
const pixelThemeToken = {
  colorPrimary: '#00ff88',
  colorSuccess: '#00ff88',
  colorWarning: '#ffd93d',
  colorError: '#ff4757',
  colorInfo: '#6bcbff',
  colorBgContainer: '#1a1a2e',
  colorBgElevated: '#16213e',
  colorBgLayout: '#0f0f23',
  colorText: '#e0e0e0',
  colorTextSecondary: '#888888',
  colorBorder: '#333366',
  borderRadius: 0,
  fontFamily: "'VT323', monospace",
};

interface ThemeProviderProps {
  children: ReactNode;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const [themeStyle, setThemeStyleState] = useState<ThemeStyle>(() => {
    const saved = localStorage.getItem(THEME_STYLE_KEY);
    return (saved as ThemeStyle) || 'default';
  });

  const [themeMode, setThemeModeState] = useState<ThemeMode>(() => {
    const saved = localStorage.getItem(THEME_MODE_KEY);
    return (saved as ThemeMode) || 'dark';
  });

  const [systemDark, setSystemDark] = useState(() =>
    window.matchMedia('(prefers-color-scheme: dark)').matches
  );

  // Listen for system theme changes
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = (e: MediaQueryListEvent) => setSystemDark(e.matches);
    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, []);

  // Save theme preferences
  useEffect(() => {
    localStorage.setItem(THEME_STYLE_KEY, themeStyle);
    
    // Apply pixel theme class to body
    if (themeStyle === 'pixel') {
      document.body.classList.add('pixel-theme');
    } else {
      document.body.classList.remove('pixel-theme');
    }
  }, [themeStyle]);

  useEffect(() => {
    localStorage.setItem(THEME_MODE_KEY, themeMode);
  }, [themeMode]);

  const setThemeStyle = (style: ThemeStyle) => {
    setThemeStyleState(style);
  };

  const setThemeMode = (mode: ThemeMode) => {
    setThemeModeState(mode);
  };

  const isDark =
    themeMode === 'dark' || (themeMode === 'system' && systemDark);

  const isPixel = themeStyle === 'pixel';

  // Build Ant Design theme config
  const antdThemeConfig = {
    algorithm: isDark ? antdTheme.darkAlgorithm : antdTheme.defaultAlgorithm,
    token: isPixel ? pixelThemeToken : undefined,
  };

  return (
    <ThemeContext.Provider
      value={{
        themeStyle,
        themeMode,
        setThemeStyle,
        setThemeMode,
        isPixel,
        isDark,
      }}
    >
      <ConfigProvider theme={antdThemeConfig}>
        <div className={isPixel ? 'pixel-theme' : ''}>
          {children}
        </div>
      </ConfigProvider>
    </ThemeContext.Provider>
  );
};

export const useTheme = (): ThemeContextType => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

export default ThemeContext;
