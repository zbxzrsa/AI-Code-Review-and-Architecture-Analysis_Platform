/**
 * VSCode + GitHub Inspired Theme
 * Cool Color Scheme (Blues, Teals, Purples)
 * 
 * Design Philosophy:
 * - VSCode's dark productivity-focused interface
 * - GitHub's clean code presentation
 * - Cool color palette for visual comfort
 */

export const colors = {
  // Primary Brand Colors (Cool Blues)
  primary: {
    50: '#e6f3ff',
    100: '#b3d9ff',
    200: '#80bfff',
    300: '#4da6ff',
    400: '#1a8cff',
    500: '#0073e6',  // Main primary
    600: '#005cb3',
    700: '#004580',
    800: '#002e4d',
    900: '#00171a',
  },

  // Accent Colors (Teal/Cyan)
  accent: {
    50: '#e6fffa',
    100: '#b3fff0',
    200: '#80ffe6',
    300: '#4dffdc',
    400: '#1affd2',
    500: '#00e6b8',  // Main accent
    600: '#00b38f',
    700: '#008066',
    800: '#004d3d',
    900: '#001a14',
  },

  // Secondary Colors (Purple)
  secondary: {
    50: '#f3e8ff',
    100: '#dbb4ff',
    200: '#c480ff',
    300: '#ac4dff',
    400: '#951aff',
    500: '#7c00e6',  // Main secondary
    600: '#6200b3',
    700: '#470080',
    800: '#2d004d',
    900: '#12001a',
  },

  // Neutral Grays (VSCode-inspired)
  gray: {
    50: '#f8f9fa',
    100: '#e9ecef',
    200: '#dee2e6',
    300: '#ced4da',
    400: '#adb5bd',
    500: '#6c757d',
    600: '#495057',
    700: '#343a40',
    800: '#212529',
    900: '#0d1117',  // GitHub dark
  },

  // Background Colors (Dark Theme - VSCode Style)
  background: {
    primary: '#0d1117',      // Main background (GitHub dark)
    secondary: '#161b22',    // Card/Panel background
    tertiary: '#21262d',     // Elevated surfaces
    hover: '#30363d',        // Hover state
    active: '#388bfd1a',     // Active/selected
    sidebar: '#010409',      // Sidebar background
    editor: '#0d1117',       // Editor background
    terminal: '#0a0c10',     // Terminal background
    modal: '#161b22',        // Modal background
    tooltip: '#21262d',      // Tooltip background
  },

  // Border Colors
  border: {
    primary: '#30363d',      // Main border
    secondary: '#21262d',    // Subtle border
    focus: '#388bfd',        // Focus ring
    error: '#f85149',        // Error border
    success: '#3fb950',      // Success border
    warning: '#d29922',      // Warning border
  },

  // Text Colors
  text: {
    primary: '#e6edf3',      // Main text
    secondary: '#8b949e',    // Secondary text
    tertiary: '#6e7681',     // Muted text
    link: '#58a6ff',         // Links
    code: '#79c0ff',         // Inline code
    heading: '#f0f6fc',      // Headings
    placeholder: '#484f58',  // Placeholder text
  },

  // Syntax Highlighting (VSCode Dark+ inspired)
  syntax: {
    keyword: '#ff79c6',      // Keywords (pink)
    function: '#79c0ff',     // Functions (blue)
    string: '#a5d6ff',       // Strings (light blue)
    number: '#79c0ff',       // Numbers (blue)
    comment: '#8b949e',      // Comments (gray)
    variable: '#ffa657',     // Variables (orange)
    type: '#7ee787',         // Types (green)
    operator: '#ff7b72',     // Operators (red)
    property: '#d2a8ff',     // Properties (purple)
    punctuation: '#8b949e',  // Punctuation (gray)
    tag: '#7ee787',          // HTML/XML tags (green)
    attribute: '#79c0ff',    // Attributes (blue)
    class: '#ffa657',        // Classes (orange)
    constant: '#79c0ff',     // Constants (blue)
    regexp: '#7ee787',       // Regular expressions (green)
  },

  // Status Colors
  status: {
    error: '#f85149',
    errorBg: '#f8514926',
    warning: '#d29922',
    warningBg: '#d2992226',
    success: '#3fb950',
    successBg: '#3fb95026',
    info: '#58a6ff',
    infoBg: '#58a6ff26',
  },

  // Git/Diff Colors
  git: {
    added: '#3fb950',
    addedBg: '#2ea04326',
    deleted: '#f85149',
    deletedBg: '#f8514926',
    modified: '#d29922',
    modifiedBg: '#d2992226',
    renamed: '#a371f7',
    untracked: '#8b949e',
  },

  // Version Badge Colors
  version: {
    v1: {
      bg: '#388bfd26',
      text: '#58a6ff',
      border: '#388bfd',
    },
    v2: {
      bg: '#3fb95026',
      text: '#3fb950',
      border: '#3fb950',
    },
    v3: {
      bg: '#f8514926',
      text: '#f85149',
      border: '#f85149',
    },
  },
};

export const spacing = {
  0: '0',
  1: '0.25rem',   // 4px
  2: '0.5rem',    // 8px
  3: '0.75rem',   // 12px
  4: '1rem',      // 16px
  5: '1.25rem',   // 20px
  6: '1.5rem',    // 24px
  8: '2rem',      // 32px
  10: '2.5rem',   // 40px
  12: '3rem',     // 48px
  16: '4rem',     // 64px
  20: '5rem',     // 80px
  24: '6rem',     // 96px
};

export const fontSize = {
  xs: '0.75rem',     // 12px
  sm: '0.8125rem',   // 13px (VSCode default)
  base: '0.875rem',  // 14px
  lg: '1rem',        // 16px
  xl: '1.125rem',    // 18px
  '2xl': '1.25rem',  // 20px
  '3xl': '1.5rem',   // 24px
  '4xl': '1.875rem', // 30px
  '5xl': '2.25rem',  // 36px
};

export const fontFamily = {
  sans: [
    '-apple-system',
    'BlinkMacSystemFont',
    '"Segoe UI"',
    'Helvetica',
    'Arial',
    'sans-serif',
    '"Apple Color Emoji"',
    '"Segoe UI Emoji"',
  ].join(', '),
  mono: [
    '"JetBrains Mono"',
    '"Fira Code"',
    '"Cascadia Code"',
    'Consolas',
    '"SF Mono"',
    'Monaco',
    '"Andale Mono"',
    'monospace',
  ].join(', '),
};

export const borderRadius = {
  none: '0',
  sm: '0.25rem',    // 4px
  md: '0.375rem',   // 6px (GitHub style)
  lg: '0.5rem',     // 8px
  xl: '0.75rem',    // 12px
  full: '9999px',
};

export const boxShadow = {
  none: 'none',
  sm: '0 1px 2px 0 rgb(0 0 0 / 0.05)',
  md: '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)',
  lg: '0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)',
  xl: '0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1)',
  // GitHub-style shadows
  overlay: '0 0 0 1px rgba(48, 54, 61, 0.1), 0 16px 32px rgba(1, 4, 9, 0.85)',
  dropdown: '0 8px 24px rgba(1, 4, 9, 0.75)',
  modal: '0 0 0 1px rgba(48, 54, 61, 0.1), 0 16px 48px rgba(1, 4, 9, 0.9)',
  focus: '0 0 0 3px rgba(56, 139, 253, 0.4)',
};

export const transition = {
  fast: '150ms ease',
  normal: '200ms ease',
  slow: '300ms ease',
  colors: 'color 150ms ease, background-color 150ms ease, border-color 150ms ease',
  transform: 'transform 200ms ease',
  all: 'all 200ms ease',
};

export const zIndex = {
  base: 0,
  dropdown: 1000,
  sticky: 1100,
  modal: 1200,
  popover: 1300,
  tooltip: 1400,
  toast: 1500,
};

// Complete theme object
export const theme = {
  colors,
  spacing,
  fontSize,
  fontFamily,
  borderRadius,
  boxShadow,
  transition,
  zIndex,
};

export type Theme = typeof theme;
export type Colors = typeof colors;

export default theme;
