/**
 * Keyboard Shortcuts Hook
 *
 * Global keyboard shortcuts management:
 * - Configurable shortcuts
 * - Conflict detection
 * - Context-aware shortcuts
 * - Shortcut hints
 */

import { useEffect, useCallback, useRef } from "react";

// ============================================
// Types
// ============================================

export interface ShortcutConfig {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  meta?: boolean;
  description: string;
  action: () => void;
  enabled?: boolean;
  context?: string;
}

export interface ShortcutGroup {
  name: string;
  shortcuts: ShortcutConfig[];
}

// ============================================
// Default Shortcuts
// ============================================

export const defaultShortcuts: ShortcutGroup[] = [
  {
    name: "Navigation",
    shortcuts: [
      { key: "g", alt: true, description: "Go to Dashboard", action: () => {} },
      { key: "p", alt: true, description: "Go to Projects", action: () => {} },
      {
        key: "r",
        alt: true,
        description: "Go to Code Review",
        action: () => {},
      },
      { key: "s", alt: true, description: "Go to Settings", action: () => {} },
    ],
  },
  {
    name: "Actions",
    shortcuts: [
      {
        key: "k",
        ctrl: true,
        description: "Open Command Palette",
        action: () => {},
      },
      { key: "n", ctrl: true, description: "New Project", action: () => {} },
      { key: "s", ctrl: true, description: "Save", action: () => {} },
      { key: "f", ctrl: true, description: "Search", action: () => {} },
    ],
  },
  {
    name: "Code Review",
    shortcuts: [
      {
        key: "Enter",
        ctrl: true,
        description: "Run Analysis",
        action: () => {},
        context: "code-review",
      },
      {
        key: "f",
        ctrl: true,
        shift: true,
        description: "Apply Auto-Fix",
        action: () => {},
        context: "code-review",
      },
      {
        key: "ArrowDown",
        alt: true,
        description: "Next Issue",
        action: () => {},
        context: "code-review",
      },
      {
        key: "ArrowUp",
        alt: true,
        description: "Previous Issue",
        action: () => {},
        context: "code-review",
      },
    ],
  },
];

// ============================================
// Hook Implementation
// ============================================

export function useKeyboardShortcuts(
  shortcuts: ShortcutConfig[] = [],
  options: {
    enabled?: boolean;
    context?: string;
  } = {}
) {
  const { enabled = true, context } = options;
  const shortcutsRef = useRef<ShortcutConfig[]>(shortcuts);

  // Update shortcuts ref when shortcuts change
  useEffect(() => {
    shortcutsRef.current = shortcuts;
  }, [shortcuts]);

  // Handle keydown event
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      // Skip if typing in an input
      const target = event.target as HTMLElement;
      if (
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.isContentEditable
      ) {
        // Allow some shortcuts even in inputs
        const allowedInInput = ["Escape", "Enter"];
        if (
          !allowedInInput.includes(event.key) &&
          !event.ctrlKey &&
          !event.metaKey
        ) {
          return;
        }
      }

      // Find matching shortcut
      const shortcut = shortcutsRef.current.find((s) => {
        if (s.enabled === false) return false;
        if (s.context && s.context !== context) return false;

        const keyMatch = s.key.toLowerCase() === event.key.toLowerCase();
        const ctrlMatch = !!s.ctrl === (event.ctrlKey || event.metaKey);
        const shiftMatch = !!s.shift === event.shiftKey;
        const altMatch = !!s.alt === event.altKey;

        return keyMatch && ctrlMatch && shiftMatch && altMatch;
      });

      if (shortcut) {
        event.preventDefault();
        event.stopPropagation();
        shortcut.action();
      }
    },
    [context]
  );

  // Add event listener
  useEffect(() => {
    if (!enabled) return;

    globalThis.addEventListener("keydown", handleKeyDown);
    return () => globalThis.removeEventListener("keydown", handleKeyDown);
  }, [enabled, handleKeyDown]);

  // Return utility functions
  return {
    registerShortcut: useCallback((config: ShortcutConfig) => {
      shortcutsRef.current.push(config);
    }, []),

    unregisterShortcut: useCallback((key: string) => {
      shortcutsRef.current = shortcutsRef.current.filter((s) => s.key !== key);
    }, []),

    getShortcuts: useCallback(() => shortcutsRef.current, []),

    formatShortcut: useCallback((config: ShortcutConfig): string => {
      const parts: string[] = [];
      if (config.ctrl) parts.push("Ctrl");
      if (config.alt) parts.push("Alt");
      if (config.shift) parts.push("Shift");
      if (config.meta) parts.push("⌘");
      parts.push(
        config.key.length === 1 ? config.key.toUpperCase() : config.key
      );
      return parts.join("+");
    }, []),
  };
}

// ============================================
// Global Shortcuts Hook
// ============================================

export function useGlobalShortcuts(navigate: (path: string) => void) {
  const shortcuts: ShortcutConfig[] = [
    // Navigation
    {
      key: "g",
      alt: true,
      description: "Go to Dashboard",
      action: () => navigate("/dashboard"),
    },
    {
      key: "p",
      alt: true,
      description: "Go to Projects",
      action: () => navigate("/projects"),
    },
    {
      key: "r",
      alt: true,
      description: "Go to Code Review",
      action: () => navigate("/review"),
    },
    {
      key: "s",
      alt: true,
      description: "Go to Settings",
      action: () => navigate("/settings"),
    },
    {
      key: "w",
      alt: true,
      description: "Go to Welcome",
      action: () => navigate("/welcome"),
    },

    // Actions
    {
      key: "k",
      ctrl: true,
      description: "Open Command Palette",
      action: () => {
        // Dispatch custom event for command palette
        globalThis.dispatchEvent(new CustomEvent("open-command-palette"));
      },
    },
    {
      key: "Escape",
      description: "Close modal/panel",
      action: () => {
        globalThis.dispatchEvent(new CustomEvent("close-modal"));
      },
    },

    // Help
    {
      key: "?",
      shift: true,
      description: "Show shortcuts",
      action: () => {
        globalThis.dispatchEvent(new CustomEvent("show-shortcuts"));
      },
    },
  ];

  return useKeyboardShortcuts(shortcuts);
}

// ============================================
// Shortcut Display Component Helper
// ============================================

export function formatShortcutKey(config: ShortcutConfig): string[] {
  const keys: string[] = [];

  // Use userAgentData when available, fallback to userAgent check
  const isMac =
    navigator.userAgentData?.platform === "macOS" ||
    navigator.userAgent.includes("Mac");
  if (config.ctrl) keys.push(isMac ? "⌘" : "Ctrl");
  if (config.alt) keys.push(isMac ? "⌥" : "Alt");
  if (config.shift) keys.push("⇧");

  // Format key name
  let keyName = config.key;
  if (keyName === " ") keyName = "Space";
  else if (keyName === "ArrowUp") keyName = "↑";
  else if (keyName === "ArrowDown") keyName = "↓";
  else if (keyName === "ArrowLeft") keyName = "←";
  else if (keyName === "ArrowRight") keyName = "→";
  else if (keyName === "Enter") keyName = "↵";
  else if (keyName === "Escape") keyName = "Esc";
  else if (keyName.length === 1) keyName = keyName.toUpperCase();

  keys.push(keyName);

  return keys;
}

export default useKeyboardShortcuts;
