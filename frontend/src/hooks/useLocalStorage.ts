/**
 * Local Storage Hook
 * 
 * Persistent storage with:
 * - Type safety
 * - Default values
 * - Expiration support
 * - Cross-tab sync
 * - Compression option
 */

import { useState, useCallback, useEffect } from 'react';

// ============================================
// Types
// ============================================

interface StorageOptions<T> {
  defaultValue?: T;
  expiration?: number; // milliseconds
  serialize?: (value: T) => string;
  deserialize?: (value: string) => T;
  sync?: boolean; // sync across tabs
}

interface StoredValue<T> {
  value: T;
  timestamp: number;
  expiration?: number;
}

// ============================================
// Main Hook
// ============================================

export function useLocalStorage<T>(
  key: string,
  options: StorageOptions<T> = {}
): [T | undefined, (value: T | ((prev: T | undefined) => T)) => void, () => void] {
  const {
    defaultValue,
    expiration,
    serialize = JSON.stringify,
    deserialize = JSON.parse,
    sync = true,
  } = options;

  const prefixedKey = `crai_${key}`;

  // Initialize state
  const [storedValue, setStoredValue] = useState<T | undefined>(() => {
    try {
      const item = localStorage.getItem(prefixedKey);
      if (!item) return defaultValue;

      const parsed: StoredValue<T> = deserialize(item);

      // Check expiration
      if (parsed.expiration && Date.now() > parsed.timestamp + parsed.expiration) {
        localStorage.removeItem(prefixedKey);
        return defaultValue;
      }

      return parsed.value;
    } catch (error) {
      console.error(`[useLocalStorage] Error reading ${key}:`, error);
      return defaultValue;
    }
  });

  // Set value
  const setValue = useCallback((value: T | ((prev: T | undefined) => T)) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;

      setStoredValue(valueToStore);

      const stored: StoredValue<T> = {
        value: valueToStore,
        timestamp: Date.now(),
        expiration,
      };

      localStorage.setItem(prefixedKey, serialize(stored));

      // Dispatch event for cross-tab sync
      if (sync) {
        window.dispatchEvent(new StorageEvent('storage', {
          key: prefixedKey,
          newValue: serialize(stored),
        }));
      }
    } catch (error) {
      console.error(`[useLocalStorage] Error writing ${key}:`, error);
    }
  }, [prefixedKey, storedValue, expiration, serialize, sync]);

  // Remove value
  const removeValue = useCallback(() => {
    try {
      localStorage.removeItem(prefixedKey);
      setStoredValue(defaultValue);
    } catch (error) {
      console.error(`[useLocalStorage] Error removing ${key}:`, error);
    }
  }, [prefixedKey, defaultValue]);

  // Cross-tab sync
  useEffect(() => {
    if (!sync) return;

    const handleStorageChange = (event: StorageEvent) => {
      if (event.key !== prefixedKey) return;

      try {
        if (event.newValue) {
          const parsed: StoredValue<T> = deserialize(event.newValue);
          setStoredValue(parsed.value);
        } else {
          setStoredValue(defaultValue);
        }
      } catch (error) {
        console.error(`[useLocalStorage] Error syncing ${key}:`, error);
      }
    };

    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, [prefixedKey, deserialize, defaultValue, sync]);

  return [storedValue, setValue, removeValue];
}

// ============================================
// Session Storage Hook
// ============================================

export function useSessionStorage<T>(
  key: string,
  defaultValue?: T
): [T | undefined, (value: T | ((prev: T | undefined) => T)) => void, () => void] {
  const prefixedKey = `crai_${key}`;

  const [storedValue, setStoredValue] = useState<T | undefined>(() => {
    try {
      const item = sessionStorage.getItem(prefixedKey);
      return item ? JSON.parse(item) : defaultValue;
    } catch {
      return defaultValue;
    }
  });

  const setValue = useCallback((value: T | ((prev: T | undefined) => T)) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      sessionStorage.setItem(prefixedKey, JSON.stringify(valueToStore));
    } catch (error) {
      console.error(`[useSessionStorage] Error writing ${key}:`, error);
    }
  }, [prefixedKey, storedValue]);

  const removeValue = useCallback(() => {
    try {
      sessionStorage.removeItem(prefixedKey);
      setStoredValue(defaultValue);
    } catch (error) {
      console.error(`[useSessionStorage] Error removing ${key}:`, error);
    }
  }, [prefixedKey, defaultValue]);

  return [storedValue, setValue, removeValue];
}

// ============================================
// Map Storage Hook (for object collections)
// ============================================

export function useStorageMap<V>(
  key: string,
  defaultValue: Map<string, V> = new Map()
): {
  map: Map<string, V>;
  get: (id: string) => V | undefined;
  set: (id: string, value: V) => void;
  remove: (id: string) => void;
  clear: () => void;
  has: (id: string) => boolean;
  size: number;
} {
  const [stored, setStored, removeStored] = useLocalStorage<[string, V][]>(key, {
    defaultValue: Array.from(defaultValue),
  });

  const map = new Map<string, V>(stored || []);

  return {
    map,
    get: (id: string) => map.get(id),
    set: (id: string, value: V) => {
      map.set(id, value);
      setStored(Array.from(map));
    },
    remove: (id: string) => {
      map.delete(id);
      setStored(Array.from(map));
    },
    clear: () => {
      removeStored();
    },
    has: (id: string) => map.has(id),
    size: map.size,
  };
}

// ============================================
// Recent Items Hook
// ============================================

export function useRecentItems<T extends { id: string }>(
  key: string,
  maxItems: number = 10
): {
  items: T[];
  add: (item: T) => void;
  remove: (id: string) => void;
  clear: () => void;
} {
  const [items, setItems] = useLocalStorage<T[]>(key, {
    defaultValue: [],
  });

  const add = useCallback((item: T) => {
    setItems(prev => {
      const filtered = (prev || []).filter(i => i.id !== item.id);
      return [item, ...filtered].slice(0, maxItems);
    });
  }, [setItems, maxItems]);

  const remove = useCallback((id: string) => {
    setItems(prev => (prev || []).filter(i => i.id !== id));
  }, [setItems]);

  const clear = useCallback(() => {
    setItems([]);
  }, [setItems]);

  return {
    items: items || [],
    add,
    remove,
    clear,
  };
}

// ============================================
// Preferences Hook
// ============================================

export interface UserPreferences {
  theme: 'light' | 'dark' | 'system';
  language: string;
  fontSize: 'small' | 'medium' | 'large';
  sidebarCollapsed: boolean;
  showLineNumbers: boolean;
  autoSave: boolean;
  notifications: {
    email: boolean;
    push: boolean;
    sound: boolean;
  };
}

const defaultPreferences: UserPreferences = {
  theme: 'system',
  language: 'en',
  fontSize: 'medium',
  sidebarCollapsed: false,
  showLineNumbers: true,
  autoSave: true,
  notifications: {
    email: true,
    push: true,
    sound: true,
  },
};

export function usePreferences(): [
  UserPreferences,
  <K extends keyof UserPreferences>(key: K, value: UserPreferences[K]) => void,
  () => void
] {
  const [preferences, setPreferences, resetPreferences] = useLocalStorage<UserPreferences>(
    'user_preferences',
    { defaultValue: defaultPreferences, sync: true }
  );

  const updatePreference = useCallback(<K extends keyof UserPreferences>(
    key: K,
    value: UserPreferences[K]
  ) => {
    setPreferences(prev => ({
      ...(prev || defaultPreferences),
      [key]: value,
    }));
  }, [setPreferences]);

  const reset = useCallback(() => {
    setPreferences(defaultPreferences);
  }, [setPreferences]);

  return [preferences || defaultPreferences, updatePreference, reset];
}

// ============================================
// Storage Quota Hook
// ============================================

export function useStorageQuota(): {
  used: number;
  total: number;
  percentage: number;
  available: number;
} {
  const [quota, setQuota] = useState({
    used: 0,
    total: 5 * 1024 * 1024, // 5MB default
    percentage: 0,
    available: 5 * 1024 * 1024,
  });

  useEffect(() => {
    // Calculate used storage
    let used = 0;
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key) {
        const value = localStorage.getItem(key);
        if (value) {
          used += key.length + value.length;
        }
      }
    }

    // Estimate total (most browsers allow 5-10MB)
    const total = 5 * 1024 * 1024;

    setQuota({
      used,
      total,
      percentage: (used / total) * 100,
      available: total - used,
    });
  }, []);

  return quota;
}

export default useLocalStorage;
