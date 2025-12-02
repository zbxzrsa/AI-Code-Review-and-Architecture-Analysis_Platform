/**
 * Storage service for managing local and session storage
 * with type safety and expiration support
 */

interface StorageItem<T> {
  value: T;
  expiresAt?: number;
  createdAt: number;
}

type StorageType = 'local' | 'session';

class StorageService {
  private getStorage(type: StorageType): Storage {
    return type === 'local' ? localStorage : sessionStorage;
  }

  /**
   * Set an item in storage
   */
  set<T>(
    key: string,
    value: T,
    options: { type?: StorageType; expiresIn?: number } = {}
  ): void {
    const { type = 'local', expiresIn } = options;
    const storage = this.getStorage(type);

    const item: StorageItem<T> = {
      value,
      createdAt: Date.now(),
      expiresAt: expiresIn ? Date.now() + expiresIn : undefined
    };

    try {
      storage.setItem(key, JSON.stringify(item));
    } catch (error) {
      console.error(`Failed to set storage item: ${key}`, error);
      // Handle quota exceeded
      if (error instanceof DOMException && error.name === 'QuotaExceededError') {
        this.clearExpired(type);
        try {
          storage.setItem(key, JSON.stringify(item));
        } catch {
          console.error('Storage quota exceeded even after cleanup');
        }
      }
    }
  }

  /**
   * Get an item from storage
   */
  get<T>(key: string, options: { type?: StorageType } = {}): T | null {
    const { type = 'local' } = options;
    const storage = this.getStorage(type);

    try {
      const data = storage.getItem(key);
      if (!data) return null;

      const item: StorageItem<T> = JSON.parse(data);

      // Check expiration
      if (item.expiresAt && item.expiresAt < Date.now()) {
        this.remove(key, { type });
        return null;
      }

      return item.value;
    } catch (error) {
      console.error(`Failed to get storage item: ${key}`, error);
      return null;
    }
  }

  /**
   * Remove an item from storage
   */
  remove(key: string, options: { type?: StorageType } = {}): void {
    const { type = 'local' } = options;
    const storage = this.getStorage(type);
    storage.removeItem(key);
  }

  /**
   * Check if an item exists in storage
   */
  has(key: string, options: { type?: StorageType } = {}): boolean {
    return this.get(key, options) !== null;
  }

  /**
   * Clear all items from storage
   */
  clear(options: { type?: StorageType } = {}): void {
    const { type = 'local' } = options;
    const storage = this.getStorage(type);
    storage.clear();
  }

  /**
   * Clear expired items from storage
   */
  clearExpired(type: StorageType = 'local'): void {
    const storage = this.getStorage(type);
    const keysToRemove: string[] = [];

    for (let i = 0; i < storage.length; i++) {
      const key = storage.key(i);
      if (!key) continue;

      try {
        const data = storage.getItem(key);
        if (!data) continue;

        const item: StorageItem<unknown> = JSON.parse(data);
        if (item.expiresAt && item.expiresAt < Date.now()) {
          keysToRemove.push(key);
        }
      } catch {
        // Skip items that can't be parsed
      }
    }

    keysToRemove.forEach((key) => storage.removeItem(key));
  }

  /**
   * Get all keys in storage
   */
  keys(options: { type?: StorageType } = {}): string[] {
    const { type = 'local' } = options;
    const storage = this.getStorage(type);
    const keys: string[] = [];

    for (let i = 0; i < storage.length; i++) {
      const key = storage.key(i);
      if (key) keys.push(key);
    }

    return keys;
  }

  /**
   * Get storage size in bytes
   */
  getSize(options: { type?: StorageType } = {}): number {
    const { type = 'local' } = options;
    const storage = this.getStorage(type);
    let size = 0;

    for (let i = 0; i < storage.length; i++) {
      const key = storage.key(i);
      if (key) {
        const value = storage.getItem(key);
        if (value) {
          size += key.length + value.length;
        }
      }
    }

    return size * 2; // UTF-16 encoding
  }
}

export const storage = new StorageService();

// Convenience methods for common storage keys
export const storageKeys = {
  AUTH_TOKEN: 'auth_token',
  REFRESH_TOKEN: 'refresh_token',
  USER: 'user',
  THEME: 'theme',
  LANGUAGE: 'language',
  RECENT_PROJECTS: 'recent_projects',
  EDITOR_SETTINGS: 'editor_settings',
  SIDEBAR_STATE: 'sidebar_state'
} as const;

// Typed storage helpers
export const authStorage = {
  getToken: () => storage.get<string>(storageKeys.AUTH_TOKEN),
  setToken: (token: string, expiresIn?: number) =>
    storage.set(storageKeys.AUTH_TOKEN, token, { expiresIn }),
  removeToken: () => storage.remove(storageKeys.AUTH_TOKEN),

  getRefreshToken: () => storage.get<string>(storageKeys.REFRESH_TOKEN),
  setRefreshToken: (token: string) =>
    storage.set(storageKeys.REFRESH_TOKEN, token),
  removeRefreshToken: () => storage.remove(storageKeys.REFRESH_TOKEN),

  clearAuth: () => {
    storage.remove(storageKeys.AUTH_TOKEN);
    storage.remove(storageKeys.REFRESH_TOKEN);
    storage.remove(storageKeys.USER);
  }
};

export const preferencesStorage = {
  getTheme: () => storage.get<'light' | 'dark' | 'system'>(storageKeys.THEME) || 'system',
  setTheme: (theme: 'light' | 'dark' | 'system') =>
    storage.set(storageKeys.THEME, theme),

  getLanguage: () => storage.get<string>(storageKeys.LANGUAGE) || 'en',
  setLanguage: (language: string) =>
    storage.set(storageKeys.LANGUAGE, language),

  getEditorSettings: () =>
    storage.get<Record<string, unknown>>(storageKeys.EDITOR_SETTINGS) || {},
  setEditorSettings: (settings: Record<string, unknown>) =>
    storage.set(storageKeys.EDITOR_SETTINGS, settings)
};

export const projectStorage = {
  getRecentProjects: () =>
    storage.get<string[]>(storageKeys.RECENT_PROJECTS) || [],
  
  addRecentProject: (projectId: string, maxItems = 10) => {
    const recent = projectStorage.getRecentProjects();
    const updated = [projectId, ...recent.filter((id) => id !== projectId)].slice(0, maxItems);
    storage.set(storageKeys.RECENT_PROJECTS, updated);
  },
  
  removeRecentProject: (projectId: string) => {
    const recent = projectStorage.getRecentProjects();
    storage.set(storageKeys.RECENT_PROJECTS, recent.filter((id) => id !== projectId));
  },
  
  clearRecentProjects: () => storage.remove(storageKeys.RECENT_PROJECTS)
};

export default storage;
