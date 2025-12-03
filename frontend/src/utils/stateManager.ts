/**
 * Enhanced State Manager
 * 
 * Utility functions for state management:
 * - Local storage with encryption
 * - Session management
 * - State synchronization
 * - Undo/Redo support
 * - State snapshots
 */

// ============================================
// Types
// ============================================

export interface StateOptions {
  encrypt?: boolean;
  compress?: boolean;
  ttl?: number;
  version?: number;
}

interface StoredState<T> {
  data: T;
  version: number;
  timestamp: number;
  ttl?: number;
}

interface HistoryEntry<T> {
  state: T;
  timestamp: number;
  label?: string;
}

// ============================================
// Simple Encryption (for non-sensitive data)
// ============================================

const encryptionKey = 'crai-state-key-2024';

function simpleEncrypt(text: string): string {
  let result = '';
  for (let i = 0; i < text.length; i++) {
    result += String.fromCharCode(
      text.charCodeAt(i) ^ encryptionKey.charCodeAt(i % encryptionKey.length)
    );
  }
  return btoa(result);
}

function simpleDecrypt(encoded: string): string {
  const text = atob(encoded);
  let result = '';
  for (let i = 0; i < text.length; i++) {
    result += String.fromCharCode(
      text.charCodeAt(i) ^ encryptionKey.charCodeAt(i % encryptionKey.length)
    );
  }
  return result;
}

// ============================================
// Persistence Manager
// ============================================

export class PersistenceManager<T> {
  private key: string;
  private options: StateOptions;
  private currentVersion: number;

  constructor(key: string, options: StateOptions = {}) {
    this.key = `crai_${key}`;
    this.options = options;
    this.currentVersion = options.version ?? 1;
  }

  public save(data: T): boolean {
    try {
      const stored: StoredState<T> = {
        data,
        version: this.currentVersion,
        timestamp: Date.now(),
        ttl: this.options.ttl,
      };

      let serialized = JSON.stringify(stored);

      if (this.options.encrypt) {
        serialized = simpleEncrypt(serialized);
      }

      localStorage.setItem(this.key, serialized);
      return true;
    } catch (e) {
      console.error(`[StateManager] Failed to save ${this.key}:`, e);
      return false;
    }
  }

  public load(): T | null {
    try {
      let serialized = localStorage.getItem(this.key);
      if (!serialized) return null;

      if (this.options.encrypt) {
        serialized = simpleDecrypt(serialized);
      }

      const stored: StoredState<T> = JSON.parse(serialized);

      // Check version
      if (stored.version !== this.currentVersion) {
        console.warn(`[StateManager] Version mismatch for ${this.key}, clearing`);
        this.clear();
        return null;
      }

      // Check TTL
      if (stored.ttl && Date.now() - stored.timestamp > stored.ttl) {
        console.warn(`[StateManager] Data expired for ${this.key}, clearing`);
        this.clear();
        return null;
      }

      return stored.data;
    } catch (e) {
      console.error(`[StateManager] Failed to load ${this.key}:`, e);
      return null;
    }
  }

  public clear(): void {
    localStorage.removeItem(this.key);
  }

  public exists(): boolean {
    return localStorage.getItem(this.key) !== null;
  }
}

// ============================================
// History Manager (Undo/Redo)
// ============================================

export class HistoryManager<T> {
  private history: HistoryEntry<T>[] = [];
  private currentIndex: number = -1;
  private maxHistory: number;
  private listeners: Set<(state: T | null, canUndo: boolean, canRedo: boolean) => void> = new Set();

  constructor(maxHistory: number = 50) {
    this.maxHistory = maxHistory;
  }

  public push(state: T, label?: string): void {
    // Remove any future history if we're not at the end
    if (this.currentIndex < this.history.length - 1) {
      this.history = this.history.slice(0, this.currentIndex + 1);
    }

    this.history.push({
      state: this.deepClone(state),
      timestamp: Date.now(),
      label,
    });

    // Trim history if too long
    if (this.history.length > this.maxHistory) {
      this.history.shift();
    } else {
      this.currentIndex++;
    }

    this.notifyListeners();
  }

  public undo(): T | null {
    if (!this.canUndo()) return null;

    this.currentIndex--;
    const entry = this.history[this.currentIndex];
    this.notifyListeners();
    return this.deepClone(entry.state);
  }

  public redo(): T | null {
    if (!this.canRedo()) return null;

    this.currentIndex++;
    const entry = this.history[this.currentIndex];
    this.notifyListeners();
    return this.deepClone(entry.state);
  }

  public canUndo(): boolean {
    return this.currentIndex > 0;
  }

  public canRedo(): boolean {
    return this.currentIndex < this.history.length - 1;
  }

  public getCurrent(): T | null {
    if (this.currentIndex < 0) return null;
    return this.deepClone(this.history[this.currentIndex].state);
  }

  public getHistory(): HistoryEntry<T>[] {
    return this.history.map(entry => ({
      ...entry,
      state: this.deepClone(entry.state),
    }));
  }

  public clear(): void {
    this.history = [];
    this.currentIndex = -1;
    this.notifyListeners();
  }

  public subscribe(
    callback: (state: T | null, canUndo: boolean, canRedo: boolean) => void
  ): () => void {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }

  private notifyListeners(): void {
    const current = this.getCurrent();
    this.listeners.forEach(callback => 
      callback(current, this.canUndo(), this.canRedo())
    );
  }

  private deepClone(obj: T): T {
    return JSON.parse(JSON.stringify(obj));
  }
}

// ============================================
// State Synchronizer (Cross-Tab)
// ============================================

export class StateSynchronizer<T> {
  private channel: BroadcastChannel | null = null;
  private key: string;
  private listeners: Set<(state: T) => void> = new Set();

  constructor(channelName: string) {
    this.key = channelName;
    
    if ('BroadcastChannel' in window) {
      this.channel = new BroadcastChannel(`crai_sync_${channelName}`);
      this.channel.onmessage = (event) => {
        this.notifyListeners(event.data);
      };
    }
  }

  public broadcast(state: T): void {
    if (this.channel) {
      this.channel.postMessage(state);
    }
  }

  public subscribe(callback: (state: T) => void): () => void {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }

  private notifyListeners(state: T): void {
    this.listeners.forEach(callback => callback(state));
  }

  public close(): void {
    if (this.channel) {
      this.channel.close();
      this.channel = null;
    }
  }
}

// ============================================
// Debounced State
// ============================================

export function createDebouncedState<T>(
  initialValue: T,
  delay: number = 300
): {
  get: () => T;
  set: (value: T) => void;
  setImmediate: (value: T) => void;
  subscribe: (callback: (value: T) => void) => () => void;
} {
  let currentValue = initialValue;
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  const listeners = new Set<(value: T) => void>();

  const notify = () => {
    listeners.forEach(callback => callback(currentValue));
  };

  return {
    get: () => currentValue,
    set: (value: T) => {
      currentValue = value;
      if (timeoutId) clearTimeout(timeoutId);
      timeoutId = setTimeout(notify, delay);
    },
    setImmediate: (value: T) => {
      currentValue = value;
      if (timeoutId) clearTimeout(timeoutId);
      notify();
    },
    subscribe: (callback: (value: T) => void) => {
      listeners.add(callback);
      return () => listeners.delete(callback);
    },
  };
}

// ============================================
// Computed State
// ============================================

export function createComputedState<T, R>(
  dependencies: (() => T)[],
  compute: (...values: T[]) => R
): () => R {
  let cachedResult: R | undefined;
  let cachedDeps: T[] | undefined;

  return () => {
    const currentDeps = dependencies.map(dep => dep());
    
    if (cachedDeps && currentDeps.every((dep, i) => dep === cachedDeps![i])) {
      return cachedResult!;
    }

    cachedDeps = currentDeps;
    cachedResult = compute(...currentDeps);
    return cachedResult;
  };
}

// ============================================
// Form State Manager
// ============================================

export interface FormState<T> {
  values: T;
  errors: Partial<Record<keyof T, string>>;
  touched: Partial<Record<keyof T, boolean>>;
  isValid: boolean;
  isDirty: boolean;
  isSubmitting: boolean;
}

export function createFormState<T extends Record<string, any>>(
  initialValues: T,
  validate?: (values: T) => Partial<Record<keyof T, string>>
): {
  getState: () => FormState<T>;
  setValue: <K extends keyof T>(field: K, value: T[K]) => void;
  setValues: (values: Partial<T>) => void;
  setTouched: (field: keyof T) => void;
  reset: () => void;
  submit: () => Promise<T | null>;
  subscribe: (callback: (state: FormState<T>) => void) => () => void;
} {
  let state: FormState<T> = {
    values: { ...initialValues },
    errors: {},
    touched: {},
    isValid: true,
    isDirty: false,
    isSubmitting: false,
  };

  const listeners = new Set<(state: FormState<T>) => void>();

  const validateForm = () => {
    if (validate) {
      state.errors = validate(state.values);
      state.isValid = Object.keys(state.errors).length === 0;
    }
  };

  const notify = () => {
    listeners.forEach(callback => callback({ ...state }));
  };

  return {
    getState: () => ({ ...state }),
    
    setValue: <K extends keyof T>(field: K, value: T[K]) => {
      state.values[field] = value;
      state.isDirty = true;
      validateForm();
      notify();
    },

    setValues: (values: Partial<T>) => {
      state.values = { ...state.values, ...values };
      state.isDirty = true;
      validateForm();
      notify();
    },

    setTouched: (field: keyof T) => {
      state.touched[field] = true;
      notify();
    },

    reset: () => {
      state = {
        values: { ...initialValues },
        errors: {},
        touched: {},
        isValid: true,
        isDirty: false,
        isSubmitting: false,
      };
      notify();
    },

    submit: async () => {
      validateForm();
      
      // Touch all fields
      Object.keys(state.values).forEach(key => {
        state.touched[key as keyof T] = true;
      });

      if (!state.isValid) {
        notify();
        return null;
      }

      state.isSubmitting = true;
      notify();

      return state.values;
    },

    subscribe: (callback: (state: FormState<T>) => void) => {
      listeners.add(callback);
      return () => listeners.delete(callback);
    },
  };
}

// ============================================
// Export Utilities
// ============================================

export const stateUtils = {
  PersistenceManager,
  HistoryManager,
  StateSynchronizer,
  createDebouncedState,
  createComputedState,
  createFormState,
};

export default stateUtils;
