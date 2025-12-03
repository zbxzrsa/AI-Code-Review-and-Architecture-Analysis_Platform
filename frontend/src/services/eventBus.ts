/**
 * Enhanced Event Bus
 * 
 * Centralized event handling with:
 * - Type-safe events
 * - Event history
 * - Priority listeners
 * - Async event handling
 * - Event filtering
 * - Middleware support
 */

// ============================================
// Types
// ============================================

export type EventPriority = 'low' | 'normal' | 'high' | 'critical';

export interface EventOptions {
  priority?: EventPriority;
  once?: boolean;
  filter?: (payload: any) => boolean;
}

export interface EventListener<T = any> {
  id: string;
  callback: (payload: T) => void | Promise<void>;
  priority: EventPriority;
  once: boolean;
  filter?: (payload: T) => boolean;
}

export interface EventEntry {
  type: string;
  payload: any;
  timestamp: Date;
  source?: string;
}

export type Middleware = (
  event: EventEntry,
  next: () => Promise<void>
) => Promise<void>;

// ============================================
// Event Definitions
// ============================================

export interface EventMap {
  // Analysis events
  'analysis:started': { id: string; file: string };
  'analysis:progress': { id: string; progress: number; file?: string };
  'analysis:completed': { id: string; result: any };
  'analysis:failed': { id: string; error: any };
  'analysis:cancelled': { id: string };

  // Project events
  'project:created': { id: string; name: string };
  'project:updated': { id: string; changes: any };
  'project:deleted': { id: string };
  'project:selected': { id: string };

  // AI events
  'ai:response:started': { requestId: string };
  'ai:response:chunk': { requestId: string; chunk: string };
  'ai:response:completed': { requestId: string; response: string };
  'ai:response:error': { requestId: string; error: any };

  // Auto-fix events
  'fix:suggested': { issueId: string; suggestion: string };
  'fix:applied': { issueId: string; success: boolean };
  'fix:rejected': { issueId: string; reason: string };

  // Security events
  'security:scan:started': { scanId: string };
  'security:scan:completed': { scanId: string; issues: number };
  'security:vulnerability:found': { vulnerability: any };

  // User events
  'user:login': { userId: string };
  'user:logout': { userId: string };
  'user:preferences:changed': { changes: any };

  // System events
  'system:online': {};
  'system:offline': {};
  'system:error': { error: any };
  'system:notification': { type: string; message: string };

  // WebSocket events
  'ws:connected': {};
  'ws:disconnected': { reason?: string };
  'ws:message': { type: string; data: any };
  'ws:error': { error: any };
}

// ============================================
// Priority Values
// ============================================

const priorityValues: Record<EventPriority, number> = {
  low: 0,
  normal: 1,
  high: 2,
  critical: 3,
};

// ============================================
// Event Bus Class
// ============================================

class EventBus {
  private listeners: Map<string, EventListener[]> = new Map();
  private history: EventEntry[] = [];
  private maxHistory: number = 100;
  private middlewares: Middleware[] = [];
  private globalListeners: Set<(event: EventEntry) => void> = new Set();

  // ============================================
  // Subscription Methods
  // ============================================

  public on<K extends keyof EventMap>(
    eventType: K,
    callback: (payload: EventMap[K]) => void | Promise<void>,
    options: EventOptions = {}
  ): () => void {
    const listener: EventListener<EventMap[K]> = {
      id: this.generateId(),
      callback: callback as any,
      priority: options.priority || 'normal',
      once: options.once || false,
      filter: options.filter,
    };

    const listeners = this.listeners.get(eventType) || [];
    listeners.push(listener);
    
    // Sort by priority (highest first)
    listeners.sort((a, b) => priorityValues[b.priority] - priorityValues[a.priority]);
    
    this.listeners.set(eventType, listeners);

    // Return unsubscribe function
    return () => this.off(eventType, listener.id);
  }

  public once<K extends keyof EventMap>(
    eventType: K,
    callback: (payload: EventMap[K]) => void | Promise<void>,
    options: EventOptions = {}
  ): () => void {
    return this.on(eventType, callback, { ...options, once: true });
  }

  public off<K extends keyof EventMap>(eventType: K, listenerId?: string): void {
    if (!listenerId) {
      this.listeners.delete(eventType);
      return;
    }

    const listeners = this.listeners.get(eventType);
    if (listeners) {
      const index = listeners.findIndex(l => l.id === listenerId);
      if (index !== -1) {
        listeners.splice(index, 1);
      }
    }
  }

  // ============================================
  // Emit Methods
  // ============================================

  public async emit<K extends keyof EventMap>(
    eventType: K,
    payload: EventMap[K],
    source?: string
  ): Promise<void> {
    const event: EventEntry = {
      type: eventType,
      payload,
      timestamp: new Date(),
      source,
    };

    // Add to history
    this.addToHistory(event);

    // Notify global listeners
    this.globalListeners.forEach(listener => listener(event));

    // Run through middlewares
    await this.runMiddlewares(event, async () => {
      await this.notifyListeners(eventType, payload);
    });
  }

  public emitSync<K extends keyof EventMap>(
    eventType: K,
    payload: EventMap[K],
    source?: string
  ): void {
    // Fire and forget
    this.emit(eventType, payload, source).catch(console.error);
  }

  private async notifyListeners<K extends keyof EventMap>(
    eventType: K,
    payload: EventMap[K]
  ): Promise<void> {
    const listeners = this.listeners.get(eventType) || [];
    const toRemove: string[] = [];

    for (const listener of listeners) {
      // Check filter
      if (listener.filter && !listener.filter(payload)) {
        continue;
      }

      try {
        await listener.callback(payload);
      } catch (error) {
        console.error(`[EventBus] Error in listener for ${eventType}:`, error);
      }

      if (listener.once) {
        toRemove.push(listener.id);
      }
    }

    // Remove one-time listeners
    toRemove.forEach(id => this.off(eventType, id));
  }

  // ============================================
  // Middleware
  // ============================================

  public use(middleware: Middleware): () => void {
    this.middlewares.push(middleware);
    return () => {
      const index = this.middlewares.indexOf(middleware);
      if (index !== -1) {
        this.middlewares.splice(index, 1);
      }
    };
  }

  private async runMiddlewares(
    event: EventEntry,
    final: () => Promise<void>
  ): Promise<void> {
    let index = 0;

    const next = async (): Promise<void> => {
      if (index < this.middlewares.length) {
        const middleware = this.middlewares[index++];
        await middleware(event, next);
      } else {
        await final();
      }
    };

    await next();
  }

  // ============================================
  // Global Listeners
  // ============================================

  public onAny(callback: (event: EventEntry) => void): () => void {
    this.globalListeners.add(callback);
    return () => this.globalListeners.delete(callback);
  }

  // ============================================
  // History
  // ============================================

  private addToHistory(event: EventEntry): void {
    this.history.unshift(event);
    if (this.history.length > this.maxHistory) {
      this.history.pop();
    }
  }

  public getHistory(eventType?: keyof EventMap): EventEntry[] {
    if (eventType) {
      return this.history.filter(e => e.type === eventType);
    }
    return [...this.history];
  }

  public clearHistory(): void {
    this.history = [];
  }

  // ============================================
  // Waiting for Events
  // ============================================

  public waitFor<K extends keyof EventMap>(
    eventType: K,
    timeout?: number,
    filter?: (payload: EventMap[K]) => boolean
  ): Promise<EventMap[K]> {
    return new Promise((resolve, reject) => {
      let timeoutId: ReturnType<typeof setTimeout> | null = null;

      const unsubscribe = this.on(
        eventType,
        (payload) => {
          if (timeoutId) clearTimeout(timeoutId);
          resolve(payload);
        },
        { once: true, filter }
      );

      if (timeout) {
        timeoutId = setTimeout(() => {
          unsubscribe();
          reject(new Error(`Timeout waiting for ${eventType}`));
        }, timeout);
      }
    });
  }

  // ============================================
  // Event Batching
  // ============================================

  private batchedEvents: Map<string, { events: any[]; timeout: ReturnType<typeof setTimeout> }> = new Map();

  public emitBatched<K extends keyof EventMap>(
    eventType: K,
    payload: EventMap[K],
    debounceMs: number = 100
  ): void {
    const key = eventType as string;
    const existing = this.batchedEvents.get(key);

    if (existing) {
      existing.events.push(payload);
      clearTimeout(existing.timeout);
    } else {
      this.batchedEvents.set(key, {
        events: [payload],
        timeout: setTimeout(() => {}, 0),
      });
    }

    const batch = this.batchedEvents.get(key)!;
    batch.timeout = setTimeout(() => {
      const events = batch.events;
      this.batchedEvents.delete(key);
      
      // Emit all batched events
      events.forEach(event => {
        this.emit(eventType, event);
      });
    }, debounceMs);
  }

  // ============================================
  // Utilities
  // ============================================

  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  public getListenerCount(eventType?: keyof EventMap): number {
    if (eventType) {
      return this.listeners.get(eventType)?.length || 0;
    }
    
    let total = 0;
    this.listeners.forEach(listeners => {
      total += listeners.length;
    });
    return total;
  }

  public clear(): void {
    this.listeners.clear();
    this.history = [];
    this.globalListeners.clear();
    this.batchedEvents.clear();
  }
}

// ============================================
// Export Singleton
// ============================================

export const eventBus = new EventBus();

// ============================================
// React Hook
// ============================================

import { useEffect, useCallback as useReactCallback } from 'react';

export function useEvent<K extends keyof EventMap>(
  eventType: K,
  callback: (payload: EventMap[K]) => void,
  deps: React.DependencyList = []
): void {
  useEffect(() => {
    const unsubscribe = eventBus.on(eventType, callback);
    return unsubscribe;
  }, [eventType, ...deps]);
}

export function useEventEmitter() {
  return useReactCallback(<K extends keyof EventMap>(
    eventType: K,
    payload: EventMap[K]
  ) => {
    eventBus.emit(eventType, payload);
  }, []);
}

export default eventBus;
