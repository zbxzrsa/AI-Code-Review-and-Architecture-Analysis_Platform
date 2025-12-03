import { useEffect, useState, useCallback } from 'react';
import { openDB, DBSchema, IDBPDatabase } from 'idb';

interface PendingOperation {
  id?: number;
  url: string;
  method: string;
  data: unknown;
  timestamp: number;
  synced: boolean;
  retryCount: number;
  lastError?: string;
}

interface CachedResponse {
  url: string;
  data: unknown;
  timestamp: number;
  expiresAt: number;
}

interface OfflineDB extends DBSchema {
  'pending-operations': {
    key: number;
    value: PendingOperation;
    indexes: {
      'by-synced': number;
      'by-timestamp': number;
    };
  };
  'cached-responses': {
    key: string;
    value: CachedResponse;
  };
}

interface OfflineSyncOptions {
  dbName?: string;
  dbVersion?: number;
  maxRetries?: number;
  cacheExpiry?: number; // in milliseconds
}

const defaultOptions: OfflineSyncOptions = {
  dbName: 'code-review-offline',
  dbVersion: 1,
  maxRetries: 3,
  cacheExpiry: 24 * 60 * 60 * 1000 // 24 hours
};

export function useOfflineSync(options: OfflineSyncOptions = {}) {
  const opts = { ...defaultOptions, ...options };
  
  const [db, setDb] = useState<IDBPDatabase<OfflineDB> | null>(null);
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [isSyncing, setIsSyncing] = useState(false);
  const [pendingCount, setPendingCount] = useState(0);

  // Initialize IndexedDB
  useEffect(() => {
    const initDB = async () => {
      try {
        const database = await openDB<OfflineDB>(opts.dbName!, opts.dbVersion!, {
          upgrade(db) {
            // Pending operations store
            if (!db.objectStoreNames.contains('pending-operations')) {
              const store = db.createObjectStore('pending-operations', {
                keyPath: 'id',
                autoIncrement: true
              });
              store.createIndex('by-synced', 'synced');
              store.createIndex('by-timestamp', 'timestamp');
            }

            // Cached responses store
            if (!db.objectStoreNames.contains('cached-responses')) {
              db.createObjectStore('cached-responses', {
                keyPath: 'url'
              });
            }
          }
        });
        setDb(database);
        
        // Update pending count (0 = not synced, 1 = synced)
        const pending = await database.getAllFromIndex(
          'pending-operations',
          'by-synced',
          0
        );
        setPendingCount(pending.length);
      } catch (error) {
        console.error('Failed to initialize IndexedDB:', error);
      }
    };

    initDB();
  }, [opts.dbName, opts.dbVersion]);

  // Listen for online/offline events
  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  // Sync pending operations when back online
  useEffect(() => {
    if (!isOnline || !db || isSyncing) return;

    const syncOperations = async () => {
      setIsSyncing(true);
      
      try {
        const operations = await db.getAllFromIndex(
          'pending-operations',
          'by-synced',
          0
        );

        for (const op of operations) {
          if (op.retryCount >= (opts.maxRetries || 3)) {
            // Mark as failed after max retries
            await db.put('pending-operations', {
              ...op,
              lastError: 'Max retries exceeded'
            });
            continue;
          }

          try {
            const response = await fetch(op.url, {
              method: op.method,
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(op.data)
            });

            if (response.ok) {
              await db.put('pending-operations', { ...op, synced: true });
            } else {
              throw new Error(`HTTP ${response.status}`);
            }
          } catch (error) {
            await db.put('pending-operations', {
              ...op,
              retryCount: op.retryCount + 1,
              lastError: error instanceof Error ? error.message : 'Unknown error'
            });
          }
        }

        // Update pending count
        const remaining = await db.getAllFromIndex(
          'pending-operations',
          'by-synced',
          0
        );
        setPendingCount(remaining.length);
      } finally {
        setIsSyncing(false);
      }
    };

    syncOperations();
  }, [isOnline, db, isSyncing, opts.maxRetries]);

  // Queue an operation for offline sync
  const queueOperation = useCallback(async (
    url: string,
    method: string,
    data: unknown
  ): Promise<number | undefined> => {
    if (!db) return undefined;

    const id = await db.add('pending-operations', {
      url,
      method,
      data,
      timestamp: Date.now(),
      synced: false,
      retryCount: 0
    });

    setPendingCount(prev => prev + 1);
    return id;
  }, [db]);

  // Cache a response for offline access
  const cacheResponse = useCallback(async (
    url: string,
    data: unknown,
    expiryMs?: number
  ): Promise<void> => {
    if (!db) return;

    const expiry = expiryMs || opts.cacheExpiry || 24 * 60 * 60 * 1000;
    
    await db.put('cached-responses', {
      url,
      data,
      timestamp: Date.now(),
      expiresAt: Date.now() + expiry
    });
  }, [db, opts.cacheExpiry]);

  // Get cached response
  const getCachedResponse = useCallback(async <T>(
    url: string
  ): Promise<T | null> => {
    if (!db) return null;

    const cached = await db.get('cached-responses', url);
    
    if (!cached) return null;
    
    // Check if expired
    if (cached.expiresAt < Date.now()) {
      await db.delete('cached-responses', url);
      return null;
    }

    return cached.data as T;
  }, [db]);

  // Clear all cached data
  const clearCache = useCallback(async (): Promise<void> => {
    if (!db) return;

    await db.clear('cached-responses');
  }, [db]);

  // Clear synced operations
  const clearSyncedOperations = useCallback(async (): Promise<void> => {
    if (!db) return;

    const synced = await db.getAllFromIndex(
      'pending-operations',
      'by-synced',
      1
    );

    for (const op of synced) {
      if (op.id) {
        await db.delete('pending-operations', op.id);
      }
    }
  }, [db]);

  // Get all pending operations
  const getPendingOperations = useCallback(async (): Promise<PendingOperation[]> => {
    if (!db) return [];

    return db.getAllFromIndex('pending-operations', 'by-synced', 0);
  }, [db]);

  // Force sync now
  const forceSync = useCallback(async (): Promise<void> => {
    if (!isOnline || !db) return;

    setIsSyncing(true);
    
    try {
      const operations = await db.getAllFromIndex(
        'pending-operations',
        'by-synced',
        0
      );

      for (const op of operations) {
        try {
          const response = await fetch(op.url, {
            method: op.method,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(op.data)
          });

          if (response.ok) {
            await db.put('pending-operations', { ...op, synced: true });
          }
        } catch (error) {
          console.error('Sync failed for operation:', op.id, error);
        }
      }

      const remaining = await db.getAllFromIndex(
        'pending-operations',
        'by-synced',
        0
      );
      setPendingCount(remaining.length);
    } finally {
      setIsSyncing(false);
    }
  }, [isOnline, db]);

  return {
    isOnline,
    isSyncing,
    pendingCount,
    queueOperation,
    cacheResponse,
    getCachedResponse,
    clearCache,
    clearSyncedOperations,
    getPendingOperations,
    forceSync
  };
}

export default useOfflineSync;
