import { useEffect, useState, useRef, useCallback } from 'react';

interface SSEState<T> {
  data: T | null;
  error: Error | null;
  isConnected: boolean;
  reconnect: () => void;
}

interface SSEOptions {
  withCredentials?: boolean;
  retryOnError?: boolean;
  maxRetries?: number;
  retryDelay?: number;
}

const defaultOptions: SSEOptions = {
  withCredentials: true,
  retryOnError: true,
  maxRetries: 3,
  retryDelay: 1000
};

export function useSSE<T>(
  url: string,
  options: SSEOptions = {}
): SSEState<T> {
  const opts = { ...defaultOptions, ...options };
  
  const [state, setState] = useState<Omit<SSEState<T>, 'reconnect'>>({
    data: null,
    error: null,
    isConnected: false
  });

  const eventSourceRef = useRef<EventSource | null>(null);
  const retryCountRef = useRef(0);
  const retryTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const connect = useCallback(() => {
    // Clean up existing connection
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    // Clear any pending retry
    if (retryTimeoutRef.current) {
      clearTimeout(retryTimeoutRef.current);
      retryTimeoutRef.current = null;
    }

    try {
      const eventSource = new EventSource(url, {
        withCredentials: opts.withCredentials
      });
      eventSourceRef.current = eventSource;

      eventSource.onopen = () => {
        retryCountRef.current = 0;
        setState(prev => ({ 
          ...prev, 
          isConnected: true, 
          error: null 
        }));
      };

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as T;
          setState(prev => ({ ...prev, data }));
        } catch (parseError) {
          setState(prev => ({
            ...prev,
            error: new Error('Failed to parse SSE data')
          }));
        }
      };

      eventSource.onerror = (event) => {
        eventSource.close();
        setState(prev => ({
          ...prev,
          isConnected: false
        }));

        // Retry logic
        if (opts.retryOnError && retryCountRef.current < (opts.maxRetries || 3)) {
          retryCountRef.current += 1;
          const delay = (opts.retryDelay || 1000) * Math.pow(2, retryCountRef.current - 1);
          
          retryTimeoutRef.current = setTimeout(() => {
            connect();
          }, delay);
        } else {
          setState(prev => ({
            ...prev,
            error: new Error('SSE connection error')
          }));
        }
      };

      // Handle custom events
      eventSource.addEventListener('error', (event: MessageEvent) => {
        try {
          const errorData = JSON.parse(event.data);
          setState(prev => ({
            ...prev,
            error: new Error(errorData.message || 'Unknown error')
          }));
        } catch {
          // Ignore parse errors for error events
        }
      });

      eventSource.addEventListener('complete', (event: MessageEvent) => {
        try {
          const data = JSON.parse(event.data) as T;
          setState(prev => ({ ...prev, data }));
          eventSource.close();
        } catch {
          // Ignore parse errors
        }
      });

    } catch (err) {
      setState(prev => ({
        ...prev,
        isConnected: false,
        error: err instanceof Error ? err : new Error('Failed to create EventSource')
      }));
    }
  }, [url, opts.withCredentials, opts.retryOnError, opts.maxRetries, opts.retryDelay]);

  const reconnect = useCallback(() => {
    retryCountRef.current = 0;
    connect();
  }, [connect]);

  useEffect(() => {
    connect();

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
      }
    };
  }, [connect]);

  return { ...state, reconnect };
}

export default useSSE;
