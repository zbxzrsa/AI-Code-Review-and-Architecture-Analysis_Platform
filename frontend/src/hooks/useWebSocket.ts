import { useEffect, useState, useRef, useCallback } from "react";
import { useAuthStore } from "../store/authStore";

interface WebSocketState<T> {
  data: T | null;
  error: Error | null;
  isConnected: boolean;
  send: (message: unknown) => void;
  reconnect: () => void;
}

interface WebSocketOptions {
  reconnect?: boolean;
  reconnectAttempts?: number;
  reconnectInterval?: number;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (error: Error) => void;
  onMessage?: (data: unknown) => void;
}

const defaultOptions: WebSocketOptions = {
  reconnect: true,
  reconnectAttempts: 5,
  reconnectInterval: 3000,
};

export function useWebSocket<T>(
  url: string,
  options: WebSocketOptions = {}
): WebSocketState<T> {
  const opts = { ...defaultOptions, ...options };
  const { token } = useAuthStore();

  const [state, setState] = useState<
    Omit<WebSocketState<T>, "send" | "reconnect">
  >({
    data: null,
    error: null,
    isConnected: false,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectCountRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const messageQueueRef = useRef<unknown[]>([]);

  const connect = useCallback(() => {
    // Clean up existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    // Clear any pending reconnect
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    try {
      // Add token to URL if available
      const wsUrl = token
        ? `${url}${url.includes("?") ? "&" : "?"}token=${token}`
        : url;

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        reconnectCountRef.current = 0;
        setState((prev) => ({
          ...prev,
          isConnected: true,
          error: null,
        }));
        opts.onOpen?.();

        // Send queued messages
        while (messageQueueRef.current.length > 0) {
          const message = messageQueueRef.current.shift();
          ws.send(JSON.stringify(message));
        }
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as T;
          setState((prev) => ({ ...prev, data }));
          opts.onMessage?.(data);
        } catch (parseError) {
          setState((prev) => ({
            ...prev,
            error: new Error("Failed to parse WebSocket message"),
          }));
        }
      };

      ws.onerror = (_event) => {
        const error = new Error("WebSocket error");
        setState((prev) => ({ ...prev, error }));
        opts.onError?.(error);
      };

      ws.onclose = (event) => {
        setState((prev) => ({ ...prev, isConnected: false }));
        opts.onClose?.();

        // Reconnect logic
        if (
          opts.reconnect &&
          reconnectCountRef.current < (opts.reconnectAttempts || 5) &&
          !event.wasClean
        ) {
          reconnectCountRef.current += 1;
          const delay =
            (opts.reconnectInterval || 3000) * reconnectCountRef.current;

          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, delay);
        }
      };
    } catch (err) {
      setState((prev) => ({
        ...prev,
        isConnected: false,
        error:
          err instanceof Error ? err : new Error("Failed to create WebSocket"),
      }));
    }
  }, [url, token, opts]);

  const send = useCallback((message: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      // Queue message for when connection is established
      messageQueueRef.current.push(message);
    }
  }, []);

  const reconnect = useCallback(() => {
    reconnectCountRef.current = 0;
    connect();
  }, [connect]);

  useEffect(() => {
    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [connect]);

  return { ...state, send, reconnect };
}

// Specialized hook for collaborative editing
export function useCollaborativeEditing(documentId: string) {
  const baseUrl = import.meta.env.VITE_WS_URL || "ws://localhost:8000";
  const url = `${baseUrl}/ws/collaborate/${documentId}`;

  interface CollaborativeMessage {
    type: "cursor" | "edit" | "selection" | "presence";
    userId: string;
    userName: string;
    data: unknown;
    timestamp: number;
  }

  const { data, error, isConnected, send, reconnect } =
    useWebSocket<CollaborativeMessage>(url);

  const sendCursorPosition = useCallback(
    (line: number, column: number) => {
      send({
        type: "cursor",
        data: { line, column },
      });
    },
    [send]
  );

  const sendEdit = useCallback(
    (changes: unknown) => {
      send({
        type: "edit",
        data: changes,
      });
    },
    [send]
  );

  const sendSelection = useCallback(
    (
      startLine: number,
      startColumn: number,
      endLine: number,
      endColumn: number
    ) => {
      send({
        type: "selection",
        data: { startLine, startColumn, endLine, endColumn },
      });
    },
    [send]
  );

  return {
    message: data,
    error,
    isConnected,
    reconnect,
    sendCursorPosition,
    sendEdit,
    sendSelection,
  };
}

export default useWebSocket;
