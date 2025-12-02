/**
 * WebSocket service for real-time communication
 */

import { useAuthStore } from '../store/authStore';

type MessageHandler = (data: unknown) => void;
type ConnectionHandler = () => void;
type ErrorHandler = (error: Error) => void;

interface WebSocketConfig {
  url: string;
  reconnect?: boolean;
  reconnectAttempts?: number;
  reconnectDelay?: number;
  heartbeatInterval?: number;
}

const defaultConfig: Partial<WebSocketConfig> = {
  reconnect: true,
  reconnectAttempts: 5,
  reconnectDelay: 3000,
  heartbeatInterval: 30000
};

class WebSocketService {
  private ws: WebSocket | null = null;
  private config: WebSocketConfig;
  private reconnectCount = 0;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private messageHandlers: Map<string, Set<MessageHandler>> = new Map();
  private connectionHandlers: Set<ConnectionHandler> = new Set();
  private disconnectionHandlers: Set<ConnectionHandler> = new Set();
  private errorHandlers: Set<ErrorHandler> = new Set();
  private messageQueue: unknown[] = [];

  constructor(config: WebSocketConfig) {
    this.config = { ...defaultConfig, ...config };
  }

  /**
   * Connect to WebSocket server
   */
  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return;
    }

    const { token } = useAuthStore.getState();
    const url = token
      ? `${this.config.url}${this.config.url.includes('?') ? '&' : '?'}token=${token}`
      : this.config.url;

    try {
      this.ws = new WebSocket(url);
      this.setupEventHandlers();
    } catch (error) {
      this.handleError(error instanceof Error ? error : new Error('Connection failed'));
    }
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect(): void {
    this.stopHeartbeat();
    this.clearReconnectTimeout();

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  /**
   * Send a message
   */
  send(type: string, data: unknown): void {
    const message = JSON.stringify({ type, data, timestamp: Date.now() });

    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(message);
    } else {
      // Queue message for when connection is established
      this.messageQueue.push({ type, data });
    }
  }

  /**
   * Subscribe to a message type
   */
  on(type: string, handler: MessageHandler): () => void {
    if (!this.messageHandlers.has(type)) {
      this.messageHandlers.set(type, new Set());
    }
    this.messageHandlers.get(type)!.add(handler);

    // Return unsubscribe function
    return () => {
      this.messageHandlers.get(type)?.delete(handler);
    };
  }

  /**
   * Subscribe to connection events
   */
  onConnect(handler: ConnectionHandler): () => void {
    this.connectionHandlers.add(handler);
    return () => this.connectionHandlers.delete(handler);
  }

  /**
   * Subscribe to disconnection events
   */
  onDisconnect(handler: ConnectionHandler): () => void {
    this.disconnectionHandlers.add(handler);
    return () => this.disconnectionHandlers.delete(handler);
  }

  /**
   * Subscribe to error events
   */
  onError(handler: ErrorHandler): () => void {
    this.errorHandlers.add(handler);
    return () => this.errorHandlers.delete(handler);
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  private setupEventHandlers(): void {
    if (!this.ws) return;

    this.ws.onopen = () => {
      this.reconnectCount = 0;
      this.startHeartbeat();
      this.flushMessageQueue();
      this.connectionHandlers.forEach((handler) => handler());
    };

    this.ws.onclose = (event) => {
      this.stopHeartbeat();
      this.disconnectionHandlers.forEach((handler) => handler());

      // Attempt reconnection
      if (
        this.config.reconnect &&
        !event.wasClean &&
        this.reconnectCount < (this.config.reconnectAttempts || 5)
      ) {
        this.scheduleReconnect();
      }
    };

    this.ws.onerror = () => {
      this.handleError(new Error('WebSocket error'));
    };

    this.ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        const { type, data } = message;

        // Handle heartbeat response
        if (type === 'pong') {
          return;
        }

        // Dispatch to handlers
        const handlers = this.messageHandlers.get(type);
        if (handlers) {
          handlers.forEach((handler) => handler(data));
        }

        // Also dispatch to wildcard handlers
        const wildcardHandlers = this.messageHandlers.get('*');
        if (wildcardHandlers) {
          wildcardHandlers.forEach((handler) => handler(message));
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };
  }

  private startHeartbeat(): void {
    if (!this.config.heartbeatInterval) return;

    this.heartbeatInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, this.config.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  private scheduleReconnect(): void {
    this.reconnectCount++;
    const delay = (this.config.reconnectDelay || 3000) * this.reconnectCount;

    this.reconnectTimeout = setTimeout(() => {
      this.connect();
    }, delay);
  }

  private clearReconnectTimeout(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
  }

  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift() as { type: string; data: unknown };
      this.send(message.type, message.data);
    }
  }

  private handleError(error: Error): void {
    this.errorHandlers.forEach((handler) => handler(error));
  }
}

// Create singleton instances for different WebSocket connections
const WS_BASE_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

export const mainWebSocket = new WebSocketService({
  url: `${WS_BASE_URL}/ws/main`
});

export const collaborationWebSocket = new WebSocketService({
  url: `${WS_BASE_URL}/ws/collaborate`
});

export const notificationWebSocket = new WebSocketService({
  url: `${WS_BASE_URL}/ws/notifications`
});

// Factory function for creating document-specific WebSocket connections
export function createDocumentWebSocket(documentId: string): WebSocketService {
  return new WebSocketService({
    url: `${WS_BASE_URL}/ws/document/${documentId}`
  });
}

export default WebSocketService;
