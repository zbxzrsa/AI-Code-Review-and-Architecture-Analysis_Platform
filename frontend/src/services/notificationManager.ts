/**
 * Enhanced Notification Manager
 * 
 * Centralized notification system with:
 * - Multiple notification types
 * - Queue management
 * - Persistence
 * - Sound/vibration support
 * - Action buttons
 * - Grouping
 */

import { message, notification } from 'antd';

// ============================================
// Types
// ============================================

export type NotificationType = 'success' | 'error' | 'warning' | 'info';

export interface NotificationAction {
  label: string;
  onClick: () => void;
  type?: 'primary' | 'default' | 'link';
}

export interface NotificationOptions {
  id?: string;
  type: NotificationType;
  title: string;
  message?: string;
  duration?: number;
  persistent?: boolean;
  actions?: NotificationAction[];
  group?: string;
  sound?: boolean;
  vibrate?: boolean;
  onClick?: () => void;
  onClose?: () => void;
  icon?: React.ReactNode;
}

export interface StoredNotification {
  id: string;
  type: NotificationType;
  title: string;
  message?: string;
  timestamp: Date;
  read: boolean;
  group?: string;
}

// ============================================
// Notification Manager Class
// ============================================

class NotificationManager {
  private history: StoredNotification[] = [];
  private maxHistory: number = 100;
  private soundEnabled: boolean = true;
  private listeners: Set<(notifications: StoredNotification[]) => void> = new Set();
  private groupCounts: Map<string, number> = new Map();

  constructor() {
    this.loadFromStorage();
  }

  // ============================================
  // Core Methods
  // ============================================

  public show(options: NotificationOptions): string {
    const id = options.id || this.generateId();
    
    // Handle grouped notifications
    if (options.group) {
      const count = (this.groupCounts.get(options.group) || 0) + 1;
      this.groupCounts.set(options.group, count);
      
      if (count > 1) {
        options.title = `${options.title} (${count})`;
      }
    }

    // Show the notification
    if (options.persistent) {
      this.showPersistent(id, options);
    } else {
      this.showToast(options);
    }

    // Play sound if enabled
    if (options.sound && this.soundEnabled) {
      this.playSound(options.type);
    }

    // Vibrate on mobile
    if (options.vibrate && 'vibrate' in navigator) {
      navigator.vibrate(options.type === 'error' ? [100, 50, 100] : [50]);
    }

    // Store in history
    this.addToHistory({
      id,
      type: options.type,
      title: options.title,
      message: options.message,
      timestamp: new Date(),
      read: false,
      group: options.group,
    });

    return id;
  }

  private showToast(options: NotificationOptions): void {
    const config = {
      content: options.message || options.title,
      duration: options.duration ?? 3,
      onClick: options.onClick,
    };

    switch (options.type) {
      case 'success':
        message.success(config);
        break;
      case 'error':
        message.error(config);
        break;
      case 'warning':
        message.warning(config);
        break;
      case 'info':
      default:
        message.info(config);
        break;
    }
  }

  private showPersistent(id: string, options: NotificationOptions): void {
    notification[options.type]({
      key: id,
      message: options.title,
      description: options.message,
      duration: options.duration ?? 0,
      onClick: options.onClick,
      onClose: options.onClose,
      icon: options.icon,
    });
  }

  // ============================================
  // Convenience Methods
  // ============================================

  public success(title: string, message?: string, options?: Partial<NotificationOptions>): string {
    return this.show({ type: 'success', title, message, ...options });
  }

  public error(title: string, message?: string, options?: Partial<NotificationOptions>): string {
    return this.show({ type: 'error', title, message, duration: 5, ...options });
  }

  public warning(title: string, message?: string, options?: Partial<NotificationOptions>): string {
    return this.show({ type: 'warning', title, message, ...options });
  }

  public info(title: string, message?: string, options?: Partial<NotificationOptions>): string {
    return this.show({ type: 'info', title, message, ...options });
  }

  public confirm(
    title: string,
    message: string,
    onConfirm: () => void,
    onCancel?: () => void
  ): string {
    return this.show({
      type: 'warning',
      title,
      message,
      persistent: true,
      actions: [
        { label: 'Cancel', onClick: onCancel || (() => {}), type: 'default' },
        { label: 'Confirm', onClick: onConfirm, type: 'primary' },
      ],
    });
  }

  // ============================================
  // API Error Handler
  // ============================================

  public handleApiError(error: any): string {
    let title = 'Error';
    let msg = 'An unexpected error occurred';

    if (error?.code) {
      switch (error.code) {
        case 'NETWORK_ERROR':
          title = 'Connection Error';
          msg = 'Please check your internet connection';
          break;
        case 'TIMEOUT':
          title = 'Request Timeout';
          msg = 'The request took too long. Please try again.';
          break;
        case 'UNAUTHORIZED':
          title = 'Session Expired';
          msg = 'Please log in again';
          break;
        case 'FORBIDDEN':
          title = 'Access Denied';
          msg = 'You do not have permission for this action';
          break;
        case 'RATE_LIMITED':
          title = 'Too Many Requests';
          msg = 'Please wait before trying again';
          break;
        case 'SERVER_ERROR':
          title = 'Server Error';
          msg = 'Something went wrong on our end';
          break;
        default:
          msg = error.message || msg;
      }
    } else if (error?.message) {
      msg = error.message;
    }

    return this.error(title, msg, { 
      persistent: error?.code === 'NETWORK_ERROR',
      sound: true,
    });
  }

  // ============================================
  // History Management
  // ============================================

  private addToHistory(notification: StoredNotification): void {
    this.history.unshift(notification);
    
    // Trim history if too long
    if (this.history.length > this.maxHistory) {
      this.history = this.history.slice(0, this.maxHistory);
    }

    this.saveToStorage();
    this.notifyListeners();
  }

  public getHistory(): StoredNotification[] {
    return [...this.history];
  }

  public getUnreadCount(): number {
    return this.history.filter(n => !n.read).length;
  }

  public markAsRead(id: string): void {
    const notification = this.history.find(n => n.id === id);
    if (notification) {
      notification.read = true;
      this.saveToStorage();
      this.notifyListeners();
    }
  }

  public markAllAsRead(): void {
    this.history.forEach(n => n.read = true);
    this.saveToStorage();
    this.notifyListeners();
  }

  public clearHistory(): void {
    this.history = [];
    this.saveToStorage();
    this.notifyListeners();
  }

  public dismiss(id: string): void {
    notification.destroy(id);
  }

  public dismissAll(): void {
    notification.destroy();
    message.destroy();
  }

  // ============================================
  // Listeners
  // ============================================

  public subscribe(callback: (notifications: StoredNotification[]) => void): () => void {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }

  private notifyListeners(): void {
    this.listeners.forEach(callback => callback(this.getHistory()));
  }

  // ============================================
  // Settings
  // ============================================

  public setSoundEnabled(enabled: boolean): void {
    this.soundEnabled = enabled;
    localStorage.setItem('notification_sound', String(enabled));
  }

  public isSoundEnabled(): boolean {
    return this.soundEnabled;
  }

  // ============================================
  // Sound
  // ============================================

  private playSound(type: NotificationType): void {
    // Simple beep using Web Audio API
    try {
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();

      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);

      // Different frequencies for different types
      const frequencies: Record<NotificationType, number> = {
        success: 800,
        error: 400,
        warning: 600,
        info: 700,
      };

      oscillator.frequency.value = frequencies[type];
      oscillator.type = 'sine';
      
      gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.2);

      oscillator.start(audioContext.currentTime);
      oscillator.stop(audioContext.currentTime + 0.2);
    } catch (e) {
      // Audio not supported
    }
  }

  // ============================================
  // Persistence
  // ============================================

  private saveToStorage(): void {
    try {
      localStorage.setItem('notification_history', JSON.stringify(this.history.slice(0, 50)));
    } catch (e) {
      // Storage full or not available
    }
  }

  private loadFromStorage(): void {
    try {
      const stored = localStorage.getItem('notification_history');
      if (stored) {
        this.history = JSON.parse(stored).map((n: any) => ({
          ...n,
          timestamp: new Date(n.timestamp),
        }));
      }

      const soundSetting = localStorage.getItem('notification_sound');
      if (soundSetting !== null) {
        this.soundEnabled = soundSetting === 'true';
      }
    } catch (e) {
      // Invalid data
    }
  }

  // ============================================
  // Utility
  // ============================================

  private generateId(): string {
    return `notif-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  public resetGroupCount(group: string): void {
    this.groupCounts.delete(group);
  }
}

// ============================================
// Export Singleton
// ============================================

export const notificationManager = new NotificationManager();

export default notificationManager;
