/**
 * Frontend Security Service
 *
 * Comprehensive security utilities for:
 * - API authentication
 * - Request signing
 * - XSS prevention
 * - CSRF protection
 * - Input validation
 * - Security audit logging
 */

import { eventBus } from "./eventBus";

// Security event types
interface SecurityEvent {
  type:
    | "auth"
    | "access"
    | "validation"
    | "xss"
    | "csrf"
    | "rate_limit"
    | "suspicious";
  severity: "low" | "medium" | "high" | "critical";
  message: string;
  details?: Record<string, unknown>;
  timestamp: number;
  userId?: string;
  sessionId?: string;
  ip?: string;
  userAgent?: string;
  page?: string;
}

// Rate limiting configuration
interface RateLimitConfig {
  maxRequests: number;
  windowMs: number;
  blockDurationMs: number;
}

// Input validation rules
interface ValidationRule {
  type: "string" | "number" | "email" | "url" | "pattern";
  required?: boolean;
  minLength?: number;
  maxLength?: number;
  min?: number;
  max?: number;
  pattern?: RegExp;
  sanitize?: boolean;
  message?: string;
}

class SecurityService {
  private securityEvents: SecurityEvent[] = [];
  private readonly rateLimitMap: Map<
    string,
    { count: number; resetTime: number; blocked: boolean }
  > = new Map();
  private csrfToken: string | null = null;
  private readonly maxEvents = 1000;

  constructor() {
    this.initCSRFToken();
    this.setupSecurityListeners();
  }

  // ==================== Authentication ====================

  /**
   * Validate JWT token structure (not signature - that's backend)
   */
  validateTokenStructure(token: string): boolean {
    try {
      const parts = token.split(".");
      if (parts.length !== 3) {
        this.logSecurityEvent({
          type: "auth",
          severity: "high",
          message: "Invalid token structure",
          details: { partsCount: parts.length },
        });
        return false;
      }

      // Decode and check payload
      const payload = JSON.parse(atob(parts[1]));

      // Check expiration
      if (payload.exp && payload.exp * 1000 < Date.now()) {
        this.logSecurityEvent({
          type: "auth",
          severity: "medium",
          message: "Token expired",
          details: { expiry: new Date(payload.exp * 1000).toISOString() },
        });
        return false;
      }

      return true;
    } catch (error) {
      this.logSecurityEvent({
        type: "auth",
        severity: "high",
        message: "Token parsing failed",
        details: { error: String(error) },
      });
      return false;
    }
  }

  /**
   * Extract token payload
   */
  getTokenPayload(token: string): Record<string, unknown> | null {
    try {
      const parts = token.split(".");
      if (parts.length !== 3) return null;
      return JSON.parse(atob(parts[1]));
    } catch {
      return null;
    }
  }

  /**
   * Check if token is about to expire
   */
  isTokenExpiringSoon(
    token: string,
    thresholdMs: number = 5 * 60 * 1000
  ): boolean {
    const payload = this.getTokenPayload(token);
    if (!payload || !payload.exp) return true;
    return (payload.exp as number) * 1000 - Date.now() < thresholdMs;
  }

  // ==================== CSRF Protection ====================

  /**
   * Initialize CSRF token
   */
  private initCSRFToken(): void {
    // Check for existing token in meta tag
    const metaToken = document.querySelector('meta[name="csrf-token"]');
    if (metaToken) {
      this.csrfToken = metaToken.getAttribute("content");
      return;
    }

    // Generate client-side token if not provided
    this.csrfToken = this.generateSecureToken();
    sessionStorage.setItem("csrf-token", this.csrfToken);
  }

  /**
   * Get CSRF token for requests
   */
  getCSRFToken(): string {
    if (!this.csrfToken) {
      this.csrfToken =
        sessionStorage.getItem("csrf-token") || this.generateSecureToken();
    }
    return this.csrfToken;
  }

  /**
   * Validate CSRF token
   */
  validateCSRFToken(token: string): boolean {
    const valid = token === this.csrfToken;
    if (!valid) {
      this.logSecurityEvent({
        type: "csrf",
        severity: "critical",
        message: "CSRF token mismatch",
      });
    }
    return valid;
  }

  /**
   * Generate secure random token
   */
  private generateSecureToken(length: number = 32): string {
    const array = new Uint8Array(length);
    crypto.getRandomValues(array);
    return Array.from(array, (byte) => byte.toString(16).padStart(2, "0")).join(
      ""
    );
  }

  // ==================== XSS Prevention ====================

  /**
   * Sanitize HTML content
   */
  sanitizeHTML(html: string): string {
    const dangerous = /<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi;
    const events = /\s*on\w+\s*=/gi;
    const javascript = /javascript:/gi;
    const dataUrls = /data:/gi;

    let sanitized = html
      .replaceAll(dangerous, "")
      .replaceAll(events, "")
      .replaceAll(javascript, "")
      .replaceAll(dataUrls, "");

    // Check if content was modified
    if (sanitized !== html) {
      this.logSecurityEvent({
        type: "xss",
        severity: "high",
        message: "XSS attempt detected and sanitized",
        details: { original: html.substring(0, 100) },
      });
    }

    return sanitized;
  }

  /**
   * Escape HTML entities
   */
  escapeHTML(text: string): string {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  /**
   * Validate and sanitize URL
   */
  sanitizeURL(url: string): string | null {
    try {
      const parsed = new URL(url);

      // Only allow http and https
      if (!["http:", "https:"].includes(parsed.protocol)) {
        this.logSecurityEvent({
          type: "xss",
          severity: "medium",
          message: "Blocked unsafe URL protocol",
          details: { url, protocol: parsed.protocol },
        });
        return null;
      }

      return parsed.href;
    } catch {
      return null;
    }
  }

  // ==================== Input Validation ====================

  /**
   * Type-specific validators
   */
  private validateString(
    value: unknown,
    rules: ValidationRule
  ): { valid: boolean; error?: string; sanitized?: unknown } | null {
    if (typeof value !== "string") {
      return { valid: false, error: "Must be a string" };
    }

    const sanitized = rules.sanitize ? this.sanitizeHTML(value) : value;

    if (rules.minLength && value.length < rules.minLength) {
      return {
        valid: false,
        error: `Minimum ${rules.minLength} characters required`,
      };
    }

    if (rules.maxLength && value.length > rules.maxLength) {
      return {
        valid: false,
        error: `Maximum ${rules.maxLength} characters allowed`,
      };
    }

    if (rules.pattern && !rules.pattern.test(value)) {
      return { valid: false, error: rules.message || "Invalid format" };
    }

    return { valid: true, sanitized };
  }

  private validateNumber(
    value: unknown,
    rules: ValidationRule
  ): { valid: boolean; error?: string; sanitized?: unknown } {
    const num = typeof value === "string" ? Number.parseFloat(value) : value;
    if (typeof num !== "number" || Number.isNaN(num)) {
      return { valid: false, error: "Must be a number" };
    }

    if (rules.min !== undefined && num < rules.min) {
      return { valid: false, error: `Minimum value is ${rules.min}` };
    }

    if (rules.max !== undefined && num > rules.max) {
      return { valid: false, error: `Maximum value is ${rules.max}` };
    }

    return { valid: true, sanitized: num };
  }

  private validateEmail(value: unknown): {
    valid: boolean;
    error?: string;
    sanitized?: unknown;
  } {
    if (typeof value !== "string") {
      return { valid: false, error: "Must be a string" };
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(value)) {
      return { valid: false, error: "Invalid email format" };
    }

    return { valid: true, sanitized: value };
  }

  private validateUrl(value: unknown): {
    valid: boolean;
    error?: string;
    sanitized?: unknown;
  } {
    if (typeof value !== "string") {
      return { valid: false, error: "Must be a string" };
    }

    const sanitizedUrl = this.sanitizeURL(value);
    if (!sanitizedUrl) {
      return { valid: false, error: "Invalid URL" };
    }

    return { valid: true, sanitized: sanitizedUrl };
  }

  private validatePattern(
    value: unknown,
    rules: ValidationRule
  ): { valid: boolean; error?: string; sanitized?: unknown } {
    if (typeof value !== "string") {
      return { valid: false, error: "Must be a string" };
    }

    if (rules.pattern && !rules.pattern.test(value)) {
      return { valid: false, error: rules.message || "Invalid format" };
    }

    return { valid: true, sanitized: value };
  }

  /**
   * Validate input against rules
   */
  validate(
    value: unknown,
    rules: ValidationRule
  ): { valid: boolean; error?: string; sanitized?: unknown } {
    // Required check
    if (
      rules.required &&
      (value === null || value === undefined || value === "")
    ) {
      return { valid: false, error: rules.message || "This field is required" };
    }

    // Empty value is valid if not required
    if (value === null || value === undefined || value === "") {
      return { valid: true, sanitized: value };
    }

    // Type-specific validation using strategy pattern
    const validators: Record<
      string,
      () => { valid: boolean; error?: string; sanitized?: unknown } | null
    > = {
      string: () => this.validateString(value, rules),
      number: () => this.validateNumber(value, rules),
      email: () => this.validateEmail(value),
      url: () => this.validateUrl(value),
      pattern: () => this.validatePattern(value, rules),
    };

    const validator = validators[rules.type];
    if (validator) {
      const result = validator();
      if (result) return result;
    }

    return { valid: true, sanitized: value };
  }

  /**
   * Validate multiple fields
   */
  validateMany(
    data: Record<string, unknown>,
    rules: Record<string, ValidationRule>
  ): {
    valid: boolean;
    errors: Record<string, string>;
    sanitized: Record<string, unknown>;
  } {
    const errors: Record<string, string> = {};
    const sanitized: Record<string, unknown> = {};
    let valid = true;

    for (const [field, rule] of Object.entries(rules)) {
      const result = this.validate(data[field], rule);
      if (!result.valid) {
        errors[field] = result.error || "Invalid value";
        valid = false;
      }
      sanitized[field] = result.sanitized ?? data[field];
    }

    return { valid, errors, sanitized };
  }

  // ==================== Rate Limiting ====================

  /**
   * Check rate limit
   */
  checkRateLimit(
    key: string,
    config: RateLimitConfig = {
      maxRequests: 100,
      windowMs: 60000,
      blockDurationMs: 300000,
    }
  ): { allowed: boolean; remaining: number; retryAfter?: number } {
    const now = Date.now();
    const existing = this.rateLimitMap.get(key);

    // Check if blocked
    if (existing?.blocked && existing.resetTime > now) {
      this.logSecurityEvent({
        type: "rate_limit",
        severity: "high",
        message: "Rate limit exceeded - request blocked",
        details: { key, retryAfter: existing.resetTime - now },
      });
      return {
        allowed: false,
        remaining: 0,
        retryAfter: existing.resetTime - now,
      };
    }

    // Reset if window expired
    if (!existing || existing.resetTime < now) {
      this.rateLimitMap.set(key, {
        count: 1,
        resetTime: now + config.windowMs,
        blocked: false,
      });
      return { allowed: true, remaining: config.maxRequests - 1 };
    }

    // Check if limit exceeded
    if (existing.count >= config.maxRequests) {
      existing.blocked = true;
      existing.resetTime = now + config.blockDurationMs;

      this.logSecurityEvent({
        type: "rate_limit",
        severity: "medium",
        message: "Rate limit exceeded",
        details: { key, count: existing.count, limit: config.maxRequests },
      });

      return {
        allowed: false,
        remaining: 0,
        retryAfter: config.blockDurationMs,
      };
    }

    // Increment counter
    existing.count++;
    return { allowed: true, remaining: config.maxRequests - existing.count };
  }

  // ==================== Security Audit Logging ====================

  /**
   * Log security event
   */
  logSecurityEvent(
    event: Omit<SecurityEvent, "timestamp" | "userAgent" | "page">
  ): void {
    const fullEvent: SecurityEvent = {
      ...event,
      timestamp: Date.now(),
      userAgent: navigator.userAgent,
      page: globalThis.location.pathname,
    };

    this.securityEvents.push(fullEvent);

    // Trim old events
    if (this.securityEvents.length > this.maxEvents) {
      this.securityEvents = this.securityEvents.slice(-this.maxEvents);
    }

    // Emit event for monitoring
    eventBus.emit("security:event", fullEvent);

    // Console warning for high severity
    if (event.severity === "high" || event.severity === "critical") {
      console.warn(
        `[Security] ${event.severity.toUpperCase()}: ${event.message}`,
        event.details
      );
    }

    // Store critical events persistently
    if (event.severity === "critical") {
      this.persistEvent(fullEvent);
    }
  }

  /**
   * Get security events
   */
  getSecurityEvents(filter?: {
    type?: SecurityEvent["type"];
    severity?: SecurityEvent["severity"];
    since?: number;
  }): SecurityEvent[] {
    let events = [...this.securityEvents];

    if (filter?.type) {
      events = events.filter((e) => e.type === filter.type);
    }

    if (filter?.severity) {
      events = events.filter((e) => e.severity === filter.severity);
    }

    if (filter?.since !== undefined) {
      const sinceTime = filter.since;
      events = events.filter((e) => e.timestamp >= sinceTime);
    }

    return events;
  }

  /**
   * Get security summary
   */
  getSecuritySummary(): {
    total: number;
    bySeverity: Record<SecurityEvent["severity"], number>;
    byType: Record<SecurityEvent["type"], number>;
    critical24h: number;
  } {
    const now = Date.now();
    const day = 24 * 60 * 60 * 1000;

    const bySeverity: Record<SecurityEvent["severity"], number> = {
      low: 0,
      medium: 0,
      high: 0,
      critical: 0,
    };

    const byType: Record<SecurityEvent["type"], number> = {
      auth: 0,
      access: 0,
      validation: 0,
      xss: 0,
      csrf: 0,
      rate_limit: 0,
      suspicious: 0,
    };

    let critical24h = 0;

    for (const event of this.securityEvents) {
      bySeverity[event.severity]++;
      byType[event.type]++;
      if (event.severity === "critical" && event.timestamp > now - day) {
        critical24h++;
      }
    }

    return {
      total: this.securityEvents.length,
      bySeverity,
      byType,
      critical24h,
    };
  }

  /**
   * Export security events
   */
  exportSecurityEvents(): string {
    return JSON.stringify(this.securityEvents, null, 2);
  }

  /**
   * Clear security events
   */
  clearSecurityEvents(): void {
    this.securityEvents = [];
  }

  // ==================== Three-Version Access Control ====================

  /**
   * Check if a user role has access to a specific version and resource type
   *
   * Access Rules:
   * - Users can only access V2 CR-AI (Code Review AI)
   * - Admins can access all versions and both AI types
   * - VC-AI (Version Control AI) is admin-only on all versions
   */
  checkVersionAccess(
    role: string,
    version: string,
    resourceType: string
  ): boolean {
    const isAdmin = role === "admin" || role === "system";
    const isVCAI = resourceType === "vc_ai";
    const isV2 = version === "v2";
    const isCRAI = resourceType === "cr_ai";

    // VC-AI is always admin-only
    if (isVCAI && !isAdmin) {
      this.logSecurityEvent({
        type: "access",
        severity: "medium",
        message: `Unauthorized VC-AI access attempt`,
        details: { role, version, resourceType },
      });
      return false;
    }

    // Admins have full access
    if (isAdmin) {
      return true;
    }

    // Regular users can only access V2 CR-AI
    if (isCRAI && isV2) {
      return true;
    }

    // Deny all other access for non-admins
    this.logSecurityEvent({
      type: "access",
      severity: "medium",
      message: `Unauthorized version access attempt`,
      details: { role, version, resourceType },
    });
    return false;
  }

  /**
   * Get user's accessible versions based on role
   */
  getAccessibleVersions(role: string): string[] {
    if (role === "admin" || role === "system") {
      return ["v1", "v2", "v3"];
    }
    return ["v2"]; // Users only get V2
  }

  /**
   * Get user's accessible AI types based on role
   */
  getAccessibleAITypes(role: string): string[] {
    if (role === "admin" || role === "system") {
      return ["cr_ai", "vc_ai"];
    }
    return ["cr_ai"]; // Users only get CR-AI
  }

  // ==================== Private Helpers ====================

  private persistEvent(event: SecurityEvent): void {
    try {
      const existing = JSON.parse(
        localStorage.getItem("security_events") || "[]"
      );
      existing.push(event);
      // Keep only last 100 critical events
      if (existing.length > 100) {
        existing.shift();
      }
      localStorage.setItem("security_events", JSON.stringify(existing));
    } catch {
      // Storage quota exceeded or not available
    }
  }

  private setupSecurityListeners(): void {
    // Listen for suspicious activity
    window.addEventListener("error", (event) => {
      if (event.message.includes("Script error")) {
        this.logSecurityEvent({
          type: "suspicious",
          severity: "low",
          message: "Cross-origin script error detected",
          details: { error: event.message },
        });
      }
    });

    // Detect DevTools open (basic detection)
    let devtoolsOpen = false;
    const threshold = 160;
    const emitDevtoolsWarning = () => {
      const widthThreshold = window.outerWidth - window.innerWidth > threshold;
      const heightThreshold =
        window.outerHeight - window.innerHeight > threshold;

      if ((widthThreshold || heightThreshold) && !devtoolsOpen) {
        devtoolsOpen = true;
        this.logSecurityEvent({
          type: "suspicious",
          severity: "low",
          message: "DevTools may be open",
        });
      } else if (!widthThreshold && !heightThreshold) {
        devtoolsOpen = false;
      }
    };

    window.addEventListener("resize", emitDevtoolsWarning);
  }
}

// Singleton instance
export const securityService = new SecurityService();

export default securityService;
