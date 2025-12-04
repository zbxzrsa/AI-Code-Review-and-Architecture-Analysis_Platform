/**
 * Security Service Unit Tests
 */

import { describe, it, expect, beforeEach, vi } from "vitest";

// Mock eventBus
vi.mock("../eventBus", () => ({
  eventBus: {
    emit: vi.fn(),
    emitSync: vi.fn(),
  },
}));

// Mock DOM APIs
vi.stubGlobal("document", {
  querySelector: () => null,
  createElement: (tag: string) => {
    let _innerHTML = "";
    return {
      textContent: "",
      get innerHTML(): string {
        return _innerHTML;
      },
      set innerHTML(v: string) {
        _innerHTML = v;
      },
    };
  },
});

vi.stubGlobal("sessionStorage", {
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {},
});

vi.stubGlobal("localStorage", {
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {},
});

vi.stubGlobal("crypto", {
  getRandomValues: (arr: Uint8Array) => {
    for (let i = 0; i < arr.length; i++) {
      arr[i] = Math.floor(Math.random() * 256);
    }
    return arr;
  },
});

vi.stubGlobal("navigator", { userAgent: "test" });
vi.stubGlobal("window", { location: { pathname: "/test" } });

import { securityService } from "../securityService";

describe("SecurityService", () => {
  const security = securityService;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("Token Validation", () => {
    it("should validate proper JWT structure", () => {
      const header = btoa(JSON.stringify({ alg: "HS256", typ: "JWT" }));
      const payload = btoa(
        JSON.stringify({
          sub: "123",
          exp: Math.floor(Date.now() / 1000) + 3600,
        })
      );
      const signature = "mock-signature";
      const token = `${header}.${payload}.${signature}`;

      const result = security.validateTokenStructure(token);
      expect(result).toBe(true);
    });

    it("should reject invalid token format", () => {
      const result = security.validateTokenStructure("invalid-token");
      expect(result).toBe(false);
    });

    it("should reject expired token", () => {
      const header = btoa(JSON.stringify({ alg: "HS256", typ: "JWT" }));
      const payload = btoa(
        JSON.stringify({
          sub: "123",
          exp: Math.floor(Date.now() / 1000) - 3600, // Expired
        })
      );
      const signature = "mock-signature";
      const token = `${header}.${payload}.${signature}`;

      const result = security.validateTokenStructure(token);
      expect(result).toBe(false);
    });

    it("should get token payload", () => {
      const header = btoa(JSON.stringify({ alg: "HS256", typ: "JWT" }));
      const payload = btoa(JSON.stringify({ sub: "123", name: "Test" }));
      const signature = "mock-signature";
      const token = `${header}.${payload}.${signature}`;

      const result = security.getTokenPayload(token);
      expect(result).toEqual({ sub: "123", name: "Test" });
    });
  });

  describe("CSRF Protection", () => {
    it("should get CSRF token", () => {
      const token = security.getCSRFToken();
      expect(token).toBeDefined();
      expect(token.length).toBeGreaterThan(0);
    });

    it("should validate correct CSRF token", () => {
      const token = security.getCSRFToken();
      const isValid = security.validateCSRFToken(token);
      expect(isValid).toBe(true);
    });

    it("should reject invalid CSRF token", () => {
      security.getCSRFToken();
      const isValid = security.validateCSRFToken("invalid-token");
      expect(isValid).toBe(false);
    });
  });

  describe("XSS Prevention", () => {
    it("should sanitize script tags", () => {
      const dirty = '<script>alert("xss")</script>Hello';
      const clean = security.sanitizeHTML(dirty);
      expect(clean).not.toContain("<script>");
      expect(clean).toContain("Hello");
    });

    it("should sanitize event handlers", () => {
      const dirty = '<div onclick="alert(1)">Click</div>';
      const clean = security.sanitizeHTML(dirty);
      expect(clean).not.toContain("onclick");
    });

    it("should allow safe content", () => {
      const safe = "Hello, World!";
      const result = security.sanitizeHTML(safe);
      expect(result).toBe("Hello, World!");
    });

    it("should sanitize javascript URLs", () => {
      const dirty = "javascript:alert(1)";
      const clean = security.sanitizeHTML(dirty);
      expect(clean).not.toContain("javascript:");
    });
  });

  describe("URL Validation", () => {
    it("should accept safe URLs", () => {
      const result = security.sanitizeURL("https://example.com");
      expect(result).toBe("https://example.com/");
    });

    it("should accept localhost URLs", () => {
      const result = security.sanitizeURL("http://localhost:3000/path");
      expect(result).toBe("http://localhost:3000/path");
    });

    it("should reject javascript URLs", () => {
      const result = security.sanitizeURL("javascript:alert(1)");
      expect(result).toBeNull();
    });
  });

  describe("Input Validation", () => {
    it("should validate email format", () => {
      const result1 = security.validate("test@example.com", { type: "email" });
      expect(result1.valid).toBe(true);

      const result2 = security.validate("invalid-email", { type: "email" });
      expect(result2.valid).toBe(false);
    });

    it("should validate required fields", () => {
      const result1 = security.validate("value", {
        type: "string",
        required: true,
      });
      expect(result1.valid).toBe(true);

      const result2 = security.validate("", { type: "string", required: true });
      expect(result2.valid).toBe(false);
    });

    it("should validate min length", () => {
      const result1 = security.validate("hello", {
        type: "string",
        minLength: 5,
      });
      expect(result1.valid).toBe(true);

      const result2 = security.validate("hi", { type: "string", minLength: 5 });
      expect(result2.valid).toBe(false);
    });

    it("should validate max length", () => {
      const result1 = security.validate("hello", {
        type: "string",
        maxLength: 5,
      });
      expect(result1.valid).toBe(true);

      const result2 = security.validate("hello world", {
        type: "string",
        maxLength: 5,
      });
      expect(result2.valid).toBe(false);
    });

    it("should validate patterns", () => {
      const result1 = security.validate("Hello", {
        type: "pattern",
        pattern: /^[a-zA-Z]+$/,
      });
      expect(result1.valid).toBe(true);

      const result2 = security.validate("Hello123", {
        type: "pattern",
        pattern: /^[a-zA-Z]+$/,
      });
      expect(result2.valid).toBe(false);
    });

    it("should validate numbers", () => {
      const result1 = security.validate(10, {
        type: "number",
        min: 5,
        max: 20,
      });
      expect(result1.valid).toBe(true);

      const result2 = security.validate(3, { type: "number", min: 5 });
      expect(result2.valid).toBe(false);
    });
  });

  describe("Rate Limiting", () => {
    it("should allow requests within limit", () => {
      const key = "test-limit-" + Date.now();
      const result = security.checkRateLimit(key, {
        maxRequests: 5,
        windowMs: 60000,
        blockDurationMs: 300000,
      });
      expect(result.allowed).toBe(true);
      expect(result.remaining).toBe(4);
    });

    it("should block requests exceeding limit", () => {
      const key = "test-exceed-" + Date.now();
      const options = {
        maxRequests: 2,
        windowMs: 60000,
        blockDurationMs: 300000,
      };

      security.checkRateLimit(key, options);
      security.checkRateLimit(key, options);
      const result = security.checkRateLimit(key, options);

      expect(result.allowed).toBe(false);
      expect(result.remaining).toBe(0);
    });
  });

  describe("Security Events", () => {
    it("should log security events", () => {
      security.logSecurityEvent({
        type: "auth",
        severity: "low",
        message: "User logged in",
        details: { userId: "123" },
      });

      const events = security.getSecurityEvents();
      expect(events.length).toBeGreaterThan(0);
    });

    it("should filter events by type", () => {
      security.logSecurityEvent({
        type: "auth",
        severity: "low",
        message: "Login",
      });

      const authEvents = security.getSecurityEvents({ type: "auth" });
      expect(authEvents.every((e) => e.type === "auth")).toBe(true);
    });

    it("should filter events by severity", () => {
      security.logSecurityEvent({
        type: "xss",
        severity: "high",
        message: "XSS blocked",
      });

      const highSeverity = security.getSecurityEvents({ severity: "high" });
      expect(highSeverity.every((e) => e.severity === "high")).toBe(true);
    });
  });

  describe("Security Summary", () => {
    it("should provide security summary", () => {
      security.logSecurityEvent({
        type: "auth",
        severity: "low",
        message: "Login",
      });

      const summary = security.getSecuritySummary();

      expect(summary.total).toBeGreaterThan(0);
      expect(summary.bySeverity).toBeDefined();
      expect(summary.byType).toBeDefined();
    });
  });

  describe("Three-Version Access Control", () => {
    it("should allow users to access V2 resources", () => {
      const userRole = "user";
      const version = "v2";
      const resourceType = "cr_ai"; // Code Review AI

      const canAccess = security.checkVersionAccess(
        userRole,
        version,
        resourceType
      );
      expect(canAccess).toBe(true);
    });

    it("should deny users access to V1 resources", () => {
      const userRole = "user";
      const version = "v1";
      const resourceType = "cr_ai";

      const canAccess = security.checkVersionAccess(
        userRole,
        version,
        resourceType
      );
      expect(canAccess).toBe(false);
    });

    it("should deny users access to V3 resources", () => {
      const userRole = "user";
      const version = "v3";
      const resourceType = "cr_ai";

      const canAccess = security.checkVersionAccess(
        userRole,
        version,
        resourceType
      );
      expect(canAccess).toBe(false);
    });

    it("should deny users access to VC-AI on any version", () => {
      const userRole = "user";
      const resourceType = "vc_ai"; // Version Control AI - admin only

      expect(security.checkVersionAccess(userRole, "v1", resourceType)).toBe(
        false
      );
      expect(security.checkVersionAccess(userRole, "v2", resourceType)).toBe(
        false
      );
      expect(security.checkVersionAccess(userRole, "v3", resourceType)).toBe(
        false
      );
    });

    it("should allow admins to access all versions", () => {
      const userRole = "admin";

      expect(security.checkVersionAccess(userRole, "v1", "cr_ai")).toBe(true);
      expect(security.checkVersionAccess(userRole, "v1", "vc_ai")).toBe(true);
      expect(security.checkVersionAccess(userRole, "v2", "cr_ai")).toBe(true);
      expect(security.checkVersionAccess(userRole, "v2", "vc_ai")).toBe(true);
      expect(security.checkVersionAccess(userRole, "v3", "cr_ai")).toBe(true);
      expect(security.checkVersionAccess(userRole, "v3", "vc_ai")).toBe(true);
    });

    it("should log unauthorized version access attempts", () => {
      const initialEvents = security.getSecurityEvents({
        type: "access",
      }).length;

      security.checkVersionAccess("user", "v1", "vc_ai");

      const newEvents = security.getSecurityEvents({ type: "access" });
      expect(newEvents.length).toBeGreaterThan(initialEvents);
    });
  });
});
