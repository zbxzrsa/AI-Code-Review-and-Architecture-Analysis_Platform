/**
 * Mock Server for Development
 *
 * Provides mock API responses when the backend is not running.
 * This allows frontend development without a running backend.
 */

import type { Plugin } from "vite";

// Mock user data
const mockUser = {
  id: "user-001",
  email: "admin@example.com",
  name: "Admin User",
  role: "admin",
  avatar: null,
  created_at: new Date().toISOString(),
  two_factor_enabled: false,
};

// Mock tokens
const mockTokens = {
  access_token: "mock_access_token_" + Date.now(),
  refresh_token: "mock_refresh_token_" + Date.now(),
  token_type: "Bearer",
  expires_in: 3600,
};

// Mock CSRF token
const mockCsrfToken =
  "mock_csrf_token_" + Math.random().toString(36).substr(2, 9);

// Mock projects
const mockProjects = [
  {
    id: "proj-001",
    name: "AI Code Review Platform",
    description: "Main platform project",
    language: "TypeScript",
    framework: "React",
    status: "active",
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    issues_count: 5,
    analyses_count: 12,
  },
  {
    id: "proj-002",
    name: "Backend Services",
    description: "Python FastAPI backend",
    language: "Python",
    framework: "FastAPI",
    status: "active",
    created_at: new Date(Date.now() - 86400000).toISOString(),
    updated_at: new Date().toISOString(),
    issues_count: 3,
    analyses_count: 8,
  },
];

// Route handlers
const mockRoutes: Record<string, (req: any) => any> = {
  "GET /api/csrf-token": () => ({
    token: mockCsrfToken,
    csrf_token: mockCsrfToken, // Also include for compatibility
  }),

  "POST /api/auth/login": (_req: any) => {
    // Simulate login - accept any credentials for dev
    return {
      ...mockTokens,
      user: mockUser,
    };
  },

  "POST /api/auth/register": () => ({
    ...mockTokens,
    user: mockUser,
  }),

  "POST /api/auth/refresh": () => ({
    ...mockTokens,
  }),

  "POST /api/auth/logout": () => ({
    success: true,
    message: "Logged out successfully",
  }),

  "GET /api/auth/me": () => mockUser,

  "GET /api/projects": () => ({
    projects: mockProjects,
    total: mockProjects.length,
  }),

  "GET /api/projects/:id": (req: any) => {
    const project = mockProjects.find((p) => p.id === req.params.id);
    return project || { error: "Not found", status: 404 };
  },

  "GET /api/dashboard/stats": () => ({
    total_projects: mockProjects.length,
    total_analyses: 20,
    issues_found: 8,
    issues_resolved: 5,
    resolution_rate: 0.625,
  }),

  "GET /api/dashboard/activity": () => ({
    activities: [
      {
        id: "act-001",
        type: "analysis_complete",
        message: "Code analysis completed for AI Code Review Platform",
        timestamp: new Date().toISOString(),
      },
      {
        id: "act-002",
        type: "issue_resolved",
        message: "Security issue resolved in auth.py",
        timestamp: new Date(Date.now() - 3600000).toISOString(),
      },
    ],
  }),

  "GET /api/auto-fix/status": () => ({
    running: true,
    phase: "idle",
    pending_fixes: 2,
    metrics: {
      cycles_completed: 15,
      vulnerabilities_detected: 25,
      fixes_applied: 15,
      fixes_verified: 12,
      fixes_failed: 3,
      fixes_rolled_back: 2,
    },
    last_cycle_at: new Date().toISOString(),
  }),

  "GET /api/evolution/status": () => ({
    running: true,
    cycle_count: 24,
    last_cycle_at: new Date().toISOString(),
    promotions: 5,
    degradations: 2,
    v1_metrics: {
      request_count: 2500,
      error_count: 75,
      error_rate: 0.03,
      avg_latency_ms: 2500,
      technology_count: 3,
    },
    v2_metrics: {
      request_count: 15000,
      error_count: 150,
      error_rate: 0.01,
      avg_latency_ms: 1800,
      technology_count: 5,
    },
    v3_metrics: {
      request_count: 500,
      error_count: 100,
      error_rate: 0.2,
      avg_latency_ms: 4000,
      technology_count: 2,
    },
  }),

  "GET /api/model-testing/models": () => [
    {
      model_id: "model-001",
      name: "GPT-4 Code Review",
      version: "v2",
      type: "code_review",
      status: "active",
      accuracy: 0.94,
      latency_ms: 1800,
      cost_per_1k: 0.03,
      requests_today: 1250,
    },
    {
      model_id: "model-002",
      name: "Claude-3 Security",
      version: "v2",
      type: "security",
      status: "active",
      accuracy: 0.92,
      latency_ms: 2100,
      cost_per_1k: 0.025,
      requests_today: 890,
    },
    {
      model_id: "model-003",
      name: "GQA Attention Model",
      version: "v1",
      type: "experimental",
      status: "testing",
      accuracy: 0.87,
      latency_ms: 2500,
      cost_per_1k: 0.02,
      requests_today: 150,
    },
  ],

  "GET /api/notifications": () => ({
    notifications: [],
    unread_count: 0,
  }),

  "GET /api/auto-fix/vulnerabilities": () => [
    {
      vuln_id: "vuln-001",
      pattern_id: "SEC-001",
      file_path: "backend/shared/security/auth.py",
      line_number: 19,
      severity: "critical",
      category: "security",
      description: "Hardcoded secret key",
      confidence: 0.95,
      detected_at: new Date().toISOString(),
    },
    {
      vuln_id: "vuln-002",
      pattern_id: "REL-001",
      file_path: "backend/shared/services/reliability.py",
      line_number: 41,
      severity: "medium",
      category: "reliability",
      description: "Deprecated datetime usage",
      confidence: 0.9,
      detected_at: new Date().toISOString(),
    },
    {
      vuln_id: "vuln-003",
      pattern_id: "SEC-003",
      file_path: "backend/shared/security/auth.py",
      line_number: 97,
      severity: "high",
      category: "security",
      description: "JWT decode without validation",
      confidence: 0.88,
      detected_at: new Date().toISOString(),
    },
  ],

  "GET /api/auto-fix/fixes": () => [
    {
      fix_id: "fix-001",
      vuln_id: "vuln-001",
      file_path: "backend/shared/security/auth.py",
      original_code: 'SECRET_KEY = "default"',
      fixed_code: 'SECRET_KEY = os.getenv("SECRET_KEY")',
      status: "verified",
      confidence: 0.95,
      applied_at: new Date().toISOString(),
      verified_at: new Date().toISOString(),
    },
    {
      fix_id: "fix-002",
      vuln_id: "vuln-002",
      file_path: "backend/shared/services/reliability.py",
      original_code: "datetime.utcnow()",
      fixed_code: "datetime.now(timezone.utc)",
      status: "applied",
      confidence: 0.9,
      applied_at: new Date().toISOString(),
    },
  ],

  "GET /api/auto-fix/fixes/pending": () => [
    {
      fix_id: "fix-003",
      vuln_id: "vuln-003",
      file_path: "backend/shared/security/auth.py",
      original_code: "jwt.decode(token, key)",
      fixed_code: 'jwt.decode(token, key, options={"verify": True})',
      status: "pending",
      confidence: 0.88,
    },
  ],

  "POST /api/auto-fix/start": () => ({
    success: true,
    message: "Auto-fix cycle started",
  }),
  "POST /api/auto-fix/stop": () => ({
    success: true,
    message: "Auto-fix cycle stopped",
  }),
  "POST /api/auto-fix/scan": () => ({
    success: true,
    message: "Scan triggered",
    vulnerabilities_found: 2,
  }),

  // Fix approval/rejection routes (parameterized)
  "POST /api/auto-fix/fixes/:id/approve": () => ({
    success: true,
    message: "Fix approved",
  }),
  "POST /api/auto-fix/fixes/:id/reject": () => ({
    success: true,
    message: "Fix rejected",
  }),
  "POST /api/auto-fix/fixes/:id/apply": () => ({
    success: true,
    message: "Fix applied",
  }),
  "POST /api/auto-fix/fixes/:id/rollback": () => ({
    success: true,
    message: "Fix rolled back",
  }),

  // Evolution cycle controls
  "POST /api/evolution/start": () => ({
    success: true,
    message: "Evolution cycle started",
  }),
  "POST /api/evolution/stop": () => ({
    success: true,
    message: "Evolution cycle stopped",
  }),
  "POST /api/evolution/promote": () => ({
    success: true,
    message: "Technology promoted",
  }),
  "POST /api/evolution/degrade": () => ({
    success: true,
    message: "Technology degraded",
  }),

  "GET /api/evolution/technologies": () => ({
    technologies: [
      {
        tech_id: "tech-001",
        name: "Multi-Head Attention",
        version: "v2",
        status: "promoted",
        metrics: { accuracy: 0.92 },
      },
      {
        tech_id: "tech-002",
        name: "GQA Attention",
        version: "v1",
        status: "experimental",
        metrics: { accuracy: 0.87 },
      },
    ],
    total: 2,
  }),

  "GET /api/evolution/experiments": () => ({
    experiments: [
      {
        experiment_id: "exp-001",
        name: "GQA Code Review Test",
        status: "running",
        accuracy: 0.87,
      },
    ],
    total: 1,
  }),

  "GET /api/users": () => ({
    users: [mockUser],
    total: 1,
  }),

  "GET /api/providers": () => [
    {
      id: "openai",
      name: "OpenAI",
      status: "active",
      models: ["gpt-4", "gpt-3.5-turbo"],
    },
    {
      id: "anthropic",
      name: "Anthropic",
      status: "active",
      models: ["claude-3-opus", "claude-3-sonnet"],
    },
  ],

  "GET /api/health": () => ({
    status: "healthy",
    services: {
      api: "healthy",
      database: "healthy",
      redis: "healthy",
      ai: "healthy",
    },
    uptime: 86400,
    version: "2.0.0",
  }),

  // Learning cycle endpoints
  "GET /api/learning/status": () => ({
    running: true,
    cycle_count: 120,
    total_knowledge_items: 32040,
    items_today: 245,
    learning_accuracy: 0.94,
    model_version: "v2.3.1",
    last_fine_tune: new Date(Date.now() - 86400000).toISOString(),
    next_scheduled_tune: new Date(Date.now() + 43200000).toISOString(),
  }),

  "GET /api/learning/sources": () => [
    {
      id: "src-001",
      name: "GitHub Repositories",
      type: "github",
      enabled: true,
      items_processed: 15420,
      status: "active",
    },
    {
      id: "src-002",
      name: "arXiv Papers",
      type: "papers",
      enabled: true,
      items_processed: 2340,
      status: "active",
    },
    {
      id: "src-003",
      name: "Tech Blogs",
      type: "blogs",
      enabled: true,
      items_processed: 8750,
      status: "syncing",
    },
    {
      id: "src-004",
      name: "Documentation",
      type: "docs",
      enabled: true,
      items_processed: 4280,
      status: "active",
    },
    {
      id: "src-005",
      name: "User Feedback",
      type: "feedback",
      enabled: true,
      items_processed: 1250,
      status: "active",
    },
  ],

  "GET /api/learning/updates": () => [
    {
      id: "upd-001",
      source: "GitHub",
      title: "New React 19 patterns detected",
      type: "Pattern",
      impact: "high",
      timestamp: new Date().toISOString(),
    },
    {
      id: "upd-002",
      source: "arXiv",
      title: "GQA attention optimization paper",
      type: "Research",
      impact: "medium",
      timestamp: new Date(Date.now() - 1800000).toISOString(),
    },
    {
      id: "upd-003",
      source: "Feedback",
      title: "Security pattern improvement",
      type: "Improvement",
      impact: "high",
      timestamp: new Date(Date.now() - 3600000).toISOString(),
    },
  ],

  "POST /api/learning/sync": () => ({ success: true, message: "Sync started" }),
  "POST /api/learning/pause": () => ({
    success: true,
    message: "Learning paused",
  }),
  "POST /api/learning/resume": () => ({
    success: true,
    message: "Learning resumed",
  }),

  // Model comparison endpoints
  "GET /api/models/comparison": () => ({
    models: [
      {
        id: "gpt4-review",
        name: "GPT-4 Code Review",
        version: "v2",
        accuracy: 0.94,
        latencyP50: 1800,
        costPer1k: 0.03,
      },
      {
        id: "claude3-security",
        name: "Claude-3 Security",
        version: "v2",
        accuracy: 0.92,
        latencyP50: 2100,
        costPer1k: 0.025,
      },
      {
        id: "gqa-attention",
        name: "GQA Attention Model",
        version: "v1",
        accuracy: 0.87,
        latencyP50: 2500,
        costPer1k: 0.02,
      },
      {
        id: "llama-quality",
        name: "LLaMA Quality Analyzer",
        version: "v1",
        accuracy: 0.85,
        latencyP50: 1500,
        costPer1k: 0.01,
      },
    ],
  }),

  "POST /api/models/compare": () => ({
    success: true,
    comparison_id: "cmp-" + Date.now(),
    status: "running",
    message: "Comparison started",
  }),

  "POST /api/models/ab-test": () => ({
    success: true,
    test_id: "ab-" + Date.now(),
    status: "scheduled",
    message: "A/B test scheduled",
  }),

  // Code quality endpoints
  "GET /api/quality/metrics": () => ({
    overall_score: 85,
    categories: [
      { name: "Security", score: 92, trend: 5 },
      { name: "Reliability", score: 88, trend: 3 },
      { name: "Maintainability", score: 78, trend: -2 },
      { name: "Performance", score: 85, trend: 8 },
      { name: "Documentation", score: 72, trend: 0 },
    ],
  }),
};

/**
 * Create Vite plugin for mock API
 */
export function mockApiPlugin(): Plugin {
  return {
    name: "mock-api",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = req.url || "";
        const method = req.method || "GET";

        // Only handle /api routes
        if (!url.startsWith("/api")) {
          return next();
        }

        // Handle CORS preflight
        if (method === "OPTIONS") {
          res.setHeader("Access-Control-Allow-Origin", "*");
          res.setHeader(
            "Access-Control-Allow-Methods",
            "GET, POST, PUT, DELETE, OPTIONS"
          );
          res.setHeader(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization, X-CSRF-Token"
          );
          res.statusCode = 204;
          res.end();
          return;
        }

        // Find matching route
        const routeKey = `${method} ${url.split("?")[0]}`;
        let handler = mockRoutes[routeKey];

        // Check for parameterized routes
        if (!handler) {
          for (const [pattern, h] of Object.entries(mockRoutes)) {
            const [m, p] = pattern.split(" ");
            if (m !== method) continue;

            const regex = new RegExp(
              "^" + p.replace(/:[^/]+/g, "([^/]+)") + "$"
            );
            const match = url.split("?")[0].match(regex);
            if (match) {
              handler = h;
              break;
            }
          }
        }

        if (handler) {
          // Parse request body for POST requests
          let body = "";
          req.on("data", (chunk: Buffer) => {
            body += chunk.toString();
          });

          req.on("end", () => {
            try {
              const parsedBody = body ? JSON.parse(body) : {};
              const result = handler({ body: parsedBody, params: {} });

              res.setHeader("Content-Type", "application/json");
              res.setHeader("Access-Control-Allow-Origin", "*");

              if (result?.status === 404) {
                res.statusCode = 404;
              }

              res.end(JSON.stringify(result));
            } catch (e) {
              res.statusCode = 500;
              res.end(JSON.stringify({ error: "Mock server error" }));
            }
          });
        } else {
          // No mock handler, let it pass through (will get 500 from real server)
          // Or return a generic mock response
          res.setHeader("Content-Type", "application/json");
          res.end(
            JSON.stringify({
              message: "Mock endpoint not implemented",
              path: url,
              method: method,
            })
          );
        }
      });
    },
  };
}

export default mockApiPlugin;
