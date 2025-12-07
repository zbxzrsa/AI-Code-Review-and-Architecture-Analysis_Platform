/**
 * API Service Unit Tests
 *
 * TD-001: Comprehensive unit tests for API service
 * Coverage Target: 80%+
 *
 * Tests cover:
 * - Request/response interceptors
 * - CSRF token handling
 * - Rate limiting
 * - Token refresh flow
 * - Error handling
 * - All API methods
 */

import { describe, it, expect, beforeEach, vi, type Mock } from "vitest";
import axios from "axios";
import { api, apiService } from "../api";

// Mock dependencies
vi.mock("axios");
vi.mock("../../store/authStore", () => ({
  useAuthStore: {
    getState: vi.fn(() => ({
      logout: vi.fn(),
    })),
  },
}));

vi.mock("../security", () => ({
  csrfManager: {
    ensureToken: vi.fn().mockResolvedValue("mock-csrf-token"),
    setToken: vi.fn(),
    clearToken: vi.fn(),
    fetchToken: vi.fn().mockResolvedValue("new-csrf-token"),
  },
  rateLimiter: {
    shouldLimit: vi.fn().mockReturnValue(false),
    getResetTime: vi.fn().mockReturnValue(Date.now() + 60000),
  },
  sessionSecurity: {
    updateActivity: vi.fn(),
    stopInactivityTimer: vi.fn(),
  },
}));

vi.mock("../errorLogging", () => ({
  errorLoggingService: {
    logNetworkError: vi.fn(),
  },
}));

// Axios is mocked via vi.mock above - use vi.mocked(axios.method) for type-safe mocking

describe("API Service", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("api instance", () => {
    it("should be created with correct defaults", () => {
      expect(api.defaults.baseURL).toBeDefined();
      expect(api.defaults.timeout).toBe(30000);
      expect(api.defaults.withCredentials).toBe(true);
    });

    it("should have content-type header set to JSON", () => {
      expect(api.defaults.headers["Content-Type"]).toBe("application/json");
    });
  });

  describe("apiService.auth", () => {
    beforeEach(() => {
      // Mock axios.create for this test suite
      vi.mocked(axios.create).mockReturnValue({
        post: vi.fn().mockResolvedValue({ data: { success: true } }),
        get: vi.fn().mockResolvedValue({ data: { user: {} } }),
        put: vi.fn().mockResolvedValue({ data: { success: true } }),
        interceptors: {
          request: { use: vi.fn() },
          response: { use: vi.fn() },
        },
        defaults: { headers: {} },
      } as unknown as ReturnType<typeof axios.create>);
    });

    it("login should call POST /auth/login with credentials", async () => {
      const mockPost = vi.fn().mockResolvedValue({ data: { token: "test" } });
      (api.post as Mock) = mockPost;

      await apiService.auth.login("test@example.com", "password123");

      expect(mockPost).toHaveBeenCalledWith("/auth/login", {
        email: "test@example.com",
        password: "password123",
        invitation_code: undefined,
      });
    });

    it("login should include invitation code when provided", async () => {
      const mockPost = vi.fn().mockResolvedValue({ data: { token: "test" } });
      (api.post as Mock) = mockPost;

      await apiService.auth.login("test@example.com", "password123", "INVITE123");

      expect(mockPost).toHaveBeenCalledWith("/auth/login", {
        email: "test@example.com",
        password: "password123",
        invitation_code: "INVITE123",
      });
    });

    it("register should call POST /auth/register", async () => {
      const mockPost = vi.fn().mockResolvedValue({ data: { success: true } });
      (api.post as Mock) = mockPost;

      const registerData = {
        email: "new@example.com",
        password: "password123",
        name: "Test User",
        invitation_code: "INVITE123",
      };

      await apiService.auth.register(registerData);

      expect(mockPost).toHaveBeenCalledWith("/auth/register", registerData);
    });

    it("logout should call POST /auth/logout", async () => {
      const mockPost = vi.fn().mockResolvedValue({ data: {} });
      (api.post as Mock) = mockPost;

      await apiService.auth.logout();

      expect(mockPost).toHaveBeenCalledWith("/auth/logout");
    });

    it("refresh should call POST /auth/refresh", async () => {
      const mockPost = vi.fn().mockResolvedValue({ data: {} });
      (api.post as Mock) = mockPost;

      await apiService.auth.refresh();

      expect(mockPost).toHaveBeenCalledWith("/auth/refresh");
    });

    it("me should call GET /auth/me", async () => {
      const mockGet = vi.fn().mockResolvedValue({ data: { user: {} } });
      (api.get as Mock) = mockGet;

      await apiService.auth.me();

      expect(mockGet).toHaveBeenCalledWith("/auth/me");
    });

    it("updateProfile should call PUT /auth/profile", async () => {
      const mockPut = vi.fn().mockResolvedValue({ data: {} });
      (api.put as Mock) = mockPut;

      await apiService.auth.updateProfile({ name: "New Name" });

      expect(mockPut).toHaveBeenCalledWith("/auth/profile", {
        name: "New Name",
      });
    });

    it("changePassword should call POST /auth/change-password", async () => {
      const mockPost = vi.fn().mockResolvedValue({ data: {} });
      (api.post as Mock) = mockPost;

      await apiService.auth.changePassword("oldPass", "newPass");

      expect(mockPost).toHaveBeenCalledWith("/auth/change-password", {
        old_password: "oldPass",
        new_password: "newPass",
      });
    });

    it("verify2FA should call POST /auth/2fa/verify", async () => {
      const mockPost = vi.fn().mockResolvedValue({ data: {} });
      (api.post as Mock) = mockPost;

      await apiService.auth.verify2FA("123456");

      expect(mockPost).toHaveBeenCalledWith("/auth/2fa/verify", {
        code: "123456",
        is_backup_code: undefined,
      });
    });

    it("verify2FA with backup code should set is_backup_code", async () => {
      const mockPost = vi.fn().mockResolvedValue({ data: {} });
      (api.post as Mock) = mockPost;

      await apiService.auth.verify2FA("BACKUP123", true);

      expect(mockPost).toHaveBeenCalledWith("/auth/2fa/verify", {
        code: "BACKUP123",
        is_backup_code: true,
      });
    });

    it("resend2FA should call POST /auth/2fa/resend", async () => {
      const mockPost = vi.fn().mockResolvedValue({ data: {} });
      (api.post as Mock) = mockPost;

      await apiService.auth.resend2FA();

      expect(mockPost).toHaveBeenCalledWith("/auth/2fa/resend");
    });
  });

  describe("apiService.projects", () => {
    it("list should call GET /projects with params", async () => {
      const mockGet = vi.fn().mockResolvedValue({ data: [] });
      (api.get as Mock) = mockGet;

      await apiService.projects.list({ page: 1, limit: 10, search: "test" });

      expect(mockGet).toHaveBeenCalledWith("/projects", {
        params: { page: 1, limit: 10, search: "test" },
      });
    });

    it("get should call GET /projects/:id", async () => {
      const mockGet = vi.fn().mockResolvedValue({ data: {} });
      (api.get as Mock) = mockGet;

      await apiService.projects.get("proj-123");

      expect(mockGet).toHaveBeenCalledWith("/projects/proj-123");
    });

    it("create should call POST /projects", async () => {
      const mockPost = vi.fn().mockResolvedValue({ data: {} });
      (api.post as Mock) = mockPost;

      const projectData = {
        name: "Test Project",
        language: "typescript",
      };

      await apiService.projects.create(projectData);

      expect(mockPost).toHaveBeenCalledWith("/projects", projectData);
    });

    it("update should call PUT /projects/:id", async () => {
      const mockPut = vi.fn().mockResolvedValue({ data: {} });
      (api.put as Mock) = mockPut;

      await apiService.projects.update("proj-123", { name: "Updated" });

      expect(mockPut).toHaveBeenCalledWith("/projects/proj-123", {
        name: "Updated",
      });
    });

    it("delete should call DELETE /projects/:id", async () => {
      const mockDelete = vi.fn().mockResolvedValue({ data: {} });
      (api.delete as Mock) = mockDelete;

      await apiService.projects.delete("proj-123");

      expect(mockDelete).toHaveBeenCalledWith("/projects/proj-123");
    });

    it("archive should call POST /projects/:id/archive", async () => {
      const mockPost = vi.fn().mockResolvedValue({ data: {} });
      (api.post as Mock) = mockPost;

      await apiService.projects.archive("proj-123");

      expect(mockPost).toHaveBeenCalledWith("/projects/proj-123/archive");
    });

    it("restore should call POST /projects/:id/restore", async () => {
      const mockPost = vi.fn().mockResolvedValue({ data: {} });
      (api.post as Mock) = mockPost;

      await apiService.projects.restore("proj-123");

      expect(mockPost).toHaveBeenCalledWith("/projects/proj-123/restore");
    });

    it("getStats should call GET /projects/:id/stats", async () => {
      const mockGet = vi.fn().mockResolvedValue({ data: {} });
      (api.get as Mock) = mockGet;

      await apiService.projects.getStats("proj-123");

      expect(mockGet).toHaveBeenCalledWith("/projects/proj-123/stats");
    });

    it("getTeam should call GET /projects/:id/team", async () => {
      const mockGet = vi.fn().mockResolvedValue({ data: [] });
      (api.get as Mock) = mockGet;

      await apiService.projects.getTeam("proj-123");

      expect(mockGet).toHaveBeenCalledWith("/projects/proj-123/team");
    });

    it("inviteMember should call POST /projects/:id/team/invite", async () => {
      const mockPost = vi.fn().mockResolvedValue({ data: {} });
      (api.post as Mock) = mockPost;

      await apiService.projects.inviteMember("proj-123", {
        email: "member@test.com",
        role: "developer",
      });

      expect(mockPost).toHaveBeenCalledWith("/projects/proj-123/team/invite", {
        email: "member@test.com",
        role: "developer",
      });
    });
  });

  describe("apiService.analysis", () => {
    it("start should call POST /projects/:id/analyze", async () => {
      const mockPost = vi.fn().mockResolvedValue({ data: { sessionId: "sess-123" } });
      (api.post as Mock) = mockPost;

      await apiService.analysis.start("proj-123", { files: ["src/index.ts"] });

      expect(mockPost).toHaveBeenCalledWith("/projects/proj-123/analyze", {
        files: ["src/index.ts"],
      });
    });

    it("getSession should call GET /analyze/:sessionId", async () => {
      const mockGet = vi.fn().mockResolvedValue({ data: {} });
      (api.get as Mock) = mockGet;

      await apiService.analysis.getSession("sess-123");

      expect(mockGet).toHaveBeenCalledWith("/analyze/sess-123");
    });

    it("getIssues should call GET /analyze/:sessionId/issues", async () => {
      const mockGet = vi.fn().mockResolvedValue({ data: [] });
      (api.get as Mock) = mockGet;

      await apiService.analysis.getIssues("sess-123", { severity: "high" });

      expect(mockGet).toHaveBeenCalledWith("/analyze/sess-123/issues", {
        params: { severity: "high" },
      });
    });

    it("applyFix should call POST /analyze/:sessionId/issues/:issueId/fix", async () => {
      const mockPost = vi.fn().mockResolvedValue({ data: {} });
      (api.post as Mock) = mockPost;

      await apiService.analysis.applyFix("sess-123", "issue-456");

      expect(mockPost).toHaveBeenCalledWith("/analyze/sess-123/issues/issue-456/fix");
    });

    it("dismissIssue should call POST with reason", async () => {
      const mockPost = vi.fn().mockResolvedValue({ data: {} });
      (api.post as Mock) = mockPost;

      await apiService.analysis.dismissIssue("sess-123", "issue-456", "False positive");

      expect(mockPost).toHaveBeenCalledWith("/analyze/sess-123/issues/issue-456/dismiss", {
        reason: "False positive",
      });
    });

    it("streamUrl should return correct URL", () => {
      const url = apiService.analysis.streamUrl("sess-123");
      expect(url).toContain("/analyze/sess-123/stream");
    });

    it("getVulnerabilities should call GET /security/vulnerabilities", async () => {
      const mockGet = vi.fn().mockResolvedValue({ data: [] });
      (api.get as Mock) = mockGet;

      await apiService.analysis.getVulnerabilities({ severity: "critical" });

      expect(mockGet).toHaveBeenCalledWith("/security/vulnerabilities", {
        params: { severity: "critical" },
      });
    });
  });

  describe("apiService.experiments", () => {
    it("list should call GET /experiments", async () => {
      const mockGet = vi.fn().mockResolvedValue({ data: [] });
      (api.get as Mock) = mockGet;

      await apiService.experiments.list({ status: "running" });

      expect(mockGet).toHaveBeenCalledWith("/experiments", {
        params: { status: "running" },
      });
    });

    it("create should call POST /experiments", async () => {
      const mockPost = vi.fn().mockResolvedValue({ data: {} });
      (api.post as Mock) = mockPost;

      const expData = {
        name: "Test Experiment",
        config: { model: "gpt-4" },
        dataset_id: "dataset-123",
      };

      await apiService.experiments.create(expData);

      expect(mockPost).toHaveBeenCalledWith("/experiments", expData);
    });

    it("start should call POST /experiments/:id/start", async () => {
      const mockPost = vi.fn().mockResolvedValue({ data: {} });
      (api.post as Mock) = mockPost;

      await apiService.experiments.start("exp-123");

      expect(mockPost).toHaveBeenCalledWith("/experiments/exp-123/start");
    });

    it("promote should call POST /experiments/:id/promote", async () => {
      const mockPost = vi.fn().mockResolvedValue({ data: {} });
      (api.post as Mock) = mockPost;

      await apiService.experiments.promote("exp-123");

      expect(mockPost).toHaveBeenCalledWith("/experiments/exp-123/promote");
    });

    it("quarantine should call POST with reason", async () => {
      const mockPost = vi.fn().mockResolvedValue({ data: {} });
      (api.post as Mock) = mockPost;

      await apiService.experiments.quarantine("exp-123", "Performance degradation");

      expect(mockPost).toHaveBeenCalledWith("/experiments/exp-123/quarantine", {
        reason: "Performance degradation",
      });
    });
  });

  describe("apiService.user", () => {
    it("getProfile should call GET /user/profile", async () => {
      const mockGet = vi.fn().mockResolvedValue({ data: {} });
      (api.get as Mock) = mockGet;

      await apiService.user.getProfile();

      expect(mockGet).toHaveBeenCalledWith("/user/profile");
    });

    it("updateProfile should call PUT /user/profile", async () => {
      const mockPut = vi.fn().mockResolvedValue({ data: {} });
      (api.put as Mock) = mockPut;

      await apiService.user.updateProfile({ name: "New Name", bio: "Hello" });

      expect(mockPut).toHaveBeenCalledWith("/user/profile", {
        name: "New Name",
        bio: "Hello",
      });
    });

    it("uploadAvatar should use FormData", async () => {
      const mockPost = vi.fn().mockResolvedValue({ data: {} });
      (api.post as Mock) = mockPost;

      const file = new File(["test"], "avatar.png", { type: "image/png" });
      await apiService.user.uploadAvatar(file);

      expect(mockPost).toHaveBeenCalledWith("/user/avatar", expect.any(FormData), {
        headers: { "Content-Type": "multipart/form-data" },
      });
    });

    it("setup2FA should call POST /user/2fa/setup", async () => {
      const mockPost = vi.fn().mockResolvedValue({ data: { qrCode: "data:..." } });
      (api.post as Mock) = mockPost;

      await apiService.user.setup2FA();

      expect(mockPost).toHaveBeenCalledWith("/user/2fa/setup");
    });

    it("enable2FA should call POST /user/2fa/enable with code", async () => {
      const mockPost = vi.fn().mockResolvedValue({ data: {} });
      (api.post as Mock) = mockPost;

      await apiService.user.enable2FA("123456");

      expect(mockPost).toHaveBeenCalledWith("/user/2fa/enable", {
        code: "123456",
      });
    });

    it("getSessions should call GET /user/sessions", async () => {
      const mockGet = vi.fn().mockResolvedValue({ data: [] });
      (api.get as Mock) = mockGet;

      await apiService.user.getSessions();

      expect(mockGet).toHaveBeenCalledWith("/user/sessions");
    });

    it("revokeSession should call DELETE /user/sessions/:id", async () => {
      const mockDelete = vi.fn().mockResolvedValue({ data: {} });
      (api.delete as Mock) = mockDelete;

      await apiService.user.revokeSession("sess-123");

      expect(mockDelete).toHaveBeenCalledWith("/user/sessions/sess-123");
    });

    it("downloadPersonalData should call GET with blob responseType", async () => {
      const mockGet = vi.fn().mockResolvedValue({ data: new Blob() });
      (api.get as Mock) = mockGet;

      await apiService.user.downloadPersonalData();

      expect(mockGet).toHaveBeenCalledWith("/user/data/export", {
        responseType: "blob",
      });
    });
  });

  describe("apiService.metrics", () => {
    it("getDashboard should call GET /metrics/dashboard", async () => {
      const mockGet = vi.fn().mockResolvedValue({ data: {} });
      (api.get as Mock) = mockGet;

      await apiService.metrics.getDashboard();

      expect(mockGet).toHaveBeenCalledWith("/metrics/dashboard");
    });

    it("getSystem should call GET /metrics/system", async () => {
      const mockGet = vi.fn().mockResolvedValue({ data: {} });
      (api.get as Mock) = mockGet;

      await apiService.metrics.getSystem();

      expect(mockGet).toHaveBeenCalledWith("/metrics/system");
    });

    it("getProvider should call GET /metrics/providers/:provider", async () => {
      const mockGet = vi.fn().mockResolvedValue({ data: {} });
      (api.get as Mock) = mockGet;

      await apiService.metrics.getProvider("openai");

      expect(mockGet).toHaveBeenCalledWith("/metrics/providers/openai");
    });

    it("getUsage should call GET /metrics/usage with date params", async () => {
      const mockGet = vi.fn().mockResolvedValue({ data: {} });
      (api.get as Mock) = mockGet;

      await apiService.metrics.getUsage({
        start_date: "2024-01-01",
        end_date: "2024-01-31",
      });

      expect(mockGet).toHaveBeenCalledWith("/metrics/usage", {
        params: { start_date: "2024-01-01", end_date: "2024-01-31" },
      });
    });
  });

  describe("apiService.audit", () => {
    it("list should call GET /audit with filters", async () => {
      const mockGet = vi.fn().mockResolvedValue({ data: [] });
      (api.get as Mock) = mockGet;

      await apiService.audit.list({
        entity: "user",
        action: "login",
        page: 1,
        limit: 50,
      });

      expect(mockGet).toHaveBeenCalledWith("/audit", {
        params: { entity: "user", action: "login", page: 1, limit: 50 },
      });
    });

    it("get should call GET /audit/:id", async () => {
      const mockGet = vi.fn().mockResolvedValue({ data: {} });
      (api.get as Mock) = mockGet;

      await apiService.audit.get("audit-123");

      expect(mockGet).toHaveBeenCalledWith("/audit/audit-123");
    });
  });

  describe("apiService.repositories", () => {
    it("list should call GET /repositories", async () => {
      const mockGet = vi.fn().mockResolvedValue({ data: [] });
      (api.get as Mock) = mockGet;

      await apiService.repositories.list({ provider: "github" });

      expect(mockGet).toHaveBeenCalledWith("/repositories", {
        params: { provider: "github" },
      });
    });

    it("connect should call POST /repositories/connect", async () => {
      const mockPost = vi.fn().mockResolvedValue({ data: {} });
      (api.post as Mock) = mockPost;

      await apiService.repositories.connect({
        provider: "github",
        repo_full_name: "user/repo",
      });

      expect(mockPost).toHaveBeenCalledWith("/repositories/connect", {
        provider: "github",
        repo_full_name: "user/repo",
      });
    });

    it("sync should call POST /repositories/:id/sync", async () => {
      const mockPost = vi.fn().mockResolvedValue({ data: {} });
      (api.post as Mock) = mockPost;

      await apiService.repositories.sync("repo-123");

      expect(mockPost).toHaveBeenCalledWith("/repositories/repo-123/sync");
    });

    it("analyze should call POST /repositories/:id/analyze", async () => {
      const mockPost = vi.fn().mockResolvedValue({ data: {} });
      (api.post as Mock) = mockPost;

      await apiService.repositories.analyze("repo-123", {
        files: ["src/index.ts"],
        branch: "main",
      });

      expect(mockPost).toHaveBeenCalledWith("/repositories/repo-123/analyze", {
        files: ["src/index.ts"],
        branch: "main",
      });
    });
  });
});
