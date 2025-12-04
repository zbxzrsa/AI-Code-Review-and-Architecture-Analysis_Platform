/**
 * Unit Tests for Lifecycle API Client
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { LifecycleApiClient } from "../lifecycleApi";
import { api } from "../api";

// Mock the api module
vi.mock("../api", () => ({
  api: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
  },
}));

describe("LifecycleApiClient", () => {
  let client: LifecycleApiClient;

  beforeEach(() => {
    client = new LifecycleApiClient();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  describe("getComparisonRequests", () => {
    it("should fetch comparison requests with default params", async () => {
      const mockResponse = {
        data: {
          requests: [
            { requestId: "req-1", code: "test", language: "javascript" },
          ],
          total: 1,
          limit: 50,
          offset: 0,
        },
      };

      vi.mocked(api.get).mockResolvedValue(mockResponse);

      const result = await client.getComparisonRequests();

      expect(api.get).toHaveBeenCalledWith(
        "/api/admin/lifecycle/comparison-requests",
        { params: undefined }
      );
      expect(result.requests).toHaveLength(1);
      expect(result.total).toBe(1);
    });

    it("should fetch comparison requests with filters", async () => {
      const mockResponse = {
        data: {
          requests: [],
          total: 0,
          limit: 10,
          offset: 0,
        },
      };

      vi.mocked(api.get).mockResolvedValue(mockResponse);

      await client.getComparisonRequests({
        limit: 10,
        offset: 0,
        language: "python",
        hasV1: true,
      });

      expect(api.get).toHaveBeenCalledWith(
        "/api/admin/lifecycle/comparison-requests",
        {
          params: {
            limit: 10,
            offset: 0,
            language: "python",
            hasV1: true,
          },
        }
      );
    });
  });

  describe("getComparisonRequest", () => {
    it("should fetch a specific comparison request", async () => {
      const mockRequest = {
        requestId: "req-123",
        code: "function test() {}",
        language: "javascript",
        timestamp: "2024-01-01T00:00:00Z",
        v1Output: null,
        v2Output: null,
      };

      vi.mocked(api.get).mockResolvedValue({ data: mockRequest });

      const result = await client.getComparisonRequest("req-123");

      expect(api.get).toHaveBeenCalledWith(
        "/api/admin/lifecycle/comparison-requests/req-123"
      );
      expect(result.requestId).toBe("req-123");
    });
  });

  describe("initiateRollback", () => {
    it("should initiate a rollback", async () => {
      const mockResponse = {
        data: {
          success: true,
          message: "Rollback initiated",
          rollbackId: "rb-123",
        },
      };

      vi.mocked(api.post).mockResolvedValue(mockResponse);

      const result = await client.initiateRollback({
        versionId: "v1-abc123",
        reason: "accuracy_regression",
        notes: "Accuracy dropped below threshold",
      });

      expect(api.post).toHaveBeenCalledWith("/api/admin/lifecycle/rollback", {
        versionId: "v1-abc123",
        reason: "accuracy_regression",
        notes: "Accuracy dropped below threshold",
      });
      expect(result.success).toBe(true);
      expect(result.rollbackId).toBe("rb-123");
    });
  });

  describe("getRollbackHistory", () => {
    it("should fetch rollback history", async () => {
      const mockHistory = [
        {
          rollbackId: "rb-1",
          versionId: "v1-abc",
          reason: "accuracy",
          timestamp: "2024-01-01T00:00:00Z",
        },
      ];

      vi.mocked(api.get).mockResolvedValue({ data: mockHistory });

      const result = await client.getRollbackHistory({ limit: 10 });

      expect(api.get).toHaveBeenCalledWith(
        "/api/admin/lifecycle/rollback/history",
        { params: { limit: 10 } }
      );
      expect(result).toHaveLength(1);
    });
  });

  describe("listVersions", () => {
    it("should list all active versions", async () => {
      const mockVersions = {
        versions: [
          { versionId: "v1-exp", currentState: "shadow" },
          { versionId: "v2-stable", currentState: "stable" },
        ],
      };

      vi.mocked(api.get).mockResolvedValue({ data: mockVersions });

      const result = await client.listVersions();

      expect(api.get).toHaveBeenCalledWith("/api/admin/lifecycle/versions");
      expect(result.versions).toHaveLength(2);
    });
  });

  describe("triggerEvaluation", () => {
    it("should trigger evaluation for a version", async () => {
      const mockResponse = {
        data: {
          versionId: "v1-new",
          metrics: {
            p95LatencyMs: 2500,
            errorRate: 0.01,
            accuracy: 0.92,
          },
          decision: {
            approved: true,
            reason: "All metrics within thresholds",
          },
        },
      };

      vi.mocked(api.post).mockResolvedValue(mockResponse);

      const result = await client.triggerEvaluation("v1-new");

      expect(api.post).toHaveBeenCalledWith(
        "/api/admin/lifecycle/versions/v1-new/evaluate"
      );
      expect(result.decision.approved).toBe(true);
    });
  });

  describe("startGoldSetEvaluation", () => {
    it("should start gold-set evaluation", async () => {
      const mockResponse = {
        data: {
          status: "started",
          message: "Gold-set evaluation started",
          checkStatusAt: "/api/admin/evaluation/status/v1-new",
        },
      };

      vi.mocked(api.post).mockResolvedValue(mockResponse);

      const result = await client.startGoldSetEvaluation({
        versionId: "v1-new",
        modelVersion: "gpt-4o",
        promptVersion: "code-review-v3",
        testSets: ["security", "quality"],
      });

      expect(api.post).toHaveBeenCalledWith(
        "/api/admin/evaluation/evaluate/gold-set",
        {
          versionId: "v1-new",
          modelVersion: "gpt-4o",
          promptVersion: "code-review-v3",
          testSets: ["security", "quality"],
        }
      );
      expect(result.status).toBe("started");
    });
  });

  describe("getEvaluationStatus", () => {
    it("should get evaluation status", async () => {
      const mockStatus = {
        evaluationId: "eval-123",
        status: "completed",
        overallPassRate: 0.95,
        promotionRecommended: true,
        recommendationReason: "All criteria met",
      };

      vi.mocked(api.get).mockResolvedValue({ data: mockStatus });

      const result = await client.getEvaluationStatus("v1-new");

      expect(api.get).toHaveBeenCalledWith(
        "/api/admin/evaluation/evaluate/status/v1-new"
      );
      expect(result.status).toBe("completed");
      expect(result.promotionRecommended).toBe(true);
    });
  });

  describe("getComparisonStats", () => {
    it("should fetch comparison statistics", async () => {
      const mockStats = {
        totalRequests: 100,
        withV1Output: 95,
        withV3Output: 10,
        languages: { javascript: 50, python: 30 },
        comparisonMetrics: {
          samples: 95,
          avgIssueCountDelta: 0.5,
          avgLatencyDeltaMs: -200,
        },
      };

      vi.mocked(api.get).mockResolvedValue({ data: mockStats });

      const result = await client.getComparisonStats();

      expect(api.get).toHaveBeenCalledWith(
        "/api/admin/lifecycle/stats/comparison"
      );
      expect(result.totalRequests).toBe(100);
    });
  });

  describe("getRollbackStats", () => {
    it("should fetch rollback statistics", async () => {
      const mockStats = {
        totalRollbacks: 5,
        last7Days: 2,
        byReason: {
          accuracy_regression: 2,
          latency_increase: 1,
          security_failure: 2,
        },
      };

      vi.mocked(api.get).mockResolvedValue({ data: mockStats });

      const result = await client.getRollbackStats();

      expect(api.get).toHaveBeenCalledWith(
        "/api/admin/lifecycle/stats/rollbacks"
      );
      expect(result.totalRollbacks).toBe(5);
    });
  });

  describe("getEventHistory", () => {
    it("should fetch lifecycle event history", async () => {
      const mockEvents = {
        events: [
          {
            timestamp: "2024-01-01T00:00:00Z",
            versionId: "v1-new",
            eventType: "evaluation_completed",
            currentState: "shadow",
            details: {},
          },
        ],
      };

      vi.mocked(api.get).mockResolvedValue({ data: mockEvents });

      const result = await client.getEventHistory(50);

      expect(api.get).toHaveBeenCalledWith("/api/admin/lifecycle/history", {
        params: { limit: 50 },
      });
      expect(result.events).toHaveLength(1);
    });
  });

  describe("checkHealth", () => {
    it("should check lifecycle controller health", async () => {
      vi.mocked(api.get).mockResolvedValue({ data: { status: "healthy" } });

      const result = await client.checkHealth();

      expect(api.get).toHaveBeenCalledWith("/api/admin/lifecycle/health");
      expect(result.status).toBe("healthy");
    });
  });
});
