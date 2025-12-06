/**
 * Lifecycle Controller API Client
 *
 * Provides typed access to the lifecycle controller endpoints
 * for version management, comparison, and rollback operations.
 */

import { api } from "./api";

// Types
export interface VersionOutput {
  version: "v1" | "v2" | "v3";
  versionId: string;
  modelVersion: string;
  promptVersion: string;
  timestamp: string;
  latencyMs: number;
  cost: number;
  issues: Issue[];
  rawOutput: string;
  confidence: number;
  securityPassed: boolean;
}

export interface Issue {
  id: string;
  type: string;
  severity: "critical" | "high" | "medium" | "low";
  message: string;
  file: string;
  line: number;
  suggestion?: string;
}

export interface ComparisonRequest {
  requestId: string;
  code: string;
  language: string;
  timestamp: string;
  v1Output?: VersionOutput;
  v2Output?: VersionOutput;
  v3Output?: VersionOutput;
}

export interface ComparisonListResponse {
  requests: ComparisonRequest[];
  total: number;
  limit: number;
  offset: number;
}

export interface RollbackRequest {
  versionId: string;
  reason: string;
  notes: string;
}

export interface RollbackResponse {
  success: boolean;
  message: string;
  rollbackId: string;
}

export interface VersionConfig {
  versionId: string;
  modelVersion: string;
  promptVersion: string;
  currentState: "shadow" | "grayscale" | "stable" | "quarantine";
  createdAt: string;
  updatedAt: string;
}

export interface VersionMetrics {
  p95LatencyMs: number;
  errorRate: number;
  accuracy: number;
  securityPassRate: number;
  costPerRequest: number;
  requestCount: number;
}

export interface EvaluationStatus {
  evaluationId: string;
  status: "pending" | "running" | "completed" | "failed";
  overallPassRate: number;
  promotionRecommended: boolean;
  recommendationReason: string;
}

export interface ComparisonStats {
  totalRequests: number;
  withV1Output: number;
  withV3Output: number;
  languages: Record<string, number>;
  comparisonMetrics: {
    samples: number;
    avgIssueCountDelta: number;
    avgLatencyDeltaMs: number;
  };
}

export interface RollbackStats {
  totalRollbacks: number;
  last7Days: number;
  byReason: Record<string, number>;
}

// API Client
class LifecycleApiClient {
  private readonly baseUrl = "/api/admin/lifecycle";

  // ============================================================
  // Comparison Requests
  // ============================================================

  /**
   * Get list of comparison requests
   */
  async getComparisonRequests(params?: {
    limit?: number;
    offset?: number;
    language?: string;
    hasV1?: boolean;
    hasV3?: boolean;
  }): Promise<ComparisonListResponse> {
    const response = await api.get<ComparisonListResponse>(
      `${this.baseUrl}/comparison-requests`,
      { params }
    );
    return response.data;
  }

  /**
   * Get a specific comparison request
   */
  async getComparisonRequest(requestId: string): Promise<ComparisonRequest> {
    const response = await api.get<ComparisonRequest>(
      `${this.baseUrl}/comparison-requests/${requestId}`
    );
    return response.data;
  }

  // ============================================================
  // Rollback Operations
  // ============================================================

  /**
   * Initiate a rollback
   */
  async initiateRollback(request: RollbackRequest): Promise<RollbackResponse> {
    const response = await api.post<RollbackResponse>(
      `${this.baseUrl}/rollback`,
      request
    );
    return response.data;
  }

  /**
   * Get rollback history
   */
  async getRollbackHistory(params?: {
    limit?: number;
    versionId?: string;
  }): Promise<
    Array<{
      rollbackId: string;
      versionId: string;
      reason: string;
      notes: string;
      timestamp: string;
      status: string;
    }>
  > {
    const response = await api.get(`${this.baseUrl}/rollback/history`, {
      params,
    });
    return response.data;
  }

  // ============================================================
  // Version Management
  // ============================================================

  /**
   * List all active versions
   */
  async listVersions(): Promise<{ versions: VersionConfig[] }> {
    const response = await api.get(`${this.baseUrl}/versions`);
    return response.data;
  }

  /**
   * Get metrics for a specific version
   */
  async getVersionMetrics(versionId: string): Promise<VersionMetrics> {
    const response = await api.get(
      `${this.baseUrl}/versions/${versionId}/metrics`
    );
    return response.data;
  }

  /**
   * Trigger evaluation for a version
   */
  async triggerEvaluation(versionId: string): Promise<{
    versionId: string;
    metrics: VersionMetrics;
    decision: {
      approved: boolean;
      reason: string;
    };
  }> {
    const response = await api.post(
      `${this.baseUrl}/versions/${versionId}/evaluate`
    );
    return response.data;
  }

  /**
   * Register a new version
   */
  async registerVersion(
    config: Omit<VersionConfig, "createdAt" | "updatedAt">
  ): Promise<{
    status: string;
    versionId: string;
  }> {
    const response = await api.post(
      `${this.baseUrl}/versions/${config.versionId}/register`,
      config
    );
    return response.data;
  }

  // ============================================================
  // Evaluation Pipeline
  // ============================================================

  /**
   * Start gold-set evaluation
   */
  async startGoldSetEvaluation(params: {
    versionId: string;
    modelVersion?: string;
    promptVersion?: string;
    testSets?: string[];
  }): Promise<{
    status: string;
    message: string;
    checkStatusAt: string;
  }> {
    const response = await api.post(
      "/api/admin/evaluation/evaluate/gold-set",
      params
    );
    return response.data;
  }

  /**
   * Get evaluation status
   */
  async getEvaluationStatus(versionId: string): Promise<EvaluationStatus> {
    const response = await api.get(
      `/api/admin/evaluation/evaluate/status/${versionId}`
    );
    return response.data;
  }

  /**
   * Get full evaluation results
   */
  async getEvaluationResults(versionId: string): Promise<{
    evaluationId: string;
    versionId: string;
    overallPassRate: number;
    totalTests: number;
    passedTests: number;
    failedTests: number;
    categoryResults: Record<
      string,
      {
        passRate: number;
        meetsRequirement: boolean;
      }
    >;
    avgPrecision: number;
    avgRecall: number;
    avgF1: number;
    avgLatencyMs: number;
    totalCostUsd: number;
    promotionRecommended: boolean;
    recommendationReason: string;
  }> {
    const response = await api.get(
      `/api/admin/evaluation/results/${versionId}`
    );
    return response.data;
  }

  /**
   * List available gold-sets
   */
  async listGoldSets(): Promise<{
    goldSets: Array<{
      id: string;
      name: string;
      category: string;
      testCount: number;
      requiredPassRate: number;
    }>;
  }> {
    const response = await api.get("/api/admin/evaluation/gold-sets");
    return response.data;
  }

  // ============================================================
  // Statistics
  // ============================================================

  /**
   * Get comparison statistics
   */
  async getComparisonStats(): Promise<ComparisonStats> {
    const response = await api.get(`${this.baseUrl}/stats/comparison`);
    return response.data;
  }

  /**
   * Get rollback statistics
   */
  async getRollbackStats(): Promise<RollbackStats> {
    const response = await api.get(`${this.baseUrl}/stats/rollbacks`);
    return response.data;
  }

  // ============================================================
  // Audit
  // ============================================================

  /**
   * Get lifecycle event history
   */
  async getEventHistory(limit: number = 100): Promise<{
    events: Array<{
      timestamp: string;
      versionId: string;
      eventType: string;
      currentState: string;
      details: Record<string, unknown>;
    }>;
  }> {
    const response = await api.get(`${this.baseUrl}/history`, {
      params: { limit },
    });
    return response.data;
  }

  // ============================================================
  // Health
  // ============================================================

  /**
   * Check lifecycle controller health
   */
  async checkHealth(): Promise<{ status: string }> {
    const response = await api.get(`${this.baseUrl}/health`);
    return response.data;
  }
}

// Export singleton instance
export const lifecycleApi = new LifecycleApiClient();

// Export class for testing
export { LifecycleApiClient };
