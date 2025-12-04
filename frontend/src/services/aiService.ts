/**
 * AI Service
 * Connects frontend to the three-version AI backend services
 */

import api from "./api";

// Types
export interface AIVersion {
  version: "v1" | "v2" | "v3";
  status: "online" | "offline" | "degraded";
  model: string;
  latency: number;
  accuracy: number;
  lastUpdated: string;
}

export interface AIMessage {
  role: "user" | "assistant";
  content: string;
  version?: "v1" | "v2" | "v3";
  model?: string;
  latency?: number;
  tokens?: number;
}

export interface AIAnalysisRequest {
  code: string;
  language: string;
  reviewTypes: ("security" | "performance" | "quality" | "bug")[];
  version?: "v1" | "v2" | "v3";
}

export interface AIIssue {
  id: string;
  type: "security" | "performance" | "quality" | "bug";
  severity: "critical" | "high" | "medium" | "low";
  title: string;
  description: string;
  line: number;
  suggestion?: string;
  fixAvailable: boolean;
}

export interface AIAnalysisResult {
  id: string;
  issues: AIIssue[];
  score: number;
  summary: string;
  model: string;
  latency: number;
  version: "v1" | "v2" | "v3";
}

export interface Technology {
  id: string;
  name: string;
  version: "v1" | "v2" | "v3";
  status: "active" | "testing" | "deprecated" | "quarantined";
  accuracy: number;
  errorRate: number;
  latency: number;
  samples: number;
  lastUpdated: string;
}

export interface EvolutionCycleStatus {
  running: boolean;
  currentPhase: string;
  cycleId: string;
  metrics: {
    experimentsRun: number;
    errorsFixed: number;
    promotionsMade: number;
    degradationsMade: number;
  };
  versions: {
    v1: AIVersion;
    v2: AIVersion;
    v3: AIVersion;
  };
}

// AI Service
export const aiService = {
  // Get all AI version statuses
  async getVersionStatuses(): Promise<AIVersion[]> {
    const response = await api.get("/api/v1/evolution/status");
    const status = response.data;
    return [
      { version: "v1", ...status.versions?.v1 },
      { version: "v2", ...status.versions?.v2 },
      { version: "v3", ...status.versions?.v3 },
    ];
  },

  // Get evolution cycle status
  async getCycleStatus(): Promise<EvolutionCycleStatus> {
    const response = await api.get("/api/v1/evolution/status");
    return response.data;
  },

  // Start evolution cycle
  async startCycle(): Promise<void> {
    await api.post("/api/v1/evolution/start");
  },

  // Stop evolution cycle
  async stopCycle(): Promise<void> {
    await api.post("/api/v1/evolution/stop");
  },

  // Chat with AI
  async chat(
    message: string,
    version: "v1" | "v2" | "v3" = "v2",
    context?: { code?: string; language?: string }
  ): Promise<AIMessage> {
    const endpoint =
      version === "v1"
        ? "/api/v1/ai/chat"
        : version === "v3"
        ? "/api/v3/ai/chat"
        : "/api/v2/ai/chat";

    const response = await api.post(endpoint, {
      message,
      context,
    });

    return {
      role: "assistant",
      content: response.data.response,
      version,
      model: response.data.model,
      latency: response.data.latency,
      tokens: response.data.tokens,
    };
  },

  // Compare responses from all versions
  async compareVersions(
    message: string,
    context?: { code?: string; language?: string }
  ): Promise<{ v1: AIMessage; v2: AIMessage; v3: AIMessage }> {
    const [v1, v2, v3] = await Promise.all([
      this.chat(message, "v1", context),
      this.chat(message, "v2", context),
      this.chat(message, "v3", context),
    ]);
    return { v1, v2, v3 };
  },

  // Analyze code
  async analyzeCode(request: AIAnalysisRequest): Promise<AIAnalysisResult> {
    const version = request.version || "v2";
    const endpoint =
      version === "v1"
        ? "/api/v1/ai/analyze"
        : version === "v3"
        ? "/api/v3/ai/analyze"
        : "/api/v2/ai/analyze";

    const response = await api.post(endpoint, {
      code: request.code,
      language: request.language,
      review_types: request.reviewTypes,
    });

    return {
      id: response.data.id,
      issues: response.data.issues,
      score: response.data.score,
      summary: response.data.summary,
      model: response.data.model,
      latency: response.data.latency,
      version,
    };
  },

  // Apply auto-fix
  async applyFix(issueId: string, code: string): Promise<string> {
    const response = await api.post("/api/v2/ai/fix", {
      issue_id: issueId,
      code,
    });
    return response.data.fixed_code;
  },

  // Get technologies
  async getTechnologies(): Promise<Technology[]> {
    const response = await api.get("/api/v1/evolution/technologies");
    return response.data;
  },

  // Promote technology
  async promoteTechnology(techId: string, reason?: string): Promise<void> {
    await api.post("/api/v1/evolution/promote", {
      tech_id: techId,
      reason,
    });
  },

  // Degrade technology
  async degradeTechnology(techId: string, reason: string): Promise<void> {
    await api.post("/api/v1/evolution/degrade", {
      tech_id: techId,
      reason,
    });
  },

  // Request re-evaluation
  async requestReEvaluation(techId: string, reason?: string): Promise<void> {
    await api.post("/api/v1/evolution/reeval", {
      tech_id: techId,
      reason,
    });
  },

  // Report V1 error
  async reportV1Error(
    techId: string,
    techName: string,
    errorType: string,
    description: string
  ): Promise<void> {
    await api.post("/api/v1/evolution/v1/errors", {
      tech_id: techId,
      tech_name: techName,
      error_type: errorType,
      description,
    });
  },

  // Provide feedback
  async provideFeedback(
    responseId: string,
    helpful: boolean,
    comment?: string
  ): Promise<void> {
    await api.post("/api/v2/ai/feedback", {
      response_id: responseId,
      helpful,
      comment,
    });
  },
};

export default aiService;
