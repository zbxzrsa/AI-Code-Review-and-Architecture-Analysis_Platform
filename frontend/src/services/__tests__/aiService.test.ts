/**
 * AI Service Unit Tests
 *
 * TD-001: Comprehensive unit tests for AI service
 * Coverage Target: 80%+
 */

import { aiService } from "../aiService";
import { api } from "../api";

// Mock the api module
jest.mock("../api", () => ({
  api: {
    post: jest.fn(),
    get: jest.fn(),
  },
}));

const mockedApi = api as jest.Mocked<typeof api>;

describe("AI Service", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("analyzeCode", () => {
    it("should call POST /ai/analyze with code and options", async () => {
      const mockResponse = { data: { issues: [], metrics: {} } };
      mockedApi.post.mockResolvedValue(mockResponse);

      const result = await aiService.analyzeCode("const x = 1;", {
        language: "javascript",
        rules: ["security", "quality"],
      });

      expect(mockedApi.post).toHaveBeenCalledWith("/ai/analyze", {
        code: "const x = 1;",
        language: "javascript",
        rules: ["security", "quality"],
      });
      expect(result).toEqual(mockResponse.data);
    });

    it("should use default options when not provided", async () => {
      const mockResponse = { data: { issues: [] } };
      mockedApi.post.mockResolvedValue(mockResponse);

      await aiService.analyzeCode("const x = 1;");

      expect(mockedApi.post).toHaveBeenCalledWith("/ai/analyze", {
        code: "const x = 1;",
      });
    });
  });

  describe("chat", () => {
    it("should call POST /ai/chat with message and context", async () => {
      const mockResponse = { data: { response: "Hello!" } };
      mockedApi.post.mockResolvedValue(mockResponse);

      const result = await aiService.chat("Hello", {
        sessionId: "sess-123",
        context: { file: "test.ts" },
      });

      expect(mockedApi.post).toHaveBeenCalledWith("/ai/chat", {
        message: "Hello",
        session_id: "sess-123",
        context: { file: "test.ts" },
      });
      expect(result).toEqual(mockResponse.data);
    });
  });

  describe("suggestFix", () => {
    it("should call POST /ai/suggest-fix with issue details", async () => {
      const mockResponse = { data: { suggestion: "Use const instead of var" } };
      mockedApi.post.mockResolvedValue(mockResponse);

      const issue = {
        id: "issue-123",
        type: "quality",
        message: "Use const",
        code: "var x = 1;",
      };

      const result = await aiService.suggestFix(issue);

      expect(mockedApi.post).toHaveBeenCalledWith("/ai/suggest-fix", { issue });
      expect(result).toEqual(mockResponse.data);
    });
  });

  describe("explainCode", () => {
    it("should call POST /ai/explain with code snippet", async () => {
      const mockResponse = { data: { explanation: "This function..." } };
      mockedApi.post.mockResolvedValue(mockResponse);

      const result = await aiService.explainCode(
        "function add(a, b) { return a + b; }"
      );

      expect(mockedApi.post).toHaveBeenCalledWith("/ai/explain", {
        code: "function add(a, b) { return a + b; }",
      });
      expect(result).toEqual(mockResponse.data);
    });
  });

  describe("getTechnologies", () => {
    it("should call GET /ai/technologies", async () => {
      const mockResponse = { data: { technologies: ["React", "TypeScript"] } };
      mockedApi.get.mockResolvedValue(mockResponse);

      const result = await aiService.getTechnologies();

      expect(mockedApi.get).toHaveBeenCalledWith("/ai/technologies");
      expect(result).toEqual(mockResponse.data);
    });
  });

  describe("getModels", () => {
    it("should call GET /ai/models", async () => {
      const mockResponse = { data: { models: ["gpt-4", "claude-3"] } };
      mockedApi.get.mockResolvedValue(mockResponse);

      const result = await aiService.getModels();

      expect(mockedApi.get).toHaveBeenCalledWith("/ai/models");
      expect(result).toEqual(mockResponse.data);
    });
  });

  describe("getCycleStatus", () => {
    it("should call GET /ai/cycle/status", async () => {
      const mockResponse = {
        data: {
          v1: { status: "active" },
          v2: { status: "stable" },
          v3: { status: "quarantine" },
        },
      };
      mockedApi.get.mockResolvedValue(mockResponse);

      const result = await aiService.getCycleStatus();

      expect(mockedApi.get).toHaveBeenCalledWith("/ai/cycle/status");
      expect(result).toEqual(mockResponse.data);
    });
  });

  describe("error handling", () => {
    it("should propagate errors from API calls", async () => {
      const error = new Error("Network error");
      mockedApi.post.mockRejectedValue(error);

      await expect(aiService.analyzeCode("code")).rejects.toThrow(
        "Network error"
      );
    });
  });
});
