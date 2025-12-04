/**
 * AI Hooks
 * React Query hooks for AI services
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { message } from "antd";
import aiService, {
  AIVersion,
  AIMessage,
  AIAnalysisRequest,
  AIAnalysisResult,
  Technology,
  EvolutionCycleStatus,
} from "../services/aiService";

// Query Keys
export const aiQueryKeys = {
  all: ["ai"] as const,
  versions: () => [...aiQueryKeys.all, "versions"] as const,
  cycleStatus: () => [...aiQueryKeys.all, "cycle-status"] as const,
  technologies: () => [...aiQueryKeys.all, "technologies"] as const,
};

// Hooks

/**
 * Get AI version statuses
 */
export function useAIVersions() {
  return useQuery<AIVersion[], Error>({
    queryKey: aiQueryKeys.versions(),
    queryFn: aiService.getVersionStatuses,
    refetchInterval: 30000, // Refresh every 30 seconds
    staleTime: 10000,
  });
}

/**
 * Get evolution cycle status
 */
export function useCycleStatus() {
  return useQuery<EvolutionCycleStatus, Error>({
    queryKey: aiQueryKeys.cycleStatus(),
    queryFn: aiService.getCycleStatus,
    refetchInterval: 10000, // Refresh every 10 seconds
    staleTime: 5000,
  });
}

/**
 * Get technologies
 */
export function useTechnologies() {
  return useQuery<Technology[], Error>({
    queryKey: aiQueryKeys.technologies(),
    queryFn: aiService.getTechnologies,
    refetchInterval: 30000,
    staleTime: 10000,
  });
}

/**
 * Start evolution cycle
 */
export function useStartCycle() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: aiService.startCycle,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: aiQueryKeys.cycleStatus() });
      message.success("Evolution cycle started");
    },
    onError: (error: Error) => {
      message.error(`Failed to start cycle: ${error.message}`);
    },
  });
}

/**
 * Stop evolution cycle
 */
export function useStopCycle() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: aiService.stopCycle,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: aiQueryKeys.cycleStatus() });
      message.success("Evolution cycle stopped");
    },
    onError: (error: Error) => {
      message.error(`Failed to stop cycle: ${error.message}`);
    },
  });
}

/**
 * Chat with AI
 */
export function useAIChat() {
  return useMutation<
    AIMessage,
    Error,
    {
      message: string;
      version?: "v1" | "v2" | "v3";
      context?: { code?: string; language?: string };
    }
  >({
    mutationFn: ({ message, version, context }) =>
      aiService.chat(message, version, context),
    onError: (error) => {
      message.error(`AI request failed: ${error.message}`);
    },
  });
}

/**
 * Compare AI versions
 */
export function useAICompare() {
  return useMutation<
    { v1: AIMessage; v2: AIMessage; v3: AIMessage },
    Error,
    { message: string; context?: { code?: string; language?: string } }
  >({
    mutationFn: ({ message, context }) =>
      aiService.compareVersions(message, context),
    onError: (error) => {
      message.error(`Comparison failed: ${error.message}`);
    },
  });
}

/**
 * Analyze code with AI
 */
export function useCodeAnalysis() {
  return useMutation<AIAnalysisResult, Error, AIAnalysisRequest>({
    mutationFn: aiService.analyzeCode,
    onError: (error) => {
      message.error(`Analysis failed: ${error.message}`);
    },
  });
}

/**
 * Apply auto-fix
 */
export function useApplyFix() {
  return useMutation<string, Error, { issueId: string; code: string }>({
    mutationFn: ({ issueId, code }) => aiService.applyFix(issueId, code),
    onSuccess: () => {
      message.success("Fix applied successfully");
    },
    onError: (error) => {
      message.error(`Failed to apply fix: ${error.message}`);
    },
  });
}

/**
 * Promote technology
 */
export function usePromoteTechnology() {
  const queryClient = useQueryClient();

  return useMutation<void, Error, { techId: string; reason?: string }>({
    mutationFn: ({ techId, reason }) =>
      aiService.promoteTechnology(techId, reason),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: aiQueryKeys.technologies() });
      queryClient.invalidateQueries({ queryKey: aiQueryKeys.cycleStatus() });
      message.success("Technology promoted to V2");
    },
    onError: (error) => {
      message.error(`Promotion failed: ${error.message}`);
    },
  });
}

/**
 * Degrade technology
 */
export function useDegradeTechnology() {
  const queryClient = useQueryClient();

  return useMutation<void, Error, { techId: string; reason: string }>({
    mutationFn: ({ techId, reason }) =>
      aiService.degradeTechnology(techId, reason),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: aiQueryKeys.technologies() });
      queryClient.invalidateQueries({ queryKey: aiQueryKeys.cycleStatus() });
      message.success("Technology degraded to V3");
    },
    onError: (error) => {
      message.error(`Degradation failed: ${error.message}`);
    },
  });
}

/**
 * Request re-evaluation
 */
export function useRequestReEvaluation() {
  const queryClient = useQueryClient();

  return useMutation<void, Error, { techId: string; reason?: string }>({
    mutationFn: ({ techId, reason }) =>
      aiService.requestReEvaluation(techId, reason),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: aiQueryKeys.technologies() });
      message.success("Re-evaluation requested");
    },
    onError: (error) => {
      message.error(`Re-evaluation request failed: ${error.message}`);
    },
  });
}

/**
 * Report V1 error
 */
export function useReportV1Error() {
  return useMutation<
    void,
    Error,
    { techId: string; techName: string; errorType: string; description: string }
  >({
    mutationFn: ({ techId, techName, errorType, description }) =>
      aiService.reportV1Error(techId, techName, errorType, description),
    onSuccess: () => {
      message.success("Error reported, V2 will analyze and fix");
    },
    onError: (error) => {
      message.error(`Failed to report error: ${error.message}`);
    },
  });
}

/**
 * Provide feedback
 */
export function useProvideFeedback() {
  return useMutation<
    void,
    Error,
    { responseId: string; helpful: boolean; comment?: string }
  >({
    mutationFn: ({ responseId, helpful, comment }) =>
      aiService.provideFeedback(responseId, helpful, comment),
    onSuccess: () => {
      message.success("Thank you for your feedback!");
    },
    onError: (error) => {
      message.error(`Failed to submit feedback: ${error.message}`);
    },
  });
}
