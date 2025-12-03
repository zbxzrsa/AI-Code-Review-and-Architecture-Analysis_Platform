/**
 * Enhanced Code Analysis Hook
 * 
 * Improved code analysis with:
 * - Streaming analysis support
 * - Incremental analysis
 * - Result caching
 * - Priority queue
 * - Batch analysis
 * - Progress tracking
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { enhancedApi, ApiError } from '../services/enhancedApi';
import { notificationManager } from '../services/notificationManager';

// ============================================
// Types
// ============================================

export type AnalysisSeverity = 'critical' | 'high' | 'medium' | 'low' | 'info';
export type AnalysisCategory = 'security' | 'performance' | 'quality' | 'style' | 'bug';

export interface AnalysisIssue {
  id: string;
  title: string;
  description: string;
  severity: AnalysisSeverity;
  category: AnalysisCategory;
  file: string;
  line: number;
  column?: number;
  endLine?: number;
  endColumn?: number;
  rule?: string;
  suggestion?: string;
  autoFixable: boolean;
  confidence: number;
}

export interface AnalysisResult {
  id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  issues: AnalysisIssue[];
  summary: {
    total: number;
    critical: number;
    high: number;
    medium: number;
    low: number;
    autoFixable: number;
  };
  metrics: {
    linesAnalyzed: number;
    filesAnalyzed: number;
    duration: number;
    coverage?: number;
  };
  timestamp: Date;
}

export interface AnalysisOptions {
  language?: string;
  rules?: string[];
  severity?: AnalysisSeverity[];
  categories?: AnalysisCategory[];
  maxIssues?: number;
  streaming?: boolean;
  incremental?: boolean;
  priority?: 'low' | 'normal' | 'high';
}

interface AnalysisState {
  isAnalyzing: boolean;
  progress: number;
  currentFile?: string;
  result: AnalysisResult | null;
  error: ApiError | null;
  queue: QueuedAnalysis[];
}

interface QueuedAnalysis {
  id: string;
  code: string;
  options: AnalysisOptions;
  priority: number;
}

// ============================================
// Analysis Cache
// ============================================

class AnalysisCache {
  private cache: Map<string, { result: AnalysisResult; expiry: number }> = new Map();
  private maxSize: number = 100;
  private ttl: number = 5 * 60 * 1000; // 5 minutes

  private generateKey(code: string, options: AnalysisOptions): string {
    const hash = this.hashCode(code);
    return `${hash}-${JSON.stringify(options)}`;
  }

  private hashCode(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return hash.toString(36);
  }

  public get(code: string, options: AnalysisOptions): AnalysisResult | null {
    const key = this.generateKey(code, options);
    const entry = this.cache.get(key);
    
    if (!entry) return null;
    
    if (Date.now() > entry.expiry) {
      this.cache.delete(key);
      return null;
    }
    
    return entry.result;
  }

  public set(code: string, options: AnalysisOptions, result: AnalysisResult): void {
    const key = this.generateKey(code, options);
    
    // Evict oldest if at capacity
    if (this.cache.size >= this.maxSize) {
      const oldest = this.cache.keys().next().value;
      if (oldest) this.cache.delete(oldest);
    }
    
    this.cache.set(key, {
      result,
      expiry: Date.now() + this.ttl,
    });
  }

  public clear(): void {
    this.cache.clear();
  }
}

const analysisCache = new AnalysisCache();

// ============================================
// Hook Implementation
// ============================================

export function useCodeAnalysis() {
  const [state, setState] = useState<AnalysisState>({
    isAnalyzing: false,
    progress: 0,
    result: null,
    error: null,
    queue: [],
  });

  const abortControllerRef = useRef<AbortController | null>(null);
  const streamReaderRef = useRef<ReadableStreamDefaultReader | null>(null);

  // ============================================
  // Cleanup
  // ============================================

  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      if (streamReaderRef.current) {
        streamReaderRef.current.cancel();
      }
    };
  }, []);

  // ============================================
  // Analysis Methods
  // ============================================

  const analyzeCode = useCallback(async (
    code: string,
    options: AnalysisOptions = {}
  ): Promise<AnalysisResult | null> => {
    // Check cache first
    const cached = analysisCache.get(code, options);
    if (cached && !options.incremental) {
      setState(prev => ({ ...prev, result: cached }));
      return cached;
    }

    // Cancel any existing analysis
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    setState(prev => ({
      ...prev,
      isAnalyzing: true,
      progress: 0,
      error: null,
    }));

    try {
      if (options.streaming) {
        return await analyzeWithStreaming(code, options);
      } else {
        return await analyzeStandard(code, options);
      }
    } catch (error) {
      const apiError = error as ApiError;
      setState(prev => ({
        ...prev,
        isAnalyzing: false,
        error: apiError,
      }));
      notificationManager.handleApiError(apiError);
      return null;
    }
  }, []);

  const analyzeStandard = async (
    code: string,
    options: AnalysisOptions
  ): Promise<AnalysisResult> => {
    const response = await enhancedApi.post<AnalysisResult>('/analysis/code', {
      code,
      ...options,
    }, {
      signal: abortControllerRef.current?.signal,
    });

    const result: AnalysisResult = {
      ...response,
      timestamp: new Date(),
    };

    // Cache the result
    analysisCache.set(code, options, result);

    setState(prev => ({
      ...prev,
      isAnalyzing: false,
      progress: 100,
      result,
    }));

    return result;
  };

  const analyzeWithStreaming = async (
    code: string,
    options: AnalysisOptions
  ): Promise<AnalysisResult> => {
    const response = await fetch('/api/analysis/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code, ...options }),
      signal: abortControllerRef.current?.signal,
    });

    if (!response.body) {
      throw new Error('Streaming not supported');
    }

    const reader = response.body.getReader();
    streamReaderRef.current = reader;
    const decoder = new TextDecoder();

    const issues: AnalysisIssue[] = [];
    let totalLines = code.split('\n').length;
    let analyzedLines = 0;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n').filter(Boolean);

        for (const line of lines) {
          try {
            const data = JSON.parse(line);

            if (data.type === 'progress') {
              analyzedLines = data.linesAnalyzed;
              setState(prev => ({
                ...prev,
                progress: Math.round((analyzedLines / totalLines) * 100),
                currentFile: data.currentFile,
              }));
            } else if (data.type === 'issue') {
              issues.push(data.issue);
            }
          } catch (e) {
            // Invalid JSON, skip
          }
        }
      }
    } finally {
      streamReaderRef.current = null;
    }

    const result: AnalysisResult = {
      id: `analysis-${Date.now()}`,
      status: 'completed',
      issues,
      summary: summarizeIssues(issues),
      metrics: {
        linesAnalyzed: totalLines,
        filesAnalyzed: 1,
        duration: 0,
      },
      timestamp: new Date(),
    };

    analysisCache.set(code, options, result);

    setState(prev => ({
      ...prev,
      isAnalyzing: false,
      progress: 100,
      result,
    }));

    return result;
  };

  // ============================================
  // Batch Analysis
  // ============================================

  const analyzeBatch = useCallback(async (
    files: { path: string; code: string }[],
    options: AnalysisOptions = {}
  ): Promise<AnalysisResult[]> => {
    const results: AnalysisResult[] = [];
    const total = files.length;

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      setState(prev => ({
        ...prev,
        progress: Math.round((i / total) * 100),
        currentFile: file.path,
      }));

      const result = await analyzeCode(file.code, {
        ...options,
        streaming: false,
      });

      if (result) {
        results.push(result);
      }
    }

    setState(prev => ({
      ...prev,
      isAnalyzing: false,
      progress: 100,
      currentFile: undefined,
    }));

    return results;
  }, [analyzeCode]);

  // ============================================
  // Auto-Fix
  // ============================================

  const applyFix = useCallback(async (
    issue: AnalysisIssue,
    code: string
  ): Promise<string | null> => {
    if (!issue.autoFixable) {
      notificationManager.warning('Cannot Auto-Fix', 'This issue requires manual intervention');
      return null;
    }

    try {
      const response = await enhancedApi.post<{ fixedCode: string }>('/analysis/fix', {
        issue,
        code,
      });

      notificationManager.success('Fix Applied', `Fixed: ${issue.title}`);
      return response.fixedCode;
    } catch (error) {
      notificationManager.handleApiError(error);
      return null;
    }
  }, []);

  const applyAllFixes = useCallback(async (
    issues: AnalysisIssue[],
    code: string
  ): Promise<string | null> => {
    const fixableIssues = issues.filter(i => i.autoFixable);

    if (fixableIssues.length === 0) {
      notificationManager.info('No Fixes Available', 'No auto-fixable issues found');
      return null;
    }

    try {
      const response = await enhancedApi.post<{ fixedCode: string; fixedCount: number }>(
        '/analysis/fix-all',
        { issues: fixableIssues, code }
      );

      notificationManager.success(
        'Fixes Applied',
        `Applied ${response.fixedCount} fixes`
      );
      return response.fixedCode;
    } catch (error) {
      notificationManager.handleApiError(error);
      return null;
    }
  }, []);

  // ============================================
  // Control Methods
  // ============================================

  const cancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    if (streamReaderRef.current) {
      streamReaderRef.current.cancel();
    }
    setState(prev => ({
      ...prev,
      isAnalyzing: false,
      progress: 0,
    }));
  }, []);

  const clearResult = useCallback(() => {
    setState(prev => ({
      ...prev,
      result: null,
      error: null,
    }));
  }, []);

  const clearCache = useCallback(() => {
    analysisCache.clear();
  }, []);

  // ============================================
  // Return Value
  // ============================================

  return {
    // State
    isAnalyzing: state.isAnalyzing,
    progress: state.progress,
    currentFile: state.currentFile,
    result: state.result,
    error: state.error,

    // Methods
    analyzeCode,
    analyzeBatch,
    applyFix,
    applyAllFixes,
    cancel,
    clearResult,
    clearCache,
  };
}

// ============================================
// Helper Functions
// ============================================

function summarizeIssues(issues: AnalysisIssue[]): AnalysisResult['summary'] {
  return {
    total: issues.length,
    critical: issues.filter(i => i.severity === 'critical').length,
    high: issues.filter(i => i.severity === 'high').length,
    medium: issues.filter(i => i.severity === 'medium').length,
    low: issues.filter(i => i.severity === 'low').length,
    autoFixable: issues.filter(i => i.autoFixable).length,
  };
}

export default useCodeAnalysis;
