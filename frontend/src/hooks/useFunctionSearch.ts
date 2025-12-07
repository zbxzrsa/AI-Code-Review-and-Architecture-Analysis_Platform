/**
 * Function Search Hook
 *
 * Provides optimized search across all system functions
 * with performance tracking and search efficiency metrics.
 */

import { useState, useMemo, useCallback, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { usePermissions } from "./usePermissions";

// =============================================================================
// Types
// =============================================================================

export interface SearchableFunction {
  id: string;
  name: string;
  description: string;
  path: string;
  category: string;
  version: "V1" | "V2" | "V3" | "Admin";
  keywords: string[];
  isAdminOnly: boolean;
  popularity?: number; // 0-100
}

export interface SearchResult extends SearchableFunction {
  score: number;
  matchedOn: ("name" | "description" | "keywords")[];
}

export interface SearchMetrics {
  totalFunctions: number;
  searchTime: number;
  resultsCount: number;
  searchEfficiency: number; // percentage
}

// =============================================================================
// Function Registry
// =============================================================================

const ALL_FUNCTIONS: SearchableFunction[] = [
  // V2 Production
  {
    id: "code-review",
    name: "Code Review",
    description: "AI-powered code review and analysis",
    path: "/review",
    category: "development",
    version: "V2",
    keywords: ["code", "review", "analysis", "ai", "quality", "lint"],
    isAdminOnly: false,
    popularity: 95,
  },
  {
    id: "projects",
    name: "Projects",
    description: "Project management and organization",
    path: "/projects",
    category: "development",
    version: "V2",
    keywords: ["project", "manage", "organize", "workspace", "folder"],
    isAdminOnly: false,
    popularity: 90,
  },
  {
    id: "repositories",
    name: "Repositories",
    description: "Repository management and integration",
    path: "/repositories",
    category: "development",
    version: "V2",
    keywords: ["repository", "git", "source", "code", "github", "gitlab"],
    isAdminOnly: false,
    popularity: 85,
  },
  {
    id: "pull-requests",
    name: "Pull Requests",
    description: "Pull request review and management",
    path: "/pull-requests",
    category: "development",
    version: "V2",
    keywords: ["pull request", "pr", "merge", "review", "branch"],
    isAdminOnly: false,
    popularity: 80,
  },
  {
    id: "analytics",
    name: "Analytics",
    description: "Code quality and performance analytics",
    path: "/analytics",
    category: "insights",
    version: "V2",
    keywords: ["analytics", "metrics", "statistics", "trends", "chart", "graph"],
    isAdminOnly: false,
    popularity: 75,
  },
  {
    id: "security-dashboard",
    name: "Security Dashboard",
    description: "Security vulnerabilities and compliance",
    path: "/security",
    category: "security",
    version: "V2",
    keywords: ["security", "vulnerability", "scan", "compliance", "threat", "cve"],
    isAdminOnly: false,
    popularity: 70,
  },
  {
    id: "reports",
    name: "Reports",
    description: "Generate and view code analysis reports",
    path: "/reports",
    category: "insights",
    version: "V2",
    keywords: ["report", "export", "document", "summary", "pdf"],
    isAdminOnly: false,
    popularity: 65,
  },
  {
    id: "code-metrics",
    name: "Code Metrics",
    description: "Detailed code quality metrics",
    path: "/metrics",
    category: "insights",
    version: "V2",
    keywords: ["metrics", "quality", "complexity", "coverage", "loc", "cyclomatic"],
    isAdminOnly: false,
    popularity: 60,
  },
  {
    id: "quality-rules",
    name: "Quality Rules",
    description: "Configure code quality rules",
    path: "/rules",
    category: "configuration",
    version: "V2",
    keywords: ["rules", "quality", "lint", "standards", "eslint", "configure"],
    isAdminOnly: false,
    popularity: 55,
  },
  {
    id: "teams",
    name: "Team Management",
    description: "Manage teams and collaboration",
    path: "/teams",
    category: "collaboration",
    version: "V2",
    keywords: ["team", "collaborate", "members", "permissions", "group"],
    isAdminOnly: false,
    popularity: 50,
  },
  {
    id: "settings",
    name: "Settings",
    description: "User and application settings",
    path: "/settings",
    category: "configuration",
    version: "V2",
    keywords: ["settings", "preferences", "config", "options"],
    isAdminOnly: false,
    popularity: 45,
  },
  {
    id: "profile",
    name: "Profile",
    description: "User profile management",
    path: "/profile",
    category: "configuration",
    version: "V2",
    keywords: ["profile", "account", "user", "avatar"],
    isAdminOnly: false,
    popularity: 40,
  },
  {
    id: "notifications",
    name: "Notifications",
    description: "Notification settings and history",
    path: "/notifications",
    category: "configuration",
    version: "V2",
    keywords: ["notification", "alert", "message", "inbox"],
    isAdminOnly: false,
    popularity: 35,
  },

  // V1 Experimental (Admin only)
  {
    id: "experiments",
    name: "Experiment Management",
    description: "Manage AI model experiments",
    path: "/admin/experiments",
    category: "ai-experiments",
    version: "V1",
    keywords: ["experiment", "test", "ai", "model", "trial"],
    isAdminOnly: true,
    popularity: 30,
  },
  {
    id: "ai-testing",
    name: "AI Model Testing",
    description: "Test AI models with sample data",
    path: "/admin/model-testing",
    category: "ai-experiments",
    version: "V1",
    keywords: ["ai", "test", "model", "validation", "benchmark"],
    isAdminOnly: true,
    popularity: 28,
  },
  {
    id: "model-comparison",
    name: "Model Comparison",
    description: "Compare AI model performance",
    path: "/admin/model-comparison",
    category: "ai-experiments",
    version: "V1",
    keywords: ["compare", "model", "performance", "benchmark", "a/b"],
    isAdminOnly: true,
    popularity: 25,
  },
  {
    id: "learning-cycle",
    name: "Learning Cycle",
    description: "Continuous learning dashboard",
    path: "/admin/learning",
    category: "ai-experiments",
    version: "V1",
    keywords: ["learning", "training", "cycle", "evolution", "ml"],
    isAdminOnly: true,
    popularity: 22,
  },
  {
    id: "evolution-cycle",
    name: "Evolution Cycle",
    description: "AI self-evolution monitoring",
    path: "/admin/evolution",
    category: "ai-experiments",
    version: "V1",
    keywords: ["evolution", "self", "improve", "cycle", "autonomous"],
    isAdminOnly: true,
    popularity: 20,
  },

  // V3 Legacy (Admin only)
  {
    id: "version-comparison",
    name: "Version Comparison",
    description: "Compare different AI versions",
    path: "/admin/version-comparison",
    category: "legacy",
    version: "V3",
    keywords: ["version", "compare", "legacy", "baseline", "diff"],
    isAdminOnly: true,
    popularity: 15,
  },
  {
    id: "three-version",
    name: "Three Version Control",
    description: "Manage V1/V2/V3 lifecycle",
    path: "/admin/three-version",
    category: "legacy",
    version: "V3",
    keywords: ["version", "lifecycle", "promote", "demote", "v1", "v2", "v3"],
    isAdminOnly: true,
    popularity: 18,
  },

  // Admin Functions
  {
    id: "user-management",
    name: "User Management",
    description: "Manage users and permissions",
    path: "/admin/users",
    category: "administration",
    version: "Admin",
    keywords: ["user", "manage", "permission", "role", "admin"],
    isAdminOnly: true,
    popularity: 40,
  },
  {
    id: "provider-management",
    name: "AI Providers",
    description: "Configure AI providers and API keys",
    path: "/admin/providers",
    category: "administration",
    version: "Admin",
    keywords: ["provider", "api", "openai", "anthropic", "key"],
    isAdminOnly: true,
    popularity: 35,
  },
  {
    id: "ai-models",
    name: "AI Models",
    description: "Manage AI models configuration",
    path: "/admin/ai-models",
    category: "administration",
    version: "Admin",
    keywords: ["ai", "model", "config", "settings", "gpt", "claude"],
    isAdminOnly: true,
    popularity: 30,
  },
  {
    id: "audit-logs",
    name: "Audit Logs",
    description: "View system audit logs",
    path: "/admin/audit",
    category: "administration",
    version: "Admin",
    keywords: ["audit", "log", "history", "track", "activity"],
    isAdminOnly: true,
    popularity: 25,
  },
  {
    id: "system-health",
    name: "System Health",
    description: "Monitor system health and performance",
    path: "/admin/health",
    category: "administration",
    version: "Admin",
    keywords: ["health", "monitor", "status", "performance", "uptime"],
    isAdminOnly: true,
    popularity: 45,
  },
  {
    id: "auto-fix",
    name: "Auto-Fix",
    description: "Automated code fix management",
    path: "/admin/auto-fix",
    category: "administration",
    version: "Admin",
    keywords: ["auto", "fix", "repair", "automated", "patch"],
    isAdminOnly: true,
    popularity: 20,
  },
  {
    id: "security-scanner",
    name: "Security Scanner",
    description: "Advanced security scanning",
    path: "/admin/security",
    category: "administration",
    version: "Admin",
    keywords: ["security", "scan", "vulnerability", "threat", "sast"],
    isAdminOnly: true,
    popularity: 28,
  },
  {
    id: "performance-monitor",
    name: "Performance Monitor",
    description: "System performance monitoring",
    path: "/admin/performance",
    category: "administration",
    version: "Admin",
    keywords: ["performance", "monitor", "latency", "throughput", "cpu"],
    isAdminOnly: true,
    popularity: 22,
  },

  // Unified Hubs
  {
    id: "unified-hub",
    name: "Unified Dashboard",
    description: "All functions organized by version",
    path: "/hub",
    category: "navigation",
    version: "V2",
    keywords: ["hub", "unified", "all", "dashboard", "central"],
    isAdminOnly: false,
    popularity: 100,
  },
  {
    id: "v2-hub",
    name: "V2 Production Hub",
    description: "Production functions for all users",
    path: "/hub/v2",
    category: "navigation",
    version: "V2",
    keywords: ["v2", "production", "hub", "stable"],
    isAdminOnly: false,
    popularity: 98,
  },
  {
    id: "v1-hub",
    name: "V1 Experimental Hub",
    description: "Experimental functions for admins",
    path: "/hub/v1",
    category: "navigation",
    version: "V1",
    keywords: ["v1", "experimental", "hub", "admin"],
    isAdminOnly: true,
    popularity: 50,
  },
  {
    id: "v3-hub",
    name: "V3 Legacy Hub",
    description: "Legacy and comparison functions",
    path: "/hub/v3",
    category: "navigation",
    version: "V3",
    keywords: ["v3", "legacy", "hub", "comparison"],
    isAdminOnly: true,
    popularity: 30,
  },
];

// =============================================================================
// Search Algorithm
// =============================================================================

function calculateScore(
  func: SearchableFunction,
  query: string
): { score: number; matchedOn: ("name" | "description" | "keywords")[] } {
  const lowerQuery = query.toLowerCase();
  const words = lowerQuery.split(/\s+/).filter((w) => w.length > 0);
  let score = 0;
  const matchedOn: ("name" | "description" | "keywords")[] = [];

  // Name matching (highest weight)
  const lowerName = func.name.toLowerCase();
  if (lowerName === lowerQuery) {
    score += 100;
    matchedOn.push("name");
  } else if (lowerName.startsWith(lowerQuery)) {
    score += 80;
    matchedOn.push("name");
  } else if (lowerName.includes(lowerQuery)) {
    score += 60;
    matchedOn.push("name");
  } else if (words.some((w) => lowerName.includes(w))) {
    score += 40;
    matchedOn.push("name");
  }

  // Description matching (medium weight)
  const lowerDesc = func.description.toLowerCase();
  if (words.some((w) => lowerDesc.includes(w))) {
    score += 20;
    if (!matchedOn.includes("description")) matchedOn.push("description");
  }

  // Keyword matching (high weight for exact matches)
  const keywordMatches = func.keywords.filter((k) =>
    words.some((w) => k.toLowerCase().includes(w) || w.includes(k.toLowerCase()))
  );
  if (keywordMatches.length > 0) {
    score += keywordMatches.length * 15;
    if (!matchedOn.includes("keywords")) matchedOn.push("keywords");
  }

  // Boost by popularity
  score += (func.popularity || 0) * 0.1;

  return { score, matchedOn };
}

// =============================================================================
// Hook
// =============================================================================

export interface UseFunctionSearchOptions {
  debounceMs?: number;
  maxResults?: number;
  includeAdminFunctions?: boolean;
}

export interface UseFunctionSearchReturn {
  query: string;
  setQuery: (query: string) => void;
  results: SearchResult[];
  metrics: SearchMetrics;
  isSearching: boolean;
  clearSearch: () => void;
  navigateToResult: (result: SearchResult) => void;
  recentSearches: string[];
  addToRecent: (query: string) => void;
}

export function useFunctionSearch(options: UseFunctionSearchOptions = {}): UseFunctionSearchReturn {
  const { debounceMs = 150, maxResults = 10 } = options;

  const navigate = useNavigate();
  const { isAdmin } = usePermissions();

  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [metrics, setMetrics] = useState<SearchMetrics>({
    totalFunctions: 0,
    searchTime: 0,
    resultsCount: 0,
    searchEfficiency: 100,
  });
  const [isSearching, setIsSearching] = useState(false);
  const [recentSearches, setRecentSearches] = useState<string[]>(() => {
    try {
      const saved = localStorage.getItem("recentFunctionSearches");
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  const debounceRef = useRef<NodeJS.Timeout>();

  // Filter functions by role
  const availableFunctions = useMemo(() => {
    return ALL_FUNCTIONS.filter((f) => !f.isAdminOnly || isAdmin);
  }, [isAdmin]);

  // Perform search
  useEffect(() => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    if (!query.trim()) {
      setResults([]);
      setMetrics({
        totalFunctions: availableFunctions.length,
        searchTime: 0,
        resultsCount: 0,
        searchEfficiency: 100,
      });
      setIsSearching(false);
      return;
    }

    setIsSearching(true);

    debounceRef.current = setTimeout(() => {
      const startTime = performance.now();

      // Calculate scores for all functions
      const scored = availableFunctions
        .map((func) => {
          const { score, matchedOn } = calculateScore(func, query);
          return { ...func, score, matchedOn };
        })
        .filter((f) => f.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, maxResults);

      const endTime = performance.now();
      const searchTime = endTime - startTime;

      // Calculate search efficiency (how quickly we narrowed down results)
      const efficiency = Math.min(
        100,
        Math.round(
          100 - (scored.length / availableFunctions.length) * 100 + (searchTime < 50 ? 30 : 0)
        )
      );

      setResults(scored);
      setMetrics({
        totalFunctions: availableFunctions.length,
        searchTime,
        resultsCount: scored.length,
        searchEfficiency: efficiency,
      });
      setIsSearching(false);
    }, debounceMs);

    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, [query, availableFunctions, debounceMs, maxResults]);

  // Clear search
  const clearSearch = useCallback(() => {
    setQuery("");
    setResults([]);
  }, []);

  // Navigate to result
  const navigateToResult = useCallback(
    (result: SearchResult) => {
      navigate(result.path);
      addToRecent(query);
    },
    [navigate, query]
  );

  // Add to recent searches
  const addToRecent = useCallback((searchQuery: string) => {
    if (!searchQuery.trim()) return;

    setRecentSearches((prev) => {
      const updated = [searchQuery, ...prev.filter((s) => s !== searchQuery)].slice(0, 5);
      try {
        localStorage.setItem("recentFunctionSearches", JSON.stringify(updated));
      } catch {
        // Ignore storage errors
      }
      return updated;
    });
  }, []);

  return {
    query,
    setQuery,
    results,
    metrics,
    isSearching,
    clearSearch,
    navigateToResult,
    recentSearches,
    addToRecent,
  };
}

export default useFunctionSearch;
