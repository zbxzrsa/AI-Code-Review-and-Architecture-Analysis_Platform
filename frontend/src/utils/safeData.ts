/**
 * Safe Data Utilities
 *
 * Utility functions to safely handle API response data
 * that might have unexpected formats (null, undefined, object instead of array, etc.)
 */

/**
 * Ensure a value is an array
 * @param data - Any data that should be an array
 * @param fallback - Optional fallback array (defaults to empty array)
 * @returns A guaranteed array
 */
export function ensureArray<T>(data: unknown, fallback: T[] = []): T[] {
  if (Array.isArray(data)) {
    return data;
  }

  // Handle common API response formats
  if (data && typeof data === "object") {
    // Check for common wrapper properties
    const obj = data as Record<string, unknown>;
    if (Array.isArray(obj.items)) return obj.items as T[];
    if (Array.isArray(obj.data)) return obj.data as T[];
    if (Array.isArray(obj.results)) return obj.results as T[];
    if (Array.isArray(obj.list)) return obj.list as T[];
    if (Array.isArray(obj.records)) return obj.records as T[];
  }

  return fallback;
}

/**
 * Safely get a nested property from an object
 * @param obj - The object to get the property from
 * @param path - Dot-separated path to the property
 * @param fallback - Fallback value if property doesn't exist
 */
export function safeGet<T>(obj: unknown, path: string, fallback: T): T {
  if (!obj || typeof obj !== "object") return fallback;

  const keys = path.split(".");
  let current: unknown = obj;

  for (const key of keys) {
    if (current === null || current === undefined) return fallback;
    if (typeof current !== "object") return fallback;
    current = (current as Record<string, unknown>)[key];
  }

  return (current as T) ?? fallback;
}

/**
 * Ensure a value is a number
 * @param value - Any value that should be a number
 * @param fallback - Fallback number (defaults to 0)
 */
export function ensureNumber(value: unknown, fallback = 0): number {
  if (typeof value === "number" && !Number.isNaN(value)) return value;
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value);
    if (!Number.isNaN(parsed)) return parsed;
  }
  return fallback;
}

/**
 * Ensure a value is a string
 * @param value - Any value that should be a string
 * @param fallback - Fallback string (defaults to empty string)
 */
export function ensureString(value: unknown, fallback = ""): string {
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return fallback;
  return String(value);
}

/**
 * Ensure a value is a boolean
 * @param value - Any value that should be a boolean
 * @param fallback - Fallback boolean (defaults to false)
 */
export function ensureBoolean(value: unknown, fallback = false): boolean {
  if (typeof value === "boolean") return value;
  if (value === "true" || value === 1) return true;
  if (value === "false" || value === 0) return false;
  return fallback;
}

/**
 * Safely parse JSON
 * @param json - JSON string to parse
 * @param fallback - Fallback value if parsing fails
 */
export function safeJsonParse<T>(json: string, fallback: T): T {
  try {
    return JSON.parse(json) as T;
  } catch {
    return fallback;
  }
}

/**
 * Create a safe data wrapper for API responses
 * Handles common response formats and provides type-safe access
 */
export interface SafeApiResponse<T> {
  items: T[];
  total: number;
  page: number;
  limit: number;
  hasMore: boolean;
}

export function wrapApiResponse<T>(data: unknown): SafeApiResponse<T> {
  const obj =
    data && typeof data === "object" ? (data as Record<string, unknown>) : {};

  return {
    items: ensureArray<T>(obj.items || obj.data || obj.results || data),
    total: ensureNumber(obj.total || obj.count || obj.totalCount, 0),
    page: ensureNumber(obj.page || obj.currentPage, 1),
    limit: ensureNumber(obj.limit || obj.pageSize || obj.per_page, 10),
    hasMore: ensureBoolean(obj.hasMore || obj.has_more, false),
  };
}

export default {
  ensureArray,
  safeGet,
  ensureNumber,
  ensureString,
  ensureBoolean,
  safeJsonParse,
  wrapApiResponse,
};
