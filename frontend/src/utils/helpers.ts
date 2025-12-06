/**
 * Utility Helper Functions
 *
 * Common utility functions used across the application:
 * - Date/Time formatting
 * - String manipulation
 * - Number formatting
 * - Validation
 * - Deep operations
 * - Performance utilities
 */

// ============================================
// Date & Time Utilities
// ============================================

/**
 * Format a date to relative time (e.g., "2 hours ago")
 */
export function formatRelativeTime(date: Date | string | number): string {
  const now = new Date();
  const target = new Date(date);
  const diffMs = now.getTime() - target.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHour = Math.floor(diffMin / 60);
  const diffDay = Math.floor(diffHour / 24);
  const diffWeek = Math.floor(diffDay / 7);
  const diffMonth = Math.floor(diffDay / 30);
  const diffYear = Math.floor(diffDay / 365);

  if (diffSec < 60) return "just now";
  if (diffMin < 60) return `${diffMin} minute${diffMin > 1 ? "s" : ""} ago`;
  if (diffHour < 24) return `${diffHour} hour${diffHour > 1 ? "s" : ""} ago`;
  if (diffDay < 7) return `${diffDay} day${diffDay > 1 ? "s" : ""} ago`;
  if (diffWeek < 4) return `${diffWeek} week${diffWeek > 1 ? "s" : ""} ago`;
  if (diffMonth < 12)
    return `${diffMonth} month${diffMonth > 1 ? "s" : ""} ago`;
  return `${diffYear} year${diffYear > 1 ? "s" : ""} ago`;
}

/**
 * Format a date to a readable string
 */
export function formatDate(
  date: Date | string | number,
  options: Intl.DateTimeFormatOptions = {
    year: "numeric",
    month: "short",
    day: "numeric",
  }
): string {
  return new Date(date).toLocaleDateString(undefined, options);
}

/**
 * Format a date with time
 */
export function formatDateTime(date: Date | string | number): string {
  return new Date(date).toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/**
 * Format duration in milliseconds to human readable
 */
export function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  if (ms < 3600000)
    return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
  return `${Math.floor(ms / 3600000)}h ${Math.floor((ms % 3600000) / 60000)}m`;
}

// ============================================
// String Utilities
// ============================================

/**
 * Truncate string with ellipsis
 */
export function truncate(
  str: string,
  maxLength: number,
  suffix: string = "..."
): string {
  if (str.length <= maxLength) return str;
  return str.slice(0, maxLength - suffix.length) + suffix;
}

/**
 * Convert string to slug
 */
export function slugify(str: string): string {
  return str
    .toLowerCase()
    .trim()
    .replaceAll(/[^\w\s-]/g, "")
    .replaceAll(/[\s_-]+/g, "-")
    .replaceAll(/(?:^-+|-+$)/g, "");
}

/**
 * Capitalize first letter
 */
export function capitalize(str: string): string {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

/**
 * Convert camelCase to Title Case
 */
export function camelToTitle(str: string): string {
  return str
    .replaceAll(/([A-Z])/g, " $1")
    .replaceAll(/^./, (s) => s.toUpperCase())
    .trim();
}

/**
 * Generate random string
 */
export function randomString(length: number = 8): string {
  const chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let result = "";
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}

/**
 * Highlight search term in text
 */
export function highlightText(text: string, search: string): string {
  if (!search) return text;
  const regex = new RegExp(`(${escapeRegExp(search)})`, "gi");
  return text.replace(regex, "<mark>$1</mark>");
}

function escapeRegExp(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// ============================================
// Number Utilities
// ============================================

/**
 * Format number with commas
 */
export function formatNumber(num: number, decimals: number = 0): string {
  return num.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

/**
 * Format bytes to human readable size
 */
export function formatBytes(bytes: number, decimals: number = 2): string {
  if (bytes === 0) return "0 B";

  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB", "TB", "PB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${Number.parseFloat((bytes / Math.pow(k, i)).toFixed(decimals))} ${
    sizes[i]
  }`;
}

/**
 * Format percentage
 */
export function formatPercent(value: number, decimals: number = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Clamp number between min and max
 */
export function clamp(num: number, min: number, max: number): number {
  return Math.min(Math.max(num, min), max);
}

/**
 * Round to nearest increment
 */
export function roundTo(num: number, increment: number): number {
  return Math.round(num / increment) * increment;
}

/**
 * Format large numbers (1K, 1M, etc.)
 */
export function formatCompact(num: number): string {
  if (num < 1000) return num.toString();
  if (num < 1000000) return `${(num / 1000).toFixed(1)}K`;
  if (num < 1000000000) return `${(num / 1000000).toFixed(1)}M`;
  return `${(num / 1000000000).toFixed(1)}B`;
}

// ============================================
// Validation Utilities
// ============================================

/**
 * Validate email format
 */
export function isValidEmail(email: string): boolean {
  const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return regex.test(email);
}

/**
 * Validate URL format
 */
export function isValidUrl(url: string): boolean {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
}

/**
 * Check if string is valid JSON
 */
export function isValidJson(str: string): boolean {
  try {
    JSON.parse(str);
    return true;
  } catch {
    return false;
  }
}

/**
 * Validate password strength
 */
export function validatePassword(password: string): {
  valid: boolean;
  score: number;
  feedback: string[];
} {
  const feedback: string[] = [];
  let score = 0;

  if (password.length >= 8) score += 1;
  else feedback.push("At least 8 characters");

  if (password.length >= 12) score += 1;

  if (/[a-z]/.test(password)) score += 1;
  else feedback.push("At least one lowercase letter");

  if (/[A-Z]/.test(password)) score += 1;
  else feedback.push("At least one uppercase letter");

  if (/\d/.test(password)) score += 1;
  else feedback.push("At least one number");

  if (/[^a-zA-Z0-9]/.test(password)) score += 1;
  else feedback.push("At least one special character");

  return {
    valid: score >= 4,
    score,
    feedback,
  };
}

// ============================================
// Deep Object Utilities
// ============================================

/**
 * Deep clone an object
 */
export function deepClone<T>(obj: T): T {
  if (obj === null || typeof obj !== "object") return obj;
  if (obj instanceof Date) return new Date(obj.valueOf()) as unknown as T;
  if (obj instanceof Array)
    return obj.map((item) => deepClone(item)) as unknown as T;
  if (obj instanceof Object) {
    const clone = {} as T;
    for (const key in obj) {
      if (Object.hasOwn(obj, key)) {
        (clone as any)[key] = deepClone((obj as any)[key]);
      }
    }
    return clone;
  }
  return obj;
}

/**
 * Deep merge objects
 */
export function deepMerge<T extends object>(
  target: T,
  ...sources: Partial<T>[]
): T {
  if (!sources.length) return target;
  const source = sources.shift();

  if (source === undefined) return target;

  for (const key in source) {
    if (Object.hasOwn(source, key)) {
      const targetValue = (target as any)[key];
      const sourceValue = (source as any)[key];

      if (isObject(targetValue) && isObject(sourceValue)) {
        (target as any)[key] = deepMerge({}, targetValue, sourceValue);
      } else {
        (target as any)[key] = sourceValue;
      }
    }
  }

  return deepMerge(target, ...sources);
}

function isObject(item: any): item is object {
  return item !== null && typeof item === "object" && !Array.isArray(item);
}

/**
 * Get nested property safely
 */
export function get<T>(obj: any, path: string, defaultValue?: T): T {
  const keys = path.split(".");
  let result = obj;

  for (const key of keys) {
    if (result === null || result === undefined) {
      return defaultValue as T;
    }
    result = result[key];
  }

  return result === undefined ? (defaultValue as T) : result;
}

/**
 * Set nested property
 */
export function set(obj: any, path: string, value: any): void {
  const keys = path.split(".");
  let current = obj;

  for (let i = 0; i < keys.length - 1; i++) {
    const key = keys[i];
    if (current[key] === undefined) {
      current[key] = {};
    }
    current = current[key];
  }

  current[keys[keys.length - 1]] = value;
}

/**
 * Pick specified keys from object
 */
export function pick<T extends object, K extends keyof T>(
  obj: T,
  keys: K[]
): Pick<T, K> {
  const result = {} as Pick<T, K>;
  for (const key of keys) {
    if (key in obj) {
      result[key] = obj[key];
    }
  }
  return result;
}

/**
 * Omit specified keys from object
 */
export function omit<T extends object, K extends keyof T>(
  obj: T,
  keys: K[]
): Omit<T, K> {
  const result = { ...obj };
  for (const key of keys) {
    delete result[key];
  }
  return result;
}

// ============================================
// Array Utilities
// ============================================

/**
 * Remove duplicates from array
 */
export function unique<T>(arr: T[], key?: keyof T): T[] {
  if (key) {
    const seen = new Set();
    return arr.filter((item) => {
      const k = item[key];
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    });
  }
  return [...new Set(arr)];
}

/**
 * Group array by key
 */
export function groupBy<T>(arr: T[], key: keyof T): Record<string, T[]> {
  return arr.reduce((result, item) => {
    const k = String(item[key]);
    if (!result[k]) result[k] = [];
    result[k].push(item);
    return result;
  }, {} as Record<string, T[]>);
}

/**
 * Sort array by key
 */
export function sortBy<T>(
  arr: T[],
  key: keyof T,
  order: "asc" | "desc" = "asc"
): T[] {
  return [...arr].sort((a, b) => {
    const aVal = a[key];
    const bVal = b[key];
    const modifier = order === "asc" ? 1 : -1;

    if (aVal < bVal) return -1 * modifier;
    if (aVal > bVal) return 1 * modifier;
    return 0;
  });
}

/**
 * Chunk array into smaller arrays
 */
export function chunk<T>(arr: T[], size: number): T[][] {
  const chunks: T[][] = [];
  for (let i = 0; i < arr.length; i += size) {
    chunks.push(arr.slice(i, i + size));
  }
  return chunks;
}

/**
 * Shuffle array
 */
export function shuffle<T>(arr: T[]): T[] {
  const result = [...arr];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

// ============================================
// Performance Utilities
// ============================================

/**
 * Debounce function
 */
export function debounce<T extends (...args: any[]) => any>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout>;

  return function (this: any, ...args: Parameters<T>) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn.apply(this, args), delay);
  };
}

/**
 * Throttle function
 */
export function throttle<T extends (...args: any[]) => any>(
  fn: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle = false;

  return function (this: any, ...args: Parameters<T>) {
    if (!inThrottle) {
      fn.apply(this, args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
}

/**
 * Memoize function
 */
export function memoize<T extends (...args: any[]) => any>(fn: T): T {
  const cache = new Map<string, ReturnType<T>>();

  return function (this: any, ...args: Parameters<T>): ReturnType<T> {
    const key = JSON.stringify(args);
    if (cache.has(key)) {
      return cache.get(key)!;
    }
    const result = fn.apply(this, args);
    cache.set(key, result);
    return result;
  } as T;
}

/**
 * Retry async function
 */
export async function retry<T>(
  fn: () => Promise<T>,
  maxRetries: number = 3,
  delay: number = 1000
): Promise<T> {
  let lastError: Error | null = null;

  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;
      if (i < maxRetries - 1) {
        await new Promise((resolve) =>
          setTimeout(resolve, delay * Math.pow(2, i))
        );
      }
    }
  }

  throw lastError;
}

/**
 * Sleep/delay function
 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ============================================
// Color Utilities
// ============================================

/**
 * Convert hex to RGB
 */
export function hexToRgb(
  hex: string
): { r: number; g: number; b: number } | null {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? {
        r: Number.parseInt(result[1], 16),
        g: Number.parseInt(result[2], 16),
        b: Number.parseInt(result[3], 16),
      }
    : null;
}

/**
 * Convert RGB to hex
 */
export function rgbToHex(r: number, g: number, b: number): string {
  return "#" + [r, g, b].map((x) => x.toString(16).padStart(2, "0")).join("");
}

/**
 * Lighten or darken a color
 */
export function adjustColor(hex: string, amount: number): string {
  const rgb = hexToRgb(hex);
  if (!rgb) return hex;

  const adjust = (value: number) => clamp(Math.round(value + amount), 0, 255);

  return rgbToHex(adjust(rgb.r), adjust(rgb.g), adjust(rgb.b));
}

// ============================================
// Export All
// ============================================

export default {
  // Date/Time
  formatRelativeTime,
  formatDate,
  formatDateTime,
  formatDuration,
  // String
  truncate,
  slugify,
  capitalize,
  camelToTitle,
  randomString,
  highlightText,
  // Number
  formatNumber,
  formatBytes,
  formatPercent,
  clamp,
  roundTo,
  formatCompact,
  // Validation
  isValidEmail,
  isValidUrl,
  isValidJson,
  validatePassword,
  // Deep operations
  deepClone,
  deepMerge,
  get,
  set,
  pick,
  omit,
  // Array
  unique,
  groupBy,
  sortBy,
  chunk,
  shuffle,
  // Performance
  debounce,
  throttle,
  memoize,
  retry,
  sleep,
  // Color
  hexToRgb,
  rgbToHex,
  adjustColor,
};
