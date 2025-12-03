/**
 * Formatters Utility
 *
 * Common formatting functions for dates, numbers, and text.
 */

// ============================================
// Date Formatters
// ============================================

/**
 * Format a date string to locale format
 */
export const formatDate = (
  dateString: string | Date,
  options: Intl.DateTimeFormatOptions = {
    year: "numeric",
    month: "long",
    day: "numeric",
  },
  locale = "en-US"
): string => {
  const date =
    typeof dateString === "string" ? new Date(dateString) : dateString;
  return date.toLocaleDateString(locale, options);
};

/**
 * Format a date to short format (e.g., "Mar 1, 2024")
 */
export const formatDateShort = (
  dateString: string | Date,
  locale = "en-US"
): string => {
  return formatDate(
    dateString,
    { month: "short", day: "numeric", year: "numeric" },
    locale
  );
};

/**
 * Format a date with time
 */
export const formatDateTime = (
  dateString: string | Date,
  locale = "en-US"
): string => {
  const date =
    typeof dateString === "string" ? new Date(dateString) : dateString;
  return date.toLocaleString(locale, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
};

/**
 * Get relative time string (e.g., "2 hours ago")
 */
export const getRelativeTime = (dateString: string | Date): string => {
  const date =
    typeof dateString === "string" ? new Date(dateString) : dateString;
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSeconds = Math.floor(diffMs / 1000);
  const diffMinutes = Math.floor(diffSeconds / 60);
  const diffHours = Math.floor(diffMinutes / 60);
  const diffDays = Math.floor(diffHours / 24);
  const diffWeeks = Math.floor(diffDays / 7);
  const diffMonths = Math.floor(diffDays / 30);
  const diffYears = Math.floor(diffDays / 365);

  if (diffSeconds < 60) return "Just now";
  if (diffMinutes < 60)
    return `${diffMinutes} minute${diffMinutes !== 1 ? "s" : ""} ago`;
  if (diffHours < 24)
    return `${diffHours} hour${diffHours !== 1 ? "s" : ""} ago`;
  if (diffDays === 1) return "Yesterday";
  if (diffDays < 7) return `${diffDays} days ago`;
  if (diffWeeks < 4)
    return `${diffWeeks} week${diffWeeks !== 1 ? "s" : ""} ago`;
  if (diffMonths < 12)
    return `${diffMonths} month${diffMonths !== 1 ? "s" : ""} ago`;
  return `${diffYears} year${diffYears !== 1 ? "s" : ""} ago`;
};

/**
 * Format duration in milliseconds to human readable
 */
export const formatDuration = (ms: number): string => {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  if (ms < 3600000)
    return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
  return `${Math.floor(ms / 3600000)}h ${Math.floor((ms % 3600000) / 60000)}m`;
};

// ============================================
// Number Formatters
// ============================================

/**
 * Format number with locale-specific separators
 */
export const formatNumber = (num: number, locale = "en-US"): string => {
  return num.toLocaleString(locale);
};

/**
 * Format number to compact form (e.g., 1.2K, 3.4M)
 */
export const formatCompactNumber = (num: number, locale = "en-US"): string => {
  return Intl.NumberFormat(locale, { notation: "compact" }).format(num);
};

/**
 * Format percentage
 */
export const formatPercent = (value: number, decimals = 1): string => {
  return `${value.toFixed(decimals)}%`;
};

/**
 * Format bytes to human readable size
 */
export const formatBytes = (bytes: number, decimals = 2): string => {
  if (bytes === 0) return "0 Bytes";

  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB", "TB", "PB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(decimals))} ${
    sizes[i]
  }`;
};

/**
 * Format currency
 */
export const formatCurrency = (
  amount: number,
  currency = "USD",
  locale = "en-US"
): string => {
  return new Intl.NumberFormat(locale, {
    style: "currency",
    currency,
  }).format(amount);
};

// ============================================
// Text Formatters
// ============================================

/**
 * Truncate text with ellipsis
 */
export const truncate = (
  text: string,
  maxLength: number,
  suffix = "..."
): string => {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength - suffix.length) + suffix;
};

/**
 * Capitalize first letter
 */
export const capitalize = (text: string): string => {
  if (!text) return "";
  return text.charAt(0).toUpperCase() + text.slice(1);
};

/**
 * Convert to title case
 */
export const toTitleCase = (text: string): string => {
  return text
    .toLowerCase()
    .split(" ")
    .map((word) => capitalize(word))
    .join(" ");
};

/**
 * Convert camelCase to Title Case
 */
export const camelToTitle = (text: string): string => {
  return text
    .replace(/([A-Z])/g, " $1")
    .replace(/^./, (str) => str.toUpperCase())
    .trim();
};

/**
 * Slugify text for URLs
 */
export const slugify = (text: string): string => {
  return text
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, "")
    .replace(/[\s_-]+/g, "-")
    .replace(/^-+|-+$/g, "");
};

/**
 * Pluralize a word based on count
 */
export const pluralize = (
  count: number,
  singular: string,
  plural?: string
): string => {
  return count === 1 ? singular : plural || `${singular}s`;
};

/**
 * Format count with label (e.g., "5 items", "1 item")
 */
export const formatCount = (count: number, label: string): string => {
  return `${formatNumber(count)} ${pluralize(count, label)}`;
};

// ============================================
// Code/Technical Formatters
// ============================================

/**
 * Format code line numbers
 */
export const formatLineNumber = (line: number, maxLines: number): string => {
  const padding = String(maxLines).length;
  return String(line).padStart(padding, " ");
};

/**
 * Format file path for display
 */
export const formatFilePath = (path: string, maxLength = 50): string => {
  if (path.length <= maxLength) return path;

  const parts = path.split("/");
  if (parts.length <= 2) return truncate(path, maxLength);

  const fileName = parts[parts.length - 1];
  const remaining = maxLength - fileName.length - 4; // 4 for ".../"

  if (remaining <= 0) return truncate(fileName, maxLength);

  const firstPart = parts.slice(0, -1).join("/");
  return truncate(firstPart, remaining) + "/" + fileName;
};

/**
 * Format version number
 */
export const formatVersion = (version: string): string => {
  // Remove 'v' prefix if present
  return version.startsWith("v") ? version : `v${version}`;
};

/**
 * Compare semantic versions
 * Returns: -1 if a < b, 0 if a = b, 1 if a > b
 */
export const compareVersions = (a: string, b: string): number => {
  const cleanA = a.replace(/^v/, "");
  const cleanB = b.replace(/^v/, "");

  const partsA = cleanA.split(".").map(Number);
  const partsB = cleanB.split(".").map(Number);

  for (let i = 0; i < Math.max(partsA.length, partsB.length); i++) {
    const numA = partsA[i] || 0;
    const numB = partsB[i] || 0;

    if (numA < numB) return -1;
    if (numA > numB) return 1;
  }

  return 0;
};

export default {
  formatDate,
  formatDateShort,
  formatDateTime,
  getRelativeTime,
  formatDuration,
  formatNumber,
  formatCompactNumber,
  formatPercent,
  formatBytes,
  formatCurrency,
  truncate,
  capitalize,
  toTitleCase,
  camelToTitle,
  slugify,
  pluralize,
  formatCount,
  formatLineNumber,
  formatFilePath,
  formatVersion,
  compareVersions,
};
