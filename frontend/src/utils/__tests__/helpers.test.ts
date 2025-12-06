/**
 * Utility Helpers Tests
 *
 * Tests for common utility functions.
 */

import {
  formatRelativeTime,
  formatDuration,
  truncate,
  slugify,
  capitalize,
  camelToTitle,
  randomString,
  formatNumber,
  formatBytes,
  formatPercent,
  clamp,
  roundTo,
  formatCompact,
  isValidEmail,
  isValidUrl,
  isValidJson,
  validatePassword,
  deepClone,
  deepMerge,
  get,
  set,
  pick,
  omit,
  unique,
  groupBy,
  sortBy,
  chunk,
  shuffle,
  debounce,
  throttle,
  memoize,
  retry,
  sleep,
  hexToRgb,
  rgbToHex,
  adjustColor,
} from "../helpers";

describe("Date & Time Utilities", () => {
  describe("formatRelativeTime", () => {
    it('should return "just now" for recent times', () => {
      const now = new Date();
      expect(formatRelativeTime(now)).toBe("just now");
    });

    it("should format minutes correctly", () => {
      const fiveMinutesAgo = new Date(Date.now() - 5 * 60 * 1000);
      expect(formatRelativeTime(fiveMinutesAgo)).toBe("5 minutes ago");
    });

    it("should format hours correctly", () => {
      const twoHoursAgo = new Date(Date.now() - 2 * 60 * 60 * 1000);
      expect(formatRelativeTime(twoHoursAgo)).toBe("2 hours ago");
    });

    it("should format days correctly", () => {
      const threeDaysAgo = new Date(Date.now() - 3 * 24 * 60 * 60 * 1000);
      expect(formatRelativeTime(threeDaysAgo)).toBe("3 days ago");
    });

    it("should handle singular forms", () => {
      const oneMinuteAgo = new Date(Date.now() - 1 * 60 * 1000);
      expect(formatRelativeTime(oneMinuteAgo)).toBe("1 minute ago");
    });
  });

  describe("formatDuration", () => {
    it("should format milliseconds", () => {
      expect(formatDuration(500)).toBe("500ms");
    });

    it("should format seconds", () => {
      expect(formatDuration(5000)).toBe("5.0s");
    });

    it("should format minutes and seconds", () => {
      expect(formatDuration(125000)).toBe("2m 5s");
    });

    it("should format hours and minutes", () => {
      expect(formatDuration(3725000)).toBe("1h 2m");
    });
  });
});

describe("String Utilities", () => {
  describe("truncate", () => {
    it("should not truncate short strings", () => {
      expect(truncate("hello", 10)).toBe("hello");
    });

    it("should truncate long strings with ellipsis", () => {
      expect(truncate("hello world", 8)).toBe("hello...");
    });

    it("should use custom suffix", () => {
      expect(truncate("hello world", 8, "…")).toBe("hello w…");
    });
  });

  describe("slugify", () => {
    it("should convert to lowercase", () => {
      expect(slugify("Hello World")).toBe("hello-world");
    });

    it("should remove special characters", () => {
      expect(slugify("Hello, World!")).toBe("hello-world");
    });

    it("should handle multiple spaces", () => {
      expect(slugify("Hello   World")).toBe("hello-world");
    });
  });

  describe("capitalize", () => {
    it("should capitalize first letter", () => {
      expect(capitalize("hello")).toBe("Hello");
    });

    it("should handle empty string", () => {
      expect(capitalize("")).toBe("");
    });
  });

  describe("camelToTitle", () => {
    it("should convert camelCase to Title Case", () => {
      expect(camelToTitle("helloWorld")).toBe("Hello World");
    });

    it("should handle multiple capitals", () => {
      expect(camelToTitle("thisIsATest")).toBe("This Is A Test");
    });
  });

  describe("randomString", () => {
    it("should generate string of correct length", () => {
      expect(randomString(10)).toHaveLength(10);
      expect(randomString(20)).toHaveLength(20);
    });

    it("should generate alphanumeric characters", () => {
      const str = randomString(100);
      expect(str).toMatch(/^[A-Za-z0-9]+$/);
    });
  });
});

describe("Number Utilities", () => {
  describe("formatNumber", () => {
    it("should format with thousand separators", () => {
      expect(formatNumber(1000000)).toBe("1,000,000");
    });

    it("should handle decimals", () => {
      expect(formatNumber(1234.5678, 2)).toBe("1,234.57");
    });
  });

  describe("formatBytes", () => {
    it("should format bytes", () => {
      expect(formatBytes(0)).toBe("0 B");
      expect(formatBytes(100)).toBe("100 B");
    });

    it("should format kilobytes", () => {
      expect(formatBytes(1024)).toBe("1 KB");
      expect(formatBytes(1536)).toBe("1.5 KB");
    });

    it("should format megabytes", () => {
      expect(formatBytes(1048576)).toBe("1 MB");
    });

    it("should format gigabytes", () => {
      expect(formatBytes(1073741824)).toBe("1 GB");
    });
  });

  describe("formatPercent", () => {
    it("should format as percentage", () => {
      expect(formatPercent(0.5)).toBe("50.0%");
      expect(formatPercent(0.123, 2)).toBe("12.30%");
    });
  });

  describe("clamp", () => {
    it("should clamp value between min and max", () => {
      expect(clamp(5, 0, 10)).toBe(5);
      expect(clamp(-5, 0, 10)).toBe(0);
      expect(clamp(15, 0, 10)).toBe(10);
    });
  });

  describe("roundTo", () => {
    it("should round to nearest increment", () => {
      expect(roundTo(7, 5)).toBe(5);
      expect(roundTo(8, 5)).toBe(10);
      expect(roundTo(12, 5)).toBe(10);
    });
  });

  describe("formatCompact", () => {
    it("should format small numbers as-is", () => {
      expect(formatCompact(999)).toBe("999");
    });

    it("should format thousands", () => {
      expect(formatCompact(1500)).toBe("1.5K");
    });

    it("should format millions", () => {
      expect(formatCompact(2500000)).toBe("2.5M");
    });

    it("should format billions", () => {
      expect(formatCompact(1200000000)).toBe("1.2B");
    });
  });
});

describe("Validation Utilities", () => {
  describe("isValidEmail", () => {
    it("should validate correct emails", () => {
      expect(isValidEmail("test@example.com")).toBe(true);
      expect(isValidEmail("user.name@domain.co.uk")).toBe(true);
    });

    it("should reject invalid emails", () => {
      expect(isValidEmail("invalid")).toBe(false);
      expect(isValidEmail("invalid@")).toBe(false);
      expect(isValidEmail("@domain.com")).toBe(false);
    });
  });

  describe("isValidUrl", () => {
    it("should validate correct URLs", () => {
      expect(isValidUrl("https://example.com")).toBe(true);
      expect(isValidUrl("http://localhost:3000")).toBe(true);
    });

    it("should reject invalid URLs", () => {
      expect(isValidUrl("not-a-url")).toBe(false);
      expect(isValidUrl("ftp://invalid")).toBe(true); // FTP is valid URL
    });
  });

  describe("isValidJson", () => {
    it("should validate correct JSON", () => {
      expect(isValidJson('{"key": "value"}')).toBe(true);
      expect(isValidJson("[]")).toBe(true);
      expect(isValidJson("null")).toBe(true);
    });

    it("should reject invalid JSON", () => {
      expect(isValidJson("{")).toBe(false);
      expect(isValidJson("undefined")).toBe(false);
    });
  });

  describe("validatePassword", () => {
    it("should validate strong passwords", () => {
      const result = validatePassword("SecureP@ss123");
      expect(result.valid).toBe(true);
      expect(result.score).toBeGreaterThanOrEqual(4);
    });

    it("should reject weak passwords", () => {
      const result = validatePassword("weak");
      expect(result.valid).toBe(false);
      expect(result.feedback.length).toBeGreaterThan(0);
    });

    it("should provide feedback for missing requirements", () => {
      const result = validatePassword("lowercase");
      expect(result.feedback).toContain("At least one uppercase letter");
      expect(result.feedback).toContain("At least one number");
    });
  });
});

describe("Deep Object Utilities", () => {
  describe("deepClone", () => {
    it("should clone primitive values", () => {
      expect(deepClone(5)).toBe(5);
      expect(deepClone("hello")).toBe("hello");
      expect(deepClone(null)).toBe(null);
    });

    it("should deep clone objects", () => {
      const original = { a: 1, b: { c: 2 } };
      const cloned = deepClone(original);

      expect(cloned).toEqual(original);
      expect(cloned).not.toBe(original);
      expect(cloned.b).not.toBe(original.b);
    });

    it("should deep clone arrays", () => {
      const original = [1, [2, 3], { a: 4 }];
      const cloned = deepClone(original);

      expect(cloned).toEqual(original);
      expect(cloned).not.toBe(original);
      expect(cloned[1]).not.toBe(original[1]);
    });

    it("should clone dates", () => {
      const date = new Date();
      const cloned = deepClone(date);

      expect(cloned.getTime()).toBe(date.getTime());
      expect(cloned).not.toBe(date);
    });
  });

  describe("deepMerge", () => {
    it("should merge objects deeply", () => {
      const target = { a: 1, b: { c: 2 } };
      const source = { b: { d: 3 }, e: 4 };

      const result = deepMerge(target, source);

      expect(result).toEqual({ a: 1, b: { c: 2, d: 3 }, e: 4 });
    });

    it("should handle multiple sources", () => {
      const result = deepMerge({}, { a: 1 }, { b: 2 }, { c: 3 });
      expect(result).toEqual({ a: 1, b: 2, c: 3 });
    });
  });

  describe("get", () => {
    it("should get nested property", () => {
      const obj = { a: { b: { c: 1 } } };
      expect(get(obj, "a.b.c")).toBe(1);
    });

    it("should return default for missing path", () => {
      const obj = { a: 1 };
      expect(get(obj, "b.c", "default")).toBe("default");
    });

    it("should handle null/undefined", () => {
      expect(get(null, "a.b", "default")).toBe("default");
    });
  });

  describe("set", () => {
    it("should set nested property", () => {
      const obj: Record<string, any> = {};
      set(obj, "a.b.c", 1);
      expect(obj.a.b.c).toBe(1);
    });
  });

  describe("pick", () => {
    it("should pick specified keys", () => {
      const obj = { a: 1, b: 2, c: 3 };
      expect(pick(obj, ["a", "c"])).toEqual({ a: 1, c: 3 });
    });
  });

  describe("omit", () => {
    it("should omit specified keys", () => {
      const obj = { a: 1, b: 2, c: 3 };
      expect(omit(obj, ["b"])).toEqual({ a: 1, c: 3 });
    });
  });
});

describe("Array Utilities", () => {
  describe("unique", () => {
    it("should remove duplicates from primitives", () => {
      expect(unique([1, 2, 2, 3, 3, 3])).toEqual([1, 2, 3]);
    });

    it("should remove duplicates by key", () => {
      const arr = [
        { id: 1, name: "a" },
        { id: 2, name: "b" },
        { id: 1, name: "c" },
      ];
      expect(unique(arr, "id")).toHaveLength(2);
    });
  });

  describe("groupBy", () => {
    it("should group by key", () => {
      const arr = [
        { type: "a", value: 1 },
        { type: "b", value: 2 },
        { type: "a", value: 3 },
      ];
      const result = groupBy(arr, "type");

      expect(result.a).toHaveLength(2);
      expect(result.b).toHaveLength(1);
    });
  });

  describe("sortBy", () => {
    it("should sort ascending", () => {
      const arr = [{ n: 3 }, { n: 1 }, { n: 2 }];
      expect(sortBy(arr, "n", "asc").map((x) => x.n)).toEqual([1, 2, 3]);
    });

    it("should sort descending", () => {
      const arr = [{ n: 3 }, { n: 1 }, { n: 2 }];
      expect(sortBy(arr, "n", "desc").map((x) => x.n)).toEqual([3, 2, 1]);
    });
  });

  describe("chunk", () => {
    it("should chunk array into smaller arrays", () => {
      expect(chunk([1, 2, 3, 4, 5], 2)).toEqual([[1, 2], [3, 4], [5]]);
    });
  });

  describe("shuffle", () => {
    it("should return array with same elements", () => {
      const arr = [1, 2, 3, 4, 5];
      const shuffled = shuffle(arr);

      expect(shuffled).toHaveLength(arr.length);
      expect(shuffled.sort((a, b) => a - b)).toEqual(arr.sort((a, b) => a - b));
    });

    it("should not modify original array", () => {
      const arr = [1, 2, 3, 4, 5];
      const original = [...arr];
      shuffle(arr);

      expect(arr).toEqual(original);
    });
  });
});

describe("Performance Utilities", () => {
  describe("debounce", () => {
    jest.useFakeTimers();

    it("should debounce function calls", () => {
      const fn = jest.fn();
      const debounced = debounce(fn, 100);

      debounced();
      debounced();
      debounced();

      expect(fn).not.toHaveBeenCalled();

      jest.advanceTimersByTime(100);

      expect(fn).toHaveBeenCalledTimes(1);
    });
  });

  describe("throttle", () => {
    jest.useFakeTimers();

    it("should throttle function calls", () => {
      const fn = jest.fn();
      const throttled = throttle(fn, 100);

      throttled();
      throttled();
      throttled();

      expect(fn).toHaveBeenCalledTimes(1);

      jest.advanceTimersByTime(100);

      throttled();
      expect(fn).toHaveBeenCalledTimes(2);
    });
  });

  describe("memoize", () => {
    it("should cache function results", () => {
      const fn = jest.fn((x: number) => x * 2);
      const memoized = memoize(fn);

      expect(memoized(5)).toBe(10);
      expect(memoized(5)).toBe(10);
      expect(memoized(5)).toBe(10);

      expect(fn).toHaveBeenCalledTimes(1);
    });

    it("should call function for different arguments", () => {
      const fn = jest.fn((x: number) => x * 2);
      const memoized = memoize(fn);

      memoized(5);
      memoized(10);

      expect(fn).toHaveBeenCalledTimes(2);
    });
  });

  describe("retry", () => {
    it("should retry on failure", async () => {
      let attempts = 0;
      const fn = jest.fn(async () => {
        attempts++;
        if (attempts < 3) throw new Error("fail");
        return "success";
      });

      const result = await retry(fn, 3, 10);

      expect(result).toBe("success");
      expect(fn).toHaveBeenCalledTimes(3);
    });

    it("should throw after max retries", async () => {
      const fn = jest.fn(async () => {
        throw new Error("always fails");
      });

      await expect(retry(fn, 2, 10)).rejects.toThrow("always fails");
      expect(fn).toHaveBeenCalledTimes(2);
    });
  });

  describe("sleep", () => {
    jest.useFakeTimers();

    it("should delay execution", async () => {
      const fn = jest.fn();

      sleep(1000).then(fn);

      expect(fn).not.toHaveBeenCalled();

      jest.advanceTimersByTime(1000);
      await Promise.resolve(); // Flush promises

      expect(fn).toHaveBeenCalled();
    });
  });
});

describe("Color Utilities", () => {
  describe("hexToRgb", () => {
    it("should convert hex to RGB", () => {
      expect(hexToRgb("#ffffff")).toEqual({ r: 255, g: 255, b: 255 });
      expect(hexToRgb("#000000")).toEqual({ r: 0, g: 0, b: 0 });
      expect(hexToRgb("#ff5733")).toEqual({ r: 255, g: 87, b: 51 });
    });

    it("should handle hex without #", () => {
      expect(hexToRgb("ffffff")).toEqual({ r: 255, g: 255, b: 255 });
    });

    it("should return null for invalid hex", () => {
      expect(hexToRgb("invalid")).toBeNull();
      expect(hexToRgb("#fff")).toBeNull(); // 3-char hex not supported
    });
  });

  describe("rgbToHex", () => {
    it("should convert RGB to hex", () => {
      expect(rgbToHex(255, 255, 255)).toBe("#ffffff");
      expect(rgbToHex(0, 0, 0)).toBe("#000000");
      expect(rgbToHex(255, 87, 51)).toBe("#ff5733");
    });
  });

  describe("adjustColor", () => {
    it("should lighten color", () => {
      const result = adjustColor("#808080", 50);
      expect(result).not.toBe("#808080");
    });

    it("should darken color", () => {
      const result = adjustColor("#808080", -50);
      expect(result).not.toBe("#808080");
    });

    it("should clamp to valid range", () => {
      const lightened = adjustColor("#ffffff", 100);
      expect(lightened).toBe("#ffffff"); // Already max

      const darkened = adjustColor("#000000", -100);
      expect(darkened).toBe("#000000"); // Already min
    });
  });
});
