/**
 * i18n Test Suite / 国际化测试套件
 *
 * Tests for:
 * 测试内容：
 * - Initial language is English / 初始语言为英语
 * - Language switching works / 语言切换有效
 * - All UI elements are translated / 所有UI元素都已翻译
 * - Language persistence / 语言持久化
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import i18n, {
  changeLanguage,
  getCurrentLanguage,
  isLanguageLoaded,
} from "../index";
import {
  SUPPORTED_LANGUAGES,
  DEFAULT_LANGUAGE,
  LANGUAGE_STORAGE_KEY,
  isRTL,
  getLanguageConfig,
  getSupportedLanguageCodes,
} from "../config";

// Mock localStorage / 模拟 localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value;
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
  };
})();

Object.defineProperty(globalThis, "localStorage", { value: localStorageMock });

describe("i18n Configuration / 国际化配置", () => {
  beforeEach(() => {
    localStorageMock.clear();
  });

  afterEach(() => {
    // Reset to default language / 重置为默认语言
    i18n.changeLanguage(DEFAULT_LANGUAGE);
  });

  describe("Initial Language / 初始语言", () => {
    it("should default to English / 应默认为英语", () => {
      expect(DEFAULT_LANGUAGE).toBe("en");
    });

    it("should have English as fallback / 应以英语作为回退语言", () => {
      const config = i18n.options;
      expect(config.fallbackLng).toBe("en");
    });

    it("should load English translations / 应加载英语翻译", () => {
      expect(isLanguageLoaded("en")).toBe(true);
    });
  });

  describe("Supported Languages / 支持的语言", () => {
    it("should support at least English and Chinese / 应至少支持英语和中文", () => {
      const codes = getSupportedLanguageCodes();
      expect(codes).toContain("en");
      expect(codes).toContain("zh-CN");
    });

    it("should have valid configuration for each language / 每种语言应有有效配置", () => {
      Object.keys(SUPPORTED_LANGUAGES).forEach((code) => {
        const config = getLanguageConfig(code);
        expect(config.code).toBe(code);
        expect(config.nativeName).toBeTruthy();
        expect(config.englishName).toBeTruthy();
        expect(["ltr", "rtl"]).toContain(config.direction);
        expect(config.flag).toBeTruthy();
      });
    });
  });

  describe("Language Switching / 语言切换", () => {
    it("should change language successfully / 应成功切换语言", async () => {
      await changeLanguage("zh-CN");
      expect(getCurrentLanguage()).toBe("zh-CN");
    });

    it("should fall back to default for unsupported languages / 不支持的语言应回退到默认语言", async () => {
      await changeLanguage("invalid-lang");
      expect(getCurrentLanguage()).toBe(DEFAULT_LANGUAGE);
    });

    it("should persist language preference / 应持久化语言偏好", async () => {
      await changeLanguage("zh-CN");
      expect(localStorageMock.getItem(LANGUAGE_STORAGE_KEY)).toBe("zh-CN");
    });
  });

  describe("RTL Support / 从右到左支持", () => {
    it("should correctly identify RTL languages / 应正确识别从右到左语言", () => {
      expect(isRTL("ar")).toBe(true);
      expect(isRTL("en")).toBe(false);
      expect(isRTL("zh-CN")).toBe(false);
    });
  });

  describe("Translation Keys / 翻译键", () => {
    it("should have common translations / 应有通用翻译", () => {
      expect(i18n.t("common.loading")).toBe("Loading...");
      expect(i18n.t("common.error")).toBe("Error");
      expect(i18n.t("common.success")).toBe("Success");
    });

    it("should have login translations / 应有登录翻译", () => {
      expect(i18n.t("login.title")).toBe("Welcome Back");
      expect(i18n.t("login.submit")).toBe("Sign In");
    });

    it("should have navigation translations / 应有导航翻译", () => {
      expect(i18n.t("nav.dashboard")).toBe("Dashboard");
      expect(i18n.t("nav.projects")).toBe("Projects");
    });

    it("should return Chinese translations when language is zh-CN / 当语言为中文时应返回中文翻译", async () => {
      await i18n.changeLanguage("zh-CN");
      expect(i18n.t("common.loading")).toBe("加载中...");
      expect(i18n.t("login.title")).toBe("欢迎回来");
    });
  });

  describe("Interpolation / 插值", () => {
    it("should interpolate values correctly / 应正确插值", () => {
      const result = i18n.t("validation.min_length", { min: 8 });
      expect(result).toContain("8");
    });
  });

  describe("Missing Keys / 缺失键", () => {
    it("should return key for missing translations / 缺失翻译应返回键", () => {
      const result = i18n.t("nonexistent.key");
      expect(result).toBe("nonexistent.key");
    });

    it("should use fallback value if provided / 如果提供应使用回退值", () => {
      const result = i18n.t("nonexistent.key", {
        defaultValue: "Default Value",
      });
      expect(result).toBe("Default Value");
    });
  });
});

describe("Translation Coverage / 翻译覆盖率", () => {
  const requiredKeys = [
    // Common / 通用
    "common.loading",
    "common.error",
    "common.success",
    "common.cancel",
    "common.save",
    "common.delete",

    // Auth / 认证
    "login.title",
    "login.submit",
    "register.title",
    "register.submit",

    // Navigation / 导航
    "nav.dashboard",
    "nav.projects",
    "nav.code_review",

    // Dashboard / 仪表板
    "dashboard.welcome",
    "dashboard.total_projects",

    // Errors / 错误
    "errors.network",
    "errors.unauthorized",
    "errors.not_found",
  ];

  it("should have all required keys in English / 应有所有必需的英语键", () => {
    requiredKeys.forEach((key) => {
      const value = i18n.t(key, { lng: "en" });
      expect(value).not.toBe(key);
    });
  });

  it("should have all required keys in Chinese / 应有所有必需的中文键", async () => {
    await i18n.changeLanguage("zh-CN");
    requiredKeys.forEach((key) => {
      const value = i18n.t(key);
      expect(value).not.toBe(key);
    });
  });
});
