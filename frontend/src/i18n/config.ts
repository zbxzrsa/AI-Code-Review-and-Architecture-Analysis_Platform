/**
 * i18n Configuration / 国际化配置
 * 
 * This module defines all supported languages and their configurations.
 * 此模块定义所有支持的语言及其配置。
 */

/**
 * Language direction type / 语言方向类型
 */
export type LanguageDirection = 'ltr' | 'rtl';

/**
 * Language configuration interface / 语言配置接口
 */
export interface LanguageConfig {
  /** Language code / 语言代码 */
  code: string;
  /** Native name / 原生名称 */
  nativeName: string;
  /** English name / 英文名称 */
  englishName: string;
  /** Text direction / 文字方向 */
  direction: LanguageDirection;
  /** Flag emoji / 旗帜表情 */
  flag: string;
  /** Date format / 日期格式 */
  dateFormat: string;
  /** Number format locale / 数字格式区域 */
  numberLocale: string;
}

/**
 * Supported languages configuration / 支持的语言配置
 */
export const SUPPORTED_LANGUAGES: Record<string, LanguageConfig> = {
  en: {
    code: 'en',
    nativeName: 'English',
    englishName: 'English',
    direction: 'ltr',
    flag: '🇺🇸',
    dateFormat: 'MM/DD/YYYY',
    numberLocale: 'en-US',
  },
  'zh-CN': {
    code: 'zh-CN',
    nativeName: '简体中文',
    englishName: 'Simplified Chinese',
    direction: 'ltr',
    flag: '🇨🇳',
    dateFormat: 'YYYY-MM-DD',
    numberLocale: 'zh-CN',
  },
  'zh-TW': {
    code: 'zh-TW',
    nativeName: '繁體中文',
    englishName: 'Traditional Chinese',
    direction: 'ltr',
    flag: '🇹🇼',
    dateFormat: 'YYYY/MM/DD',
    numberLocale: 'zh-TW',
  },
  ar: {
    code: 'ar',
    nativeName: 'العربية',
    englishName: 'Arabic',
    direction: 'rtl',
    flag: '🇸🇦',
    dateFormat: 'DD/MM/YYYY',
    numberLocale: 'ar-SA',
  },
};

/**
 * Default language code / 默认语言代码
 */
export const DEFAULT_LANGUAGE = 'en';

/**
 * Fallback language code / 回退语言代码
 */
export const FALLBACK_LANGUAGE = 'en';

/**
 * LocalStorage key for language preference / 语言偏好的本地存储键
 */
export const LANGUAGE_STORAGE_KEY = 'app-language';

/**
 * Namespace definitions for modular translations / 模块化翻译的命名空间定义
 */
export const NAMESPACES = [
  'common',      // Common UI elements / 通用UI元素
  'auth',        // Authentication / 认证
  'dashboard',   // Dashboard / 仪表板
  'projects',    // Projects / 项目
  'codeReview',  // Code Review / 代码审查
  'settings',    // Settings / 设置
  'admin',       // Admin panel / 管理面板
  'errors',      // Error messages / 错误消息
  'validation',  // Validation messages / 验证消息
] as const;

export type Namespace = typeof NAMESPACES[number];

/**
 * Get language configuration by code / 通过代码获取语言配置
 */
export function getLanguageConfig(code: string): LanguageConfig {
  return SUPPORTED_LANGUAGES[code] || SUPPORTED_LANGUAGES[DEFAULT_LANGUAGE];
}

/**
 * Check if a language is RTL / 检查语言是否为从右到左
 */
export function isRTL(code: string): boolean {
  const config = getLanguageConfig(code);
  return config.direction === 'rtl';
}

/**
 * Get all supported language codes / 获取所有支持的语言代码
 */
export function getSupportedLanguageCodes(): string[] {
  return Object.keys(SUPPORTED_LANGUAGES);
}
