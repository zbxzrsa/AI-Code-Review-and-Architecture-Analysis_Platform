/**
 * useLanguage Hook / 语言钩子
 * 
 * Custom hook for language management with:
 * 自定义语言管理钩子，提供：
 * - Current language state / 当前语言状态
 * - Language switching / 语言切换
 * - RTL detection / 从右到左检测
 * - Loading states / 加载状态
 */

import { useState, useCallback, useMemo, useEffect } from 'react';
import { useTranslation } from 'react-i18next';

import {
  SUPPORTED_LANGUAGES,
  type LanguageConfig,
  getLanguageConfig,
  isRTL as checkRTL,
  getSupportedLanguageCodes,
  DEFAULT_LANGUAGE,
} from '../i18n/config';
import { changeLanguage, preloadLanguage, isLanguageLoaded } from '../i18n';

/**
 * Return type for useLanguage hook / useLanguage 钩子的返回类型
 */
export interface UseLanguageReturn {
  /** Current language code / 当前语言代码 */
  currentLanguage: string;
  /** Current language configuration / 当前语言配置 */
  languageConfig: LanguageConfig;
  /** List of supported languages / 支持的语言列表 */
  supportedLanguages: LanguageConfig[];
  /** Whether current language is RTL / 当前语言是否从右到左 */
  isRTL: boolean;
  /** Whether language is loading / 语言是否正在加载 */
  isLoading: boolean;
  /** Whether language is loaded / 语言是否已加载 */
  isLoaded: boolean;
  /** Change language function / 切换语言函数 */
  setLanguage: (code: string) => Promise<boolean>;
  /** Preload language function / 预加载语言函数 */
  preload: (code: string) => Promise<void>;
  /** Format date according to locale / 根据区域格式化日期 */
  formatDate: (date: Date | string | number, options?: Intl.DateTimeFormatOptions) => string;
  /** Format number according to locale / 根据区域格式化数字 */
  formatNumber: (number: number, options?: Intl.NumberFormatOptions) => string;
  /** Format currency according to locale / 根据区域格式化货币 */
  formatCurrency: (amount: number, currency?: string) => string;
}

/**
 * useLanguage Hook / 语言钩子
 * 
 * Provides language management utilities for React components.
 * 为 React 组件提供语言管理工具。
 */
export function useLanguage(): UseLanguageReturn {
  const { i18n } = useTranslation();
  
  // Loading state / 加载状态
  const [isLoading, setIsLoading] = useState(false);

  // Current language code / 当前语言代码
  const currentLanguage = i18n.language || DEFAULT_LANGUAGE;

  // Current language config / 当前语言配置
  const languageConfig = useMemo(() => {
    return getLanguageConfig(currentLanguage);
  }, [currentLanguage]);

  // Supported languages list / 支持的语言列表
  const supportedLanguages = useMemo(() => {
    return Object.values(SUPPORTED_LANGUAGES);
  }, []);

  // RTL check / 从右到左检查
  const isRTL = useMemo(() => {
    return checkRTL(currentLanguage);
  }, [currentLanguage]);

  // Check if current language is loaded / 检查当前语言是否已加载
  const isLoaded = useMemo(() => {
    return isLanguageLoaded(currentLanguage);
  }, [currentLanguage]);

  /**
   * Change language / 切换语言
   */
  const setLanguage = useCallback(async (code: string): Promise<boolean> => {
    if (code === currentLanguage) {
      return true;
    }

    if (!getSupportedLanguageCodes().includes(code)) {
      console.warn(`Language "${code}" is not supported`);
      return false;
    }

    setIsLoading(true);

    try {
      await changeLanguage(code);
      return true;
    } catch (error) {
      console.error('Failed to change language:', error);
      return false;
    } finally {
      setIsLoading(false);
    }
  }, [currentLanguage]);

  /**
   * Preload language / 预加载语言
   */
  const preload = useCallback(async (code: string): Promise<void> => {
    if (!getSupportedLanguageCodes().includes(code)) {
      console.warn(`Language "${code}" is not supported`);
      return;
    }

    try {
      await preloadLanguage(code);
    } catch (error) {
      console.error('Failed to preload language:', error);
    }
  }, []);

  /**
   * Format date according to locale / 根据区域格式化日期
   */
  const formatDate = useCallback((
    date: Date | string | number,
    options?: Intl.DateTimeFormatOptions
  ): string => {
    const dateObj = date instanceof Date ? date : new Date(date);
    const locale = languageConfig.numberLocale;
    
    return new Intl.DateTimeFormat(locale, {
      dateStyle: 'medium',
      ...options,
    }).format(dateObj);
  }, [languageConfig.numberLocale]);

  /**
   * Format number according to locale / 根据区域格式化数字
   */
  const formatNumber = useCallback((
    number: number,
    options?: Intl.NumberFormatOptions
  ): string => {
    const locale = languageConfig.numberLocale;
    return new Intl.NumberFormat(locale, options).format(number);
  }, [languageConfig.numberLocale]);

  /**
   * Format currency according to locale / 根据区域格式化货币
   */
  const formatCurrency = useCallback((
    amount: number,
    currency: string = 'USD'
  ): string => {
    const locale = languageConfig.numberLocale;
    return new Intl.NumberFormat(locale, {
      style: 'currency',
      currency,
    }).format(amount);
  }, [languageConfig.numberLocale]);

  // Update document direction when language changes
  // 语言变更时更新文档方向
  useEffect(() => {
    document.documentElement.dir = isRTL ? 'rtl' : 'ltr';
    document.documentElement.lang = currentLanguage;
  }, [currentLanguage, isRTL]);

  return {
    currentLanguage,
    languageConfig,
    supportedLanguages,
    isRTL,
    isLoading,
    isLoaded,
    setLanguage,
    preload,
    formatDate,
    formatNumber,
    formatCurrency,
  };
}

export default useLanguage;
