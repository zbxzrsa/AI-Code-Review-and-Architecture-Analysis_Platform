/**
 * i18n Initialization / 国际化初始化
 * 
 * This module sets up i18next with:
 * 此模块设置i18next，包含：
 * - Dynamic language loading / 动态语言加载
 * - Browser language detection / 浏览器语言检测
 * - LocalStorage persistence / 本地存储持久化
 * - RTL support / 从右到左支持
 * - Fallback mechanism / 回退机制
 */

import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';
import Backend from 'i18next-http-backend';

import {
  DEFAULT_LANGUAGE,
  FALLBACK_LANGUAGE,
  LANGUAGE_STORAGE_KEY,
  SUPPORTED_LANGUAGES,
  isRTL,
} from './config';

// Static imports for bundled translations / 静态导入打包的翻译
import en from './locales/en/translation.json';
import zhCN from './locales/zh-CN/translation.json';

/**
 * Bundled resources for initial load / 初始加载的打包资源
 */
const bundledResources = {
  en: { translation: en },
  'zh-CN': { translation: zhCN },
};

/**
 * Language change event handler / 语言变更事件处理器
 * Updates document direction and lang attribute
 * 更新文档方向和语言属性
 */
function handleLanguageChange(lng: string): void {
  // Update document direction / 更新文档方向
  const direction = isRTL(lng) ? 'rtl' : 'ltr';
  document.documentElement.dir = direction;
  document.documentElement.lang = lng;
  
  // Update body class for styling hooks / 更新body类以供样式钩子使用
  document.body.classList.remove('lang-ltr', 'lang-rtl');
  document.body.classList.add(`lang-${direction}`);
  
  // Store preference / 存储偏好
  try {
    localStorage.setItem(LANGUAGE_STORAGE_KEY, lng);
  } catch (error) {
    console.warn('Failed to save language preference:', error);
  }
}

/**
 * Initialize i18n instance / 初始化i18n实例
 */
i18n
  // Load translations using http backend (for dynamic loading)
  // 使用http后端加载翻译（用于动态加载）
  .use(Backend)
  // Detect user language / 检测用户语言
  .use(LanguageDetector)
  // Pass the i18n instance to react-i18next
  // 将i18n实例传递给react-i18next
  .use(initReactI18next)
  // Initialize i18next / 初始化i18next
  .init({
    // Bundled resources for faster initial load
    // 打包的资源用于更快的初始加载
    resources: bundledResources,
    
    // Default language / 默认语言
    lng: DEFAULT_LANGUAGE,
    
    // Fallback language / 回退语言
    fallbackLng: FALLBACK_LANGUAGE,
    
    // Supported languages / 支持的语言
    supportedLngs: Object.keys(SUPPORTED_LANGUAGES),
    
    // Don't load missing translations from backend for bundled langs
    // 对于打包的语言不从后端加载缺失的翻译
    partialBundledLanguages: true,
    
    // Debug mode in development / 开发环境中的调试模式
    debug: import.meta.env.DEV,
    
    // Namespace configuration / 命名空间配置
    ns: ['translation'],
    defaultNS: 'translation',
    
    // Interpolation options / 插值选项
    interpolation: {
      escapeValue: false, // React already escapes / React已经转义
      formatSeparator: ',',
    },
    
    // Language detection configuration / 语言检测配置
    detection: {
      // Detection order / 检测顺序
      order: ['localStorage', 'querystring', 'navigator', 'htmlTag'],
      // Cache user language / 缓存用户语言
      caches: ['localStorage'],
      // localStorage key / 本地存储键
      lookupLocalStorage: LANGUAGE_STORAGE_KEY,
      // Query string key / 查询字符串键
      lookupQuerystring: 'lang',
      // Check whitelist / 检查白名单
      checkWhitelist: true,
    },
    
    // Backend configuration for dynamic loading
    // 动态加载的后端配置
    backend: {
      loadPath: '/locales/{{lng}}/{{ns}}.json',
      requestOptions: {
        cache: 'default',
      },
    },
    
    // React specific options / React特定选项
    react: {
      useSuspense: true,
      bindI18n: 'languageChanged loaded',
      bindI18nStore: 'added removed',
      transEmptyNodeValue: '',
      transSupportBasicHtmlNodes: true,
      transKeepBasicHtmlNodesFor: ['br', 'strong', 'i', 'p', 'span'],
    },
    
    // Missing key handler / 缺失键处理器
    saveMissing: import.meta.env.DEV,
    missingKeyHandler: (lngs, ns, key, fallbackValue) => {
      if (import.meta.env.DEV) {
        console.warn(`Missing translation: [${lngs}] ${ns}:${key}`);
      }
    },
  });

// Listen for language changes / 监听语言变更
i18n.on('languageChanged', handleLanguageChange);

// Apply initial language settings / 应用初始语言设置
handleLanguageChange(i18n.language);

/**
 * Change language programmatically / 程序化更改语言
 * 
 * @param lng - Language code / 语言代码
 * @returns Promise that resolves when language is changed / 语言更改完成时解析的Promise
 */
export async function changeLanguage(lng: string): Promise<void> {
  if (!SUPPORTED_LANGUAGES[lng]) {
    console.warn(`Language "${lng}" is not supported, falling back to "${FALLBACK_LANGUAGE}"`);
    lng = FALLBACK_LANGUAGE;
  }
  await i18n.changeLanguage(lng);
}

/**
 * Get current language / 获取当前语言
 */
export function getCurrentLanguage(): string {
  return i18n.language;
}

/**
 * Check if language is loaded / 检查语言是否已加载
 */
export function isLanguageLoaded(lng: string): boolean {
  return i18n.hasResourceBundle(lng, 'translation');
}

/**
 * Preload a language / 预加载语言
 */
export async function preloadLanguage(lng: string): Promise<void> {
  if (!isLanguageLoaded(lng)) {
    await i18n.loadLanguages(lng);
  }
}

export default i18n;
