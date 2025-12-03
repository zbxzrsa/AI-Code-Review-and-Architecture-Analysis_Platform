/**
 * I18nProvider Component / 国际化提供者组件
 * 
 * Wraps the application with i18n support:
 * 为应用提供国际化支持：
 * - Suspense fallback for async loading / 异步加载的 Suspense 回退
 * - Loading indicator during language switch / 语言切换时的加载指示器
 * - RTL layout support / 从右到左布局支持
 */

import React, { Suspense, useEffect } from 'react';
import { I18nextProvider } from 'react-i18next';
import { Spin, ConfigProvider } from 'antd';
import { LoadingOutlined } from '@ant-design/icons';

// Ant Design locales / Ant Design 语言包
import enUS from 'antd/locale/en_US';
import zhCN from 'antd/locale/zh_CN';
import zhTW from 'antd/locale/zh_TW';
import arEG from 'antd/locale/ar_EG';

import i18n from '../../../i18n';
import { useLanguage } from '../../../hooks/useLanguage';

import './I18nProvider.css';

/**
 * Ant Design locale map / Ant Design 语言包映射
 */
const antdLocales: Record<string, typeof enUS> = {
  en: enUS,
  'zh-CN': zhCN,
  'zh-TW': zhTW,
  ar: arEG,
};

/**
 * Loading Fallback Component / 加载回退组件
 */
const LoadingFallback: React.FC = () => (
  <div className="i18n-loading-fallback">
    <Spin indicator={<LoadingOutlined style={{ fontSize: 32 }} spin />} />
    <span className="loading-text">Loading translations...</span>
  </div>
);

/**
 * Props for I18nProvider / I18nProvider 的属性
 */
interface I18nProviderProps {
  children: React.ReactNode;
}

/**
 * Inner provider with locale sync / 带有语言同步的内部提供者
 */
const I18nInnerProvider: React.FC<I18nProviderProps> = ({ children }) => {
  const { currentLanguage, isRTL: rtl } = useLanguage();

  // Get Ant Design locale / 获取 Ant Design 语言包
  const antdLocale = antdLocales[currentLanguage] || enUS;

  // Update document attributes / 更新文档属性
  useEffect(() => {
    document.documentElement.dir = rtl ? 'rtl' : 'ltr';
    document.documentElement.lang = currentLanguage;
    
    // Add class for CSS hooks / 添加 CSS 钩子类
    document.body.classList.remove('lang-ltr', 'lang-rtl');
    document.body.classList.add(rtl ? 'lang-rtl' : 'lang-ltr');
    
    // Update theme direction for Ant Design / 更新 Ant Design 的主题方向
    document.body.setAttribute('data-direction', rtl ? 'rtl' : 'ltr');
  }, [currentLanguage, rtl]);

  return (
    <ConfigProvider 
      locale={antdLocale}
      direction={rtl ? 'rtl' : 'ltr'}
    >
      {children}
    </ConfigProvider>
  );
};

/**
 * I18nProvider Component / 国际化提供者组件
 */
export const I18nProvider: React.FC<I18nProviderProps> = ({ children }) => {
  return (
    <I18nextProvider i18n={i18n}>
      <Suspense fallback={<LoadingFallback />}>
        <I18nInnerProvider>
          {children}
        </I18nInnerProvider>
      </Suspense>
    </I18nextProvider>
  );
};

export default I18nProvider;
