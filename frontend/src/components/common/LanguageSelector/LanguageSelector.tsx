/**
 * LanguageSelector Component / 语言选择器组件
 * 
 * A responsive language selector with the following features:
 * 具有以下功能的响应式语言选择器：
 * - Dropdown menu for desktop / 桌面端下拉菜单
 * - Modal dialog for mobile / 移动端模态对话框
 * - Loading states during language switch / 语言切换时的加载状态
 * - Persistence via localStorage / 通过 localStorage 持久化
 * - RTL language support / 从右到左语言支持
 */

import React, { useState, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { 
  Dropdown, 
  Button, 
  Modal, 
  List, 
  Spin, 
  Typography, 
  Space,
  message,
  Grid
} from 'antd';
import { 
  GlobalOutlined, 
  CheckOutlined, 
  LoadingOutlined,
  DownOutlined
} from '@ant-design/icons';
import type { MenuProps } from 'antd';

import { 
  SUPPORTED_LANGUAGES, 
  type LanguageConfig,
  getLanguageConfig,
  isRTL 
} from '../../../i18n/config';
import { changeLanguage, preloadLanguage } from '../../../i18n';

import './LanguageSelector.css';

const { Text } = Typography;
const { useBreakpoint } = Grid;

/**
 * Props for LanguageSelector component / LanguageSelector 组件的属性
 */
interface LanguageSelectorProps {
  /** Display mode / 显示模式 */
  mode?: 'dropdown' | 'inline' | 'icon-only';
  /** Size of the selector / 选择器大小 */
  size?: 'small' | 'middle' | 'large';
  /** Show flags / 显示旗帜 */
  showFlag?: boolean;
  /** Show native name / 显示原生名称 */
  showNativeName?: boolean;
  /** Custom class name / 自定义类名 */
  className?: string;
  /** Callback when language changes / 语言变更时的回调 */
  onLanguageChange?: (language: string) => void;
}

/**
 * LanguageSelector Component / 语言选择器组件
 */
export const LanguageSelector: React.FC<LanguageSelectorProps> = ({
  mode = 'dropdown',
  size = 'middle',
  showFlag = true,
  showNativeName = true,
  className = '',
  onLanguageChange,
}) => {
  const { t, i18n } = useTranslation();
  const screens = useBreakpoint();
  
  // State / 状态
  const [isLoading, setIsLoading] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [loadingLanguage, setLoadingLanguage] = useState<string | null>(null);

  // Current language config / 当前语言配置
  const currentLanguage = useMemo(() => {
    return getLanguageConfig(i18n.language);
  }, [i18n.language]);

  // Language list / 语言列表
  const languageList = useMemo(() => {
    return Object.values(SUPPORTED_LANGUAGES);
  }, []);

  /**
   * Handle language change / 处理语言变更
   */
  const handleLanguageChange = useCallback(async (langCode: string) => {
    if (langCode === i18n.language) {
      setIsModalOpen(false);
      return;
    }

    setIsLoading(true);
    setLoadingLanguage(langCode);

    try {
      // Preload language if needed / 如果需要则预加载语言
      await preloadLanguage(langCode);
      
      // Change language / 切换语言
      await changeLanguage(langCode);
      
      // Show success message / 显示成功消息
      const langConfig = getLanguageConfig(langCode);
      message.success(
        t('common.language_changed', { 
          defaultValue: `Language changed to ${langConfig.nativeName}` 
        })
      );

      // Call callback / 调用回调
      onLanguageChange?.(langCode);
      
      // Close modal if open / 如果模态框打开则关闭
      setIsModalOpen(false);
    } catch (error) {
      console.error('Failed to change language:', error);
      message.error(t('language_selector.error', 'Failed to load language'));
    } finally {
      setIsLoading(false);
      setLoadingLanguage(null);
    }
  }, [i18n.language, t, onLanguageChange]);

  /**
   * Preload language on hover / 悬停时预加载语言
   */
  const handleLanguageHover = useCallback((langCode: string) => {
    if (langCode !== i18n.language) {
      preloadLanguage(langCode);
    }
  }, [i18n.language]);

  /**
   * Render language item / 渲染语言项
   */
  const renderLanguageItem = useCallback((lang: LanguageConfig, isSelected: boolean) => {
    const isCurrentlyLoading = loadingLanguage === lang.code;
    
    return (
      <Space 
        className={`language-item ${isSelected ? 'selected' : ''} ${isRTL(lang.code) ? 'rtl' : 'ltr'}`}
      >
        {showFlag && <span className="language-flag">{lang.flag}</span>}
        <span className="language-name">
          {showNativeName ? lang.nativeName : lang.englishName}
        </span>
        {isCurrentlyLoading && <LoadingOutlined spin />}
        {isSelected && !isCurrentlyLoading && <CheckOutlined className="check-icon" />}
      </Space>
    );
  }, [showFlag, showNativeName, loadingLanguage]);

  /**
   * Dropdown menu items / 下拉菜单项
   */
  const menuItems: MenuProps['items'] = useMemo(() => {
    return languageList.map((lang) => ({
      key: lang.code,
      label: renderLanguageItem(lang, lang.code === i18n.language),
      onClick: () => handleLanguageChange(lang.code),
      onMouseEnter: () => handleLanguageHover(lang.code),
    }));
  }, [languageList, i18n.language, renderLanguageItem, handleLanguageChange, handleLanguageHover]);

  /**
   * Render button content / 渲染按钮内容
   */
  const renderButtonContent = () => {
    if (mode === 'icon-only') {
      return <GlobalOutlined />;
    }

    return (
      <Space>
        <GlobalOutlined />
        {showFlag && <span>{currentLanguage.flag}</span>}
        {showNativeName && !screens.xs && (
          <span className="current-language-name">{currentLanguage.nativeName}</span>
        )}
        <DownOutlined className="dropdown-arrow" />
      </Space>
    );
  };

  // Mobile: Use modal / 移动端：使用模态框
  if (screens.xs && mode !== 'inline') {
    return (
      <>
        <Button
          className={`language-selector-button ${className}`}
          size={size}
          icon={<GlobalOutlined />}
          onClick={() => setIsModalOpen(true)}
          loading={isLoading}
        >
          {!screens.xs && currentLanguage.nativeName}
        </Button>

        <Modal
          title={
            <Space>
              <GlobalOutlined />
              {t('language_selector.title', 'Select Language')}
            </Space>
          }
          open={isModalOpen}
          onCancel={() => setIsModalOpen(false)}
          footer={null}
          className="language-selector-modal"
          centered
        >
          {isLoading && (
            <div className="language-loading-overlay">
              <Spin indicator={<LoadingOutlined spin />} />
              <span className="loading-text">{t('language_selector.loading', 'Loading language...')}</span>
            </div>
          )}
          
          <List
            dataSource={languageList}
            renderItem={(lang) => {
              const isSelected = lang.code === i18n.language;
              return (
                <List.Item
                  className={`language-list-item ${isSelected ? 'selected' : ''}`}
                  onClick={() => handleLanguageChange(lang.code)}
                >
                  {renderLanguageItem(lang, isSelected)}
                </List.Item>
              );
            }}
          />
        </Modal>
      </>
    );
  }

  // Inline mode: Show all languages / 内联模式：显示所有语言
  if (mode === 'inline') {
    return (
      <div className={`language-selector-inline ${className}`}>
        <Text type="secondary" className="language-selector-label">
          {t('settings.language', 'Language')}:
        </Text>
        <Space wrap className="language-buttons">
          {languageList.map((lang) => {
            const isSelected = lang.code === i18n.language;
            const isCurrentlyLoading = loadingLanguage === lang.code;
            
            return (
              <Button
                key={lang.code}
                type={isSelected ? 'primary' : 'default'}
                size={size}
                onClick={() => handleLanguageChange(lang.code)}
                onMouseEnter={() => handleLanguageHover(lang.code)}
                loading={isCurrentlyLoading}
                className={`language-inline-button ${isRTL(lang.code) ? 'rtl' : 'ltr'}`}
              >
                {showFlag && <span className="language-flag">{lang.flag}</span>}
                {showNativeName ? lang.nativeName : lang.englishName}
              </Button>
            );
          })}
        </Space>
      </div>
    );
  }

  // Default: Dropdown mode / 默认：下拉模式
  return (
    <Dropdown
      menu={{ items: menuItems }}
      trigger={['click']}
      placement="bottomRight"
      className={`language-selector-dropdown ${className}`}
      disabled={isLoading}
    >
      <Button
        className="language-selector-button"
        size={size}
        loading={isLoading}
      >
        {renderButtonContent()}
      </Button>
    </Dropdown>
  );
};

export default LanguageSelector;
