/**
 * Theme Switcher Component
 * 
 * Toggle between default and pixel themes.
 */

import React from 'react';
import { Switch, Space, Tooltip, Typography, Segmented } from 'antd';
import { useTheme, ThemeStyle, ThemeMode } from '../../contexts/ThemeContext';

const { Text } = Typography;

interface ThemeSwitcherProps {
  showLabel?: boolean;
  compact?: boolean;
}

export const ThemeSwitcher: React.FC<ThemeSwitcherProps> = ({
  showLabel = true,
  compact = false,
}) => {
  const { themeStyle, themeMode, setThemeStyle, setThemeMode, isPixel, isDark } = useTheme();

  if (compact) {
    return (
      <Tooltip title={isPixel ? 'Switch to Modern Theme' : 'Switch to Pixel Theme'}>
        <Switch
          checked={isPixel}
          onChange={(checked) => setThemeStyle(checked ? 'pixel' : 'default')}
          checkedChildren="🎮"
          unCheckedChildren="✨"
        />
      </Tooltip>
    );
  }

  return (
    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
      {/* Theme Style */}
      <div>
        {showLabel && <Text type="secondary" style={{ display: 'block', marginBottom: 8 }}>Theme Style</Text>}
        <Segmented
          value={themeStyle}
          onChange={(value) => setThemeStyle(value as ThemeStyle)}
          options={[
            { label: '✨ Modern', value: 'default' },
            { label: '🎮 Pixel', value: 'pixel' },
          ]}
          block
        />
      </div>

      {/* Theme Mode */}
      <div>
        {showLabel && <Text type="secondary" style={{ display: 'block', marginBottom: 8 }}>Theme Mode</Text>}
        <Segmented
          value={themeMode}
          onChange={(value) => setThemeMode(value as ThemeMode)}
          options={[
            { label: '☀️ Light', value: 'light' },
            { label: '🌙 Dark', value: 'dark' },
            { label: '💻 System', value: 'system' },
          ]}
          block
        />
      </div>
    </Space>
  );
};

// Quick toggle for header/toolbar
export const ThemeQuickToggle: React.FC = () => {
  const { isPixel, setThemeStyle } = useTheme();

  return (
    <Tooltip title={isPixel ? 'Modern Theme' : 'Pixel Theme'}>
      <div
        onClick={() => setThemeStyle(isPixel ? 'default' : 'pixel')}
        style={{
          cursor: 'pointer',
          padding: '4px 8px',
          borderRadius: isPixel ? 0 : 4,
          border: isPixel ? '2px solid #333366' : '1px solid transparent',
          transition: 'all 0.2s',
          display: 'flex',
          alignItems: 'center',
          gap: 4,
        }}
      >
        {isPixel ? '🎮' : '✨'}
        <Text style={{ fontSize: 12 }}>{isPixel ? 'PIXEL' : 'Modern'}</Text>
      </div>
    </Tooltip>
  );
};

export default ThemeSwitcher;
