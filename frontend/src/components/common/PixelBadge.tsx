/**
 * Pixel Badge Component
 * 
 * Pixel-style badges and status indicators.
 */

import React from 'react';
import { Typography } from 'antd';

const { Text } = Typography;

interface PixelBadgeProps {
  text: string;
  color?: 'green' | 'red' | 'yellow' | 'blue' | 'purple' | 'default';
  size?: 'small' | 'medium' | 'large';
  animated?: boolean;
  blinking?: boolean;
}

const colorMap = {
  green: { bg: '#00ff88', text: '#000', border: '#00cc6a' },
  red: { bg: '#ff4757', text: '#fff', border: '#cc3a47' },
  yellow: { bg: '#ffd93d', text: '#000', border: '#ccae31' },
  blue: { bg: '#6bcbff', text: '#000', border: '#56a3cc' },
  purple: { bg: '#a855f7', text: '#fff', border: '#8644c5' },
  default: { bg: '#333366', text: '#fff', border: '#222244' },
};

const sizeMap = {
  small: { padding: '2px 6px', fontSize: 10 },
  medium: { padding: '4px 10px', fontSize: 12 },
  large: { padding: '6px 14px', fontSize: 14 },
};

export const PixelBadge: React.FC<PixelBadgeProps> = ({
  text,
  color = 'default',
  size = 'medium',
  animated = false,
  blinking = false,
}) => {
  const colors = colorMap[color];
  const sizes = sizeMap[size];

  return (
    <span
      style={{
        display: 'inline-block',
        fontFamily: "'Press Start 2P', cursive",
        fontSize: sizes.fontSize,
        padding: sizes.padding,
        backgroundColor: colors.bg,
        color: colors.text,
        border: `2px solid ${colors.border}`,
        boxShadow: '2px 2px 0 rgba(0,0,0,0.3)',
        textTransform: 'uppercase',
        letterSpacing: '1px',
        animation: blinking ? 'pixel-blink 1s step-end infinite' : undefined,
      }}
      className={animated ? 'pixel-glow' : undefined}
    >
      {text}
    </span>
  );
};

// Pixel-style status indicator
interface PixelStatusProps {
  status: 'online' | 'offline' | 'busy' | 'idle';
  showLabel?: boolean;
}

export const PixelStatus: React.FC<PixelStatusProps> = ({ status, showLabel = false }) => {
  const statusConfig = {
    online: { color: '#00ff88', label: 'ONLINE' },
    offline: { color: '#ff4757', label: 'OFFLINE' },
    busy: { color: '#ffd93d', label: 'BUSY' },
    idle: { color: '#888888', label: 'IDLE' },
  };

  const config = statusConfig[status];

  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
      <span
        style={{
          width: 8,
          height: 8,
          backgroundColor: config.color,
          boxShadow: `0 0 4px ${config.color}`,
          animation: status === 'online' ? 'pixel-blink 2s step-end infinite' : undefined,
        }}
      />
      {showLabel && (
        <Text style={{ fontFamily: "'VT323', monospace", fontSize: 14, color: config.color }}>
          {config.label}
        </Text>
      )}
    </span>
  );
};

// Pixel-style progress bar
interface PixelProgressProps {
  percent: number;
  color?: string;
  height?: number;
  showLabel?: boolean;
}

export const PixelProgress: React.FC<PixelProgressProps> = ({
  percent,
  color = '#00ff88',
  height = 16,
  showLabel = true,
}) => {
  const segments = 10;
  const filledSegments = Math.round((percent / 100) * segments);

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div
        style={{
          display: 'flex',
          gap: 2,
          padding: 2,
          backgroundColor: '#0f0f23',
          border: '2px solid #333366',
        }}
      >
        {Array.from({ length: segments }).map((_, i) => (
          <div
            key={i}
            style={{
              width: height - 4,
              height: height - 4,
              backgroundColor: i < filledSegments ? color : '#1a1a2e',
              boxShadow: i < filledSegments ? `0 0 4px ${color}` : undefined,
            }}
          />
        ))}
      </div>
      {showLabel && (
        <Text style={{ fontFamily: "'VT323', monospace", fontSize: 16, color }}>
          {percent}%
        </Text>
      )}
    </div>
  );
};

// Pixel-style loading spinner
export const PixelSpinner: React.FC<{ size?: number }> = ({ size = 32 }) => {
  return (
    <div
      style={{
        width: size,
        height: size,
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gridTemplateRows: 'repeat(3, 1fr)',
        gap: 2,
      }}
    >
      {[0, 1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
        <div
          key={i}
          style={{
            backgroundColor: '#00ff88',
            animation: 'pixel-blink 1s step-end infinite',
            animationDelay: `${i * 0.1}s`,
          }}
        />
      ))}
    </div>
  );
};

export default PixelBadge;
