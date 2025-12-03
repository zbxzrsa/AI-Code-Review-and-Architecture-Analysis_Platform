/**
 * Artistic UI Widgets
 * 
 * Beautiful, reusable components with soothing aesthetics:
 * - Glassmorphism cards
 * - Gradient stats
 * - Animated progress indicators
 * - Artistic data visualizations
 */

import React from 'react';
import { Card, Typography, Space } from 'antd';
import {
  RiseOutlined,
  FallOutlined,
  ClockCircleOutlined,
} from '@ant-design/icons';

const { Text } = Typography;

// ============================================
// Glassmorphism Card
// ============================================
interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
  hover?: boolean;
}

export const GlassCard: React.FC<GlassCardProps> = ({ 
  children, 
  className = '', 
  style = {},
  hover = true 
}) => (
  <div
    className={`glass-card ${hover ? 'glass-card-hover' : ''} ${className}`}
    style={{
      background: 'rgba(255, 255, 255, 0.7)',
      backdropFilter: 'blur(10px)',
      WebkitBackdropFilter: 'blur(10px)',
      borderRadius: 16,
      border: '1px solid rgba(255, 255, 255, 0.3)',
      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
      padding: 24,
      transition: 'all 0.3s ease',
      ...style,
    }}
  >
    {children}
  </div>
);

// ============================================
// Gradient Stat Card
// ============================================
interface GradientStatProps {
  title: string;
  value: number | string;
  suffix?: string;
  prefix?: React.ReactNode;
  trend?: number;
  gradient?: 'ocean' | 'sunset' | 'aurora' | 'mint';
  icon?: React.ReactNode;
}

const gradients = {
  ocean: 'linear-gradient(135deg, #0091c3 0%, #00c9a7 100%)',
  sunset: 'linear-gradient(135deg, #ff9b33 0%, #ff6b6b 100%)',
  aurora: 'linear-gradient(135deg, #a855f7 0%, #3b82f6 50%, #10b981 100%)',
  mint: 'linear-gradient(135deg, #10b981 0%, #34d399 100%)',
};

export const GradientStat: React.FC<GradientStatProps> = ({
  title,
  value,
  suffix,
  prefix,
  trend,
  gradient = 'ocean',
  icon,
}) => (
  <Card
    style={{
      borderRadius: 16,
      border: 'none',
      boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
      overflow: 'hidden',
      position: 'relative',
    }}
    bodyStyle={{ padding: 24 }}
  >
    {/* Decorative background circle */}
    <div
      style={{
        position: 'absolute',
        top: -30,
        right: -30,
        width: 120,
        height: 120,
        background: gradients[gradient],
        opacity: 0.1,
        borderRadius: '50%',
      }}
    />
    
    <Space direction="vertical" size={8} style={{ width: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Text type="secondary" style={{ fontSize: 13, textTransform: 'uppercase', letterSpacing: 1 }}>
          {title}
        </Text>
        {icon && (
          <div
            style={{
              width: 40,
              height: 40,
              borderRadius: 12,
              background: gradients[gradient],
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'white',
              fontSize: 18,
            }}
          >
            {icon}
          </div>
        )}
      </div>
      
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
        {prefix}
        <span
          style={{
            fontSize: 32,
            fontWeight: 700,
            background: gradients[gradient],
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
          }}
        >
          {value}
        </span>
        {suffix && <Text type="secondary">{suffix}</Text>}
      </div>
      
      {trend !== undefined && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          {trend >= 0 ? (
            <RiseOutlined style={{ color: '#10b981' }} />
          ) : (
            <FallOutlined style={{ color: '#ef4444' }} />
          )}
          <Text style={{ color: trend >= 0 ? '#10b981' : '#ef4444', fontSize: 13 }}>
            {trend >= 0 ? '+' : ''}{trend}%
          </Text>
          <Text type="secondary" style={{ fontSize: 12 }}>vs last period</Text>
        </div>
      )}
    </Space>
  </Card>
);

// ============================================
// Artistic Progress Ring
// ============================================
interface ProgressRingProps {
  percent: number;
  size?: number;
  strokeWidth?: number;
  gradient?: 'ocean' | 'sunset' | 'aurora' | 'mint';
  label?: string;
  sublabel?: string;
}

export const ProgressRing: React.FC<ProgressRingProps> = ({
  percent,
  size = 120,
  strokeWidth = 8,
  gradient = 'ocean',
  label,
  sublabel,
}) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (percent / 100) * circumference;

  const gradientId = `progress-gradient-${gradient}-${Math.random().toString(36).substr(2, 9)}`;

  return (
    <div style={{ position: 'relative', width: size, height: size }}>
      <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
        <defs>
          <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="100%">
            {gradient === 'ocean' && (
              <>
                <stop offset="0%" stopColor="#0091c3" />
                <stop offset="100%" stopColor="#00c9a7" />
              </>
            )}
            {gradient === 'sunset' && (
              <>
                <stop offset="0%" stopColor="#ff9b33" />
                <stop offset="100%" stopColor="#ff6b6b" />
              </>
            )}
            {gradient === 'aurora' && (
              <>
                <stop offset="0%" stopColor="#a855f7" />
                <stop offset="50%" stopColor="#3b82f6" />
                <stop offset="100%" stopColor="#10b981" />
              </>
            )}
            {gradient === 'mint' && (
              <>
                <stop offset="0%" stopColor="#10b981" />
                <stop offset="100%" stopColor="#34d399" />
              </>
            )}
          </linearGradient>
        </defs>
        
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#f0f0f0"
          strokeWidth={strokeWidth}
        />
        
        {/* Progress circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={`url(#${gradientId})`}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          style={{ transition: 'stroke-dashoffset 0.5s ease' }}
        />
      </svg>
      
      {/* Center text */}
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          textAlign: 'center',
        }}
      >
        <div style={{ fontSize: size / 4, fontWeight: 700, color: '#1f2937' }}>
          {percent}%
        </div>
        {label && (
          <div style={{ fontSize: 12, color: '#6b7280' }}>{label}</div>
        )}
        {sublabel && (
          <div style={{ fontSize: 10, color: '#9ca3af' }}>{sublabel}</div>
        )}
      </div>
    </div>
  );
};

// ============================================
// Artistic Status Badge
// ============================================
interface StatusBadgeProps {
  status: 'success' | 'warning' | 'error' | 'info' | 'processing';
  text: string;
  animated?: boolean;
}

const statusColors = {
  success: { bg: '#d1fae5', color: '#065f46', dot: '#10b981' },
  warning: { bg: '#fef3c7', color: '#92400e', dot: '#f59e0b' },
  error: { bg: '#fee2e2', color: '#991b1b', dot: '#ef4444' },
  info: { bg: '#dbeafe', color: '#1e40af', dot: '#3b82f6' },
  processing: { bg: '#e0e7ff', color: '#3730a3', dot: '#6366f1' },
};

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, text, animated }) => (
  <span
    style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 6,
      padding: '4px 12px',
      borderRadius: 9999,
      background: statusColors[status].bg,
      color: statusColors[status].color,
      fontSize: 12,
      fontWeight: 500,
    }}
  >
    <span
      style={{
        width: 6,
        height: 6,
        borderRadius: '50%',
        background: statusColors[status].dot,
        animation: animated ? 'pulse 2s infinite' : 'none',
      }}
    />
    {text}
  </span>
);

// ============================================
// Artistic Timeline Item
// ============================================
interface TimelineItemProps {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  time?: string;
  status?: 'success' | 'warning' | 'error' | 'info';
}

export const ArtisticTimeline: React.FC<{ items: TimelineItemProps[] }> = ({ items }) => (
  <div style={{ position: 'relative' }}>
    {items.map((item, index) => (
      <div
        key={index}
        style={{
          display: 'flex',
          gap: 16,
          paddingBottom: index < items.length - 1 ? 24 : 0,
          position: 'relative',
        }}
      >
        {/* Connector line */}
        {index < items.length - 1 && (
          <div
            style={{
              position: 'absolute',
              left: 15,
              top: 32,
              width: 2,
              height: 'calc(100% - 8px)',
              background: 'linear-gradient(to bottom, #e5e7eb, transparent)',
            }}
          />
        )}
        
        {/* Icon */}
        <div
          style={{
            width: 32,
            height: 32,
            borderRadius: 8,
            background: item.status ? statusColors[item.status].bg : '#f3f4f6',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: item.status ? statusColors[item.status].color : '#6b7280',
            flexShrink: 0,
          }}
        >
          {item.icon || <ClockCircleOutlined />}
        </div>
        
        {/* Content */}
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: 500, color: '#1f2937' }}>{item.title}</div>
          {item.description && (
            <div style={{ fontSize: 13, color: '#6b7280', marginTop: 2 }}>
              {item.description}
            </div>
          )}
          {item.time && (
            <div style={{ fontSize: 12, color: '#9ca3af', marginTop: 4 }}>
              {item.time}
            </div>
          )}
        </div>
      </div>
    ))}
  </div>
);

// ============================================
// Artistic Metric Bar
// ============================================
interface MetricBarProps {
  label: string;
  value: number;
  max?: number;
  gradient?: 'ocean' | 'sunset' | 'aurora' | 'mint';
  showValue?: boolean;
}

export const MetricBar: React.FC<MetricBarProps> = ({
  label,
  value,
  max = 100,
  gradient = 'ocean',
  showValue = true,
}) => {
  const percent = Math.min((value / max) * 100, 100);
  
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
        <Text style={{ fontSize: 13, color: '#4b5563' }}>{label}</Text>
        {showValue && (
          <Text style={{ fontSize: 13, fontWeight: 500 }}>
            {value}{max !== 100 ? `/${max}` : '%'}
          </Text>
        )}
      </div>
      <div
        style={{
          height: 8,
          background: '#f3f4f6',
          borderRadius: 9999,
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            height: '100%',
            width: `${percent}%`,
            background: gradients[gradient],
            borderRadius: 9999,
            transition: 'width 0.5s ease',
          }}
        />
      </div>
    </div>
  );
};

export default {
  GlassCard,
  GradientStat,
  ProgressRing,
  StatusBadge,
  ArtisticTimeline,
  MetricBar,
};
