/**
 * Pixel Art Icons for AI Code Review Platform
 * Cool, retro-style icons with a modern twist
 */

import React from 'react';

interface IconProps {
  size?: number;
  className?: string;
}

// Pixel Art Code Review Icon
export const PixelCodeIcon: React.FC<IconProps> = ({ size = 32, className }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" className={className} style={{ imageRendering: 'pixelated' }}>
    <rect x="4" y="4" width="24" height="24" fill="#ecfeff" stroke="#06b6d4" strokeWidth="2"/>
    <rect x="8" y="10" width="8" height="2" fill="#06b6d4"/>
    <rect x="10" y="14" width="10" height="2" fill="#6366f1"/>
    <rect x="8" y="18" width="12" height="2" fill="#14b8a6"/>
    <rect x="10" y="22" width="6" height="2" fill="#06b6d4"/>
    <rect x="20" y="8" width="4" height="4" fill="#22c55e"/>
  </svg>
);

// Pixel Art Bug/Issue Icon
export const PixelBugIcon: React.FC<IconProps> = ({ size = 32, className }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" className={className} style={{ imageRendering: 'pixelated' }}>
    <ellipse cx="16" cy="18" rx="8" ry="10" fill="#fef2f2" stroke="#ef4444" strokeWidth="2"/>
    <circle cx="16" cy="8" r="4" fill="#fef2f2" stroke="#ef4444" strokeWidth="2"/>
    <rect x="4" y="12" width="4" height="2" fill="#ef4444"/>
    <rect x="24" y="12" width="4" height="2" fill="#ef4444"/>
    <rect x="4" y="18" width="4" height="2" fill="#ef4444"/>
    <rect x="24" y="18" width="4" height="2" fill="#ef4444"/>
    <rect x="4" y="24" width="4" height="2" fill="#ef4444"/>
    <rect x="24" y="24" width="4" height="2" fill="#ef4444"/>
    <rect x="12" y="14" width="2" height="2" fill="#ef4444"/>
    <rect x="18" y="14" width="2" height="2" fill="#ef4444"/>
  </svg>
);

// Pixel Art Shield/Security Icon
export const PixelShieldIcon: React.FC<IconProps> = ({ size = 32, className }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" className={className} style={{ imageRendering: 'pixelated' }}>
    <path d="M16 4 L4 10 L4 18 C4 24 10 28 16 30 C22 28 28 24 28 18 L28 10 Z" 
          fill="#ecfeff" stroke="#06b6d4" strokeWidth="2"/>
    <rect x="14" y="12" width="4" height="8" fill="#22c55e"/>
    <rect x="10" y="16" width="12" height="4" fill="#22c55e"/>
  </svg>
);

// Pixel Art Project/Folder Icon
export const PixelFolderIcon: React.FC<IconProps> = ({ size = 32, className }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" className={className} style={{ imageRendering: 'pixelated' }}>
    <path d="M4 8 L4 26 L28 26 L28 12 L16 12 L14 8 Z" 
          fill="#fef3c7" stroke="#f59e0b" strokeWidth="2"/>
    <rect x="4" y="12" width="24" height="2" fill="#f59e0b"/>
  </svg>
);

// Pixel Art Dashboard/Grid Icon
export const PixelDashboardIcon: React.FC<IconProps> = ({ size = 32, className }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" className={className} style={{ imageRendering: 'pixelated' }}>
    <rect x="4" y="4" width="10" height="10" fill="#ecfeff" stroke="#06b6d4" strokeWidth="2"/>
    <rect x="18" y="4" width="10" height="10" fill="#e0e7ff" stroke="#6366f1" strokeWidth="2"/>
    <rect x="4" y="18" width="10" height="10" fill="#ccfbf1" stroke="#14b8a6" strokeWidth="2"/>
    <rect x="18" y="18" width="10" height="10" fill="#f0fdf4" stroke="#22c55e" strokeWidth="2"/>
  </svg>
);

// Pixel Art AI Robot Icon
export const PixelRobotIcon: React.FC<IconProps> = ({ size = 32, className }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" className={className} style={{ imageRendering: 'pixelated' }}>
    <rect x="8" y="8" width="16" height="16" fill="#ecfeff" stroke="#06b6d4" strokeWidth="2"/>
    <rect x="6" y="4" width="4" height="4" fill="#6366f1"/>
    <rect x="22" y="4" width="4" height="4" fill="#6366f1"/>
    <rect x="12" y="12" width="4" height="4" fill="#06b6d4"/>
    <rect x="20" y="12" width="4" height="4" fill="#06b6d4"/>
    <rect x="10" y="20" width="12" height="2" fill="#14b8a6"/>
    <rect x="4" y="14" width="4" height="6" fill="#c7d2fe"/>
    <rect x="24" y="14" width="4" height="6" fill="#c7d2fe"/>
  </svg>
);

// Pixel Art Chart/Analytics Icon
export const PixelChartIcon: React.FC<IconProps> = ({ size = 32, className }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" className={className} style={{ imageRendering: 'pixelated' }}>
    <rect x="4" y="4" width="24" height="24" fill="#f8fafc" stroke="#94a3b8" strokeWidth="2"/>
    <rect x="8" y="18" width="4" height="8" fill="#06b6d4"/>
    <rect x="14" y="12" width="4" height="14" fill="#6366f1"/>
    <rect x="20" y="8" width="4" height="18" fill="#14b8a6"/>
  </svg>
);

// Pixel Art Team/Users Icon
export const PixelTeamIcon: React.FC<IconProps> = ({ size = 32, className }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" className={className} style={{ imageRendering: 'pixelated' }}>
    <circle cx="16" cy="10" r="4" fill="#ecfeff" stroke="#06b6d4" strokeWidth="2"/>
    <rect x="10" y="16" width="12" height="12" rx="2" fill="#ecfeff" stroke="#06b6d4" strokeWidth="2"/>
    <circle cx="6" cy="12" r="3" fill="#e0e7ff" stroke="#6366f1" strokeWidth="1"/>
    <rect x="3" y="17" width="6" height="8" fill="#e0e7ff" stroke="#6366f1" strokeWidth="1"/>
    <circle cx="26" cy="12" r="3" fill="#e0e7ff" stroke="#6366f1" strokeWidth="1"/>
    <rect x="23" y="17" width="6" height="8" fill="#e0e7ff" stroke="#6366f1" strokeWidth="1"/>
  </svg>
);

// Pixel Art Settings/Gear Icon
export const PixelSettingsIcon: React.FC<IconProps> = ({ size = 32, className }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" className={className} style={{ imageRendering: 'pixelated' }}>
    <circle cx="16" cy="16" r="6" fill="#ecfeff" stroke="#06b6d4" strokeWidth="2"/>
    <rect x="14" y="2" width="4" height="6" fill="#6366f1"/>
    <rect x="14" y="24" width="4" height="6" fill="#6366f1"/>
    <rect x="2" y="14" width="6" height="4" fill="#6366f1"/>
    <rect x="24" y="14" width="6" height="4" fill="#6366f1"/>
    <rect x="5" y="5" width="4" height="4" fill="#14b8a6" transform="rotate(45 7 7)"/>
    <rect x="23" y="5" width="4" height="4" fill="#14b8a6" transform="rotate(45 25 7)"/>
    <rect x="5" y="23" width="4" height="4" fill="#14b8a6" transform="rotate(45 7 25)"/>
    <rect x="23" y="23" width="4" height="4" fill="#14b8a6" transform="rotate(45 25 25)"/>
  </svg>
);

// Pixel Art Rocket/Deploy Icon
export const PixelRocketIcon: React.FC<IconProps> = ({ size = 32, className }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" className={className} style={{ imageRendering: 'pixelated' }}>
    <path d="M16 4 L20 12 L20 20 L16 28 L12 20 L12 12 Z" fill="#ecfeff" stroke="#06b6d4" strokeWidth="2"/>
    <rect x="14" y="8" width="4" height="4" fill="#6366f1"/>
    <polygon points="8,22 12,18 12,22" fill="#f59e0b"/>
    <polygon points="24,22 20,18 20,22" fill="#f59e0b"/>
    <rect x="14" y="24" width="4" height="4" fill="#ef4444"/>
    <rect x="12" y="26" width="2" height="4" fill="#fbbf24"/>
    <rect x="18" y="26" width="2" height="4" fill="#fbbf24"/>
  </svg>
);

// Pixel Art Check/Success Icon
export const PixelCheckIcon: React.FC<IconProps> = ({ size = 32, className }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" className={className} style={{ imageRendering: 'pixelated' }}>
    <circle cx="16" cy="16" r="12" fill="#f0fdf4" stroke="#22c55e" strokeWidth="2"/>
    <path d="M10 16 L14 20 L22 12" stroke="#22c55e" strokeWidth="3" fill="none"/>
  </svg>
);

// Pixel Art Warning Icon
export const PixelWarningIcon: React.FC<IconProps> = ({ size = 32, className }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" className={className} style={{ imageRendering: 'pixelated' }}>
    <path d="M16 4 L28 28 L4 28 Z" fill="#fffbeb" stroke="#f59e0b" strokeWidth="2"/>
    <rect x="14" y="12" width="4" height="8" fill="#f59e0b"/>
    <rect x="14" y="22" width="4" height="4" fill="#f59e0b"/>
  </svg>
);

// Pixel Art Error Icon
export const PixelErrorIcon: React.FC<IconProps> = ({ size = 32, className }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" className={className} style={{ imageRendering: 'pixelated' }}>
    <circle cx="16" cy="16" r="12" fill="#fef2f2" stroke="#ef4444" strokeWidth="2"/>
    <path d="M10 10 L22 22 M22 10 L10 22" stroke="#ef4444" strokeWidth="3"/>
  </svg>
);

// Pixel Art Clock/Time Icon
export const PixelClockIcon: React.FC<IconProps> = ({ size = 32, className }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" className={className} style={{ imageRendering: 'pixelated' }}>
    <circle cx="16" cy="16" r="12" fill="#ecfeff" stroke="#06b6d4" strokeWidth="2"/>
    <rect x="15" y="8" width="2" height="10" fill="#06b6d4"/>
    <rect x="15" y="15" width="8" height="2" fill="#6366f1"/>
    <rect x="15" y="15" width="2" height="2" fill="#14b8a6"/>
  </svg>
);

// Pixel Art Star/Favorite Icon
export const PixelStarIcon: React.FC<IconProps> = ({ size = 32, className }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" className={className} style={{ imageRendering: 'pixelated' }}>
    <path d="M16 4 L19 12 L28 12 L21 18 L24 28 L16 22 L8 28 L11 18 L4 12 L13 12 Z" 
          fill="#fef3c7" stroke="#f59e0b" strokeWidth="2"/>
  </svg>
);

// Pixel Art Lock/Security Icon
export const PixelLockIcon: React.FC<IconProps> = ({ size = 32, className }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" className={className} style={{ imageRendering: 'pixelated' }}>
    <rect x="8" y="14" width="16" height="14" fill="#ecfeff" stroke="#06b6d4" strokeWidth="2"/>
    <path d="M10 14 L10 10 C10 6 13 4 16 4 C19 4 22 6 22 10 L22 14" 
          fill="none" stroke="#6366f1" strokeWidth="2"/>
    <rect x="14" y="18" width="4" height="6" fill="#14b8a6"/>
  </svg>
);

// Main App Logo - Pixel Art Style
export const PixelAppLogo: React.FC<IconProps> = ({ size = 48, className }) => (
  <svg width={size} height={size} viewBox="0 0 48 48" className={className} style={{ imageRendering: 'pixelated' }}>
    {/* Background */}
    <rect x="4" y="4" width="40" height="40" rx="8" fill="url(#logoGradient)"/>
    
    {/* Code brackets */}
    <path d="M14 16 L10 24 L14 32" stroke="white" strokeWidth="3" fill="none" strokeLinecap="square"/>
    <path d="M34 16 L38 24 L34 32" stroke="white" strokeWidth="3" fill="none" strokeLinecap="square"/>
    
    {/* AI Eye */}
    <circle cx="24" cy="24" r="6" fill="white"/>
    <circle cx="24" cy="24" r="3" fill="#06b6d4"/>
    <rect x="22" y="22" width="2" height="2" fill="white"/>
    
    {/* Pixel dots */}
    <rect x="18" y="12" width="2" height="2" fill="rgba(255,255,255,0.6)"/>
    <rect x="28" y="12" width="2" height="2" fill="rgba(255,255,255,0.6)"/>
    <rect x="18" y="34" width="2" height="2" fill="rgba(255,255,255,0.6)"/>
    <rect x="28" y="34" width="2" height="2" fill="rgba(255,255,255,0.6)"/>
    
    <defs>
      <linearGradient id="logoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#06b6d4"/>
        <stop offset="100%" stopColor="#6366f1"/>
      </linearGradient>
    </defs>
  </svg>
);

export default {
  PixelCodeIcon,
  PixelBugIcon,
  PixelShieldIcon,
  PixelFolderIcon,
  PixelDashboardIcon,
  PixelRobotIcon,
  PixelChartIcon,
  PixelTeamIcon,
  PixelSettingsIcon,
  PixelRocketIcon,
  PixelCheckIcon,
  PixelWarningIcon,
  PixelErrorIcon,
  PixelClockIcon,
  PixelStarIcon,
  PixelLockIcon,
  PixelAppLogo,
};
