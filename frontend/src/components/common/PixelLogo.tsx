/**
 * Pixel Logo Component
 * 
 * 8-bit style logo for the application.
 */

import React from 'react';
import { Typography } from 'antd';

const { Text } = Typography;

interface PixelLogoProps {
  collapsed?: boolean;
  size?: 'small' | 'medium' | 'large';
}

export const PixelLogo: React.FC<PixelLogoProps> = ({ collapsed = false, size = 'medium' }) => {
  const sizeConfig = {
    small: { icon: 24, fontSize: 10, gap: 6 },
    medium: { icon: 32, fontSize: 12, gap: 8 },
    large: { icon: 48, fontSize: 16, gap: 12 },
  };

  const config = sizeConfig[size];

  // Simple pixel art code icon
  const PixelIcon = () => (
    <svg
      width={config.icon}
      height={config.icon}
      viewBox="0 0 16 16"
      style={{ imageRendering: 'pixelated' }}
    >
      {/* Background */}
      <rect x="2" y="2" width="12" height="12" fill="#1a1a2e" />
      {/* Border */}
      <rect x="1" y="1" width="14" height="1" fill="#00ff88" />
      <rect x="1" y="14" width="14" height="1" fill="#00ff88" />
      <rect x="1" y="1" width="1" height="14" fill="#00ff88" />
      <rect x="14" y="1" width="1" height="14" fill="#00ff88" />
      {/* Code brackets */}
      <rect x="4" y="5" width="1" height="1" fill="#00ff88" />
      <rect x="3" y="6" width="1" height="4" fill="#00ff88" />
      <rect x="4" y="10" width="1" height="1" fill="#00ff88" />
      <rect x="11" y="5" width="1" height="1" fill="#00ff88" />
      <rect x="12" y="6" width="1" height="4" fill="#00ff88" />
      <rect x="11" y="10" width="1" height="1" fill="#00ff88" />
      {/* AI dots */}
      <rect x="6" y="7" width="2" height="2" fill="#ff6b6b" />
      <rect x="9" y="7" width="1" height="2" fill="#4ecdc4" />
    </svg>
  );

  if (collapsed) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center',
        padding: '12px 0',
      }}>
        <PixelIcon />
      </div>
    );
  }

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: config.gap,
        padding: '12px 16px',
      }}
    >
      <PixelIcon />
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        <Text
          style={{
            fontFamily: "'Press Start 2P', cursive",
            fontSize: config.fontSize,
            color: '#00ff88',
            textShadow: '2px 2px 0 #006633',
            lineHeight: 1.2,
          }}
        >
          CODE
        </Text>
        <Text
          style={{
            fontFamily: "'Press Start 2P', cursive",
            fontSize: config.fontSize * 0.8,
            color: '#4ecdc4',
            lineHeight: 1.2,
          }}
        >
          REVIEW AI
        </Text>
      </div>
    </div>
  );
};

// ASCII art version for terminal/console style
export const ASCIILogo: React.FC = () => {
  const asciiArt = `
╔═══════════════════════╗
║  ▄▄▄▄▄▄▄  CODE  ▄▄▄▄▄▄▄║
║  █░░░░░█ REVIEW █░░░░░█║
║  █░▄▄▄░█   AI   █░▄▄▄░█║
║  █░░░░░█ ═════ █░░░░░█║
║  ▀▀▀▀▀▀▀  v2.0  ▀▀▀▀▀▀▀║
╚═══════════════════════╝
  `;

  return (
    <pre
      style={{
        fontFamily: "'VT323', monospace",
        fontSize: 10,
        color: '#00ff88',
        lineHeight: 1,
        margin: 0,
        textShadow: '0 0 4px #00ff88',
      }}
    >
      {asciiArt}
    </pre>
  );
};

export default PixelLogo;
