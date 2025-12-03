import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { Modal, Input, List, Tag, Typography, Space } from 'antd';
import {
  SearchOutlined,
  FileOutlined,
  SettingOutlined,
  UserOutlined,
  ProjectOutlined,
  BugOutlined,
  ExperimentOutlined,
  DashboardOutlined,
  LogoutOutlined
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useUIStore } from '../../../store/uiStore';
import { useAuth } from '../../../hooks/useAuth';
import './CommandPalette.css';

const { Text } = Typography;

// Simple keyboard shortcut display component
const Kbd: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <span style={{
    display: 'inline-block',
    padding: '2px 6px',
    fontSize: '11px',
    fontFamily: 'monospace',
    lineHeight: '1.4',
    color: '#666',
    backgroundColor: '#f5f5f5',
    border: '1px solid #d9d9d9',
    borderRadius: '3px',
    boxShadow: '0 1px 0 rgba(0,0,0,0.1)',
  }}>
    {children}
  </span>
);

interface Command {
  id: string;
  title: string;
  description?: string;
  icon: React.ReactNode;
  shortcut?: string[];
  category: 'navigation' | 'action' | 'settings';
  action: () => void;
  keywords?: string[];
}

export const CommandPalette: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { isAdmin, logout } = useAuth();
  const { isCommandPaletteOpen, closeCommandPalette, setTheme } = useUIStore();
  
  const [search, setSearch] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);

  // Define commands
  const commands: Command[] = useMemo(() => {
    const baseCommands: Command[] = [
      // Navigation
      {
        id: 'go-dashboard',
        title: t('commands.go_dashboard', 'Go to Dashboard'),
        icon: <DashboardOutlined />,
        shortcut: ['g', 'd'],
        category: 'navigation',
        action: () => navigate('/dashboard'),
        keywords: ['home', 'main']
      },
      {
        id: 'go-projects',
        title: t('commands.go_projects', 'Go to Projects'),
        icon: <ProjectOutlined />,
        shortcut: ['g', 'p'],
        category: 'navigation',
        action: () => navigate('/projects'),
        keywords: ['list', 'all']
      },
      {
        id: 'go-review',
        title: t('commands.go_review', 'Go to Code Review'),
        icon: <BugOutlined />,
        shortcut: ['g', 'r'],
        category: 'navigation',
        action: () => navigate('/review'),
        keywords: ['analyze', 'code']
      },
      {
        id: 'go-settings',
        title: t('commands.go_settings', 'Go to Settings'),
        icon: <SettingOutlined />,
        shortcut: ['g', 's'],
        category: 'navigation',
        action: () => navigate('/settings'),
        keywords: ['preferences', 'config']
      },
      {
        id: 'go-profile',
        title: t('commands.go_profile', 'Go to Profile'),
        icon: <UserOutlined />,
        shortcut: ['g', 'u'],
        category: 'navigation',
        action: () => navigate('/profile'),
        keywords: ['account', 'user']
      },
      
      // Actions
      {
        id: 'new-project',
        title: t('commands.new_project', 'Create New Project'),
        icon: <ProjectOutlined />,
        category: 'action',
        action: () => navigate('/projects/new'),
        keywords: ['add', 'create']
      },
      {
        id: 'start-analysis',
        title: t('commands.start_analysis', 'Start New Analysis'),
        icon: <BugOutlined />,
        category: 'action',
        action: () => navigate('/review/new'),
        keywords: ['analyze', 'scan']
      },
      
      // Settings
      {
        id: 'theme-light',
        title: t('commands.theme_light', 'Switch to Light Theme'),
        icon: <SettingOutlined />,
        category: 'settings',
        action: () => setTheme('light'),
        keywords: ['light', 'white', 'day']
      },
      {
        id: 'theme-dark',
        title: t('commands.theme_dark', 'Switch to Dark Theme'),
        icon: <SettingOutlined />,
        category: 'settings',
        action: () => setTheme('dark'),
        keywords: ['dark', 'black', 'night']
      },
      {
        id: 'theme-system',
        title: t('commands.theme_system', 'Use System Theme'),
        icon: <SettingOutlined />,
        category: 'settings',
        action: () => setTheme('system'),
        keywords: ['auto', 'system']
      },
      
      // Account
      {
        id: 'logout',
        title: t('commands.logout', 'Sign Out'),
        icon: <LogoutOutlined />,
        category: 'action',
        action: () => logout(),
        keywords: ['signout', 'exit']
      }
    ];

    // Admin-only commands
    if (isAdmin()) {
      baseCommands.push(
        {
          id: 'go-experiments',
          title: t('commands.go_experiments', 'Go to Experiments'),
          icon: <ExperimentOutlined />,
          shortcut: ['g', 'e'],
          category: 'navigation',
          action: () => navigate('/admin/experiments'),
          keywords: ['v1', 'test']
        },
        {
          id: 'go-audit',
          title: t('commands.go_audit', 'Go to Audit Log'),
          icon: <FileOutlined />,
          shortcut: ['g', 'a'],
          category: 'navigation',
          action: () => navigate('/admin/audit'),
          keywords: ['logs', 'history']
        }
      );
    }

    return baseCommands;
  }, [t, navigate, isAdmin, setTheme, logout]);

  // Filter commands based on search
  const filteredCommands = useMemo(() => {
    if (!search) return commands;
    
    const searchLower = search.toLowerCase();
    return commands.filter((cmd) => {
      return (
        cmd.title.toLowerCase().includes(searchLower) ||
        cmd.description?.toLowerCase().includes(searchLower) ||
        cmd.keywords?.some((kw) => kw.includes(searchLower))
      );
    });
  }, [commands, search]);

  // Reset selection when filtered commands change
  useEffect(() => {
    setSelectedIndex(0);
  }, [filteredCommands]);

  // Execute selected command
  const executeCommand = useCallback((command: Command) => {
    closeCommandPalette();
    setSearch('');
    command.action();
  }, [closeCommandPalette]);

  // Keyboard navigation
  useEffect(() => {
    if (!isCommandPaletteOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          setSelectedIndex((prev) => 
            prev < filteredCommands.length - 1 ? prev + 1 : 0
          );
          break;
        case 'ArrowUp':
          e.preventDefault();
          setSelectedIndex((prev) => 
            prev > 0 ? prev - 1 : filteredCommands.length - 1
          );
          break;
        case 'Enter':
          e.preventDefault();
          if (filteredCommands[selectedIndex]) {
            executeCommand(filteredCommands[selectedIndex]);
          }
          break;
        case 'Escape':
          closeCommandPalette();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isCommandPaletteOpen, filteredCommands, selectedIndex, executeCommand, closeCommandPalette]);

  // Global keyboard shortcut to open
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        useUIStore.getState().toggleCommandPalette();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const getCategoryLabel = (category: string) => {
    switch (category) {
      case 'navigation':
        return t('commands.category_navigation', 'Navigation');
      case 'action':
        return t('commands.category_action', 'Actions');
      case 'settings':
        return t('commands.category_settings', 'Settings');
      default:
        return category;
    }
  };

  return (
    <Modal
      open={isCommandPaletteOpen}
      onCancel={closeCommandPalette}
      footer={null}
      closable={false}
      centered
      width={600}
      className="command-palette-modal"
      maskClosable
    >
      <div className="command-palette">
        <div className="command-palette-input">
          <Input
            prefix={<SearchOutlined />}
            placeholder={t('commands.placeholder', 'Type a command or search...')}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            autoFocus
            size="large"
            bordered={false}
          />
        </div>
        
        <div className="command-palette-list">
          {filteredCommands.length === 0 ? (
            <div className="command-palette-empty">
              <Text type="secondary">
                {t('commands.no_results', 'No commands found')}
              </Text>
            </div>
          ) : (
            <List
              dataSource={filteredCommands}
              renderItem={(cmd, index) => (
                <List.Item
                  className={`command-item ${index === selectedIndex ? 'selected' : ''}`}
                  onClick={() => executeCommand(cmd)}
                >
                  <Space>
                    <span className="command-icon">{cmd.icon}</span>
                    <div className="command-content">
                      <Text strong>{cmd.title}</Text>
                      {cmd.description && (
                        <Text type="secondary" className="command-description">
                          {cmd.description}
                        </Text>
                      )}
                    </div>
                  </Space>
                  <Space>
                    <Tag>{getCategoryLabel(cmd.category)}</Tag>
                    {cmd.shortcut && (
                      <span className="command-shortcut">
                        {cmd.shortcut.map((key, i) => (
                          <kbd key={i}>{key}</kbd>
                        ))}
                      </span>
                    )}
                  </Space>
                </List.Item>
              )}
            />
          )}
        </div>
        
        <div className="command-palette-footer">
          <Space>
            <span><kbd>↑↓</kbd> {t('commands.navigate', 'Navigate')}</span>
            <span><kbd>↵</kbd> {t('commands.select', 'Select')}</span>
            <span><kbd>esc</kbd> {t('commands.close', 'Close')}</span>
          </Space>
        </div>
      </div>
    </Modal>
  );
};

export default CommandPalette;
