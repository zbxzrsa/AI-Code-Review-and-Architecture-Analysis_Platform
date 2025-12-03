import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useSearchParams } from 'react-router-dom';
import {
  Layout,
  Button,
  Space,
  Typography,
  Tag,
  List,
  Tooltip,
  Spin,
  Empty,
  Dropdown,
  Badge,
  Segmented
} from 'antd';
import {
  PlayCircleOutlined,
  BugOutlined,
  WarningOutlined,
  InfoCircleOutlined,
  CheckCircleOutlined,
  DownloadOutlined,
  ReloadOutlined
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { CodeEditor, Issue } from '../components/code/CodeEditor';
import { StreamingResponse, AnalysisResult } from '../components/chat/StreamingResponse';
import { apiService } from '../services/api';
import { useProjectStore } from '../store/projectStore';
import './CodeReview.css';

const { Sider, Content } = Layout;
const { Title, Text } = Typography;

type SeverityFilter = 'all' | 'error' | 'warning' | 'info' | 'hint';

export const CodeReview: React.FC = () => {
  const { t } = useTranslation();
  const { projectId } = useParams<{ projectId: string }>();
  const [searchParams] = useSearchParams();
  const filePath = searchParams.get('file');

  const { selectedFile, setSelectedFile, currentSession, setCurrentSession } = useProjectStore();

  const [code, setCode] = useState('');
  const [issues, setIssues] = useState<Issue[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [severityFilter, setSeverityFilter] = useState<SeverityFilter>('all');
  const [showAIResponse, setShowAIResponse] = useState(false);
  const [loading, setLoading] = useState(false);

  // Load file content
  useEffect(() => {
    if (!projectId || !filePath) return;

    const loadFile = async () => {
      setLoading(true);
      try {
        const response = await apiService.projects.getFile(projectId, filePath);
        setCode(response.data.content);
        setSelectedFile({
          path: filePath,
          content: response.data.content,
          language: response.data.language || 'plaintext',
          size: response.data.size || 0,
          last_modified: response.data.last_modified || new Date().toISOString()
        });
      } catch (error) {
        console.error('Failed to load file:', error);
      } finally {
        setLoading(false);
      }
    };

    loadFile();
  }, [projectId, filePath, setSelectedFile]);

  // Start analysis
  const handleAnalyze = useCallback(async () => {
    if (!projectId) return;

    setIsAnalyzing(true);
    setShowAIResponse(true);
    setIssues([]);

    try {
      const response = await apiService.analysis.start(projectId, {
        files: filePath ? [filePath] : undefined
      });
      setCurrentSession(response.data);
    } catch (error) {
      console.error('Failed to start analysis:', error);
      setIsAnalyzing(false);
    }
  }, [projectId, filePath, setCurrentSession]);

  // Handle analysis complete
  const handleAnalysisComplete = useCallback((result: AnalysisResult) => {
    setIsAnalyzing(false);
    setIssues(result.issues.map((issue, index) => ({
      id: `issue-${index}`,
      type: issue.type,
      severity: issue.severity as Issue['severity'],
      line_start: issue.line_start,
      line_end: issue.line_end,
      description: issue.description,
      fix: issue.fix
    })));
  }, []);

  // Filter issues by severity
  const filteredIssues = issues.filter(
    (issue) => severityFilter === 'all' || issue.severity === severityFilter
  );

  // Count issues by severity
  const issueCounts = {
    error: issues.filter((i) => i.severity === 'error').length,
    warning: issues.filter((i) => i.severity === 'warning').length,
    info: issues.filter((i) => i.severity === 'info').length,
    hint: issues.filter((i) => i.severity === 'hint').length
  };

  // Get severity icon
  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'error':
        return <BugOutlined style={{ color: '#ff4d4f' }} />;
      case 'warning':
        return <WarningOutlined style={{ color: '#faad14' }} />;
      case 'info':
        return <InfoCircleOutlined style={{ color: '#1890ff' }} />;
      case 'hint':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      default:
        return <InfoCircleOutlined />;
    }
  };

  // Get language from file extension
  const getLanguage = (path: string): string => {
    const ext = path.split('.').pop()?.toLowerCase();
    const languageMap: Record<string, string> = {
      py: 'python',
      js: 'javascript',
      ts: 'typescript',
      tsx: 'typescriptreact',
      jsx: 'javascriptreact',
      java: 'java',
      go: 'go',
      rs: 'rust',
      cpp: 'cpp',
      c: 'c',
      cs: 'csharp',
      rb: 'ruby',
      php: 'php',
      swift: 'swift',
      kt: 'kotlin',
      scala: 'scala',
      sql: 'sql',
      json: 'json',
      yaml: 'yaml',
      yml: 'yaml',
      md: 'markdown',
      html: 'html',
      css: 'css',
      scss: 'scss',
      less: 'less'
    };
    return languageMap[ext || ''] || 'plaintext';
  };

  if (loading) {
    return (
      <div className="code-review-loading">
        <Spin size="large" />
      </div>
    );
  }

  return (
    <Layout className="code-review-container">
      {/* Main Editor Area */}
      <Content className="code-review-content">
        <div className="code-review-toolbar">
          <Space>
            <Title level={4} style={{ margin: 0 }}>
              {filePath || t('code_review.untitled', 'Untitled')}
            </Title>
            {selectedFile && (
              <Tag>{getLanguage(selectedFile.path)}</Tag>
            )}
          </Space>
          <Space>
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={handleAnalyze}
              loading={isAnalyzing}
            >
              {t('code_review.analyze', 'Analyze')}
            </Button>
            <Dropdown
              menu={{
                items: [
                  { key: 'json', label: 'Export as JSON' },
                  { key: 'csv', label: 'Export as CSV' },
                  { key: 'sarif', label: 'Export as SARIF' }
                ]
              }}
            >
              <Button icon={<DownloadOutlined />}>
                {t('code_review.export', 'Export')}
              </Button>
            </Dropdown>
          </Space>
        </div>

        <div className="code-review-editor">
          <CodeEditor
            value={code}
            language={selectedFile ? getLanguage(selectedFile.path) : 'plaintext'}
            onChange={setCode}
            issues={filteredIssues}
            height="100%"
          />
        </div>

        {/* AI Response Panel */}
        {showAIResponse && currentSession && (
          <div className="code-review-ai-panel">
            <StreamingResponse
              sessionId={currentSession.id}
              onComplete={handleAnalysisComplete}
            />
          </div>
        )}
      </Content>

      {/* Issues Sidebar */}
      <Sider width={350} className="code-review-sidebar">
        <div className="issues-header">
          <Title level={5}>{t('code_review.issues', 'Issues')}</Title>
          <Space>
            <Badge count={issues.length} showZero>
              <Button
                icon={<ReloadOutlined />}
                size="small"
                onClick={handleAnalyze}
                loading={isAnalyzing}
              />
            </Badge>
          </Space>
        </div>

        {/* Severity Filter */}
        <div className="issues-filter">
          <Segmented
            value={severityFilter}
            onChange={(value) => setSeverityFilter(value as SeverityFilter)}
            options={[
              { label: t('code_review.all', 'All'), value: 'all' },
              {
                label: (
                  <Badge count={issueCounts.error} size="small">
                    <BugOutlined style={{ color: '#ff4d4f' }} />
                  </Badge>
                ),
                value: 'error'
              },
              {
                label: (
                  <Badge count={issueCounts.warning} size="small">
                    <WarningOutlined style={{ color: '#faad14' }} />
                  </Badge>
                ),
                value: 'warning'
              },
              {
                label: (
                  <Badge count={issueCounts.info} size="small">
                    <InfoCircleOutlined style={{ color: '#1890ff' }} />
                  </Badge>
                ),
                value: 'info'
              }
            ]}
            block
          />
        </div>

        {/* Issues List */}
        <div className="issues-list">
          {filteredIssues.length === 0 ? (
            <Empty
              description={
                issues.length === 0
                  ? t('code_review.no_issues', 'No issues found')
                  : t('code_review.no_matching_issues', 'No matching issues')
              }
            />
          ) : (
            <List
              dataSource={filteredIssues}
              renderItem={(issue) => (
                <List.Item
                  className="issue-item"
                  onClick={() => {
                    // Jump to line in editor
                  }}
                >
                  <List.Item.Meta
                    avatar={getSeverityIcon(issue.severity)}
                    title={
                      <Space>
                        <Text strong>{issue.type}</Text>
                        <Tag>L{issue.line_start}</Tag>
                      </Space>
                    }
                    description={
                      <Text type="secondary" ellipsis={{ tooltip: issue.description }}>
                        {issue.description}
                      </Text>
                    }
                  />
                  {issue.fix && (
                    <Tooltip title={t('code_review.quick_fix', 'Quick fix available')}>
                      <Button type="link" size="small">
                        {t('code_review.fix', 'Fix')}
                      </Button>
                    </Tooltip>
                  )}
                </List.Item>
              )}
            />
          )}
        </div>
      </Sider>
    </Layout>
  );
};

export default CodeReview;
