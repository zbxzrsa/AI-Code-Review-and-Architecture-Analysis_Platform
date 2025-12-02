import React, { useState, useMemo } from 'react';
import { List, Tag, Button, Space, Input, Select, Empty, Tooltip, Badge } from 'antd';
import {
  BugOutlined,
  WarningOutlined,
  InfoCircleOutlined,
  CheckCircleOutlined,
  FilterOutlined,
  SortAscendingOutlined,
  ThunderboltOutlined
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import './IssueList.css';

const { Search } = Input;

export interface Issue {
  id: string;
  type: string;
  severity: 'error' | 'warning' | 'info' | 'hint';
  line_start: number;
  line_end?: number;
  column_start?: number;
  column_end?: number;
  description: string;
  fix?: string;
  rule?: string;
  file?: string;
}

interface IssueListProps {
  issues: Issue[];
  onIssueClick?: (issue: Issue) => void;
  onApplyFix?: (issue: Issue) => void;
  onDismiss?: (issue: Issue) => void;
  loading?: boolean;
  showFileColumn?: boolean;
}

type SortField = 'severity' | 'line' | 'type';
type SortOrder = 'asc' | 'desc';

const severityOrder: Record<string, number> = {
  error: 0,
  warning: 1,
  info: 2,
  hint: 3
};

const severityConfig: Record<string, { color: string; icon: React.ReactNode }> = {
  error: { color: 'red', icon: <BugOutlined /> },
  warning: { color: 'orange', icon: <WarningOutlined /> },
  info: { color: 'blue', icon: <InfoCircleOutlined /> },
  hint: { color: 'green', icon: <CheckCircleOutlined /> }
};

export const IssueList: React.FC<IssueListProps> = ({
  issues,
  onIssueClick,
  onApplyFix,
  onDismiss,
  loading = false,
  showFileColumn = false
}) => {
  const { t } = useTranslation();
  const [searchValue, setSearchValue] = useState('');
  const [severityFilter, setSeverityFilter] = useState<string>('all');
  const [sortField, setSortField] = useState<SortField>('severity');
  const [sortOrder, setSortOrder] = useState<SortOrder>('asc');

  // Filter and sort issues
  const filteredIssues = useMemo(() => {
    let result = [...issues];

    // Filter by search
    if (searchValue) {
      const search = searchValue.toLowerCase();
      result = result.filter(
        (issue) =>
          issue.type.toLowerCase().includes(search) ||
          issue.description.toLowerCase().includes(search) ||
          issue.rule?.toLowerCase().includes(search)
      );
    }

    // Filter by severity
    if (severityFilter !== 'all') {
      result = result.filter((issue) => issue.severity === severityFilter);
    }

    // Sort
    result.sort((a, b) => {
      let comparison = 0;
      switch (sortField) {
        case 'severity':
          comparison = severityOrder[a.severity] - severityOrder[b.severity];
          break;
        case 'line':
          comparison = a.line_start - b.line_start;
          break;
        case 'type':
          comparison = a.type.localeCompare(b.type);
          break;
      }
      return sortOrder === 'asc' ? comparison : -comparison;
    });

    return result;
  }, [issues, searchValue, severityFilter, sortField, sortOrder]);

  // Count by severity
  const severityCounts = useMemo(() => {
    return issues.reduce(
      (acc, issue) => {
        acc[issue.severity] = (acc[issue.severity] || 0) + 1;
        return acc;
      },
      {} as Record<string, number>
    );
  }, [issues]);

  const renderIssue = (issue: Issue) => {
    const config = severityConfig[issue.severity] || severityConfig.info;

    return (
      <List.Item
        className="issue-list-item"
        onClick={() => onIssueClick?.(issue)}
        actions={[
          issue.fix && onApplyFix && (
            <Tooltip title={t('issues.apply_fix', 'Apply fix')}>
              <Button
                type="link"
                size="small"
                icon={<ThunderboltOutlined />}
                onClick={(e) => {
                  e.stopPropagation();
                  onApplyFix(issue);
                }}
              >
                {t('issues.fix', 'Fix')}
              </Button>
            </Tooltip>
          ),
          onDismiss && (
            <Button
              type="link"
              size="small"
              onClick={(e) => {
                e.stopPropagation();
                onDismiss(issue);
              }}
            >
              {t('issues.dismiss', 'Dismiss')}
            </Button>
          )
        ].filter(Boolean)}
      >
        <List.Item.Meta
          avatar={
            <Tag color={config.color} icon={config.icon}>
              {issue.severity.toUpperCase()}
            </Tag>
          }
          title={
            <Space>
              <span className="issue-type">{issue.type}</span>
              <Tag size="small">L{issue.line_start}</Tag>
              {showFileColumn && issue.file && (
                <span className="issue-file">{issue.file}</span>
              )}
            </Space>
          }
          description={
            <div className="issue-description">
              {issue.description}
              {issue.rule && (
                <span className="issue-rule">({issue.rule})</span>
              )}
            </div>
          }
        />
      </List.Item>
    );
  };

  return (
    <div className="issue-list">
      {/* Toolbar */}
      <div className="issue-list-toolbar">
        <Search
          placeholder={t('issues.search', 'Search issues...')}
          value={searchValue}
          onChange={(e) => setSearchValue(e.target.value)}
          allowClear
          style={{ width: 200 }}
        />
        <Space>
          <Select
            value={severityFilter}
            onChange={setSeverityFilter}
            style={{ width: 120 }}
            options={[
              { label: t('issues.all', 'All'), value: 'all' },
              {
                label: (
                  <Badge count={severityCounts.error || 0} size="small">
                    <span style={{ paddingRight: 16 }}>{t('issues.errors', 'Errors')}</span>
                  </Badge>
                ),
                value: 'error'
              },
              {
                label: (
                  <Badge count={severityCounts.warning || 0} size="small">
                    <span style={{ paddingRight: 16 }}>{t('issues.warnings', 'Warnings')}</span>
                  </Badge>
                ),
                value: 'warning'
              },
              {
                label: (
                  <Badge count={severityCounts.info || 0} size="small">
                    <span style={{ paddingRight: 16 }}>{t('issues.info', 'Info')}</span>
                  </Badge>
                ),
                value: 'info'
              }
            ]}
          />
          <Select
            value={sortField}
            onChange={setSortField}
            style={{ width: 120 }}
            options={[
              { label: t('issues.sort_severity', 'Severity'), value: 'severity' },
              { label: t('issues.sort_line', 'Line'), value: 'line' },
              { label: t('issues.sort_type', 'Type'), value: 'type' }
            ]}
          />
          <Button
            icon={<SortAscendingOutlined rotate={sortOrder === 'desc' ? 180 : 0} />}
            onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
          />
        </Space>
      </div>

      {/* Summary */}
      <div className="issue-list-summary">
        <Space>
          {Object.entries(severityCounts).map(([severity, count]) => {
            const config = severityConfig[severity];
            return (
              <Tag key={severity} color={config?.color} icon={config?.icon}>
                {count} {severity}
              </Tag>
            );
          })}
        </Space>
      </div>

      {/* List */}
      <div className="issue-list-content">
        {filteredIssues.length === 0 ? (
          <Empty
            description={
              searchValue || severityFilter !== 'all'
                ? t('issues.no_matching', 'No matching issues')
                : t('issues.no_issues', 'No issues found')
            }
          />
        ) : (
          <List
            dataSource={filteredIssues}
            renderItem={renderIssue}
            loading={loading}
          />
        )}
      </div>
    </div>
  );
};

export default IssueList;
