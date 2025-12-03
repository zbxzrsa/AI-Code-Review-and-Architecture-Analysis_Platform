/**
 * Import/Export Page
 * 导入导出页面
 * 
 * Features:
 * - Export configuration
 * - Import settings
 * - Backup/restore projects
 * - Data migration
 */

import React, { useState } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Button,
  Upload,
  List,
  Tag,
  Checkbox,
  Alert,
  Progress,
  Steps,
  message,
  Modal,
} from 'antd';
import {
  ExportOutlined,
  ImportOutlined,
  DownloadOutlined,
  UploadOutlined,
  FileZipOutlined,
  SettingOutlined,
  SafetyCertificateOutlined,
  TeamOutlined,
  ApiOutlined,
  CodeOutlined,
  CheckCircleOutlined,
  DatabaseOutlined,
  CloudDownloadOutlined,
  CloudUploadOutlined,
  HistoryOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

const { Title, Text, Paragraph } = Typography;

interface ExportItem {
  key: string;
  label: string;
  description: string;
  icon: React.ReactNode;
  size?: string;
}

const exportItems: ExportItem[] = [
  { key: 'rules', label: 'Quality Rules', description: 'Custom linting rules and configurations', icon: <CodeOutlined />, size: '12 KB' },
  { key: 'webhooks', label: 'Webhooks', description: 'Webhook configurations and subscriptions', icon: <ApiOutlined />, size: '3 KB' },
  { key: 'team', label: 'Team Settings', description: 'Team members and permissions', icon: <TeamOutlined />, size: '8 KB' },
  { key: 'integrations', label: 'Integrations', description: 'Connected services and OAuth tokens', icon: <SettingOutlined />, size: '5 KB' },
  { key: 'security', label: 'Security Policies', description: 'Security rules and OWASP settings', icon: <SafetyCertificateOutlined />, size: '15 KB' },
];

interface BackupItem {
  id: string;
  name: string;
  createdAt: string;
  size: string;
  type: 'full' | 'partial';
}

const mockBackups: BackupItem[] = [
  { id: '1', name: 'Full Backup - March 2024', createdAt: '2024-03-01T10:00:00Z', size: '45.2 MB', type: 'full' },
  { id: '2', name: 'Config Backup', createdAt: '2024-02-28T15:30:00Z', size: '128 KB', type: 'partial' },
  { id: '3', name: 'Full Backup - February 2024', createdAt: '2024-02-01T10:00:00Z', size: '42.8 MB', type: 'full' },
];

export const ImportExport: React.FC = () => {
  const { t: _t } = useTranslation();
  const [selectedExports, setSelectedExports] = useState<string[]>(['rules', 'webhooks', 'security']);
  const [exportProgress, setExportProgress] = useState<number | null>(null);
  const [importModalOpen, setImportModalOpen] = useState(false);
  const [importStep, setImportStep] = useState(0);

  const handleExport = () => {
    setExportProgress(0);
    const interval = setInterval(() => {
      setExportProgress(prev => {
        if (prev === null) return 0;
        if (prev >= 100) {
          clearInterval(interval);
          message.success('Export completed! Download starting...');
          setTimeout(() => setExportProgress(null), 1000);
          return 100;
        }
        return prev + 10;
      });
    }, 200);
  };

  const handleSelectAll = (checked: boolean) => {
    setSelectedExports(checked ? exportItems.map(i => i.key) : []);
  };

  return (
    <div className="import-export-page" style={{ maxWidth: 1200, margin: '0 auto' }}>
      {/* Header */}
      <div className="page-header" style={{ marginBottom: 24 }}>
        <div>
          <Title level={3} style={{ margin: 0 }}>
            <DatabaseOutlined style={{ color: '#2563eb' }} /> Import / Export
          </Title>
          <Text type="secondary">Backup and restore your configuration and data</Text>
        </div>
      </div>

      <Row gutter={24}>
        {/* Export Section */}
        <Col xs={24} lg={12}>
          <Card
            title={<><ExportOutlined /> Export Configuration</>}
            style={{ borderRadius: 12, marginBottom: 24 }}
          >
            <Paragraph type="secondary">
              Select the items you want to export. The configuration will be downloaded as a JSON file.
            </Paragraph>

            <div style={{ marginBottom: 16 }}>
              <Checkbox
                checked={selectedExports.length === exportItems.length}
                indeterminate={selectedExports.length > 0 && selectedExports.length < exportItems.length}
                onChange={e => handleSelectAll(e.target.checked)}
              >
                Select All
              </Checkbox>
            </div>

            <List
              dataSource={exportItems}
              renderItem={item => (
                <List.Item style={{ padding: '12px 0' }}>
                  <Checkbox
                    checked={selectedExports.includes(item.key)}
                    onChange={e => {
                      setSelectedExports(prev =>
                        e.target.checked
                          ? [...prev, item.key]
                          : prev.filter(k => k !== item.key)
                      );
                    }}
                  >
                    <Space>
                      <span style={{ color: '#2563eb' }}>{item.icon}</span>
                      <div>
                        <Text strong>{item.label}</Text>
                        <div>
                          <Text type="secondary" style={{ fontSize: 12 }}>{item.description}</Text>
                        </div>
                      </div>
                    </Space>
                  </Checkbox>
                  <Tag style={{ marginLeft: 'auto' }}>{item.size}</Tag>
                </List.Item>
              )}
            />

            {exportProgress !== null && (
              <div style={{ marginTop: 16 }}>
                <Progress percent={exportProgress} status={exportProgress === 100 ? 'success' : 'active'} />
              </div>
            )}

            <Button
              type="primary"
              icon={<DownloadOutlined />}
              onClick={handleExport}
              disabled={selectedExports.length === 0 || exportProgress !== null}
              style={{ marginTop: 16 }}
              block
            >
              Export Selected ({selectedExports.length} items)
            </Button>
          </Card>
        </Col>

        {/* Import Section */}
        <Col xs={24} lg={12}>
          <Card
            title={<><ImportOutlined /> Import Configuration</>}
            style={{ borderRadius: 12, marginBottom: 24 }}
          >
            <Alert
              type="warning"
              showIcon
              message="Importing will overwrite existing settings"
              description="Make sure to backup your current configuration before importing."
              style={{ marginBottom: 16 }}
            />

            <Upload.Dragger
              accept=".json,.zip"
              showUploadList={false}
              beforeUpload={_file => {
                message.loading('Validating import file...');
                setTimeout(() => {
                  setImportModalOpen(true);
                  setImportStep(0);
                }, 1000);
                return false;
              }}
            >
              <p className="ant-upload-drag-icon">
                <CloudUploadOutlined style={{ fontSize: 48, color: '#2563eb' }} />
              </p>
              <p className="ant-upload-text">Click or drag file to import</p>
              <p className="ant-upload-hint">Supports .json or .zip backup files</p>
            </Upload.Dragger>
          </Card>
        </Col>
      </Row>

      {/* Backup History */}
      <Card
        title={<><HistoryOutlined /> Backup History</>}
        style={{ borderRadius: 12 }}
        extra={
          <Button icon={<CloudDownloadOutlined />}>Create Backup</Button>
        }
      >
        <List
          dataSource={mockBackups}
          renderItem={backup => (
            <List.Item
              actions={[
                <Button key="download" size="small" icon={<DownloadOutlined />}>Download</Button>,
                <Button key="restore" size="small" icon={<ImportOutlined />}>Restore</Button>,
              ]}
            >
              <List.Item.Meta
                avatar={
                  <div style={{
                    width: 40,
                    height: 40,
                    borderRadius: 10,
                    background: backup.type === 'full' ? '#dbeafe' : '#f1f5f9',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: backup.type === 'full' ? '#2563eb' : '#64748b',
                  }}>
                    <FileZipOutlined />
                  </div>
                }
                title={
                  <Space>
                    <Text strong>{backup.name}</Text>
                    <Tag color={backup.type === 'full' ? 'blue' : 'default'}>
                      {backup.type === 'full' ? 'Full Backup' : 'Partial'}
                    </Tag>
                  </Space>
                }
                description={
                  <Space>
                    <Text type="secondary">{new Date(backup.createdAt).toLocaleDateString()}</Text>
                    <Text type="secondary">•</Text>
                    <Text type="secondary">{backup.size}</Text>
                  </Space>
                }
              />
            </List.Item>
          )}
        />
      </Card>

      {/* Import Modal */}
      <Modal
        title={<><ImportOutlined /> Import Configuration</>}
        open={importModalOpen}
        onCancel={() => setImportModalOpen(false)}
        footer={
          importStep === 2 ? (
            <Button type="primary" onClick={() => {
              setImportModalOpen(false);
              message.success('Import completed successfully!');
            }}>
              Done
            </Button>
          ) : (
            <Button type="primary" onClick={() => setImportStep(prev => prev + 1)}>
              {importStep === 0 ? 'Validate' : 'Import'}
            </Button>
          )
        }
        width={600}
      >
        <Steps
          current={importStep}
          items={[
            { title: 'Upload', icon: <UploadOutlined /> },
            { title: 'Validate', icon: <SafetyCertificateOutlined /> },
            { title: 'Complete', icon: <CheckCircleOutlined /> },
          ]}
          style={{ marginBottom: 24 }}
        />

        {importStep === 0 && (
          <Alert
            type="info"
            message="File uploaded: config-backup.json"
            description="3 configurations detected: Quality Rules, Webhooks, Security Policies"
          />
        )}

        {importStep === 1 && (
          <div>
            <Alert type="success" message="Validation passed!" style={{ marginBottom: 16 }} />
            <List
              size="small"
              dataSource={[
                { label: 'Quality Rules', count: 15 },
                { label: 'Webhooks', count: 3 },
                { label: 'Security Policies', count: 8 },
              ]}
              renderItem={item => (
                <List.Item>
                  <Space>
                    <CheckCircleOutlined style={{ color: '#22c55e' }} />
                    <Text>{item.label}</Text>
                  </Space>
                  <Tag>{item.count} items</Tag>
                </List.Item>
              )}
            />
          </div>
        )}

        {importStep === 2 && (
          <Alert
            type="success"
            icon={<CheckCircleOutlined />}
            message="Import Successful!"
            description="All configurations have been imported successfully."
          />
        )}
      </Modal>
    </div>
  );
};

export default ImportExport;
