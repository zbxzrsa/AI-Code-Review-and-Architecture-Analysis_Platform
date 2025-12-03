/**
 * New Project Wizard
 * 
 * Multi-step form for creating a new project with:
 * - Step 1: Basic Information
 * - Step 2: Analysis Settings
 * - Step 3: Notification Settings
 * - Step 4: Review & Submit
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Card,
  Steps,
  Form,
  Input,
  Select,
  Button,
  Space,
  Typography,
  Row,
  Col,
  Switch,
  InputNumber,
  Divider,
  Tag,
  Alert,
  message,
  Modal,
  Descriptions,
  Spin,
  Result,
} from 'antd';
import {
  ArrowLeftOutlined,
  ArrowRightOutlined,
  SaveOutlined,
  CheckCircleOutlined,
  FolderOutlined,
  SettingOutlined,
  BellOutlined,
  FileSearchOutlined,
  GithubOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useProjectStore, type ProjectDraft, defaultDraft } from '../../store/projectStore';
import { useCreateProject } from '../../hooks/useProjects';
import './NewProject.css';

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;

/** Language options */
const languageOptions = [
  { value: 'python', label: 'Python', color: '#3572A5' },
  { value: 'javascript', label: 'JavaScript', color: '#f1e05a' },
  { value: 'typescript', label: 'TypeScript', color: '#2b7489' },
  { value: 'java', label: 'Java', color: '#b07219' },
  { value: 'go', label: 'Go', color: '#00ADD8' },
  { value: 'rust', label: 'Rust', color: '#dea584' },
  { value: 'cpp', label: 'C++', color: '#f34b7d' },
  { value: 'csharp', label: 'C#', color: '#178600' },
  { value: 'ruby', label: 'Ruby', color: '#701516' },
  { value: 'php', label: 'PHP', color: '#4F5D95' },
  { value: 'swift', label: 'Swift', color: '#ffac45' },
  { value: 'kotlin', label: 'Kotlin', color: '#F18E33' },
];

/** AI Model options */
const aiModelOptions = [
  { value: 'gpt-4', label: 'GPT-4 (Most Capable)', description: 'Best for complex analysis' },
  { value: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo (Fast)', description: 'Good balance of speed and quality' },
  { value: 'claude-3-opus', label: 'Claude 3 Opus (Advanced)', description: 'Excellent for detailed reviews' },
  { value: 'claude-3-sonnet', label: 'Claude 3 Sonnet (Balanced)', description: 'Fast and reliable' },
];

/** Analysis frequency options */
const frequencyOptions = [
  { value: 'manual', label: 'Manual', description: 'Run analysis manually' },
  { value: 'on_push', label: 'On Push', description: 'Analyze on every push' },
  { value: 'on_pr', label: 'On Pull Request', description: 'Analyze on PR creation' },
  { value: 'scheduled', label: 'Scheduled', description: 'Run on a schedule' },
];

/** Priority options */
const priorityOptions = [
  { value: 'low', label: 'Low', color: 'default' },
  { value: 'medium', label: 'Medium', color: 'blue' },
  { value: 'high', label: 'High', color: 'orange' },
];

/** Email digest options */
const digestOptions = [
  { value: 'none', label: 'None' },
  { value: 'daily', label: 'Daily Digest' },
  { value: 'weekly', label: 'Weekly Digest' },
];

/** Step 1: Basic Information */
const BasicInfoStep: React.FC<{
  form: any;
  draft: ProjectDraft;
  onValuesChange: (values: any) => void;
}> = ({ form, draft, onValuesChange }) => {
  const { t } = useTranslation();

  return (
    <div className="wizard-step">
      <Title level={4}>
        <FolderOutlined /> {t('projects.wizard.basic_info', 'Basic Information')}
      </Title>
      <Paragraph type="secondary">
        {t('projects.wizard.basic_info_desc', 'Provide the essential details about your project')}
      </Paragraph>

      <Form
        form={form}
        layout="vertical"
        initialValues={draft.basic_info}
        onValuesChange={(_, allValues) => onValuesChange({ basic_info: allValues })}
      >
        <Form.Item
          name="name"
          label={t('projects.form.name', 'Project Name')}
          rules={[
            { required: true, message: t('projects.form.name_required', 'Please enter a project name') },
            { min: 3, message: t('projects.form.name_min', 'Name must be at least 3 characters') },
            { max: 100, message: t('projects.form.name_max', 'Name must be less than 100 characters') },
            { 
              pattern: /^[a-zA-Z0-9][a-zA-Z0-9-_]*$/,
              message: t('projects.form.name_pattern', 'Name must start with a letter or number')
            },
          ]}
          tooltip={t('projects.form.name_tooltip', 'A unique identifier for your project')}
        >
          <Input 
            placeholder="my-awesome-project" 
            prefix={<FolderOutlined />}
            size="large"
            aria-label={t('projects.form.name', 'Project Name')}
          />
        </Form.Item>

        <Form.Item
          name="description"
          label={t('projects.form.description', 'Description')}
          rules={[
            { max: 500, message: t('projects.form.description_max', 'Description must be less than 500 characters') },
          ]}
        >
          <TextArea
            rows={4}
            placeholder={t('projects.form.description_placeholder', 'Describe what this project does...')}
            showCount
            maxLength={500}
            aria-label={t('projects.form.description', 'Description')}
          />
        </Form.Item>

        <Row gutter={16}>
          <Col xs={24} sm={12}>
            <Form.Item
              name="repository_url"
              label={t('projects.form.repo', 'Repository URL')}
              rules={[
                {
                  pattern: /^https?:\/\/(github\.com|gitlab\.com|bitbucket\.org)\/[\w.-]+\/[\w.-]+/,
                  message: t('projects.form.repo_invalid', 'Please enter a valid repository URL'),
                },
              ]}
            >
              <Input
                placeholder="https://github.com/username/repo"
                prefix={<GithubOutlined />}
                aria-label={t('projects.form.repo', 'Repository URL')}
              />
            </Form.Item>
          </Col>
          <Col xs={24} sm={12}>
            <Form.Item
              name="language"
              label={t('projects.form.language', 'Primary Language')}
              rules={[{ required: true, message: t('projects.form.language_required', 'Please select a language') }]}
            >
              <Select
                placeholder={t('projects.form.select_language', 'Select language')}
                options={languageOptions.map(opt => ({
                  value: opt.value,
                  label: (
                    <Space>
                      <span 
                        style={{ 
                          width: 12, 
                          height: 12, 
                          borderRadius: '50%', 
                          backgroundColor: opt.color,
                          display: 'inline-block',
                        }} 
                      />
                      {opt.label}
                    </Space>
                  ),
                }))}
                aria-label={t('projects.form.language', 'Primary Language')}
              />
            </Form.Item>
          </Col>
        </Row>
      </Form>
    </div>
  );
};

/** Step 2: Analysis Settings */
const AnalysisSettingsStep: React.FC<{
  form: any;
  draft: ProjectDraft;
  onValuesChange: (values: any) => void;
}> = ({ form, draft, onValuesChange }) => {
  const { t } = useTranslation();

  return (
    <div className="wizard-step">
      <Title level={4}>
        <SettingOutlined /> {t('projects.wizard.analysis_settings', 'Analysis Settings')}
      </Title>
      <Paragraph type="secondary">
        {t('projects.wizard.analysis_settings_desc', 'Configure how your code will be analyzed')}
      </Paragraph>

      <Form
        form={form}
        layout="vertical"
        initialValues={draft.analysis_settings}
        onValuesChange={(_, allValues) => onValuesChange({ analysis_settings: allValues })}
      >
        <Row gutter={16}>
          <Col xs={24} sm={12}>
            <Form.Item
              name="ai_model"
              label={t('projects.form.ai_model', 'AI Model')}
              tooltip={t('projects.form.ai_model_tooltip', 'Choose the AI model for code analysis')}
            >
              <Select
                placeholder={t('projects.form.select_model', 'Select AI model')}
                options={aiModelOptions.map(opt => ({
                  value: opt.value,
                  label: (
                    <div>
                      <div>{opt.label}</div>
                      <Text type="secondary" style={{ fontSize: 12 }}>{opt.description}</Text>
                    </div>
                  ),
                }))}
                aria-label={t('projects.form.ai_model', 'AI Model')}
              />
            </Form.Item>
          </Col>
          <Col xs={24} sm={12}>
            <Form.Item
              name="analysis_frequency"
              label={t('projects.form.frequency', 'Analysis Frequency')}
            >
              <Select
                placeholder={t('projects.form.select_frequency', 'Select frequency')}
                options={frequencyOptions.map(opt => ({
                  value: opt.value,
                  label: (
                    <div>
                      <div>{opt.label}</div>
                      <Text type="secondary" style={{ fontSize: 12 }}>{opt.description}</Text>
                    </div>
                  ),
                }))}
                aria-label={t('projects.form.frequency', 'Analysis Frequency')}
              />
            </Form.Item>
          </Col>
        </Row>

        <Row gutter={16}>
          <Col xs={24} sm={12}>
            <Form.Item
              name="priority"
              label={t('projects.form.priority', 'Priority Level')}
            >
              <Select
                placeholder={t('projects.form.select_priority', 'Select priority')}
                options={priorityOptions.map(opt => ({
                  value: opt.value,
                  label: <Tag color={opt.color}>{opt.label}</Tag>,
                }))}
                aria-label={t('projects.form.priority', 'Priority Level')}
              />
            </Form.Item>
          </Col>
          <Col xs={24} sm={12}>
            <Form.Item
              name="max_files_per_analysis"
              label={t('projects.form.max_files', 'Max Files per Analysis')}
              tooltip={t('projects.form.max_files_tooltip', 'Limit the number of files analyzed in one run')}
            >
              <InputNumber
                min={1}
                max={1000}
                style={{ width: '100%' }}
                aria-label={t('projects.form.max_files', 'Max Files per Analysis')}
              />
            </Form.Item>
          </Col>
        </Row>

        <Form.Item
          name="excluded_patterns"
          label={t('projects.form.excluded_patterns', 'Excluded Patterns')}
          tooltip={t('projects.form.excluded_patterns_tooltip', 'Glob patterns for files to exclude')}
        >
          <Select
            mode="tags"
            style={{ width: '100%' }}
            placeholder={t('projects.form.excluded_patterns_placeholder', 'e.g., node_modules/**, *.test.js')}
            defaultValue={['node_modules/**', '.git/**', 'dist/**']}
            aria-label={t('projects.form.excluded_patterns', 'Excluded Patterns')}
          />
        </Form.Item>
      </Form>
    </div>
  );
};

/** Step 3: Notification Settings */
const NotificationSettingsStep: React.FC<{
  form: any;
  draft: ProjectDraft;
  onValuesChange: (values: any) => void;
}> = ({ form, draft, onValuesChange }) => {
  const { t } = useTranslation();

  return (
    <div className="wizard-step">
      <Title level={4}>
        <BellOutlined /> {t('projects.wizard.notification_settings', 'Notification Settings')}
      </Title>
      <Paragraph type="secondary">
        {t('projects.wizard.notification_settings_desc', 'Configure how you want to be notified')}
      </Paragraph>

      <Form
        form={form}
        layout="vertical"
        initialValues={draft.notification_settings}
        onValuesChange={(_, allValues) => onValuesChange({ notification_settings: allValues })}
      >
        <Form.Item
          name="email_on_analysis_complete"
          label={t('projects.form.email_on_complete', 'Email on Analysis Complete')}
          valuePropName="checked"
        >
          <Switch aria-label={t('projects.form.email_on_complete', 'Email on Analysis Complete')} />
        </Form.Item>

        <Form.Item
          name="email_on_critical_issues"
          label={t('projects.form.email_on_critical', 'Email on Critical Issues')}
          valuePropName="checked"
        >
          <Switch aria-label={t('projects.form.email_on_critical', 'Email on Critical Issues')} />
        </Form.Item>

        <Form.Item
          name="email_digest"
          label={t('projects.form.email_digest', 'Email Digest')}
        >
          <Select
            options={digestOptions}
            aria-label={t('projects.form.email_digest', 'Email Digest')}
          />
        </Form.Item>

        <Divider>{t('projects.form.integrations', 'Integrations')}</Divider>

        <Form.Item
          name="slack_webhook_url"
          label={t('projects.form.slack_webhook', 'Slack Webhook URL')}
          rules={[
            {
              pattern: /^https:\/\/hooks\.slack\.com\/services\//,
              message: t('projects.form.slack_webhook_invalid', 'Please enter a valid Slack webhook URL'),
            },
          ]}
        >
          <Input
            placeholder="https://hooks.slack.com/services/..."
            aria-label={t('projects.form.slack_webhook', 'Slack Webhook URL')}
          />
        </Form.Item>

        <Form.Item
          name="teams_webhook_url"
          label={t('projects.form.teams_webhook', 'Microsoft Teams Webhook URL')}
          rules={[
            {
              pattern: /^https:\/\/.*\.webhook\.office\.com\//,
              message: t('projects.form.teams_webhook_invalid', 'Please enter a valid Teams webhook URL'),
            },
          ]}
        >
          <Input
            placeholder="https://...webhook.office.com/..."
            aria-label={t('projects.form.teams_webhook', 'Microsoft Teams Webhook URL')}
          />
        </Form.Item>
      </Form>
    </div>
  );
};

/** Step 4: Review */
const ReviewStep: React.FC<{
  draft: ProjectDraft;
}> = ({ draft }) => {
  const { t } = useTranslation();
  const language = languageOptions.find(l => l.value === draft.basic_info.language);
  const model = aiModelOptions.find(m => m.value === draft.analysis_settings.ai_model);
  const frequency = frequencyOptions.find(f => f.value === draft.analysis_settings.analysis_frequency);

  return (
    <div className="wizard-step">
      <Title level={4}>
        <FileSearchOutlined /> {t('projects.wizard.review', 'Review & Create')}
      </Title>
      <Paragraph type="secondary">
        {t('projects.wizard.review_desc', 'Review your project settings before creating')}
      </Paragraph>

      <Card title={t('projects.wizard.basic_info', 'Basic Information')} className="review-card">
        <Descriptions column={{ xs: 1, sm: 2 }}>
          <Descriptions.Item label={t('projects.form.name', 'Name')}>
            {draft.basic_info.name || '-'}
          </Descriptions.Item>
          <Descriptions.Item label={t('projects.form.language', 'Language')}>
            {language ? (
              <Space>
                <span 
                  style={{ 
                    width: 12, 
                    height: 12, 
                    borderRadius: '50%', 
                    backgroundColor: language.color,
                    display: 'inline-block',
                  }} 
                />
                {language.label}
              </Space>
            ) : '-'}
          </Descriptions.Item>
          <Descriptions.Item label={t('projects.form.repo', 'Repository')} span={2}>
            {draft.basic_info.repository_url || t('common.not_set', 'Not set')}
          </Descriptions.Item>
          <Descriptions.Item label={t('projects.form.description', 'Description')} span={2}>
            {draft.basic_info.description || t('common.not_set', 'Not set')}
          </Descriptions.Item>
        </Descriptions>
      </Card>

      <Card title={t('projects.wizard.analysis_settings', 'Analysis Settings')} className="review-card">
        <Descriptions column={{ xs: 1, sm: 2 }}>
          <Descriptions.Item label={t('projects.form.ai_model', 'AI Model')}>
            {model?.label || '-'}
          </Descriptions.Item>
          <Descriptions.Item label={t('projects.form.frequency', 'Frequency')}>
            {frequency?.label || '-'}
          </Descriptions.Item>
          <Descriptions.Item label={t('projects.form.priority', 'Priority')}>
            {draft.analysis_settings.priority && (
              <Tag color={priorityOptions.find(p => p.value === draft.analysis_settings.priority)?.color}>
                {draft.analysis_settings.priority}
              </Tag>
            )}
          </Descriptions.Item>
          <Descriptions.Item label={t('projects.form.max_files', 'Max Files')}>
            {draft.analysis_settings.max_files_per_analysis || '-'}
          </Descriptions.Item>
          <Descriptions.Item label={t('projects.form.excluded_patterns', 'Excluded')} span={2}>
            <Space wrap>
              {draft.analysis_settings.excluded_patterns?.map((pattern, i) => (
                <Tag key={i}>{pattern}</Tag>
              )) || '-'}
            </Space>
          </Descriptions.Item>
        </Descriptions>
      </Card>

      <Card title={t('projects.wizard.notification_settings', 'Notification Settings')} className="review-card">
        <Descriptions column={{ xs: 1, sm: 2 }}>
          <Descriptions.Item label={t('projects.form.email_on_complete', 'Email on Complete')}>
            {draft.notification_settings.email_on_analysis_complete ? 
              <Tag color="green">{t('common.enabled', 'Enabled')}</Tag> : 
              <Tag>{t('common.disabled', 'Disabled')}</Tag>
            }
          </Descriptions.Item>
          <Descriptions.Item label={t('projects.form.email_on_critical', 'Email on Critical')}>
            {draft.notification_settings.email_on_critical_issues ? 
              <Tag color="green">{t('common.enabled', 'Enabled')}</Tag> : 
              <Tag>{t('common.disabled', 'Disabled')}</Tag>
            }
          </Descriptions.Item>
          <Descriptions.Item label={t('projects.form.email_digest', 'Email Digest')}>
            {draft.notification_settings.email_digest || 'none'}
          </Descriptions.Item>
          <Descriptions.Item label={t('projects.form.integrations', 'Integrations')}>
            <Space>
              {draft.notification_settings.slack_webhook_url && <Tag color="purple">Slack</Tag>}
              {draft.notification_settings.teams_webhook_url && <Tag color="blue">Teams</Tag>}
              {!draft.notification_settings.slack_webhook_url && !draft.notification_settings.teams_webhook_url && '-'}
            </Space>
          </Descriptions.Item>
        </Descriptions>
      </Card>

      <Alert
        message={t('projects.wizard.ready', 'Ready to Create')}
        description={t('projects.wizard.ready_desc', 'Click "Create Project" to finish setting up your project.')}
        type="success"
        showIcon
        icon={<CheckCircleOutlined />}
        style={{ marginTop: 16 }}
      />
    </div>
  );
};

/**
 * New Project Wizard Component
 */
export const NewProject: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  
  // Store
  const { 
    draft, 
    setDraft, 
    updateDraft, 
    saveDraftToStorage, 
    loadDraftFromStorage, 
    clearDraft 
  } = useProjectStore();
  
  // Mutation
  const createProject = useCreateProject();
  
  // Local state
  const [currentStep, setCurrentStep] = useState(0);
  const [showExitModal, setShowExitModal] = useState(false);
  
  // Forms
  const [basicForm] = Form.useForm();
  const [analysisForm] = Form.useForm();
  const [notificationForm] = Form.useForm();
  
  // Initialize draft on mount
  useEffect(() => {
    loadDraftFromStorage();
    if (!draft) {
      setDraft({ ...defaultDraft });
    }
    // Intentionally run only on mount to initialize draft
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  
  // Set current step from draft
  useEffect(() => {
    if (draft?.step) {
      setCurrentStep(draft.step);
    }
  }, [draft?.step]);
  
  // Auto-save draft
  useEffect(() => {
    const autoSaveInterval = setInterval(() => {
      if (draft) {
        saveDraftToStorage();
      }
    }, 30000); // Auto-save every 30 seconds
    
    return () => clearInterval(autoSaveInterval);
  }, [draft, saveDraftToStorage]);
  
  // Current draft with fallback
  const currentDraft = useMemo(() => draft || defaultDraft, [draft]);
  
  // Handle values change
  const handleValuesChange = useCallback((updates: Partial<ProjectDraft>) => {
    updateDraft(updates);
  }, [updateDraft]);
  
  // Validate current step
  const validateCurrentStep = useCallback(async (): Promise<boolean> => {
    try {
      switch (currentStep) {
        case 0:
          await basicForm.validateFields();
          break;
        case 1:
          await analysisForm.validateFields();
          break;
        case 2:
          await notificationForm.validateFields();
          break;
      }
      return true;
    } catch {
      return false;
    }
  }, [currentStep, basicForm, analysisForm, notificationForm]);
  
  // Navigation
  const handleNext = useCallback(async () => {
    const isValid = await validateCurrentStep();
    if (isValid && currentStep < 3) {
      setCurrentStep(currentStep + 1);
      updateDraft({ step: currentStep + 1 });
    }
  }, [currentStep, validateCurrentStep, updateDraft]);
  
  const handlePrevious = useCallback(() => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
      updateDraft({ step: currentStep - 1 });
    }
  }, [currentStep, updateDraft]);
  
  // Save draft manually
  const handleSaveDraft = useCallback(() => {
    saveDraftToStorage();
    message.success(t('projects.wizard.draft_saved', 'Draft saved successfully'));
  }, [saveDraftToStorage, t]);
  
  // Create project
  const handleCreate = useCallback(async () => {
    const isValid = await validateCurrentStep();
    if (!isValid) return;
    
    const projectData = {
      name: currentDraft.basic_info.name,
      description: currentDraft.basic_info.description,
      repository_url: currentDraft.basic_info.repository_url,
      language: currentDraft.basic_info.language,
      settings: {
        auto_review: currentDraft.analysis_settings.analysis_frequency !== 'manual',
        review_on_push: currentDraft.analysis_settings.analysis_frequency === 'on_push',
        review_on_pr: currentDraft.analysis_settings.analysis_frequency === 'on_pr',
        severity_threshold: 'warning' as const,
        enabled_rules: [],
        ignored_paths: currentDraft.analysis_settings.excluded_patterns || [],
        analysis: currentDraft.analysis_settings,
        notifications: currentDraft.notification_settings,
      },
    };
    
    createProject.mutate(projectData, {
      onSuccess: (data) => {
        clearDraft();
        navigate(`/projects/${data.id}/settings`);
      },
    });
  }, [currentDraft, createProject, clearDraft, navigate, validateCurrentStep]);
  
  // Handle navigation away
  const handleBack = useCallback(() => {
    if (currentDraft.basic_info.name || currentDraft.basic_info.description) {
      setShowExitModal(true);
    } else {
      navigate('/projects');
    }
  }, [currentDraft, navigate]);
  
  const confirmExit = useCallback((saveDraft: boolean) => {
    if (saveDraft) {
      saveDraftToStorage();
    } else {
      clearDraft();
    }
    navigate('/projects');
  }, [saveDraftToStorage, clearDraft, navigate]);
  
  // Steps configuration
  const steps = [
    {
      title: t('projects.wizard.step_basic', 'Basic Info'),
      icon: <FolderOutlined />,
    },
    {
      title: t('projects.wizard.step_analysis', 'Analysis'),
      icon: <SettingOutlined />,
    },
    {
      title: t('projects.wizard.step_notifications', 'Notifications'),
      icon: <BellOutlined />,
    },
    {
      title: t('projects.wizard.step_review', 'Review'),
      icon: <FileSearchOutlined />,
    },
  ];
  
  // Show success result after creation
  if (createProject.isSuccess) {
    return (
      <div className="new-project-container">
        <Result
          status="success"
          title={t('projects.wizard.success_title', 'Project Created Successfully!')}
          subTitle={t('projects.wizard.success_subtitle', 'Your project is ready to use.')}
          extra={[
            <Button 
              type="primary" 
              key="settings"
              onClick={() => navigate(`/projects/${createProject.data.id}/settings`)}
            >
              {t('projects.wizard.go_to_settings', 'Go to Settings')}
            </Button>,
            <Button 
              key="list"
              onClick={() => navigate('/projects')}
            >
              {t('projects.wizard.go_to_list', 'View All Projects')}
            </Button>,
          ]}
        />
      </div>
    );
  }
  
  return (
    <div className="new-project-container" role="main" aria-label={t('projects.wizard.title', 'Create New Project')}>
      {/* Header */}
      <div className="new-project-header">
        <Space>
          <Button 
            icon={<ArrowLeftOutlined />} 
            onClick={handleBack}
            aria-label={t('common.back', 'Back')}
          />
          <div>
            <Title level={3} style={{ margin: 0 }}>
              {t('projects.wizard.title', 'Create New Project')}
            </Title>
            {currentDraft.last_saved_at && (
              <Text type="secondary" style={{ fontSize: 12 }}>
                {t('projects.wizard.last_saved', 'Last saved')}: {new Date(currentDraft.last_saved_at).toLocaleTimeString()}
              </Text>
            )}
          </div>
        </Space>
        <Button 
          icon={<SaveOutlined />} 
          onClick={handleSaveDraft}
        >
          {t('projects.wizard.save_draft', 'Save Draft')}
        </Button>
      </div>

      {/* Steps */}
      <Card className="steps-card">
        <Steps 
          current={currentStep} 
          items={steps}
          responsive
        />
      </Card>

      {/* Content */}
      <Card className="content-card">
        <Spin spinning={createProject.isPending}>
          {currentStep === 0 && (
            <BasicInfoStep
              form={basicForm}
              draft={currentDraft}
              onValuesChange={handleValuesChange}
            />
          )}
          {currentStep === 1 && (
            <AnalysisSettingsStep
              form={analysisForm}
              draft={currentDraft}
              onValuesChange={handleValuesChange}
            />
          )}
          {currentStep === 2 && (
            <NotificationSettingsStep
              form={notificationForm}
              draft={currentDraft}
              onValuesChange={handleValuesChange}
            />
          )}
          {currentStep === 3 && (
            <ReviewStep draft={currentDraft} />
          )}
        </Spin>
      </Card>

      {/* Navigation */}
      <Card className="navigation-card">
        <Row justify="space-between">
          <Col>
            {currentStep > 0 && (
              <Button 
                icon={<ArrowLeftOutlined />} 
                onClick={handlePrevious}
                disabled={createProject.isPending}
              >
                {t('common.previous', 'Previous')}
              </Button>
            )}
          </Col>
          <Col>
            <Space>
              <Button onClick={handleBack} disabled={createProject.isPending}>
                {t('common.cancel', 'Cancel')}
              </Button>
              {currentStep < 3 ? (
                <Button 
                  type="primary" 
                  icon={<ArrowRightOutlined />}
                  onClick={handleNext}
                  disabled={createProject.isPending}
                >
                  {t('common.next', 'Next')}
                </Button>
              ) : (
                <Button 
                  type="primary" 
                  icon={<CheckCircleOutlined />}
                  onClick={handleCreate}
                  loading={createProject.isPending}
                >
                  {t('projects.wizard.create', 'Create Project')}
                </Button>
              )}
            </Space>
          </Col>
        </Row>
      </Card>

      {/* Exit Confirmation Modal */}
      <Modal
        title={t('projects.wizard.exit_title', 'Save your progress?')}
        open={showExitModal}
        onCancel={() => setShowExitModal(false)}
        footer={[
          <Button key="discard" danger onClick={() => confirmExit(false)}>
            {t('projects.wizard.discard', 'Discard')}
          </Button>,
          <Button key="save" type="primary" onClick={() => confirmExit(true)}>
            {t('projects.wizard.save_and_exit', 'Save & Exit')}
          </Button>,
        ]}
      >
        <p>{t('projects.wizard.exit_message', 'You have unsaved changes. Would you like to save your draft before leaving?')}</p>
      </Modal>
    </div>
  );
};

export default NewProject;
