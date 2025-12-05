import React, { useState } from 'react';
import {
  Card,
  Row,
  Col,
  Avatar,
  Typography,
  Button,
  Form,
  Input,
  Upload,
  Space,
  Statistic,
  Tag,
  message,
  List,
  Tooltip
} from 'antd';
import {
  UserOutlined,
  EditOutlined,
  CameraOutlined,
  MailOutlined,
  CalendarOutlined,
  ProjectOutlined,
  BugOutlined,
  CheckCircleOutlined,
  TrophyOutlined,
  FireOutlined
} from '@ant-design/icons';
import type { UploadProps } from 'antd';
import { useTranslation } from 'react-i18next';
import { useAuthStore } from '../store/authStore';
import { apiService } from '../services/api';
import './Profile.css';

const { Title, Text } = Typography;

interface Activity {
  id: string;
  type: 'analysis' | 'fix' | 'review' | 'project';
  description: string;
  project: string;
  timestamp: string;
}

interface Achievement {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  unlocked: boolean;
  unlockedAt?: string;
}

export const Profile: React.FC = () => {
  const { t } = useTranslation();
  const { user, setUser } = useAuthStore();
  const [isEditing, setIsEditing] = useState(false);
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);

  // Mock data
  const stats = {
    projects: 12,
    analyses: 156,
    issues_found: 847,
    issues_fixed: 723,
    streak: 15
  };

  const recentActivity: Activity[] = [
    {
      id: '1',
      type: 'analysis',
      description: 'Completed security analysis',
      project: 'backend-api',
      timestamp: new Date().toISOString()
    },
    {
      id: '2',
      type: 'fix',
      description: 'Fixed 12 code style issues',
      project: 'frontend-app',
      timestamp: new Date(Date.now() - 3600000).toISOString()
    },
    {
      id: '3',
      type: 'project',
      description: 'Created new project',
      project: 'ml-pipeline',
      timestamp: new Date(Date.now() - 86400000).toISOString()
    }
  ];

  const achievements: Achievement[] = [
    {
      id: '1',
      name: 'First Analysis',
      description: 'Complete your first code analysis',
      icon: <BugOutlined />,
      unlocked: true,
      unlockedAt: '2024-01-15'
    },
    {
      id: '2',
      name: 'Bug Hunter',
      description: 'Find 100 issues',
      icon: <TrophyOutlined />,
      unlocked: true,
      unlockedAt: '2024-02-20'
    },
    {
      id: '3',
      name: 'Streak Master',
      description: 'Maintain a 30-day streak',
      icon: <FireOutlined />,
      unlocked: false
    },
    {
      id: '4',
      name: 'Perfectionist',
      description: 'Fix 1000 issues',
      icon: <CheckCircleOutlined />,
      unlocked: false
    }
  ];

  // Handle profile update
  const handleUpdateProfile = async (values: any) => {
    setLoading(true);
    try {
      await apiService.auth.updateProfile(values);
      setUser({ ...user!, ...values });
      message.success(t('profile.updated', 'Profile updated successfully'));
      setIsEditing(false);
    } catch (error) {
      message.error(t('profile.update_error', 'Failed to update profile'));
    } finally {
      setLoading(false);
    }
  };

  // Handle avatar upload
  const uploadProps: UploadProps = {
    name: 'avatar',
    showUploadList: false,
    beforeUpload: (file) => {
      const isImage = file.type.startsWith('image/');
      if (!isImage) {
        message.error(t('profile.avatar_type_error', 'You can only upload image files!'));
        return Upload.LIST_IGNORE;  // Reject: wrong type
      }
      const isLt2M = file.size / 1024 / 1024 < 2;
      if (!isLt2M) {
        message.error(t('profile.avatar_size_error', 'Image must be smaller than 2MB!'));
        return Upload.LIST_IGNORE;  // Reject: too large
      }
      // In production, this would upload to server
      const reader = new FileReader();
      reader.onload = () => {
        // Mock avatar update
        message.success(t('profile.avatar_updated', 'Avatar updated'));
      };
      reader.readAsDataURL(file);
      return false;  // Prevent auto upload, handle manually
    }
  };

  // Get activity icon
  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'analysis': return <BugOutlined style={{ color: '#1890ff' }} />;
      case 'fix': return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'review': return <EditOutlined style={{ color: '#722ed1' }} />;
      case 'project': return <ProjectOutlined style={{ color: '#fa8c16' }} />;
      default: return <UserOutlined />;
    }
  };

  return (
    <div className="profile-container">
      <Row gutter={[24, 24]}>
        {/* Profile Card */}
        <Col xs={24} lg={8}>
          <Card>
            <div style={{ textAlign: 'center', marginBottom: 24 }}>
              <div style={{ position: 'relative', display: 'inline-block' }}>
                <Avatar
                  size={120}
                  src={user?.avatar}
                  icon={!user?.avatar && <UserOutlined />}
                  style={{ backgroundColor: '#1890ff' }}
                />
                <Upload {...uploadProps}>
                  <Button
                    type="primary"
                    shape="circle"
                    icon={<CameraOutlined />}
                    size="small"
                    style={{
                      position: 'absolute',
                      bottom: 0,
                      right: 0
                    }}
                  />
                </Upload>
              </div>
              <Title level={3} style={{ marginTop: 16, marginBottom: 4 }}>
                {user?.name || 'User'}
              </Title>
              <Tag color="blue">{user?.role || 'user'}</Tag>
            </div>

            {isEditing ? (
              <Form
                form={form}
                layout="vertical"
                onFinish={handleUpdateProfile}
                initialValues={{
                  name: user?.name,
                  email: user?.email
                }}
              >
                <Form.Item
                  name="name"
                  label={t('profile.name', 'Name')}
                  rules={[{ required: true, message: t('profile.name_required', 'Please enter your name') }]}
                >
                  <Input prefix={<UserOutlined />} />
                </Form.Item>
                <Form.Item
                  name="email"
                  label={t('profile.email', 'Email')}
                  rules={[
                    { required: true, message: t('profile.email_required', 'Please enter your email') },
                    { type: 'email', message: t('profile.email_invalid', 'Please enter a valid email') }
                  ]}
                >
                  <Input prefix={<MailOutlined />} disabled />
                </Form.Item>
                <Form.Item>
                  <Space>
                    <Button type="primary" htmlType="submit" loading={loading}>
                      {t('common.save', 'Save')}
                    </Button>
                    <Button onClick={() => setIsEditing(false)}>
                      {t('common.cancel', 'Cancel')}
                    </Button>
                  </Space>
                </Form.Item>
              </Form>
            ) : (
              <>
                <div style={{ marginBottom: 16 }}>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div>
                      <MailOutlined style={{ marginRight: 8 }} />
                      <Text>{user?.email}</Text>
                    </div>
                    <div>
                      <CalendarOutlined style={{ marginRight: 8 }} />
                      <Text type="secondary">
                        {t('profile.joined', 'Joined')} {user?.createdAt ? new Date(user.createdAt).toLocaleDateString() : 'Recently'}
                      </Text>
                    </div>
                  </Space>
                </div>
                <Button
                  type="primary"
                  icon={<EditOutlined />}
                  block
                  onClick={() => {
                    form.setFieldsValue({
                      name: user?.name,
                      email: user?.email
                    });
                    setIsEditing(true);
                  }}
                >
                  {t('profile.edit', 'Edit Profile')}
                </Button>
              </>
            )}
          </Card>

          {/* Achievements */}
          <Card title={t('profile.achievements', 'Achievements')} style={{ marginTop: 24 }}>
            <Row gutter={[16, 16]}>
              {achievements.map((achievement) => (
                <Col span={12} key={achievement.id}>
                  <Tooltip
                    title={
                      <>
                        <div>{achievement.name}</div>
                        <div style={{ fontSize: 12, opacity: 0.8 }}>{achievement.description}</div>
                        {achievement.unlockedAt && (
                          <div style={{ fontSize: 12, opacity: 0.6 }}>
                            Unlocked: {new Date(achievement.unlockedAt).toLocaleDateString()}
                          </div>
                        )}
                      </>
                    }
                  >
                    <div
                      style={{
                        textAlign: 'center',
                        padding: 12,
                        borderRadius: 8,
                        background: achievement.unlocked ? 'rgba(24, 144, 255, 0.1)' : 'rgba(0, 0, 0, 0.04)',
                        opacity: achievement.unlocked ? 1 : 0.5,
                        cursor: 'pointer'
                      }}
                    >
                      <div style={{ fontSize: 24, marginBottom: 4 }}>
                        {achievement.icon}
                      </div>
                      <Text strong style={{ fontSize: 12 }}>
                        {achievement.name}
                      </Text>
                    </div>
                  </Tooltip>
                </Col>
              ))}
            </Row>
          </Card>
        </Col>

        {/* Stats and Activity */}
        <Col xs={24} lg={16}>
          {/* Stats */}
          <Card>
            <Row gutter={[16, 16]}>
              <Col xs={12} sm={8}>
                <Statistic
                  title={t('profile.projects', 'Projects')}
                  value={stats.projects}
                  prefix={<ProjectOutlined />}
                />
              </Col>
              <Col xs={12} sm={8}>
                <Statistic
                  title={t('profile.analyses', 'Analyses')}
                  value={stats.analyses}
                  prefix={<BugOutlined />}
                />
              </Col>
              <Col xs={12} sm={8}>
                <Statistic
                  title={t('profile.streak', 'Day Streak')}
                  value={stats.streak}
                  prefix={<FireOutlined />}
                  valueStyle={{ color: '#fa8c16' }}
                />
              </Col>
              <Col xs={12} sm={8}>
                <Statistic
                  title={t('profile.issues_found', 'Issues Found')}
                  value={stats.issues_found}
                  valueStyle={{ color: '#cf1322' }}
                />
              </Col>
              <Col xs={12} sm={8}>
                <Statistic
                  title={t('profile.issues_fixed', 'Issues Fixed')}
                  value={stats.issues_fixed}
                  valueStyle={{ color: '#3f8600' }}
                />
              </Col>
              <Col xs={12} sm={8}>
                <Statistic
                  title={t('profile.fix_rate', 'Fix Rate')}
                  value={Math.round((stats.issues_fixed / stats.issues_found) * 100)}
                  suffix="%"
                  valueStyle={{ color: '#1890ff' }}
                />
              </Col>
            </Row>
          </Card>

          {/* Recent Activity */}
          <Card
            title={t('profile.recent_activity', 'Recent Activity')}
            style={{ marginTop: 24 }}
          >
            <List
              dataSource={recentActivity}
              renderItem={(item) => (
                <List.Item>
                  <List.Item.Meta
                    avatar={
                      <Avatar icon={getActivityIcon(item.type)} />
                    }
                    title={
                      <Space>
                        <Text strong>{item.project}</Text>
                        <Text type="secondary">•</Text>
                        <Text type="secondary">{item.description}</Text>
                      </Space>
                    }
                    description={
                      <Text type="secondary">
                        {new Date(item.timestamp).toLocaleString()}
                      </Text>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>

          {/* Contribution Graph (Placeholder) */}
          <Card
            title={t('profile.contributions', 'Contribution Graph')}
            style={{ marginTop: 24 }}
          >
            <div
              style={{
                height: 120,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'rgba(0, 0, 0, 0.02)',
                borderRadius: 8
              }}
            >
              <Text type="secondary">
                {t('profile.contributions_placeholder', 'Contribution graph coming soon')}
              </Text>
            </div>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Profile;
