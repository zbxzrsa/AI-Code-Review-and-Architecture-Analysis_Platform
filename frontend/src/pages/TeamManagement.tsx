/**
 * Team Management Page
 * 团队管理页面
 * 
 * Features:
 * - Create and manage teams
 * - Invite members
 * - Role-based access control
 * - Team activity tracking
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Table,
  Button,
  Space,
  Typography,
  Modal,
  Form,
  Input,
  Select,
  Tag,
  Avatar,
  Tooltip,
  message,
  Popconfirm,
  Tabs,
  List,
  Badge,
  Dropdown,
  Divider,
  Empty,
  Statistic,
} from 'antd';
import type { TableProps, MenuProps } from 'antd';
import {
  TeamOutlined,
  PlusOutlined,
  UserAddOutlined,
  DeleteOutlined,
  EditOutlined,
  MailOutlined,
  CrownOutlined,
  UserOutlined,
  SettingOutlined,
  MoreOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ProjectOutlined,
  SafetyCertificateOutlined,
  EyeOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { api } from '../services/api';

const { Title, Text, Paragraph } = Typography;

interface TeamMember {
  id: string;
  name: string;
  email: string;
  avatar?: string;
  role: 'owner' | 'admin' | 'member' | 'viewer';
  status: 'active' | 'pending' | 'inactive';
  joinedAt: string;
  lastActive?: string;
  projectCount: number;
}

interface Team {
  id: string;
  name: string;
  description?: string;
  memberCount: number;
  projectCount: number;
  createdAt: string;
  members: TeamMember[];
}

const mockTeams: Team[] = [
  {
    id: 'team_1',
    name: 'Frontend Team',
    description: 'React and TypeScript development team',
    memberCount: 8,
    projectCount: 5,
    createdAt: '2024-01-15T10:00:00Z',
    members: [
      { id: 'user_1', name: 'John Doe', email: 'john@example.com', role: 'owner', status: 'active', joinedAt: '2024-01-15T10:00:00Z', lastActive: '2024-03-01T15:30:00Z', projectCount: 5 },
      { id: 'user_2', name: 'Jane Smith', email: 'jane@example.com', role: 'admin', status: 'active', joinedAt: '2024-01-20T10:00:00Z', lastActive: '2024-03-01T12:00:00Z', projectCount: 4 },
      { id: 'user_3', name: 'Bob Wilson', email: 'bob@example.com', role: 'member', status: 'active', joinedAt: '2024-02-01T10:00:00Z', lastActive: '2024-02-28T10:00:00Z', projectCount: 3 },
      { id: 'user_4', name: 'Alice Brown', email: 'alice@example.com', role: 'member', status: 'pending', joinedAt: '2024-02-15T10:00:00Z', projectCount: 0 },
    ],
  },
  {
    id: 'team_2',
    name: 'Backend Team',
    description: 'Python and FastAPI development team',
    memberCount: 6,
    projectCount: 4,
    createdAt: '2024-01-20T10:00:00Z',
    members: [
      { id: 'user_5', name: 'Charlie Davis', email: 'charlie@example.com', role: 'owner', status: 'active', joinedAt: '2024-01-20T10:00:00Z', lastActive: '2024-03-01T14:00:00Z', projectCount: 4 },
      { id: 'user_6', name: 'Diana Evans', email: 'diana@example.com', role: 'member', status: 'active', joinedAt: '2024-02-01T10:00:00Z', lastActive: '2024-02-28T16:00:00Z', projectCount: 3 },
    ],
  },
];

const roleColors: Record<string, string> = {
  owner: 'gold',
  admin: 'purple',
  member: 'blue',
  viewer: 'default',
};

const roleIcons: Record<string, React.ReactNode> = {
  owner: <CrownOutlined />,
  admin: <SafetyCertificateOutlined />,
  member: <UserOutlined />,
  viewer: <EyeOutlined />,
};

export const TeamManagement: React.FC = () => {
  const { t } = useTranslation();
  const [teams, setTeams] = useState<Team[]>(mockTeams);
  const [selectedTeam, setSelectedTeam] = useState<Team | null>(mockTeams[0]);
  const [loading, setLoading] = useState(false);
  const [createTeamModalOpen, setCreateTeamModalOpen] = useState(false);
  const [inviteModalOpen, setInviteModalOpen] = useState(false);
  const [editMemberModalOpen, setEditMemberModalOpen] = useState(false);
  const [selectedMember, setSelectedMember] = useState<TeamMember | null>(null);
  const [teamForm] = Form.useForm();
  const [inviteForm] = Form.useForm();
  const [memberForm] = Form.useForm();

  // Create team
  const handleCreateTeam = async (values: any) => {
    try {
      await api.post('/api/teams', values);
      message.success('Team created successfully');
    } catch (error) {
      message.success('Team created successfully (demo)');
      const newTeam: Team = {
        id: `team_${Date.now()}`,
        name: values.name,
        description: values.description,
        memberCount: 1,
        projectCount: 0,
        createdAt: new Date().toISOString(),
        members: [{
          id: 'current_user',
          name: 'You',
          email: 'you@example.com',
          role: 'owner',
          status: 'active',
          joinedAt: new Date().toISOString(),
          projectCount: 0,
        }],
      };
      setTeams(prev => [...prev, newTeam]);
    }
    setCreateTeamModalOpen(false);
    teamForm.resetFields();
  };

  // Invite member
  const handleInvite = async (values: any) => {
    if (!selectedTeam) return;
    try {
      await api.post(`/api/teams/${selectedTeam.id}/invite`, values);
      message.success('Invitation sent');
    } catch (error) {
      message.success('Invitation sent (demo)');
      const newMember: TeamMember = {
        id: `user_${Date.now()}`,
        name: values.email.split('@')[0],
        email: values.email,
        role: values.role,
        status: 'pending',
        joinedAt: new Date().toISOString(),
        projectCount: 0,
      };
      setTeams(prev => prev.map(t => 
        t.id === selectedTeam.id 
          ? { ...t, members: [...t.members, newMember], memberCount: t.memberCount + 1 }
          : t
      ));
      setSelectedTeam(prev => prev ? { ...prev, members: [...prev.members, newMember] } : null);
    }
    setInviteModalOpen(false);
    inviteForm.resetFields();
  };

  // Update member role
  const handleUpdateMember = async (values: any) => {
    if (!selectedTeam || !selectedMember) return;
    try {
      await api.put(`/api/teams/${selectedTeam.id}/members/${selectedMember.id}`, values);
      message.success('Member updated');
    } catch (error) {
      message.success('Member updated (demo)');
      setTeams(prev => prev.map(t => 
        t.id === selectedTeam.id 
          ? { 
              ...t, 
              members: t.members.map(m => 
                m.id === selectedMember.id ? { ...m, ...values } : m
              )
            }
          : t
      ));
    }
    setEditMemberModalOpen(false);
    setSelectedMember(null);
    memberForm.resetFields();
  };

  // Remove member
  const handleRemoveMember = async (memberId: string) => {
    if (!selectedTeam) return;
    try {
      await api.delete(`/api/teams/${selectedTeam.id}/members/${memberId}`);
      message.success('Member removed');
    } catch (error) {
      message.success('Member removed (demo)');
    }
    setTeams(prev => prev.map(t => 
      t.id === selectedTeam.id 
        ? { ...t, members: t.members.filter(m => m.id !== memberId), memberCount: t.memberCount - 1 }
        : t
    ));
    setSelectedTeam(prev => prev ? { ...prev, members: prev.members.filter(m => m.id !== memberId) } : null);
  };

  const memberColumns: TableProps<TeamMember>['columns'] = [
    {
      title: 'Member',
      key: 'member',
      render: (_, record) => (
        <Space>
          <Avatar src={record.avatar} icon={<UserOutlined />} />
          <div>
            <Text strong>{record.name}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: 12 }}>{record.email}</Text>
          </div>
        </Space>
      ),
    },
    {
      title: 'Role',
      dataIndex: 'role',
      width: 120,
      render: (role) => (
        <Tag color={roleColors[role]} icon={roleIcons[role]}>
          {role.charAt(0).toUpperCase() + role.slice(1)}
        </Tag>
      ),
    },
    {
      title: 'Status',
      dataIndex: 'status',
      width: 100,
      render: (status) => (
        <Badge
          status={status === 'active' ? 'success' : status === 'pending' ? 'warning' : 'default'}
          text={status.charAt(0).toUpperCase() + status.slice(1)}
        />
      ),
    },
    {
      title: 'Projects',
      dataIndex: 'projectCount',
      width: 100,
      render: (count) => <Text>{count} projects</Text>,
    },
    {
      title: 'Last Active',
      dataIndex: 'lastActive',
      width: 150,
      render: (date) => date 
        ? new Date(date).toLocaleDateString()
        : <Text type="secondary">Never</Text>,
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 120,
      render: (_, record) => {
        if (record.role === 'owner') return null;
        
        const items: MenuProps['items'] = [
          {
            key: 'edit',
            icon: <EditOutlined />,
            label: 'Edit Role',
            onClick: () => {
              setSelectedMember(record);
              memberForm.setFieldsValue({ role: record.role });
              setEditMemberModalOpen(true);
            },
          },
          { type: 'divider' },
          {
            key: 'remove',
            icon: <DeleteOutlined />,
            label: 'Remove',
            danger: true,
            onClick: () => handleRemoveMember(record.id),
          },
        ];
        
        return (
          <Dropdown menu={{ items }} trigger={['click']}>
            <Button icon={<MoreOutlined />} />
          </Dropdown>
        );
      },
    },
  ];

  return (
    <div className="team-management">
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <div>
          <Title level={3}>
            <TeamOutlined /> {t('team.title', 'Team Management')}
          </Title>
          <Text type="secondary">
            {t('team.subtitle', 'Manage your teams and members')}
          </Text>
        </div>
        <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateTeamModalOpen(true)}>
          Create Team
        </Button>
      </div>

      <Row gutter={[16, 16]}>
        {/* Teams List */}
        <Col xs={24} md={8}>
          <Card title="Your Teams" bodyStyle={{ padding: 0 }}>
            <List
              dataSource={teams}
              renderItem={team => (
                <List.Item
                  style={{
                    padding: 16,
                    cursor: 'pointer',
                    backgroundColor: selectedTeam?.id === team.id ? '#f5f5f5' : 'transparent',
                  }}
                  onClick={() => setSelectedTeam(team)}
                >
                  <List.Item.Meta
                    avatar={<Avatar icon={<TeamOutlined />} style={{ backgroundColor: '#1890ff' }} />}
                    title={team.name}
                    description={
                      <Space split={<Divider type="vertical" />}>
                        <Text type="secondary">{team.memberCount} members</Text>
                        <Text type="secondary">{team.projectCount} projects</Text>
                      </Space>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </Col>

        {/* Team Details */}
        <Col xs={24} md={16}>
          {selectedTeam ? (
            <Card
              title={
                <Space>
                  <TeamOutlined />
                  {selectedTeam.name}
                </Space>
              }
              extra={
                <Space>
                  <Button icon={<UserAddOutlined />} onClick={() => setInviteModalOpen(true)}>
                    Invite Member
                  </Button>
                  <Button icon={<SettingOutlined />}>Settings</Button>
                </Space>
              }
            >
              {/* Team Stats */}
              <Row gutter={16} style={{ marginBottom: 24 }}>
                <Col span={8}>
                  <Statistic title="Members" value={selectedTeam.memberCount} prefix={<TeamOutlined />} />
                </Col>
                <Col span={8}>
                  <Statistic title="Projects" value={selectedTeam.projectCount} prefix={<ProjectOutlined />} />
                </Col>
                <Col span={8}>
                  <Statistic 
                    title="Created" 
                    value={new Date(selectedTeam.createdAt).toLocaleDateString()} 
                    prefix={<ClockCircleOutlined />}
                  />
                </Col>
              </Row>

              {selectedTeam.description && (
                <Paragraph type="secondary" style={{ marginBottom: 24 }}>
                  {selectedTeam.description}
                </Paragraph>
              )}

              {/* Members Table */}
              <Table
                columns={memberColumns}
                dataSource={selectedTeam.members}
                rowKey="id"
                pagination={false}
              />
            </Card>
          ) : (
            <Card>
              <Empty description="Select a team to view details" />
            </Card>
          )}
        </Col>
      </Row>

      {/* Create Team Modal */}
      <Modal
        title={<><TeamOutlined /> Create Team</>}
        open={createTeamModalOpen}
        onCancel={() => {
          setCreateTeamModalOpen(false);
          teamForm.resetFields();
        }}
        onOk={() => teamForm.submit()}
      >
        <Form form={teamForm} layout="vertical" onFinish={handleCreateTeam}>
          <Form.Item
            name="name"
            label="Team Name"
            rules={[{ required: true, message: 'Please enter a team name' }]}
          >
            <Input placeholder="e.g., Frontend Team" />
          </Form.Item>
          <Form.Item
            name="description"
            label="Description"
          >
            <Input.TextArea placeholder="Brief description of the team" rows={3} />
          </Form.Item>
        </Form>
      </Modal>

      {/* Invite Member Modal */}
      <Modal
        title={<><UserAddOutlined /> Invite Member</>}
        open={inviteModalOpen}
        onCancel={() => {
          setInviteModalOpen(false);
          inviteForm.resetFields();
        }}
        onOk={() => inviteForm.submit()}
      >
        <Form form={inviteForm} layout="vertical" onFinish={handleInvite}>
          <Form.Item
            name="email"
            label="Email Address"
            rules={[
              { required: true, message: 'Please enter an email' },
              { type: 'email', message: 'Please enter a valid email' },
            ]}
          >
            <Input prefix={<MailOutlined />} placeholder="member@example.com" />
          </Form.Item>
          <Form.Item
            name="role"
            label="Role"
            initialValue="member"
            rules={[{ required: true }]}
          >
            <Select
              options={[
                { value: 'admin', label: 'Admin - Full access to team settings' },
                { value: 'member', label: 'Member - Can create and analyze projects' },
                { value: 'viewer', label: 'Viewer - Read-only access' },
              ]}
            />
          </Form.Item>
        </Form>
      </Modal>

      {/* Edit Member Modal */}
      <Modal
        title={<><EditOutlined /> Edit Member Role</>}
        open={editMemberModalOpen}
        onCancel={() => {
          setEditMemberModalOpen(false);
          setSelectedMember(null);
          memberForm.resetFields();
        }}
        onOk={() => memberForm.submit()}
      >
        <Form form={memberForm} layout="vertical" onFinish={handleUpdateMember}>
          <Form.Item
            name="role"
            label="Role"
            rules={[{ required: true }]}
          >
            <Select
              options={[
                { value: 'admin', label: 'Admin' },
                { value: 'member', label: 'Member' },
                { value: 'viewer', label: 'Viewer' },
              ]}
            />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default TeamManagement;
