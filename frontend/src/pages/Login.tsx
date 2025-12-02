import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Form, Input, Button, Card, Typography, Alert, Divider } from 'antd';
import { UserOutlined, LockOutlined, KeyOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../hooks/useAuth';
import './Login.css';

const { Title, Text } = Typography;

interface LoginFormValues {
  email: string;
  password: string;
  invitation_code?: string;
}

export const Login: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { login, isLoading, error } = useAuth();
  const [showInvitationCode, setShowInvitationCode] = useState(false);

  const onFinish = async (values: LoginFormValues) => {
    const success = await login(values);
    if (success) {
      navigate('/dashboard');
    }
  };

  return (
    <div className="login-container">
      <div className="login-background" />
      
      <Card className="login-card">
        <div className="login-header">
          <Title level={2}>{t('login.title', 'Welcome Back')}</Title>
          <Text type="secondary">
            {t('login.subtitle', 'Sign in to continue to Code Review Platform')}
          </Text>
        </div>

        {error && (
          <Alert
            message={error}
            type="error"
            showIcon
            closable
            className="login-error"
          />
        )}

        <Form
          name="login"
          onFinish={onFinish}
          layout="vertical"
          size="large"
          requiredMark={false}
        >
          <Form.Item
            name="email"
            rules={[
              { required: true, message: t('login.email_required', 'Please enter your email') },
              { type: 'email', message: t('login.email_invalid', 'Please enter a valid email') }
            ]}
          >
            <Input
              prefix={<UserOutlined />}
              placeholder={t('login.email_placeholder', 'Email')}
              autoComplete="email"
            />
          </Form.Item>

          <Form.Item
            name="password"
            rules={[
              { required: true, message: t('login.password_required', 'Please enter your password') }
            ]}
          >
            <Input.Password
              prefix={<LockOutlined />}
              placeholder={t('login.password_placeholder', 'Password')}
              autoComplete="current-password"
            />
          </Form.Item>

          {showInvitationCode && (
            <Form.Item
              name="invitation_code"
              rules={[
                { required: true, message: t('login.invitation_required', 'Please enter invitation code') }
              ]}
            >
              <Input
                prefix={<KeyOutlined />}
                placeholder={t('login.invitation_placeholder', 'Invitation Code')}
              />
            </Form.Item>
          )}

          <Form.Item>
            <Button
              type="primary"
              htmlType="submit"
              loading={isLoading}
              block
            >
              {t('login.submit', 'Sign In')}
            </Button>
          </Form.Item>
        </Form>

        <div className="login-footer">
          <Button
            type="link"
            onClick={() => setShowInvitationCode(!showInvitationCode)}
          >
            {showInvitationCode
              ? t('login.hide_invitation', 'Hide invitation code')
              : t('login.have_invitation', 'Have an invitation code?')}
          </Button>

          <Divider plain>{t('login.or', 'or')}</Divider>

          <Text>
            {t('login.no_account', "Don't have an account?")}{' '}
            <Link to="/register">{t('login.register', 'Register')}</Link>
          </Text>
        </div>
      </Card>
    </div>
  );
};

export default Login;
