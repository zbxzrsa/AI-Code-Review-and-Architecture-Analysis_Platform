/**
 * Register Page Component
 * 注册页面组件
 * 
 * This component handles new user registration with invitation code validation.
 * 此组件处理新用户注册，包含邀请码验证功能。
 * 
 * Features / 功能:
 * - Email and password validation / 邮箱和密码验证
 * - Invitation code requirement / 邀请码要求
 * - Password strength indicator / 密码强度指示器
 * - Form validation with i18n / 带国际化的表单验证
 */

import React, { useState, useMemo } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { 
  Form, 
  Input, 
  Button, 
  Card, 
  Typography, 
  Alert, 
  Divider,
  Progress,
  Space
} from 'antd';
import { 
  UserOutlined, 
  LockOutlined, 
  KeyOutlined, 
  MailOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../hooks/useAuth';
import './Register.css';

const { Title, Text } = Typography;

/**
 * Registration form values interface
 * 注册表单值接口
 */
interface RegisterFormValues {
  name: string;           // 用户名 / Username
  email: string;          // 邮箱 / Email
  password: string;       // 密码 / Password
  confirmPassword: string; // 确认密码 / Confirm password
  invitation_code: string; // 邀请码 / Invitation code
}

/**
 * Password strength calculation
 * 密码强度计算
 * 
 * @param password - The password to evaluate / 要评估的密码
 * @returns Strength score (0-100) and status / 强度分数(0-100)和状态
 */
const calculatePasswordStrength = (password: string): { score: number; status: 'exception' | 'normal' | 'success' } => {
  let score = 0;
  
  // 长度检查 / Length check
  if (password.length >= 8) score += 25;
  if (password.length >= 12) score += 15;
  
  // 包含小写字母 / Contains lowercase
  if (/[a-z]/.test(password)) score += 15;
  
  // 包含大写字母 / Contains uppercase
  if (/[A-Z]/.test(password)) score += 15;
  
  // 包含数字 / Contains number
  if (/\d/.test(password)) score += 15;
  
  // 包含特殊字符 / Contains special character
  if (/[!@#$%^&*(),.?":{}|<>]/.test(password)) score += 15;
  
  // 确定状态 / Determine status
  const status = score < 40 ? 'exception' : score < 70 ? 'normal' : 'success';
  
  return { score: Math.min(score, 100), status };
};

/**
 * Register Component
 * 注册组件
 */
export const Register: React.FC = () => {
  // 国际化钩子 / Internationalization hook
  const { t } = useTranslation();
  
  // 路由导航钩子 / Router navigation hook
  const navigate = useNavigate();
  
  // 认证钩子 / Authentication hook
  const { register, isLoading, error } = useAuth();
  
  // 密码状态用于强度计算 / Password state for strength calculation
  const [password, setPassword] = useState('');
  
  // 表单实例 / Form instance
  const [form] = Form.useForm();

  /**
   * Calculate password strength memo
   * 计算密码强度的缓存值
   */
  const passwordStrength = useMemo(() => {
    return calculatePasswordStrength(password);
  }, [password]);

  /**
   * Password validation rules
   * 密码验证规则列表
   */
  const passwordRules = useMemo(() => [
    { 
      valid: password.length >= 8, 
      text: t('register.password_length', 'At least 8 characters / 至少8个字符') 
    },
    { 
      valid: /[a-z]/.test(password), 
      text: t('register.password_lowercase', 'One lowercase letter / 一个小写字母') 
    },
    { 
      valid: /[A-Z]/.test(password), 
      text: t('register.password_uppercase', 'One uppercase letter / 一个大写字母') 
    },
    { 
      valid: /\d/.test(password), 
      text: t('register.password_number', 'One number / 一个数字') 
    },
  ], [password, t]);

  /**
   * Handle form submission
   * 处理表单提交
   * 
   * @param values - Form values / 表单值
   */
  const onFinish = async (values: RegisterFormValues) => {
    // 调用注册API / Call register API
    const success = await register({
      name: values.name,
      email: values.email,
      password: values.password,
      invitation_code: values.invitation_code,
    });
    
    // 注册成功后跳转到仪表板 / Navigate to dashboard on success
    if (success) {
      navigate('/dashboard');
    }
  };

  return (
    <div className="register-container">
      {/* 背景装饰 / Background decoration */}
      <div className="register-background" />
      
      <Card className="register-card">
        {/* 页面标题 / Page header */}
        <div className="register-header">
          <Title level={2}>{t('register.title', 'Create Account')}</Title>
          <Text type="secondary">
            {t('register.subtitle', 'Join the AI Code Review Platform')}
          </Text>
        </div>

        {/* 错误提示 / Error alert */}
        {error && (
          <Alert
            message={typeof error === 'string' ? error : (error as any)?.message || (error as any)?.msg || t('register.error', 'Registration failed')}
            type="error"
            showIcon
            closable
            className="register-error"
          />
        )}

        {/* 注册表单 / Registration form */}
        <Form
          form={form}
          name="register"
          onFinish={onFinish}
          layout="vertical"
          size="large"
          requiredMark={false}
        >
          {/* 用户名输入 / Name input */}
          <Form.Item
            name="name"
            rules={[
              { required: true, message: t('register.name_required', 'Please enter your name') },
              { min: 2, message: t('register.name_min', 'Name must be at least 2 characters') },
              { max: 50, message: t('register.name_max', 'Name cannot exceed 50 characters') }
            ]}
          >
            <Input
              prefix={<UserOutlined />}
              placeholder={t('register.name_placeholder', 'Full Name / 姓名')}
              autoComplete="name"
            />
          </Form.Item>

          {/* 邮箱输入 / Email input */}
          <Form.Item
            name="email"
            rules={[
              { required: true, message: t('register.email_required', 'Please enter your email') },
              { type: 'email', message: t('register.email_invalid', 'Please enter a valid email') }
            ]}
          >
            <Input
              prefix={<MailOutlined />}
              placeholder={t('register.email_placeholder', 'Email / 邮箱')}
              autoComplete="email"
            />
          </Form.Item>

          {/* 密码输入 / Password input */}
          <Form.Item
            name="password"
            rules={[
              { required: true, message: t('register.password_required', 'Please enter a password') },
              { min: 8, message: t('register.password_min', 'Password must be at least 8 characters') }
            ]}
          >
            <Input.Password
              prefix={<LockOutlined />}
              placeholder={t('register.password_placeholder', 'Password / 密码')}
              autoComplete="new-password"
              onChange={(e) => setPassword(e.target.value)}
            />
          </Form.Item>

          {/* 密码强度指示器 / Password strength indicator */}
          {password && (
            <div className="password-strength">
              <Progress 
                percent={passwordStrength.score} 
                status={passwordStrength.status}
                showInfo={false}
                size="small"
              />
              <Space direction="vertical" size={4} className="password-rules">
                {passwordRules.map((rule, index) => (
                  <Text 
                    key={index} 
                    type={rule.valid ? 'success' : 'secondary'}
                    className="password-rule"
                  >
                    {rule.valid ? <CheckCircleOutlined /> : <CloseCircleOutlined />}
                    {' '}{rule.text}
                  </Text>
                ))}
              </Space>
            </div>
          )}

          {/* 确认密码输入 / Confirm password input */}
          <Form.Item
            name="confirmPassword"
            dependencies={['password']}
            rules={[
              { required: true, message: t('register.confirm_required', 'Please confirm your password') },
              ({ getFieldValue }) => ({
                validator(_, value) {
                  if (!value || getFieldValue('password') === value) {
                    return Promise.resolve();
                  }
                  return Promise.reject(new Error(t('register.password_mismatch', 'Passwords do not match')));
                },
              }),
            ]}
          >
            <Input.Password
              prefix={<LockOutlined />}
              placeholder={t('register.confirm_placeholder', 'Confirm Password / 确认密码')}
              autoComplete="new-password"
            />
          </Form.Item>

          {/* 邀请码输入 / Invitation code input */}
          <Form.Item
            name="invitation_code"
            rules={[
              { required: true, message: t('register.invitation_required', 'Invitation code is required') }
            ]}
            extra={
              <Text type="secondary" style={{ fontSize: 12 }}>
                {t('register.invitation_help', 'Contact admin to get an invitation code / 联系管理员获取邀请码')}
              </Text>
            }
          >
            <Input
              prefix={<KeyOutlined />}
              placeholder={t('register.invitation_placeholder', 'Invitation Code / 邀请码')}
            />
          </Form.Item>

          {/* 提交按钮 / Submit button */}
          <Form.Item>
            <Button
              type="primary"
              htmlType="submit"
              loading={isLoading}
              block
            >
              {t('register.submit', 'Create Account / 创建账户')}
            </Button>
          </Form.Item>
        </Form>

        {/* 页脚链接 / Footer links */}
        <div className="register-footer">
          <Divider plain>{t('register.or', 'or')}</Divider>
          
          <Text>
            {t('register.have_account', 'Already have an account?')}{' '}
            <Link to="/login">{t('register.login', 'Sign In / 登录')}</Link>
          </Text>
        </div>
      </Card>
    </div>
  );
};

export default Register;
