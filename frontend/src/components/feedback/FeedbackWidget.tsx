/**
 * User Feedback Widget
 * 
 * Collects user satisfaction and feedback for product iteration:
 * - Quick rating (1-5 stars)
 * - Detailed feedback form
 * - Feature request submission
 * - Bug report
 * - NPS scoring
 */

import React, { useState, useCallback } from 'react';
import {
  Button,
  Modal,
  Rate,
  Input,
  Select,
  Form,
  Space,
  Typography,
  Tooltip,
  message,
  Segmented,
  Tag,
} from 'antd';
import {
  SmileOutlined,
  FrownOutlined,
  MehOutlined,
  BugOutlined,
  BulbOutlined,
  MessageOutlined,
  SendOutlined,
  HeartOutlined,
  StarOutlined,
} from '@ant-design/icons';
import './FeedbackWidget.css';

const { TextArea } = Input;
const { Text, Title } = Typography;

// Feedback types
type FeedbackType = 'rating' | 'feature' | 'bug' | 'general' | 'nps';

interface FeedbackData {
  type: FeedbackType;
  rating?: number;
  npsScore?: number;
  category?: string;
  title?: string;
  description: string;
  email?: string;
  page?: string;
  timestamp: number;
  userAgent: string;
  screenResolution: string;
}

interface FeedbackWidgetProps {
  position?: 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left';
  defaultType?: FeedbackType;
  onSubmit?: (feedback: FeedbackData) => Promise<void>;
  showTriggerButton?: boolean;
  triggerButtonText?: string;
}

const FEEDBACK_CATEGORIES = [
  { value: 'ui', label: 'User Interface' },
  { value: 'performance', label: 'Performance' },
  { value: 'features', label: 'Features' },
  { value: 'documentation', label: 'Documentation' },
  { value: 'analysis', label: 'Code Analysis' },
  { value: 'ai', label: 'AI Features' },
  { value: 'other', label: 'Other' },
];

const SATISFACTION_EMOJIS = [
  { icon: <FrownOutlined />, label: 'Very Dissatisfied', color: '#ef4444' },
  { icon: <FrownOutlined />, label: 'Dissatisfied', color: '#f97316' },
  { icon: <MehOutlined />, label: 'Neutral', color: '#eab308' },
  { icon: <SmileOutlined />, label: 'Satisfied', color: '#22c55e' },
  { icon: <SmileOutlined />, label: 'Very Satisfied', color: '#06b6d4' },
];

export const FeedbackWidget: React.FC<FeedbackWidgetProps> = ({
  position = 'bottom-right',
  defaultType = 'rating',
  onSubmit,
  showTriggerButton = true,
  triggerButtonText = 'Feedback',
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [feedbackType, setFeedbackType] = useState<FeedbackType>(defaultType);
  const [rating, setRating] = useState<number>(0);
  const [npsScore, setNpsScore] = useState<number | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [form] = Form.useForm();

  const handleOpen = useCallback(() => {
    setIsOpen(true);
    setFeedbackType(defaultType);
    setRating(0);
    setNpsScore(null);
    form.resetFields();
  }, [defaultType, form]);

  const handleClose = useCallback(() => {
    setIsOpen(false);
  }, []);

  const handleSubmit = useCallback(async () => {
    try {
      const values = await form.validateFields();
      setIsSubmitting(true);

      const feedbackData: FeedbackData = {
        type: feedbackType,
        rating: feedbackType === 'rating' ? rating : undefined,
        npsScore: feedbackType === 'nps' ? (npsScore ?? undefined) : undefined,
        category: values.category,
        title: values.title,
        description: values.description || '',
        email: values.email,
        page: window.location.pathname,
        timestamp: Date.now(),
        userAgent: navigator.userAgent,
        screenResolution: `${window.screen.width}x${window.screen.height}`,
      };

      if (onSubmit) {
        await onSubmit(feedbackData);
      } else {
        // Default: store in localStorage for demo
        const existingFeedback = JSON.parse(localStorage.getItem('user_feedback') || '[]');
        existingFeedback.push(feedbackData);
        localStorage.setItem('user_feedback', JSON.stringify(existingFeedback));
      }

      message.success('Thank you for your feedback! 🎉');
      handleClose();
    } catch (error) {
      console.error('Feedback submission error:', error);
      message.error('Failed to submit feedback. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  }, [feedbackType, rating, npsScore, form, onSubmit, handleClose]);

  const renderRatingForm = () => (
    <div className="feedback-rating-form">
      <div className="rating-header">
        <Title level={4}>How satisfied are you?</Title>
        <Text type="secondary">Rate your experience with our platform</Text>
      </div>

      <div className="emoji-rating">
        {SATISFACTION_EMOJIS.map((emoji, index) => (
          <Tooltip key={index} title={emoji.label}>
            <div
              className={`emoji-option ${rating === index + 1 ? 'selected' : ''}`}
              onClick={() => setRating(index + 1)}
              style={{ 
                '--emoji-color': emoji.color,
                borderColor: rating === index + 1 ? emoji.color : undefined,
              } as React.CSSProperties}
            >
              {React.cloneElement(emoji.icon as React.ReactElement, {
                style: { fontSize: 32, color: rating === index + 1 ? emoji.color : '#94a3b8' },
              })}
            </div>
          </Tooltip>
        ))}
      </div>

      <div className="star-rating">
        <Rate
          value={rating}
          onChange={setRating}
          character={<StarOutlined />}
          style={{ fontSize: 28 }}
        />
        <Text type="secondary" style={{ marginLeft: 12 }}>
          {rating > 0 ? SATISFACTION_EMOJIS[rating - 1].label : 'Select a rating'}
        </Text>
      </div>

      <Form form={form} layout="vertical" style={{ marginTop: 24 }}>
        <Form.Item name="description" label="What can we improve?">
          <TextArea
            rows={4}
            placeholder="Tell us about your experience..."
            maxLength={1000}
            showCount
          />
        </Form.Item>
        <Form.Item name="email" label="Email (optional)">
          <Input placeholder="your@email.com" type="email" />
        </Form.Item>
      </Form>
    </div>
  );

  const renderNPSForm = () => (
    <div className="feedback-nps-form">
      <div className="nps-header">
        <Title level={4}>How likely are you to recommend us?</Title>
        <Text type="secondary">On a scale of 0-10</Text>
      </div>

      <div className="nps-scale">
        {Array.from({ length: 11 }, (_, i) => (
          <div
            key={i}
            className={`nps-option ${npsScore === i ? 'selected' : ''} ${
              i <= 6 ? 'detractor' : i <= 8 ? 'passive' : 'promoter'
            }`}
            onClick={() => setNpsScore(i)}
          >
            {i}
          </div>
        ))}
      </div>

      <div className="nps-labels">
        <Text type="secondary">Not likely</Text>
        <Text type="secondary">Very likely</Text>
      </div>

      {npsScore !== null && (
        <div className="nps-feedback">
          <Tag color={npsScore <= 6 ? 'error' : npsScore <= 8 ? 'warning' : 'success'}>
            {npsScore <= 6 ? 'Detractor' : npsScore <= 8 ? 'Passive' : 'Promoter'}
          </Tag>
        </div>
      )}

      <Form form={form} layout="vertical" style={{ marginTop: 24 }}>
        <Form.Item
          name="description"
          label={
            npsScore !== null && npsScore <= 6
              ? "What would make you more likely to recommend us?"
              : "What do you like most about our platform?"
          }
        >
          <TextArea rows={4} placeholder="Your thoughts..." maxLength={1000} showCount />
        </Form.Item>
      </Form>
    </div>
  );

  const renderFeatureForm = () => (
    <div className="feedback-feature-form">
      <div className="form-header">
        <BulbOutlined style={{ fontSize: 32, color: '#f59e0b' }} />
        <Title level={4}>Feature Request</Title>
        <Text type="secondary">Help us build what you need</Text>
      </div>

      <Form form={form} layout="vertical">
        <Form.Item
          name="category"
          label="Category"
          rules={[{ required: true, message: 'Please select a category' }]}
        >
          <Select options={FEEDBACK_CATEGORIES} placeholder="Select category" />
        </Form.Item>

        <Form.Item
          name="title"
          label="Feature Title"
          rules={[{ required: true, message: 'Please provide a title' }]}
        >
          <Input placeholder="Brief title for your feature request" maxLength={100} />
        </Form.Item>

        <Form.Item
          name="description"
          label="Description"
          rules={[{ required: true, message: 'Please describe the feature' }]}
        >
          <TextArea
            rows={4}
            placeholder="Describe the feature you'd like to see..."
            maxLength={2000}
            showCount
          />
        </Form.Item>

        <Form.Item name="email" label="Email (for updates)">
          <Input placeholder="your@email.com" type="email" />
        </Form.Item>
      </Form>
    </div>
  );

  const renderBugForm = () => (
    <div className="feedback-bug-form">
      <div className="form-header">
        <BugOutlined style={{ fontSize: 32, color: '#ef4444' }} />
        <Title level={4}>Bug Report</Title>
        <Text type="secondary">Help us fix issues</Text>
      </div>

      <Form form={form} layout="vertical">
        <Form.Item
          name="category"
          label="Area"
          rules={[{ required: true, message: 'Please select an area' }]}
        >
          <Select options={FEEDBACK_CATEGORIES} placeholder="Select area" />
        </Form.Item>

        <Form.Item
          name="title"
          label="Bug Summary"
          rules={[{ required: true, message: 'Please provide a summary' }]}
        >
          <Input placeholder="Brief description of the issue" maxLength={100} />
        </Form.Item>

        <Form.Item
          name="description"
          label="Steps to Reproduce"
          rules={[{ required: true, message: 'Please describe how to reproduce' }]}
        >
          <TextArea
            rows={4}
            placeholder="1. Go to...&#10;2. Click on...&#10;3. See error..."
            maxLength={2000}
            showCount
          />
        </Form.Item>

        <Form.Item name="expected" label="Expected Behavior">
          <TextArea rows={2} placeholder="What should happen?" maxLength={500} />
        </Form.Item>

        <Form.Item name="email" label="Email (for follow-up)">
          <Input placeholder="your@email.com" type="email" />
        </Form.Item>
      </Form>

      <div className="system-info">
        <Text type="secondary">
          System info will be automatically included: Browser, Screen size, Page URL
        </Text>
      </div>
    </div>
  );

  const renderGeneralForm = () => (
    <div className="feedback-general-form">
      <div className="form-header">
        <MessageOutlined style={{ fontSize: 32, color: '#06b6d4' }} />
        <Title level={4}>General Feedback</Title>
        <Text type="secondary">Share your thoughts</Text>
      </div>

      <Form form={form} layout="vertical">
        <Form.Item name="category" label="Topic">
          <Select options={FEEDBACK_CATEGORIES} placeholder="Select topic (optional)" />
        </Form.Item>

        <Form.Item
          name="description"
          label="Your Feedback"
          rules={[{ required: true, message: 'Please enter your feedback' }]}
        >
          <TextArea
            rows={5}
            placeholder="What's on your mind?"
            maxLength={2000}
            showCount
          />
        </Form.Item>

        <Form.Item name="email" label="Email (optional)">
          <Input placeholder="your@email.com" type="email" />
        </Form.Item>
      </Form>
    </div>
  );

  const renderContent = () => {
    switch (feedbackType) {
      case 'rating':
        return renderRatingForm();
      case 'nps':
        return renderNPSForm();
      case 'feature':
        return renderFeatureForm();
      case 'bug':
        return renderBugForm();
      case 'general':
        return renderGeneralForm();
      default:
        return renderRatingForm();
    }
  };

  const canSubmit = () => {
    if (feedbackType === 'rating') return rating > 0;
    if (feedbackType === 'nps') return npsScore !== null;
    return true;
  };

  return (
    <>
      {showTriggerButton && (
        <Tooltip title="Share your feedback">
          <Button
            className={`feedback-trigger-btn ${position}`}
            type="primary"
            icon={<HeartOutlined />}
            onClick={handleOpen}
          >
            {triggerButtonText}
          </Button>
        </Tooltip>
      )}

      <Modal
        open={isOpen}
        onCancel={handleClose}
        footer={null}
        width={520}
        className="feedback-modal"
        destroyOnClose
      >
        <div className="feedback-type-selector">
          <Segmented
            value={feedbackType}
            onChange={(value) => setFeedbackType(value as FeedbackType)}
            options={[
              { value: 'rating', label: 'Rate', icon: <StarOutlined /> },
              { value: 'nps', label: 'NPS', icon: <SmileOutlined /> },
              { value: 'feature', label: 'Feature', icon: <BulbOutlined /> },
              { value: 'bug', label: 'Bug', icon: <BugOutlined /> },
              { value: 'general', label: 'Other', icon: <MessageOutlined /> },
            ]}
            block
          />
        </div>

        <div className="feedback-content">{renderContent()}</div>

        <div className="feedback-actions">
          <Button onClick={handleClose}>Cancel</Button>
          <Button
            type="primary"
            icon={<SendOutlined />}
            onClick={handleSubmit}
            loading={isSubmitting}
            disabled={!canSubmit()}
          >
            Submit Feedback
          </Button>
        </div>
      </Modal>
    </>
  );
};

// Quick feedback component for inline use
export const QuickFeedback: React.FC<{
  question?: string;
  onFeedback?: (positive: boolean, comment?: string) => void;
}> = ({ question = 'Was this helpful?', onFeedback }) => {
  const [submitted, setSubmitted] = useState(false);
  const [showComment, setShowComment] = useState(false);
  const [comment, setComment] = useState('');

  const handleFeedback = (positive: boolean) => {
    if (!positive) {
      setShowComment(true);
      return;
    }
    
    onFeedback?.(positive);
    setSubmitted(true);
    message.success('Thanks for your feedback!');
  };

  const handleSubmitComment = () => {
    onFeedback?.(false, comment);
    setSubmitted(true);
    setShowComment(false);
    message.success('Thanks for your feedback!');
  };

  if (submitted) {
    return (
      <div className="quick-feedback submitted">
        <HeartOutlined style={{ color: '#06b6d4' }} />
        <Text type="secondary">Thank you!</Text>
      </div>
    );
  }

  return (
    <div className="quick-feedback">
      <Text type="secondary">{question}</Text>
      <Space>
        <Button size="small" icon={<SmileOutlined />} onClick={() => handleFeedback(true)}>
          Yes
        </Button>
        <Button size="small" icon={<FrownOutlined />} onClick={() => handleFeedback(false)}>
          No
        </Button>
      </Space>
      
      {showComment && (
        <div className="quick-feedback-comment">
          <Input.TextArea
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="How can we improve?"
            rows={2}
            maxLength={500}
          />
          <Button size="small" type="primary" onClick={handleSubmitComment}>
            Submit
          </Button>
        </div>
      )}
    </div>
  );
};

export default FeedbackWidget;
