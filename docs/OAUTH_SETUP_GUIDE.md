# OAuth 配置指南 / OAuth Setup Guide

本指南帮助您配置 GitHub、GitLab 和 Bitbucket 的 OAuth 集成。

## 快速开始

### 1. GitHub OAuth 配置

#### 步骤 1: 创建 GitHub OAuth App

1. 访问 https://github.com/settings/developers
2. 点击 **"New OAuth App"**
3. 填写信息:

   - **Application name**: `AI Code Review Platform`
   - **Homepage URL**: `http://localhost:5173` (开发环境)
   - **Authorization callback URL**: `http://localhost:5173/oauth/callback/github`

4. 点击 **"Register application"**
5. 复制 **Client ID**
6. 点击 **"Generate a new client secret"** 并复制

#### 步骤 2: 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，添加以下内容
GITHUB_CLIENT_ID=your_client_id_here
GITHUB_CLIENT_SECRET=your_client_secret_here
```

#### 步骤 3: 重启后端服务

```bash
# 如果使用 Python 直接运行
# 停止当前服务，然后重新启动
cd backend
python dev-api-server.py

# 如果使用 Docker
docker compose restart auth-service
```

---

### 2. GitLab OAuth 配置

#### 步骤 1: 创建 GitLab Application

1. 访问 https://gitlab.com/-/profile/applications
2. 填写信息:

   - **Name**: `AI Code Review Platform`
   - **Redirect URI**: `http://localhost:5173/oauth/callback/gitlab`
   - **Scopes**: 勾选 `read_user`, `read_repository`, `api`

3. 点击 **"Save application"**
4. 复制 **Application ID** 和 **Secret**

#### 步骤 2: 配置环境变量

```bash
GITLAB_CLIENT_ID=your_application_id
GITLAB_CLIENT_SECRET=your_secret
```

---

### 3. Bitbucket API Token 配置

> ⚠️ **注意**: 自 2025 年 9 月起，Bitbucket 已弃用 OAuth，改用 API Token。

#### 步骤 1: 创建 API Token

1. 访问 https://bitbucket.org/account/settings/api-tokens/
2. 点击 **"Create API token"**
3. 填写信息:

   - **Name**: `AI Code Review Platform`
   - **Scopes**: 勾选需要的权限 (repository:read, repository:write 等)

4. 复制生成的 **API Token**

#### 步骤 2: 配置环境变量

```bash
BITBUCKET_API_TOKEN=your_api_token_here
BITBUCKET_USERNAME=your_bitbucket_username
```

> 💡 API Token 配置后立即生效，无需 OAuth 回调流程。

---

## 验证配置

配置完成后，可以通过以下 API 验证:

```bash
# 检查 OAuth 提供商状态
curl http://localhost:8000/api/auth/oauth/providers

# 预期响应:
{
  "providers": [
    {
      "name": "github",
      "configured": true,  # 应该显示 true
      "message": "Ready to connect"
    }
    ...
  ]
}
```

---

## 生产环境配置

在生产环境中，需要更新以下 URL:

```bash
# 生产环境
GITHUB_CALLBACK_URL=https://your-domain.com/oauth/callback/github
GITLAB_CALLBACK_URL=https://your-domain.com/oauth/callback/gitlab
BITBUCKET_CALLBACK_URL=https://your-domain.com/oauth/callback/bitbucket
```

同时需要在各 OAuth 提供商后台更新对应的回调 URL。

---

## 故障排除

### 问题: OAuth 连接失败

**可能原因:**

1. 环境变量未正确设置
2. 回调 URL 不匹配
3. Client Secret 已过期

**解决方法:**

1. 检查 `.env` 文件中的值是否正确
2. 确认 OAuth App 中的回调 URL 与代码中一致
3. 在 OAuth 提供商后台重新生成 Secret

### 问题: "OAuth not configured" 消息

**解决方法:**

1. 确保已创建 `.env` 文件
2. 确保环境变量名称正确（区分大小写）
3. 重启后端服务以加载新的环境变量

---

## 安全建议

1. **永远不要**将 OAuth Secret 提交到 Git
2. 使用 `.gitignore` 忽略 `.env` 文件
3. 在生产环境使用 Kubernetes Secrets 或环境变量注入
4. 定期轮换 Client Secret

---

_最后更新: 2024-12-05_
