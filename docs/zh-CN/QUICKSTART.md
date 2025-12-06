# 快速入门指南

| **文档信息** |            |
| ------------ | ---------- |
| **版本**     | 1.0.0      |
| **最后更新** | 2024-12-06 |
| **语言**     | 中文       |

---

## 变更历史

| 版本  | 日期       | 作者     | 变更描述     |
| ----- | ---------- | -------- | ------------ |
| 1.0.0 | 2024-12-06 | 开发团队 | 初始文档创建 |

---

## 目录

1. [概述](#1-概述)
2. [系统要求](#2-系统要求)
3. [快速安装](#3-快速安装)
4. [首次运行](#4-首次运行)
5. [基本使用](#5-基本使用)
6. [常见问题](#6-常见问题)

---

## 1. 概述

AI 代码审查平台是一个智能化的代码分析和审查系统，采用三版本自演进架构，集成多种 AI 模型提供全面的代码质量分析。

### 1.1 核心功能

- 🔍 **智能代码分析** - AI 驱动的代码质量检测
- 🔒 **安全漏洞扫描** - 识别潜在安全风险
- 📊 **架构分析** - 代码结构和依赖关系可视化
- 🤖 **AI 辅助修复** - 智能修复建议
- 📈 **持续监控** - 代码质量趋势追踪

### 1.2 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                    三版本自演进架构                          │
├─────────────────────────────────────────────────────────────┤
│  V1 实验区    │    V2 生产区    │    V3 隔离区              │
│  (新模型测试)  │   (稳定服务)    │   (问题模型隔离)          │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 系统要求

### 2.1 最低配置

| 组件           | 最低要求                                |
| -------------- | --------------------------------------- |
| 操作系统       | Windows 10 / macOS 10.15 / Ubuntu 20.04 |
| CPU            | 4 核心                                  |
| 内存           | 8 GB                                    |
| 磁盘空间       | 20 GB                                   |
| Docker         | 20.10+                                  |
| Docker Compose | 2.0+                                    |

### 2.2 推荐配置

| 组件     | 推荐配置          |
| -------- | ----------------- |
| CPU      | 8 核心+           |
| 内存     | 16 GB+            |
| 磁盘空间 | 50 GB+ SSD        |
| GPU      | NVIDIA GPU (可选) |

### 2.3 软件依赖

- Git 2.30+
- Node.js 18+ (前端开发)
- Python 3.10+ (后端开发)

---

## 3. 快速安装

### 3.1 克隆仓库

```bash
git clone https://github.com/your-org/ai-code-review-platform.git
cd ai-code-review-platform
```

### 3.2 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑配置文件
# Windows: notepad .env
# Linux/Mac: nano .env
```

### 3.3 关键配置项

```bash
# 基础配置
ENVIRONMENT=development
MOCK_MODE=true          # 开发模式无需AI密钥

# 数据库配置
POSTGRES_PASSWORD=your_secure_password
REDIS_PASSWORD=your_secure_password

# AI提供商密钥 (生产环境必需)
OPENAI_API_KEY=sk-your-key        # 可选
ANTHROPIC_API_KEY=sk-ant-your-key # 可选
```

### 3.4 启动服务

```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 3.5 验证安装

```bash
# 检查API健康状态
curl http://localhost:8000/health

# 预期响应
# {"status": "healthy", "version": "1.0.0"}
```

---

## 4. 首次运行

### 4.1 访问应用

| 服务         | 地址                       |
| ------------ | -------------------------- |
| 前端应用     | http://localhost:3000      |
| API 服务     | http://localhost:8000      |
| API 文档     | http://localhost:8000/docs |
| Grafana 监控 | http://localhost:3001      |

### 4.2 创建账户

1. 打开浏览器访问 http://localhost:3000
2. 点击 **注册** 按钮
3. 填写邮箱和密码
4. 完成邮箱验证

### 4.3 首次配置

1. **选择语言**: 设置界面语言（中文/English）
2. **连接代码仓库**: 绑定 GitHub/GitLab 账号
3. **创建项目**: 导入您的第一个代码仓库

---

## 5. 基本使用

### 5.1 代码分析

#### 步骤一：创建项目

```
1. 点击 "新建项目"
2. 选择代码来源（GitHub/GitLab/本地上传）
3. 配置分析选项
4. 点击 "创建"
```

#### 步骤二：运行分析

```
1. 在项目页面点击 "开始分析"
2. 选择分析类型：
   - 快速扫描：基础问题检测
   - 深度分析：全面代码审查
   - 安全扫描：专注安全漏洞
3. 等待分析完成
```

#### 步骤三：查看结果

```
分析结果包括：
├── 问题列表
│   ├── 严重程度（高/中/低）
│   ├── 问题类型
│   └── 修复建议
├── 代码质量评分
└── 趋势图表
```

### 5.2 AI 对话

与 AI 助手进行代码相关的对话：

```
用户：请解释这段代码的作用
AI：这段代码实现了用户认证功能...

用户：如何优化这个函数的性能？
AI：建议以下优化方案...
```

### 5.3 自动修复

1. 在问题详情页点击 **AI 修复**
2. 查看修复建议
3. 选择 **应用修复** 或 **手动编辑**
4. 提交变更

---

## 6. 常见问题

### 6.1 安装问题

**Q: Docker 启动失败？**

```bash
# 检查Docker状态
docker info

# 重启Docker服务
# Windows: 重启 Docker Desktop
# Linux: sudo systemctl restart docker
```

**Q: 端口被占用？**

```bash
# 检查端口占用
# Windows: netstat -ano | findstr :8000
# Linux: lsof -i :8000

# 修改.env中的端口配置
API_PORT=8001
FRONTEND_PORT=3001
```

### 6.2 运行问题

**Q: 分析超时？**

- 检查网络连接
- 减少单次分析的代码量
- 增加超时配置

**Q: AI 功能不可用？**

```bash
# 检查MOCK_MODE配置
# 开发环境: MOCK_MODE=true
# 生产环境: MOCK_MODE=false + 有效API密钥
```

### 6.3 获取帮助

- 📖 完整文档: [docs/README.md](../README.md)
- 🐛 问题反馈: [GitHub Issues](https://github.com/your-org/ai-code-review-platform/issues)
- 💬 社区讨论: [Discord](https://discord.gg/your-server)
- 📧 技术支持: support@example.com

---

## 下一步

- 阅读 [完整用户手册](./user-manual.md)
- 了解 [API 参考文档](./api-reference.md)
- 查看 [架构设计文档](./architecture.md)
- 学习 [最佳实践指南](./best-practices.md)

---

**版权所有 © 2024 AI 代码审查平台。保留所有权利。**
