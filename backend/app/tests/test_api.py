"""
API Integration Tests / API 集成测试

Run with: pytest backend/app/tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

from ..main import app


@pytest.fixture
def client():
    """Create test client / 创建测试客户端"""
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints / 健康检查端点测试"""
    
    def test_root(self, client):
        """Test root endpoint / 测试根端点"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
    
    def test_health_check(self, client):
        """Test health check / 测试健康检查"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_api_health(self, client):
        """Test API health / 测试 API 健康"""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data


class TestProjectEndpoints:
    """Tests for project endpoints / 项目端点测试"""
    
    def test_list_projects(self, client):
        """Test list projects / 测试列出项目"""
        response = client.get("/api/projects")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
    
    def test_list_projects_with_pagination(self, client):
        """Test pagination / 测试分页"""
        response = client.get("/api/projects?page=1&limit=5")
        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        assert data["limit"] == 5
    
    def test_create_project(self, client):
        """Test create project / 测试创建项目"""
        response = client.post("/api/projects", json={
            "name": "Test Project",
            "language": "Python",
            "description": "A test project"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Project"
        assert data["language"] == "Python"


class TestAdminEndpoints:
    """Tests for admin endpoints / 管理员端点测试"""
    
    def test_list_users(self, client):
        """Test list users / 测试列出用户"""
        response = client.get("/api/admin/users")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
    
    def test_user_stats(self, client):
        """Test user stats / 测试用户统计"""
        response = client.get("/api/admin/users/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "active" in data
    
    def test_list_providers(self, client):
        """Test list providers / 测试列出提供商"""
        response = client.get("/api/admin/providers")
        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
    
    def test_admin_stats(self, client):
        """Test admin stats / 测试管理统计"""
        response = client.get("/api/admin/stats")
        assert response.status_code == 200
        data = response.json()
        assert "users" in data
        assert "projects" in data


class TestOAuthEndpoints:
    """Tests for OAuth endpoints / OAuth 端点测试"""
    
    def test_list_providers(self, client):
        """Test list OAuth providers / 测试列出 OAuth 提供商"""
        response = client.get("/api/oauth/providers")
        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
        assert len(data["providers"]) >= 2
    
    def test_get_connections(self, client):
        """Test get connections / 测试获取连接"""
        response = client.get("/api/oauth/connections")
        assert response.status_code == 200
        data = response.json()
        assert "connections" in data


class TestAnalysisEndpoints:
    """Tests for analysis endpoints / 分析端点测试"""
    
    def test_ai_analyze(self, client):
        """Test AI analysis / 测试 AI 分析"""
        response = client.post("/api/ai/analyze", json={
            "code": "const x = 1;",
            "language": "typescript"
        })
        assert response.status_code == 200
        data = response.json()
        assert "analysis_id" in data
        assert "issues" in data
    
    def test_get_vulnerabilities(self, client):
        """Test get vulnerabilities / 测试获取漏洞"""
        response = client.get("/api/security/vulnerabilities")
        assert response.status_code == 200
        data = response.json()
        assert "vulnerabilities" in data
        assert "summary" in data
    
    def test_security_metrics(self, client):
        """Test security metrics / 测试安全指标"""
        response = client.get("/api/security/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "overall_score" in data
        assert "compliance" in data


class TestUserEndpoints:
    """Tests for user endpoints / 用户端点测试"""
    
    def test_get_profile(self, client):
        """Test get profile / 测试获取资料"""
        response = client.get("/api/user/profile")
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "email" in data
    
    def test_get_privacy_settings(self, client):
        """Test get privacy settings / 测试获取隐私设置"""
        response = client.get("/api/user/settings/privacy")
        assert response.status_code == 200
        data = response.json()
        assert "profile_visibility" in data
    
    def test_get_notification_settings(self, client):
        """Test get notification settings / 测试获取通知设置"""
        response = client.get("/api/user/settings/notifications")
        assert response.status_code == 200
        data = response.json()
        assert "email_notifications" in data
    
    def test_list_api_keys(self, client):
        """Test list API keys / 测试列出 API 密钥"""
        response = client.get("/api/user/api-keys")
        assert response.status_code == 200
        data = response.json()
        assert "keys" in data
    
    def test_create_api_key(self, client):
        """Test create API key / 测试创建 API 密钥"""
        response = client.post("/api/user/api-keys", json={
            "name": "Test Key",
            "scopes": ["read"]
        })
        assert response.status_code == 200
        data = response.json()
        assert "key" in data
        assert data["key"]["name"] == "Test Key"
