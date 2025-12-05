"""
Unit Tests for Services / 服务单元测试

Run with: pytest backend/app/tests/test_services.py -v
"""

import pytest
from datetime import datetime

from ..services import AnalysisService, ProjectService, UserService


class TestAnalysisService:
    """Tests for AnalysisService / AnalysisService 测试"""
    
    def setup_method(self):
        """Setup test fixtures / 设置测试夹具"""
        self.service = AnalysisService(mock_mode=True)
    
    def test_generate_session_id(self):
        """Test session ID generation / 测试会话 ID 生成"""
        session_id = self.service.generate_session_id("proj_1")
        assert session_id.startswith("session_proj_1_")
        assert len(session_id) > 20
    
    def test_calculate_code_hash(self):
        """Test code hash calculation / 测试代码哈希计算"""
        code = "const x = 1;"
        hash1 = self.service.calculate_code_hash(code)
        hash2 = self.service.calculate_code_hash(code)
        
        assert hash1 == hash2
        assert len(hash1) == 16
    
    def test_calculate_code_hash_different_code(self):
        """Test different code produces different hash / 测试不同代码产生不同哈希"""
        hash1 = self.service.calculate_code_hash("const x = 1;")
        hash2 = self.service.calculate_code_hash("const x = 2;")
        
        assert hash1 != hash2
    
    @pytest.mark.asyncio
    async def test_analyze_code_mock(self):
        """Test mock code analysis / 测试模拟代码分析"""
        code = """
        var x = 1;
        console.log(x);
        """
        
        result = await self.service.analyze_code(code)
        
        assert result["status"] == "completed"
        assert "issues" in result
        assert "metrics" in result
        assert result["from_cache"] is False
    
    @pytest.mark.asyncio
    async def test_analyze_code_caching(self):
        """Test analysis result caching / 测试分析结果缓存"""
        code = "const y = 2;"
        
        # First call
        result1 = await self.service.analyze_code(code)
        assert result1["from_cache"] is False
        
        # Second call should be cached
        result2 = await self.service.analyze_code(code)
        assert result2["from_cache"] is True
    
    def test_clear_cache(self):
        """Test cache clearing / 测试清除缓存"""
        self.service._analysis_cache["test"] = {"data": "test"}
        self.service.clear_cache()
        
        assert len(self.service._analysis_cache) == 0
    
    def test_get_cache_stats(self):
        """Test cache statistics / 测试缓存统计"""
        self.service._analysis_cache["test"] = {"issues": [1, 2, 3]}
        stats = self.service.get_cache_stats()
        
        assert stats["cached_analyses"] == 1
        assert stats["total_issues_cached"] == 3


class TestProjectService:
    """Tests for ProjectService / ProjectService 测试"""
    
    def setup_method(self):
        """Setup test fixtures / 设置测试夹具"""
        self.service = ProjectService()
    
    def test_create_project(self):
        """Test project creation / 测试创建项目"""
        project = self.service.create_project(
            name="Test Project",
            language="Python",
            description="A test project"
        )
        
        assert project.id.startswith("proj_")
        assert project.name == "Test Project"
        assert project.language == "Python"
        assert project.status == "active"
    
    def test_get_project(self):
        """Test getting project by ID / 测试根据 ID 获取项目"""
        created = self.service.create_project(
            name="Test",
            language="JavaScript"
        )
        
        fetched = self.service.get_project(created.id)
        
        assert fetched is not None
        assert fetched.id == created.id
    
    def test_get_nonexistent_project(self):
        """Test getting nonexistent project / 测试获取不存在的项目"""
        result = self.service.get_project("nonexistent_id")
        assert result is None
    
    def test_list_projects_with_filter(self):
        """Test listing projects with filter / 测试带过滤的项目列表"""
        self.service.create_project(name="Project A", language="Python")
        self.service.create_project(name="Project B", language="JavaScript")
        
        result = self.service.list_projects(search="Project A")
        
        assert result["total"] == 1
        assert result["items"][0].name == "Project A"
    
    def test_update_project(self):
        """Test project update / 测试更新项目"""
        project = self.service.create_project(
            name="Original",
            language="Python"
        )
        
        updated = self.service.update_project(
            project.id,
            name="Updated"
        )
        
        assert updated is not None
        assert updated.name == "Updated"
    
    def test_delete_project(self):
        """Test project deletion / 测试删除项目"""
        project = self.service.create_project(
            name="To Delete",
            language="Python"
        )
        
        result = self.service.delete_project(project.id)
        assert result is True
        
        fetched = self.service.get_project(project.id)
        assert fetched is None
    
    def test_get_stats(self):
        """Test project statistics / 测试项目统计"""
        self.service.create_project(name="P1", language="Python")
        self.service.create_project(name="P2", language="Python")
        self.service.create_project(name="P3", language="JavaScript")
        
        stats = self.service.get_stats()
        
        assert stats["total"] == 3
        assert stats["by_language"]["Python"] == 2
        assert stats["by_language"]["JavaScript"] == 1


class TestUserService:
    """Tests for UserService / UserService 测试"""
    
    def setup_method(self):
        """Setup test fixtures / 设置测试夹具"""
        self.service = UserService()
    
    def test_demo_user_exists(self):
        """Test demo user is created / 测试演示用户已创建"""
        user = self.service.get_user("user_1")
        
        assert user is not None
        assert user["email"] == "demo@example.com"
        assert user["role"] == "admin"
    
    def test_get_user_by_email(self):
        """Test getting user by email / 测试根据邮箱获取用户"""
        user = self.service.get_user_by_email("demo@example.com")
        
        assert user is not None
        assert user["id"] == "user_1"
    
    def test_create_api_key(self):
        """Test API key creation / 测试创建 API 密钥"""
        key = self.service.create_api_key(
            user_id="user_1",
            name="Test Key",
            scopes=["read", "write"]
        )
        
        assert key["name"] == "Test Key"
        assert "full_key" in key
        assert key["full_key"].startswith("sk_")
    
    def test_list_api_keys(self):
        """Test listing API keys / 测试列出 API 密钥"""
        self.service.create_api_key(
            user_id="user_1",
            name="Key 1",
            scopes=["read"]
        )
        
        keys = self.service.list_api_keys("user_1")
        
        assert len(keys) >= 1
        assert any(k["name"] == "Key 1" for k in keys)
    
    def test_revoke_api_key(self):
        """Test API key revocation / 测试撤销 API 密钥"""
        key = self.service.create_api_key(
            user_id="user_1",
            name="To Revoke",
            scopes=["read"]
        )
        
        result = self.service.revoke_api_key(key["id"], "user_1")
        assert result is True
        
        keys = self.service.list_api_keys("user_1")
        assert not any(k["id"] == key["id"] for k in keys)
    
    def test_create_session(self):
        """Test session creation / 测试创建会话"""
        session_id = self.service.create_session("user_1")
        
        assert len(session_id) > 20
    
    def test_validate_session(self):
        """Test session validation / 测试验证会话"""
        session_id = self.service.create_session("user_1")
        
        user = self.service.validate_session(session_id)
        
        assert user is not None
        assert user["id"] == "user_1"
    
    def test_invalidate_session(self):
        """Test session invalidation / 测试使会话失效"""
        session_id = self.service.create_session("user_1")
        
        result = self.service.invalidate_session(session_id)
        assert result is True
        
        user = self.service.validate_session(session_id)
        assert user is None
    
    def test_get_user_stats(self):
        """Test user statistics / 测试用户统计"""
        stats = self.service.get_user_stats()
        
        assert "total" in stats
        assert "active" in stats
        assert stats["total"] >= 1
