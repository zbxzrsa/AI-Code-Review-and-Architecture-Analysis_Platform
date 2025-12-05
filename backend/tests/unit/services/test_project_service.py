"""
Project Service Unit Tests
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.services.project_service import ProjectService
from app.models.project import Project, ProjectFile, ProjectVersion


class TestProjectService:
    """Test ProjectService class"""

    @pytest.fixture
    def project_service(self):
        """Create ProjectService instance with mocked dependencies"""
        service = ProjectService()
        service.db = AsyncMock()
        service.redis = AsyncMock()
        return service

    @pytest.fixture
    def mock_project(self):
        """Create mock project"""
        return Project(
            id=str(uuid4()),
            name="Test Project",
            description="Test description",
            language="python",
            owner_id="user-123",
            status="active",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )


class TestProjectCRUD(TestProjectService):
    """Test project CRUD operations"""

    @pytest.mark.asyncio
    async def test_create_project(self, project_service):
        """Test project creation"""
        project_service.db.add = MagicMock()
        project_service.db.commit = AsyncMock()
        project_service.db.refresh = AsyncMock()

        result = await project_service.create(
            name="New Project",
            description="New description",
            language="python",
            owner_id="user-123"
        )

        assert result is not None
        project_service.db.add.assert_called_once()
        project_service.db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_project_by_id(self, project_service, mock_project):
        """Test getting project by ID"""
        project_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=mock_project))
        )

        result = await project_service.get(mock_project.id)

        assert result is not None
        assert result.id == mock_project.id

    @pytest.mark.asyncio
    async def test_get_project_not_found(self, project_service):
        """Test getting non-existent project"""
        project_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=None))
        )

        result = await project_service.get("nonexistent-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_update_project(self, project_service, mock_project):
        """Test project update"""
        project_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=mock_project))
        )
        project_service.db.commit = AsyncMock()

        result = await project_service.update(
            project_id=mock_project.id,
            name="Updated Name",
            description="Updated description"
        )

        assert result is not None
        project_service.db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_project(self, project_service, mock_project):
        """Test project deletion"""
        project_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=mock_project))
        )
        project_service.db.delete = AsyncMock()
        project_service.db.commit = AsyncMock()

        result = await project_service.delete(mock_project.id)

        assert result is True
        project_service.db.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_projects(self, project_service, mock_project):
        """Test listing projects"""
        project_service.db.execute = AsyncMock(
            return_value=MagicMock(
                scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[mock_project])))
            )
        )

        result = await project_service.list(owner_id="user-123")

        assert len(result) == 1
        assert result[0].id == mock_project.id

    @pytest.mark.asyncio
    async def test_list_projects_with_pagination(self, project_service, mock_project):
        """Test listing projects with pagination"""
        projects = [mock_project for _ in range(5)]
        project_service.db.execute = AsyncMock(
            return_value=MagicMock(
                scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=projects[:2])))
            )
        )

        result = await project_service.list(
            owner_id="user-123",
            page=1,
            limit=2
        )

        assert len(result) == 2


class TestProjectFiles(TestProjectService):
    """Test project file operations"""

    @pytest.fixture
    def mock_file(self):
        """Create mock project file"""
        return ProjectFile(
            id=str(uuid4()),
            project_id="project-123",
            path="src/main.py",
            content="print('hello')",
            language="python",
            created_at=datetime.now(timezone.utc),
        )

    @pytest.mark.asyncio
    async def test_add_file(self, project_service, mock_project):
        """Test adding file to project"""
        project_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=mock_project))
        )
        project_service.db.add = MagicMock()
        project_service.db.commit = AsyncMock()

        result = await project_service.add_file(
            project_id=mock_project.id,
            path="src/main.py",
            content="print('hello')"
        )

        assert result is not None
        project_service.db.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_file(self, project_service, mock_file):
        """Test getting project file"""
        project_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=mock_file))
        )

        result = await project_service.get_file(
            project_id="project-123",
            path="src/main.py"
        )

        assert result is not None
        assert result.path == "src/main.py"

    @pytest.mark.asyncio
    async def test_update_file(self, project_service, mock_file):
        """Test updating project file"""
        project_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=mock_file))
        )
        project_service.db.commit = AsyncMock()

        result = await project_service.update_file(
            project_id="project-123",
            path="src/main.py",
            content="print('updated')"
        )

        assert result is not None
        project_service.db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_file(self, project_service, mock_file):
        """Test deleting project file"""
        project_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=mock_file))
        )
        project_service.db.delete = AsyncMock()
        project_service.db.commit = AsyncMock()

        result = await project_service.delete_file(
            project_id="project-123",
            path="src/main.py"
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_list_files(self, project_service, mock_file):
        """Test listing project files"""
        project_service.db.execute = AsyncMock(
            return_value=MagicMock(
                scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[mock_file])))
            )
        )

        result = await project_service.list_files("project-123")

        assert len(result) == 1


class TestProjectVersioning(TestProjectService):
    """Test project versioning"""

    @pytest.fixture
    def mock_version(self, mock_project):
        """Create mock project version"""
        return ProjectVersion(
            id=str(uuid4()),
            project_id=mock_project.id,
            version="1.0.0",
            description="Initial version",
            created_at=datetime.now(timezone.utc),
        )

    @pytest.mark.asyncio
    async def test_create_version(self, project_service, mock_project):
        """Test creating project version"""
        project_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=mock_project))
        )
        project_service.db.add = MagicMock()
        project_service.db.commit = AsyncMock()

        result = await project_service.create_version(
            project_id=mock_project.id,
            version="1.0.0",
            description="Initial version"
        )

        assert result is not None
        project_service.db.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_versions(self, project_service, mock_version):
        """Test getting project versions"""
        project_service.db.execute = AsyncMock(
            return_value=MagicMock(
                scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[mock_version])))
            )
        )

        result = await project_service.get_versions("project-123")

        assert len(result) == 1
        assert result[0].version == "1.0.0"


class TestProjectPermissions(TestProjectService):
    """Test project permission checks"""

    @pytest.mark.asyncio
    async def test_check_owner_permission(self, project_service, mock_project):
        """Test owner has full permissions"""
        project_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=mock_project))
        )

        result = await project_service.check_permission(
            project_id=mock_project.id,
            user_id=mock_project.owner_id,
            permission="delete"
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_check_member_permission(self, project_service, mock_project):
        """Test member permissions"""
        mock_project.members = [{"user_id": "member-123", "role": "editor"}]
        project_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=mock_project))
        )

        result = await project_service.check_permission(
            project_id=mock_project.id,
            user_id="member-123",
            permission="edit"
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_check_unauthorized_permission(self, project_service, mock_project):
        """Test unauthorized user permissions"""
        project_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=mock_project))
        )

        result = await project_service.check_permission(
            project_id=mock_project.id,
            user_id="other-user",
            permission="edit"
        )

        assert result is False


class TestProjectSearch(TestProjectService):
    """Test project search functionality"""

    @pytest.mark.asyncio
    async def test_search_projects(self, project_service, mock_project):
        """Test searching projects"""
        project_service.db.execute = AsyncMock(
            return_value=MagicMock(
                scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[mock_project])))
            )
        )

        result = await project_service.search(
            query="Test",
            owner_id="user-123"
        )

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_search_by_language(self, project_service, mock_project):
        """Test searching by language"""
        project_service.db.execute = AsyncMock(
            return_value=MagicMock(
                scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[mock_project])))
            )
        )

        result = await project_service.search(
            language="python",
            owner_id="user-123"
        )

        assert len(result) == 1
        assert result[0].language == "python"


class TestProjectStats(TestProjectService):
    """Test project statistics"""

    @pytest.mark.asyncio
    async def test_get_project_stats(self, project_service, mock_project):
        """Test getting project statistics"""
        project_service.db.execute = AsyncMock(
            return_value=MagicMock(
                first=MagicMock(return_value=(10, 1000, 5))  # files, lines, issues
            )
        )

        result = await project_service.get_stats(mock_project.id)

        assert result["files_count"] == 10
        assert result["lines_of_code"] == 1000
        assert result["issues_count"] == 5
