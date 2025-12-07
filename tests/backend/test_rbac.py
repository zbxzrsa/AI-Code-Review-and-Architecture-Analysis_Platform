"""
RBAC System Tests

Tests for Role-Based Access Control:
1. Admin functions are not visible to regular users
2. Direct URL access returns 403 error
3. Permission status updates in real-time on account switch
4. Unauthorized access is logged
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock

# Add backend to path
_backend_path = Path(__file__).parent.parent.parent / "backend"
if str(_backend_path) not in sys.path:
    sys.path.insert(0, str(_backend_path))


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def admin_user():
    """Admin user fixture."""
    return {
        "id": "admin-123",
        "role": "admin",
        "email": "admin@example.com",
    }


@pytest.fixture
def regular_user():
    """Regular user fixture."""
    return {
        "id": "user-456",
        "role": "user",
        "email": "user@example.com",
    }


@pytest.fixture
def viewer_user():
    """Viewer user fixture."""
    return {
        "id": "viewer-789",
        "role": "viewer",
        "email": "viewer@example.com",
    }


@pytest.fixture
def guest_user():
    """Guest user fixture."""
    return {
        "id": "guest-000",
        "role": "guest",
        "email": "guest@example.com",
    }


# =============================================================================
# Role Hierarchy Tests
# =============================================================================

class TestRoleHierarchy:
    """Test role hierarchy and permissions."""

    def test_admin_has_highest_level(self):
        """Admin role should have highest hierarchy level."""
        from shared.security.rbac import ROLE_HIERARCHY, Role

        admin_level = ROLE_HIERARCHY[Role.ADMIN]
        user_level = ROLE_HIERARCHY[Role.USER]
        viewer_level = ROLE_HIERARCHY[Role.VIEWER]
        guest_level = ROLE_HIERARCHY[Role.GUEST]

        assert admin_level > user_level > viewer_level > guest_level

    def test_has_role_admin(self, admin_user):
        """Admin should pass all role checks."""
        from shared.security.rbac import has_role, Role

        assert has_role(admin_user["role"], Role.ADMIN) is True
        assert has_role(admin_user["role"], Role.USER) is True
        assert has_role(admin_user["role"], Role.VIEWER) is True
        assert has_role(admin_user["role"], Role.GUEST) is True

    def test_has_role_user(self, regular_user):
        """User should not have admin role."""
        from shared.security.rbac import has_role, Role

        assert has_role(regular_user["role"], Role.ADMIN) is False
        assert has_role(regular_user["role"], Role.USER) is True
        assert has_role(regular_user["role"], Role.VIEWER) is True
        assert has_role(regular_user["role"], Role.GUEST) is True

    def test_has_role_viewer(self, viewer_user):
        """Viewer should not have user or admin roles."""
        from shared.security.rbac import has_role, Role

        assert has_role(viewer_user["role"], Role.ADMIN) is False
        assert has_role(viewer_user["role"], Role.USER) is False
        assert has_role(viewer_user["role"], Role.VIEWER) is True
        assert has_role(viewer_user["role"], Role.GUEST) is True

    def test_has_role_guest(self, guest_user):
        """Guest should only have guest role."""
        from shared.security.rbac import has_role, Role

        assert has_role(guest_user["role"], Role.ADMIN) is False
        assert has_role(guest_user["role"], Role.USER) is False
        assert has_role(guest_user["role"], Role.VIEWER) is False
        assert has_role(guest_user["role"], Role.GUEST) is True


# =============================================================================
# Permission Tests
# =============================================================================

class TestPermissions:
    """Test permission checking."""

    def test_admin_has_all_permissions(self, admin_user):
        """Admin should have all permissions."""
        from shared.security.rbac import has_permission, Permission

        assert has_permission(admin_user["role"], Permission.ADMIN_ALL) is True
        assert has_permission(admin_user["role"], Permission.ADMIN_USERS) is True
        assert has_permission(admin_user["role"], Permission.PROJECT_CREATE) is True
        assert has_permission(admin_user["role"], Permission.PROJECT_DELETE) is True

    def test_user_has_limited_permissions(self, regular_user):
        """User should not have admin permissions."""
        from shared.security.rbac import has_permission, Permission

        # User permissions
        assert has_permission(regular_user["role"], Permission.PROJECT_CREATE) is True
        assert has_permission(regular_user["role"], Permission.PROJECT_READ) is True
        assert has_permission(regular_user["role"], Permission.ANALYSIS_CREATE) is True

        # Admin permissions (should be denied)
        assert has_permission(regular_user["role"], Permission.ADMIN_ALL) is False
        assert has_permission(regular_user["role"], Permission.ADMIN_USERS) is False
        assert has_permission(regular_user["role"], Permission.ADMIN_SYSTEM) is False

    def test_viewer_read_only_permissions(self, viewer_user):
        """Viewer should only have read permissions."""
        from shared.security.rbac import has_permission, Permission

        # Read permissions
        assert has_permission(viewer_user["role"], Permission.PROJECT_READ) is True
        assert has_permission(viewer_user["role"], Permission.ANALYSIS_READ) is True
        assert has_permission(viewer_user["role"], Permission.REVIEW_READ) is True

        # Write permissions (should be denied)
        assert has_permission(viewer_user["role"], Permission.PROJECT_CREATE) is False
        assert has_permission(viewer_user["role"], Permission.ANALYSIS_CREATE) is False
        assert has_permission(viewer_user["role"], Permission.REVIEW_CREATE) is False

    def test_has_any_permission(self, regular_user):
        """Test checking for any of multiple permissions."""
        from shared.security.rbac import has_any_permission, Permission

        # User has at least one
        assert has_any_permission(
            regular_user["role"],
            [Permission.PROJECT_CREATE, Permission.ADMIN_USERS]
        ) is True

        # User has none
        assert has_any_permission(
            regular_user["role"],
            [Permission.ADMIN_USERS, Permission.ADMIN_SYSTEM]
        ) is False

    def test_has_all_permissions(self, regular_user):
        """Test checking for all permissions."""
        from shared.security.rbac import has_all_permissions, Permission

        # User has all
        assert has_all_permissions(
            regular_user["role"],
            [Permission.PROJECT_CREATE, Permission.PROJECT_READ]
        ) is True

        # User missing one
        assert has_all_permissions(
            regular_user["role"],
            [Permission.PROJECT_CREATE, Permission.ADMIN_USERS]
        ) is False


# =============================================================================
# Admin Route Protection Tests
# =============================================================================

class TestAdminRouteProtection:
    """Test admin route protection."""

    def test_is_admin_route(self):
        """Test admin route detection."""
        from shared.security.rbac import is_admin_route

        # Admin routes
        assert is_admin_route("/api/admin/users") is True
        assert is_admin_route("/api/admin/providers") is True
        assert is_admin_route("/api/admin/audit") is True
        assert is_admin_route("/api/admin/any-path") is True

        # Non-admin routes
        assert is_admin_route("/api/projects") is False
        assert is_admin_route("/api/analyses") is False
        assert is_admin_route("/dashboard") is False

    def test_is_admin_function(self, admin_user, regular_user):
        """Test admin role checking."""
        from shared.security.rbac import is_admin

        assert is_admin(admin_user["role"]) is True
        assert is_admin(regular_user["role"]) is False


# =============================================================================
# Access Audit Logging Tests
# =============================================================================

class TestAccessAuditLogging:
    """Test unauthorized access logging."""

    def test_log_access_denial(self, regular_user):
        """Test that access denials are logged."""
        from shared.security.rbac import AccessAuditLogger

        # Log a denial
        AccessAuditLogger.log_access_attempt(
            user_id=regular_user["id"],
            user_role=regular_user["role"],
            requested_path="/api/admin/users",
            requested_method="GET",
            granted=False,
            denial_reason="Admin access required",
            required_role="admin",
        )

        # Check recent denials
        denials = AccessAuditLogger.get_recent_denials(10)
        assert len(denials) > 0

        latest = denials[-1]
        assert latest.user_id == regular_user["id"]
        assert latest.user_role == regular_user["role"]
        assert latest.requested_path == "/api/admin/users"
        assert latest.granted is False

    def test_get_user_denials(self, regular_user):
        """Test getting denials for specific user."""
        from shared.security.rbac import AccessAuditLogger

        # Log multiple denials
        for path in ["/api/admin/users", "/api/admin/audit", "/api/admin/providers"]:
            AccessAuditLogger.log_access_attempt(
                user_id=regular_user["id"],
                user_role=regular_user["role"],
                requested_path=path,
                requested_method="GET",
                granted=False,
                denial_reason="Admin access required",
            )

        # Get user-specific denials
        user_denials = AccessAuditLogger.get_user_denials(regular_user["id"])
        assert len(user_denials) >= 3

    def test_access_statistics(self):
        """Test access statistics."""
        from shared.security.rbac import AccessAuditLogger

        stats = AccessAuditLogger.get_statistics()

        assert "total_attempts" in stats
        assert "denied_attempts" in stats
        assert "denial_rate" in stats


# =============================================================================
# RBAC Dependency Tests
# =============================================================================

class TestRBACDependency:
    """Test FastAPI RBAC dependencies."""

    @pytest.mark.asyncio
    async def test_admin_dependency_blocks_user(self, regular_user):
        """Test that admin dependency blocks regular users."""
        from shared.security.rbac import require_admin, RBACDependency
        from fastapi import HTTPException

        # Create mock request
        mock_request = Mock()
        mock_request.url.path = "/api/admin/users"
        mock_request.method = "GET"
        mock_request.client = Mock(host="127.0.0.1")
        mock_request.headers = {"user-agent": "test"}

        # Create mock credentials
        mock_credentials = Mock()
        mock_credentials.credentials = "mock_token"

        # Mock token verification to return user role
        with patch("shared.security.rbac.TokenManager") as mock_token_manager:
            mock_token_manager.verify_token.return_value = {
                "sub": regular_user["id"],
                "role": regular_user["role"],
            }

            dependency = require_admin()

            with pytest.raises(HTTPException) as exc_info:
                await dependency(mock_request, mock_credentials)

            assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_user_dependency_allows_user(self, regular_user):
        """Test that user dependency allows regular users."""
        from shared.security.rbac import require_user, RBACDependency

        # Create mock request
        mock_request = Mock()
        mock_request.url.path = "/api/projects"
        mock_request.method = "GET"
        mock_request.client = Mock(host="127.0.0.1")
        mock_request.headers = {"user-agent": "test"}

        # Create mock credentials
        mock_credentials = Mock()
        mock_credentials.credentials = "mock_token"

        # Mock token verification
        with patch("shared.security.rbac.TokenManager") as mock_token_manager:
            mock_token_manager.verify_token.return_value = {
                "sub": regular_user["id"],
                "role": regular_user["role"],
            }

            dependency = require_user()

            result = await dependency(mock_request, mock_credentials)

            assert result["id"] == regular_user["id"]
            assert result["role"] == regular_user["role"]


# =============================================================================
# Direct URL Access Tests
# =============================================================================

class TestDirectURLAccess:
    """Test that direct URL access to admin routes returns 403."""

    def test_admin_routes_list(self):
        """Verify admin routes are properly configured."""
        from shared.security.rbac import ADMIN_ONLY_ROUTES

        assert len(ADMIN_ONLY_ROUTES) > 0
        assert "/api/admin/users" in ADMIN_ONLY_ROUTES
        assert "/api/admin/providers" in ADMIN_ONLY_ROUTES
        assert "/api/admin/audit" in ADMIN_ONLY_ROUTES

    def test_access_denied_response_format(self):
        """Test access denied response has correct format."""
        from shared.security.rbac import get_access_denied_response

        response = get_access_denied_response(
            error_code="ADMIN_REQUIRED",
            message="Admin access required",
            required_role="admin",
        )

        assert response["error"] == "Access Denied"
        assert response["code"] == "ADMIN_REQUIRED"
        assert response["message"] == "Admin access required"
        assert response["required_role"] == "admin"
        assert "contact" in response
        assert "timestamp" in response


# =============================================================================
# Permission Response Tests
# =============================================================================

class TestPermissionResponse:
    """Test permission response formatting."""

    def test_admin_permissions_response(self, admin_user):
        """Test admin permissions response."""
        from shared.security.rbac import get_user_permissions_response

        response = get_user_permissions_response(admin_user["role"])

        assert response["role"] == "admin"
        assert response["is_admin"] is True
        assert response["can_access_admin_panel"] is True
        assert len(response["permissions"]) > 0
        assert len(response["admin_permissions"]) > 0

    def test_user_permissions_response(self, regular_user):
        """Test user permissions response."""
        from shared.security.rbac import get_user_permissions_response

        response = get_user_permissions_response(regular_user["role"])

        assert response["role"] == "user"
        assert response["is_admin"] is False
        assert response["can_access_admin_panel"] is False
        assert len(response["permissions"]) > 0
        assert len(response["admin_permissions"]) == 0


# =============================================================================
# Admin Contact Info Tests
# =============================================================================

class TestAdminContactInfo:
    """Test admin contact information."""

    def test_admin_contact_info_exists(self):
        """Test admin contact info is configured."""
        from shared.security.rbac import ADMIN_CONTACT_INFO

        assert "email" in ADMIN_CONTACT_INFO
        assert "support_url" in ADMIN_CONTACT_INFO
        assert "documentation_url" in ADMIN_CONTACT_INFO


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
