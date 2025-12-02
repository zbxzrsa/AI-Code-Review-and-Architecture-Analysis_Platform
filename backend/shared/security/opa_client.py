"""
OPA (Open Policy Agent) client for policy-based access control.

Integrates with OPA for fine-grained authorization decisions.
"""
import logging
import json
from typing import Dict, Any, Optional, List
from enum import Enum

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class PolicyAction(str, Enum):
    """Policy actions."""
    READ = "read"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    PROMOTE = "promote"
    QUARANTINE = "quarantine"
    EXECUTE = "execute"


class ResourceType(str, Enum):
    """Resource types."""
    EXPERIMENT = "experiment"
    VERSION = "version"
    PROJECT = "project"
    API_KEY = "api_key"
    USER = "user"
    PROVIDER = "provider"
    ANALYSIS = "analysis"


class OPAClient:
    """OPA policy client."""

    def __init__(self, opa_url: str = "http://localhost:8181", timeout: int = 5):
        """Initialize OPA client."""
        self.opa_url = opa_url.rstrip("/")
        self.timeout = timeout
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy."""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def health_check(self) -> bool:
        """Check OPA server health."""
        try:
            response = self.session.get(
                f"{self.opa_url}/health",
                timeout=self.timeout
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"OPA health check failed: {e}")
            return False

    # ============================================
    # Access Control
    # ============================================

    def check_permission(
        self,
        user: Dict[str, Any],
        resource: Dict[str, Any],
        action: str = PolicyAction.READ.value
    ) -> bool:
        """Check if user has permission for action on resource."""
        try:
            policy_input = {
                "input": {
                    "user": {
                        "id": user.get("id"),
                        "role": user.get("role")
                    },
                    "resource": resource,
                    "action": action
                }
            }

            response = self.session.post(
                f"{self.opa_url}/v1/data/code_review/access/allow",
                json=policy_input,
                timeout=self.timeout
            )

            if response.status_code != 200:
                logger.warning(f"OPA permission check failed: {response.status_code}")
                return False

            result = response.json().get("result", False)
            logger.debug(f"Permission check: {action} on {resource} = {result}")
            return result
        except Exception as e:
            logger.error(f"OPA permission check error: {e}")
            return False

    def check_resource_access(
        self,
        user: Dict[str, Any],
        resource_type: str,
        resource_id: str,
        action: str = PolicyAction.READ.value
    ) -> bool:
        """Check access to specific resource."""
        resource = {
            "type": resource_type,
            "id": resource_id
        }
        return self.check_permission(user, resource, action)

    def check_version_access(
        self,
        user: Dict[str, Any],
        version: str
    ) -> bool:
        """Check if user can access version."""
        try:
            policy_input = {
                "input": {
                    "user": {
                        "id": user.get("id"),
                        "role": user.get("role")
                    },
                    "resource": {
                        "type": ResourceType.VERSION.value,
                        "version": version
                    }
                }
            }

            response = self.session.post(
                f"{self.opa_url}/v1/data/code_review/access/allow_version_access",
                json=policy_input,
                timeout=self.timeout
            )

            if response.status_code != 200:
                return False

            return response.json().get("result", False)
        except Exception as e:
            logger.error(f"Version access check failed: {e}")
            return False

    # ============================================
    # Version Promotion
    # ============================================

    def can_promote_version(
        self,
        user: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> bool:
        """Check if version can be promoted based on metrics."""
        try:
            policy_input = {
                "input": {
                    "user": {
                        "id": user.get("id"),
                        "role": user.get("role")
                    },
                    "action": PolicyAction.PROMOTE.value,
                    "metrics": metrics
                }
            }

            response = self.session.post(
                f"{self.opa_url}/v1/data/code_review/access/allow_promotion",
                json=policy_input,
                timeout=self.timeout
            )

            if response.status_code != 200:
                logger.warning(f"Promotion check failed: {response.status_code}")
                return False

            result = response.json().get("result", False)
            logger.info(f"Promotion allowed: {result} (metrics: {metrics})")
            return result
        except Exception as e:
            logger.error(f"Promotion check error: {e}")
            return False

    def get_promotion_requirements(self) -> Dict[str, Any]:
        """Get promotion requirements from policy."""
        try:
            response = self.session.get(
                f"{self.opa_url}/v1/data/code_review/access/promotion_requirements",
                timeout=self.timeout
            )

            if response.status_code != 200:
                return {}

            return response.json().get("result", {})
        except Exception as e:
            logger.error(f"Failed to get promotion requirements: {e}")
            return {}

    # ============================================
    # Baseline Violations
    # ============================================

    def should_alert_on_violation(
        self,
        event: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> bool:
        """Check if baseline violation should trigger alert."""
        try:
            policy_input = {
                "input": {
                    "event": event,
                    "metrics": metrics
                }
            }

            response = self.session.post(
                f"{self.opa_url}/v1/data/code_review/alerts/alert_on_violation",
                json=policy_input,
                timeout=self.timeout
            )

            if response.status_code != 200:
                return False

            return response.json().get("result", False)
        except Exception as e:
            logger.error(f"Violation alert check failed: {e}")
            return False

    # ============================================
    # Data Access Control
    # ============================================

    def can_access_api_key(
        self,
        user: Dict[str, Any],
        api_key_owner_id: str
    ) -> bool:
        """Check if user can access API key."""
        try:
            policy_input = {
                "input": {
                    "user": {
                        "id": user.get("id"),
                        "role": user.get("role")
                    },
                    "resource": {
                        "type": ResourceType.API_KEY.value,
                        "owner_id": api_key_owner_id
                    },
                    "action": PolicyAction.READ.value
                }
            }

            response = self.session.post(
                f"{self.opa_url}/v1/data/code_review/access/allow",
                json=policy_input,
                timeout=self.timeout
            )

            if response.status_code != 200:
                return False

            return response.json().get("result", False)
        except Exception as e:
            logger.error(f"API key access check failed: {e}")
            return False

    def can_access_user_data(
        self,
        user: Dict[str, Any],
        target_user_id: str
    ) -> bool:
        """Check if user can access another user's data."""
        try:
            policy_input = {
                "input": {
                    "user": {
                        "id": user.get("id"),
                        "role": user.get("role")
                    },
                    "resource": {
                        "type": ResourceType.USER.value,
                        "id": target_user_id
                    },
                    "action": PolicyAction.READ.value
                }
            }

            response = self.session.post(
                f"{self.opa_url}/v1/data/code_review/access/allow",
                json=policy_input,
                timeout=self.timeout
            )

            if response.status_code != 200:
                return False

            return response.json().get("result", False)
        except Exception as e:
            logger.error(f"User data access check failed: {e}")
            return False

    # ============================================
    # Quota and Cost Control
    # ============================================

    def can_execute_analysis(
        self,
        user: Dict[str, Any],
        analysis_config: Dict[str, Any],
        current_usage: Dict[str, Any]
    ) -> bool:
        """Check if user can execute analysis based on quotas."""
        try:
            policy_input = {
                "input": {
                    "user": {
                        "id": user.get("id"),
                        "role": user.get("role")
                    },
                    "action": PolicyAction.EXECUTE.value,
                    "analysis": analysis_config,
                    "usage": current_usage
                }
            }

            response = self.session.post(
                f"{self.opa_url}/v1/data/code_review/quotas/allow_execution",
                json=policy_input,
                timeout=self.timeout
            )

            if response.status_code != 200:
                return False

            return response.json().get("result", False)
        except Exception as e:
            logger.error(f"Analysis execution check failed: {e}")
            return False

    def get_quota_status(self, user_id: str) -> Dict[str, Any]:
        """Get user quota status from policy."""
        try:
            response = self.session.post(
                f"{self.opa_url}/v1/data/code_review/quotas/user_quota",
                json={"input": {"user_id": user_id}},
                timeout=self.timeout
            )

            if response.status_code != 200:
                return {}

            return response.json().get("result", {})
        except Exception as e:
            logger.error(f"Failed to get quota status: {e}")
            return {}

    # ============================================
    # Audit and Compliance
    # ============================================

    def should_audit_action(
        self,
        user: Dict[str, Any],
        action: str,
        resource: Dict[str, Any]
    ) -> bool:
        """Check if action should be audited."""
        try:
            policy_input = {
                "input": {
                    "user": {
                        "id": user.get("id"),
                        "role": user.get("role")
                    },
                    "action": action,
                    "resource": resource
                }
            }

            response = self.session.post(
                f"{self.opa_url}/v1/data/code_review/audit/should_audit",
                json=policy_input,
                timeout=self.timeout
            )

            if response.status_code != 200:
                return True  # Audit by default on error

            return response.json().get("result", True)
        except Exception as e:
            logger.error(f"Audit check failed: {e}")
            return True  # Audit by default on error

    # ============================================
    # Policy Management
    # ============================================

    def load_policy(self, policy_name: str, policy_code: str) -> bool:
        """Load policy into OPA."""
        try:
            response = self.session.put(
                f"{self.opa_url}/v1/policies/{policy_name}",
                data=policy_code,
                headers={"Content-Type": "text/plain"},
                timeout=self.timeout
            )

            if response.status_code not in [200, 201]:
                logger.error(f"Failed to load policy: {response.status_code}")
                return False

            logger.info(f"Policy {policy_name} loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load policy: {e}")
            return False

    def get_policy(self, policy_name: str) -> Optional[str]:
        """Get policy from OPA."""
        try:
            response = self.session.get(
                f"{self.opa_url}/v1/policies/{policy_name}",
                timeout=self.timeout
            )

            if response.status_code != 200:
                return None

            return response.json().get("result")
        except Exception as e:
            logger.error(f"Failed to get policy: {e}")
            return None

    def delete_policy(self, policy_name: str) -> bool:
        """Delete policy from OPA."""
        try:
            response = self.session.delete(
                f"{self.opa_url}/v1/policies/{policy_name}",
                timeout=self.timeout
            )

            if response.status_code != 204:
                logger.error(f"Failed to delete policy: {response.status_code}")
                return False

            logger.info(f"Policy {policy_name} deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to delete policy: {e}")
            return False

    # ============================================
    # Batch Operations
    # ============================================

    def batch_check_permissions(
        self,
        user: Dict[str, Any],
        resources: List[Dict[str, Any]],
        action: str = PolicyAction.READ.value
    ) -> Dict[str, bool]:
        """Check permissions for multiple resources."""
        results = {}
        for resource in resources:
            resource_id = resource.get("id", "unknown")
            results[resource_id] = self.check_permission(user, resource, action)
        return results

    # ============================================
    # Error Handling
    # ============================================

    def is_available(self) -> bool:
        """Check if OPA is available."""
        return self.health_check()
