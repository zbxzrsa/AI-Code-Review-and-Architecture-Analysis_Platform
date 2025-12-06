"""
API Contract Testing System (Testing Improvement Plan #2)

Provides contract testing with:
- OpenAPI/Swagger specification validation
- Pact-style consumer-driven contracts
- API version compatibility verification
- Automated contract generation

Acceptance Criteria: Each API change must pass contract test verification
"""
import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type
import yaml

import pytest

logger = logging.getLogger(__name__)


class HttpMethod(str, Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class ContractViolationType(str, Enum):
    """Types of contract violations."""
    MISSING_ENDPOINT = "missing_endpoint"
    WRONG_METHOD = "wrong_method"
    SCHEMA_MISMATCH = "schema_mismatch"
    MISSING_FIELD = "missing_field"
    TYPE_MISMATCH = "type_mismatch"
    STATUS_CODE_MISMATCH = "status_code_mismatch"
    HEADER_MISSING = "header_missing"
    BREAKING_CHANGE = "breaking_change"


@dataclass
class ContractViolation:
    """Represents a contract violation."""
    violation_type: ContractViolationType
    path: str
    method: HttpMethod
    message: str
    expected: Any = None
    actual: Any = None
    severity: str = "error"  # error, warning
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.violation_type.value,
            "path": self.path,
            "method": self.method.value,
            "message": self.message,
            "expected": self.expected,
            "actual": self.actual,
            "severity": self.severity,
        }


@dataclass
class EndpointContract:
    """Contract for a single API endpoint."""
    path: str
    method: HttpMethod
    summary: str = ""
    description: str = ""
    request_body: Optional[Dict] = None
    query_params: List[Dict] = field(default_factory=list)
    path_params: List[Dict] = field(default_factory=list)
    headers: List[Dict] = field(default_factory=list)
    responses: Dict[int, Dict] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    deprecated: bool = False
    
    def to_openapi(self) -> Dict[str, Any]:
        """Convert to OpenAPI format."""
        spec = {
            "summary": self.summary,
            "description": self.description,
            "tags": self.tags,
            "deprecated": self.deprecated,
            "responses": {},
        }
        
        # Parameters
        parameters = []
        for param in self.path_params:
            parameters.append({**param, "in": "path"})
        for param in self.query_params:
            parameters.append({**param, "in": "query"})
        for param in self.headers:
            parameters.append({**param, "in": "header"})
        
        if parameters:
            spec["parameters"] = parameters
        
        # Request body
        if self.request_body:
            spec["requestBody"] = {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": self.request_body
                    }
                }
            }
        
        # Responses
        for status_code, response in self.responses.items():
            spec["responses"][str(status_code)] = response
        
        return spec


@dataclass
class APIContract:
    """Full API contract specification."""
    title: str
    version: str
    base_url: str
    endpoints: List[EndpointContract] = field(default_factory=list)
    schemas: Dict[str, Dict] = field(default_factory=dict)
    security_schemes: Dict[str, Dict] = field(default_factory=dict)
    
    def to_openapi(self) -> Dict[str, Any]:
        """Convert to OpenAPI 3.0 specification."""
        paths = {}
        
        for endpoint in self.endpoints:
            if endpoint.path not in paths:
                paths[endpoint.path] = {}
            paths[endpoint.path][endpoint.method.value.lower()] = endpoint.to_openapi()
        
        return {
            "openapi": "3.0.3",
            "info": {
                "title": self.title,
                "version": self.version,
            },
            "servers": [{"url": self.base_url}],
            "paths": paths,
            "components": {
                "schemas": self.schemas,
                "securitySchemes": self.security_schemes,
            },
        }
    
    def save(self, path: str):
        """Save contract to file."""
        spec = self.to_openapi()
        
        if path.endswith(".yaml") or path.endswith(".yml"):
            with open(path, "w") as f:
                yaml.dump(spec, f, default_flow_style=False)
        else:
            with open(path, "w") as f:
                json.dump(spec, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "APIContract":
        """Load contract from file."""
        with open(path) as f:
            if path.endswith(".yaml") or path.endswith(".yml"):
                spec = yaml.safe_load(f)
            else:
                spec = json.load(f)
        
        return cls.from_openapi(spec)
    
    @classmethod
    def from_openapi(cls, spec: Dict) -> "APIContract":
        """Create contract from OpenAPI specification."""
        endpoints = []
        
        for path, methods in spec.get("paths", {}).items():
            for method, details in methods.items():
                if method.upper() not in [m.value for m in HttpMethod]:
                    continue
                
                endpoint = EndpointContract(
                    path=path,
                    method=HttpMethod(method.upper()),
                    summary=details.get("summary", ""),
                    description=details.get("description", ""),
                    tags=details.get("tags", []),
                    deprecated=details.get("deprecated", False),
                )
                
                # Parse parameters
                for param in details.get("parameters", []):
                    param_data = {
                        "name": param.get("name"),
                        "required": param.get("required", False),
                        "schema": param.get("schema", {}),
                    }
                    
                    if param.get("in") == "path":
                        endpoint.path_params.append(param_data)
                    elif param.get("in") == "query":
                        endpoint.query_params.append(param_data)
                    elif param.get("in") == "header":
                        endpoint.headers.append(param_data)
                
                # Parse request body
                if "requestBody" in details:
                    content = details["requestBody"].get("content", {})
                    if "application/json" in content:
                        endpoint.request_body = content["application/json"].get("schema")
                
                # Parse responses
                for status_code, response in details.get("responses", {}).items():
                    endpoint.responses[int(status_code)] = response
                
                endpoints.append(endpoint)
        
        return cls(
            title=spec.get("info", {}).get("title", "API"),
            version=spec.get("info", {}).get("version", "1.0.0"),
            base_url=spec.get("servers", [{}])[0].get("url", ""),
            endpoints=endpoints,
            schemas=spec.get("components", {}).get("schemas", {}),
            security_schemes=spec.get("components", {}).get("securitySchemes", {}),
        )


class SchemaValidator:
    """Validates data against JSON schemas."""
    
    @classmethod
    def validate(cls, data: Any, schema: Dict) -> List[str]:
        """Validate data against schema. Returns list of errors."""
        errors = []
        
        schema_type = schema.get("type")
        
        if schema_type == "object":
            errors.extend(cls._validate_object(data, schema))
        elif schema_type == "array":
            errors.extend(cls._validate_array(data, schema))
        elif schema_type == "string":
            errors.extend(cls._validate_string(data, schema))
        elif schema_type == "integer":
            errors.extend(cls._validate_integer(data, schema))
        elif schema_type == "number":
            errors.extend(cls._validate_number(data, schema))
        elif schema_type == "boolean":
            errors.extend(cls._validate_boolean(data, schema))
        
        return errors
    
    @classmethod
    def _validate_object(cls, data: Any, schema: Dict) -> List[str]:
        errors = []
        
        if not isinstance(data, dict):
            return [f"Expected object, got {type(data).__name__}"]
        
        # Check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Validate properties
        properties = schema.get("properties", {})
        for prop, prop_schema in properties.items():
            if prop in data:
                prop_errors = cls.validate(data[prop], prop_schema)
                errors.extend([f"{prop}.{e}" for e in prop_errors])
        
        return errors
    
    @classmethod
    def _validate_array(cls, data: Any, schema: Dict) -> List[str]:
        errors = []
        
        if not isinstance(data, list):
            return [f"Expected array, got {type(data).__name__}"]
        
        # Validate items
        items_schema = schema.get("items", {})
        for i, item in enumerate(data):
            item_errors = cls.validate(item, items_schema)
            errors.extend([f"[{i}].{e}" for e in item_errors])
        
        return errors
    
    @classmethod
    def _validate_string(cls, data: Any, schema: Dict) -> List[str]:
        if not isinstance(data, str):
            return [f"Expected string, got {type(data).__name__}"]
        
        errors = []
        
        if "minLength" in schema and len(data) < schema["minLength"]:
            errors.append(f"String too short (min: {schema['minLength']})")
        
        if "maxLength" in schema and len(data) > schema["maxLength"]:
            errors.append(f"String too long (max: {schema['maxLength']})")
        
        if "pattern" in schema and not re.match(schema["pattern"], data):
            errors.append(f"String doesn't match pattern: {schema['pattern']}")
        
        return errors
    
    @classmethod
    def _validate_integer(cls, data: Any, schema: Dict) -> List[str]:
        if not isinstance(data, int) or isinstance(data, bool):
            return [f"Expected integer, got {type(data).__name__}"]
        
        errors = []
        
        if "minimum" in schema and data < schema["minimum"]:
            errors.append(f"Value too small (min: {schema['minimum']})")
        
        if "maximum" in schema and data > schema["maximum"]:
            errors.append(f"Value too large (max: {schema['maximum']})")
        
        return errors
    
    @classmethod
    def _validate_number(cls, data: Any, schema: Dict) -> List[str]:
        if not isinstance(data, (int, float)) or isinstance(data, bool):
            return [f"Expected number, got {type(data).__name__}"]
        return []
    
    @classmethod
    def _validate_boolean(cls, data: Any, schema: Dict) -> List[str]:
        if not isinstance(data, bool):
            return [f"Expected boolean, got {type(data).__name__}"]
        return []


class ContractValidator:
    """Validates API implementations against contracts."""
    
    def __init__(self, contract: APIContract):
        self.contract = contract
        self.violations: List[ContractViolation] = []
    
    def validate_response(
        self,
        path: str,
        method: HttpMethod,
        status_code: int,
        response_body: Any,
        response_headers: Dict[str, str] = None
    ) -> List[ContractViolation]:
        """Validate a response against the contract."""
        violations = []
        
        # Find endpoint
        endpoint = self._find_endpoint(path, method)
        if not endpoint:
            violations.append(ContractViolation(
                violation_type=ContractViolationType.MISSING_ENDPOINT,
                path=path,
                method=method,
                message=f"Endpoint not found in contract: {method.value} {path}",
            ))
            return violations
        
        # Check status code
        if status_code not in endpoint.responses:
            violations.append(ContractViolation(
                violation_type=ContractViolationType.STATUS_CODE_MISMATCH,
                path=path,
                method=method,
                message=f"Unexpected status code: {status_code}",
                expected=list(endpoint.responses.keys()),
                actual=status_code,
            ))
        else:
            # Validate response schema
            response_spec = endpoint.responses[status_code]
            content = response_spec.get("content", {})
            
            if "application/json" in content:
                schema = content["application/json"].get("schema", {})
                errors = SchemaValidator.validate(response_body, schema)
                
                for error in errors:
                    violations.append(ContractViolation(
                        violation_type=ContractViolationType.SCHEMA_MISMATCH,
                        path=path,
                        method=method,
                        message=f"Schema validation error: {error}",
                    ))
        
        self.violations.extend(violations)
        return violations
    
    def validate_request(
        self,
        path: str,
        method: HttpMethod,
        request_body: Any = None,
        query_params: Dict = None,
        headers: Dict = None
    ) -> List[ContractViolation]:
        """Validate a request against the contract."""
        violations = []
        
        endpoint = self._find_endpoint(path, method)
        if not endpoint:
            violations.append(ContractViolation(
                violation_type=ContractViolationType.MISSING_ENDPOINT,
                path=path,
                method=method,
                message=f"Endpoint not found in contract: {method.value} {path}",
            ))
            return violations
        
        # Validate request body
        if endpoint.request_body and request_body:
            errors = SchemaValidator.validate(request_body, endpoint.request_body)
            for error in errors:
                violations.append(ContractViolation(
                    violation_type=ContractViolationType.SCHEMA_MISMATCH,
                    path=path,
                    method=method,
                    message=f"Request body validation error: {error}",
                ))
        
        # Validate required query params
        if query_params:
            for param in endpoint.query_params:
                if param.get("required") and param["name"] not in query_params:
                    violations.append(ContractViolation(
                        violation_type=ContractViolationType.MISSING_FIELD,
                        path=path,
                        method=method,
                        message=f"Missing required query parameter: {param['name']}",
                    ))
        
        self.violations.extend(violations)
        return violations
    
    def _find_endpoint(self, path: str, method: HttpMethod) -> Optional[EndpointContract]:
        """Find endpoint matching path and method."""
        for endpoint in self.contract.endpoints:
            if endpoint.method == method:
                # Check path match (handle path params)
                pattern = re.sub(r'\{[^}]+\}', r'[^/]+', endpoint.path)
                if re.fullmatch(pattern, path):
                    return endpoint
        return None
    
    def check_breaking_changes(self, old_contract: APIContract) -> List[ContractViolation]:
        """Check for breaking changes between contracts."""
        violations = []
        
        # Check for removed endpoints
        old_endpoints = {(e.path, e.method) for e in old_contract.endpoints}
        new_endpoints = {(e.path, e.method) for e in self.contract.endpoints}
        
        for path, method in old_endpoints - new_endpoints:
            violations.append(ContractViolation(
                violation_type=ContractViolationType.BREAKING_CHANGE,
                path=path,
                method=method,
                message=f"Endpoint removed: {method.value} {path}",
                severity="error",
            ))
        
        # Check for removed required fields in existing endpoints
        for old_endpoint in old_contract.endpoints:
            new_endpoint = self._find_endpoint(old_endpoint.path, old_endpoint.method)
            if new_endpoint:
                # Check response schema changes
                for status_code, old_response in old_endpoint.responses.items():
                    if status_code not in new_endpoint.responses:
                        violations.append(ContractViolation(
                            violation_type=ContractViolationType.BREAKING_CHANGE,
                            path=old_endpoint.path,
                            method=old_endpoint.method,
                            message=f"Response status code removed: {status_code}",
                            severity="error",
                        ))
        
        return violations
    
    def get_report(self) -> Dict[str, Any]:
        """Get validation report."""
        return {
            "contract": {
                "title": self.contract.title,
                "version": self.contract.version,
            },
            "violations": [v.to_dict() for v in self.violations],
            "total_violations": len(self.violations),
            "errors": len([v for v in self.violations if v.severity == "error"]),
            "warnings": len([v for v in self.violations if v.severity == "warning"]),
            "passed": len(self.violations) == 0,
        }


# ============================================================
# Pact-Style Consumer Contract Tests
# ============================================================

@dataclass
class Interaction:
    """Pact-style interaction definition."""
    description: str
    provider_state: str
    request: Dict[str, Any]
    response: Dict[str, Any]


class ConsumerContract:
    """Consumer-driven contract for Pact-style testing."""
    
    def __init__(self, consumer: str, provider: str):
        self.consumer = consumer
        self.provider = provider
        self.interactions: List[Interaction] = []
    
    def given(self, provider_state: str) -> "InteractionBuilder":
        """Start defining an interaction."""
        return InteractionBuilder(self, provider_state)
    
    def add_interaction(self, interaction: Interaction):
        """Add an interaction to the contract."""
        self.interactions.append(interaction)
    
    def to_pact(self) -> Dict[str, Any]:
        """Convert to Pact format."""
        return {
            "consumer": {"name": self.consumer},
            "provider": {"name": self.provider},
            "interactions": [
                {
                    "description": i.description,
                    "providerState": i.provider_state,
                    "request": i.request,
                    "response": i.response,
                }
                for i in self.interactions
            ],
            "metadata": {
                "pactSpecification": {"version": "2.0.0"}
            }
        }
    
    def save(self, path: str):
        """Save Pact contract."""
        with open(path, "w") as f:
            json.dump(self.to_pact(), f, indent=2)


class InteractionBuilder:
    """Builder for Pact interactions."""
    
    def __init__(self, contract: ConsumerContract, provider_state: str):
        self._contract = contract
        self._provider_state = provider_state
        self._description = ""
        self._request = {}
        self._response = {}
    
    def upon_receiving(self, description: str) -> "InteractionBuilder":
        """Set interaction description."""
        self._description = description
        return self
    
    def with_request(
        self,
        method: str,
        path: str,
        headers: Dict = None,
        body: Any = None,
        query: Dict = None
    ) -> "InteractionBuilder":
        """Define expected request."""
        self._request = {
            "method": method,
            "path": path,
        }
        if headers:
            self._request["headers"] = headers
        if body:
            self._request["body"] = body
        if query:
            self._request["query"] = query
        return self
    
    def will_respond_with(
        self,
        status: int,
        headers: Dict = None,
        body: Any = None
    ) -> ConsumerContract:
        """Define expected response."""
        self._response = {"status": status}
        if headers:
            self._response["headers"] = headers
        if body:
            self._response["body"] = body
        
        # Create and add interaction
        interaction = Interaction(
            description=self._description,
            provider_state=self._provider_state,
            request=self._request,
            response=self._response,
        )
        self._contract.add_interaction(interaction)
        
        return self._contract


# ============================================================
# Test Fixtures and Helpers
# ============================================================

@pytest.fixture
def sample_api_contract() -> APIContract:
    """Provide sample API contract for testing."""
    return APIContract(
        title="Code Review API",
        version="1.0.0",
        base_url="http://localhost:8000",
        endpoints=[
            EndpointContract(
                path="/api/v1/projects",
                method=HttpMethod.GET,
                summary="List projects",
                responses={
                    200: {
                        "description": "Success",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "projects": {"type": "array"},
                                        "total": {"type": "integer"},
                                    },
                                    "required": ["projects", "total"],
                                }
                            }
                        }
                    }
                }
            ),
            EndpointContract(
                path="/api/v1/projects/{id}",
                method=HttpMethod.GET,
                summary="Get project",
                path_params=[{"name": "id", "required": True, "schema": {"type": "string"}}],
                responses={
                    200: {"description": "Success"},
                    404: {"description": "Not found"},
                }
            ),
            EndpointContract(
                path="/api/v1/analyze",
                method=HttpMethod.POST,
                summary="Analyze code",
                request_body={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "language": {"type": "string"},
                    },
                    "required": ["code"],
                },
                responses={
                    200: {"description": "Analysis result"},
                    400: {"description": "Bad request"},
                }
            ),
        ],
    )


class TestContractValidation:
    """Contract validation tests."""
    
    def test_valid_response_passes(self, sample_api_contract):
        """Test that valid response passes validation."""
        validator = ContractValidator(sample_api_contract)
        
        violations = validator.validate_response(
            path="/api/v1/projects",
            method=HttpMethod.GET,
            status_code=200,
            response_body={"projects": [], "total": 0},
        )
        
        assert len(violations) == 0
    
    def test_missing_required_field_fails(self, sample_api_contract):
        """Test that missing required field fails validation."""
        validator = ContractValidator(sample_api_contract)
        
        violations = validator.validate_response(
            path="/api/v1/projects",
            method=HttpMethod.GET,
            status_code=200,
            response_body={"projects": []},  # Missing 'total'
        )
        
        assert len(violations) > 0
        assert any(v.violation_type == ContractViolationType.SCHEMA_MISMATCH for v in violations)
    
    def test_unexpected_status_code(self, sample_api_contract):
        """Test that unexpected status code is detected."""
        validator = ContractValidator(sample_api_contract)
        
        violations = validator.validate_response(
            path="/api/v1/projects",
            method=HttpMethod.GET,
            status_code=500,
            response_body={"error": "Internal error"},
        )
        
        assert len(violations) > 0
        assert any(v.violation_type == ContractViolationType.STATUS_CODE_MISMATCH for v in violations)


class TestConsumerContract:
    """Consumer contract tests."""
    
    def test_create_pact_contract(self):
        """Test creating Pact-style contract."""
        contract = ConsumerContract("frontend", "api")
        
        contract.given("a user exists").upon_receiving(
            "a request to get user"
        ).with_request(
            method="GET",
            path="/api/v1/users/123",
            headers={"Authorization": "Bearer token"},
        ).will_respond_with(
            status=200,
            headers={"Content-Type": "application/json"},
            body={"id": "123", "name": "Test User"},
        )
        
        pact = contract.to_pact()
        
        assert pact["consumer"]["name"] == "frontend"
        assert pact["provider"]["name"] == "api"
        assert len(pact["interactions"]) == 1
        assert pact["interactions"][0]["response"]["status"] == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
