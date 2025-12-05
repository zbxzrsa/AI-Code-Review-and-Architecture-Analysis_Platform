"""
Security Penetration Testing Suite

Phase 5: OWASP Top 10 Security Testing
- Injection attacks
- Authentication bypass
- Sensitive data exposure
- XSS and CSRF
- Security misconfiguration
"""

import asyncio
import pytest
import httpx
import json
import base64
import hashlib
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SecurityFinding:
    """Security test finding."""
    category: str
    severity: str  # critical, high, medium, low, info
    title: str
    description: str
    evidence: str
    remediation: str
    owasp_category: str


class SecurityTestSuite:
    """
    Comprehensive security testing suite.
    
    Tests against OWASP Top 10:
    1. Injection
    2. Broken Authentication
    3. Sensitive Data Exposure
    4. XML External Entities (XXE)
    5. Broken Access Control
    6. Security Misconfiguration
    7. Cross-Site Scripting (XSS)
    8. Insecure Deserialization
    9. Using Components with Known Vulnerabilities
    10. Insufficient Logging & Monitoring
    """
    
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.findings: List[SecurityFinding] = []
    
    def _headers(self, auth: bool = True) -> Dict[str, str]:
        """Get request headers."""
        headers = {'Content-Type': 'application/json'}
        if auth and self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers
    
    def _add_finding(
        self,
        category: str,
        severity: str,
        title: str,
        description: str,
        evidence: str,
        remediation: str,
        owasp: str,
    ):
        """Add security finding."""
        self.findings.append(SecurityFinding(
            category=category,
            severity=severity,
            title=title,
            description=description,
            evidence=evidence,
            remediation=remediation,
            owasp_category=owasp,
        ))
    
    # =========================================================================
    # A1: Injection
    # =========================================================================
    
    async def test_sql_injection(self, client: httpx.AsyncClient):
        """Test for SQL injection vulnerabilities."""
        print("Testing SQL Injection...")
        
        sql_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "1' UNION SELECT * FROM users--",
            "admin'--",
            "' OR 1=1--",
            "1; SELECT * FROM information_schema.tables",
        ]
        
        endpoints = [
            ("/api/v2/projects", "GET", {"search": "{payload}"}),
            ("/api/v2/users", "GET", {"filter": "{payload}"}),
            ("/api/auth/login", "POST", {"username": "{payload}", "password": "test"}),
        ]
        
        for endpoint, method, params in endpoints:
            for payload in sql_payloads:
                try:
                    test_params = {
                        k: v.replace("{payload}", payload) 
                        for k, v in params.items()
                    }
                    
                    if method == "GET":
                        resp = await client.get(
                            f"{self.base_url}{endpoint}",
                            params=test_params,
                            headers=self._headers(),
                        )
                    else:
                        resp = await client.post(
                            f"{self.base_url}{endpoint}",
                            json=test_params,
                            headers=self._headers(auth=False),
                        )
                    
                    # Check for SQL error indicators
                    if resp.status_code == 500:
                        body = resp.text.lower()
                        sql_errors = ['sql', 'syntax', 'query', 'mysql', 'postgresql', 'sqlite']
                        if any(err in body for err in sql_errors):
                            self._add_finding(
                                category="Injection",
                                severity="critical",
                                title=f"SQL Injection at {endpoint}",
                                description=f"SQL error returned with payload: {payload}",
                                evidence=f"Status: {resp.status_code}, Body: {resp.text[:200]}",
                                remediation="Use parameterized queries",
                                owasp="A1:2021-Injection",
                            )
                except (httpx.RequestError, httpx.HTTPStatusError):
                    pass  # Endpoint might not exist or be unreachable
    
    async def test_command_injection(self, client: httpx.AsyncClient):
        """Test for command injection."""
        print("Testing Command Injection...")
        
        cmd_payloads = [
            "; ls -la",
            "| cat /etc/passwd",
            "`id`",
            "$(whoami)",
            "& dir",
        ]
        
        # Test file-related endpoints
        for payload in cmd_payloads:
            try:
                resp = await client.post(
                    f"{self.base_url}/api/v2/projects/analyze",
                    json={"file_path": f"/tmp/test{payload}"},
                    headers=self._headers(),
                )
                
                if resp.status_code == 200 and ("root" in resp.text or "uid=" in resp.text):
                    self._add_finding(
                        category="Injection",
                        severity="critical",
                        title="Command Injection Detected",
                        description=f"Command execution with payload: {payload}",
                        evidence=resp.text[:200],
                        remediation="Sanitize all user inputs, avoid shell commands",
                        owasp="A1:2021-Injection",
                    )
            except (httpx.RequestError, httpx.HTTPStatusError):
                pass  # Endpoint might not exist or be unreachable
    
    # =========================================================================
    # A2: Broken Authentication
    # =========================================================================
    
    async def test_authentication_bypass(self, client: httpx.AsyncClient):
        """Test for authentication bypass."""
        print("Testing Authentication Bypass...")
        
        # Test without authentication
        protected_endpoints = [
            "/api/v2/projects",
            "/api/v2/cr-ai/review",
            "/api/v2/vc-ai/versions",
            "/api/admin/users",
        ]
        
        for endpoint in protected_endpoints:
            resp = await client.get(
                f"{self.base_url}{endpoint}",
                headers={'Content-Type': 'application/json'},  # No auth
            )
            
            if resp.status_code == 200:
                self._add_finding(
                    category="Authentication",
                    severity="high",
                    title=f"Missing Authentication at {endpoint}",
                    description="Endpoint accessible without authentication",
                    evidence=f"Status: {resp.status_code}",
                    remediation="Require authentication for all protected endpoints",
                    owasp="A2:2021-Broken Authentication",
                )
    
    async def test_jwt_vulnerabilities(self, client: httpx.AsyncClient):
        """Test for JWT vulnerabilities."""
        print("Testing JWT Vulnerabilities...")
        
        # Test JWT none algorithm
        none_token = base64.b64encode(b'{"alg":"none","typ":"JWT"}').decode() + "." + \
                     base64.b64encode(b'{"sub":"admin","role":"admin"}').decode() + "."
        
        resp = await client.get(
            f"{self.base_url}/api/v2/projects",
            headers={'Authorization': f'Bearer {none_token}'},
        )
        
        if resp.status_code == 200:
            self._add_finding(
                category="Authentication",
                severity="critical",
                title="JWT None Algorithm Accepted",
                description="Server accepts JWT tokens with 'none' algorithm",
                evidence=f"Token: {none_token[:50]}..., Status: {resp.status_code}",
                remediation="Reject tokens with 'none' algorithm",
                owasp="A2:2021-Broken Authentication",
            )
        
        # Test expired token handling
        expired_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwiZXhwIjoxfQ.invalid"
        
        resp = await client.get(
            f"{self.base_url}/api/v2/projects",
            headers={'Authorization': f'Bearer {expired_token}'},
        )
        
        if resp.status_code == 200:
            self._add_finding(
                category="Authentication",
                severity="high",
                title="Expired JWT Token Accepted",
                description="Server accepts expired JWT tokens",
                evidence=f"Status: {resp.status_code}",
                remediation="Properly validate token expiration",
                owasp="A2:2021-Broken Authentication",
            )
    
    async def test_brute_force_protection(self, client: httpx.AsyncClient):
        """Test for brute force protection."""
        print("Testing Brute Force Protection...")
        
        # Attempt multiple failed logins
        for i in range(15):
            await client.post(
                f"{self.base_url}/api/auth/login",
                json={"username": "admin", "password": f"wrong{i}"},
                headers={'Content-Type': 'application/json'},
            )
        
        # Check if still accepting requests
        resp = await client.post(
            f"{self.base_url}/api/auth/login",
            json={"username": "admin", "password": "wrong"},
            headers={'Content-Type': 'application/json'},
        )
        
        if resp.status_code not in [429, 423]:  # Too Many Requests or Locked
            self._add_finding(
                category="Authentication",
                severity="medium",
                title="No Brute Force Protection",
                description="No rate limiting or account lockout after multiple failed attempts",
                evidence=f"15+ attempts accepted, Status: {resp.status_code}",
                remediation="Implement rate limiting and account lockout",
                owasp="A2:2021-Broken Authentication",
            )
    
    # =========================================================================
    # A3: Sensitive Data Exposure
    # =========================================================================
    
    async def test_sensitive_data_exposure(self, client: httpx.AsyncClient):
        """Test for sensitive data exposure."""
        print("Testing Sensitive Data Exposure...")
        
        # Check error responses for sensitive info
        resp = await client.get(
            f"{self.base_url}/api/v2/nonexistent",
            headers=self._headers(),
        )
        
        sensitive_patterns = [
            'password', 'secret', 'api_key', 'token',
            'stack trace', 'traceback', 'exception',
            '/var/', '/home/', 'C:\\',
        ]
        
        body = resp.text.lower()
        for pattern in sensitive_patterns:
            if pattern.lower() in body:
                self._add_finding(
                    category="Data Exposure",
                    severity="medium",
                    title="Sensitive Information in Error Response",
                    description=f"Found '{pattern}' in error response",
                    evidence=resp.text[:200],
                    remediation="Sanitize error responses, don't expose internal details",
                    owasp="A3:2021-Sensitive Data Exposure",
                )
                break
        
        # Check for missing security headers
        security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': ['DENY', 'SAMEORIGIN'],
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': None,
            'Content-Security-Policy': None,
        }
        
        resp = await client.get(f"{self.base_url}/health")
        
        for header, expected in security_headers.items():
            if header not in resp.headers:
                self._add_finding(
                    category="Data Exposure",
                    severity="low",
                    title=f"Missing Security Header: {header}",
                    description=f"Response missing {header} header",
                    evidence=f"Headers: {dict(resp.headers)}",
                    remediation=f"Add {header} header to responses",
                    owasp="A3:2021-Sensitive Data Exposure",
                )
    
    # =========================================================================
    # A5: Broken Access Control
    # =========================================================================
    
    async def test_broken_access_control(self, client: httpx.AsyncClient):
        """Test for broken access control."""
        print("Testing Broken Access Control...")
        
        # Test IDOR (Insecure Direct Object Reference)
        idor_endpoints = [
            "/api/v2/projects/{id}",
            "/api/v2/users/{id}",
            "/api/v2/reviews/{id}",
        ]
        
        for endpoint in idor_endpoints:
            for test_id in ["1", "0", "-1", "admin", "../../../etc/passwd"]:
                resp = await client.get(
                    f"{self.base_url}{endpoint.format(id=test_id)}",
                    headers=self._headers(),
                )
                
                if resp.status_code == 200:
                    # Check if accessing others' data
                    body = resp.text.lower()
                    if "password" in body or "secret" in body:
                        self._add_finding(
                            category="Access Control",
                            severity="high",
                            title=f"IDOR at {endpoint}",
                            description=f"Can access resource with ID: {test_id}",
                            evidence=resp.text[:200],
                            remediation="Implement proper authorization checks",
                            owasp="A5:2021-Broken Access Control",
                        )
        
        # Test privilege escalation
        resp = await client.post(
            f"{self.base_url}/api/admin/users",
            json={"role": "admin"},
            headers=self._headers(),
        )
        
        if resp.status_code == 200:
            self._add_finding(
                category="Access Control",
                severity="critical",
                title="Privilege Escalation Possible",
                description="Non-admin can create admin users",
                evidence=f"Status: {resp.status_code}",
                remediation="Enforce role-based access control",
                owasp="A5:2021-Broken Access Control",
            )
    
    # =========================================================================
    # A7: Cross-Site Scripting (XSS)
    # =========================================================================
    
    async def test_xss(self, client: httpx.AsyncClient):
        """Test for XSS vulnerabilities."""
        print("Testing XSS...")
        
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg onload=alert('XSS')>",
            "'\"><script>alert('XSS')</script>",
        ]
        
        for payload in xss_payloads:
            resp = await client.post(
                f"{self.base_url}/api/v2/cr-ai/review",
                json={"code": payload, "language": "html"},
                headers=self._headers(),
            )
            
            if resp.status_code == 200 and payload in resp.text:
                self._add_finding(
                    category="XSS",
                    severity="high",
                    title="Reflected XSS",
                    description=f"XSS payload reflected in response",
                    evidence=f"Payload: {payload}",
                    remediation="Escape output, use Content-Security-Policy",
                    owasp="A7:2021-XSS",
                )
    
    # =========================================================================
    # Run All Tests
    # =========================================================================
    
    async def run_all_tests(self) -> List[SecurityFinding]:
        """Run all security tests."""
        print(f"\n{'='*60}")
        print("SECURITY PENETRATION TEST SUITE")
        print(f"Target: {self.base_url}")
        print(f"Started: {datetime.now().isoformat()}")
        print(f"{'='*60}\n")
        
        # Security Note: verify=False is intentional for penetration testing
        # This test suite needs to test against local/staging servers that may use self-signed certs
        # In production environments, SSL verification should always be enabled
        ssl_context = httpx.create_ssl_context(verify=False)  # noqa: S501  # nosec B501
        async with httpx.AsyncClient(timeout=30.0, verify=ssl_context) as client:
            # A1: Injection
            await self.test_sql_injection(client)
            await self.test_command_injection(client)
            
            # A2: Broken Authentication
            await self.test_authentication_bypass(client)
            await self.test_jwt_vulnerabilities(client)
            await self.test_brute_force_protection(client)
            
            # A3: Sensitive Data Exposure
            await self.test_sensitive_data_exposure(client)
            
            # A5: Broken Access Control
            await self.test_broken_access_control(client)
            
            # A7: XSS
            await self.test_xss(client)
        
        return self.findings
    
    def generate_report(self) -> str:
        """Generate security test report."""
        report = f"""
{'='*60}
SECURITY TEST REPORT
{'='*60}
Generated: {datetime.now().isoformat()}
Target: {self.base_url}

SUMMARY
-------
Total Findings: {len(self.findings)}
Critical: {len([f for f in self.findings if f.severity == 'critical'])}
High: {len([f for f in self.findings if f.severity == 'high'])}
Medium: {len([f for f in self.findings if f.severity == 'medium'])}
Low: {len([f for f in self.findings if f.severity == 'low'])}

FINDINGS
--------
"""
        
        for i, finding in enumerate(self.findings, 1):
            report += f"""
{i}. [{finding.severity.upper()}] {finding.title}
   Category: {finding.category}
   OWASP: {finding.owasp_category}
   Description: {finding.description}
   Evidence: {finding.evidence[:100]}...
   Remediation: {finding.remediation}
"""
        
        if not self.findings:
            report += "\nNo vulnerabilities found!\n"
        
        report += f"\n{'='*60}\n"
        return report


# Pytest integration
@pytest.fixture
def security_suite():
    """Create security test suite."""
    import os
    base_url = os.environ.get('TEST_BASE_URL', 'http://localhost:8000')
    api_key = os.environ.get('TEST_API_KEY', 'test-key')
    return SecurityTestSuite(base_url, api_key)


@pytest.mark.asyncio
async def test_security_scan(security_suite):
    """Run full security scan."""
    findings = await security_suite.run_all_tests()
    
    print(security_suite.generate_report())
    
    # Fail if critical vulnerabilities found
    critical = [f for f in findings if f.severity == 'critical']
    assert len(critical) == 0, f"Critical vulnerabilities found: {len(critical)}"


# CLI runner
if __name__ == "__main__":
    import sys
    
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    api_key = sys.argv[2] if len(sys.argv) > 2 else None
    
    suite = SecurityTestSuite(base_url, api_key)
    asyncio.run(suite.run_all_tests())
    print(suite.generate_report())
