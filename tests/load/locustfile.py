"""
Load Testing Configuration using Locust

Run with:
    locust -f tests/load/locustfile.py --host=http://localhost
    
Web UI: http://localhost:8089
"""

import json
import random
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner


# =============================================================================
# Test Data
# =============================================================================

SAMPLE_CODE = '''
def calculate_sum(numbers):
    """Calculate sum of numbers"""
    total = 0
    for num in numbers:
        total += num
    return total

class UserService:
    def __init__(self, db):
        self.db = db
    
    def get_user(self, user_id):
        query = f"SELECT * FROM users WHERE id = {user_id}"  # SQL Injection!
        return self.db.execute(query)
'''

TEST_USERS = [
    {"email": f"loadtest{i}@example.com", "password": "LoadTest123!"}
    for i in range(100)
]

# API endpoint constants
API_PROJECTS = "/api/projects/"
API_ANALYSIS_ANALYZE = "/api/analysis/analyze"


# =============================================================================
# Auth Service Load Tests
# =============================================================================

class AuthServiceUser(HttpUser):
    """Load test for Auth Service"""
    
    wait_time = between(1, 3)
    weight = 3
    
    def on_start(self):
        """Setup - register and login"""
        self.user_data = random.choice(TEST_USERS)
        self.token = None
    
    @task(1)
    def health_check(self):
        """Test health endpoint"""
        self.client.get("/api/auth/health", name="Auth Health")
    
    @task(2)
    def login(self):
        """Test login endpoint"""
        response = self.client.post(
            "/api/auth/login",
            json={
                "email": self.user_data["email"],
                "password": self.user_data["password"]
            },
            name="Auth Login"
        )
        if response.status_code == 200:
            # Extract token if returned in body (for testing)
            data = response.json()
            self.token = data.get("access_token")
    
    @task(1)
    def get_me(self):
        """Test get current user"""
        if self.token:
            self.client.get(
                "/api/auth/me",
                headers={"Authorization": f"Bearer {self.token}"},
                name="Auth Me"
            )


# =============================================================================
# Project Service Load Tests
# =============================================================================

class ProjectServiceUser(HttpUser):
    """Load test for Project Service"""
    
    wait_time = between(1, 5)
    weight = 2
    
    def on_start(self):
        """Setup - login first"""
        self.token = self._get_token()
        self.project_ids = []
    
    def _get_token(self):
        """Get auth token"""
        response = self.client.post(
            "/api/auth/login",
            json={"email": "loadtest0@example.com", "password": "LoadTest123!"}
        )
        if response.status_code == 200:
            return response.json().get("access_token", "test-token")
        return "test-token"
    
    @task(1)
    def health_check(self):
        """Test health endpoint"""
        self.client.get("/api/projects/health", name="Project Health")
    
    @task(3)
    def list_projects(self):
        """Test list projects"""
        self.client.get(
            API_PROJECTS,
            headers={"Authorization": f"Bearer {self.token}"},
            name="List Projects"
        )
    
    @task(2)
    def create_project(self):
        """Test create project"""
        response = self.client.post(
            API_PROJECTS,
            json={
                "name": f"LoadTest Project {random.randint(1, 10000)}",
                "description": "Load test project",
                "language": random.choice(["python", "javascript", "typescript"])
            },
            headers={"Authorization": f"Bearer {self.token}"},
            name="Create Project"
        )
        if response.status_code in [200, 201]:
            data = response.json()
            if "id" in data:
                self.project_ids.append(data["id"])
    
    @task(2)
    def get_project(self):
        """Test get single project"""
        if self.project_ids:
            project_id = random.choice(self.project_ids)
            self.client.get(
                f"/api/projects/{project_id}",
                headers={"Authorization": f"Bearer {self.token}"},
                name="Get Project"
            )


# =============================================================================
# Analysis Service Load Tests
# =============================================================================

class AnalysisServiceUser(HttpUser):
    """Load test for Analysis Service"""
    
    wait_time = between(5, 15)  # Analysis takes longer
    weight = 1
    
    def on_start(self):
        """Setup"""
        self.token = "test-token"
    
    @task(1)
    def health_check(self):
        """Test health endpoint"""
        self.client.get("/api/analysis/health", name="Analysis Health")
    
    @task(2)
    def analyze_code(self):
        """Test code analysis"""
        self.client.post(
            API_ANALYSIS_ANALYZE,
            json={
                "code": SAMPLE_CODE,
                "language": "python",
                "analysis_type": "quick"
            },
            headers={"Authorization": f"Bearer {self.token}"},
            name="Analyze Code",
            timeout=60
        )
    
    @task(1)
    def analyze_code_deep(self):
        """Test deep code analysis"""
        self.client.post(
            API_ANALYSIS_ANALYZE,
            json={
                "code": SAMPLE_CODE,
                "language": "python",
                "analysis_type": "deep"
            },
            headers={"Authorization": f"Bearer {self.token}"},
            name="Deep Analysis",
            timeout=120
        )


# =============================================================================
# API Gateway Load Tests
# =============================================================================

class APIGatewayUser(HttpUser):
    """Load test for API Gateway"""
    
    wait_time = between(0.5, 2)
    weight = 5
    
    @task(5)
    def health_check(self):
        """Test gateway health"""
        self.client.get("/health", name="Gateway Health")
    
    @task(3)
    def auth_health(self):
        """Test auth through gateway"""
        self.client.get("/api/auth/health", name="Gateway -> Auth")
    
    @task(3)
    def project_health(self):
        """Test project through gateway"""
        self.client.get("/api/projects/health", name="Gateway -> Project")
    
    @task(2)
    def analysis_health(self):
        """Test analysis through gateway"""
        self.client.get("/api/analysis/health", name="Gateway -> Analysis")


# =============================================================================
# Mixed Workload
# =============================================================================

class MixedWorkloadUser(HttpUser):
    """Simulate realistic mixed workload"""
    
    wait_time = between(1, 5)
    weight = 4
    
    def on_start(self):
        self.token = None
        self.project_id = None
    
    @task(10)
    def browse_projects(self):
        """Simulate browsing projects"""
        self.client.get(API_PROJECTS, name="Browse Projects")
    
    @task(5)
    def view_project(self):
        """Simulate viewing a project"""
        if self.project_id:
            self.client.get(f"/api/projects/{self.project_id}", name="View Project")
    
    @task(2)
    def run_analysis(self):
        """Simulate running analysis"""
        self.client.post(
            API_ANALYSIS_ANALYZE,
            json={
                "code": "print('hello')",
                "language": "python"
            },
            name="Quick Analysis",
            timeout=30
        )
    
    @task(1)
    def create_project(self):
        """Simulate creating a project"""
        response = self.client.post(
            API_PROJECTS,
            json={
                "name": f"Project {random.randint(1, 10000)}",
                "language": "python"
            },
            name="Create Project"
        )
        if response.status_code in [200, 201]:
            data = response.json()
            self.project_id = data.get("id")


# =============================================================================
# Event Handlers
# =============================================================================

@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Initialize load test"""
    if isinstance(environment.runner, MasterRunner):
        print("Load test initialized on master")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts"""
    print(f"Load test starting with {environment.runner.user_count} users")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops"""
    print("Load test completed")
    
    # Print summary stats
    stats = environment.runner.stats
    print(f"\nTotal requests: {stats.total.num_requests}")
    print(f"Failed requests: {stats.total.num_failures}")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"Requests per second: {stats.total.total_rps:.2f}")
