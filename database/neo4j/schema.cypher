// ============================================
// Neo4j Graph Schema for Code Architecture Analysis
// ============================================

// ============================================
// Constraints
// ============================================

// Code entity constraints
CREATE CONSTRAINT unique_function IF NOT EXISTS
FOR (f:Function) REQUIRE f.id IS UNIQUE;

CREATE CONSTRAINT unique_class IF NOT EXISTS
FOR (c:Class) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT unique_file IF NOT EXISTS
FOR (f:File) REQUIRE f.path IS UNIQUE;

CREATE CONSTRAINT unique_module IF NOT EXISTS
FOR (m:Module) REQUIRE m.name IS UNIQUE;

CREATE CONSTRAINT unique_package IF NOT EXISTS
FOR (p:Package) REQUIRE p.name IS UNIQUE;

CREATE CONSTRAINT unique_interface IF NOT EXISTS
FOR (i:Interface) REQUIRE i.id IS UNIQUE;

CREATE CONSTRAINT unique_variable IF NOT EXISTS
FOR (v:Variable) REQUIRE v.id IS UNIQUE;

// Analysis constraints
CREATE CONSTRAINT unique_analysis IF NOT EXISTS
FOR (a:Analysis) REQUIRE a.id IS UNIQUE;

CREATE CONSTRAINT unique_metric IF NOT EXISTS
FOR (m:Metric) REQUIRE m.id IS UNIQUE;

// ============================================
// Indexes for Performance
// ============================================

CREATE INDEX idx_function_name IF NOT EXISTS
FOR (f:Function) ON (f.name);

CREATE INDEX idx_class_name IF NOT EXISTS
FOR (c:Class) ON (c.name);

CREATE INDEX idx_module_name IF NOT EXISTS
FOR (m:Module) ON (m.name);

CREATE INDEX idx_file_path IF NOT EXISTS
FOR (f:File) ON (f.path);

CREATE INDEX idx_function_complexity IF NOT EXISTS
FOR (f:Function) ON (f.cyclomatic_complexity);

CREATE INDEX idx_function_loc IF NOT EXISTS
FOR (f:Function) ON (f.lines_of_code);

CREATE INDEX idx_created_at IF NOT EXISTS
FOR (n) ON (n.created_at);

CREATE INDEX idx_modified_at IF NOT EXISTS
FOR (n) ON (n.last_modified);

// ============================================
// Node Creation Examples
// ============================================

// Package node
CREATE (pkg:Package {
    id: 'auth',
    name: 'auth',
    description: 'Authentication and authorization module',
    version: 'v2.1.0',
    created_at: datetime('2024-01-01T00:00:00Z'),
    last_modified: datetime('2024-12-01T10:00:00Z'),
    maintainer: 'security-team',
    stability: 'stable'
})

// Module node
CREATE (mod:Module {
    id: 'auth.jwt',
    name: 'jwt',
    package: 'auth',
    description: 'JWT token handling',
    version: 'v2.1.0',
    created_at: datetime('2024-01-01T00:00:00Z'),
    last_modified: datetime('2024-12-01T10:00:00Z'),
    lines_of_code: 250,
    test_coverage: 0.95,
    documentation_score: 0.85
})

// File node
CREATE (file:File {
    id: 'auth/jwt.py',
    path: 'backend/services/auth-service/src/jwt.py',
    name: 'jwt.py',
    language: 'python',
    lines_of_code: 250,
    lines_of_comments: 45,
    created_at: datetime('2024-01-01T00:00:00Z'),
    last_modified: datetime('2024-12-01T10:00:00Z'),
    size_bytes: 8500,
    complexity_score: 0.72
})

// Class node
CREATE (cls:Class {
    id: 'auth.jwt.JWTManager',
    name: 'JWTManager',
    module: 'auth.jwt',
    file: 'auth/jwt.py',
    methods_count: 12,
    fields_count: 8,
    lines_of_code: 180,
    cyclomatic_complexity: 15,
    version: 'v2.1.0',
    created_at: datetime('2024-01-01T00:00:00Z'),
    last_modified: datetime('2024-12-01T10:00:00Z'),
    is_abstract: false,
    is_interface: false,
    documentation_score: 0.9
})

// Interface node
CREATE (iface:Interface {
    id: 'auth.jwt.TokenProvider',
    name: 'TokenProvider',
    module: 'auth.jwt',
    methods_count: 5,
    version: 'v2.1.0',
    created_at: datetime('2024-01-01T00:00:00Z'),
    documentation_score: 0.95
})

// Function node
CREATE (func:Function {
    id: 'auth.jwt.JWTManager.create_token',
    name: 'create_token',
    module: 'auth.jwt',
    class: 'JWTManager',
    file: 'auth/jwt.py',
    lines_of_code: 45,
    cyclomatic_complexity: 8,
    parameters_count: 4,
    return_type: 'str',
    is_public: true,
    is_async: true,
    version: 'v2.1.0',
    created_at: datetime('2024-01-01T00:00:00Z'),
    last_modified: datetime('2024-12-01T10:00:00Z'),
    test_coverage: 1.0,
    documentation_score: 0.95,
    avg_execution_time_ms: 5.2
})

// Variable node
CREATE (var:Variable {
    id: 'auth.jwt.JWTManager.secret_key',
    name: 'secret_key',
    class: 'JWTManager',
    type: 'str',
    is_public: false,
    is_static: true,
    version: 'v2.1.0',
    created_at: datetime('2024-01-01T00:00:00Z')
})

// Analysis node
CREATE (analysis:Analysis {
    id: 'analysis_2024_12_01',
    name: 'Architecture Analysis',
    type: 'dependency',
    version: 'v2.1.0',
    created_at: datetime('2024-12-01T10:00:00Z'),
    project: 'ai-code-review',
    analyzed_files: 150,
    total_functions: 450,
    total_classes: 80,
    total_modules: 25,
    circular_dependencies_found: 2,
    avg_complexity: 7.5,
    avg_test_coverage: 0.82
})

// Metric node
CREATE (metric:Metric {
    id: 'metric_coupling_2024_12_01',
    name: 'Module Coupling',
    type: 'coupling',
    analysis_id: 'analysis_2024_12_01',
    value: 0.68,
    threshold: 0.75,
    status: 'warning',
    created_at: datetime('2024-12-01T10:00:00Z')
})

// ============================================
// Relationship Creation Examples
// ============================================

// Package contains Module
CREATE (pkg:Package {name: 'auth'})-[:CONTAINS]->(mod:Module {name: 'jwt'})

// Module contains File
CREATE (mod:Module {name: 'jwt'})-[:CONTAINS]->(file:File {path: 'backend/services/auth-service/src/jwt.py'})

// File contains Class
CREATE (file:File {path: 'backend/services/auth-service/src/jwt.py'})-[:DEFINES]->(cls:Class {name: 'JWTManager'})

// Class implements Interface
CREATE (cls:Class {name: 'JWTManager'})-[:IMPLEMENTS {version: 'v2.1.0'}]->(iface:Interface {name: 'TokenProvider'})

// Class contains Method (Function)
CREATE (cls:Class {name: 'JWTManager'})-[:HAS_METHOD {version: 'v2.1.0'}]->(func:Function {name: 'create_token'})

// Class has Field (Variable)
CREATE (cls:Class {name: 'JWTManager'})-[:HAS_FIELD {version: 'v2.1.0'}]->(var:Variable {name: 'secret_key'})

// Function calls Function (with temporal support)
CREATE (f1:Function {id: 'auth.jwt.JWTManager.create_token'})-[:CALLS {
    valid_from: datetime('2024-01-01T00:00:00Z'),
    valid_to: NULL,
    call_count: 150,
    avg_latency_ms: 5.2,
    version: 'v2.1.0',
    is_direct: true
}]->(f2:Function {id: 'auth.jwt.JWTManager.encode'})

// Module depends on Module (with temporal support)
CREATE (m1:Module {name: 'jwt'})-[:DEPENDS_ON {
    valid_from: datetime('2024-01-01T00:00:00Z'),
    valid_to: NULL,
    dependency_type: 'import',
    is_circular: false,
    version: 'v2.1.0',
    strength: 'strong'
}]->(m2:Module {name: 'security'})

// Class inherits from Class
CREATE (c1:Class {name: 'AdminUser'})-[:INHERITS {version: 'v2.1.0'}]->(c2:Class {name: 'User'})

// Function uses Variable
CREATE (func:Function {name: 'create_token'})-[:USES {
    access_type: 'read',
    version: 'v2.1.0'
}]->(var:Variable {name: 'secret_key'})

// Function throws Exception
CREATE (func:Function {name: 'create_token'})-[:THROWS {
    version: 'v2.1.0'
}]->(exc:Exception {name: 'InvalidTokenError'})

// Analysis analyzes Entity
CREATE (analysis:Analysis {id: 'analysis_2024_12_01'})-[:ANALYZES]->(pkg:Package {name: 'auth'})

// Analysis produces Metric
CREATE (analysis:Analysis {id: 'analysis_2024_12_01'})-[:PRODUCES]->(metric:Metric {name: 'Module Coupling'})

// ============================================
// Temporal Relationship Updates
// ============================================

// Mark old relationship as invalid (dependency removed)
MATCH (m1:Module {name: 'jwt'})-[r:DEPENDS_ON]->(m2:Module {name: 'old_module'})
SET r.valid_to = datetime('2024-12-01T10:00:00Z')

// Create new relationship with temporal validity
CREATE (m1:Module {name: 'jwt'})-[:DEPENDS_ON {
    valid_from: datetime('2024-12-01T10:00:00Z'),
    valid_to: NULL,
    dependency_type: 'import',
    is_circular: false,
    version: 'v2.1.0',
    strength: 'strong'
}]->(m2:Module {name: 'new_module'})
