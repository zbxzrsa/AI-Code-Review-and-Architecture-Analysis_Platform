# Neo4j Graph Database Design

## Overview

Comprehensive Neo4j graph schema for code architecture analysis, dependency tracking, and complexity visualization. Supports temporal queries, circular dependency detection, and impact analysis.

---

## Node Types

### Code Structure Nodes

#### Package

Represents a logical grouping of modules.

**Properties**:

- `id` (String, unique) - Package identifier
- `name` (String, unique) - Package name
- `description` (Text) - Package description
- `version` (String) - Current version
- `created_at` (DateTime) - Creation timestamp
- `last_modified` (DateTime) - Last modification
- `maintainer` (String) - Maintainer name
- `stability` (String) - stable, beta, experimental

**Example**:

```cypher
CREATE (pkg:Package {
    id: 'auth',
    name: 'auth',
    description: 'Authentication module',
    version: 'v2.1.0',
    stability: 'stable'
})
```

#### Module

Represents a code module or namespace.

**Properties**:

- `id` (String, unique) - Module identifier
- `name` (String, unique) - Module name
- `package` (String) - Parent package
- `description` (Text) - Module description
- `version` (String) - Current version
- `created_at` (DateTime)
- `last_modified` (DateTime)
- `lines_of_code` (Integer)
- `test_coverage` (Float) - 0-1 scale
- `documentation_score` (Float) - 0-1 scale

#### File

Represents a source code file.

**Properties**:

- `id` (String, unique) - File identifier
- `path` (String, unique) - Full file path
- `name` (String) - File name
- `language` (String) - Programming language
- `lines_of_code` (Integer)
- `lines_of_comments` (Integer)
- `created_at` (DateTime)
- `last_modified` (DateTime)
- `size_bytes` (Integer)
- `complexity_score` (Float)

#### Class

Represents a class or struct.

**Properties**:

- `id` (String, unique) - Class identifier
- `name` (String) - Class name
- `module` (String) - Parent module
- `file` (String) - Source file
- `methods_count` (Integer)
- `fields_count` (Integer)
- `lines_of_code` (Integer)
- `cyclomatic_complexity` (Integer)
- `version` (String)
- `created_at` (DateTime)
- `last_modified` (DateTime)
- `is_abstract` (Boolean)
- `is_interface` (Boolean)
- `documentation_score` (Float)

#### Interface

Represents an interface or protocol.

**Properties**:

- `id` (String, unique) - Interface identifier
- `name` (String) - Interface name
- `module` (String) - Parent module
- `methods_count` (Integer)
- `version` (String)
- `created_at` (DateTime)
- `documentation_score` (Float)

#### Function

Represents a function or method.

**Properties**:

- `id` (String, unique) - Function identifier
- `name` (String) - Function name
- `module` (String) - Parent module
- `class` (String) - Parent class (if applicable)
- `file` (String) - Source file
- `lines_of_code` (Integer)
- `cyclomatic_complexity` (Integer)
- `parameters_count` (Integer)
- `return_type` (String)
- `is_public` (Boolean)
- `is_async` (Boolean)
- `version` (String)
- `created_at` (DateTime)
- `last_modified` (DateTime)
- `test_coverage` (Float)
- `documentation_score` (Float)
- `avg_execution_time_ms` (Float)

#### Variable

Represents a class field or global variable.

**Properties**:

- `id` (String, unique) - Variable identifier
- `name` (String) - Variable name
- `class` (String) - Parent class
- `type` (String) - Data type
- `is_public` (Boolean)
- `is_static` (Boolean)
- `version` (String)
- `created_at` (DateTime)

#### Exception

Represents an exception type.

**Properties**:

- `id` (String, unique) - Exception identifier
- `name` (String) - Exception name
- `module` (String) - Parent module
- `parent_exception` (String) - Parent exception type
- `version` (String)

### Analysis Nodes

#### Analysis

Represents a code analysis run.

**Properties**:

- `id` (String, unique) - Analysis identifier
- `name` (String) - Analysis name
- `type` (String) - dependency, complexity, coverage, etc.
- `version` (String) - Code version analyzed
- `created_at` (DateTime)
- `project` (String) - Project name
- `analyzed_files` (Integer)
- `total_functions` (Integer)
- `total_classes` (Integer)
- `total_modules` (Integer)
- `circular_dependencies_found` (Integer)
- `avg_complexity` (Float)
- `avg_test_coverage` (Float)

#### Metric

Represents a calculated metric.

**Properties**:

- `id` (String, unique) - Metric identifier
- `name` (String) - Metric name
- `type` (String) - coupling, complexity, coverage, etc.
- `analysis_id` (String) - Associated analysis
- `value` (Float) - Metric value
- `threshold` (Float) - Warning threshold
- `status` (String) - ok, warning, critical
- `created_at` (DateTime)

---

## Relationship Types

### Structural Relationships

#### CONTAINS

Package contains Module, Module contains File.

**Properties**:

- None

**Example**:

```cypher
CREATE (pkg:Package {name: 'auth'})-[:CONTAINS]->(mod:Module {name: 'jwt'})
```

#### DEFINES

File defines Class, Class defines Method.

**Properties**:

- None

#### HAS_METHOD

Class has Method (Function).

**Properties**:

- `version` (String)

#### HAS_FIELD

Class has Field (Variable).

**Properties**:

- `version` (String)

### Behavioral Relationships

#### CALLS

Function calls another Function.

**Properties**:

- `valid_from` (DateTime) - When relationship started
- `valid_to` (DateTime, nullable) - When relationship ended (NULL = current)
- `call_count` (Integer) - Number of calls
- `avg_latency_ms` (Float) - Average execution time
- `version` (String) - Code version
- `is_direct` (Boolean) - Direct vs indirect call

**Example**:

```cypher
CREATE (f1:Function)-[:CALLS {
    valid_from: datetime('2024-01-01T00:00:00Z'),
    valid_to: NULL,
    call_count: 150,
    avg_latency_ms: 5.2,
    version: 'v2.1.0',
    is_direct: true
}]->(f2:Function)
```

#### DEPENDS_ON

Module depends on another Module.

**Properties**:

- `valid_from` (DateTime) - When dependency started
- `valid_to` (DateTime, nullable) - When dependency ended
- `dependency_type` (String) - import, require, include
- `is_circular` (Boolean) - Part of circular dependency
- `version` (String)
- `strength` (String) - weak, medium, strong

**Example**:

```cypher
CREATE (m1:Module)-[:DEPENDS_ON {
    valid_from: datetime('2024-01-01T00:00:00Z'),
    valid_to: NULL,
    dependency_type: 'import',
    is_circular: false,
    version: 'v2.1.0',
    strength: 'strong'
}]->(m2:Module)
```

### Type Relationships

#### IMPLEMENTS

Class implements Interface.

**Properties**:

- `version` (String)

#### INHERITS

Class inherits from another Class.

**Properties**:

- `version` (String)

#### USES

Function uses Variable.

**Properties**:

- `access_type` (String) - read, write, read_write
- `version` (String)

#### THROWS

Function throws Exception.

**Properties**:

- `version` (String)

### Analysis Relationships

#### ANALYZES

Analysis analyzes Entity (Package, Module, etc.).

**Properties**:

- None

#### PRODUCES

Analysis produces Metric.

**Properties**:

- None

---

## Constraints

```cypher
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

CREATE CONSTRAINT unique_analysis IF NOT EXISTS
FOR (a:Analysis) REQUIRE a.id IS UNIQUE;

CREATE CONSTRAINT unique_metric IF NOT EXISTS
FOR (m:Metric) REQUIRE m.id IS UNIQUE;
```

---

## Indexes

```cypher
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
```

---

## Key Analysis Queries

### 1. Circular Dependency Detection

Find all circular dependencies in the codebase:

```cypher
MATCH path = (m:Module)-[:DEPENDS_ON*]->(m)
WHERE ALL(r IN relationships(path) WHERE r.valid_to IS NULL)
RETURN path, length(path) as cycle_length
ORDER BY cycle_length DESC
```

### 2. Coupling Analysis

Find highly coupled modules:

```cypher
MATCH (m:Module)-[r:DEPENDS_ON]->()
WHERE r.valid_to IS NULL
WITH m, count(r) as fan_out
WHERE fan_out > 10
RETURN m.name, fan_out
ORDER BY fan_out DESC
```

### 3. Critical Function Detection

Find functions called by many others:

```cypher
MATCH (caller:Function)-[r:CALLS]->(f:Function)
WHERE r.valid_to IS NULL
WITH f, count(caller) as caller_count, avg(r.avg_latency_ms) as avg_latency
WHERE caller_count > 20
RETURN f.name, caller_count, avg_latency
ORDER BY caller_count DESC
```

### 4. Dependency Drift

Find dependencies changed in last 30 days:

```cypher
MATCH (m:Module)-[r:DEPENDS_ON]->(target)
WHERE r.valid_from > datetime() - duration({days: 30})
RETURN m.name, target.name, r.valid_from
ORDER BY r.valid_from DESC
```

### 5. Time-Travel Query

What did dependencies look like on a specific date?

```cypher
MATCH (m:Module)-[r:DEPENDS_ON]->(target)
WHERE r.valid_from <= datetime('2024-06-01T00:00:00Z')
  AND (r.valid_to IS NULL OR r.valid_to > datetime('2024-06-01T00:00:00Z'))
RETURN m.name, collect(target.name) as dependencies
```

---

## Temporal Support

All relationships support temporal validity:

- `valid_from` (DateTime) - When relationship became active
- `valid_to` (DateTime, nullable) - When relationship ended (NULL = still active)

This enables:

- **Time-travel queries** - See state at any point in time
- **Dependency tracking** - Track when dependencies were added/removed
- **Change detection** - Find what changed between versions
- **Impact analysis** - Understand evolution of architecture

### Updating Temporal Relationships

```cypher
// Mark old relationship as invalid
MATCH (m1:Module)-[r:DEPENDS_ON]->(m2:Module)
WHERE m1.name = 'jwt' AND m2.name = 'old_module'
SET r.valid_to = datetime('2024-12-01T10:00:00Z')

// Create new relationship
CREATE (m1:Module {name: 'jwt'})-[:DEPENDS_ON {
    valid_from: datetime('2024-12-01T10:00:00Z'),
    valid_to: NULL,
    dependency_type: 'import',
    is_circular: false,
    version: 'v2.1.0',
    strength: 'strong'
}]->(m2:Module {name: 'new_module'})
```

---

## Use Cases

### 1. Architecture Visualization

- Visualize module dependencies
- Show inheritance hierarchies
- Display call graphs

### 2. Quality Analysis

- Identify complexity hotspots
- Find untested critical functions
- Detect dead code

### 3. Impact Analysis

- Determine impact of changes
- Find affected modules
- Trace dependency chains

### 4. Performance Analysis

- Find slow functions
- Identify bottlenecks
- Trace call chains with high latency

### 5. Refactoring Guidance

- Suggest decoupling opportunities
- Identify circular dependencies
- Find modules to consolidate

### 6. Compliance & Governance

- Track architectural decisions
- Enforce dependency rules
- Monitor test coverage

---

## Integration with PostgreSQL

**PostgreSQL**: Stores operational data

- User accounts
- Analysis sessions
- Metrics and results
- Audit logs

**Neo4j**: Stores structural data

- Code entities (functions, classes, modules)
- Dependencies and relationships
- Temporal change history
- Architecture patterns

**Synchronization**:

- Analysis service populates Neo4j from code
- PostgreSQL stores analysis metadata
- Queries combine both databases

---

## Performance Considerations

### Query Optimization

1. **Use indexes** on frequently queried properties
2. **Limit path lengths** in recursive queries
3. **Use APOC** for complex operations
4. **Cache results** for expensive queries

### Data Management

1. **Partition by version** for large codebases
2. **Archive old analyses** periodically
3. **Monitor graph size** and query performance
4. **Use relationship properties** instead of separate nodes

### Scaling

- Neo4j handles millions of nodes efficiently
- Use clustering for high availability
- Consider sharding for very large graphs

---

## Best Practices

1. **Consistent naming** - Use clear, consistent identifiers
2. **Temporal accuracy** - Always set valid_from/valid_to
3. **Regular analysis** - Run analysis regularly to keep graph fresh
4. **Query optimization** - Profile slow queries
5. **Backup strategy** - Regular backups of Neo4j database
6. **Version tracking** - Include version in all nodes/relationships

---

## Future Enhancements

- [ ] Machine learning for anomaly detection
- [ ] Real-time graph updates
- [ ] Advanced visualization
- [ ] Custom rule engine
- [ ] Integration with IDE plugins
- [ ] Automated refactoring suggestions
