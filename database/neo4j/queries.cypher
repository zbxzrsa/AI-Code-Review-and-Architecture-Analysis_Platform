// ============================================
// Neo4j Analysis Queries
// ============================================

// ============================================
// 1. CIRCULAR DEPENDENCY DETECTION
// ============================================

// Find all circular dependencies
MATCH path = (m:Module)-[:DEPENDS_ON*]->(m)
WHERE ALL(r IN relationships(path) WHERE r.valid_to IS NULL)
RETURN path, length(path) as cycle_length
ORDER BY cycle_length DESC

// Find circular dependencies with details
MATCH path = (m:Module)-[:DEPENDS_ON*2..5]->(m)
WHERE ALL(r IN relationships(path) WHERE r.valid_to IS NULL)
WITH m, path, [node IN nodes(path) | node.name] as cycle_path
RETURN m.name as module, cycle_path, length(path) as depth

// ============================================
// 2. COUPLING ANALYSIS
// ============================================

// Find highly coupled modules (fan-out > 10)
MATCH (m:Module)-[r:DEPENDS_ON]->()
WHERE r.valid_to IS NULL
WITH m, count(r) as fan_out
WHERE fan_out > 10
RETURN m.name, fan_out, m.lines_of_code, m.test_coverage
ORDER BY fan_out DESC

// Find modules with high fan-in (many dependents)
MATCH (m:Module)<-[r:DEPENDS_ON]-()
WHERE r.valid_to IS NULL
WITH m, count(r) as fan_in
WHERE fan_in > 5
RETURN m.name, fan_in
ORDER BY fan_in DESC

// Calculate coupling metrics
MATCH (m:Module)-[out:DEPENDS_ON]->()
WHERE out.valid_to IS NULL
WITH m, count(out) as fan_out
MATCH (m)<-[in:DEPENDS_ON]-()
WHERE in.valid_to IS NULL
WITH m, fan_out, count(in) as fan_in
RETURN m.name,
       fan_out,
       fan_in,
       fan_out + fan_in as total_coupling,
       CASE WHEN (fan_out + fan_in) > 15 THEN 'HIGH'
            WHEN (fan_out + fan_in) > 10 THEN 'MEDIUM'
            ELSE 'LOW' END as coupling_level
ORDER BY total_coupling DESC

// ============================================
// 3. DEPENDENCY DRIFT DETECTION
// ============================================

// Find dependencies changed in last 30 days
MATCH (m:Module)-[r:DEPENDS_ON]->(target)
WHERE r.valid_from > datetime() - duration({days: 30})
RETURN m.name as module,
       target.name as new_dependency,
       r.valid_from as changed_at,
       r.dependency_type,
       r.strength
ORDER BY r.valid_from DESC

// Find removed dependencies in last 30 days
MATCH (m:Module)-[r:DEPENDS_ON]->(target)
WHERE r.valid_to > datetime() - duration({days: 30})
RETURN m.name as module,
       target.name as removed_dependency,
       r.valid_to as removed_at,
       r.valid_from as added_at
ORDER BY r.valid_to DESC

// ============================================
// 4. CRITICAL FUNCTION ANALYSIS
// ============================================

// Find critical functions (called by many others)
MATCH (caller:Function)-[r:CALLS]->(f:Function)
WHERE r.valid_to IS NULL
WITH f, count(caller) as caller_count, avg(r.avg_latency_ms) as avg_latency
WHERE caller_count > 20
RETURN f.name,
       f.module,
       caller_count,
       avg_latency,
       f.cyclomatic_complexity,
       f.test_coverage
ORDER BY caller_count DESC

// Find functions with high complexity
MATCH (f:Function)
WHERE f.cyclomatic_complexity > 10
RETURN f.name,
       f.module,
       f.cyclomatic_complexity,
       f.lines_of_code,
       f.test_coverage
ORDER BY f.cyclomatic_complexity DESC

// Find untested critical functions
MATCH (caller:Function)-[r:CALLS]->(f:Function)
WHERE r.valid_to IS NULL AND f.test_coverage < 0.8
WITH f, count(caller) as caller_count
WHERE caller_count > 10
RETURN f.name,
       f.module,
       caller_count,
       f.test_coverage,
       f.cyclomatic_complexity
ORDER BY caller_count DESC

// ============================================
// 5. TIME-TRAVEL QUERIES
// ============================================

// What did dependencies look like on a specific date?
MATCH (m:Module)-[r:DEPENDS_ON]->(target)
WHERE r.valid_from <= datetime('2024-06-01T00:00:00Z')
  AND (r.valid_to IS NULL OR r.valid_to > datetime('2024-06-01T00:00:00Z'))
RETURN m.name, collect(target.name) as dependencies

// Track dependency evolution over time
MATCH (m:Module)-[r:DEPENDS_ON]->(target)
RETURN m.name,
       target.name,
       r.valid_from as added_at,
       r.valid_to as removed_at,
       CASE WHEN r.valid_to IS NULL THEN 'ACTIVE' ELSE 'REMOVED' END as status
ORDER BY r.valid_from DESC

// ============================================
// 6. COMPLEXITY HOTSPOTS
// ============================================

// Find modules with high average complexity
MATCH (m:Module)-[:CONTAINS]->(f:Function)
WITH m, avg(f.cyclomatic_complexity) as avg_complexity, count(f) as function_count
WHERE avg_complexity > 8
RETURN m.name,
       avg_complexity,
       function_count,
       m.lines_of_code,
       m.test_coverage
ORDER BY avg_complexity DESC

// Find classes with many methods
MATCH (c:Class)-[:HAS_METHOD]->(f:Function)
WITH c, count(f) as method_count, avg(f.cyclomatic_complexity) as avg_complexity
WHERE method_count > 15
RETURN c.name,
       method_count,
       avg_complexity,
       c.lines_of_code
ORDER BY method_count DESC

// ============================================
// 7. TEST COVERAGE ANALYSIS
// ============================================

// Find modules with low test coverage
MATCH (m:Module)
WHERE m.test_coverage < 0.8
RETURN m.name,
       m.test_coverage,
       m.lines_of_code,
       m.created_at
ORDER BY m.test_coverage ASC

// Find untested functions in critical modules
MATCH (m:Module)-[:CONTAINS]->(f:Function)
WHERE m.name IN ['auth', 'security', 'payment']
  AND f.test_coverage < 0.9
RETURN m.name,
       f.name,
       f.test_coverage,
       f.cyclomatic_complexity
ORDER BY f.cyclomatic_complexity DESC

// ============================================
// 8. INHERITANCE HIERARCHY
// ============================================

// Find inheritance chains
MATCH path = (c:Class)-[:INHERITS*]->(parent:Class)
RETURN [node IN nodes(path) | node.name] as inheritance_chain,
       length(path) as depth
ORDER BY depth DESC

// Find classes with multiple inheritance levels
MATCH (c:Class)-[:INHERITS*2..]->(ancestor:Class)
WITH c, count(DISTINCT ancestor) as ancestor_count
WHERE ancestor_count > 1
RETURN c.name, ancestor_count

// ============================================
// 9. INTERFACE IMPLEMENTATION
// ============================================

// Find interfaces with many implementations
MATCH (c:Class)-[:IMPLEMENTS]->(i:Interface)
WITH i, count(c) as implementation_count
WHERE implementation_count > 5
RETURN i.name,
       implementation_count,
       collect(c.name) as implementations
ORDER BY implementation_count DESC

// ============================================
// 10. DEAD CODE DETECTION
// ============================================

// Find functions never called
MATCH (f:Function)
WHERE NOT (f)<-[:CALLS]-()
RETURN f.name,
       f.module,
       f.lines_of_code,
       f.last_modified
ORDER BY f.last_modified DESC

// Find unused classes
MATCH (c:Class)
WHERE NOT (c)<-[:DEFINES]-()
  AND NOT (c)<-[:INHERITS]-()
  AND NOT (c)<-[:IMPLEMENTS]-()
RETURN c.name,
       c.module,
       c.lines_of_code

// ============================================
// 11. PERFORMANCE ANALYSIS
// ============================================

// Find slow functions
MATCH (caller:Function)-[r:CALLS]->(f:Function)
WHERE r.valid_to IS NULL
WITH f, avg(r.avg_latency_ms) as avg_latency, count(r) as call_count
WHERE avg_latency > 100
RETURN f.name,
       f.module,
       avg_latency,
       call_count,
       f.cyclomatic_complexity
ORDER BY avg_latency DESC

// Find call chains with high latency
MATCH path = (start:Function)-[:CALLS*1..5]->(end:Function)
WHERE ALL(r IN relationships(path) WHERE r.valid_to IS NULL)
WITH path, [r IN relationships(path) | r.avg_latency_ms] as latencies
WITH path, reduce(total = 0, lat IN latencies | total + lat) as total_latency
WHERE total_latency > 500
RETURN [node IN nodes(path) | node.name] as call_chain,
       total_latency
ORDER BY total_latency DESC
LIMIT 10

// ============================================
// 12. ARCHITECTURE METRICS
// ============================================

// Calculate overall architecture health
MATCH (m:Module)
WITH count(m) as total_modules,
     avg(m.test_coverage) as avg_coverage,
     avg(m.lines_of_code) as avg_loc
MATCH (m:Module)-[r:DEPENDS_ON]->()
WHERE r.valid_to IS NULL
WITH total_modules, avg_coverage, avg_loc, count(r) as total_dependencies
RETURN total_modules,
       total_dependencies,
       avg_coverage,
       avg_loc,
       CASE WHEN avg_coverage > 0.85 THEN 'GOOD'
            WHEN avg_coverage > 0.75 THEN 'ACCEPTABLE'
            ELSE 'POOR' END as coverage_status

// ============================================
// 13. IMPACT ANALYSIS
// ============================================

// Find impact of changing a module
MATCH (m:Module {name: 'auth'})-[:DEPENDS_ON*1..3]->(affected:Module)
RETURN m.name,
       affected.name,
       affected.lines_of_code,
       affected.test_coverage

// Find all modules that would be affected by a change
MATCH (m:Module {name: 'core'})<-[:DEPENDS_ON*1..5]-(dependent:Module)
RETURN DISTINCT dependent.name,
       dependent.lines_of_code,
       dependent.test_coverage
ORDER BY dependent.lines_of_code DESC

// ============================================
// 14. DOCUMENTATION ANALYSIS
// ============================================

// Find undocumented critical functions
MATCH (caller:Function)-[r:CALLS]->(f:Function)
WHERE r.valid_to IS NULL
  AND f.documentation_score < 0.7
WITH f, count(caller) as caller_count
WHERE caller_count > 10
RETURN f.name,
       f.module,
       f.documentation_score,
       caller_count
ORDER BY caller_count DESC

// ============================================
// 15. VERSION TRACKING
// ============================================

// Find changes between versions
MATCH (n)
WHERE n.version = 'v2.1.0'
RETURN labels(n)[0] as entity_type,
       count(n) as count
GROUP BY labels(n)[0]

// Track entity modifications over time
MATCH (f:Function)
WHERE f.last_modified > datetime() - duration({days: 7})
RETURN f.name,
       f.module,
       f.last_modified,
       f.cyclomatic_complexity,
       f.test_coverage
ORDER BY f.last_modified DESC
