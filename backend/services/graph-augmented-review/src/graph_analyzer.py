"""
Graph-Augmented Code Review Analyzer

整合Neo4j知识图谱的代码审查分析器，提供架构上下文智能分析。
支持AST、CFG、数据流图的构建和查询，检测架构违规和依赖风险。
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
import hashlib

from neo4j import GraphDatabase
import networkx as nx

logger = logging.getLogger(__name__)


class GraphRelationshipType(str, Enum):
    """图关系类型"""
    CALLS = "CALLS"
    DEPENDS_ON = "DEPENDS_ON"
    USES = "USES"
    DEFINES = "DEFINES"
    IMPLEMENTS = "IMPLEMENTS"
    INHERITS = "INHERITS"
    DATAFLOW = "DATAFLOW"
    CONTROLFLOW = "CONTROLFLOW"


class ArchitecturalViolationType(str, Enum):
    """架构违规类型"""
    CIRCULAR_DEPENDENCY = "circular_dependency"
    HIGH_COUPLING = "high_coupling"
    LAYER_VIOLATION = "layer_violation"
    DEPENDENCY_DRIFT = "dependency_drift"
    CYCLE_DETECTED = "cycle_detected"
    MODULE_BOUNDARY_VIOLATION = "module_boundary_violation"


@dataclass
class GraphNode:
    """图节点表示"""
    id: str
    type: str  # Function, Class, Module, File, etc.
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphRelationship:
    """图关系表示"""
    source_id: str
    target_id: str
    type: GraphRelationshipType
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArchitecturalImpact:
    """架构影响分析结果"""
    violation_type: ArchitecturalViolationType
    severity: str  # critical, high, medium, low
    affected_components: List[str]
    impact_path: List[str]  # 影响路径
    confidence: float
    explanation: str
    suggested_fix: Optional[str] = None
    historical_similar_fixes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DataflowAnalysis:
    """数据流分析结果"""
    source: str
    sink: str
    data_path: List[str]
    sensitive_data_detected: bool = False
    security_risk: Optional[str] = None


@dataclass
class GraphReviewResult:
    """图增强审查结果"""
    review_id: str
    code_hash: str
    
    # 架构分析
    architectural_violations: List[ArchitecturalImpact]
    dependency_risks: List[Dict[str, Any]]
    dataflow_analysis: List[DataflowAnalysis]
    
    # 影响分析
    affected_modules: Set[str]
    call_hierarchy_depth: int
    coupling_metrics: Dict[str, float]
    
    # 证据链
    evidence_chain: List[Dict[str, Any]]
    confidence_score: float
    
    timestamp: datetime


class Neo4jGraphAnalyzer:
    """
    Neo4j图数据库分析器
    
    功能：
    - 构建代码关系图（AST, CFG, 数据流）
    - 检测架构违规
    - 分析依赖风险
    - 追踪数据流
    - 提供证据链
    """
    
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        database: str = "neo4j"
    ):
        """
        初始化Neo4j连接
        
        Args:
            neo4j_uri: Neo4j连接URI
            neo4j_user: 用户名
            neo4j_password: 密码
            database: 数据库名称
        """
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.database = database
        self._verify_connection()
    
    def _verify_connection(self):
        """验证Neo4j连接"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                result.single()
            logger.info("Neo4j connection verified")
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            raise
    
    def close(self):
        """关闭连接"""
        self.driver.close()
    
    async def analyze_code_change(
        self,
        code: str,
        file_path: str,
        language: str,
        project_id: str,
        commit_hash: Optional[str] = None
    ) -> GraphReviewResult:
        """
        分析代码变更的架构影响
        
        Args:
            code: 源代码
            file_path: 文件路径
            language: 编程语言
            project_id: 项目ID
            commit_hash: 提交哈希（可选）
        
        Returns:
            GraphReviewResult: 图增强审查结果
        """
        review_id = str(hashlib.sha256(f"{file_path}:{code}".encode()).hexdigest()[:16])
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        
        logger.info(f"Starting graph analysis for {file_path}")
        
        # 1. 构建代码图
        graph_nodes, graph_relationships = await self._build_code_graph(
            code, file_path, language, project_id
        )
        
        # 2. 存储到Neo4j
        await self._store_graph(graph_nodes, graph_relationships, project_id, commit_hash)
        
        # 3. 检测架构违规
        violations = await self._detect_architectural_violations(
            file_path, project_id
        )
        
        # 4. 分析依赖风险
        dependency_risks = await self._analyze_dependency_risks(
            file_path, project_id
        )
        
        # 5. 数据流分析
        dataflow_analysis = await self._analyze_dataflow(
            file_path, project_id
        )
        
        # 6. 影响分析
        affected_modules = await self._get_affected_modules(file_path, project_id)
        call_hierarchy = await self._analyze_call_hierarchy(file_path, project_id)
        coupling_metrics = await self._calculate_coupling_metrics(file_path, project_id)
        
        # 7. 构建证据链
        evidence_chain = await self._build_evidence_chain(
            violations, dependency_risks, project_id
        )
        
        # 8. 计算置信度
        confidence = self._calculate_confidence(violations, dependency_risks, evidence_chain)
        
        return GraphReviewResult(
            review_id=review_id,
            code_hash=code_hash,
            architectural_violations=violations,
            dependency_risks=dependency_risks,
            dataflow_analysis=dataflow_analysis,
            affected_modules=affected_modules,
            call_hierarchy_depth=call_hierarchy.get("max_depth", 0),
            coupling_metrics=coupling_metrics,
            evidence_chain=evidence_chain,
            confidence_score=confidence,
            timestamp=datetime.now(timezone.utc)
        )
    
    async def _build_code_graph(
        self,
        code: str,
        file_path: str,
        language: str,
        project_id: str
    ) -> Tuple[List[GraphNode], List[GraphRelationship]]:
        """
        构建代码关系图
        
        返回节点和关系的列表
        """
        nodes: List[GraphNode] = []
        relationships: List[GraphRelationship] = []
        
        # 创建文件节点
        file_node = GraphNode(
            id=f"file:{file_path}",
            type="File",
            name=file_path.split("/")[-1],
            properties={
                "path": file_path,
                "language": language,
                "project_id": project_id,
                "lines_of_code": len(code.splitlines())
            }
        )
        nodes.append(file_node)
        
        # TODO: 集成AST解析器（tree-sitter, libcst等）
        # 这里简化处理，实际应该解析AST、CFG、数据流
        
        # 示例：检测函数调用
        if language == "python":
            import re
            function_pattern = r'def\s+(\w+)\s*\('
            functions = re.findall(function_pattern, code)
            
            for func_name in functions:
                func_node = GraphNode(
                    id=f"function:{file_path}:{func_name}",
                    type="Function",
                    name=func_name,
                    properties={
                        "file": file_path,
                        "language": language
                    }
                )
                nodes.append(func_node)
                
                # 文件包含函数
                relationships.append(GraphRelationship(
                    source_id=file_node.id,
                    target_id=func_node.id,
                    type=GraphRelationshipType.DEFINES
                ))
        
        return nodes, relationships
    
    async def _store_graph(
        self,
        nodes: List[GraphNode],
        relationships: List[GraphRelationship],
        project_id: str,
        commit_hash: Optional[str]
    ):
        """将图存储到Neo4j"""
        with self.driver.session(database=self.database) as session:
            # 创建节点
            for node in nodes:
                props = {**node.properties, "name": node.name, "project_id": project_id}
                if commit_hash:
                    props["commit_hash"] = commit_hash
                
                query = f"""
                MERGE (n:{node.type} {{id: $id}})
                SET n += $props
                """
                session.run(query, id=node.id, props=props)
            
            # 创建关系
            for rel in relationships:
                query = f"""
                MATCH (source {{id: $source_id}})
                MATCH (target {{id: $target_id}})
                MERGE (source)-[r:{rel.type.value}]->(target)
                SET r += $props
                """
                session.run(
                    query,
                    source_id=rel.source_id,
                    target_id=rel.target_id,
                    props={**rel.properties, "project_id": project_id}
                )
    
    async def _detect_architectural_violations(
        self,
        file_path: str,
        project_id: str
    ) -> List[ArchitecturalImpact]:
        """
        检测架构违规
        
        使用Neo4j查询检测：
        - 循环依赖
        - 高耦合
        - 层级违规
        - 依赖漂移
        """
        violations: List[ArchitecturalImpact] = []
        
        with self.driver.session(database=self.database) as session:
            # 1. 检测循环依赖
            circular_query = """
            MATCH path = (m:Module)-[:DEPENDS_ON*]->(m)
            WHERE ALL(r IN relationships(path) WHERE r.valid_to IS NULL)
            AND m.project_id = $project_id
            RETURN path, length(path) as cycle_length
            ORDER BY cycle_length DESC
            LIMIT 10
            """
            result = session.run(circular_query, project_id=project_id)
            
            for record in result:
                path = record["path"]
                cycle_length = record["cycle_length"]
                
                affected = [node["name"] for node in path.nodes]
                
                violations.append(ArchitecturalImpact(
                    violation_type=ArchitecturalViolationType.CIRCULAR_DEPENDENCY,
                    severity="high" if cycle_length <= 3 else "medium",
                    affected_components=affected,
                    impact_path=affected,
                    confidence=0.95,
                    explanation=f"检测到长度为{cycle_length}的循环依赖",
                    suggested_fix="考虑引入接口或依赖注入来打破循环"
                ))
            
            # 2. 检测高耦合
            coupling_query = """
            MATCH (m:Module)-[r:DEPENDS_ON]->()
            WHERE r.valid_to IS NULL AND m.project_id = $project_id
            WITH m, count(r) as fan_out
            WHERE fan_out > 10
            RETURN m.name as module, fan_out
            ORDER BY fan_out DESC
            """
            result = session.run(coupling_query, project_id=project_id)
            
            for record in result:
                module = record["module"]
                fan_out = record["fan_out"]
                
                violations.append(ArchitecturalImpact(
                    violation_type=ArchitecturalViolationType.HIGH_COUPLING,
                    severity="medium",
                    affected_components=[module],
                    impact_path=[module],
                    confidence=0.85,
                    explanation=f"模块{module}的扇出为{fan_out}，超过阈值10",
                    suggested_fix="考虑重构以减少依赖数量"
                ))
        
        return violations
    
    async def _analyze_dependency_risks(
        self,
        file_path: str,
        project_id: str
    ) -> List[Dict[str, Any]]:
        """分析依赖风险"""
        risks = []
        
        with self.driver.session(database=self.database) as session:
            # 查找不稳定的依赖
            query = """
            MATCH (m:Module)-[r:DEPENDS_ON]->(dep:Module)
            WHERE r.valid_to IS NULL 
            AND m.project_id = $project_id
            AND dep.stability IN ['beta', 'experimental']
            RETURN m.name as module, dep.name as dependency, dep.stability as stability
            """
            result = session.run(query, project_id=project_id)
            
            for record in result:
                risks.append({
                    "type": "unstable_dependency",
                    "module": record["module"],
                    "dependency": record["dependency"],
                    "stability": record["stability"],
                    "risk_level": "medium"
                })
        
        return risks
    
    async def _analyze_dataflow(
        self,
        file_path: str,
        project_id: str
    ) -> List[DataflowAnalysis]:
        """分析数据流"""
        dataflows = []
        
        with self.driver.session(database=self.database) as session:
            # 查找数据流路径
            query = """
            MATCH path = (source:Function)-[:DATAFLOW*]->(sink:Function)
            WHERE source.file = $file_path OR sink.file = $file_path
            RETURN source.name as source, sink.name as sink, 
                   [n IN nodes(path) | n.name] as path
            LIMIT 20
            """
            result = session.run(query, file_path=file_path)
            
            for record in result:
                dataflows.append(DataflowAnalysis(
                    source=record["source"],
                    sink=record["sink"],
                    data_path=record["path"],
                    sensitive_data_detected=False,  # TODO: 集成敏感数据检测
                    security_risk=None
                ))
        
        return dataflows
    
    async def _get_affected_modules(
        self,
        file_path: str,
        project_id: str
    ) -> Set[str]:
        """获取受影响的模块"""
        affected = set()
        
        with self.driver.session(database=self.database) as session:
            # 查找所有依赖此文件的模块
            query = """
            MATCH (f:File {path: $file_path, project_id: $project_id})
            MATCH (m:Module)-[:DEPENDS_ON*]->(f)
            RETURN DISTINCT m.name as module
            """
            result = session.run(query, file_path=file_path, project_id=project_id)
            
            for record in result:
                affected.add(record["module"])
        
        return affected
    
    async def _analyze_call_hierarchy(
        self,
        file_path: str,
        project_id: str
    ) -> Dict[str, Any]:
        """分析调用层次"""
        with self.driver.session(database=self.database) as session:
            query = """
            MATCH path = (root:Function)-[:CALLS*]->(leaf:Function)
            WHERE root.file = $file_path OR leaf.file = $file_path
            RETURN max(length(path)) as max_depth, 
                   count(DISTINCT path) as total_paths
            """
            result = session.run(query, file_path=file_path, project_id=project_id)
            record = result.single()
            
            return {
                "max_depth": record["max_depth"] if record else 0,
                "total_paths": record["total_paths"] if record else 0
            }
    
    async def _calculate_coupling_metrics(
        self,
        file_path: str,
        project_id: str
    ) -> Dict[str, float]:
        """计算耦合指标"""
        with self.driver.session(database=self.database) as session:
            # 扇入扇出
            query = """
            MATCH (m:Module)
            WHERE m.project_id = $project_id
            OPTIONAL MATCH (m)-[out:DEPENDS_ON]->()
            WHERE out.valid_to IS NULL
            OPTIONAL MATCH (m)<-[in:DEPENDS_ON]-()
            WHERE in.valid_to IS NULL
            RETURN m.name as module,
                   count(DISTINCT out) as fan_out,
                   count(DISTINCT in) as fan_in
            """
            result = session.run(query, project_id=project_id)
            
            metrics = {}
            for record in result:
                module = record["module"]
                fan_out = record["fan_out"] or 0
                fan_in = record["fan_in"] or 0
                
                metrics[f"{module}_fan_out"] = float(fan_out)
                metrics[f"{module}_fan_in"] = float(fan_in)
                metrics[f"{module}_total_coupling"] = float(fan_out + fan_in)
        
        return metrics
    
    async def _build_evidence_chain(
        self,
        violations: List[ArchitecturalImpact],
        dependency_risks: List[Dict[str, Any]],
        project_id: str
    ) -> List[Dict[str, Any]]:
        """
        构建证据链
        
        链接到类似的历史修复案例
        """
        evidence_chain = []
        
        with self.driver.session(database=self.database) as session:
            # 查找类似的历史问题
            for violation in violations:
                query = """
                MATCH (a:Analysis)-[:ANALYZES]->(v:Violation)
                WHERE v.type = $violation_type
                AND a.project_id = $project_id
                RETURN a.id as analysis_id, a.created_at as timestamp,
                       v.description as description
                ORDER BY a.created_at DESC
                LIMIT 5
                """
                result = session.run(
                    query,
                    violation_type=violation.violation_type.value,
                    project_id=project_id
                )
                
                similar_cases = []
                for record in result:
                    similar_cases.append({
                        "analysis_id": record["analysis_id"],
                        "timestamp": record["timestamp"],
                        "description": record["description"]
                    })
                
                evidence_chain.append({
                    "violation": violation.violation_type.value,
                    "similar_cases": similar_cases,
                    "confidence_boost": len(similar_cases) * 0.1
                })
        
        return evidence_chain
    
    def _calculate_confidence(
        self,
        violations: List[ArchitecturalImpact],
        dependency_risks: List[Dict[str, Any]],
        evidence_chain: List[Dict[str, Any]]
    ) -> float:
        """计算整体置信度"""
        if not violations and not dependency_risks:
            return 1.0
        
        # 基于证据链增强置信度
        base_confidence = 0.8
        evidence_boost = sum(e.get("confidence_boost", 0) for e in evidence_chain)
        
        confidence = min(1.0, base_confidence + evidence_boost)
        return confidence

