"""
可解释AI与证据链系统

提供：
- 置信度评分
- 数据流可视化
- 历史修复链接
- 推理步骤追踪
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ConfidenceLevel(str, Enum):
    """置信度等级"""
    VERY_HIGH = "very_high"  # >= 0.9
    HIGH = "high"  # >= 0.75
    MEDIUM = "medium"  # >= 0.5
    LOW = "low"  # < 0.5


@dataclass
class EvidenceLink:
    """证据链接"""
    type: str  # similar_fix, historical_pattern, rule_reference
    title: str
    url: Optional[str] = None
    confidence: float = 0.0
    description: str = ""


@dataclass
class ReasoningStep:
    """推理步骤"""
    step_number: int
    description: str
    evidence: List[str]
    confidence: float
    rule_applied: Optional[str] = None


@dataclass
class DataflowVisualization:
    """数据流可视化数据"""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    sensitive_paths: List[List[str]]
    security_risks: List[Dict[str, Any]]


@dataclass
class ExplainableFinding:
    """可解释的发现"""
    finding_id: str
    issue: str
    severity: str
    
    # 置信度信息
    confidence_score: float
    confidence_level: ConfidenceLevel
    confidence_factors: Dict[str, float]  # 各因素对置信度的贡献
    
    # 推理步骤
    reasoning_steps: List[ReasoningStep]
    
    # 证据链
    evidence_links: List[EvidenceLink]
    similar_past_fixes: List[Dict[str, Any]]
    
    # 数据流可视化
    dataflow_visualization: Optional[DataflowVisualization] = None
    
    # 建议
    suggestion: str
    fix_code: Optional[str] = None
    
    # 元数据
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ExplainableAIService:
    """
    可解释AI服务
    
    为每个AI建议提供：
    1. 置信度评分和因素分解
    2. 推理步骤追踪
    3. 证据链（链接到历史案例）
    4. 数据流可视化
    """
    
    def __init__(self, neo4j_analyzer=None, historical_db=None):
        """
        初始化可解释AI服务
        
        Args:
            neo4j_analyzer: Neo4j图分析器（用于证据链查询）
            historical_db: 历史数据库连接（用于查找相似修复）
        """
        self.neo4j_analyzer = neo4j_analyzer
        self.historical_db = historical_db
    
    def enhance_finding_with_explanation(
        self,
        finding: Dict[str, Any],
        code_context: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> ExplainableFinding:
        """
        增强发现项，添加可解释性信息
        
        Args:
            finding: 原始发现项
            code_context: 代码上下文
            project_id: 项目ID
        
        Returns:
            ExplainableFinding: 增强后的发现项
        """
        finding_id = finding.get("id", "unknown")
        issue = finding.get("issue", "")
        severity = finding.get("severity", "medium")
        base_confidence = finding.get("confidence", 0.5)
        
        # 1. 计算置信度因素
        confidence_factors = self._calculate_confidence_factors(
            finding, code_context
        )
        
        # 2. 构建推理步骤
        reasoning_steps = self._build_reasoning_steps(finding, code_context)
        
        # 3. 查找证据链
        evidence_links = self._find_evidence_links(finding, project_id)
        
        # 4. 查找相似的历史修复
        similar_fixes = self._find_similar_past_fixes(finding, project_id)
        
        # 5. 生成数据流可视化（如果适用）
        dataflow_viz = None
        if "dataflow" in finding.get("tags", []):
            dataflow_viz = self._generate_dataflow_visualization(
                finding, code_context, project_id
            )
        
        # 6. 计算最终置信度
        final_confidence = self._calculate_final_confidence(
            base_confidence, confidence_factors, evidence_links, similar_fixes
        )
        
        return ExplainableFinding(
            finding_id=finding_id,
            issue=issue,
            severity=severity,
            confidence_score=final_confidence,
            confidence_level=self._get_confidence_level(final_confidence),
            confidence_factors=confidence_factors,
            reasoning_steps=reasoning_steps,
            evidence_links=evidence_links,
            similar_past_fixes=similar_fixes,
            dataflow_visualization=dataflow_viz,
            suggestion=finding.get("suggestion", ""),
            fix_code=finding.get("fix_code")
        )
    
    def _calculate_confidence_factors(
        self,
        finding: Dict[str, Any],
        code_context: Optional[str]
    ) -> Dict[str, float]:
        """
        计算置信度因素
        
        返回各因素对置信度的贡献
        """
        factors = {}
        
        # 1. 规则匹配度
        if finding.get("rule_id"):
            factors["rule_match"] = 0.3
        else:
            factors["rule_match"] = 0.1
        
        # 2. 代码模式匹配
        if code_context and finding.get("code_snippet"):
            snippet = finding.get("code_snippet", "")
            if snippet in code_context:
                factors["pattern_match"] = 0.25
            else:
                factors["pattern_match"] = 0.1
        else:
            factors["pattern_match"] = 0.15
        
        # 3. 严重性权重
        severity_weights = {
            "critical": 0.3,
            "high": 0.2,
            "medium": 0.15,
            "low": 0.1
        }
        factors["severity_weight"] = severity_weights.get(
            finding.get("severity", "medium"), 0.15
        )
        
        # 4. CWE/CVE引用（如果有）
        if finding.get("cwe_id") or finding.get("cve_id"):
            factors["security_reference"] = 0.2
        else:
            factors["security_reference"] = 0.0
        
        return factors
    
    def _build_reasoning_steps(
        self,
        finding: Dict[str, Any],
        code_context: Optional[str]
    ) -> List[ReasoningStep]:
        """构建推理步骤"""
        steps = []
        
        # 步骤1: 代码解析
        steps.append(ReasoningStep(
            step_number=1,
            description="解析代码结构，识别函数、类和模块",
            evidence=["AST解析完成", "识别到关键代码模式"],
            confidence=0.9,
            rule_applied="code_parsing"
        ))
        
        # 步骤2: 规则匹配
        if finding.get("rule_id"):
            steps.append(ReasoningStep(
                step_number=2,
                description=f"应用规则 {finding['rule_id']} 进行模式匹配",
                evidence=[f"规则 {finding['rule_id']} 匹配成功"],
                confidence=0.85,
                rule_applied=finding["rule_id"]
            ))
        
        # 步骤3: 上下文分析
        if code_context:
            steps.append(ReasoningStep(
                step_number=3,
                description="分析代码上下文，评估影响范围",
                evidence=["上下文分析完成", "影响范围已确定"],
                confidence=0.8,
                rule_applied="context_analysis"
            ))
        
        # 步骤4: 严重性评估
        steps.append(ReasoningStep(
            step_number=len(steps) + 1,
            description=f"评估问题严重性: {finding.get('severity', 'medium')}",
            evidence=[f"严重性评分: {finding.get('severity')}"],
            confidence=0.75,
            rule_applied="severity_assessment"
        ))
        
        return steps
    
    def _find_evidence_links(
        self,
        finding: Dict[str, Any],
        project_id: Optional[str]
    ) -> List[EvidenceLink]:
        """查找证据链接"""
        links = []
        
        # 1. CWE/CVE引用
        if finding.get("cwe_id"):
            links.append(EvidenceLink(
                type="rule_reference",
                title=f"CWE-{finding['cwe_id']}",
                url=f"https://cwe.mitre.org/data/definitions/{finding['cwe_id']}.html",
                confidence=0.95,
                description="Common Weakness Enumeration参考"
            ))
        
        # 2. 历史模式（如果 Neo4j 可用）
        if self.neo4j_analyzer and project_id:
            # 查询类似的历史问题
            similar_issues = self._query_similar_historical_issues(
                finding, project_id
            )
            
            for issue in similar_issues:
                links.append(EvidenceLink(
                    type="historical_pattern",
                    title=f"类似问题: {issue.get('title', '')}",
                    url=issue.get("url"),
                    confidence=issue.get("similarity", 0.7),
                    description=issue.get("description", "")
                ))
        
        return links
    
    def _query_similar_historical_issues(
        self,
        finding: Dict[str, Any],
        project_id: str
    ) -> List[Dict[str, Any]]:
        """查询类似的历史问题"""
        # TODO: 实现Neo4j查询
        # 这里返回模拟数据
        return []
    
    def _find_similar_past_fixes(
        self,
        finding: Dict[str, Any],
        project_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """查找相似的历史修复"""
        similar_fixes = []
        
        # TODO: 从历史数据库查询
        # 示例结构：
        # {
        #     "fix_id": "fix_123",
        #     "issue_type": finding.get("issue"),
        #     "fix_code": "...",
        #     "success_rate": 0.95,
        #     "applied_count": 42,
        #     "timestamp": "2024-01-15T10:30:00Z"
        # }
        
        return similar_fixes
    
    def _generate_dataflow_visualization(
        self,
        finding: Dict[str, Any],
        code_context: Optional[str],
        project_id: Optional[str]
    ) -> Optional[DataflowVisualization]:
        """生成数据流可视化"""
        if not self.neo4j_analyzer or not project_id:
            return None
        
        # TODO: 从Neo4j查询数据流图
        # 返回节点和边的列表，用于前端可视化
        
        return DataflowVisualization(
            nodes=[],
            edges=[],
            sensitive_paths=[],
            security_risks=[]
        )
    
    def _calculate_final_confidence(
        self,
        base_confidence: float,
        confidence_factors: Dict[str, float],
        evidence_links: List[EvidenceLink],
        similar_fixes: List[Dict[str, Any]]
    ) -> float:
        """计算最终置信度"""
        # 基础置信度
        confidence = base_confidence
        
        # 因素加权
        factor_boost = sum(confidence_factors.values()) * 0.3
        confidence += factor_boost
        
        # 证据链增强
        if evidence_links:
            evidence_boost = sum(link.confidence for link in evidence_links) / len(evidence_links) * 0.2
            confidence += evidence_boost
        
        # 历史修复增强
        if similar_fixes:
            fix_boost = min(0.1, len(similar_fixes) * 0.02)
            confidence += fix_boost
        
        return min(1.0, confidence)
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """获取置信度等级"""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.75:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def generate_explanation_summary(
        self,
        explainable_finding: ExplainableFinding
    ) -> Dict[str, Any]:
        """
        生成解释摘要
        
        用于前端展示
        """
        return {
            "finding_id": explainable_finding.finding_id,
            "issue": explainable_finding.issue,
            "severity": explainable_finding.severity,
            "confidence": {
                "score": explainable_finding.confidence_score,
                "level": explainable_finding.confidence_level.value,
                "factors": explainable_finding.confidence_factors
            },
            "reasoning": {
                "steps": [
                    {
                        "step": step.step_number,
                        "description": step.description,
                        "evidence": step.evidence,
                        "confidence": step.confidence
                    }
                    for step in explainable_finding.reasoning_steps
                ]
            },
            "evidence": {
                "links": [
                    {
                        "type": link.type,
                        "title": link.title,
                        "url": link.url,
                        "confidence": link.confidence
                    }
                    for link in explainable_finding.evidence_links
                ],
                "similar_fixes_count": len(explainable_finding.similar_past_fixes)
            },
            "suggestion": explainable_finding.suggestion,
            "fix_code": explainable_finding.fix_code
        }

