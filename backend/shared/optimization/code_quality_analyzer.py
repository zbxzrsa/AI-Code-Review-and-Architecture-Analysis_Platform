"""
Code Quality Analyzer

分析代码质量，识别重复代码、冗余逻辑和低效实现。
提供代码质量报告和改进建议。
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import ast
import hashlib
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CodeIssue:
    """代码问题"""
    issue_id: str
    file_path: str
    line_number: int
    issue_type: str  # duplicate, complexity, maintainability, etc.
    severity: str  # critical, high, medium, low
    description: str
    suggestion: str
    code_snippet: Optional[str] = None


@dataclass
class DuplicateCode:
    """重复代码块"""
    block_id: str
    files: List[str]
    line_ranges: List[Tuple[int, int]]
    similarity: float
    code_hash: str
    suggestion: str


@dataclass
class CodeQualityReport:
    """代码质量报告"""
    report_id: str
    timestamp: datetime
    total_files: int
    total_lines: int
    
    # 问题统计
    issues: List[CodeIssue]
    duplicate_blocks: List[DuplicateCode]
    
    # 指标
    code_duplication_rate: float
    average_complexity: float
    maintainability_index: float
    
    # 建议
    recommendations: List[str]


class CodeQualityAnalyzer:
    """
    代码质量分析器
    
    功能：
    1. 检测重复代码
    2. 分析代码复杂度
    3. 评估可维护性
    4. 识别冗余逻辑
    5. 生成改进建议
    """
    
    def __init__(self, project_root: str):
        """
        初始化分析器
        
        Args:
            project_root: 项目根目录
        """
        self.project_root = Path(project_root)
        self.issues: List[CodeIssue] = []
        self.duplicate_blocks: List[DuplicateCode] = []
        
        # 代码块哈希表（用于检测重复）
        self.code_blocks: Dict[str, List[Tuple[str, int, int]]] = defaultdict(list)
    
    def analyze_project(self) -> CodeQualityReport:
        """
        分析整个项目
        
        Returns:
            CodeQualityReport: 代码质量报告
        """
        logger.info("Starting code quality analysis")
        
        # 1. 收集所有Python文件
        python_files = list(self.project_root.rglob("*.py"))
        
        # 2. 分析每个文件
        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue
            
            try:
                self._analyze_file(file_path)
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
        
        # 3. 检测重复代码
        self._detect_duplicate_code()
        
        # 4. 计算指标
        metrics = self._calculate_metrics()
        
        # 5. 生成建议
        recommendations = self._generate_recommendations()
        
        return CodeQualityReport(
            report_id=f"report_{datetime.now().isoformat()}",
            timestamp=datetime.now(),
            total_files=len(python_files),
            total_lines=sum(self._count_lines(f) for f in python_files),
            issues=self.issues,
            duplicate_blocks=self.duplicate_blocks,
            code_duplication_rate=metrics["duplication_rate"],
            average_complexity=metrics["avg_complexity"],
            maintainability_index=metrics["maintainability"],
            recommendations=recommendations
        )
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """判断是否跳过文件"""
        skip_patterns = [
            "__pycache__",
            ".venv",
            "venv",
            "node_modules",
            ".git",
            "migrations",
            "tests"  # 可以配置是否分析测试文件
        ]
        
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def _analyze_file(self, file_path: Path) -> None:
        """分析单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content, filename=str(file_path))
            
            # 分析AST
            analyzer = FileAnalyzer(str(file_path), content, tree)
            file_issues = analyzer.analyze()
            self.issues.extend(file_issues)
            
            # 提取代码块用于重复检测
            self._extract_code_blocks(str(file_path), content)
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            self.issues.append(CodeIssue(
                issue_id=f"syntax_{hash(str(file_path))}",
                file_path=str(file_path),
                line_number=e.lineno or 0,
                issue_type="syntax_error",
                severity="high",
                description=f"Syntax error: {e.msg}",
                suggestion="Fix syntax error"
            ))
    
    def _extract_code_blocks(self, file_path: str, content: str) -> None:
        """提取代码块用于重复检测"""
        lines = content.split('\n')
        
        # 提取函数和方法
        for i, line in enumerate(lines, 1):
            # 简单检测：以def开头的函数
            if line.strip().startswith('def '):
                # 提取函数体（简化版，实际应该解析AST）
                func_start = i
                indent = len(line) - len(line.lstrip())
                
                # 找到函数结束（下一个相同或更少缩进的非空行）
                func_end = func_start
                for j in range(i, min(i + 50, len(lines))):  # 限制函数长度
                    if j + 1 < len(lines):
                        next_line = lines[j]
                        if next_line.strip() and len(next_line) - len(next_line.lstrip()) <= indent:
                            if not next_line.strip().startswith('#'):
                                break
                    func_end = j + 1
                
                if func_end > func_start:
                    func_code = '\n'.join(lines[func_start-1:func_end])
                    code_hash = hashlib.md5(func_code.encode()).hexdigest()
                    
                    # 标准化代码（移除空白和注释）
                    normalized = self._normalize_code(func_code)
                    normalized_hash = hashlib.md5(normalized.encode()).hexdigest()
                    
                    self.code_blocks[normalized_hash].append((file_path, func_start, func_end))
    
    def _normalize_code(self, code: str) -> str:
        """标准化代码（用于比较）"""
        lines = code.split('\n')
        normalized = []
        
        for line in lines:
            # 移除注释
            if '#' in line:
                line = line[:line.index('#')]
            
            # 移除多余空白
            line = line.strip()
            
            if line:
                normalized.append(line)
        
        return '\n'.join(normalized)
    
    def _detect_duplicate_code(self) -> None:
        """检测重复代码"""
        for code_hash, occurrences in self.code_blocks.items():
            if len(occurrences) > 1:
                # 发现重复
                files = [occ[0] for occ in occurrences]
                line_ranges = [(occ[1], occ[2]) for occ in occurrences]
                
                # 计算相似度（简化：相同哈希=100%相似）
                similarity = 1.0
                
                self.duplicate_blocks.append(DuplicateCode(
                    block_id=f"dup_{code_hash[:8]}",
                    files=files,
                    line_ranges=line_ranges,
                    similarity=similarity,
                    code_hash=code_hash,
                    suggestion=f"提取到共享模块，在 {len(files)} 个文件中重复"
                ))
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """计算代码质量指标"""
        total_issues = len(self.issues)
        total_duplicates = len(self.duplicate_blocks)
        
        # 重复率（简化计算）
        duplication_rate = (total_duplicates / max(1, total_issues)) * 100
        
        # 平均复杂度（需要实际计算，这里简化）
        avg_complexity = 5.0  # 占位符
        
        # 可维护性指数（0-100，越高越好）
        maintainability = max(0, 100 - (total_issues * 2) - (total_duplicates * 5))
        
        return {
            "duplication_rate": min(100, duplication_rate),
            "avg_complexity": avg_complexity,
            "maintainability": maintainability
        }
    
    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if self.duplicate_blocks:
            recommendations.append(
                f"发现 {len(self.duplicate_blocks)} 处重复代码，建议提取到共享模块"
            )
        
        critical_issues = [i for i in self.issues if i.severity == "critical"]
        if critical_issues:
            recommendations.append(
                f"发现 {len(critical_issues)} 个严重问题，需要立即修复"
            )
        
        high_complexity_files = [i for i in self.issues if i.issue_type == "high_complexity"]
        if high_complexity_files:
            recommendations.append(
                f"发现 {len(high_complexity_files)} 个高复杂度文件，建议重构"
            )
        
        return recommendations
    
    def _count_lines(self, file_path: Path) -> int:
        """统计文件行数"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return len(f.readlines())
        except Exception:
            return 0


class FileAnalyzer:
    """文件分析器"""
    
    def __init__(self, file_path: str, content: str, tree: ast.AST):
        self.file_path = file_path
        self.content = content
        self.tree = tree
        self.issues: List[CodeIssue] = []
    
    def analyze(self) -> List[CodeIssue]:
        """分析文件"""
        # 1. 检查函数长度
        self._check_function_length()
        
        # 2. 检查复杂度
        self._check_complexity()
        
        # 3. 检查命名规范
        self._check_naming_conventions()
        
        # 4. 检查异常处理
        self._check_exception_handling()
        
        return self.issues
    
    def _check_function_length(self) -> None:
        """检查函数长度"""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                
                if func_lines > 100:
                    self.issues.append(CodeIssue(
                        issue_id=f"long_func_{node.lineno}",
                        file_path=self.file_path,
                        line_number=node.lineno,
                        issue_type="long_function",
                        severity="medium",
                        description=f"函数 {node.name} 过长 ({func_lines} 行)",
                        suggestion="将函数拆分为更小的函数"
                    ))
    
    def _check_complexity(self) -> None:
        """检查代码复杂度"""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_cyclomatic_complexity(node)
                
                if complexity > 15:
                    self.issues.append(CodeIssue(
                        issue_id=f"complex_{node.lineno}",
                        file_path=self.file_path,
                        line_number=node.lineno,
                        issue_type="high_complexity",
                        severity="high",
                        description=f"函数 {node.name} 圈复杂度为 {complexity}（建议 < 10）",
                        suggestion="简化控制流，提取子函数"
                    ))
    
    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """计算圈复杂度（简化版）"""
        complexity = 1  # 基础复杂度
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _check_naming_conventions(self) -> None:
        """检查命名规范"""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                if not node.name.islower() and '_' not in node.name:
                    # 可能是驼峰命名，不符合PEP 8
                    if not node.name.startswith('_'):
                        self.issues.append(CodeIssue(
                            issue_id=f"naming_{node.lineno}",
                            file_path=self.file_path,
                            line_number=node.lineno,
                            issue_type="naming_convention",
                            severity="low",
                            description=f"函数名 {node.name} 不符合PEP 8规范",
                            suggestion="使用snake_case命名"
                        ))
    
    def _check_exception_handling(self) -> None:
        """检查异常处理"""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    # 裸except
                    self.issues.append(CodeIssue(
                        issue_id=f"bare_except_{node.lineno}",
                        file_path=self.file_path,
                        line_number=node.lineno,
                        issue_type="bare_except",
                        severity="high",
                        description="使用了裸except，会捕获所有异常",
                        suggestion="指定具体的异常类型"
                    ))
                elif isinstance(node.type, ast.Name) and node.type.id == "Exception":
                    # 过于宽泛的Exception
                    self.issues.append(CodeIssue(
                        issue_id=f"broad_except_{node.lineno}",
                        file_path=self.file_path,
                        line_number=node.lineno,
                        issue_type="broad_except",
                        severity="medium",
                        description="捕获Exception过于宽泛",
                        suggestion="捕获更具体的异常类型"
                    ))

