"""
Technical Debt Tracker (R-004 Mitigation)

Tracks and manages technical debt with:
- Code duplication detection
- Documentation coverage analysis
- Complexity metrics
- Debt prioritization
- Sprint planning integration

Targets:
- Code duplication < 10%
- Documentation coverage > 90%
- Cyclomatic complexity average < 10
"""
import ast
import hashlib
import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class DebtCategory(str, Enum):
    """Categories of technical debt."""
    DUPLICATION = "duplication"
    COMPLEXITY = "complexity"
    DOCUMENTATION = "documentation"
    TEST_COVERAGE = "test_coverage"
    DEPRECATED = "deprecated"
    SECURITY = "security"
    PERFORMANCE = "performance"
    ARCHITECTURE = "architecture"


class DebtPriority(str, Enum):
    """Priority levels for debt items."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DebtStatus(str, Enum):
    """Status of debt items."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    WONT_FIX = "wont_fix"


@dataclass
class DebtItem:
    """Represents a technical debt item."""
    id: str
    title: str
    description: str
    category: DebtCategory
    priority: DebtPriority
    status: DebtStatus = DebtStatus.OPEN
    file_path: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    estimated_hours: float = 0
    actual_hours: float = 0
    created_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_date: Optional[datetime] = None
    assignee: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "estimated_hours": self.estimated_hours,
            "actual_hours": self.actual_hours,
            "created_date": self.created_date.isoformat(),
            "resolved_date": self.resolved_date.isoformat() if self.resolved_date else None,
            "assignee": self.assignee,
            "tags": self.tags,
        }


@dataclass
class DuplicationBlock:
    """Represents a block of duplicated code."""
    hash: str
    files: List[Tuple[str, int, int]]  # (file_path, start_line, end_line)
    lines: int
    tokens: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hash": self.hash,
            "files": [{"path": f[0], "start": f[1], "end": f[2]} for f in self.files],
            "lines": self.lines,
            "tokens": self.tokens,
            "occurrences": len(self.files),
        }


@dataclass
class ComplexityMetrics:
    """Complexity metrics for a file/function."""
    file_path: str
    function_name: Optional[str]
    cyclomatic_complexity: int
    cognitive_complexity: int
    lines_of_code: int
    maintainability_index: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "function_name": self.function_name,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "cognitive_complexity": self.cognitive_complexity,
            "lines_of_code": self.lines_of_code,
            "maintainability_index": round(self.maintainability_index, 2),
        }


class DuplicationDetector:
    """
    Detects code duplication in the codebase.
    
    Uses token-based comparison for accurate detection.
    """
    
    def __init__(self, min_lines: int = 6, min_tokens: int = 50):
        self.min_lines = min_lines
        self.min_tokens = min_tokens
    
    def analyze(self, source_dirs: List[str]) -> Tuple[float, List[DuplicationBlock]]:
        """
        Analyze code duplication.
        
        Returns:
            Tuple of (duplication_percentage, list of duplication blocks)
        """
        # Collect all code blocks
        blocks: Dict[str, List[Tuple[str, int, int, str]]] = defaultdict(list)
        total_lines = 0
        duplicated_lines = 0
        
        for source_dir in source_dirs:
            for py_file in Path(source_dir).rglob("*.py"):
                if "__pycache__" in str(py_file) or "test" in py_file.name.lower():
                    continue
                
                try:
                    file_blocks, file_lines = self._extract_blocks(py_file)
                    total_lines += file_lines
                    
                    for block_hash, start, end, content in file_blocks:
                        blocks[block_hash].append((str(py_file), start, end, content))
                except Exception as e:
                    logger.warning(f"Failed to analyze {py_file}: {e}")
        
        # Find duplications
        duplications = []
        counted_lines: Set[Tuple[str, int]] = set()
        
        for block_hash, occurrences in blocks.items():
            if len(occurrences) > 1:
                # This is a duplication
                lines = occurrences[0][2] - occurrences[0][1]
                tokens = len(occurrences[0][3].split())
                
                dup = DuplicationBlock(
                    hash=block_hash[:16],
                    files=[(f, s, e) for f, s, e, _ in occurrences],
                    lines=lines,
                    tokens=tokens,
                )
                duplications.append(dup)
                
                # Count duplicated lines (excluding first occurrence)
                for file_path, start, end, _ in occurrences[1:]:
                    for line in range(start, end + 1):
                        key = (file_path, line)
                        if key not in counted_lines:
                            counted_lines.add(key)
                            duplicated_lines += 1
        
        duplication_percentage = (duplicated_lines / total_lines * 100) if total_lines > 0 else 0
        
        return duplication_percentage, duplications
    
    def _extract_blocks(self, file_path: Path) -> Tuple[List[Tuple[str, int, int, str]], int]:
        """Extract code blocks from a file."""
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        
        blocks = []
        total_lines = len([l for l in lines if l.strip() and not l.strip().startswith("#")])
        
        # Extract blocks of consecutive non-empty lines
        for window_size in range(self.min_lines, min(50, len(lines))):
            for start in range(len(lines) - window_size + 1):
                block_lines = lines[start:start + window_size]
                
                # Skip if mostly comments or whitespace
                code_lines = [l for l in block_lines if l.strip() and not l.strip().startswith("#")]
                if len(code_lines) < self.min_lines:
                    continue
                
                # Normalize and hash
                normalized = self._normalize_code("".join(code_lines))
                if len(normalized.split()) >= self.min_tokens:
                    block_hash = hashlib.sha256(normalized.encode()).hexdigest()
                    blocks.append((block_hash, start + 1, start + window_size, normalized))
        
        return blocks, total_lines
    
    def _normalize_code(self, code: str) -> str:
        """Normalize code for comparison."""
        # Remove extra whitespace
        code = re.sub(r'\s+', ' ', code)
        # Remove string literals
        code = re.sub(r'"[^"]*"', '""', code)
        code = re.sub(r"'[^']*'", "''", code)
        return code.strip()


class ComplexityAnalyzer:
    """
    Analyzes code complexity metrics.
    """
    
    def __init__(self, max_complexity: int = 10):
        self.max_complexity = max_complexity
    
    def analyze(self, source_dirs: List[str]) -> Tuple[float, List[ComplexityMetrics]]:
        """
        Analyze code complexity.
        
        Returns:
            Tuple of (average_complexity, list of complexity metrics)
        """
        metrics = []
        total_complexity = 0
        count = 0
        
        for source_dir in source_dirs:
            for py_file in Path(source_dir).rglob("*.py"):
                if "__pycache__" in str(py_file):
                    continue
                
                try:
                    file_metrics = self._analyze_file(py_file)
                    metrics.extend(file_metrics)
                    
                    for m in file_metrics:
                        total_complexity += m.cyclomatic_complexity
                        count += 1
                except Exception as e:
                    logger.warning(f"Failed to analyze {py_file}: {e}")
        
        avg_complexity = total_complexity / count if count > 0 else 0
        
        return avg_complexity, metrics
    
    def _analyze_file(self, file_path: Path) -> List[ComplexityMetrics]:
        """Analyze complexity of a single file."""
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            source = f.read()
        
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        
        metrics = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cc = self._calculate_cyclomatic_complexity(node)
                cog = self._calculate_cognitive_complexity(node)
                loc = node.end_lineno - node.lineno + 1 if hasattr(node, "end_lineno") else 10
                mi = self._calculate_maintainability_index(cc, loc)
                
                metrics.append(ComplexityMetrics(
                    file_path=str(file_path),
                    function_name=node.name,
                    cyclomatic_complexity=cc,
                    cognitive_complexity=cog,
                    lines_of_code=loc,
                    maintainability_index=mi,
                ))
        
        return metrics
    
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
            elif isinstance(child, ast.Assert):
                complexity += 1
        
        return complexity
    
    def _calculate_cognitive_complexity(self, node: ast.AST) -> int:
        """Calculate cognitive complexity (simplified)."""
        # Simplified cognitive complexity
        return self._calculate_cyclomatic_complexity(node) + self._count_nesting_depth(node)
    
    def _count_nesting_depth(self, node: ast.AST, depth: int = 0) -> int:
        """Count maximum nesting depth."""
        max_depth = depth
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                child_depth = self._count_nesting_depth(child, depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._count_nesting_depth(child, depth)
                max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def _calculate_maintainability_index(self, cc: int, loc: int) -> float:
        """Calculate maintainability index (0-100)."""
        import math
        
        # Simplified MI formula
        # MI = 171 - 5.2 * ln(Halstead Volume) - 0.23 * CC - 16.2 * ln(LOC)
        # Using simplified version
        if loc <= 0:
            return 100.0
        
        mi = 171 - 0.23 * cc - 16.2 * math.log(loc)
        mi = max(0, min(100, mi))
        
        return mi


class DocumentationAnalyzer:
    """
    Analyzes documentation coverage.
    """
    
    def analyze(self, source_dirs: List[str]) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Analyze documentation coverage.
        
        Returns:
            Tuple of (coverage_percentage, list of undocumented items)
        """
        total_items = 0
        documented_items = 0
        undocumented = []
        
        for source_dir in source_dirs:
            for py_file in Path(source_dir).rglob("*.py"):
                if "__pycache__" in str(py_file):
                    continue
                
                try:
                    with open(py_file, encoding="utf-8", errors="ignore") as f:
                        source = f.read()
                    
                    tree = ast.parse(source)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            total_items += 1
                            
                            docstring = ast.get_docstring(node)
                            if docstring and len(docstring) > 10:
                                documented_items += 1
                            else:
                                undocumented.append({
                                    "file": str(py_file),
                                    "name": node.name,
                                    "type": "class" if isinstance(node, ast.ClassDef) else "function",
                                    "line": node.lineno,
                                })
                                
                except Exception as e:
                    logger.warning(f"Failed to analyze {py_file}: {e}")
        
        coverage = (documented_items / total_items * 100) if total_items > 0 else 100
        
        return coverage, undocumented


class TechnicalDebtTracker:
    """
    Main technical debt tracking and management system.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self.debt_items: Dict[str, DebtItem] = {}
        self.duplication_detector = DuplicationDetector()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.documentation_analyzer = DocumentationAnalyzer()
    
    def analyze_codebase(self, source_dirs: List[str]) -> Dict[str, Any]:
        """
        Perform comprehensive technical debt analysis.
        
        Returns:
            Analysis report with metrics and identified debt items
        """
        logger.info("Starting technical debt analysis...")
        
        # Duplication analysis
        dup_percentage, duplications = self.duplication_detector.analyze(source_dirs)
        
        # Complexity analysis
        avg_complexity, complexity_metrics = self.complexity_analyzer.analyze(source_dirs)
        
        # Documentation analysis
        doc_coverage, undocumented = self.documentation_analyzer.analyze(source_dirs)
        
        # Create debt items from findings
        self._create_debt_items_from_analysis(duplications, complexity_metrics, undocumented)
        
        # Calculate overall health score
        health_score = self._calculate_health_score(dup_percentage, avg_complexity, doc_coverage)
        
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "health_score": round(health_score, 1),
                "duplication_percentage": round(dup_percentage, 2),
                "duplication_target": 10.0,
                "duplication_passed": dup_percentage < 10,
                "average_complexity": round(avg_complexity, 2),
                "complexity_target": 10.0,
                "complexity_passed": avg_complexity < 10,
                "documentation_coverage": round(doc_coverage, 2),
                "documentation_target": 90.0,
                "documentation_passed": doc_coverage >= 90,
                "total_debt_items": len(self.debt_items),
            },
            "duplication": {
                "percentage": round(dup_percentage, 2),
                "blocks": [d.to_dict() for d in duplications[:20]],
            },
            "complexity": {
                "average": round(avg_complexity, 2),
                "high_complexity_functions": [
                    m.to_dict() for m in complexity_metrics 
                    if m.cyclomatic_complexity > 10
                ][:20],
            },
            "documentation": {
                "coverage": round(doc_coverage, 2),
                "undocumented": undocumented[:50],
            },
            "debt_items": [d.to_dict() for d in self.debt_items.values()],
        }
        
        logger.info(f"Analysis complete. Health score: {health_score:.1f}/100")
        
        return report
    
    def _create_debt_items_from_analysis(
        self,
        duplications: List[DuplicationBlock],
        complexity_metrics: List[ComplexityMetrics],
        undocumented: List[Dict[str, Any]]
    ):
        """Create debt items from analysis findings."""
        # Duplication debt items
        for i, dup in enumerate(duplications[:10]):  # Top 10
            item = DebtItem(
                id=f"DUP-{i+1:04d}",
                title=f"Code duplication ({dup.lines} lines, {dup.occurrences} occurrences)",
                description=f"Duplicated code found in {dup.occurrences} locations",
                category=DebtCategory.DUPLICATION,
                priority=DebtPriority.MEDIUM if dup.lines < 20 else DebtPriority.HIGH,
                file_path=dup.files[0][0] if dup.files else None,
                line_start=dup.files[0][1] if dup.files else None,
                line_end=dup.files[0][2] if dup.files else None,
                estimated_hours=dup.lines * 0.1,  # Rough estimate
                tags=["duplication", "refactoring"],
            )
            self.debt_items[item.id] = item
        
        # Complexity debt items
        high_complexity = [m for m in complexity_metrics if m.cyclomatic_complexity > 15]
        for i, metric in enumerate(high_complexity[:10]):
            item = DebtItem(
                id=f"CMP-{i+1:04d}",
                title=f"High complexity: {metric.function_name} (CC={metric.cyclomatic_complexity})",
                description=f"Function has cyclomatic complexity of {metric.cyclomatic_complexity}, exceeds threshold of 10",
                category=DebtCategory.COMPLEXITY,
                priority=DebtPriority.HIGH if metric.cyclomatic_complexity > 20 else DebtPriority.MEDIUM,
                file_path=metric.file_path,
                estimated_hours=metric.cyclomatic_complexity * 0.5,
                tags=["complexity", "refactoring"],
            )
            self.debt_items[item.id] = item
        
        # Documentation debt items
        if len(undocumented) > 10:
            item = DebtItem(
                id="DOC-0001",
                title=f"Missing documentation ({len(undocumented)} items)",
                description=f"{len(undocumented)} functions/classes lack proper documentation",
                category=DebtCategory.DOCUMENTATION,
                priority=DebtPriority.MEDIUM,
                estimated_hours=len(undocumented) * 0.25,
                tags=["documentation"],
            )
            self.debt_items[item.id] = item
    
    def _calculate_health_score(
        self,
        dup_percentage: float,
        avg_complexity: float,
        doc_coverage: float
    ) -> float:
        """Calculate overall codebase health score (0-100)."""
        # Duplication score (target < 10%)
        dup_score = max(0, 100 - (dup_percentage * 5))  # Penalize above 0%
        
        # Complexity score (target < 10)
        complexity_score = max(0, 100 - (avg_complexity * 5))
        
        # Documentation score
        doc_score = doc_coverage
        
        # Weighted average
        health_score = (dup_score * 0.3) + (complexity_score * 0.3) + (doc_score * 0.4)
        
        return min(100, max(0, health_score))
    
    def add_debt_item(self, item: DebtItem) -> str:
        """Add a manual debt item."""
        self.debt_items[item.id] = item
        return item.id
    
    def update_debt_item(self, item_id: str, **updates) -> Optional[DebtItem]:
        """Update a debt item."""
        item = self.debt_items.get(item_id)
        if not item:
            return None
        
        for key, value in updates.items():
            if hasattr(item, key):
                setattr(item, key, value)
        
        if updates.get("status") == DebtStatus.RESOLVED:
            item.resolved_date = datetime.now(timezone.utc)
        
        return item
    
    def get_sprint_backlog(self, max_hours: float = 40) -> List[DebtItem]:
        """Get prioritized debt items for a sprint."""
        open_items = [
            item for item in self.debt_items.values()
            if item.status == DebtStatus.OPEN
        ]
        
        # Sort by priority and estimated hours
        priority_order = {
            DebtPriority.CRITICAL: 0,
            DebtPriority.HIGH: 1,
            DebtPriority.MEDIUM: 2,
            DebtPriority.LOW: 3,
        }
        
        sorted_items = sorted(open_items, key=lambda x: (priority_order[x.priority], x.estimated_hours))
        
        # Select items within budget
        backlog = []
        total_hours = 0
        
        for item in sorted_items:
            if total_hours + item.estimated_hours <= max_hours:
                backlog.append(item)
                total_hours += item.estimated_hours
        
        return backlog
    
    def save(self, report: Dict[str, Any], output_path: str):
        """Save analysis report."""
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved: {output_path}")


# CLI entry point
def main():
    """CLI entry point for technical debt analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze technical debt")
    parser.add_argument("--source", "-s", nargs="+", default=["backend", "ai_core"], help="Source directories")
    parser.add_argument("--output", "-o", default="technical_debt_report.json", help="Output report path")
    
    args = parser.parse_args()
    
    tracker = TechnicalDebtTracker()
    report = tracker.analyze_codebase(args.source)
    tracker.save(report, args.output)
    
    print(f"\n{'='*50}")
    print(f"Technical Debt Analysis Complete")
    print(f"{'='*50}")
    print(f"Health Score: {report['summary']['health_score']}/100")
    print(f"Duplication: {report['summary']['duplication_percentage']:.1f}% ({'✅' if report['summary']['duplication_passed'] else '❌'} target: <10%)")
    print(f"Complexity: {report['summary']['average_complexity']:.1f} ({'✅' if report['summary']['complexity_passed'] else '❌'} target: <10)")
    print(f"Documentation: {report['summary']['documentation_coverage']:.1f}% ({'✅' if report['summary']['documentation_passed'] else '❌'} target: >90%)")
    print(f"Debt Items: {report['summary']['total_debt_items']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
