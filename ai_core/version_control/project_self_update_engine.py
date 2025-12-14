"""
Project Self-Update Engine

Core Features:
- Scan entire project codebase
- Identify improvement points (performance, security, architecture, code quality, etc.)
- Automatically generate code improvement suggestions and patches
- Automatically apply improvements via CI/CD or PR
- Monitor improvement effects and form feedback loops
- Integrate with three-version system

This is a core extension of the Version Control AI that enables the entire project
to enter a self-updating cycle.
"""

import os
import json
import asyncio
import logging
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import hashlib
import difflib

logger = logging.getLogger(__name__)


class ImprovementCategory(str, Enum):
    """Improvement category"""
    PERFORMANCE = "performance"
    SECURITY = "security"
    ARCHITECTURE = "architecture"
    CODE_QUALITY = "code_quality"
    TEST_COVERAGE = "test_coverage"
    DOCUMENTATION = "documentation"
    DEPENDENCY = "dependency"
    REFACTORING = "refactoring"


class ImprovementPriority(str, Enum):
    """Improvement priority"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ImprovementStatus(str, Enum):
    """Improvement status"""
    IDENTIFIED = "identified"
    PATCH_GENERATED = "patch_generated"
    REVIEWING = "reviewing"
    APPROVED = "approved"
    APPLIED = "applied"
    TESTING = "testing"
    VERIFIED = "verified"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


@dataclass
class CodeIssue:
    """Code issue"""
    issue_id: str
    file_path: str
    line_start: int
    line_end: int
    category: ImprovementCategory
    priority: ImprovementPriority
    description: str
    current_code: str
    suggested_code: Optional[str] = None
    impact_analysis: Optional[str] = None
    estimated_effort: Optional[str] = None
    related_files: List[str] = field(default_factory=list)
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ImprovementPatch:
    """Improvement patch"""
    patch_id: str
    issue_id: str
    file_path: str
    patch_type: str  # "diff", "full_file", "insert", "delete"
    original_code: str
    improved_code: str
    diff: str
    description: str
    category: ImprovementCategory
    priority: ImprovementPriority
    estimated_impact: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: ImprovementStatus = ImprovementStatus.PATCH_GENERATED
    applied_at: Optional[datetime] = None
    applied_by: Optional[str] = None
    verification_results: Optional[Dict[str, Any]] = None


@dataclass
class ProjectScanResult:
    """Project scan result"""
    scan_id: str
    scan_timestamp: datetime
    total_files_scanned: int
    total_lines_scanned: int
    issues_found: List[CodeIssue]
    issues_by_category: Dict[str, int]
    issues_by_priority: Dict[str, int]
    project_metrics: Dict[str, Any]
    scan_duration_seconds: float


@dataclass
class ImprovementCycle:
    """Improvement cycle"""
    cycle_id: str
    start_time: datetime
    end_time: Optional[datetime]
    patches_generated: int
    patches_applied: int
    patches_verified: int
    patches_rolled_back: int
    overall_impact: Dict[str, Any]
    status: str  # "running", "completed", "failed"


class ProjectSelfUpdateEngine:
    """
    Project Self-Update Engine

    Responsible for automatic scanning, analysis, improvement, and application
    of the entire project. Integrates with the three-version system to form
    a complete self-updating cycle.
    """

    def __init__(
        self,
        project_root: str,
        version_manager: Optional[Any] = None,
        code_analysis_engine: Optional[Any] = None,
        ai_model: Optional[Any] = None,
        auto_apply: bool = False,
        create_pr: bool = True,
        git_repo_path: Optional[str] = None,
    ):
        """
        Initialize Project Self-Update Engine

        Args:
            project_root: Project root directory
            version_manager: Three-version manager instance
            code_analysis_engine: Code analysis engine instance
            ai_model: AI model instance (for generating improvement suggestions)
            auto_apply: Whether to auto-apply improvements (use with caution)
            create_pr: Whether to create PR (recommended)
            git_repo_path: Git repository path (for creating PR)
        """
        self.project_root = Path(project_root)
        self.version_manager = version_manager
        self.code_analysis_engine = code_analysis_engine
        self.ai_model = ai_model
        self.auto_apply = auto_apply
        self.create_pr = create_pr
        self.git_repo_path = git_repo_path or str(self.project_root)

        # Storage
        self.scans: Dict[str, ProjectScanResult] = {}
        self.patches: Dict[str, ImprovementPatch] = {}
        self.cycles: Dict[str, ImprovementCycle] = {}
        self.applied_patches: List[str] = []

        # Configuration
        self.ignore_patterns = {
            "node_modules", "__pycache__", ".git", ".venv", "venv",
            "dist", "build", ".mypy_cache", ".pytest_cache",
            "*.min.js", "*.min.css", ".next", "target", "*.pyc"
        }

        # Scannable file extensions
        self.scannable_extensions = {
            ".py", ".ts", ".tsx", ".js", ".jsx", ".java", ".go",
            ".rs", ".cpp", ".c", ".h", ".hpp", ".cs", ".rb", ".php"
        }

        # Mechanism configuration: timeout, retry, limits
        self.max_files_per_scan = 10000  # Maximum files per scan
        self.max_file_size_mb = 10  # Maximum file size (MB)
        self.scan_timeout_seconds = 3600  # Scan timeout (1 hour)
        self.max_concurrent_scans = 5  # Maximum concurrent scans
        self.retry_attempts = 3  # Retry attempts
        self.retry_delay_seconds = 1  # Retry delay (seconds)

        logger.info(f"Project Self-Update Engine initialized: {project_root}")

    async def scan_project(
        self,
        scan_id: Optional[str] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> ProjectScanResult:
        """
        Scan entire project codebase

        Args:
            scan_id: Scan ID (optional)
            include_patterns: Include file patterns
            exclude_patterns: Exclude file patterns

        Returns:
            Project scan result
        """
        scan_id = scan_id or f"scan_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now(timezone.utc)

        logger.info(f"Starting project scan: {scan_id}")

        # Collect all scannable files
        files_to_scan = await self._collect_files(include_patterns, exclude_patterns)

        logger.info(f"Found {len(files_to_scan)} files to scan")

        # Scan files (with concurrency control)
        issues: List[CodeIssue] = []
        total_lines = 0

        # Use semaphore to control concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent_scans)

        async def scan_with_limit(file_path: str):
            """Scan with concurrency limit"""
            async with semaphore:
                return await self._scan_file_with_retry(file_path)

        # Concurrent scanning
        tasks = [scan_with_limit(fp) for fp in files_to_scan]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to scan file {files_to_scan[i]}: {result}")
                continue

            if result:
                file_issues, line_count = result
                issues.extend(file_issues)
                total_lines += line_count

        # Statistics
        issues_by_category = {}
        issues_by_priority = {}

        for issue in issues:
            issues_by_category[issue.category.value] = issues_by_category.get(issue.category.value, 0) + 1
            issues_by_priority[issue.priority.value] = issues_by_priority.get(issue.priority.value, 0) + 1

        # Calculate project metrics
        project_metrics = await self._calculate_project_metrics(files_to_scan, issues)

        scan_duration = (datetime.now(timezone.utc) - start_time).total_seconds()

        result = ProjectScanResult(
            scan_id=scan_id,
            scan_timestamp=start_time,
            total_files_scanned=len(files_to_scan),
            total_lines_scanned=total_lines,
            issues_found=issues,
            issues_by_category=issues_by_category,
            issues_by_priority=issues_by_priority,
            project_metrics=project_metrics,
            scan_duration_seconds=scan_duration,
        )

        self.scans[scan_id] = result

        logger.info(
            f"Scan completed: {scan_id}, "
            f"files: {len(files_to_scan)}, "
            f"issues: {len(issues)}, "
            f"duration: {scan_duration:.2f}s"
        )

        return result

    async def _collect_files(
        self,
        include_patterns: Optional[List[str]],
        exclude_patterns: Optional[List[str]],
    ) -> List[str]:
        """
        Collect files to scan

        Mechanism optimizations:
        - Limit maximum file count to prevent infinite loops
        - Check file size, skip oversized files
        - Add timeout protection
        """
        files = []
        file_count = 0
        start_time = datetime.now(timezone.utc)

        try:
            for file_path in self.project_root.rglob("*"):
                # Timeout check
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                if elapsed > self.scan_timeout_seconds:
                    logger.warning(f"File collection timeout, collected {len(files)} files")
                    break

                # File count limit
                if file_count >= self.max_files_per_scan:
                    logger.warning(f"Reached maximum file count limit {self.max_files_per_scan}")
                    break

                if not file_path.is_file():
                    continue

                # Check file size
                try:
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    if file_size_mb > self.max_file_size_mb:
                        logger.debug(f"Skipping oversized file: {file_path} ({file_size_mb:.2f}MB)")
                        continue
                except (OSError, PermissionError) as e:
                    logger.warning(f"Cannot get file size {file_path}: {e}")
                    continue

                # Check extension
                if file_path.suffix not in self.scannable_extensions:
                    continue

                rel_path = str(file_path.relative_to(self.project_root))

                # Check ignore patterns
                if self._should_ignore(rel_path):
                    continue

                # Check include/exclude patterns
                if include_patterns and not any(
                    Path(rel_path).match(pattern) for pattern in include_patterns
                ):
                    continue

                if exclude_patterns and any(
                    Path(rel_path).match(pattern) for pattern in exclude_patterns
                ):
                    continue

                files.append(str(file_path))
                file_count += 1

        except Exception as e:
            logger.error(f"Error collecting files: {e}")
            # Return collected files, don't interrupt flow

        logger.info(f"File collection completed: {len(files)} files")
        return files

    def _should_ignore(self, path: str) -> bool:
        """检查路径是否应被忽略"""
        path_parts = Path(path).parts
        for pattern in self.ignore_patterns:
            if pattern in path_parts:
                return True
            if pattern.startswith("*") and path.endswith(pattern[1:]):
                return True
        return False

    async def _scan_file_with_retry(self, file_path: str) -> Optional[Tuple[List[CodeIssue], int]]:
        """
        带重试机制的文件扫描

        机制优化：
        - 重试机制
        - 超时保护
        - 异常处理
        """
        last_error = None

        for attempt in range(self.retry_attempts):
            try:
                return await asyncio.wait_for(
                    self._scan_file(file_path),
                    timeout=30.0  # 单文件扫描超时30秒
                )
            except asyncio.TimeoutError:
                logger.warning(f"文件扫描超时 {file_path} (尝试 {attempt + 1}/{self.retry_attempts})")
                last_error = TimeoutError(f"扫描超时: {file_path}")
            except Exception as e:
                logger.warning(f"文件扫描失败 {file_path} (尝试 {attempt + 1}/{self.retry_attempts}): {e}")
                last_error = e

            if attempt < self.retry_attempts - 1:
                await asyncio.sleep(self.retry_delay_seconds * (attempt + 1))

        logger.error(f"文件扫描最终失败 {file_path}: {last_error}")
        return None

    async def _scan_file(self, file_path: str) -> Tuple[List[CodeIssue], int]:
        """
        扫描单个文件

        Returns:
            (问题列表, 行数)
        """
        issues = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                line_count = len(lines)

            # 使用代码分析引擎（如果可用）
            if self.code_analysis_engine:
                try:
                    analysis_result = await self.code_analysis_engine.analyze_file(file_path)
                    # 转换分析结果为CodeIssue
                    for issue in analysis_result.issues:
                        code_issue = CodeIssue(
                            issue_id=f"{file_path}_{issue.line}_{hash(issue.message)}",
                            file_path=file_path,
                            line_start=issue.line,
                            line_end=issue.line,
                            category=self._map_severity_to_category(issue.severity),
                            priority=self._map_severity_to_priority(issue.severity),
                            description=issue.message,
                            current_code=lines[issue.line - 1] if issue.line <= len(lines) else "",
                        )
                        issues.append(code_issue)
                except Exception as e:
                    logger.warning(f"代码分析引擎分析失败 {file_path}: {e}")

            # 基础检查（如果没有分析引擎）
            if not self.code_analysis_engine:
                issues.extend(await self._basic_checks(file_path, content, lines))

        except Exception as e:
            logger.error(f"读取文件失败 {file_path}: {e}")

        return issues, line_count

    async def _basic_checks(
        self, file_path: str, content: str, lines: List[str]
    ) -> List[CodeIssue]:
        """基础代码检查"""
        issues = []

        # 检查TODO/FIXME
        for i, line in enumerate(lines, 1):
            if "TODO" in line.upper() or "FIXME" in line.upper():
                issues.append(CodeIssue(
                    issue_id=f"{file_path}_{i}_todo",
                    file_path=file_path,
                    line_start=i,
                    line_end=i,
                    category=ImprovementCategory.CODE_QUALITY,
                    priority=ImprovementPriority.LOW,
                    description=f"发现待办事项: {line.strip()}",
                    current_code=line,
                ))

        # 检查长行
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                issues.append(CodeIssue(
                    issue_id=f"{file_path}_{i}_long_line",
                    file_path=file_path,
                    line_start=i,
                    line_end=i,
                    category=ImprovementCategory.CODE_QUALITY,
                    priority=ImprovementPriority.LOW,
                    description=f"行过长 ({len(line)} 字符), 建议拆分",
                    current_code=line,
                ))

        return issues

    async def _calculate_project_metrics(
        self, files: List[str], issues: List[CodeIssue]
    ) -> Dict[str, Any]:
        """计算项目指标"""
        total_lines = 0
        total_files = len(files)

        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    total_lines += len(f.readlines())
            except:
                pass

        critical_issues = sum(1 for i in issues if i.priority == ImprovementPriority.CRITICAL)
        high_issues = sum(1 for i in issues if i.priority == ImprovementPriority.HIGH)

        return {
            "total_files": total_files,
            "total_lines": total_lines,
            "total_issues": len(issues),
            "critical_issues": critical_issues,
            "high_issues": high_issues,
            "issues_per_file": len(issues) / total_files if total_files > 0 else 0,
            "issues_per_1000_lines": (len(issues) / total_lines * 1000) if total_lines > 0 else 0,
        }

    def _map_severity_to_category(self, severity: str) -> ImprovementCategory:
        """映射严重程度到改进类别"""
        severity_lower = severity.lower()
        if "security" in severity_lower or "vulnerability" in severity_lower:
            return ImprovementCategory.SECURITY
        elif "performance" in severity_lower or "slow" in severity_lower:
            return ImprovementCategory.PERFORMANCE
        elif "architecture" in severity_lower or "design" in severity_lower:
            return ImprovementCategory.ARCHITECTURE
        else:
            return ImprovementCategory.CODE_QUALITY

    def _map_severity_to_priority(self, severity: str) -> ImprovementPriority:
        """映射严重程度到优先级"""
        severity_lower = severity.lower()
        if "critical" in severity_lower or "error" in severity_lower:
            return ImprovementPriority.CRITICAL
        elif "high" in severity_lower or "warning" in severity_lower:
            return ImprovementPriority.HIGH
        elif "medium" in severity_lower:
            return ImprovementPriority.MEDIUM
        else:
            return ImprovementPriority.LOW

    async def generate_improvement_patches(
        self,
        scan_result: ProjectScanResult,
        max_patches: Optional[int] = None,
        priority_filter: Optional[List[ImprovementPriority]] = None,
    ) -> List[ImprovementPatch]:
        """
        为扫描结果生成改进补丁

        Args:
            scan_result: 扫描结果
            max_patches: 最大补丁数
            priority_filter: 优先级过滤

        Returns:
            改进补丁列表
        """
        logger.info(f"开始生成改进补丁: {scan_result.scan_id}")

        # 过滤问题
        issues_to_patch = scan_result.issues_found
        if priority_filter:
            issues_to_patch = [i for i in issues_to_patch if i.priority in priority_filter]

        # 按优先级排序
        priority_order = {
            ImprovementPriority.CRITICAL: 0,
            ImprovementPriority.HIGH: 1,
            ImprovementPriority.MEDIUM: 2,
            ImprovementPriority.LOW: 3,
        }
        issues_to_patch.sort(key=lambda x: priority_order.get(x.priority, 99))

        if max_patches:
            issues_to_patch = issues_to_patch[:max_patches]

        patches = []

        # 并发控制：限制同时生成的补丁数
        semaphore = asyncio.Semaphore(10)  # 最多10个并发补丁生成

        async def generate_with_limit(issue: CodeIssue):
            """带并发限制的补丁生成"""
            async with semaphore:
                try:
                    patch = await asyncio.wait_for(
                        self._generate_patch_for_issue(issue),
                        timeout=60.0  # 单补丁生成超时60秒
                    )
                    return patch
                except asyncio.TimeoutError:
                    logger.error(f"补丁生成超时 {issue.issue_id}")
                    return None
                except Exception as e:
                    logger.error(f"生成补丁失败 {issue.issue_id}: {e}")
                    return None

        # 并发生成补丁
        tasks = [generate_with_limit(issue) for issue in issues_to_patch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                continue
            if result:
                patches.append(result)
                self.patches[result.patch_id] = result

        logger.info(f"生成 {len(patches)} 个改进补丁")

        return patches

    async def _generate_patch_for_issue(self, issue: CodeIssue) -> Optional[ImprovementPatch]:
        """为单个问题生成补丁"""
        try:
            file_path = Path(issue.file_path)
            if not file_path.exists():
                return None

            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
                original_lines = original_content.split('\n')

            # 生成改进代码（使用AI模型或规则）
            improved_code = await self._suggest_improvement(issue, original_lines)

            if not improved_code:
                return None

            # 生成diff
            diff = self._generate_diff(original_lines, improved_code.split('\n'))

            # 计算影响
            estimated_impact = await self._estimate_impact(issue, original_content, improved_code)

            patch_id = f"patch_{hash(issue.issue_id)}_{datetime.now(timezone.utc).timestamp()}"

            patch = ImprovementPatch(
                patch_id=patch_id,
                issue_id=issue.issue_id,
                file_path=str(file_path),
                patch_type="diff",
                original_code=original_content,
                improved_code=improved_code,
                diff=diff,
                description=issue.description,
                category=issue.category,
                priority=issue.priority,
                estimated_impact=estimated_impact,
            )

            return patch

        except Exception as e:
            logger.error(f"生成补丁失败 {issue.issue_id}: {e}")
            return None

    async def _suggest_improvement(
        self, issue: CodeIssue, original_lines: List[str]
    ) -> Optional[str]:
        """建议改进代码"""
        # 如果有AI模型，使用AI生成建议
        if self.ai_model:
            try:
                prompt = self._build_improvement_prompt(issue, original_lines)
                suggestion = await self.ai_model.generate(prompt)
                # 解析AI返回的建议代码
                return self._extract_code_from_suggestion(suggestion)
            except Exception as e:
                logger.warning(f"AI生成建议失败: {e}")

        # 否则使用规则生成
        return self._rule_based_improvement(issue, original_lines)

    def _build_improvement_prompt(self, issue: CodeIssue, lines: List[str]) -> str:
        """构建AI提示"""
        context_lines = lines[max(0, issue.line_start - 5):issue.line_end + 5]
        context = '\n'.join(context_lines)

        return f"""
请改进以下代码问题：

文件: {issue.file_path}
行: {issue.line_start}-{issue.line_end}
类别: {issue.category.value}
优先级: {issue.priority.value}
描述: {issue.description}

当前代码上下文:
```{Path(issue.file_path).suffix[1:]}
{context}
```

请提供改进后的完整代码片段，保持代码风格一致。
"""

    def _extract_code_from_suggestion(self, suggestion: str) -> Optional[str]:
        """从AI建议中提取代码"""
        # 尝试提取代码块
        import re
        code_blocks = re.findall(r'```[\w]*\n(.*?)```', suggestion, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        return suggestion.strip()

    def _rule_based_improvement(
        self, issue: CodeIssue, original_lines: List[str]
    ) -> Optional[str]:
        """基于规则的改进"""
        # 简单的规则改进示例
        if issue.category == ImprovementCategory.CODE_QUALITY:
            if "TODO" in issue.description.upper():
                # 移除TODO注释或转换为正式实现
                improved_lines = original_lines.copy()
                if issue.line_start <= len(improved_lines):
                    line = improved_lines[issue.line_start - 1]
                    improved_lines[issue.line_start - 1] = line.split('#')[0].strip()
                return '\n'.join(improved_lines)

        return None

    def _generate_diff(self, original_lines: List[str], improved_lines: List[str]) -> str:
        """生成diff"""
        diff = difflib.unified_diff(
            original_lines,
            improved_lines,
            lineterm='',
            n=3,
        )
        return '\n'.join(diff)

    async def _estimate_impact(
        self, issue: CodeIssue, original: str, improved: str
    ) -> Dict[str, Any]:
        """估算改进影响"""
        return {
            "lines_changed": abs(len(improved.split('\n')) - len(original.split('\n'))),
            "complexity_change": "unknown",
            "performance_impact": "unknown",
            "risk_level": issue.priority.value,
        }

    async def apply_patches(
        self,
        patch_ids: Optional[List[str]] = None,
        auto_approve: bool = False,
    ) -> Dict[str, Any]:
        """
        应用补丁

        Args:
            patch_ids: 要应用的补丁ID列表（None表示应用所有）
            auto_approve: 是否自动批准

        Returns:
            应用结果
        """
        if patch_ids is None:
            patch_ids = list(self.patches.keys())

        applied = []
        failed = []

        for patch_id in patch_ids:
            if patch_id not in self.patches:
                failed.append({"patch_id": patch_id, "reason": "补丁不存在"})
                continue

            patch = self.patches[patch_id]

            try:
                if not auto_approve and patch.status != ImprovementStatus.APPROVED:
                    logger.warning(f"补丁 {patch_id} 未批准，跳过")
                    continue

                # 应用补丁
                await self._apply_single_patch(patch)

                patch.status = ImprovementStatus.APPLIED
                patch.applied_at = datetime.now(timezone.utc)
                patch.applied_by = "self_update_engine"

                applied.append(patch_id)
                self.applied_patches.append(patch_id)

                logger.info(f"补丁应用成功: {patch_id}")

            except Exception as e:
                logger.error(f"应用补丁失败 {patch_id}: {e}")
                failed.append({"patch_id": patch_id, "reason": str(e)})

        return {
            "applied": applied,
            "failed": failed,
            "total": len(patch_ids),
        }

    async def _apply_single_patch(self, patch: ImprovementPatch):
        """应用单个补丁"""
        file_path = Path(patch.file_path)

        # 备份原文件
        backup_path = file_path.with_suffix(file_path.suffix + '.backup')
        if file_path.exists():
            import shutil
            shutil.copy2(file_path, backup_path)

        # 写入改进后的代码
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(patch.improved_code)

        logger.info(f"补丁已应用到: {file_path}")

    async def create_improvement_pr(
        self,
        patch_ids: List[str],
        branch_name: Optional[str] = None,
        pr_title: Optional[str] = None,
        pr_description: Optional[str] = None,
    ) -> Optional[str]:
        """
        创建改进PR

        Args:
            patch_ids: 补丁ID列表
            branch_name: 分支名
            pr_title: PR标题
            pr_description: PR描述

        Returns:
            PR URL或ID
        """
        if not self.create_pr:
            logger.warning("PR创建已禁用")
            return None

        try:
            # 创建分支
            branch_name = branch_name or f"auto-improve-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
            await self._create_git_branch(branch_name)

            # 应用补丁
            await self.apply_patches(patch_ids, auto_approve=True)

            # 提交更改
            commit_message = pr_title or f"自动改进: {len(patch_ids)} 个补丁"
            await self._commit_changes(commit_message)

            # 推送分支
            await self._push_branch(branch_name)

            # 创建PR（需要GitHub API或类似工具）
            pr_url = await self._create_pr_via_api(
                branch_name,
                pr_title or commit_message,
                pr_description or self._generate_pr_description(patch_ids),
            )

            logger.info(f"PR创建成功: {pr_url}")
            return pr_url

        except Exception as e:
            logger.error(f"创建PR失败: {e}")
            return None

    async def _create_git_branch(self, branch_name: str):
        """创建Git分支"""
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=self.git_repo_path,
            check=True,
        )

    async def _commit_changes(self, message: str):
        """提交更改"""
        subprocess.run(
            ["git", "add", "."],
            cwd=self.git_repo_path,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=self.git_repo_path,
            check=True,
        )

    async def _push_branch(self, branch_name: str):
        """推送分支"""
        subprocess.run(
            ["git", "push", "-u", "origin", branch_name],
            cwd=self.git_repo_path,
            check=True,
        )

    async def _create_pr_via_api(
        self, branch: str, title: str, description: str
    ) -> str:
        """通过API创建PR（占位，需要实际实现）"""
        # 这里需要集成GitHub/GitLab API
        # 返回PR URL
        return f"https://github.com/example/repo/pull/123"

    def _generate_pr_description(self, patch_ids: List[str]) -> str:
        """生成PR描述"""
        patches = [self.patches[pid] for pid in patch_ids if pid in self.patches]

        description = f"## 自动改进PR\n\n"
        description += f"本PR包含 {len(patches)} 个自动生成的代码改进。\n\n"
        description += f"### 改进类别统计\n\n"

        categories = {}
        for patch in patches:
            cat = patch.category.value
            categories[cat] = categories.get(cat, 0) + 1

        for cat, count in categories.items():
            description += f"- {cat}: {count}\n"

        description += f"\n### 补丁列表\n\n"
        for patch in patches[:10]:  # 只显示前10个
            description += f"- `{Path(patch.file_path).name}`: {patch.description}\n"

        return description

    async def monitor_improvements(
        self,
        patch_ids: List[str],
        duration_hours: int = 24,
    ) -> Dict[str, Any]:
        """
        监控改进效果

        Args:
            patch_ids: 补丁ID列表
            duration_hours: 监控时长（小时）

        Returns:
            监控结果
        """
        # 这里应该集成监控系统，跟踪：
        # - 错误率变化
        # - 性能指标变化
        # - 用户反馈
        # - 测试通过率

        return {
            "monitored_patches": len(patch_ids),
            "duration_hours": duration_hours,
            "metrics": {},
            "status": "monitoring",
        }

    async def run_full_cycle(
        self,
        max_patches: Optional[int] = 50,
        priority_filter: Optional[List[ImprovementPriority]] = None,
        create_pr: bool = True,
    ) -> ImprovementCycle:
        """
        运行完整的自更新周期

        1. 扫描项目
        2. 生成补丁
        3. 创建PR或应用
        4. 监控效果

        Returns:
            改进周期记录
        """
        cycle_id = f"cycle_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now(timezone.utc)

        logger.info(f"开始自更新周期: {cycle_id}")

        # 1. 扫描
        scan_result = await self.scan_project()

        # 2. 生成补丁
        patches = await self.generate_improvement_patches(
            scan_result,
            max_patches=max_patches,
            priority_filter=priority_filter or [ImprovementPriority.CRITICAL, ImprovementPriority.HIGH],
        )

        # 3. 应用或创建PR
        if create_pr and patches:
            patch_ids = [p.patch_id for p in patches]
            pr_url = await self.create_improvement_pr(patch_ids)
            logger.info(f"PR创建: {pr_url}")
        elif self.auto_apply and patches:
            result = await self.apply_patches([p.patch_id for p in patches], auto_approve=True)
            logger.info(f"补丁应用结果: {result}")

        # 4. 创建周期记录
        cycle = ImprovementCycle(
            cycle_id=cycle_id,
            start_time=start_time,
            end_time=datetime.now(timezone.utc),
            patches_generated=len(patches),
            patches_applied=len([p for p in patches if p.status == ImprovementStatus.APPLIED]),
            patches_verified=0,
            patches_rolled_back=0,
            overall_impact={},
            status="completed",
        )

        self.cycles[cycle_id] = cycle

        logger.info(f"自更新周期完成: {cycle_id}")

        return cycle

    def get_status_report(self) -> Dict[str, Any]:
        """获取状态报告"""
        return {
            "total_scans": len(self.scans),
            "total_patches": len(self.patches),
            "applied_patches": len(self.applied_patches),
            "total_cycles": len(self.cycles),
            "recent_scans": [
                {
                    "scan_id": s.scan_id,
                    "timestamp": s.scan_timestamp.isoformat(),
                    "issues_found": len(s.issues_found),
                }
                for s in list(self.scans.values())[-5:]
            ],
            "recent_patches": [
                {
                    "patch_id": p.patch_id,
                    "file": Path(p.file_path).name,
                    "status": p.status.value,
                    "category": p.category.value,
                }
                for p in list(self.patches.values())[-10:]
            ],
        }

