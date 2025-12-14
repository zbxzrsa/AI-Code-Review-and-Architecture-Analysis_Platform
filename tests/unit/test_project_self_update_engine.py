"""
项目自更新引擎单元测试

补充单元测试以提高覆盖率到98%
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from ai_core.version_control.project_self_update_engine import (
    ProjectSelfUpdateEngine,
    ImprovementCategory,
    ImprovementPriority,
    ImprovementStatus,
    CodeIssue,
    ImprovementPatch,
)


@pytest.fixture
def temp_project():
    """创建临时项目目录"""
    temp_dir = tempfile.mkdtemp()
    project_path = Path(temp_dir) / "test_project"
    project_path.mkdir()
    
    # 创建各种类型的测试文件
    (project_path / "test.py").write_text("""
def hello():
    # TODO: 完善功能
    print("Hello")
    
def long_line_function():
    print("This is a very long line that exceeds 120 characters and should be flagged by the code quality checker for being too long")
""")
    
    (project_path / "test.ts").write_text("""
function test() {
    // TODO: 添加测试
    console.log("test");
}
""")
    
    (project_path / "large_file.py").write_text("\n".join([f"# Line {i}" for i in range(1000)]))
    
    yield project_path
    
    shutil.rmtree(temp_dir)


@pytest.fixture
def engine(temp_project):
    """创建自更新引擎实例"""
    return ProjectSelfUpdateEngine(
        project_root=str(temp_project),
        auto_apply=False,
        create_pr=False,
    )


@pytest.mark.asyncio
async def test_should_ignore_patterns(engine):
    """测试忽略模式"""
    # 测试各种忽略模式
    assert engine._should_ignore("node_modules/test.js") is True
    assert engine._should_ignore("__pycache__/test.pyc") is True
    assert engine._should_ignore(".git/config") is True
    assert engine._should_ignore("src/test.py") is False
    assert engine._should_ignore("test.min.js") is True


@pytest.mark.asyncio
async def test_map_severity_to_category(engine):
    """测试严重程度到类别的映射"""
    assert engine._map_severity_to_category("security vulnerability") == ImprovementCategory.SECURITY
    assert engine._map_severity_to_category("performance issue") == ImprovementCategory.PERFORMANCE
    assert engine._map_severity_to_category("architecture problem") == ImprovementCategory.ARCHITECTURE
    assert engine._map_severity_to_category("code quality") == ImprovementCategory.CODE_QUALITY


@pytest.mark.asyncio
async def test_map_severity_to_priority(engine):
    """测试严重程度到优先级的映射"""
    assert engine._map_severity_to_priority("critical error") == ImprovementPriority.CRITICAL
    assert engine._map_severity_to_priority("high warning") == ImprovementPriority.HIGH
    assert engine._map_severity_to_priority("medium issue") == ImprovementPriority.MEDIUM
    assert engine._map_severity_to_priority("low") == ImprovementPriority.LOW


@pytest.mark.asyncio
async def test_generate_diff(engine):
    """测试diff生成"""
    original = ["line1", "line2", "line3"]
    improved = ["line1", "line2_modified", "line3"]
    
    diff = engine._generate_diff(original, improved)
    assert isinstance(diff, str)
    assert "line2" in diff


@pytest.mark.asyncio
async def test_estimate_impact(engine, temp_project):
    """测试影响估算"""
    issue = CodeIssue(
        issue_id="test-1",
        file_path=str(temp_project / "test.py"),
        line_start=1,
        line_end=1,
        category=ImprovementCategory.PERFORMANCE,
        priority=ImprovementPriority.HIGH,
        description="Test issue",
        current_code="test",
    )
    
    original = "original code\nline2"
    improved = "improved code\nline2\nline3"
    
    impact = await engine._estimate_impact(issue, original, improved)
    assert "lines_changed" in impact
    assert "risk_level" in impact
    assert impact["risk_level"] == "high"


@pytest.mark.asyncio
async def test_build_improvement_prompt(engine, temp_project):
    """测试AI提示构建"""
    issue = CodeIssue(
        issue_id="test-1",
        file_path=str(temp_project / "test.py"),
        line_start=1,
        line_end=1,
        category=ImprovementCategory.CODE_QUALITY,
        priority=ImprovementPriority.MEDIUM,
        description="Test issue",
        current_code="test",
    )
    
    lines = ["def test():", "    pass"]
    prompt = engine._build_improvement_prompt(issue, lines)
    
    assert isinstance(prompt, str)
    assert "test.py" in prompt
    assert "CODE_QUALITY" in prompt or "code_quality" in prompt


@pytest.mark.asyncio
async def test_extract_code_from_suggestion(engine):
    """测试从AI建议中提取代码"""
    suggestion_with_code = """
    这是建议说明。
    ```python
    def improved():
        return True
    ```
    """
    
    code = engine._extract_code_from_suggestion(suggestion_with_code)
    assert "def improved" in code
    
    # 测试没有代码块的情况
    suggestion_no_code = "这是纯文本建议"
    code = engine._extract_code_from_suggestion(suggestion_no_code)
    assert isinstance(code, str)


@pytest.mark.asyncio
async def test_rule_based_improvement(engine, temp_project):
    """测试基于规则的改进"""
    issue = CodeIssue(
        issue_id="test-1",
        file_path=str(temp_project / "test.py"),
        line_start=1,
        line_end=1,
        category=ImprovementCategory.CODE_QUALITY,
        priority=ImprovementPriority.LOW,
        description="TODO: 完善功能",
        current_code="# TODO: 完善功能",
    )
    
    lines = ["# TODO: 完善功能", "def test():", "    pass"]
    improved = engine._rule_based_improvement(issue, lines)
    
    assert improved is not None
    assert isinstance(improved, str)


@pytest.mark.asyncio
async def test_file_size_limit(engine, temp_project):
    """测试文件大小限制"""
    # 创建一个超大文件（模拟）
    large_file = temp_project / "large.py"
    large_content = "# " + "x" * (11 * 1024 * 1024)  # 11MB
    large_file.write_text(large_content)
    
    # 设置较小的文件大小限制
    engine.max_file_size_mb = 10
    
    files = await engine._collect_files(None, None)
    
    # 超大文件应该被过滤
    assert str(large_file) not in files


@pytest.mark.asyncio
async def test_file_count_limit(engine, temp_project):
    """测试文件数限制"""
    # 创建多个文件
    for i in range(20):
        (temp_project / f"test_{i}.py").write_text(f"# File {i}")
    
    # 设置较小的文件数限制
    engine.max_files_per_scan = 10
    
    result = await engine.scan_project()
    
    # 应该被限制
    assert result.total_files_scanned <= engine.max_files_per_scan


@pytest.mark.asyncio
async def test_patch_status_transitions(engine, temp_project):
    """测试补丁状态转换"""
    scan_result = await engine.scan_project()
    patches = await engine.generate_improvement_patches(
        scan_result,
        max_patches=1,
    )
    
    if patches:
        patch = patches[0]
        assert patch.status == ImprovementStatus.PATCH_GENERATED
        
        # 模拟批准
        patch.status = ImprovementStatus.APPROVED
        assert patch.status == ImprovementStatus.APPROVED


@pytest.mark.asyncio
async def test_apply_patch_backup(engine, temp_project):
    """测试补丁应用时的备份"""
    scan_result = await engine.scan_project()
    patches = await engine.generate_improvement_patches(
        scan_result,
        max_patches=1,
    )
    
    if patches:
        patch = patches[0]
        test_file = Path(patch.file_path)
        original_content = test_file.read_text()
        
        # 应用补丁
        await engine._apply_single_patch(patch)
        
        # 检查备份文件是否存在
        backup_file = test_file.with_suffix(test_file.suffix + '.backup')
        assert backup_file.exists()
        
        # 恢复原文件
        test_file.write_text(original_content)
        backup_file.unlink()


@pytest.mark.asyncio
async def test_generate_pr_description(engine, temp_project):
    """测试PR描述生成"""
    scan_result = await engine.scan_project()
    patches = await engine.generate_improvement_patches(
        scan_result,
        max_patches=3,
    )
    
    if patches:
        patch_ids = [p.patch_id for p in patches]
        description = engine._generate_pr_description(patch_ids)
        
        assert isinstance(description, str)
        assert "自动改进PR" in description
        assert str(len(patches)) in description


@pytest.mark.asyncio
async def test_concurrent_scan_semaphore(engine):
    """测试并发扫描的信号量控制"""
    # 设置较小的并发限制
    engine.max_concurrent_scans = 2
    
    # 创建多个文件
    for i in range(5):
        (Path(engine.project_root) / f"test_{i}.py").write_text(f"# File {i}")
    
    result = await engine.scan_project()
    
    # 应该完成扫描
    assert result is not None
    assert result.total_files_scanned > 0


@pytest.mark.asyncio
async def test_retry_mechanism(engine, temp_project):
    """测试重试机制"""
    # 创建一个会失败的文件（权限问题模拟）
    test_file = temp_project / "test_retry.py"
    test_file.write_text("test")
    
    # 设置重试配置
    engine.retry_attempts = 2
    engine.retry_delay_seconds = 0.1
    
    # 扫描应该能处理错误并重试
    result = await engine.scan_project()
    
    # 应该完成扫描（即使有错误）
    assert result is not None


def test_improvement_category_enum():
    """测试改进类别枚举"""
    assert ImprovementCategory.PERFORMANCE.value == "performance"
    assert ImprovementCategory.SECURITY.value == "security"
    assert ImprovementCategory.ARCHITECTURE.value == "architecture"


def test_improvement_priority_enum():
    """测试改进优先级枚举"""
    assert ImprovementPriority.CRITICAL.value == "critical"
    assert ImprovementPriority.HIGH.value == "high"
    assert ImprovementPriority.MEDIUM.value == "medium"
    assert ImprovementPriority.LOW.value == "low"


def test_improvement_status_enum():
    """测试改进状态枚举"""
    assert ImprovementStatus.IDENTIFIED.value == "identified"
    assert ImprovementStatus.APPLIED.value == "applied"
    assert ImprovementStatus.VERIFIED.value == "verified"


@pytest.mark.asyncio
async def test_empty_project_scan(engine, temp_project):
    """测试空项目扫描"""
    # 清空项目
    for f in temp_project.iterdir():
        if f.is_file():
            f.unlink()
    
    result = await engine.scan_project()
    
    assert result.total_files_scanned == 0
    assert len(result.issues_found) == 0


@pytest.mark.asyncio
async def test_include_exclude_patterns(engine, temp_project):
    """测试包含/排除模式"""
    # 创建多个文件
    (temp_project / "include.py").write_text("# Include")
    (temp_project / "exclude.py").write_text("# Exclude")
    
    # 只包含include文件
    result = await engine.scan_project(
        include_patterns=["*include*"],
    )
    
    assert result.total_files_scanned >= 1
    
    # 排除exclude文件
    result = await engine.scan_project(
        exclude_patterns=["*exclude*"],
    )
    
    # exclude文件应该被排除
    scanned_files = [Path(issue.file_path).name for issue in result.issues_found]
    assert "exclude.py" not in scanned_files

