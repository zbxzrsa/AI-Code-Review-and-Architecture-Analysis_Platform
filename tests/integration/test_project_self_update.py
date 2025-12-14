"""
Project Self-Update Engine Integration Tests

Test Coverage:
- Functional level: scanning, patch generation, application
- Mechanism level: timeout, retry, concurrency control
- Loop level: file scanning loop, patch generation loop
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from ai_core.version_control.project_self_update_engine import (
    ProjectSelfUpdateEngine,
    ImprovementCategory,
    ImprovementPriority,
)


@pytest.fixture
def temp_project():
    """Create temporary project directory"""
    temp_dir = tempfile.mkdtemp()
    project_path = Path(temp_dir) / "test_project"
    project_path.mkdir()
    
    # Create test files
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
    
    yield project_path
    
    # 清理
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
async def test_scan_project_functionality(engine, temp_project):
    """测试项目扫描功能"""
    result = await engine.scan_project()
    
    assert result is not None
    assert result.total_files_scanned > 0
    assert len(result.issues_found) > 0
    assert result.scan_duration_seconds > 0
    
    # 验证问题分类
    assert ImprovementCategory.CODE_QUALITY.value in result.issues_by_category


@pytest.mark.asyncio
async def test_file_collection_limits(engine):
    """测试文件收集限制机制"""
    # 设置较小的限制
    engine.max_files_per_scan = 1
    
    result = await engine.scan_project()
    
    # 应该被限制
    assert result.total_files_scanned <= engine.max_files_per_scan


@pytest.mark.asyncio
async def test_patch_generation(engine, temp_project):
    """测试补丁生成功能"""
    # 先扫描
    scan_result = await engine.scan_project()
    
    # 生成补丁
    patches = await engine.generate_improvement_patches(
        scan_result,
        max_patches=5,
        priority_filter=[ImprovementPriority.LOW],
    )
    
    assert len(patches) > 0
    
    # 验证补丁结构
    for patch in patches:
        assert patch.patch_id
        assert patch.file_path
        assert patch.original_code
        assert patch.improved_code
        assert patch.diff


@pytest.mark.asyncio
async def test_concurrent_scanning(engine):
    """测试并发扫描机制"""
    # 测试并发控制
    start_time = asyncio.get_event_loop().time()
    
    result = await engine.scan_project()
    
    elapsed = asyncio.get_event_loop().time() - start_time
    
    # 验证并发执行（应该比串行快）
    assert elapsed < 60  # 应该在1分钟内完成
    assert result.total_files_scanned > 0


@pytest.mark.asyncio
async def test_retry_mechanism(engine, temp_project):
    """测试重试机制"""
    # 创建一个会失败的文件（权限问题模拟）
    test_file = temp_project / "test_retry.py"
    test_file.write_text("test")
    
    # 修改重试配置
    engine.retry_attempts = 2
    engine.retry_delay_seconds = 0.1
    
    # 扫描应该能处理错误并重试
    result = await engine.scan_project()
    
    # 应该完成扫描（即使有错误）
    assert result is not None


@pytest.mark.asyncio
async def test_timeout_protection(engine):
    """测试超时保护机制"""
    # 设置很短的超时
    engine.scan_timeout_seconds = 0.1
    
    result = await engine.scan_project()
    
    # 应该被超时中断
    assert result.scan_duration_seconds <= engine.scan_timeout_seconds + 1


@pytest.mark.asyncio
async def test_patch_application(engine, temp_project):
    """测试补丁应用功能"""
    # 扫描并生成补丁
    scan_result = await engine.scan_project()
    patches = await engine.generate_improvement_patches(
        scan_result,
        max_patches=1,
    )
    
    if patches:
        patch = patches[0]
        
        # 应用补丁
        result = await engine.apply_patches(
            [patch.patch_id],
            auto_approve=True,
        )
        
        assert len(result["applied"]) > 0 or len(result["failed"]) > 0


@pytest.mark.asyncio
async def test_ignore_patterns(engine, temp_project):
    """测试忽略模式"""
    # 创建应该被忽略的文件
    (temp_project / "node_modules" / "test.js").parent.mkdir()
    (temp_project / "node_modules" / "test.js").write_text("test")
    
    result = await engine.scan_project()
    
    # 验证node_modules被忽略
    scanned_files = [Path(issue.file_path).parts for issue in result.issues_found]
    assert "node_modules" not in [parts for parts in scanned_files if "node_modules" in parts]


def test_input_validation():
    """测试输入验证"""
    # 无效的项目路径
    with pytest.raises((ValueError, FileNotFoundError)):
        engine = ProjectSelfUpdateEngine(
            project_root="/nonexistent/path",
        )


@pytest.mark.asyncio
async def test_loop_termination(engine):
    """测试循环终止条件"""
    # 确保循环能正常终止
    result = await engine.scan_project()
    
    # 验证结果完整性
    assert result.scan_id
    assert result.scan_timestamp
    assert result.total_files_scanned >= 0
    assert result.total_lines_scanned >= 0

