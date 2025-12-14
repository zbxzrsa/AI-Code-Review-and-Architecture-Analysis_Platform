# 项目自更新系统文档

## 概述

项目自更新系统是版本控制AI的核心扩展，使整个项目进入自动改进循环。系统不仅管理模型版本，还持续扫描、分析、改进整个项目代码库。

## 核心功能

### 1. 项目扫描
- 自动扫描整个项目代码库
- 识别性能、安全、架构、代码质量等问题
- 生成详细的问题报告和统计

### 2. 改进补丁生成
- 基于扫描结果自动生成改进补丁
- 使用AI模型或规则生成改进建议
- 生成diff和影响分析

### 3. 三版本验证流程
- **V1实验环境**：在沙盒中测试补丁
- **V3基准对比**：与基准版本对比性能（要求≥15%提升）
- **V2生产环境**：通过验证后应用到生产

### 4. 自动应用
- 创建PR（推荐）
- 直接应用（需谨慎）
- 监控改进效果

## 架构

```
项目自更新循环
├── 扫描项目
│   ├── 收集文件
│   ├── 代码分析
│   └── 问题识别
├── 生成补丁
│   ├── AI生成建议
│   ├── 规则改进
│   └── 影响分析
├── 三版本验证
│   ├── V1实验测试
│   ├── V3基准对比
│   └── V2生产准备
├── 应用改进
│   ├── 创建PR
│   └── 或直接应用
└── 监控反馈
    ├── 性能指标
    ├── 错误率
    └── 用户反馈
```

## 使用方法

### 基础使用

```python
from ai_core.version_control import (
    ProjectSelfUpdateService,
    create_self_update_service,
)

# 创建服务
service = await create_self_update_service(
    project_root="./",
    scan_interval_hours=24,
    create_pr=True,
    auto_improve=False,  # 建议设为False，使用PR流程
)

# 启动持续改进循环
await service.start()

# 或运行一次
result = await service.run_once()
print(f"生成补丁: {result['patches_generated']}")
print(f"应用补丁: {result['patches_applied']}")
```

### 手动扫描

```python
# 扫描项目
scan_result = await service.scan_project()
print(f"发现问题: {scan_result['total_issues']}")
print(f"关键问题: {scan_result['critical_issues']}")
```

### 获取状态

```python
status = await service.get_status()
print(f"运行状态: {status['running']}")
print(f"最新扫描: {status['health']['latest_scan']}")
```

## 配置选项

### VersionControlAIConfig

```python
from ai_core.version_control import (
    VersionControlAIConfig,
    ImprovementPriority,
)

config = VersionControlAIConfig(
    project_root="./",
    scan_interval_hours=24,  # 扫描间隔（小时）
    auto_improve=False,  # 是否自动应用
    create_pr=True,  # 是否创建PR
    max_patches_per_cycle=50,  # 每周期最大补丁数
    priority_filter=[
        ImprovementPriority.CRITICAL,
        ImprovementPriority.HIGH,
    ],
    integration_with_v1=True,  # 与V1集成
    integration_with_v2=True,  # 与V2集成
    integration_with_v3=True,  # 与V3集成
)
```

## 改进类别

- `PERFORMANCE`: 性能优化
- `SECURITY`: 安全修复
- `ARCHITECTURE`: 架构改进
- `CODE_QUALITY`: 代码质量
- `TEST_COVERAGE`: 测试覆盖
- `DOCUMENTATION`: 文档完善
- `DEPENDENCY`: 依赖更新
- `REFACTORING`: 代码重构

## 改进优先级

- `CRITICAL`: 关键问题（安全漏洞、严重bug）
- `HIGH`: 高优先级（性能问题、重要改进）
- `MEDIUM`: 中等优先级（代码质量、优化）
- `LOW`: 低优先级（代码风格、文档）

## 三版本验证流程

### V1实验环境
- 在隔离沙盒中应用补丁
- 运行完整测试套件
- 检查功能正确性
- 测量性能指标

### V3基准对比
- 获取基准版本性能指标
- 应用补丁后重新测量
- 对比性能差异
- **要求：性能提升≥15%或关键指标显著改善**

### V2生产环境
- 通过验证的补丁才可应用
- 创建PR或直接应用
- 监控生产环境指标
- 如有问题立即回滚

## 最佳实践

### 1. 使用PR流程
```python
# 推荐：创建PR，人工审查
service = await create_self_update_service(
    project_root="./",
    create_pr=True,
    auto_improve=False,
)
```

### 2. 设置合理的优先级过滤
```python
# 只处理关键和高优先级问题
config = VersionControlAIConfig(
    priority_filter=[
        ImprovementPriority.CRITICAL,
        ImprovementPriority.HIGH,
    ],
)
```

### 3. 定期监控
```python
# 定期检查状态
status = await service.get_status()
if status['health']['latest_scan']['critical_issues'] > 0:
    # 处理关键问题
    pass
```

### 4. 集成到CI/CD
```yaml
# .github/workflows/auto-improve.yml
name: Auto Improve
on:
  schedule:
    - cron: '0 2 * * *'  # 每天凌晨2点
jobs:
  improve:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Self-Update
        run: python -m ai_core.version_control.self_update_service
```

## 监控和反馈

### 改进效果监控

系统会自动监控：
- 错误率变化
- 性能指标变化
- 用户反馈
- 测试通过率

### 反馈循环

```
改进应用 → 监控效果 → 分析结果 → 调整策略 → 下一轮改进
```

## 安全考虑

1. **不要在生产环境直接自动应用**
   - 使用PR流程，人工审查
   - 或设置严格的自动应用条件

2. **备份重要文件**
   - 系统会自动备份，但建议额外备份

3. **测试覆盖**
   - 确保有足够的测试覆盖
   - 改进补丁必须通过测试

4. **回滚机制**
   - 系统支持回滚
   - 监控异常时自动回滚

## 故障排除

### 扫描失败
- 检查项目路径是否正确
- 检查文件权限
- 查看日志了解详细错误

### 补丁生成失败
- 检查AI模型是否可用
- 检查代码分析引擎配置
- 查看具体错误信息

### PR创建失败
- 检查Git配置
- 检查GitHub/GitLab API权限
- 查看网络连接

## API参考

### ProjectSelfUpdateService

#### `start()`
启动持续改进循环

#### `stop()`
停止服务

#### `run_once() -> Dict`
运行一次改进周期

#### `scan_project() -> Dict`
扫描项目

#### `get_status() -> Dict`
获取服务状态

### ProjectSelfUpdateEngine

#### `scan_project() -> ProjectScanResult`
扫描项目代码库

#### `generate_improvement_patches() -> List[ImprovementPatch]`
生成改进补丁

#### `apply_patches() -> Dict`
应用补丁

#### `create_improvement_pr() -> Optional[str]`
创建改进PR

## 示例

### 完整示例

```python
import asyncio
from ai_core.version_control import create_self_update_service

async def main():
    # 创建服务
    service = await create_self_update_service(
        project_root=".",
        scan_interval_hours=24,
        create_pr=True,
        max_patches_per_cycle=20,
    )
    
    # 运行一次改进周期
    result = await service.run_once()
    print(f"周期ID: {result['cycle_id']}")
    print(f"生成补丁: {result['patches_generated']}")
    print(f"应用补丁: {result['patches_applied']}")
    
    # 获取状态
    status = await service.get_status()
    print(f"项目健康: {status['health']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 更新日志

### v1.0.0 (2024-12-07)
- 初始版本
- 项目扫描功能
- 改进补丁生成
- 三版本验证流程
- PR创建支持

