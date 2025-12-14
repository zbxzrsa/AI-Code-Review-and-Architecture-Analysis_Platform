"""
版本架构使用示例

演示如何使用版本架构系统进行技术更新和监控
"""

import asyncio
import logging
from ai_core.version_architecture import (
    VersionArchitectureOrchestrator,
    VersionType,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def example_basic_usage():
    """基本使用示例"""
    logger.info("=== 基本使用示例 ===")
    
    # 创建协调器
    orchestrator = VersionArchitectureOrchestrator()
    
    try:
        # 启动系统
        await orchestrator.start()
        logger.info("系统已启动")
        
        # 获取系统状态
        status = orchestrator.get_system_status()
        logger.info(f"系统状态: {status}")
        
        # 等待一段时间让系统运行
        await asyncio.sleep(5)
        
    finally:
        # 停止系统
        await orchestrator.stop()
        logger.info("系统已停止")


async def example_tech_update():
    """技术更新示例"""
    logger.info("=== 技术更新示例 ===")
    
    orchestrator = VersionArchitectureOrchestrator()
    
    try:
        await orchestrator.start()
        
        # 处理技术更新
        result = await orchestrator.process_tech_update(
            tech_name="new_attention_mechanism",
            tech_config={
                "type": "attention",
                "heads": 8,
                "dim": 512,
            },
        )
        
        logger.info(f"技术更新结果: {result}")
        
        await asyncio.sleep(10)
        
    finally:
        await orchestrator.stop()


async def example_monitoring():
    """监控示例"""
    logger.info("=== 监控示例 ===")
    
    orchestrator = VersionArchitectureOrchestrator()
    
    try:
        await orchestrator.start()
        
        # 定期检查系统状态
        for i in range(5):
            status = orchestrator.get_system_status()
            
            # 检查V2状态
            v2_status = status["versions"]["v2"]
            logger.info(f"V2状态: {v2_status['monitor_status']}")
            
            await asyncio.sleep(2)
            
    finally:
        await orchestrator.stop()


async def example_rollback():
    """回滚示例"""
    logger.info("=== 回滚示例 ===")
    
    orchestrator = VersionArchitectureOrchestrator()
    
    try:
        await orchestrator.start()
        
        # 获取V2回滚管理器
        rollback_manager = orchestrator.rollback_managers[VersionType.V2_STABLE]
        
        # 创建回滚点
        rollback_point = await rollback_manager.create_rollback_point(
            state_snapshot={"version": "2.0.0"},
            tech_stack=["tech1", "tech2"],
            performance_baseline={"latency": 2000, "throughput": 100},
        )
        
        logger.info(f"创建回滚点: {rollback_point.point_id}")
        
        # 获取回滚点列表
        points = rollback_manager.get_rollback_points(limit=5)
        logger.info(f"回滚点数量: {len(points)}")
        
        # 执行回滚（示例，实际不应在正常运行时回滚）
        # success = await rollback_manager.rollback(
        #     point_id=rollback_point.point_id,
        #     reason="示例回滚"
        # )
        # logger.info(f"回滚结果: {success}")
        
    finally:
        await orchestrator.stop()


async def main():
    """主函数"""
    logger.info("开始运行版本架构示例")
    
    # 运行各个示例
    await example_basic_usage()
    await asyncio.sleep(1)
    
    await example_tech_update()
    await asyncio.sleep(1)
    
    await example_monitoring()
    await asyncio.sleep(1)
    
    await example_rollback()
    
    logger.info("所有示例运行完成")


if __name__ == "__main__":
    asyncio.run(main())

