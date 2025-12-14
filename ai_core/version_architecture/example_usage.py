"""
版本架构系统使用示例

演示如何使用版本架构系统实现完整的技术升级流程
"""

import asyncio
import logging
from ai_core.version_architecture import (
    create_orchestrator,
    VersionRole,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """主函数"""
    # 1. 创建并启动协调器
    logger.info("初始化版本架构系统...")
    orchestrator = create_orchestrator()
    await orchestrator.start()
    
    try:
        # 2. 获取系统状态
        status = orchestrator.get_system_status()
        logger.info(f"系统状态: v1={status.v1_status}, v2={status.v2_status}, v3={status.v3_status}")
        
        # 3. 模拟技术升级流程
        logger.info("开始技术升级流程...")
        
        technology_data = {
            "name": "新技术方案A",
            "description": "基于新AI模型的代码审查技术",
            "performance_metrics": {
                "latency_improvement_pct": 20.0,
                "throughput_improvement_pct": 15.0,
            },
        }
        
        # 尝试升级技术
        result = await orchestrator.promote_technology(
            technology_id="tech_001",
            technology_data=technology_data,
            auto_approve=False,  # 需要人工审核
        )
        
        if result["success"]:
            logger.info("技术升级成功！")
            logger.info(f"决策: {result['decision']}")
        else:
            logger.warning(f"技术升级失败: {result['reason']}")
        
        # 4. 获取版本切换手册
        manual = orchestrator.get_version_switch_manual("v1", "v2")
        if manual:
            logger.info(f"版本切换手册已找到，包含 {len(manual.steps)} 个步骤")
        
        # 5. 等待一段时间以观察系统运行
        logger.info("系统运行中，等待30秒...")
        await asyncio.sleep(30)
        
        # 6. 再次检查系统状态
        status = orchestrator.get_system_status()
        logger.info(f"最终系统状态: 活动工作流={status.active_workflows}")
        
    finally:
        # 7. 停止系统
        logger.info("停止版本架构系统...")
        await orchestrator.stop()
        logger.info("系统已停止")


if __name__ == "__main__":
    asyncio.run(main())

