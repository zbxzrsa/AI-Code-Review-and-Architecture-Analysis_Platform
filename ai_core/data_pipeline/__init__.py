"""
数据管道模块 (Data Pipeline Module)

模块功能描述:
    提供数据清洗、质量评估和异常检测功能。

主要功能:
    - 自动化数据清洗
    - 数据质量评估和报告
    - 异常检测和自动修复
    - 多模态数据处理

主要组件:
    - DataCleaningPipeline: 数据清洗管道
    - QualityAssessor: 质量评估器
    - AnomalyDetector: 异常检测器
    - MultiModalCleaner: 多模态清洗器

最后修改日期: 2024-12-07
"""

from .data_cleaning import DataCleaningPipeline
from .quality_assessor import QualityAssessor, DataQualityReport
from .anomaly_detector import AnomalyDetector, AutoRepair
from .multimodal_cleaner import MultiModalCleaner

__all__ = [
    'DataCleaningPipeline',
    'QualityAssessor',
    'DataQualityReport',
    'AnomalyDetector',
    'AutoRepair',
    'MultiModalCleaner'
]
