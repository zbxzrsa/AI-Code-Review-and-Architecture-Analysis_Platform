"""
基础模型训练基础设施 (Foundation Model Training Infrastructure)

模块功能描述:
    企业级基础模型训练系统，用于增强 VC-AI 和 CR-AI。

两种部署方案:
================

方案 A - 实用部署（推荐用于生产环境）:
    - 冻结基础模型 + LoRA 适配器
    - RAG 系统实现实时信息检索
    - 定期重训练（每周/每月）
    - 量化（INT8/INT4）提高效率
    - 成本: 约 $1-5M/年

方案 B - 完全自主学习（研究/大型企业）:
    - 完整 500B-1T MoE 模型
    - 7×24 持续学习
    - 自演化能力
    - 多时间尺度记忆
    - 成本: 约 $10-100M/年

技术规格:
    - 训练数据: 10-15 万亿 tokens
    - 架构: Transformer + MoE + Flash Attention + RoPE + 稀疏注意力
    - 参数: 500B-1T（MoE 效率约 50B 活跃）
    - 训练: 多节点分布式训练 + 4D 并行
    - 上下文: 128K-1M tokens

主要组件:
    1. Architecture: MoE Transformer 和高级注意力机制
    2. Data Pipeline: 网页爬取、清洗、去重、HDF5/Parquet 存储
    3. Pre-training: AdamW、余弦衰减、BF16/FP8、4D 并行
    4. Post-training: SFT、RLHF (PPO)、DPO、Constitutional AI
    5. Continuous Learning: CPT、DAP、CFT + 防遗忘技术
    6. Autonomous Learning: 自修改、多时间尺度、关联记忆
    7. Practical Deployment: LoRA、RAG、量化、成本控制
    8. VC-AI/CR-AI Integration: 三版本演化循环

最后修改日期: 2024-12-07
"""

from .architecture import (
    MoETransformer,
    MoEConfig,
    ExpertRouter,
    SparseAttention,
    RoPEEmbedding,
    FlashAttentionLayer,
)

from .data_pipeline import (
    DataPipeline,
    DataCleaner,
    Deduplicator,
    CommonCrawlProcessor,
    CodeDataProcessor,
    HDF5Storage,
    ParquetStorage,
)

from .pretraining import (
    PretrainingConfig,
    PretrainingEngine,
    DistributedTrainer4D,
    MixedPrecisionManager,
    CheckpointManager,
)

from .posttraining import (
    SFTTrainer,
    RLHFTrainer,
    PPOOptimizer,
    RewardModel,
    ConstitutionalAI,
    ValueAligner,
)

from .continual_learning import (
    ContinualPretraining,
    DomainAdaptive,
    ContinualFineTuning,
    EWCRegularizer,
    ExperienceReplay,
    ProgressiveNetworks,
)

from .autonomous_learning import (
    AutonomousLearningAgent,
    OnlineLearningModule,
    MemoryManagement,
    SelfEvaluationSystem,
    KnowledgeIntegration,
    SafetyMonitor,
)

# Practical Deployment (Plan A) - New modular structure (recommended)
from .deployment import (
    PracticalDeploymentSystem,
    PracticalDeploymentConfig,
    LoRAAdapterManager,
    LoRALayer,
    RAGSystem,
    RAGIndex,
    RetrainingScheduler,
    ModelQuantizer,
    INT4Quantizer,
    ModelDistiller,
    CostController,
    FaultToleranceManager,
    HealthChecker,
)

# Advanced Quantization (INT4/GPTQ/AWQ)
from .quantization import (
    AdvancedQuantizer,
    QuantizationConfig,
    QuantizationMethod,
    QuantizationStats,
    QuantizationStatus,
    INT4BitsAndBytesQuantizer,
    GPTQQuantizer,
    AWQQuantizer,
    estimate_quantized_size,
    benchmark_quantization,
)

# Vector indexing module for production
from .vector_index import (
    BaseVectorIndex,
    NumpyVectorIndex,
    FAISSVectorIndex,
    MilvusVectorIndex,
    IndexConfig,
    IndexType,
    create_vector_index,
)

from .vcai_crai_integration import (
    EnhancedVCAI,
    EnhancedCRAI,
    EnhancedDualAICoordinator,
    EnhancedVersionAIEngine,
    TrainingPipelineIntegration,
    create_enhanced_coordinator,
    create_enhanced_vcai,
    create_enhanced_crai,
)

__all__ = [
    # Architecture
    "MoETransformer",
    "MoEConfig", 
    "ExpertRouter",
    "SparseAttention",
    "RoPEEmbedding",
    "FlashAttentionLayer",
    # Data Pipeline
    "DataPipeline",
    "DataCleaner",
    "Deduplicator",
    "CommonCrawlProcessor",
    "CodeDataProcessor",
    "HDF5Storage",
    "ParquetStorage",
    # Pre-training
    "PretrainingConfig",
    "PretrainingEngine",
    "DistributedTrainer4D",
    "MixedPrecisionManager",
    "CheckpointManager",
    # Post-training
    "SFTTrainer",
    "RLHFTrainer",
    "PPOOptimizer",
    "RewardModel",
    "ConstitutionalAI",
    "ValueAligner",
    # Continual Learning
    "ContinualPretraining",
    "DomainAdaptive",
    "ContinualFineTuning",
    "EWCRegularizer",
    "ExperienceReplay",
    "ProgressiveNetworks",
    # Autonomous Learning
    "AutonomousLearningAgent",
    "OnlineLearningModule",
    "MemoryManagement",
    "SelfEvaluationSystem",
    "KnowledgeIntegration",
    "SafetyMonitor",
    # Practical Deployment (Plan A)
    "PracticalDeploymentSystem",
    "PracticalDeploymentConfig",
    "LoRAAdapterManager",
    "LoRALayer",
    "RAGSystem",
    "RAGIndex",
    "RetrainingScheduler",
    "ModelQuantizer",
    "INT4Quantizer",
    "ModelDistiller",
    "CostController",
    "FaultToleranceManager",
    "HealthChecker",
    # Advanced Quantization
    "AdvancedQuantizer",
    "QuantizationConfig",
    "QuantizationMethod",
    "QuantizationStats",
    "QuantizationStatus",
    "INT4BitsAndBytesQuantizer",
    "GPTQQuantizer",
    "AWQQuantizer",
    "estimate_quantized_size",
    "benchmark_quantization",
    # Vector Indexing
    "BaseVectorIndex",
    "NumpyVectorIndex",
    "FAISSVectorIndex",
    "MilvusVectorIndex",
    "IndexConfig",
    "IndexType",
    "create_vector_index",
    # VC-AI/CR-AI Integration
    "EnhancedVCAI",
    "EnhancedCRAI",
    "EnhancedDualAICoordinator",
    "EnhancedVersionAIEngine",
    "TrainingPipelineIntegration",
    "create_enhanced_coordinator",
    "create_enhanced_vcai",
    "create_enhanced_crai",
]

__version__ = "2.0.0"
