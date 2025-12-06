"""
Async Processing Module (Performance Optimization #2)

Provides optimized asynchronous task processing with:
- Priority-based scheduling
- Batch processing
- Rate limiting
- Automatic retries
"""
from .task_queue import (
    # Core classes
    AsyncTaskQueue,
    TaskScheduler,
    DeadLetterQueue,
    RateLimiter,
    
    # Task types
    Task,
    TaskBatch,
    TaskConfig,
    TaskPriority,
    TaskStatus,
    TaskHandler,
    TaskQueueStats,
    
    # Example handlers
    AnalysisTaskHandler,
    EmbeddingTaskHandler,
    
    # Factory functions
    get_task_queue,
    init_task_queue,
)
