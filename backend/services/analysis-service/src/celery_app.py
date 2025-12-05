"""
Celery Application Configuration
Handles async task processing for code analysis
"""

import os
from celery import Celery

# Redis URL for broker and backend
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Celery app
celery_app = Celery(
    "analysis_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["src.tasks"]
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task execution settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=300,  # 5 minutes max per task
    task_soft_time_limit=240,  # 4 minutes soft limit
    
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
    
    # Result settings
    result_expires=3600,  # 1 hour
    
    # Beat schedule for periodic tasks
    beat_schedule={
        "cleanup-old-results": {
            "task": "src.tasks.cleanup_old_results",
            "schedule": 3600.0,  # Every hour
        },
    },
)

# Alias for celery -A src.celery_app
app = celery_app
