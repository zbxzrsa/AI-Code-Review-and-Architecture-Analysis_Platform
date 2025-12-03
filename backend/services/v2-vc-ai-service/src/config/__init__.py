"""V2 VC-AI Configuration Package"""

from .settings import settings
from .model_config import MODEL_CONFIG, CONSISTENCY_GUARANTEES, FAILOVER_STRATEGY
from .slo_config import SLO_DEFINITIONS, MONITORING_SETUP

__all__ = [
    "settings",
    "MODEL_CONFIG",
    "CONSISTENCY_GUARANTEES",
    "FAILOVER_STRATEGY",
    "SLO_DEFINITIONS",
    "MONITORING_SETUP",
]
