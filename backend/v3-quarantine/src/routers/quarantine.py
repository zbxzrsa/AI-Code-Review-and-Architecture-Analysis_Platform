"""
Quarantine management endpoints for V3.
"""
import logging
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Query

from models.experiment import QuarantineRecord

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/quarantine")

# In-memory storage for demo
quarantine_store = {}


@router.get("/records")
async def list_quarantine_records(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List all quarantined experiments."""
    records = list(quarantine_store.values())
    total = len(records)
    records = records[offset : offset + limit]

    return {
        "records": [
            {
                "id": r.id,
                "experiment_id": r.experiment_id,
                "reason": r.reason,
                "quarantined_at": r.quarantined_at,
                "can_re_evaluate": r.can_re_evaluate,
            }
            for r in records
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/records/{record_id}")
async def get_quarantine_record(record_id: str):
    """Get detailed quarantine record."""
    if record_id not in quarantine_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Quarantine record not found",
        )

    record = quarantine_store[record_id]

    return {
        "id": record.id,
        "experiment_id": record.experiment_id,
        "reason": record.reason,
        "failure_analysis": record.failure_analysis,
        "metrics_at_failure": record.metrics_at_failure.to_dict() if record.metrics_at_failure else None,
        "quarantined_at": record.quarantined_at,
        "quarantined_by": record.quarantined_by,
        "can_re_evaluate": record.can_re_evaluate,
        "impact_analysis": record.impact_analysis,
        "related_experiments": record.related_experiments,
    }


@router.post("/records/{experiment_id}")
async def create_quarantine_record(
    experiment_id: str,
    reason: str,
    failure_analysis: Optional[dict] = None,
    impact_analysis: Optional[dict] = None,
):
    """Create a new quarantine record."""
    record = QuarantineRecord(
        experiment_id=experiment_id,
        reason=reason,
        failure_analysis=failure_analysis or {},
        quarantined_by="system",
        impact_analysis=impact_analysis or {},
    )

    quarantine_store[record.id] = record

    logger.info(
        "Experiment quarantined",
        record_id=record.id,
        experiment_id=experiment_id,
        reason=reason,
    )

    return {
        "id": record.id,
        "experiment_id": experiment_id,
        "status": "quarantined",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/records/{record_id}/request-re-evaluation")
async def request_re_evaluation(
    record_id: str,
    notes: str,
):
    """Request re-evaluation of a quarantined experiment."""
    if record_id not in quarantine_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Quarantine record not found",
        )

    record = quarantine_store[record_id]

    if not record.can_re_evaluate:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This record cannot be re-evaluated",
        )

    record.re_evaluation_requested_at = datetime.utcnow()
    record.re_evaluation_requested_by = "admin"
    record.re_evaluation_notes = notes

    logger.info(
        "Re-evaluation requested",
        record_id=record_id,
        experiment_id=record.experiment_id,
    )

    return {
        "record_id": record_id,
        "re_evaluation_status": "requested",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/statistics")
async def quarantine_statistics():
    """Get statistics about quarantined experiments."""
    records = list(quarantine_store.values())

    return {
        "total_quarantined": len(records),
        "can_re_evaluate": sum(1 for r in records if r.can_re_evaluate),
        "permanently_blacklisted": sum(1 for r in records if not r.can_re_evaluate),
        "re_evaluation_pending": sum(1 for r in records if r.re_evaluation_requested_at),
        "quarantine_reasons": {
            "low_accuracy": sum(1 for r in records if "accuracy" in r.reason.lower()),
            "high_latency": sum(1 for r in records if "latency" in r.reason.lower()),
            "high_error_rate": sum(1 for r in records if "error" in r.reason.lower()),
            "other": sum(1 for r in records if not any(
                keyword in r.reason.lower() for keyword in ["accuracy", "latency", "error"]
            )),
        },
    }


@router.get("/impact-analysis/{record_id}")
async def get_impact_analysis(record_id: str):
    """Get impact analysis of a quarantined experiment."""
    if record_id not in quarantine_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Quarantine record not found",
        )

    record = quarantine_store[record_id]

    return {
        "record_id": record_id,
        "experiment_id": record.experiment_id,
        "impact_analysis": record.impact_analysis,
        "related_experiments": record.related_experiments,
        "recommendation": "Do not promote similar configurations to V2",
    }
