"""Evaluation API endpoints."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlmodel import Session, select

from ..db import get_session
from ..models import EvaluationMetric, EvaluationRun, GoldStandard, Meeting
from ..services.evaluation import create_gold_standard, evaluate_meeting

logger = logging.getLogger(__name__)
router = APIRouter()


def _json_loads_safe(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _evaluation_summary(run: EvaluationRun) -> Dict[str, Any]:
    details = _json_loads_safe(run.details_json, {})
    return {
        "id": run.id,
        "meeting_id": run.meeting_id,
        "gold_id": run.gold_standard_id,
        "overall_score": run.overall_quality_score,
        "model_whisper": run.model_whisper,
        "model_task": run.model_task,
        "run_date": run.run_date.isoformat() if run.run_date else None,
        "wer": details.get("wer"),
        "cer": details.get("cer"),
        "transcript_quality": details.get("transcript_quality"),
        "task_set_f1": details.get("task_set_f1"),
        "task_set_precision": details.get("task_set_precision"),
        "task_set_recall": details.get("task_set_recall"),
        "assignee_accuracy": details.get("assignee_accuracy"),
        "deadline_accuracy": details.get("deadline_accuracy"),
        "hallucination_rate": details.get("hallucination_rate"),
        "predicted_tasks": details.get("predicted_tasks"),
        "gold_tasks": details.get("gold_tasks"),
        "matched_tasks": details.get("matched_tasks"),
        "task_provider": details.get("task_provider"),
        "task_model": details.get("task_model"),
        "task_parse_stage": details.get("task_parse_stage"),
        "task_fallback_used": details.get("task_fallback_used"),
        "task_fallback_merged": details.get("task_fallback_merged"),
        "has_diarization": details.get("has_diarization"),
    }


@router.get("/gold_standards")
def list_gold_standards(limit: int = Query(20, ge=1, le=100), session: Session = Depends(get_session)):
    golds = session.exec(select(GoldStandard).limit(limit)).all()
    return {
        "gold_standards": [
            {
                "id": g.id,
                "meeting_ref": g.meeting_ref,
                "source": g.transcript_source,
                "language": g.language,
                "tasks_count": len(_json_loads_safe(g.tasks_json, [])),
                "created_at": g.created_at.isoformat() if g.created_at else None,
                "notes": g.notes,
            }
            for g in golds
        ]
    }


@router.post("/gold_standards")
def add_gold_standard(gold: GoldStandard, session: Session = Depends(get_session)):
    existing = session.exec(select(GoldStandard).where(GoldStandard.meeting_ref == gold.meeting_ref)).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Gold standard with meeting_ref '{gold.meeting_ref}' already exists")
    session.add(gold)
    session.commit()
    session.refresh(gold)
    return gold


@router.post("/gold_standards/upload")
async def upload_gold_standard_from_audio(
    file: UploadFile = File(...),
    meeting_ref: str = Query(..., description="Unique reference for this gold standard"),
    transcript: str = Query(..., description="Gold transcript (exact)"),
    tasks_json: str = Query(..., description="JSON array of gold tasks"),
    session: Session = Depends(get_session),
):
    try:
        tasks = json.loads(tasks_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid tasks_json")

    _ = await file.read()  # keep API backward-compatible even if audio is not used yet

    gold = GoldStandard(
        meeting_ref=meeting_ref,
        transcript=transcript,
        transcript_source="manual",
        tasks_json=json.dumps(tasks, ensure_ascii=False),
        notes=f"Uploaded via API with file {file.filename}",
    )
    session.add(gold)
    session.commit()
    session.refresh(gold)

    return {"gold_id": gold.id, "status": "created", "tasks_count": len(tasks)}


@router.post("/evaluate/meeting/{meeting_id}")
def evaluate_meeting_endpoint(meeting_id: int, session: Session = Depends(get_session)):
    """Evaluate a processed meeting against its gold standard."""
    meeting = session.get(Meeting, meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    try:
        run, metrics = evaluate_meeting(meeting=meeting, session=session)
        return {
            "evaluation_run_id": run.id if run else None,
            "overall_quality_score": run.overall_quality_score if run else None,
            "metrics": metrics,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Evaluation failed for meeting {meeting_id}")
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")


@router.get("/evaluations")
def list_evaluations(
    limit: int = Query(20, ge=1, le=100),
    meeting_id: int | None = Query(None),
    session: Session = Depends(get_session),
):
    query = select(EvaluationRun)
    if meeting_id is not None:
        query = query.where(EvaluationRun.meeting_id == meeting_id)
    query = query.order_by(EvaluationRun.run_date.desc()).limit(limit)
    runs = session.exec(query).all()
    return {"evaluations": [_evaluation_summary(r) for r in runs]}


@router.get("/evaluation/{run_id}")
def get_evaluation_details(run_id: int, session: Session = Depends(get_session)):
    run = session.get(EvaluationRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")

    details = _json_loads_safe(run.details_json, {})
    metrics = session.exec(select(EvaluationMetric).where(EvaluationMetric.evaluation_run_id == run_id)).all()
    details["individual_metrics"] = [
        {
            "name": m.metric_name,
            "value": m.value,
            "breakdown": _json_loads_safe(m.breakdown_json, None),
        }
        for m in metrics
    ]
    return {
        "run": {
            "id": run.id,
            "meeting_id": run.meeting_id,
            "gold_standard_id": run.gold_standard_id,
            "overall_quality_score": run.overall_quality_score,
            "model_whisper": run.model_whisper,
            "model_task": run.model_task,
            "run_date": run.run_date.isoformat() if run.run_date else None,
            "notes": run.notes,
        },
        "metrics": details,
    }
