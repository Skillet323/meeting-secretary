"""
Evaluation API endpoints.
"""
import json
from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File
from sqlmodel import Session, select
from ..db import get_session
from ..models import Meeting, GoldStandard, EvaluationRun, EvaluationMetric
from ..services.evaluation import (
    evaluate_meeting,
    load_gold_dataset,
    create_gold_standard
)
from ..services.transcription import transcribe_from_bytes
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/gold_standards")
def list_gold_standards(
    limit: int = Query(20, ge=1, le=100),
    session: Session = Depends(get_session)
):
    """List available gold standard annotations."""
    golds = session.exec(select(GoldStandard).limit(limit)).all()
    return {
        "gold_standards": [
            {
                "id": g.id,
                "meeting_ref": g.meeting_ref,
                "source": g.transcript_source,
                "language": g.language,
                "tasks_count": len(json.loads(g.tasks_json)) if g.tasks_json else 0,
                "created_at": g.created_at
            }
            for g in golds
        ]
    }


@router.post("/gold_standards")
def add_gold_standard(
    gold: GoldStandard,
    session: Session = Depends(get_session)
):
    """
    Manually add a gold standard annotation.
    Useful for building evaluation dataset.
    """
    # Check if meeting_ref already exists
    existing = session.exec(
        select(GoldStandard).where(GoldStandard.meeting_ref == gold.meeting_ref)
    ).first()
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
    session: Session = Depends(get_session)
):
    """
    Upload audio file and gold annotations together.
    This will store the gold standard and optionally evaluate against current pipeline.
    """
    try:
        tasks = json.loads(tasks_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid tasks_json")

    audio_bytes = await file.read()

    # Create gold standard
    gold = GoldStandard(
        meeting_ref=meeting_ref,
        transcript=transcript,
        transcript_source="manual",
        tasks_json=json.dumps(tasks, ensure_ascii=False),
        notes=f"Uploaded via API with file {file.filename}"
    )
    session.add(gold)
    session.commit()
    session.refresh(gold)

    # Optionally run evaluation now?
    # Could be separate step to avoid long wait

    return {"gold_id": gold.id, "status": "created", "tasks_count": len(tasks)}


@router.post("/evaluate/meeting/{meeting_id}")
def evaluate_meeting_endpoint(
    meeting_id: int,
    session: Session = Depends(get_session)
):
    """
    Evaluate a processed meeting against its gold standard (if exists).
    The gold standard must have meeting_ref equal to the meeting.id.
    Returns evaluation metrics.
    """
    meeting = session.get(Meeting, meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    try:
        run, metrics = evaluate_meeting(meeting=meeting, session=session)
        return {
            "evaluation_run_id": run.id if run else None,
            "overall_quality_score": run.overall_quality_score if run else None,
            "metrics": metrics
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
    session: Session = Depends(get_session)
):
    """List evaluation runs."""
    query = select(EvaluationRun)
    if meeting_id is not None:
        query = query.where(EvaluationRun.meeting_id == meeting_id)
    query = query.order_by(EvaluationRun.run_date.desc()).limit(limit)
    runs = session.exec(query).all()
    return {
        "evaluations": [
            {
                "id": r.id,
                "meeting_id": r.meeting_id,
                "gold_id": r.gold_standard_id,
                "overall_score": r.overall_quality_score,
                "model_whisper": r.model_whisper,
                "model_task": r.model_task,
                "run_date": r.run_date.isoformat() if r.run_date else None
            }
            for r in runs
        ]
    }


@router.get("/evaluation/{run_id}")
def get_evaluation_details(
    run_id: int,
    session: Session = Depends(get_session)
):
    """Get full evaluation details including all metrics."""
    run = session.get(EvaluationRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    details = json.loads(run.details_json) if run.details_json else {}
    # Fetch individual metrics
    metrics = session.exec(
        select(EvaluationMetric).where(EvaluationMetric.evaluation_run_id == run_id)
    ).all()
    details["individual_metrics"] = [
        {"name": m.metric_name, "value": m.value, "breakdown": json.loads(m.breakdown_json) if m.breakdown_json else None}
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
            "notes": run.notes
        },
        "metrics": details
    }