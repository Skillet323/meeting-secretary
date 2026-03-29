"""
API Routes with metrics, webhooks, and enhanced pipeline.
"""
import time
import json
import os
import httpx
from io import StringIO
import shutil
import tempfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, Response
from sqlmodel import Session, select
from ..db import init_db, get_session, engine
from ..models import Meeting, Task, Participant, Rule, ProcessingMetrics
from ..services.transcription import transcribe_from_bytes as transcribe
from ..services.task_extraction import extract_tasks as extract
from ..services.assignment_engine import assign_tasks_to_participants as assign
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


def _serialize_meeting(session: Session, meeting_id: int) -> dict:
    meeting = session.get(Meeting, meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="meeting not found")

    tasks = session.exec(select(Task).where(Task.meeting_id == meeting_id)).all()
    info = json.loads(meeting.info) if meeting.info else {}
    segments = info.pop("segments", [])
    return {
        "meeting": {
            "id": meeting.id,
            "transcript": meeting.transcript,
            "created_at": meeting.created_at.isoformat() if meeting.created_at else None,
        },
        "tasks": [
            {
                "id": task.id,
                "meeting_id": task.meeting_id,
                "description": task.description,
                "assignee": task.assignee,
                "deadline": task.deadline,
                "raw": task.raw,
                "status": task.status,
                "priority": task.priority,
            }
            for task in tasks
        ],
        "metadata": info,
        "segments": segments,
    }


# === Background Processing with Progress Tracking ===

def _update_meeting_progress(meeting_id: int, progress: int, stage: str, message: str = "", status: str = "processing"):
    """Update meeting info with progress data in database."""
    from ..db import Session as DBSession
    try:
        with DBSession(engine) as session:
            meeting = session.get(Meeting, meeting_id)
            if meeting:
                info = json.loads(meeting.info) if meeting.info else {}
                info.update({
                    "status": status,
                    "progress": progress,
                    "current_stage": stage,
                    "message": message
                })
                meeting.info = json.dumps(info, ensure_ascii=False)
                session.add(meeting)
                session.commit()
        logger.info(f"[Progress] Meeting {meeting_id}: {progress}% - {stage}: {message}")
    except Exception as e:
        logger.error(f"Failed to update progress for meeting {meeting_id}: {e}")


async def _process_meeting_background(meeting_id: int, audio_source: str, filename: str, audio_size_bytes: int | None = None):
    """Background task to process meeting with progress updates."""
    start_time = time.time()
    try:
        # Initialize progress
        _update_meeting_progress(meeting_id, 0, "Initializing", "Starting processing")

        # Step 1: Transcription
        _update_meeting_progress(meeting_id, 10, "Loading model", "Loading Whisper model...")
        transcribe_start = time.time()
        res = transcribe(audio_source, filename)
        transcript = res.get("text", "")
        segments = res.get("segments", [])
        language = res.get("language")
        confidence = res.get("confidence")
        transcribe_time = time.time() - transcribe_start
        logger.info(f"Transcription completed: {transcribe_time:.2f}s, {len(segments)} segments, language={language}")
        _update_meeting_progress(meeting_id, 60, "Transcription", f"Transcribed {len(segments)} segments")
        
        # Step 2: Task extraction
        _update_meeting_progress(meeting_id, 65, "Loading model", "Loading task extraction model...")
        task_start = time.time()
        tasks_list = await extract(transcript)
        task_time = time.time() - task_start
        logger.info(f"Task extraction completed: {len(tasks_list)} tasks, {task_time:.2f}s")
        _update_meeting_progress(meeting_id, 85, "Task extraction", f"Found {len(tasks_list)} tasks")
        
        # Step 3: Assignment
        _update_meeting_progress(meeting_id, 88, "Assignment", "Assigning tasks to participants")
        assign_start = time.time()
        from ..db import Session as DBSession
        with DBSession(engine) as session:
            assigned_tasks = assign(tasks_list, session)
        assign_time = time.time() - assign_start
        logger.info(f"Assignment completed: {assign_time:.2f}s")
        _update_meeting_progress(meeting_id, 95, "Assignment", "Tasks assigned")
        
        # Step 4: Save full results
        metadata = {
            "filename": filename,
            "audio_size_bytes": audio_size_bytes or 0,
            "language": language,
            "transcript_confidence": confidence,
            "transcribe_time_sec": transcribe_time,
            "task_time_sec": task_time,
            "assign_time_sec": assign_time,
            "segments_count": len(segments),
            "has_diarization": res.get("has_diarization", False),
            "model_whisper": os.environ.get("WHISPER_MODEL", "default"),
            "model_task": os.environ.get("TASK_MODEL", "default"),
            "segments": segments,  # Store segments for later retrieval
            "status": "completed",
            "progress": 100,
            "current_stage": "Completed",
            "message": "All done"
        }
        with DBSession(engine) as session:
            meeting = session.get(Meeting, meeting_id)
            if meeting:
                meeting.transcript = transcript
                meeting.info = json.dumps(metadata, ensure_ascii=False)
                session.add(meeting)
                # Save tasks
                for t in assigned_tasks:
                    task = Task(
                        meeting_id=meeting.id,
                        description=t["description"][:500],
                        assignee=t.get("assignee"),
                        deadline=t.get("deadline_hint"),
                        raw=json.dumps(t, ensure_ascii=False)
                    )
                    session.add(task)
                # Save metrics
                metrics = ProcessingMetrics(
                    meeting_id=meeting.id,
                    audio_size_bytes=audio_size_bytes or 0,
                    audio_duration_sec=segments[-1]["end"] if segments else 0,
                    transcribe_latency_sec=transcribe_time,
                    task_latency_sec=task_time,
                    assign_latency_sec=assign_time,
                    total_latency_sec=time.time() - start_time,
                    transcript_confidence=confidence,
                    segments_count=len(segments),
                    tasks_count=len(assigned_tasks),
                    language=language,
                    model_whisper=os.environ.get("WHISPER_MODEL", "default"),
                    model_task=os.environ.get("TASK_MODEL", "default"),
                    has_diarization=metadata["has_diarization"]
                )
                session.add(metrics)
                session.commit()
        total_time = time.time() - start_time
        logger.info(f"Meeting {meeting_id} fully processed in {total_time:.2f}s")
        
        # Trigger webhooks asynchronously
        try:
            await trigger_webhooks(meeting_id, assigned_tasks)
        except Exception as e:
            logger.warning(f"Webhook trigger failed: {e}")
            
    except Exception as e:
        logger.exception(f"Processing failed for meeting {meeting_id}: {e}")
        # Update status to failed
        _update_meeting_progress(meeting_id, 0, "Error", str(e), status="failed")
    finally:
        if audio_source and os.path.exists(audio_source):
            try:
                os.unlink(audio_source)
            except OSError:
                pass


@router.on_event("startup")
def on_startup():
    init_db()


# === Health & Metrics ===
@router.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@router.get("/meetings")
def list_meetings(limit: int = Query(20, ge=1, le=100), session: Session = Depends(get_session)):
    meetings = session.exec(select(Meeting).order_by(Meeting.created_at.desc()).limit(limit)).all()
    return {"meetings": [_serialize_meeting(session, meeting.id) for meeting in meetings if meeting.id is not None]}


@router.get("/metrics")
def get_metrics(limit: int = Query(10, ge=1, le=100), session: Session = Depends(get_session)):
    """Recent processing metrics (latency, quality scores)."""
    metrics = session.exec(select(ProcessingMetrics).order_by(ProcessingMetrics.created_at.desc()).limit(limit)).all()
    return {
        "metrics": [
            {
                "id": m.id,
                "meeting_id": m.meeting_id,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "audio_size_bytes": m.audio_size_bytes,
                "audio_duration_sec": m.audio_duration_sec,
                "transcribe_latency_sec": m.transcribe_latency_sec,
                "task_latency_sec": m.task_latency_sec,
                "assign_latency_sec": m.assign_latency_sec,
                "total_latency_sec": m.total_latency_sec,
                "transcript_confidence": m.transcript_confidence,
                "segments_count": m.segments_count,
                "tasks_count": m.tasks_count,
                "language": m.language,
                "model_whisper": m.model_whisper,
                "model_task": m.model_task,
                "has_diarization": m.has_diarization,
            }
            for m in metrics
        ]
    }


# === Meeting Upload & Processing ===
@router.post("/upload_meeting")
async def upload_meeting(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    background_tasks: BackgroundTasks = None,
):
    """
    Upload audio file for asynchronous processing.
    Returns immediate response with meeting_id; use /meeting/{id}/progress to track status.
    """
    temp_audio_path = None
    try:
        suffix = Path(file.filename or "upload.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_audio_path = tmp.name

        audio_size_bytes = os.path.getsize(temp_audio_path)
        filename = file.filename or Path(temp_audio_path).name

        # Create meeting record with initial status
        initial_info = {
            "filename": filename,
            "audio_size_bytes": audio_size_bytes,
            "status": "processing",
            "progress": 0,
            "current_stage": "Uploading",
            "message": "File received, queued for processing",
        }
        meeting = Meeting(transcript="", info=json.dumps(initial_info, ensure_ascii=False))
        session.add(meeting)
        session.commit()
        session.refresh(meeting)

        # Kick off background processing
        if background_tasks:
            background_tasks.add_task(_process_meeting_background, meeting.id, temp_audio_path, filename, audio_size_bytes)
        else:
            logger.warning("No background_tasks; running processing synchronously")
            await _process_meeting_background(meeting.id, temp_audio_path, filename, audio_size_bytes)

        return {"meeting_id": meeting.id, "status": "processing", "progress": 0, "message": "Upload successful, processing started"}

    except Exception as e:
        logger.exception("Upload failed")
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except OSError:
                pass
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@router.get("/meeting/{meeting_id}")
def get_meeting(meeting_id: int, session: Session = Depends(get_session)):
    return _serialize_meeting(session, meeting_id)


@router.get("/meeting/{meeting_id}/progress")
def get_meeting_progress(meeting_id: int, session: Session = Depends(get_session)):
    """Get current processing progress for a meeting."""
    meeting = session.get(Meeting, meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    info = json.loads(meeting.info) if meeting.info else {}
    return {
        "meeting_id": meeting_id,
        "status": info.get("status", "unknown"),
        "progress": info.get("progress", 0),
        "current_stage": info.get("current_stage", ""),
        "message": info.get("message", "")
    }


@router.get("/meeting/{meeting_id}/export")
def export_meeting(
    meeting_id: int,
    format: str = Query("json", pattern="^(json|csv|txt|md)$"),
    session: Session = Depends(get_session)
):
    """Export meeting data in various formats."""
    meeting = session.get(Meeting, meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="meeting not found")

    tasks = session.exec(select(Task).where(Task.meeting_id == meeting_id)).all()
    metadata = json.loads(meeting.info) if meeting.info else {}

    if format == "json":
        # Convert datetime to ISO string
        meeting_data = {
            "id": meeting.id,
            "transcript": meeting.transcript,
            "created_at": meeting.created_at.isoformat() if meeting.created_at else None
        }
        return JSONResponse({
            "meeting": meeting_data,
            "tasks": [{"description": t.description, "assignee": t.assignee, "deadline": t.deadline} for t in tasks],
            "metadata": metadata
        })

    elif format == "txt":
        content = f"Meeting #{meeting_id}\n"
        content += f"Date: {meeting.created_at}\n\n"
        content += "Transcript:\n" + meeting.transcript + "\n\n"
        content += "Tasks:\n"
        for i, t in enumerate(tasks, 1):
            content += f"{i}. {t.description}\n   Assignee: {t.assignee or 'Unassigned'}\n   Deadline: {t.deadline or 'None'}\n\n"
        return Response(content, media_type="text/plain")

    elif format == "md":
        content = f"# Meeting #{meeting_id}\n\n"
        content += f"**Date:** {meeting.created_at}\n\n"
        content += "## Transcript\n\n" + meeting.transcript + "\n\n"
        content += "## Tasks\n\n"
        for i, t in enumerate(tasks, 1):
            content += f"{i}. **{t.description}**\n"
            content += f"   - Assignee: {t.assignee or 'Unassigned'}\n"
            content += f"   - Deadline: {t.deadline or 'None'}\n\n"
        return Response(content, media_type="text/markdown")

    elif format == "csv":
        import csv, io
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Task", "Assignee", "Deadline"])
        for t in tasks:
            writer.writerow([t.description, t.assignee or "", t.deadline or ""])
        return Response(output.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f"attachment; filename=meeting-{meeting_id}.csv"})


# === Webhook Configuration ===
WEBHOOKS_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "webhooks.json")


@router.post("/webhooks")
def configure_webhook(url: str, events: list = ["meeting_completed"]):
    """Register a webhook URL to receive notifications."""
    try:
        if os.path.exists(WEBHOOKS_FILE):
            with open(WEBHOOKS_FILE, "r") as f:
                hooks = json.load(f)
        else:
            hooks = []
        hooks.append({"url": url, "events": events, "created_at": time.time()})
        with open(WEBHOOKS_FILE, "w") as f:
            json.dump(hooks, f, indent=2)
        return {"ok": True, "webhooks": len(hooks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def trigger_webhooks(meeting_id: int, tasks: list):
    """Send webhook notifications for completed meetings."""
    try:
        if not os.path.exists(WEBHOOKS_FILE):
            return
        with open(WEBHOOKS_FILE, "r") as f:
            hooks = json.load(f)
        async with httpx.AsyncClient(timeout=10.0) as client:
            for hook in hooks:
                if "meeting_completed" in hook["events"]:
                    await client.post(
                        hook["url"],
                        json={
                            "event": "meeting_completed",
                            "meeting_id": meeting_id,
                            "tasks_count": len(tasks),
                            "tasks": tasks[:10]  # Limit size
                        }
                    )
    except Exception as e:
        logger.warning(f"Webhook failed: {e}")


# === Participant CRUD (unchanged) ===
@router.post("/participant")
def create_participant(p: Participant, session: Session = Depends(get_session)):
    session.add(p)
    session.commit()
    session.refresh(p)
    return p

@router.get("/participants")
def list_participants(session: Session = Depends(get_session)):
    return session.exec(select(Participant)).all()

@router.put("/participant/{participant_id}")
def update_participant(participant_id: int, p: Participant, session: Session = Depends(get_session)):
    db_p = session.get(Participant, participant_id)
    if not db_p:
        raise HTTPException(status_code=404, detail="participant not found")
    db_p.name = p.name
    db_p.email = p.email
    db_p.role = p.role
    db_p.tags = p.tags
    session.add(db_p)
    session.commit()
    session.refresh(db_p)
    return db_p

@router.delete("/participant/{participant_id}")
def delete_participant(participant_id: int, session: Session = Depends(get_session)):
    p = session.get(Participant, participant_id)
    if not p:
        raise HTTPException(status_code=404, detail="participant not found")
    session.delete(p)
    session.commit()
    return {"ok": True}


# === Rule CRUD (unchanged) ===
@router.post("/rule")
def create_rule(r: Rule, session: Session = Depends(get_session)):
    session.add(r)
    session.commit()
    session.refresh(r)
    return r

@router.get("/rules")
def list_rules(session: Session = Depends(get_session)):
    return session.exec(select(Rule).order_by(Rule.priority)).all()

@router.put("/rule/{rule_id}")
def update_rule(rule_id: int, r: Rule, session: Session = Depends(get_session)):
    db_r = session.get(Rule, rule_id)
    if not db_r:
        raise HTTPException(status_code=404, detail="rule not found")
    db_r.name = r.name
    db_r.kind = r.kind
    db_r.pattern = r.pattern
    db_r.priority = r.priority
    session.add(db_r)
    session.commit()
    session.refresh(db_r)
    return db_r

@router.delete("/rule/{rule_id}")
def delete_rule(rule_id: int, session: Session = Depends(get_session)):
    r = session.get(Rule, rule_id)
    if not r:
        raise HTTPException(status_code=404, detail="rule not found")
    session.delete(r)
    session.commit()
    return {"ok": True}