from sqlmodel import SQLModel, Field, Relationship
from typing import List, Optional
from datetime import datetime
import json

class Participant(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    email: Optional[str] = None
    role: Optional[str] = None
    tags: Optional[str] = None  # comma separated tags

class Rule(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    kind: str  # explicit_mention | regex | role_lookup | fallback
    pattern: Optional[str] = None  # for regex or role mapping (json)
    priority: int = 100

class Meeting(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    transcript: Optional[str] = None
    info: Optional[str] = None  # JSON metadata like participants list, original filename

class Task(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    meeting_id: Optional[int] = Field(default=None, foreign_key="meeting.id")
    description: str
    assignee: Optional[str] = None
    deadline: Optional[str] = None
    raw: Optional[str] = None  # raw extraction json
    status: str = "pending"  # pending, done, blocked
    priority: int = 1  # 1=high, 2=medium, 3=low

class ProcessingMetrics(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    meeting_id: Optional[int] = Field(default=None, foreign_key="meeting.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    audio_size_bytes: int
    audio_duration_sec: float
    transcribe_latency_sec: float
    task_latency_sec: float
    assign_latency_sec: float
    total_latency_sec: float
    transcript_confidence: Optional[float] = None
    segments_count: int
    tasks_count: int
    language: Optional[str] = None
    model_whisper: str
    model_task: str
    has_diarization: bool = False

class GoldStandard(SQLModel, table=True):
    """Reference transcript and tasks for evaluation."""
    id: Optional[int] = Field(default=None, primary_key=True)
    meeting_ref: str = Field(unique=True)  # e.g., "ami_001" or meeting_id as string
    transcript: str
    transcript_source: Optional[str] = None  # "manual", "ami", "icsi"
    tasks_json: str  # JSON array of gold tasks
    audio_file_path: Optional[str] = None
    language: str = "en"
    duration_sec: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    notes: Optional[str] = None

class EvaluationRun(SQLModel, table=True):
    """A single evaluation run (comparing system output to gold)."""
    id: Optional[int] = Field(default=None, primary_key=True)
    meeting_id: Optional[int] = Field(default=None, foreign_key="meeting.id")
    gold_standard_id: Optional[int] = Field(default=None, foreign_key="goldstandard.id")
    run_date: datetime = Field(default_factory=datetime.utcnow)
    model_whisper: str
    model_task: str
    overall_quality_score: Optional[float] = None  # 0-100 weighted score
    details_json: str  # Full metrics JSON
    notes: Optional[str] = None

class EvaluationMetric(SQLModel, table=True):
    """Individual metric value for an evaluation run."""
    id: Optional[int] = Field(default=None, primary_key=True)
    evaluation_run_id: int = Field(foreign_key="evaluationrun.id")
    metric_name: str
    value: float
    breakdown_json: Optional[str] = None  # Per-task or per-field details
    created_at: datetime = Field(default_factory=datetime.utcnow)