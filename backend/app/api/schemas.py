from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class EvaluationRunSummary(BaseModel):
    id: int
    meeting_id: int
    gold_id: int
    overall_score: float
    model_whisper: str
    model_task: str
    run_date: Optional[str] = None


class EvaluationMetricBase(BaseModel):
    metric_name: str
    value: float


class EvaluationDetail(BaseModel):
    run: EvaluationRunSummary
    metrics: List[EvaluationMetricBase]


class EvaluationList(BaseModel):
    evaluations: List[EvaluationRunSummary]
