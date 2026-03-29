from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Tuple, Optional, Any

from sqlmodel import Session, select

from ..models import GoldStandard, EvaluationRun, EvaluationMetric, Meeting, Task

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def evaluate_transcription(reference: str, hypothesis: str) -> Tuple[float, float]:
    """Return (WER, CER). Uses jiwer if available, otherwise a conservative fallback."""
    ref = reference or ""
    hyp = hypothesis or ""

    try:
        from jiwer import wer, cer
        return float(wer(ref, hyp)), float(cer(ref, hyp))
    except Exception:
        # Fallback: normalized token/character coverage approximation
        ref_n = normalize_text(ref)
        hyp_n = normalize_text(hyp)
        if not ref_n:
            return 1.0, 1.0
        ref_tokens = ref_n.split()
        hyp_tokens = set(hyp_n.split())
        token_cov = len(set(ref_tokens) & hyp_tokens) / max(1, len(set(ref_tokens)))
        ref_chars = set(ref_n)
        hyp_chars = set(hyp_n)
        char_cov = len(ref_chars & hyp_chars) / max(1, len(ref_chars))
        return 1.0 - token_cov, 1.0 - char_cov


def _task_desc(task: dict) -> str:
    return normalize_text(str(task.get("description", "")))


def _task_name_like(text: str) -> str:
    return normalize_text(text)


def _best_match_by_jaccard(pred_desc: str, gold_tasks: List[dict], used: set[int]) -> Tuple[int, float]:
    pred_tokens = set(pred_desc.split())
    best_idx, best_score = -1, 0.0
    if not pred_tokens:
        return best_idx, best_score

    for i, gold in enumerate(gold_tasks):
        if i in used:
            continue
        gold_tokens = set(_task_desc(gold).split())
        if not gold_tokens:
            continue
        inter = len(pred_tokens & gold_tokens)
        union = len(pred_tokens | gold_tokens)
        score = inter / union if union else 0.0
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx, best_score


def evaluate_tasks(pred_tasks: List[dict], gold_tasks: List[dict]) -> Dict[str, Any]:
    """
    Compute task-set precision/recall/F1 plus assignee/deadline accuracy.
    """
    pred_tasks = pred_tasks or []
    gold_tasks = gold_tasks or []

    matches: Dict[int, int] = {}
    used_gold: set[int] = set()

    for i, pred in enumerate(pred_tasks):
        idx, score = _best_match_by_jaccard(_task_desc(pred), gold_tasks, used_gold)
        if idx >= 0 and score >= 0.35:
            matches[i] = idx
            used_gold.add(idx)

    tp = len(matches)
    fp = max(0, len(pred_tasks) - tp)
    fn = max(0, len(gold_tasks) - tp)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    assignee_correct = 0
    deadline_correct = 0
    for pred_idx, gold_idx in matches.items():
        pred = pred_tasks[pred_idx]
        gold = gold_tasks[gold_idx]
        if _task_name_like(pred.get("assignee_hint", "")) == _task_name_like(gold.get("assignee_hint", "")):
            assignee_correct += 1
        if _task_name_like(pred.get("deadline_hint", "")) == _task_name_like(gold.get("deadline_hint", "")):
            deadline_correct += 1

    assignee_accuracy = assignee_correct / tp if tp else None
    deadline_accuracy = deadline_correct / tp if tp else None

    # Hallucination proxy: very short descriptions are often bad extractions.
    hallucination_rate = sum(
        1 for t in pred_tasks if len(_task_desc(t).split()) < 3
    ) / max(1, len(pred_tasks))

    return {
        "task_set_f1": f1,
        "task_set_precision": precision,
        "task_set_recall": recall,
        "assignee_accuracy": assignee_accuracy,
        "deadline_accuracy": deadline_accuracy,
        "hallucination_rate": hallucination_rate,
        "predicted_tasks": len(pred_tasks),
        "gold_tasks": len(gold_tasks),
        "matched_tasks": tp,
    }


def evaluate_meeting(meeting: Meeting, session: Session) -> tuple[EvaluationRun, dict]:
    """
    Evaluate processed meeting against a gold standard with meeting_ref == meeting.id.
    Creates EvaluationRun and EvaluationMetric rows.
    """
    gold = session.exec(
        select(GoldStandard).where(GoldStandard.meeting_ref == str(meeting.id))
    ).first()
    if not gold:
        raise ValueError(f"No gold standard found for meeting_id={meeting.id}")

    try:
        gold_tasks = json.loads(gold.tasks_json) if gold.tasks_json else []
    except Exception:
        gold_tasks = []

    pred_tasks = session.exec(
        select(__import__("..models", fromlist=["Task"]).Task).where(
            __import__("..models", fromlist=["Task"]).Task.meeting_id == meeting.id
        )
    ).all()

    pred_task_dicts = [
        {
            "description": t.description,
            "assignee_hint": t.assignee,
            "deadline_hint": t.deadline,
        }
        for t in pred_tasks
    ]

    wer, cer = evaluate_transcription(gold.transcript, meeting.transcript or "")
    task_metrics = evaluate_tasks(pred_task_dicts, gold_tasks)

    assign_acc = task_metrics.get("assignee_accuracy") or 0.0
    deadline_acc = task_metrics.get("deadline_accuracy") or 0.0
    task_f1 = task_metrics.get("task_set_f1", 0.0)

    overall = (task_f1 * 0.6 + assign_acc * 0.2 + deadline_acc * 0.2) * 100.0

    details = {
        "wer": wer,
        "cer": cer,
        **task_metrics,
        "overall_quality_score": overall,
    }

    run = EvaluationRun(
        meeting_id=meeting.id,
        gold_standard_id=gold.id,
        model_whisper="unknown",
        model_task="unknown",
        overall_quality_score=overall,
        details_json=json.dumps(details, ensure_ascii=False),
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    for metric_name, value in {
        "wer": wer,
        "cer": cer,
        "task_set_f1": task_f1,
        "task_set_precision": task_metrics["task_set_precision"],
        "task_set_recall": task_metrics["task_set_recall"],
        "assignee_accuracy": assignee_acc,
        "deadline_accuracy": deadline_acc,
        "hallucination_rate": task_metrics["hallucination_rate"],
        "overall_quality_score": overall,
    }.items():
        if value is None:
            continue
        session.add(
            EvaluationMetric(
                evaluation_run_id=run.id,
                metric_name=metric_name,
                value=float(value),
            )
        )

    session.commit()
    return run, details


def load_gold_dataset(gold_dir: str) -> list[dict]:
    from pathlib import Path
    gold_path = Path(gold_dir)
    data = []
    for json_file in gold_path.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            item = json.load(f)
            item["_source_file"] = json_file.name
            data.append(item)
    return data


def create_gold_standard(session: Session, data: dict):
    gold = GoldStandard(
        meeting_ref=data["meeting_ref"],
        transcript=data["transcript"],
        transcript_source=data.get("source", "manual"),
        tasks_json=json.dumps(data.get("tasks", []), ensure_ascii=False),
        audio_file_path=data.get("audio_file_path"),
        language=data.get("language", "en"),
        duration_sec=data.get("duration_sec"),
        notes=data.get("notes"),
    )
    session.add(gold)
    session.commit()
    session.refresh(gold)
    return gold
