from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlmodel import Session, select

from ..models import GoldStandard, EvaluationRun, EvaluationMetric, Meeting, Task, ProcessingMetrics

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "have", "has", "had",
    "he", "her", "him", "his", "i", "if", "in", "into", "is", "it", "its", "me", "my",
    "of", "on", "or", "our", "she", "so", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "to", "we", "were", "what", "when", "where", "which", "who",
    "will", "with", "would", "you", "your", "the", "um", "uh", "okay", "right", "yeah",
}

def _transcript_quality_score(wer: float, cer: float) -> float:
    # simple conservative score in [0..100]
    return max(0.0, 100.0 * (1.0 - (0.7 * wer + 0.3 * cer)))

def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokens(text: str) -> List[str]:
    return [t for t in normalize_text(text).split() if t and t not in _STOPWORDS]


def _sim(a: str, b: str) -> float:
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    if not a_n or not b_n:
        return 0.0
    seq = SequenceMatcher(None, a_n, b_n).ratio()
    a_tokens = set(_tokens(a_n))
    b_tokens = set(_tokens(b_n))
    if not a_tokens or not b_tokens:
        return seq
    jacc = len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))
    return 0.55 * jacc + 0.45 * seq


def evaluate_transcription(reference: str, hypothesis: str) -> Tuple[float, float]:
    """Return (WER, CER). Uses jiwer if available, otherwise a conservative fallback."""
    ref = reference or ""
    hyp = hypothesis or ""

    try:
        from jiwer import cer, wer
        return float(wer(ref, hyp)), float(cer(ref, hyp))
    except Exception:
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


def _best_match(pred_desc: str, gold_tasks: List[dict], used: set[int]) -> Tuple[int, float]:
    best_idx, best_score = -1, 0.0
    for i, gold in enumerate(gold_tasks):
        if i in used:
            continue
        gold_desc = _task_desc(gold)
        score = _sim(pred_desc, gold_desc)
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx, best_score

def evaluate_tasks(pred_tasks: List[dict], gold_tasks: List[dict]) -> Dict[str, Any]:
    """Compute task-set precision/recall/F1 plus assignee/deadline accuracy."""
    pred_tasks = pred_tasks or []
    gold_tasks = gold_tasks or []

    matches: Dict[int, int] = {}
    used_gold: set[int] = set()

    for i, pred in enumerate(pred_tasks):
        idx, score = _best_match(_task_desc(pred), gold_tasks, used_gold)
        if idx >= 0 and score >= 0.28:
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
    assignment_checks = 0
    deadline_checks = 0
    matched_details: List[Dict[str, Any]] = []

    for pred_idx, gold_idx in matches.items():
        pred = pred_tasks[pred_idx]
        gold = gold_tasks[gold_idx]
        pred_assignee = _task_name_like(pred.get("assignee_hint", "") or pred.get("assignee", ""))
        gold_assignee = _task_name_like(gold.get("assignee_hint", "") or gold.get("assignee", ""))
        pred_deadline = _task_name_like(pred.get("deadline_hint", "") or pred.get("deadline", ""))
        gold_deadline = _task_name_like(gold.get("deadline_hint", "") or gold.get("deadline", ""))

        if pred_assignee or gold_assignee:
            assignment_checks += 1
            if not pred_assignee and not gold_assignee:
                assignee_correct += 1
            elif pred_assignee and gold_assignee and (pred_assignee == gold_assignee or pred_assignee in gold_assignee or gold_assignee in pred_assignee):
                assignee_correct += 1

        if pred_deadline or gold_deadline:
            deadline_checks += 1
            if not pred_deadline and not gold_deadline:
                deadline_correct += 1
            elif pred_deadline and gold_deadline and (pred_deadline == gold_deadline or pred_deadline in gold_deadline or gold_deadline in pred_deadline):
                deadline_correct += 1

        matched_details.append(
            {
                "pred_idx": pred_idx,
                "gold_idx": gold_idx,
                "similarity": round(_sim(_task_desc(pred), _task_desc(gold)), 3),
            }
        )

    assignee_accuracy = assignee_correct / assignment_checks if assignment_checks else None
    deadline_accuracy = deadline_correct / deadline_checks if deadline_checks else None

    hallucination_rate = sum(1 for t in pred_tasks if len(_task_desc(t).split()) < 3) / max(1, len(pred_tasks))

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
        "assignment_checks": assignment_checks,
        "deadline_checks": deadline_checks,
        "matched_details": matched_details,
    }


def meeting_ref_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    return re.sub(r"\.Mix-Headset$", "", stem)


def _meeting_ref_candidates(meeting: Meeting) -> List[str]:
    candidates: List[str] = []
    info = json.loads(meeting.info) if meeting.info else {}
    filename = info.get("filename", "")
    if filename:
        candidates.append(meeting_ref_from_filename(filename))
    if meeting.id is not None:
        candidates.append(str(meeting.id))
    if meeting.info:
        # Sometimes the imported gold uses a source key.
        for key in ("source_key", "meeting_ref", "gold_ref"):
            if info.get(key):
                candidates.append(str(info[key]))
    # preserve order, unique
    out = []
    seen = set()
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _find_gold_for_meeting(session: Session, meeting: Meeting) -> Optional[GoldStandard]:
    for candidate in _meeting_ref_candidates(meeting):
        gold = session.exec(select(GoldStandard).where(GoldStandard.meeting_ref == candidate)).first()
        if gold:
            return gold
    return None


def evaluate_meeting(meeting: Meeting, session: Session) -> tuple[EvaluationRun, dict]:
    """Evaluate a processed meeting against the available gold standard."""
    meeting_info = json.loads(meeting.info) if meeting.info else {}
    gold = _find_gold_for_meeting(session, meeting)
    if not gold:
        raise ValueError(f"No gold standard found for meeting candidates={_meeting_ref_candidates(meeting)}")

    try:
        gold_tasks = json.loads(gold.tasks_json) if gold.tasks_json else []
    except Exception:
        gold_tasks = []

    pred_tasks = session.exec(select(Task).where(Task.meeting_id == meeting.id)).all()
    pred_task_dicts = [
        {
            "description": t.description,
            "assignee_hint": t.assignee,
            "deadline_hint": t.deadline,
            "raw": t.raw,
        }
        for t in pred_tasks
    ]

    wer, cer = evaluate_transcription(gold.transcript, meeting.transcript or "")
    
    task_metrics = evaluate_tasks(pred_task_dicts, gold_tasks)

    assign_acc = float(task_metrics.get("assignee_accuracy") or 0.0)
    deadline_acc = float(task_metrics.get("deadline_accuracy") or 0.0)
    task_f1 = float(task_metrics.get("task_set_f1", 0.0))
    transcript_quality = _transcript_quality_score(wer, cer)

    # A blended score that does not collapse to zero when task matching is still immature.
    overall = (transcript_quality * 0.45 + task_f1 * 0.35 + assign_acc * 0.10 + deadline_acc * 0.10) * 100.0

    latest_metrics = session.exec(
        select(ProcessingMetrics)
        .where(ProcessingMetrics.meeting_id == meeting.id)
        .order_by(ProcessingMetrics.created_at.desc())
    ).first()

    model_whisper = meeting_info.get("model_whisper") or (latest_metrics.model_whisper if latest_metrics else "unknown")
    model_task = meeting_info.get("model_task") or (latest_metrics.model_task if latest_metrics else "unknown")

    details = {
        "wer": wer,
        "cer": cer,
        "transcript_quality": transcript_quality,
        **task_metrics,
        "overall_quality_score": overall,
        "task_provider": meeting_info.get("task_provider"),
        "task_model": meeting_info.get("task_model"),
        "task_parse_stage": meeting_info.get("task_parse_stage"),
        "task_fallback_used": meeting_info.get("task_fallback_used"),
        "task_fallback_merged": meeting_info.get("task_fallback_merged"),
        "has_diarization": meeting_info.get("has_diarization"),
        "speaker_aliases": meeting_info.get("speaker_aliases", {}),
    }

    run = EvaluationRun(
        meeting_id=meeting.id,
        gold_standard_id=gold.id,
        model_whisper=model_whisper,
        model_task=model_task,
        overall_quality_score=overall,
        details_json=json.dumps(details, ensure_ascii=False),
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    metric_values = {
        "wer": wer,
        "cer": cer,
        "transcript_quality": transcript_quality,
        "task_set_f1": task_f1,
        "task_set_precision": task_metrics["task_set_precision"],
        "task_set_recall": task_metrics["task_set_recall"],
        "assignee_accuracy": assign_acc if task_metrics.get("assignment_checks") else None,
        "deadline_accuracy": deadline_acc if task_metrics.get("deadline_checks") else None,
        "hallucination_rate": task_metrics["hallucination_rate"],
        "overall_quality_score": overall,
    }

    for metric_name, value in metric_values.items():
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
        transcript_source=data.get("transcript_source") or data.get("source", "manual"),
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
