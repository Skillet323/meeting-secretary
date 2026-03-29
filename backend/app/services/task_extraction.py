# Task extraction service using T5, with a strong rule-based fallback.
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ..config import settings

logger = logging.getLogger(__name__)

_tokenizer = None
_model = None


def load_local_model():
    """Lazy-load T5 model."""
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    logger.info("[TASK] Loading model: %s", settings.TASK_MODEL)
    _tokenizer = AutoTokenizer.from_pretrained(settings.TASK_MODEL)
    _model = AutoModelForSeq2SeqLM.from_pretrained(settings.TASK_MODEL)

    device = "cuda" if torch.cuda.is_available() and settings.WHISPER_DEVICE == "cuda" else "cpu"
    _model = _model.to(device)
    _model.eval()

    logger.info("[TASK] Model loaded on %s", device.upper())
    return _tokenizer, _model


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _guess_deadline(sentence: str) -> Optional[str]:
    m = re.search(
        r"(?:by|до|к|к\s+)(\d{1,2}(?:[./-]\d{1,2})?(?:[./-]\d{2,4})?|"
        r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"понедельник|вторник|среда|четверг|пятница|суббота|воскресенье))",
        sentence,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1)
    return None


def _guess_assignee(sentence: str) -> Optional[str]:
    # Patterns like "for Ivan", "to Maria", "Ивану", "Марии", "@name"
    patterns = [
        r"(?:for|to|assigned to|by)\s+([A-ZА-ЯЁ][a-zа-яё]+(?:\s+[A-ZА-ЯЁ][a-zа-яё]+)?)",
        r"(?:для|от|к)\s+([A-ZА-ЯЁ][a-zа-яё]+(?:\s+[A-ZА-ЯЁ][a-zа-яё]+)?)",
        r"@([\w.-]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, sentence)
        if m:
            return _normalize_space(m.group(1))
    return None


def _looks_like_task(sentence: str) -> bool:
    markers = [
        "should", "must", "need to", "needs to", "please", "action item", "task",
        "to do", "follow up", "let's", "let us", "will", "have to", "required to",
        "нужно", "надо", "нужна", "нужен", "сделать", "подготовить", "проверить",
        "отправить", "согласовать", "обновить", "выполнить", "завершить", "доработать",
        "назначить", "созвониться", "позвонить",
    ]
    s = sentence.lower()
    return any(marker in s for marker in markers)


def _extract_tasks_simple(transcript: str) -> List[Dict[str, Any]]:
    """Heuristic extraction for transcripts when the LLM is unavailable or fails."""
    tasks: List[Dict[str, Any]] = []
    sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", transcript)

    for raw_sentence in sentences:
        sentence = _normalize_space(raw_sentence)
        if not sentence or len(sentence) < 12:
            continue
        if not _looks_like_task(sentence):
            continue

        # Trim obvious lead-ins.
        sentence = re.sub(
            r"^(?:ну|okay|ok|please|let's|we should|we need to|нужно|надо)\s*,?\s*",
            "",
            sentence,
            flags=re.IGNORECASE,
        )

        description = sentence[:500]
        assignee = _guess_assignee(sentence)
        deadline = _guess_deadline(sentence)

        tasks.append({
            "description": description,
            "assignee_hint": assignee,
            "deadline_hint": deadline,
            "source": "rule_based",
        })

    # Deduplicate by normalized description prefix
    unique = []
    seen = set()
    for task in tasks:
        key = _normalize_space(task["description"].lower())[:120]
        if key not in seen:
            seen.add(key)
            unique.append(task)
    return unique


def _parse_generated_tasks(generated: str) -> List[Dict[str, Any]]:
    """Try to parse JSON array from model output, with a few recovery strategies."""
    candidates = []

    start = generated.find("[")
    end = generated.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidates.append(generated[start:end + 1])

    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", generated, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            tasks: List[Dict[str, Any]] = []
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        desc = str(item.get("description") or item.get("task") or "").strip()
                        if desc:
                            tasks.append({
                                "description": desc[:500],
                                "assignee_hint": item.get("assignee_hint") or item.get("assignee"),
                                "deadline_hint": item.get("deadline_hint") or item.get("deadline"),
                                "source": "llm",
                            })
            if tasks:
                return tasks
        except Exception:
            continue

    tasks = []
    for line in generated.splitlines():
        line = line.strip().lstrip("-•*0123456789. )")
        if len(line) > 15 and _looks_like_task(line):
            tasks.append({
                "description": line[:500],
                "assignee_hint": _guess_assignee(line),
                "deadline_hint": _guess_deadline(line),
                "source": "llm_text",
            })
    return tasks


async def extract_tasks(transcript: str) -> List[Dict[str, Any]]:
    """
    Extract tasks using T5 where possible, with a reliable heuristic fallback.
    """
    transcript = (transcript or "").strip()
    if not transcript:
        return []

    fallback_tasks = _extract_tasks_simple(transcript)

    try:
        tokenizer, model = load_local_model()
    except Exception as e:
        logger.warning("[TASK] Model load failed, using rule-based fallback: %s", e)
        return fallback_tasks

    prompt = (
        "Extract action items from this meeting transcript. "
        "Return ONLY a JSON array. Each item must contain keys: "
        "description, assignee_hint, deadline_hint. "
        "Use null for unknown values. "
        "Do not add commentary.\n\n"
        f"Transcript:\n{transcript[:3000]}"
    )

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=min(settings.MAX_TASK_MODEL_TOKENS, 1024),
            truncation=True,
            padding=False,
        )

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                num_beams=2,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        logger.info("[TASK] Raw output (first 400 chars): %s", generated[:400])

        tasks = _parse_generated_tasks(generated)

        if not tasks:
            return fallback_tasks

        if len(tasks) < 2 and fallback_tasks:
            combined = tasks + fallback_tasks
        else:
            combined = tasks

        unique = []
        seen = set()
        for t in combined:
            key = _normalize_space((t.get("description") or "").lower())[:120]
            if key and key not in seen:
                seen.add(key)
                unique.append(t)

        return unique[:20]

    except Exception as e:
        logger.exception("[TASK] Extraction failed, falling back to heuristic rules: %s", e)
        return fallback_tasks


def extract_tasks_rule_based(transcript: str) -> List[Dict[str, Any]]:
    """Compatibility wrapper for rule-based extraction."""
    return _extract_tasks_simple(transcript)
