# backend/app/services/task_extraction.py
"""
Task extraction via OpenRouter (free tier) with a robust rule-based fallback.

Model strategy
--------------
Primary  : OpenRouter free router  → set OPENROUTER_API_KEY + OPENROUTER_TASK_MODEL=openrouter/free
           When you want reproducibility, pin to a specific free model, e.g.:
               meta-llama/llama-3.3-70b-instruct:free
               google/gemma-3-27b-it:free
               mistralai/mistral-7b-instruct:free
           The `response.model` field in each reply tells you which model
           was actually used — handy for later evaluation.

Fallback : Pure regex / heuristic extraction (no network, always works).

Environment variables (.env)
----------------------------
TASK_PROVIDER          = openrouter          # or "rules" to skip API entirely
OPENROUTER_API_KEY     = sk-or-...
OPENROUTER_TASK_MODEL  = openrouter/free     # or any :free model slug

GitHub Codespaces note
----------------------
Add OPENROUTER_API_KEY to repo/codespace secrets; no GPU or local weights needed.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from ..config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers shared by both paths
# ---------------------------------------------------------------------------

def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _guess_deadline(sentence: str) -> Optional[str]:
    m = re.search(
        r"(?:by|до|к|к\s+)"
        r"(\d{1,2}(?:[./-]\d{1,2})?(?:[./-]\d{2,4})?|"
        r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"понедельник|вторник|среда|четверг|пятница|суббота|воскресенье))",
        sentence,
        flags=re.IGNORECASE,
    )
    return m.group(1) if m else None


def _guess_assignee(sentence: str) -> Optional[str]:
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


# ---------------------------------------------------------------------------
# Rule-based fallback (always available)
# ---------------------------------------------------------------------------

def _extract_tasks_simple(transcript: str) -> List[Dict[str, Any]]:
    """Heuristic extraction — no external dependencies."""
    tasks: List[Dict[str, Any]] = []
    sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", transcript)

    for raw in sentences:
        sentence = _normalize_space(raw)
        if not sentence or len(sentence) < 12:
            continue
        if not _looks_like_task(sentence):
            continue

        sentence = re.sub(
            r"^(?:ну|okay|ok|please|let's|we should|we need to|нужно|надо)\s*,?\s*",
            "",
            sentence,
            flags=re.IGNORECASE,
        )

        tasks.append({
            "description": sentence[:500],
            "assignee_hint": _guess_assignee(sentence),
            "deadline_hint": _guess_deadline(sentence),
            "source": "rule_based",
        })

    # Deduplicate
    unique: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for t in tasks:
        key = _normalize_space(t["description"].lower())[:120]
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


# ---------------------------------------------------------------------------
# JSON parser for LLM output
# ---------------------------------------------------------------------------

def _parse_llm_output(raw: str) -> List[Dict[str, Any]]:
    """
    Try to extract a JSON array from free-form LLM text.
    Falls back to line-by-line heuristic if JSON parsing fails.
    """
    candidates: List[str] = []

    # 1. Fenced code block
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1))

    # 2. Bare JSON array anywhere in the text
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end > start:
        candidates.append(raw[start : end + 1])

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
                                "source": "openrouter",
                            })
            if tasks:
                return tasks
        except Exception:
            continue

    # 3. Line-by-line text fallback
    tasks = []
    for line in raw.splitlines():
        line = line.strip().lstrip("-•*0123456789.) ")
        if len(line) > 15 and _looks_like_task(line):
            tasks.append({
                "description": line[:500],
                "assignee_hint": _guess_assignee(line),
                "deadline_hint": _guess_deadline(line),
                "source": "openrouter_text",
            })
    return tasks


# ---------------------------------------------------------------------------
# OpenRouter API call
# ---------------------------------------------------------------------------

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_SYSTEM_PROMPT = (
    "You are an assistant that extracts action items from meeting transcripts. "
    "Return ONLY a valid JSON array — no prose, no markdown fences, no commentary. "
    "Each element must be a JSON object with exactly these keys: "
    '"description" (string), "assignee_hint" (string or null), "deadline_hint" (string or null). '
    "If no tasks are found, return an empty array []."
)

_USER_TEMPLATE = (
    "Extract all action items from the following meeting transcript.\n\n"
    "Transcript:\n{transcript}"
)


async def _call_openrouter(transcript: str) -> List[Dict[str, Any]]:
    api_key: str = getattr(settings, "OPENROUTER_API_KEY", "") or ""
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    model: str = getattr(settings, "OPENROUTER_TASK_MODEL", "openrouter/free") or "openrouter/free"

    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TEMPLATE.format(transcript=transcript[:4000])},
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional: helps OpenRouter leaderboard attribution
        "HTTP-Referer": "https://github.com/your-org/meeting-secretary",
        "X-Title": "Meeting Secretary",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(_OPENROUTER_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    actual_model = data.get("model", model)
    logger.info("[TASK] OpenRouter model used: %s", actual_model)

    raw = data["choices"][0]["message"]["content"] or "[]"
    logger.debug("[TASK] Raw output (first 400 chars): %s", raw[:400])

    tasks = _parse_llm_output(raw)
    # Stamp actual model for traceability
    for t in tasks:
        t["model"] = actual_model
    return tasks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def extract_tasks(transcript: str) -> List[Dict[str, Any]]:
    """
    Main entry point.  Called by routes and data-generation scripts.

    Strategy:
        1. If TASK_PROVIDER == "rules" → rule-based only (no network).
        2. Otherwise, try OpenRouter; on any failure fall back to rules.
        3. Merge LLM + rule results when LLM returns < 2 items.
        4. Deduplicate and cap at 20 tasks.
    """
    transcript = (transcript or "").strip()
    if not transcript:
        return []

    provider: str = getattr(settings, "TASK_PROVIDER", "openrouter") or "openrouter"

    fallback_tasks = _extract_tasks_simple(transcript)

    if provider == "rules":
        logger.info("[TASK] Using rule-based extraction (TASK_PROVIDER=rules)")
        return fallback_tasks

    # --- OpenRouter path ---
    try:
        llm_tasks = await _call_openrouter(transcript)
    except Exception as exc:
        logger.warning("[TASK] OpenRouter failed, using rule-based fallback: %s", exc)
        return fallback_tasks

    if not llm_tasks:
        logger.info("[TASK] LLM returned no tasks; using fallback")
        return fallback_tasks

    # Merge when LLM result is suspiciously thin
    combined = llm_tasks if len(llm_tasks) >= 2 else llm_tasks + fallback_tasks

    # Deduplicate
    unique: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for t in combined:
        key = _normalize_space((t.get("description") or "").lower())[:120]
        if key and key not in seen:
            seen.add(key)
            unique.append(t)

    return unique[:20]


def extract_tasks_rule_based(transcript: str) -> List[Dict[str, Any]]:
    """Sync compatibility shim for scripts that don't use asyncio."""
    return _extract_tasks_simple(transcript)
