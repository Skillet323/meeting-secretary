"""
Task extraction with OpenRouter + robust parser + strict rule-based fallback.

This module is designed for two workflows:
1) runtime extraction in the backend
2) batch generation of pseudo-gold AMI JSON files

Returned task dicts are normalized to:
    {
        "description": str,
        "assignee_hint": str|None,
        "deadline_hint": str|None,
        "speaker_hint": str|None,      # optional
        "source": "openrouter"|"openrouter_text"|"rule_based",
        "source_snippet": str|None,    # optional
        "model": str|None,             # only for LLM outputs
    }
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from typing import Any, Dict, List, Optional

import httpx

from ..config import settings

logger = logging.getLogger(__name__)

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_OPENROUTER_RETRIES = 3
BASE_BACKOFF_SEC = 2.5

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "had", "are", "was", "were",
    "will", "would", "could", "should", "need", "needs", "please", "about", "into", "onto", "then",
    "than", "them", "they", "their", "there", "here", "what", "when", "where", "why", "how", "who",
    "whom", "which", "your", "our", "you", "we", "uh", "um", "mm", "yeah", "okay", "ok", "right",
    "just", "also", "still", "very", "really", "maybe"
}


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _preview(text: str, limit: int = 800) -> str:
    text = (text or "").replace("\r", " ").strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def _split_speaker_prefix(text: str) -> tuple[Optional[str], str]:
    """
    Split speaker-labeled lines such as:
      - SPEAKER_00: hello
      - Speaker 1: hello
      - A: hello
      - Laura: hello
      - David Smith: hello
    """
    m = re.match(
        r"^\s*((?:SPEAKER_\d+)|(?:Speaker\s+\d+)|(?:[A-Z])|(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}))\s*:\s*(.+)$",
        text or "",
    )
    if not m:
        return None, _normalize_space(text)

    speaker = _normalize_space(m.group(1))
    body = _normalize_space(m.group(2))
    return speaker, body


def _tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    return [t for t in text.split() if t and t not in STOPWORDS and len(t) > 2]


def _task_supported_by_transcript(description: str, transcript: str, min_overlap: float = 0.15) -> bool:
    """
    Keep only tasks with some lexical support in the current transcript.
    This is intentionally strict for short/noisy transcripts.
    """
    desc_tokens = set(_tokenize(description))
    tr_tokens = set(_tokenize(transcript))

    if len(desc_tokens) < 3 or not tr_tokens:
        return False

    overlap = len(desc_tokens & tr_tokens) / max(1, len(desc_tokens))
    if overlap >= min_overlap:
        return True

    if len(desc_tokens & tr_tokens) >= 2:
        return True

    return False


def _guess_deadline(sentence: str) -> Optional[str]:
    m = re.search(
        r"(?:by|до|к|к\s+)"
        r"(\d{1,2}(?:[./-]\d{1,2})?(?:[./-]\d{2,4})?|"
        r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"понедельник|вторник|среда|четверг|пятница|суббота|воскресенье))",
        sentence,
        flags=re.IGNORECASE,
    )
    return _normalize_space(m.group(1)) if m else None


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
    """Conservative rule-based detection for fallback."""
    s = normalize_text(sentence)

    deny = [
        "agenda",
        "project manager",
        "we re developing",
        "we are developing",
        "first meeting",
        "icebreaker",
        "favourite animal",
        "favourite characteristic",
        "white board",
        "design stages",
        "finance",
        "marketing",
        "introduction",
        "introduce ourselves",
        "introduce self",
        "meeting agenda",
        "good morning",
        "hello everybody",
        "i am",
        "my name is",
    ]
    if any(d in s for d in deny):
        return False

    markers = [
        "should",
        "must",
        "need to",
        "needs to",
        "please",
        "action item",
        "task",
        "to do",
        "follow up",
        "let s",
        "let us",
        "have to",
        "required to",
        "нужно",
        "надо",
        "нужна",
        "нужен",
        "сделать",
        "подготовить",
        "проверить",
        "отправить",
        "согласовать",
        "обновить",
        "выполнить",
        "завершить",
        "доработать",
        "назначить",
        "созвониться",
        "позвонить",
    ]
    return any(marker in s for marker in markers)


def _is_meta_task(text: str) -> bool:
    """Filter out obvious model meta-output or meeting narration."""
    s = normalize_text(text)
    bad_phrases = [
        "review and summarize action items",
        "extract action items from the meeting transcript",
        "meeting transcript",
        "summarize action items",
        "introduce participants",
        "confirm meeting agenda",
        "project goal and objectives",
        "outline the project structure",
        "describe the functional design process",
        "explain the tool training exercise",
        "introduce the user interface design approach",
        "confirm attendance",
        "review current remote control features",
        "start the meeting",
        "confirm everyone is ready",
        "read the entire transcript first",
        "meeting id",
        "language",
        "duration",
    ]
    return any(p in s for p in bad_phrases)


def _normalize_assignee_hint(hint: Any) -> Optional[str]:
    """
    Keep only plausible specific names/roles.
    Drop generic instruction-like text.
    """
    if hint is None:
        return None

    text = _normalize_space(str(hint))
    if not text:
        return None

    lower = normalize_text(text)

    generic_markers = [
        "assign tasks",
        "ensure",
        "review",
        "summarize",
        "prepare",
        "confirm",
        "lead",
        "team",
        "members",
        "based on meeting content",
        "meeting content",
        "project manager",
        "facilitator",
        "design team",
        "marketing expert",
        "product designer",
        "assign specific tasks",
        "should do",
        "task owner",
    ]

    if len(text.split()) > 5:
        return None

    if any(marker in lower for marker in generic_markers):
        # Keep if it is a short title-like role from the transcript,
        # otherwise drop generic pseudo-assignees.
        if not re.fullmatch(r"[A-Za-zА-ЯЁ][\w.-]+(?:\s+[A-Za-zА-ЯЁ][\w.-]+)?", text):
            return None

    return text


def _normalize_deadline_hint(hint: Any) -> Optional[str]:
    if hint is None:
        return None
    text = _normalize_space(str(hint))
    return text or None


def _add_task(tasks: List[Dict[str, Any]], item: Dict[str, Any], transcript: str) -> None:
    desc = _normalize_space(str(item.get("description") or item.get("task") or ""))
    if not desc or _is_meta_task(desc):
        return

    # For short/noisy transcripts, be stricter.
    if not _task_supported_by_transcript(desc, transcript):
        return

    speaker_hint = item.get("speaker_hint") or None
    speaker_hint = _normalize_space(str(speaker_hint)) if speaker_hint else None

    assignee_hint = _normalize_assignee_hint(item.get("assignee_hint") or item.get("assignee"))
    deadline_hint = _normalize_deadline_hint(item.get("deadline_hint") or item.get("deadline"))

    source_snippet = item.get("source_snippet") or item.get("evidence")
    source_snippet = _normalize_space(str(source_snippet)) if source_snippet else None

    out: Dict[str, Any] = {
        "description": desc[:500],
        "assignee_hint": assignee_hint,
        "deadline_hint": deadline_hint,
        "source": item.get("source") or "openrouter",
    }
    if speaker_hint:
        out["speaker_hint"] = speaker_hint
    if source_snippet:
        out["source_snippet"] = source_snippet[:120]
    if item.get("model"):
        out["model"] = item["model"]

    tasks.append(out)


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

def _extract_tasks_simple(transcript: str) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", transcript)

    for raw in sentences:
        speaker, body = _split_speaker_prefix(raw)
        sentence = _normalize_space(body)
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

        item: Dict[str, Any] = {
            "description": sentence[:500],
            "assignee_hint": _guess_assignee(sentence),
            "deadline_hint": _guess_deadline(sentence),
            "source": "rule_based",
        }
        if speaker:
            item["speaker_hint"] = speaker

        if _task_supported_by_transcript(item["description"], transcript):
            tasks.append(item)

    unique: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for t in tasks:
        key = _normalize_space(t["description"].lower())[:120]
        if key and key not in seen:
            seen.add(key)
            unique.append(t)

    return unique


# ---------------------------------------------------------------------------
# LLM output parsing
# ---------------------------------------------------------------------------

def _collect_json_tasks(node: Any, tasks: List[Dict[str, Any]], transcript: str) -> None:
    """
    Recursively collect task-like objects from arbitrarily nested JSON.
    """
    if isinstance(node, dict):
        desc = str(node.get("description") or node.get("task") or "").strip()
        if desc and not _is_meta_task(desc) and _task_supported_by_transcript(desc, transcript):
            item: Dict[str, Any] = {
                "description": desc[:500],
                "assignee_hint": _normalize_assignee_hint(node.get("assignee_hint") or node.get("assignee")),
                "deadline_hint": _normalize_deadline_hint(node.get("deadline_hint") or node.get("deadline")),
                "source": "openrouter",
            }
            if node.get("source_snippet"):
                item["source_snippet"] = str(node["source_snippet"])[:120]
            tasks.append(item)

        # Recurse only into non-scalar content and ignore task metadata fields.
        for key, value in node.items():
            if key in {"description", "task", "assignee_hint", "assignee", "deadline_hint", "deadline", "source_snippet", "evidence"}:
                continue
            _collect_json_tasks(value, tasks, transcript)

    elif isinstance(node, list):
        for item in node:
            _collect_json_tasks(item, tasks, transcript)

    elif isinstance(node, str):
        text = _normalize_space(node)
        if len(text) < 10 or _is_meta_task(text):
            return
        if _task_supported_by_transcript(text, transcript):
            tasks.append(
                {
                    "description": text[:500],
                    "assignee_hint": _guess_assignee(text),
                    "deadline_hint": _guess_deadline(text),
                    "source": "openrouter_text",
                }
            )


def _parse_llm_output(raw: str, transcript: str, *, trace_id: str | None = None) -> tuple[List[Dict[str, Any]], dict[str, Any]]:
    debug: dict[str, Any] = {
        "raw": raw,
        "raw_preview": _preview(raw, 1200),
        "parse_stage": None,
        "parsed_tasks": 0,
    }

    candidates: List[str] = []

    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())

    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end > start:
        candidates.append(raw[start : end + 1].strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue

        tasks: List[Dict[str, Any]] = []
        _collect_json_tasks(parsed, tasks, transcript)

        if tasks:
            unique: List[Dict[str, Any]] = []
            seen: set[str] = set()
            for t in tasks:
                desc = _normalize_space((t.get("description") or "").lower())[:120]
                if not desc or desc in seen:
                    continue
                seen.add(desc)
                unique.append(t)

            debug["parse_stage"] = "json_recursive"
            debug["parsed_tasks"] = len(unique)
            return unique, debug

    tasks: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = _normalize_space(line)
        if not line:
            continue

        low = normalize_text(line)
        if low in {"action items", "task", "tasks", "extracted action items"}:
            continue
        if low.startswith("[") and low.endswith("]"):
            continue
        if low.startswith("extracted action items"):
            continue

        if len(line) > 15 and _looks_like_task(line) and _task_supported_by_transcript(line, transcript):
            tasks.append(
                {
                    "description": line[:500],
                    "assignee_hint": _guess_assignee(line),
                    "deadline_hint": _guess_deadline(line),
                    "source": "openrouter_text",
                }
            )

    debug["parse_stage"] = "line_fallback"
    debug["parsed_tasks"] = len(tasks)
    return tasks, debug


# ---------------------------------------------------------------------------
# OpenRouter call
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a strict action-item extractor for meeting transcripts.\n"
    "Return ONLY a valid JSON array. No prose. No markdown. No explanation.\n"
    "If no valid tasks exist, return [].\n\n"
    "Rules:\n"
    "- Extract only tasks that are clearly intended to happen AFTER the meeting.\n"
    "- Do NOT extract introductions, agenda items, discussion topics, or summaries.\n"
    "- Do NOT invent people, names, roles, deadlines, or tasks.\n"
    "- If the transcript is short, noisy, or mostly off-topic, prefer returning [].\n"
    "- Keep descriptions concrete and short.\n"
    "- Use assignee_hint only if the person/role is actually mentioned or strongly implied in the transcript.\n"
    "- Use deadline_hint only if explicitly stated or very clearly implied.\n"
    "- speaker_hint should be copied only from actual speaker labels found in the transcript (e.g. SPEAKER_00, A). Do not turn them into human names.\n\n"
    "Output schema for each item:\n"
    "{"
    '"description": "string", '
    '"assignee_hint": "string or null", '
    '"deadline_hint": "string or null", '
    '"speaker_hint": "string or null", '
    '"source_snippet": "string or null"'
    "}"
)

_USER_TEMPLATE = (
    "Extract action items from the meeting transcript below.\n\n"
    "Meeting ID: {meeting_ref}\n"
    "Language: {language}\n"
    "Duration: {duration_sec:.1f} seconds\n"
    "Transcript confidence: {transcript_confidence}\n\n"
    "Transcript:\n"
    "{transcript}\n\n"
    "Return only a JSON array."
)

async def _call_openrouter(
    transcript: str,
    *,
    trace_id: str | None = None,
    meeting_ref: str | None = None,
    language: str = "en",
    duration_sec: float | None = None,
    transcript_confidence: float | None = None,
) -> tuple[List[Dict[str, Any]], dict[str, Any]]:
    api_key: str = getattr(settings, "OPENROUTER_API_KEY", "") or ""
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    model: str = getattr(settings, "OPENROUTER_TASK_MODEL", "openrouter/free") or "openrouter/free"

    user_content = _USER_TEMPLATE.format(
        meeting_ref=meeting_ref or "unknown",
        language=language or "en",
        duration_sec=duration_sec or 0.0,
        transcript_confidence=f"{transcript_confidence:.3f}" if transcript_confidence is not None else "unknown",
        transcript=transcript[:8000],
    )

    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 1200,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-org/meeting-secretary",
        "X-Title": "Meeting Secretary",
    }

    last_error: Exception | None = None

    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(MAX_OPENROUTER_RETRIES):
            try:
                resp = await client.post(_OPENROUTER_URL, json=payload, headers=headers)

                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    logger.warning(
                        "[TASK][%s] OpenRouter 429 rate limit%s",
                        trace_id or "-",
                        f", Retry-After={retry_after}" if retry_after else "",
                    )
                    return [], {
                        "provider": "openrouter",
                        "model": model,
                        "raw_preview": None,
                        "parse_stage": None,
                        "parsed_tasks": 0,
                        "fallback_used": True,
                        "error": "rate_limited",
                    }

                if resp.status_code in (408, 502, 503, 529):
                    wait_sec = BASE_BACKOFF_SEC * (2**attempt) + random.uniform(0, 0.75)
                    logger.warning(
                        "[TASK][%s] OpenRouter transient error %s, retry in %.1fs (attempt %d/%d)",
                        trace_id or "-",
                        resp.status_code,
                        wait_sec,
                        attempt + 1,
                        MAX_OPENROUTER_RETRIES,
                    )
                    await asyncio.sleep(wait_sec)
                    continue

                resp.raise_for_status()
                data = resp.json()

                actual_model = data.get("model", model)
                raw = data["choices"][0]["message"]["content"] or "[]"

                logger.info("[TASK][%s] OpenRouter model used: %s", trace_id or "-", actual_model)
                logger.info("[TASK][%s] OpenRouter raw output:\n%s", trace_id or "-", raw)

                tasks, parse_debug = _parse_llm_output(raw, transcript, trace_id=trace_id)
                for t in tasks:
                    t["model"] = actual_model

                debug = {
                    "provider": "openrouter",
                    "model": actual_model,
                    "raw_preview": _preview(raw, 1200),
                    "parse_stage": parse_debug.get("parse_stage"),
                    "parsed_tasks": parse_debug.get("parsed_tasks", 0),
                    "fallback_used": False,
                }
                return tasks, debug

            except Exception as exc:
                last_error = exc
                if attempt < MAX_OPENROUTER_RETRIES - 1:
                    wait_sec = BASE_BACKOFF_SEC * (2**attempt) + random.uniform(0, 0.75)
                    logger.warning(
                        "[TASK][%s] OpenRouter error, retry in %.1fs (attempt %d/%d): %s",
                        trace_id or "-",
                        wait_sec,
                        attempt + 1,
                        MAX_OPENROUTER_RETRIES,
                        exc,
                    )
                    await asyncio.sleep(wait_sec)
                    continue
                break

    raise RuntimeError(f"OpenRouter failed after retries: {last_error}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def extract_tasks(
    transcript: str,
    *,
    return_debug: bool = False,
    trace_id: str | None = None,
    meeting_ref: str | None = None,
    language: str = "en",
    duration_sec: float | None = None,
    transcript_confidence: float | None = None,
) -> List[Dict[str, Any]] | tuple[List[Dict[str, Any]], dict[str, Any]]:
    transcript = _normalize_space(transcript)
    provider: str = getattr(settings, "TASK_PROVIDER", "openrouter") or "openrouter"
    fallback_tasks = _extract_tasks_simple(transcript)

    debug: dict[str, Any] = {
        "provider": provider,
        "trace_id": trace_id,
        "fallback_tasks": len(fallback_tasks),
        "model": None,
        "raw_preview": None,
        "parse_stage": None,
        "fallback_used": provider == "rules",
        "conservative_mode": False,
    }

    if not transcript:
        debug["reason"] = "empty transcript"
        return ([], debug) if return_debug else []

    words_count = len(transcript.split())
    short_or_noisy = (
        (duration_sec is not None and duration_sec < 180)
        or words_count < 40
        or (transcript_confidence is not None and transcript_confidence < 0.60)
    )

    # For short/noisy transcripts, better a conservative baseline than hallucinated tasks.
    if short_or_noisy and provider != "rules":
        logger.info(
            "[TASK][%s] Conservative mode enabled: short/noisy transcript, using rule-based fallback only",
            trace_id or "-",
        )
        debug["conservative_mode"] = True
        debug["fallback_used"] = True
        return (fallback_tasks, debug) if return_debug else fallback_tasks

    if provider == "rules":
        logger.info("[TASK][%s] Using rule-based extraction", trace_id or "-")
        return (fallback_tasks, debug) if return_debug else fallback_tasks

    try:
        llm_tasks, llm_debug = await _call_openrouter(
            transcript,
            trace_id=trace_id,
            meeting_ref=meeting_ref,
            language=language,
            duration_sec=duration_sec,
            transcript_confidence=transcript_confidence,
        )
        debug.update(llm_debug)
    except Exception as exc:
        logger.warning("[TASK][%s] OpenRouter failed, using rule-based fallback: %s", trace_id or "-", exc)
        debug["error"] = str(exc)
        debug["fallback_used"] = True
        return (fallback_tasks, debug) if return_debug else fallback_tasks

    if not llm_tasks:
        logger.info("[TASK][%s] LLM returned no tasks; using fallback", trace_id or "-")
        debug["llm_tasks"] = 0
        debug["fallback_used"] = True
        return (fallback_tasks, debug) if return_debug else fallback_tasks

    debug["llm_tasks"] = len(llm_tasks)

    # If the LLM result is thin, merge with fallback to preserve some coverage.
    combined = llm_tasks if len(llm_tasks) >= 2 else llm_tasks + fallback_tasks
    debug["fallback_merged"] = len(llm_tasks) < 2 and len(fallback_tasks) > 0

    unique: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for t in combined:
        key = _normalize_space((t.get("description") or "").lower())[:120]
        if key and key not in seen:
            seen.add(key)
            unique.append(t)

    result = unique[:20]
    debug["final_tasks"] = len(result)

    return (result, debug) if return_debug else result


def extract_tasks_rule_based(transcript: str) -> List[Dict[str, Any]]:
    return _extract_tasks_simple(transcript)