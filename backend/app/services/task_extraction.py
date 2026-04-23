"""Task extraction with OpenRouter + robust parser + rule-based fallback.

This module is designed for two workflows:
1) runtime extraction in the backend
2) batch generation of pseudo-gold AMI JSON files

Public API:
    async extract_tasks(transcript, return_debug=False, trace_id=None,
                        meeting_ref=None, language='en', duration_sec=None)
    def extract_tasks_rule_based(transcript)

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
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ..config import settings

logger = logging.getLogger(__name__)

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_OPENROUTER_RETRIES = 3
BASE_BACKOFF_SEC = 2.5


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _preview(text: str, limit: int = 800) -> str:
    text = (text or "").replace("\r", " ").strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def _split_speaker_prefix(text: str) -> tuple[Optional[str], str]:
    """Split 'SPEAKER_00: hello' or 'A: hello' into (speaker, text)."""
    m = re.match(r"^\s*((?:SPEAKER_\d+)|(?:Speaker\s+\d+)|(?:[A-Z]))\s*:\s*(.+)$", text or "")
    if not m:
        return None, _normalize_space(text)
    speaker = _normalize_space(m.group(1))
    body = _normalize_space(m.group(2))
    return speaker, body


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
    s = sentence.lower()

    deny = [
        "agenda",
        "project manager",
        "we're developing",
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
        "let's",
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
    s = _normalize_space(text.lower())
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
    ]
    return any(p in s for p in bad_phrases)


def _normalize_assignee_hint(hint: Any) -> Optional[str]:
    if hint is None:
        return None
    text = _normalize_space(str(hint))
    if not text:
        return None

    # Drop obviously generic instructions.
    generic_markers = [
        "assign",
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
    ]
    if len(text.split()) > 6:
        return None
    if any(marker in text.lower() for marker in generic_markers) and not re.search(r"[A-ZА-ЯЁ][a-zа-яё]+", text):
        return None
    return text


def _add_task(tasks: List[Dict[str, Any]], item: Dict[str, Any]) -> None:
    desc = _normalize_space(str(item.get("description") or item.get("task") or ""))
    if not desc or _is_meta_task(desc):
        return

    speaker_hint = item.get("speaker_hint") or None
    speaker_hint = _normalize_space(str(speaker_hint)) if speaker_hint else None
    assignee_hint = _normalize_assignee_hint(item.get("assignee_hint") or item.get("assignee"))
    deadline_hint = _normalize_space(str(item.get("deadline_hint") or item.get("deadline") or "")) or None
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


def _collect_json_tasks(node: Any, tasks: List[Dict[str, Any]]) -> None:
    if isinstance(node, dict):
        desc = str(node.get("description") or node.get("task") or "").strip()
        if desc:
            if not _is_meta_task(desc):
                item: Dict[str, Any] = {
                    "description": desc[:500],
                    "assignee_hint": node.get("assignee_hint") or node.get("assignee"),
                    "deadline_hint": node.get("deadline_hint") or node.get("deadline"),
                    "speaker_hint": node.get("speaker_hint") or node.get("speaker"),
                    "source": node.get("source") or "openrouter",
                }
                if node.get("source_snippet"):
                    item["source_snippet"] = str(node["source_snippet"])[:120]
                tasks.append(item)
            return

        for value in node.values():
            if isinstance(value, (dict, list)):
                _collect_json_tasks(value, tasks)

    elif isinstance(node, list):
        for item in node:
            _collect_json_tasks(item, tasks)

    elif isinstance(node, str):
        speaker, text = _split_speaker_prefix(node)
        text = _normalize_space(text)
        if len(text) < 10 or _is_meta_task(text):
            return
        tasks.append(
            {
                "description": text[:500],
                "assignee_hint": _guess_assignee(text),
                "deadline_hint": _guess_deadline(text),
                "speaker_hint": speaker,
                "source": "openrouter_text",
                "source_snippet": text[:120],
            }
        )


def _parse_llm_output(raw: str) -> tuple[List[Dict[str, Any]], dict[str, Any]]:
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
        _collect_json_tasks(parsed, tasks)
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

    # Line fallback for malformed JSON-like output.
    tasks = []
    for line in raw.splitlines():
        speaker, text = _split_speaker_prefix(line)
        text = _normalize_space(text)
        if not text:
            continue

        low = text.lower()
        if low in {"action items", "task", "tasks", "extracted action items"}:
            continue
        if low.startswith("[") and low.endswith("]"):
            continue
        if low.startswith("extracted action items"):
            continue

        if len(text) > 15 and _looks_like_task(text):
            tasks.append(
                {
                    "description": text[:500],
                    "assignee_hint": _guess_assignee(text),
                    "deadline_hint": _guess_deadline(text),
                    "speaker_hint": speaker,
                    "source": "openrouter_text",
                    "source_snippet": text[:120],
                }
            )

    # Deduplicate.
    unique: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for t in tasks:
        key = _normalize_space(t["description"].lower())[:120]
        if key and key not in seen:
            seen.add(key)
            unique.append(t)

    debug["parse_stage"] = "line_fallback"
    debug["parsed_tasks"] = len(unique)
    return unique, debug


# ---------------------------------------------------------------------------
# OpenRouter call
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a precise action-item extractor for meeting transcripts. "
    "Your ONLY output must be a valid JSON array — no prose, no markdown, no commentary. "
    "If no valid tasks are found, return [].\n\n"
    "### DEFINITION OF A VALID ACTION ITEM ###\n"
    "A valid task MUST meet ALL criteria:\n"
    "1. FUTURE-ORIENTED: The action must happen AFTER the meeting ends. "
    "Ignore introductions, discussions, decisions already made, or retrospective comments.\n"
    "2. CONCRETE & ACTIONABLE: Must contain a specific verb (e.g., 'draft', 'review', 'send', 'build'). "
    "Vague items like 'discuss further' or 'think about' are invalid unless tied to a deliverable.\n"
    "3. ASSIGNABLE: Must be possible to assign to a specific person or role mentioned in the transcript.\n\n"
    "### EXTRACTION RULES ###\n"
    "- assignee_hint: Extract the EXACT name or role spoken (e.g., 'David', 'marketing expert'). "
    "If no assignee is mentioned or strongly implied, use null.\n"
    "- speaker_hint: If the transcript line contains a speaker label, copy it exactly (e.g. 'SPEAKER_00', 'A'). "
    "Otherwise use null.\n"
    "- deadline_hint: Extract ONLY if explicitly stated ('by Friday', 'before next meeting') "
    "or strongly implied by context. Never invent deadlines. If unclear, use null.\n"
    "- description: Be specific. Include key constraints mentioned (e.g., 'design remote control casing, max cost €12.50').\n"
    "- source_snippet: Optional short quote (≤15 words) from the transcript that justifies this task.\n\n"
    "### NEGATIVE EXAMPLES (DO NOT EXTRACT) ###\n"
    "'Introduce participants' → already happened during meeting\n"
    "'Discuss project finance' → discussion topic, not an action\n"
    "'Review project announcement' → past event (they already received it)\n"
    "'Confirm attendance' → administrative note, not a task\n\n"
    "### POSITIVE EXAMPLES (EXTRACT THESE) ###\n"
    "'David to draft industrial design concepts for remote casing by next session'\n"
    "'Craig to define UI technical functions before 30-min break'\n"
    "'Andrew to research target market positioning for €25 price point'\n\n"
    "### OUTPUT FORMAT ###\n"
    "Each element must be a JSON object with EXACTLY these keys:\n"
    "{\n"
    '  "description": "string",\n'
    '  "assignee_hint": "string or null",\n'
    '  "deadline_hint": "string or null",\n'
    '  "speaker_hint": "string or null",\n'
    '  "source_snippet": "string or null"\n'
    "}\n\n"
    "Ground every task in the transcript text. When in doubt, exclude."
)

_USER_TEMPLATE = (
    "Extract action items from the meeting transcript below.\n\n"
    "### MEETING METADATA ###\n"
    "Meeting ID: {meeting_ref}\n"
    "Language: {language}\n"
    "Duration: {duration_sec:.1f} seconds\n\n"
    "### TRANSCRIPT ###\n"
    "{transcript}\n\n"
    "### INSTRUCTIONS ###\n"
    "1. Read the ENTIRE transcript first.\n"
    "2. Identify ONLY tasks that participants agreed to complete AFTER this meeting.\n"
    "3. For each task, extract: description, assignee_hint, deadline_hint, speaker_hint, source_snippet.\n"
    "4. Return ONLY a valid JSON array. No other text.\n\n"
    "Remember: If the action already happened DURING the meeting, DO NOT extract it."
)


async def _call_openrouter(
    transcript: str,
    *,
    trace_id: str | None = None,
    meeting_ref: str | None = None,
    language: str = "en",
    duration_sec: float | None = None,
) -> tuple[List[Dict[str, Any]], dict[str, Any]]:
    api_key: str = getattr(settings, "OPENROUTER_API_KEY", "") or ""
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    model: str = getattr(settings, "OPENROUTER_TASK_MODEL", "openrouter/free") or "openrouter/free"

    user_content = _USER_TEMPLATE.format(
        meeting_ref=meeting_ref or "unknown",
        language=language or "en",
        duration_sec=duration_sec or 0.0,
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
                    raise RuntimeError("OPENROUTER_RATE_LIMITED")

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

                tasks, parse_debug = _parse_llm_output(raw)
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

            except RuntimeError as exc:
                if str(exc) == "OPENROUTER_RATE_LIMITED":
                    return [], {
                        "provider": "openrouter",
                        "model": model,
                        "raw_preview": None,
                        "parse_stage": None,
                        "parsed_tasks": 0,
                        "fallback_used": True,
                        "error": "rate_limited",
                    }
                last_error = exc
                break
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
) -> List[Dict[str, Any]] | tuple[List[Dict[str, Any]], dict[str, Any]]:
    transcript = _normalize_space(transcript)
    if not transcript:
        empty_debug = {"provider": "none", "reason": "empty transcript"}
        return ([], empty_debug) if return_debug else []

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
    }

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

    # If the LLM returns a thin list, merge with fallback so we still have coverage.
    combined = llm_tasks if len(llm_tasks) >= 2 else llm_tasks + fallback_tasks

    unique: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for t in combined:
        key = _normalize_space((t.get("description") or "").lower())[:120]
        if key and key not in seen:
            seen.add(key)
            unique.append(t)

    result = unique[:20]
    debug["final_tasks"] = len(result)
    debug["fallback_merged"] = len(llm_tasks) < 2 and len(fallback_tasks) > 0
    debug.setdefault("fallback_used", False)

    return (result, debug) if return_debug else result


def extract_tasks_rule_based(transcript: str) -> List[Dict[str, Any]]:
    return _extract_tasks_simple(transcript)
