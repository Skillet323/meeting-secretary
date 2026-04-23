"""Task assignment engine.

Strategy order:
1. Resolve speaker_hint using meeting-level speaker aliases.
2. Match assignee_hint to participant name/email/role.
3. Infer role from task text and match participant.role.
4. Extract a concrete name from the description/snippet.
5. Apply explicit regex/role rules.
6. Use round-robin fallback only when there are participants.
7. If no participants exist, keep assignee None rather than inventing one.

The engine enriches each task with:
    assignee, assignee_source, assignment_confidence, speaker_resolved
"""
from __future__ import annotations

import json
import re
from difflib import get_close_matches
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from ..models import Participant, Rule


def load_participants(session: Session) -> List[Participant]:
    return session.exec(select(Participant)).all()


def load_rules(session: Session) -> List[Rule]:
    return session.exec(select(Rule).order_by(Rule.priority)).all()


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _tags_as_text(tags: Optional[str]) -> str:
    if not tags:
        return ""
    try:
        parsed = json.loads(tags)
        if isinstance(parsed, dict):
            return _norm(" ".join(f"{k}:{v}" for k, v in parsed.items()))
        if isinstance(parsed, list):
            return _norm(" ".join(str(x) for x in parsed))
    except Exception:
        pass
    return _norm(tags.replace(",", " "))


def _participant_blob(p: Participant) -> str:
    return _norm(" ".join(filter(None, [p.name, p.email, p.role, _tags_as_text(p.tags)])))


def _match_name(participants: List[Participant], name_hint: Optional[str]) -> Optional[Participant]:
    if not name_hint:
        return None
    hint = _norm(name_hint)
    if not hint:
        return None

    for p in participants:
        blob = _participant_blob(p)
        if hint in blob:
            return p

    # If hint is a single token, allow fuzzy match only on names.
    names = [p.name for p in participants if p.name]
    close = get_close_matches(name_hint, names, n=1, cutoff=0.72)
    if close:
        for p in participants:
            if p.name == close[0]:
                return p

    return None

def _match_role(participants: List[Participant], role_hint: Optional[str]) -> Optional[Participant]:
    if not role_hint:
        return None
    hint = _norm(role_hint)
    if not hint:
        return None

    for p in participants:
        role = _norm(p.role)
        if role and (hint in role or role in hint):
            return p
    return None


def _speaker_alias_map(meeting_info: Optional[dict]) -> dict[str, str]:
    if not meeting_info:
        return {}
    aliases = meeting_info.get("speaker_aliases") or meeting_info.get("speaker_name_map") or {}
    if isinstance(aliases, dict):
        return {str(k): str(v) for k, v in aliases.items() if v}
    return {}

def _looks_like_real_name(text: str) -> bool:
    s = _norm(text)
    if not s:
        return False
    # one or two capitalized tokens
    return bool(re.fullmatch(r"[a-zа-яё][a-zа-яё'\-]+(?:\s+[a-zа-яё][a-zа-яё'\-]+)?", s, flags=re.IGNORECASE))

def _resolve_speaker_hint(task: dict, meeting_info: Optional[dict], participants: List[Participant]) -> Optional[str]:
    speaker_hint = task.get("speaker_hint") or task.get("speaker") or task.get("speaker_label")
    if not speaker_hint:
        return None

    speaker_hint_str = str(speaker_hint).strip()
    if not speaker_hint_str:
        return None

    aliases = _speaker_alias_map(meeting_info)
    resolved = aliases.get(speaker_hint_str)
    if not resolved:
        # Normalize speaker variants like 'Speaker 0' / 'SPEAKER_00'.
        digits = re.search(r"(\d+)", speaker_hint_str)
        if digits:
            idx = digits.group(1)
            for key, value in aliases.items():
                if re.search(rf"\b{re.escape(idx)}\b", key) or re.search(rf"\b{re.escape(idx)}\b", value):
                    resolved = value
                    break

    if resolved:
        return resolved

    # If speaker hint already looks like a real name, keep it.
    if _match_name(participants, speaker_hint_str):
        return speaker_hint_str
    return None


_ROLE_KEYWORDS = {
    "project manager": ["project manager", "pm", "manager"],
    "industrial designer": ["industrial designer", "designer", "design"],
    "marketing expert": ["marketing expert", "marketing", "market"],
    "user interface": ["user interface", "ui", "interface"],
    "technical": ["technical", "tech", "engineer"],
    "facilitator": ["facilitator", "moderator", "host"],
}


def _infer_role_hint(text: str) -> Optional[str]:
    s = _norm(text)
    if not s:
        return None
    for role, keys in _ROLE_KEYWORDS.items():
        if any(k in s for k in keys):
            return role
    return None


def _extract_candidate_name(text: str) -> Optional[str]:
    """Try to extract a concrete person name from task text/snippet."""
    patterns = [
        r"(?:for|to|assigned to|by|from)\s+([A-ZА-ЯЁ][a-zа-яё]+(?:\s+[A-ZА-ЯЁ][a-zа-яё]+)?)",
        r"(?:speaking|speaker)\s+([A-ZА-ЯЁ][a-zа-яё]+(?:\s+[A-ZА-ЯЁ][a-zа-яё]+)?)",
        r"@([\w.-]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()
    return None


def _rule_assignee(desc: str, rules: List[Rule], participants: List[Participant]) -> tuple[Optional[str], Optional[str]]:
    """Return (assignee, source)."""
    for r in rules:
        if r.kind == "regex" and r.pattern:
            try:
                m = re.search(r.pattern, desc, re.IGNORECASE)
            except re.error:
                continue
            if not m:
                continue

            extracted = None
            if m.lastindex:
                extracted = m.group(1)
            else:
                extracted = m.group(0)
            extracted = re.sub(r"[^\w\s@.-]", "", str(extracted)).strip()
            if extracted:
                p = _match_name(participants, extracted)
                if p:
                    return (p.email or p.name, "rule:regex:name")
                return (extracted, "rule:regex:raw")
            return (None, "rule:regex")

        if r.kind == "role_lookup" and r.pattern:
            try:
                obj = json.loads(r.pattern)
            except json.JSONDecodeError:
                continue
            role = _norm(obj.get("role"))
            default_assignee = obj.get("assignee")
            if role and role in _norm(desc):
                return (default_assignee, "rule:role_lookup")

    return (None, None)


def assign_task_to_participant(task: dict, meeting_info: Optional[dict], participants: List[Participant], round_robin_idx: int) -> tuple[Optional[str], str, float]:
    # 1) speaker alias -> participant
    speaker_resolved = _resolve_speaker_hint(task, meeting_info, participants)
    if speaker_resolved:
        p = _match_name(participants, speaker_resolved)
        if p:
            return p.name, "speaker_alias", 0.95

    # 2) explicit assignee_hint
    assignee_hint = task.get("assignee_hint")
    p = _match_name(participants, assignee_hint)
    if p:
        return p.name, "assignee_hint", 0.90

    # 3) role hint from task text
    role_hint = _infer_role_hint(task.get("description", ""))
    p = _match_role(participants, role_hint)
    if p:
        return p.name, "role_hint", 0.70

    # 4) concrete name in task / snippet
    candidate = _extract_candidate_name(
        " ".join(filter(None, [task.get("description", ""), task.get("source_snippet", "")]))
    )
    if candidate:
        p = _match_name(participants, candidate)
        if p:
            return p.name, "name_in_text", 0.80

    # 5) explicit role words in text
    role_text = _infer_role_hint(" ".join(filter(None, [task.get("description", ""), task.get("assignee_hint", "")])))
    p = _match_role(participants, role_text)
    if p:
        return p.name, "role_text", 0.65

    # 6) round-robin only as a very last resort and only for plausible tasks
    desc_norm = _norm(task.get("description", ""))
    if participants and len(desc_norm.split()) >= 4:
        p = participants[round_robin_idx % len(participants)]
        return p.name, "round_robin", 0.15

    return None, "unassigned", 0.0