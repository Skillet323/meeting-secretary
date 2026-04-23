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

    # Direct / substring match first.
    for p in participants:
        blob = _participant_blob(p)
        if hint in blob:
            return p

    # Fuzzy name match.
    names = [p.name for p in participants if p.name]
    close = get_close_matches(name_hint, names, n=1, cutoff=0.55)
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


def assign_tasks_to_participants(
    tasks: List[dict],
    session: Session,
    meeting_info: Optional[dict] = None,
) -> List[dict]:
    """Assign tasks to participants using name/role/speaker mapping."""
    participants = load_participants(session)
    rules = load_rules(session)
    rr_index = 0
    aliases = _speaker_alias_map(meeting_info)

    for t in tasks:
        desc = str(t.get("description") or "")
        hint = t.get("assignee_hint") or t.get("assignee")
        speaker_hint = _resolve_speaker_hint(t, meeting_info, participants)
        speaker_resolved = None
        assigned = None
        source = None
        confidence = 0.0

        # 1) Speaker hint -> alias -> participant.
        if speaker_hint:
            speaker_resolved = aliases.get(str(t.get("speaker_hint") or t.get("speaker") or t.get("speaker_label") or "").strip())
            if speaker_resolved:
                p = _match_name(participants, speaker_resolved)
                if p:
                    assigned = p.email or p.name
                    source = "speaker_alias"
                    confidence = 0.95
                elif not participants:
                    assigned = speaker_resolved
                    source = "speaker_alias_text"
                    confidence = 0.65
            if not assigned:
                p = _match_name(participants, speaker_hint)
                if p:
                    assigned = p.email or p.name
                    source = "speaker_name"
                    confidence = 0.9
                elif not participants:
                    assigned = speaker_hint
                    source = "speaker_text"
                    confidence = 0.5

        # 2) Explicit assignee hint from model.
        if not assigned and hint:
            p = _match_name(participants, hint)
            if p:
                assigned = p.email or p.name
                source = "assignee_hint:name"
                confidence = 0.85
            else:
                p = _match_role(participants, hint)
                if p:
                    assigned = p.email or p.name
                    source = "assignee_hint:role"
                    confidence = 0.8
                elif not participants:
                    assigned = str(hint)
                    source = "assignee_hint:text"
                    confidence = 0.45

        # 3) Look for names in task description.
        if not assigned:
            name = _extract_candidate_name(desc)
            if name:
                p = _match_name(participants, name)
                if p:
                    assigned = p.email or p.name
                    source = "description:name"
                    confidence = 0.75

        # 4) Infer a role from the description.
        if not assigned:
            role_hint = _infer_role_hint(desc) or _infer_role_hint(str(hint or ""))
            if role_hint:
                p = _match_role(participants, role_hint)
                if p:
                    assigned = p.email or p.name
                    source = "inferred_role"
                    confidence = 0.7
                elif not participants:
                    assigned = role_hint
                    source = "inferred_role_text"
                    confidence = 0.35

        # 5) Regex / role rules.
        if not assigned:
            rule_assignee, rule_source = _rule_assignee(desc, rules, participants)
            if rule_assignee:
                assigned = rule_assignee
                source = rule_source
                confidence = 0.6

        # 6) Round robin fallback if participants exist.
        if not assigned and participants:
            p = participants[rr_index % len(participants)]
            assigned = p.email or p.name
            rr_index += 1
            source = "round_robin"
            confidence = 0.25

        # Keep tasks unassigned if there is nobody to assign to.
        t["assignee"] = assigned
        t["assignee_source"] = source
        t["assignment_confidence"] = round(confidence, 3) if confidence else None
        if speaker_resolved:
            t["speaker_resolved"] = speaker_resolved

    return tasks
