"""
Smart task assignment engine:
- Direct name matching (participants)
- Speaker label matching (from diarization)
- Regex rules with capture groups
- Role-based lookup
- Round-robin fallback
- Fuzzy name matching (partial, case-insensitive)
"""
import re
import json
from typing import List, Optional
from difflib import get_close_matches
from ..models import Participant, Rule
from sqlmodel import Session, select


def load_participants(session: Session) -> List[Participant]:
    return session.exec(select(Participant)).all()


def load_rules(session: Session) -> List[Rule]:
    return session.exec(select(Rule).order_by(Rule.priority)).all()


def _match_name(participants: List[Participant], name_hint: Optional[str], threshold: float = 0.6) -> Optional[Participant]:
    """Find participant by name/email with fuzzy matching."""
    if not name_hint:
        return None

    hint_lower = name_hint.lower()

    # Exact substring match
    for p in participants:
        if p.name and hint_lower in p.name.lower():
            return p
        if p.email and hint_lower in p.email.lower():
            return p

    # Fuzzy match (edit distance)
    all_names = [p.name for p in participants if p.name]
    close = get_close_matches(name_hint, all_names, n=1, cutoff=0.5)
    if close:
        matched_name = close[0]
        for p in participants:
            if p.name == matched_name:
                return p

    return None


def _find_participant_by_name(participants: List[Participant], name_hint: Optional[str]) -> Optional[Participant]:
    """Compatibility wrapper for older tests/code paths."""
    return _match_name(participants, name_hint)


def _match_speaker_to_participant(speaker_label: str, participants: List[Participant]) -> Optional[Participant]:
    """
    Match speaker label (e.g. "Speaker 0", "SPEAKER_00") to participant.
    Strategy: use participant alias/tags or enrollment if stored.
    """
    if not speaker_label:
        return None

    # Extract number
    num_match = re.search(r"(\d+)", speaker_label)
    if not num_match:
        return None

    spk_id = int(num_match.group(1))

    # Look for participant with matching speaker_id in tags/metadata
    for p in participants:
        if p.tags:
            try:
                tags = json.loads(p.tags)
                if tags.get("speaker_id") == spk_id:
                    return p
            except json.JSONDecodeError:
                pass

    # If no explicit mapping, skip
    return None


def assign_tasks_to_participants(tasks: List[dict], session: Session) -> List[dict]:
    """
    Assign tasks to participants using multi-stage strategy.
    Modifies each task dict to add 'assignee' field.
    """
    participants = load_participants(session)
    rules = load_rules(session)
    assignments = []
    rr_index = 0

    for t in tasks:
        desc = t.get("description", "")
        hint = t.get("assignee_hint")
        assigned = None

        # 1) Direct name match from LLM hint
        if hint:
            p = _match_name(participants, hint)
            if p:
                assigned = p.email or p.name

        # 2) Look for participant names directly in description text
        if not assigned:
            for p in participants:
                if p.name and re.search(rf"\b{re.escape(p.name)}\b", desc, re.IGNORECASE):
                    assigned = p.email or p.name
                    break

        # 3) Regex rules with capture groups to extract name
        if not assigned:
            for r in rules:
                if r.kind == "regex" and r.pattern:
                    try:
                        m = re.search(r.pattern, desc, re.IGNORECASE)
                        if m:
                            # Try to get named group "assignee" or first group
                            if m.lastindex:
                                # Group 1 might be the assignee
                                extracted = m.group(1) if m.lastindex >= 1 else m.group(0)
                                # Remove punctuation
                                extracted = re.sub(r"[^\w\s@.-]", "", extracted).strip()
                                if extracted:
                                    # Try to match to participant
                                    p = _match_name(participants, extracted)
                                    if p:
                                        assigned = p.email or p.name
                                    else:
                                        assigned = extracted  # Raw value
                                else:
                                    assigned = r.pattern  # Fallback to pattern
                            else:
                                assigned = r.pattern
                            break
                    except re.error:
                        continue

        # 4) Role-based lookup
        if not assigned:
            for r in rules:
                if r.kind == "role_lookup" and r.pattern:
                    try:
                        obj = json.loads(r.pattern)
                        role = obj.get("role", "").lower()
                        default_assignee = obj.get("assignee")
                        if role and role in desc.lower():
                            assigned = default_assignee
                            break
                    except json.JSONDecodeError:
                        continue

        # 5) Round-robin fallback
        if not assigned and participants:
            p = participants[rr_index % len(participants)]
            assigned = p.email or p.name
            rr_index += 1

        t["assignee"] = assigned
        assignments.append(t)

    return assignments