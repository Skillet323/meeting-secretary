from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import httpx


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MAX_TOKENS = 900
DEFAULT_TIMEOUT = 60.0

SYSTEM_PROMPT = (
    "You are a strict action-item extractor for meeting transcripts.\n"
    "Return ONLY a valid JSON array. No prose. No markdown. No explanation.\n"
    "If no valid tasks exist, return [].\n\n"
    "Rules:\n"
    "- Extract only tasks that are clearly intended to happen AFTER the meeting.\n"
    "- Do NOT extract introductions, agenda items, discussion topics, or summaries.\n"
    "- Do NOT invent people, names, roles, deadlines, or tasks.\n"
    "- If the transcript is short, noisy, or mostly off-topic, prefer returning [].\n"
    "- Keep descriptions concrete and short.\n"
    "- assignee_hint must be a real name/role actually mentioned or strongly implied in the transcript.\n"
    "- deadline_hint must be explicit or very clearly implied.\n"
    "- speaker_hint, if present, must copy an actual speaker label from the transcript (e.g. SPEAKER_00, A). Do not turn it into a human name.\n\n"
    "Output schema for each item:\n"
    '{'
    '"description": "string", '
    '"assignee_hint": "string or null", '
    '"deadline_hint": "string or null", '
    '"speaker_hint": "string or null", '
    '"source_snippet": "string or null"'
    '}'
)

USER_TEMPLATE = (
    "Extract action items from the meeting transcript below.\n\n"
    "Meeting ID: {meeting_ref}\n"
    "Language: {language}\n"
    "Duration: {duration_sec:.1f} seconds\n\n"
    "Transcript:\n"
    "{transcript}\n\n"
    "Return only a JSON array."
)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "have", "has", "had",
    "he", "her", "him", "his", "i", "if", "in", "into", "is", "it", "its", "me", "my",
    "of", "on", "or", "our", "she", "so", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "to", "we", "were", "what", "when", "where", "which", "who",
    "will", "with", "would", "you", "your", "um", "uh", "okay", "ok", "right", "yeah",
    "just", "also", "still", "very", "really", "maybe", "please", "could", "should", "need",
    "needs", "want", "wanna",
}


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    return [t for t in text.split() if t and t not in STOPWORDS and len(t) > 2]


def is_meta_task(text: str) -> bool:
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


def parse_gold_dir(gold_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for fp in sorted(gold_dir.glob("*.json")):
        with fp.open("r", encoding="utf-8") as f:
            item = json.load(f)
        if not isinstance(item, dict):
            continue
        item["_source_file"] = fp.name
        records.append(item)
    return records


def _candidate_texts(raw: str) -> list[str]:
    candidates: list[str] = []
    raw = raw or ""

    for m in re.finditer(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE):
        block = m.group(1).strip()
        if block:
            candidates.append(block)

    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end > start:
        candidates.append(raw[start:end + 1].strip())

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start:end + 1].strip())

    candidates.append(raw.strip())
    return candidates


def _collect_tasks(node: Any, out: list[dict[str, Any]]) -> None:
    if isinstance(node, dict):
        desc = normalize_space(node.get("description") or node.get("task") or "")
        if desc and not is_meta_task(desc):
            out.append(
                {
                    "description": desc[:500],
                    "assignee_hint": node.get("assignee_hint") or node.get("assignee"),
                    "deadline_hint": node.get("deadline_hint") or node.get("deadline"),
                    "speaker_hint": node.get("speaker_hint") or node.get("speaker"),
                    "source_snippet": node.get("source_snippet") or node.get("evidence"),
                }
            )

        for key, value in node.items():
            if key in {
                "description", "task", "assignee_hint", "assignee",
                "deadline_hint", "deadline", "speaker_hint", "speaker",
                "source_snippet", "evidence",
            }:
                continue
            _collect_tasks(value, out)

    elif isinstance(node, list):
        for item in node:
            _collect_tasks(item, out)

    elif isinstance(node, str):
        text = normalize_space(node)
        if len(text) >= 10 and not is_meta_task(text):
            out.append(
                {
                    "description": text[:500],
                    "assignee_hint": None,
                    "deadline_hint": None,
                    "speaker_hint": None,
                    "source_snippet": None,
                }
            )


def parse_tasks_from_raw(raw: str) -> tuple[list[dict[str, Any]], str]:
    for candidate in _candidate_texts(raw):
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue

        tasks: list[dict[str, Any]] = []
        _collect_tasks(parsed, tasks)

        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for t in tasks:
            desc = normalize_space(t.get("description", ""))
            key = normalize_text(desc)[:120]
            if not desc or key in seen:
                continue
            seen.add(key)
            unique.append(t)

        if unique:
            return unique, "json"

    tasks = []
    for line in raw.splitlines():
        line = normalize_space(line)
        if not line or len(line) < 10:
            continue
        low = normalize_text(line)
        if low in {"action items", "task", "tasks", "extracted action items"}:
            continue
        if low.startswith("extracted action items"):
            continue
        if line[0].isdigit():
            line = re.sub(r"^\d+[\).\-\s]+", "", line).strip()
        if len(line) >= 10 and not is_meta_task(line):
            tasks.append(
                {
                    "description": line[:500],
                    "assignee_hint": None,
                    "deadline_hint": None,
                    "speaker_hint": None,
                    "source_snippet": None,
                }
            )

    unique = []
    seen = set()
    for t in tasks:
        key = normalize_text(t["description"])[:120]
        if key and key not in seen:
            seen.add(key)
            unique.append(t)

    return unique, "line"


def _sim(a: str, b: str) -> float:
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    if not a_n or not b_n:
        return 0.0

    seq = SequenceMatcher(None, a_n, b_n).ratio()
    a_tokens = set(tokenize(a_n))
    b_tokens = set(tokenize(b_n))
    if not a_tokens or not b_tokens:
        return seq

    jacc = len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))
    return 0.55 * jacc + 0.45 * seq


def evaluate_tasks(
    pred_tasks: list[dict[str, Any]],
    gold_tasks: list[dict[str, Any]],
    threshold: float = 0.28,
) -> dict[str, Any]:
    pred_tasks = pred_tasks or []
    gold_tasks = gold_tasks or []

    matches: dict[int, int] = {}
    used_gold: set[int] = set()
    gold_descs = [normalize_space(g.get("description") or g.get("task") or "") for g in gold_tasks]

    for i, pred in enumerate(pred_tasks):
        pdesc = normalize_space(pred.get("description") or pred.get("task") or "")
        best_idx, best_score = -1, 0.0
        for j, gdesc in enumerate(gold_descs):
            if j in used_gold:
                continue
            score = _sim(pdesc, gdesc)
            if score > best_score:
                best_idx, best_score = j, score
        if best_idx >= 0 and best_score >= threshold:
            matches[i] = best_idx
            used_gold.add(best_idx)

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

    for pred_idx, gold_idx in matches.items():
        pred = pred_tasks[pred_idx]
        gold = gold_tasks[gold_idx]
        pred_assignee = normalize_text(pred.get("assignee_hint") or pred.get("assignee") or "")
        gold_assignee = normalize_text(gold.get("assignee_hint") or gold.get("assignee") or "")
        pred_deadline = normalize_text(pred.get("deadline_hint") or pred.get("deadline") or "")
        gold_deadline = normalize_text(gold.get("deadline_hint") or gold.get("deadline") or "")

        if pred_assignee or gold_assignee:
            assignment_checks += 1
            if not pred_assignee and not gold_assignee:
                assignee_correct += 1
            elif pred_assignee and gold_assignee and (
                pred_assignee == gold_assignee
                or pred_assignee in gold_assignee
                or gold_assignee in pred_assignee
            ):
                assignee_correct += 1

        if pred_deadline or gold_deadline:
            deadline_checks += 1
            if not pred_deadline and not gold_deadline:
                deadline_correct += 1
            elif pred_deadline and gold_deadline and (
                pred_deadline == gold_deadline
                or pred_deadline in gold_deadline
                or gold_deadline in pred_deadline
            ):
                deadline_correct += 1

    assignee_accuracy = assignee_correct / assignment_checks if assignment_checks else None
    deadline_accuracy = deadline_correct / deadline_checks if deadline_checks else None
    hallucination_rate = sum(
        1 for t in pred_tasks if len(normalize_space(t.get("description", "")).split()) < 3
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
        "assignment_checks": assignment_checks,
        "deadline_checks": deadline_checks,
    }


@dataclass
class RequestResult:
    status: str
    http_status: int | None = None
    error: str | None = None
    retry_after: float | None = None
    response_model: str | None = None
    raw: str | None = None
    raw_preview: str | None = None
    parse_stage: str | None = None
    tasks: list[dict[str, Any]] = dataclasses.field(default_factory=list)


def _openrouter_post(client: httpx.Client, api_key: str, payload: dict[str, Any]) -> httpx.Response:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://example.com"),
        "X-Title": os.getenv("OPENROUTER_APP_TITLE", "Meeting Secretary Benchmark"),
    }
    return client.post(OPENROUTER_URL, json=payload, headers=headers)


def probe_model(client: httpx.Client, api_key: str, model: str) -> RequestResult:
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 1,
        "messages": [
            {"role": "system", "content": "Reply with OK."},
            {"role": "user", "content": "OK"},
        ],
    }

    try:
        resp = _openrouter_post(client, api_key, payload)
    except Exception as exc:
        return RequestResult(status="error", error=str(exc))

    if resp.status_code == 429:
        retry_after = resp.headers.get("Retry-After")
        retry_after_val = None
        if retry_after:
            try:
                retry_after_val = float(retry_after)
            except ValueError:
                retry_after_val = None
        return RequestResult(status="rate_limited", http_status=429, error="rate_limited", retry_after=retry_after_val)

    if resp.status_code == 403:
        return RequestResult(status="forbidden", http_status=403, error="forbidden")

    if resp.status_code != 200:
        return RequestResult(status="error", http_status=resp.status_code, error=resp.text[:300])

    try:
        data = resp.json()
    except Exception as exc:
        return RequestResult(status="error", http_status=200, error=f"bad_json:{exc}")

    response_model = data.get("model") or model
    return RequestResult(status="ok", http_status=200, response_model=response_model)


def request_tasks(
    client: httpx.Client,
    api_key: str,
    model: str,
    transcript: str,
    *,
    meeting_ref: str,
    language: str = "en",
    duration_sec: float | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    retries: int = 2,
) -> RequestResult:
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    meeting_ref=meeting_ref or "unknown",
                    language=language or "en",
                    duration_sec=duration_sec or 0.0,
                    transcript=(transcript or "")[:8000],
                ),
            },
        ],
    }

    for attempt in range(retries + 1):
        try:
            resp = _openrouter_post(client, api_key, payload)
        except Exception as exc:
            if attempt >= retries:
                return RequestResult(status="error", error=str(exc))
            time.sleep(1.0 + attempt)
            continue

        if resp.status_code == 429:
            retry_after_hdr = resp.headers.get("Retry-After")
            retry_after = None
            if retry_after_hdr:
                try:
                    retry_after = float(retry_after_hdr)
                except ValueError:
                    retry_after = None
            return RequestResult(
                status="rate_limited",
                http_status=429,
                error="rate_limited",
                retry_after=retry_after,
            )

        if resp.status_code == 403:
            return RequestResult(status="forbidden", http_status=403, error="forbidden")

        if resp.status_code in (408, 502, 503, 529) or resp.status_code >= 500:
            if attempt < retries:
                retry_after_hdr = resp.headers.get("Retry-After")
                if retry_after_hdr:
                    try:
                        time.sleep(float(retry_after_hdr))
                    except ValueError:
                        time.sleep(1.0)
                else:
                    time.sleep(1.5 * (attempt + 1))
                continue
            return RequestResult(status="error", http_status=resp.status_code, error=f"http_{resp.status_code}")

        if resp.status_code != 200:
            return RequestResult(status="error", http_status=resp.status_code, error=resp.text[:300])

        try:
            data = resp.json()
        except Exception as exc:
            return RequestResult(status="error", http_status=200, error=f"bad_json:{exc}")

        response_model = data.get("model") or model
        raw = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        tasks, parse_stage = parse_tasks_from_raw(raw)

        return RequestResult(
            status="ok" if tasks else "parse_failed",
            http_status=200,
            response_model=response_model,
            raw=raw,
            raw_preview=normalize_space(raw)[:1000],
            parse_stage=parse_stage,
            tasks=tasks,
        )

    return RequestResult(status="error", error="unknown_error")


def summarize_numeric(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def fmt_float(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def benchmark_model(
    client: httpx.Client,
    api_key: str,
    model: str,
    records: list[dict[str, Any]],
    *,
    delay_sec: float,
    preflight: bool = True,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    retries: int = 2,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []

    if preflight:
        probe = probe_model(client, api_key, model)
        if probe.status in {"rate_limited", "forbidden", "error"}:
            return (
                {
                    "model": model,
                    "response_model": None,
                    "status": probe.status,
                    "n_total": len(records),
                    "n_ok": 0,
                    "n_failed": 0,
                    "n_rate_limited": 1 if probe.status == "rate_limited" else 0,
                    "n_forbidden": 1 if probe.status == "forbidden" else 0,
                    "n_parse_failed": 0,
                    "avg_task_f1": None,
                    "avg_precision": None,
                    "avg_recall": None,
                    "avg_assignee_accuracy": None,
                    "avg_deadline_accuracy": None,
                    "avg_hallucination_rate": None,
                    "avg_predicted_tasks": None,
                    "avg_matched_tasks": None,
                    "avg_latency_sec": None,
                    "avg_raw_chars": None,
                    "notes": probe.error or "preflight_failed",
                },
                [],
            )

    response_models: list[str] = []
    metrics_acc = defaultdict(list)

    for idx, rec in enumerate(records, start=1):
        transcript = rec.get("transcript", "") or ""
        gold_tasks = rec.get("tasks", []) or []
        meeting_ref = rec.get("meeting_ref") or rec.get("id") or rec.get("_source_file") or f"row_{idx}"
        duration_sec = rec.get("duration_sec")
        language = rec.get("language", "en")

        time.sleep(max(0.0, delay_sec))
        started = time.time()
        rr = request_tasks(
            client,
            api_key,
            model,
            transcript,
            meeting_ref=str(meeting_ref),
            language=language,
            duration_sec=duration_sec,
            max_tokens=max_tokens,
            retries=retries,
        )
        latency = time.time() - started
        response_models.append(rr.response_model or model)

        row = {
            "model": model,
            "response_model": rr.response_model or model,
            "meeting_ref": str(meeting_ref),
            "source_file": rec.get("_source_file"),
            "status": rr.status,
            "http_status": rr.http_status,
            "error": rr.error,
            "parse_stage": rr.parse_stage,
            "predicted_tasks": len(rr.tasks),
            "gold_tasks": len(gold_tasks),
            "raw_preview": rr.raw_preview,
            "latency_sec": fmt_float(latency, 3),
            "raw_chars": len(rr.raw or ""),
        }

        if rr.status == "ok":
            task_metrics = evaluate_tasks(rr.tasks, gold_tasks)
            row.update(task_metrics)

            metrics_acc["task_set_f1"].append(task_metrics["task_set_f1"])
            metrics_acc["task_set_precision"].append(task_metrics["task_set_precision"])
            metrics_acc["task_set_recall"].append(task_metrics["task_set_recall"])
            metrics_acc["assignee_accuracy"].append(task_metrics["assignee_accuracy"])
            metrics_acc["deadline_accuracy"].append(task_metrics["deadline_accuracy"])
            metrics_acc["hallucination_rate"].append(task_metrics["hallucination_rate"])
            metrics_acc["predicted_tasks"].append(task_metrics["predicted_tasks"])
            metrics_acc["matched_tasks"].append(task_metrics["matched_tasks"])
            metrics_acc["latency_sec"].append(latency)
            metrics_acc["raw_chars"].append(len(rr.raw or ""))

        elif rr.status == "rate_limited":
            row["note"] = "stopped_after_rate_limit"
            rows.append(row)
            break

        rows.append(row)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    parse_failed = sum(1 for r in rows if r.get("status") == "parse_failed")
    rate_limited = sum(1 for r in rows if r.get("status") == "rate_limited")
    forbidden = sum(1 for r in rows if r.get("status") == "forbidden")
    failed = sum(1 for r in rows if r.get("status") == "error")

    summary = {
        "model": model,
        "response_model": response_models[-1] if response_models else None,
        "status": "ok" if ok_rows else ("rate_limited" if rate_limited else ("forbidden" if forbidden else "failed")),
        "n_total": len(records),
        "n_ok": len(ok_rows),
        "n_failed": failed,
        "n_rate_limited": rate_limited,
        "n_forbidden": forbidden,
        "n_parse_failed": parse_failed,
        "avg_task_f1": fmt_float(summarize_numeric(metrics_acc["task_set_f1"]), 4),
        "avg_precision": fmt_float(summarize_numeric(metrics_acc["task_set_precision"]), 4),
        "avg_recall": fmt_float(summarize_numeric(metrics_acc["task_set_recall"]), 4),
        "avg_assignee_accuracy": fmt_float(summarize_numeric([v for v in metrics_acc["assignee_accuracy"] if v is not None]), 4),
        "avg_deadline_accuracy": fmt_float(summarize_numeric([v for v in metrics_acc["deadline_accuracy"] if v is not None]), 4),
        "avg_hallucination_rate": fmt_float(summarize_numeric(metrics_acc["hallucination_rate"]), 4),
        "avg_predicted_tasks": fmt_float(summarize_numeric(metrics_acc["predicted_tasks"]), 4),
        "avg_matched_tasks": fmt_float(summarize_numeric(metrics_acc["matched_tasks"]), 4),
        "avg_latency_sec": fmt_float(summarize_numeric(metrics_acc["latency_sec"]), 3),
        "avg_raw_chars": fmt_float(summarize_numeric(metrics_acc["raw_chars"]), 1),
        "notes": None,
    }

    if rate_limited:
        summary["notes"] = "rate_limited"
    elif forbidden:
        summary["notes"] = "forbidden"
    elif parse_failed and not ok_rows:
        summary["notes"] = "all_parse_failed"

    return summary, rows


def print_summary_table(summary: list[dict[str, Any]]) -> None:
    if not summary:
        print("No results.")
        return

    headers = [
        "model",
        "status",
        "n_ok",
        "n_total",
        "avg_f1",
        "avg_prec",
        "avg_rec",
        "avg_assignee",
        "avg_deadline",
        "rate_limited",
        "forbidden",
    ]
    widths = {h: len(h) for h in headers}
    rows = []
    for s in summary:
        row = {
            "model": s["model"],
            "status": s["status"],
            "n_ok": str(s["n_ok"]),
            "n_total": str(s["n_total"]),
            "avg_f1": "" if s["avg_task_f1"] is None else f'{s["avg_task_f1"]:.4f}',
            "avg_prec": "" if s["avg_precision"] is None else f'{s["avg_precision"]:.4f}',
            "avg_rec": "" if s["avg_recall"] is None else f'{s["avg_recall"]:.4f}',
            "avg_assignee": "" if s["avg_assignee_accuracy"] is None else f'{s["avg_assignee_accuracy"]:.4f}',
            "avg_deadline": "" if s["avg_deadline_accuracy"] is None else f'{s["avg_deadline_accuracy"]:.4f}',
            "rate_limited": str(s["n_rate_limited"]),
            "forbidden": str(s["n_forbidden"]),
        }
        rows.append(row)
        for h in headers:
            widths[h] = max(widths[h], len(row[h]))

    def fmt_row(row: dict[str, str]) -> str:
        return " | ".join(row[h].ljust(widths[h]) for h in headers)

    print(fmt_row({h: h for h in headers}))
    print("-+-".join("-" * widths[h] for h in headers))
    for row in rows:
        print(fmt_row(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare task extraction models on gold annotations.")
    parser.add_argument("--gold-dir", required=True, help="Folder with gold JSON files")
    parser.add_argument("--models", required=True, help="Comma-separated OpenRouter model slugs")
    parser.add_argument("--out-csv", required=True, help="Output CSV summary path")
    parser.add_argument("--out-json", required=True, help="Output JSON report path")
    parser.add_argument("--delay-sec", type=float, default=3.0, help="Delay between API requests")
    parser.add_argument("--retries", type=int, default=2, help="Retries for transient errors per request")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens for completion")
    parser.add_argument("--no-preflight", action="store_true", help="Disable preflight availability check")
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    gold_dir = Path(args.gold_dir).expanduser().resolve()
    records = parse_gold_dir(gold_dir)
    if not records:
        raise SystemExit(f"No JSON files found in {gold_dir}")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise SystemExit("No models provided")

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    detailed_runs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    timeout = httpx.Timeout(DEFAULT_TIMEOUT, connect=20.0)
    limits = httpx.Limits(max_keepalive_connections=2, max_connections=4)

    with httpx.Client(timeout=timeout, limits=limits) as client:
        for model in models:
            print(f"\n=== {model} ===")
            summary, runs = benchmark_model(
                client,
                api_key,
                model,
                records,
                delay_sec=args.delay_sec,
                preflight=not args.no_preflight,
                max_tokens=args.max_tokens,
                retries=args.retries,
            )
            summaries.append(summary)
            detailed_runs.extend(runs)

            if summary["status"] in {"rate_limited", "forbidden"}:
                print(f"[SKIP] {model}: {summary['status']}")
            else:
                print(
                    f"[DONE] {model}: ok={summary['n_ok']}/{summary['n_total']} "
                    f"f1={summary['avg_task_f1']} prec={summary['avg_precision']} rec={summary['avg_recall']}"
                )

            time.sleep(max(0.0, args.delay_sec))

    summaries_sorted = sorted(
        summaries,
        key=lambda s: (s["avg_task_f1"] or 0.0, s["avg_precision"] or 0.0, s["avg_recall"] or 0.0),
        reverse=True,
    )

    report = {
        "generated_at": started_at,
        "gold_dir": str(gold_dir),
        "records": len(records),
        "models_requested": models,
        "settings": {
            "delay_sec": args.delay_sec,
            "retries": args.retries,
            "max_tokens": args.max_tokens,
            "preflight": not args.no_preflight,
        },
        "summary": summaries_sorted,
        "runs": detailed_runs,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "model",
            "status",
            "n_ok",
            "n_total",
            "avg_task_f1",
            "avg_precision",
            "avg_recall",
            "avg_assignee_accuracy",
            "avg_deadline_accuracy",
            "avg_hallucination_rate",
            "avg_predicted_tasks",
            "avg_matched_tasks",
            "avg_latency_sec",
            "avg_raw_chars",
            "n_rate_limited",
            "n_forbidden",
            "n_parse_failed",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries_sorted:
            writer.writerow({k: s.get(k) for k in fieldnames})

    print("\n=== FINAL SUMMARY ===")
    print_summary_table(summaries_sorted)
    print(f"\nCSV: {out_csv}")
    print(f"JSON: {out_json}")


if __name__ == "__main__":
    main()