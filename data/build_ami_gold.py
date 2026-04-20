from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

from xml.etree import ElementTree as ET

TURN_GAP_SEC = 1.2


def meeting_ref_from_audio(audio_path: Path) -> str:
    stem = audio_path.stem  # ES2002a.Mix-Headset
    stem = re.sub(r"\.Mix-Headset$", "", stem)
    stem = re.sub(r"\.Headset$", "", stem)
    return stem


def meeting_ref_from_words_file(xml_path: Path) -> str:
    # ES2002a.A.words.xml -> ES2002a
    stem = xml_path.stem
    m = re.match(r"^(?P<meeting>.+?)\.[A-D]\.words$", stem)
    if m:
        return m.group("meeting")
    return stem.split(".")[0]


def speaker_from_words_file(xml_path: Path) -> str:
    stem = xml_path.stem
    m = re.match(r"^(?P<meeting>.+?)\.(?P<speaker>[A-D])\.words$", stem)
    if m:
        return m.group("speaker")
    parts = stem.split(".")
    return parts[1] if len(parts) > 1 else "UNK"


def parse_words_xml(xml_path: Path) -> list[dict[str, Any]]:
    root = ET.parse(xml_path).getroot()
    speaker = speaker_from_words_file(xml_path)
    tokens: list[dict[str, Any]] = []

    for elem in root.iter():
        if not elem.tag.endswith("w"):
            continue

        text = (elem.text or "").strip()
        if not text:
            continue

        start = float(elem.attrib.get("starttime", "0") or 0)
        end = float(elem.attrib.get("endtime", start) or start)
        is_punc = elem.attrib.get("punc") == "true"

        tokens.append(
            {
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text,
                "is_punc": is_punc,
            }
        )

    return tokens


def tokens_to_text(tokens: list[dict[str, Any]]) -> str:
    out = ""
    for tok in tokens:
        text = tok["text"]
        if tok["is_punc"]:
            out = out.rstrip() + text
        else:
            if out and not out.endswith((" ", "\n")):
                out += " "
            out += text
    return out.strip()


def build_transcript(all_tokens: list[dict[str, Any]]) -> str:
    all_tokens = sorted(all_tokens, key=lambda x: (x["start"], x["end"]))
    turns: list[tuple[str, list[dict[str, Any]]]] = []

    current_speaker = None
    current_tokens: list[dict[str, Any]] = []
    last_end = None

    def flush():
        nonlocal current_tokens, current_speaker
        if current_tokens and current_speaker is not None:
            turns.append((current_speaker, current_tokens))
        current_tokens = []

    for tok in all_tokens:
        speaker = tok["speaker"]
        if (
            current_speaker is None
            or speaker != current_speaker
            or (last_end is not None and tok["start"] - last_end > TURN_GAP_SEC)
        ):
            flush()
            current_speaker = speaker

        current_tokens.append(tok)
        last_end = tok["end"]

    flush()

    lines = []
    for speaker, toks in turns:
        text = tokens_to_text(toks)
        if text:
            lines.append(f"{speaker}: {text}")

    return "\n".join(lines).strip()


def load_backend_extractor():
    """
    Пытается использовать extractor из текущего backend проекта.
    Если импорт не удался, вернёт None, и тогда включится rule-based fallback.
    """
    repo_root = Path(__file__).resolve().parents[1]
    backend_dir = repo_root / "backend"
    if backend_dir.exists():
        sys.path.insert(0, str(backend_dir))

    try:
        from app.services.task_extraction import extract_tasks as backend_extract_tasks
        return backend_extract_tasks
    except Exception:
        return None


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def heuristic_tasks(transcript: str) -> list[dict[str, Any]]:
    """
    Очень простой fallback.
    Основная ставка — на backend extractor.
    """
    cue_patterns = [
        r"\bneed to\b",
        r"\bshould\b",
        r"\bmust\b",
        r"\bplease\b",
        r"\baction item\b",
        r"\btask\b",
        r"\bresponsible\b",
        r"\bby\s+\w+",
    ]
    cue_re = re.compile("|".join(cue_patterns), re.IGNORECASE)

    tasks: list[dict[str, Any]] = []
    seen = set()

    for sent in split_sentences(transcript):
        if len(sent.split()) < 5:
            continue
        if not cue_re.search(sent):
            continue

        desc = sent[:500].strip()
        key = desc.lower()[:80]
        if key in seen:
            continue
        seen.add(key)

        tasks.append(
            {
                "description": desc,
                "assignee_hint": None,
                "deadline_hint": None,
            }
        )

    return tasks


async def extract_tasks_auto(transcript: str, mode: str = "backend") -> list[dict[str, Any]]:
    if mode == "backend":
        backend_extract_tasks = load_backend_extractor()
        if backend_extract_tasks is not None:
            try:
                return await backend_extract_tasks(transcript)
            except Exception as e:
                print(f"[WARN] backend extractor failed, fallback to heuristic: {e}")

    return heuristic_tasks(transcript)


def build_record(audio_path: Path, words_files: list[Path], tasks: list[dict[str, Any]]) -> dict[str, Any]:
    all_tokens: list[dict[str, Any]] = []
    max_end = 0.0

    for xml_path in words_files:
        tokens = parse_words_xml(xml_path)
        all_tokens.extend(tokens)
        for t in tokens:
            if float(t["end"]) > max_end:
                max_end = float(t["end"])

    transcript = build_transcript(all_tokens)

    meeting_ref = meeting_ref_from_audio(audio_path)
    return {
        "meeting_ref": meeting_ref,
        "transcript": transcript,
        "transcript_source": "ami",
        "tasks": tasks,
        "audio_file_path": str(audio_path.as_posix()),
        "language": "en",
        "duration_sec": round(max_end, 2) if max_end else None,
        "notes": "auto-generated from AMI XML transcripts; treat as silver/pseudo-gold unless manually checked",
    }


def group_ami_files(audio_dir: Path, words_dir: Path) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}

    for audio_path in sorted(audio_dir.glob("*.wav")):
        ref = meeting_ref_from_audio(audio_path)
        groups.setdefault(ref, {"audio": audio_path, "words": []})

    for xml_path in sorted(words_dir.glob("*.xml")):
        ref = meeting_ref_from_words_file(xml_path)
        if ref not in groups:
            # Если аудио ещё не найдено, создадим группу всё равно
            groups.setdefault(ref, {"audio": None, "words": []})
        groups[ref]["words"].append(xml_path)

    return groups


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", required=True, help="Папка с wav файлами, например ES2002a.Mix-Headset.wav")
    parser.add_argument("--words-dir", required=True, help="Папка с XML транскриптами, например ES2002a.A.words.xml")
    parser.add_argument("--out-dir", required=True, help="Куда сохранять JSON")
    parser.add_argument(
        "--task-mode",
        choices=["backend", "rules"],
        default="backend",
        help="backend = использовать extractor из проекта; rules = только эвристика",
    )
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir).resolve()
    words_dir = Path(args.words_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = group_ami_files(audio_dir, words_dir)

    for ref, item in sorted(groups.items()):
        audio_path = item["audio"]
        words_files = item["words"]

        if audio_path is None:
            print(f"[SKIP] {ref}: audio file not found")
            continue
        if not words_files:
            print(f"[SKIP] {ref}: words XML not found")
            continue

        # Если есть несколько XML-файлов по спикерам — всё объединяем в один transcript.
        # Для задач используем общий текст.
        all_tokens = []
        for xml_path in words_files:
            all_tokens.extend(parse_words_xml(xml_path))
        transcript = build_transcript(all_tokens)

        tasks = await extract_tasks_auto(transcript, mode=args.task_mode)

        record = build_record(audio_path, words_files, tasks)
        record["transcript"] = transcript

        out_path = out_dir / f"{ref}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] {ref}: {out_path.name} | words_files={len(words_files)} | tasks={len(tasks)}")

if __name__ == "__main__":
    asyncio.run(main())