from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.db import init_db, Session as DBSession, engine  # noqa: E402
from app.models import GoldStandard  # noqa: E402
from sqlmodel import select  # noqa: E402


def load_json_files(gold_dir: Path) -> list[Path]:
    return sorted([p for p in gold_dir.glob("*.json") if p.is_file()])


def import_one(session, path: Path, replace: bool = False) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    meeting_ref = str(data.get("meeting_ref") or "").strip()
    if not meeting_ref:
        return f"[SKIP] {path.name}: missing meeting_ref"

    existing = session.exec(select(GoldStandard).where(GoldStandard.meeting_ref == meeting_ref)).first()
    if existing and not replace:
        return f"[SKIP] {path.name}: meeting_ref={meeting_ref} already exists"

    if existing and replace:
        session.delete(existing)
        session.commit()

    gold = GoldStandard(
        meeting_ref=meeting_ref,
        transcript=str(data.get("transcript") or ""),
        transcript_source=data.get("transcript_source") or data.get("source", "manual"),
        tasks_json=json.dumps(data.get("tasks", []), ensure_ascii=False),
        audio_file_path=data.get("audio_file_path"),
        language=data.get("language", "en"),
        duration_sec=data.get("duration_sec"),
        notes=data.get("notes"),
    )
    session.add(gold)
    session.commit()
    return f"[OK] {path.name}: imported as meeting_ref={meeting_ref}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-dir", default="gold_annotations", help="Folder with *.json gold files")
    parser.add_argument("--replace", action="store_true", help="Replace existing DB gold rows")
    args = parser.parse_args()

    init_db()

    gold_dir = Path(args.gold_dir).resolve()
    files = load_json_files(gold_dir)
    if not files:
        print(f"No JSON files found in {gold_dir}")
        return

    with DBSession(engine) as session:
        for path in files:
            try:
                print(import_one(session, path, replace=args.replace))
            except Exception as e:
                print(f"[ERR] {path.name}: {e}")


if __name__ == "__main__":
    main()