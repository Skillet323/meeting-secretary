from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sqlmodel import Session, select

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.db import engine, init_db  # noqa: E402
from app.models import GoldStandard  # noqa: E402
from app.services.evaluation import create_gold_standard  # noqa: E402


def upsert_gold(session: Session, data: dict) -> str:
    meeting_ref = data["meeting_ref"]
    existing = session.exec(
        select(GoldStandard).where(GoldStandard.meeting_ref == meeting_ref)
    ).first()

    if existing:
        existing.transcript = data["transcript"]
        existing.transcript_source = data.get("transcript_source", "ami")
        existing.tasks_json = json.dumps(data.get("tasks", []), ensure_ascii=False)
        existing.audio_file_path = data.get("audio_file_path")
        existing.language = data.get("language", "en")
        existing.duration_sec = data.get("duration_sec")
        existing.notes = data.get("notes")
        session.add(existing)
        session.commit()
        return f"updated gold {meeting_ref} (id={existing.id})"

    gold = create_gold_standard(session, data)
    return f"created gold {meeting_ref} (id={gold.id})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Папка с JSON-файлами, созданными build_ami_gold.py")
    args = parser.parse_args()

    init_db()
    json_dir = Path(args.dir).resolve()

    files = sorted(json_dir.glob("*.json"))
    if not files:
        print(f"No JSON files found in {json_dir}")
        return

    with Session(engine) as session:
        for path in files:
            data = json.loads(path.read_text(encoding="utf-8"))
            msg = upsert_gold(session, data)
            print(f"[OK] {path.name}: {msg}")


if __name__ == "__main__":
    main()