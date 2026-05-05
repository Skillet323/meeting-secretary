from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import httpx


async def wait_for_completion(client: httpx.AsyncClient, base_url: str, meeting_id: int, poll_sec: float) -> dict:
    while True:
        resp = await client.get(f"{base_url}/meeting/{meeting_id}/progress")
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        progress = data.get("progress")
        stage = data.get("current_stage")
        msg = data.get("message")
        print(f"[{meeting_id}] {status} {progress}% | {stage} | {msg}")

        if status in {"completed", "failed"}:
            return data

        await asyncio.sleep(poll_sec)


async def upload_one(client: httpx.AsyncClient, base_url: str, path: Path, poll_sec: float) -> None:
    print(f"\nUploading: {path.name}")
    with path.open("rb") as f:
        files = {"file": (path.name, f, "audio/wav")}
        resp = await client.post(f"{base_url}/upload_meeting", files=files)
    resp.raise_for_status()
    data = resp.json()
    meeting_id = data["meeting_id"]
    print(f"Started meeting_id={meeting_id}")

    await wait_for_completion(client, base_url, meeting_id, poll_sec)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", required=True, help="Directory with audio files")
    parser.add_argument("--base-url", default=os.environ.get("API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--poll-sec", type=float, default=10.0)
    parser.add_argument("--pattern", default="*.wav", help="Glob pattern, default *.wav")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir).resolve()
    files = sorted(audio_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files found in {audio_dir} matching {args.pattern}")

    async with httpx.AsyncClient(timeout=60 * 60) as client:
        for path in files:
            try:
                await upload_one(client, args.base_url.rstrip("/"), path, args.poll_sec)
            except Exception as e:
                print(f"[ERR] {path.name}: {e}")


if __name__ == "__main__":
    asyncio.run(main())