#!/usr/bin/env python3
"""Persistent per-GPU Qwen caption worker for semantic_alignment."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.semantic_alignment import caption_video_entries, load_qwen_captioner  # noqa: E402

@contextlib.contextmanager
def _redirect_stdout_to_stderr():
    stdout_fd = sys.stdout.fileno()
    saved_fd = os.dup(stdout_fd)
    try:
        os.dup2(sys.stderr.fileno(), stdout_fd)
        yield
    finally:
        os.dup2(saved_fd, stdout_fd)
        os.close(saved_fd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--shard-path", type=Path, required=True)
    return parser.parse_args()


def _load_shard(path: Path) -> dict[str, object]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_shard(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    model, processor = load_qwen_captioner(args.model_path)
    shard = _load_shard(args.shard_path)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "STOP":
            break
        job = json.loads(line)
        episode = str(job.get("episode") or "")
        entries = job.get("entries") or []
        try:
            pending = []
            for entry in entries:
                key = str(entry["key"])
                if key in shard:
                    continue
                pending.append(entry)
            if pending:
                with _redirect_stdout_to_stderr():
                    updates = caption_video_entries(model, processor, pending)
                shard.update(updates)
                _save_shard(args.shard_path, shard)
            print(json.dumps({"status": "success", "episode": episode}), flush=True)
        except Exception as exc:
            print(json.dumps({
                "status": "failed",
                "episode": episode,
                "error": str(exc),
            }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
