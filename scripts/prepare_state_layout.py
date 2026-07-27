#!/usr/bin/env python3
"""Build flat episode STATE layout expected by protocol_0710 from flat STATE files."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metrics._common.episode_sort import episode_sort_key, sort_episode_names
EP_RE = re.compile(r"^episode(\d+)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-root",
        type=Path,
        default=PROJECT_ROOT / "test_data" / "STATE",
        help="Flat STATE root (episodeN.json files).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output with flat episodeN.json layout.",
    )
    parser.add_argument(
        "--episodes",
        nargs="*",
        default=None,
        help="Only link STATE for these episode ids (default: all under state-root).",
    )
    parser.add_argument(
        "--episodes-file",
        type=Path,
        default=None,
        help="JSON file with episode list (manifest records or plain list).",
    )
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def load_episode_filter(args: argparse.Namespace) -> set[str] | None:
    if args.episodes:
        return set(args.episodes)
    if args.episodes_file and args.episodes_file.is_file():
        payload = json.loads(args.episodes_file.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return {str(item) for item in payload}
        if isinstance(payload, dict):
            if isinstance(payload.get("records"), list):
                return {
                    str(item["episode"])
                    for item in payload["records"]
                    if isinstance(item, dict) and item.get("episode")
                }
            if isinstance(payload.get("entries"), list):
                return {
                    str(item["episode"])
                    for item in payload["entries"]
                    if isinstance(item, dict) and item.get("episode")
                }
    return None


def main() -> int:
    args = parse_args()
    state_root = args.state_root.resolve()
    output_root = args.output_root.resolve()
    episode_filter = load_episode_filter(args)

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0
    for path in sorted(state_root.glob("episode*.json"), key=lambda p: episode_sort_key(p.stem)):
        episode = path.stem
        if not EP_RE.match(episode):
            continue
        if episode_filter is not None and episode not in episode_filter:
            skipped += 1
            continue
        dst = output_root / f"{episode}.json"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(path.resolve(), dst)
        count += 1

    if episode_filter is not None:
        missing = sort_episode_names(episode_filter - {p.stem for p in output_root.glob("episode*.json")})
        if missing:
            print(
                f"[warn] {len(missing)} episode(s) missing STATE: {missing[:5]}{'...' if len(missing) > 5 else ''}",
                file=sys.stderr,
            )

    print(f"[ok] {count} STATE files -> {output_root}" + (f" (skipped {skipped})" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
