#!/usr/bin/env python3
"""Continuously merge completed shard episode reports into global reports."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from merge_shard_reports import comparable_config, episode_key
from run_yes_no_vlm_eval import write_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--done-file", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--num-shards", type=int, required=True)
    return parser.parse_args()


def load_config(shard_root: Path, num_shards: int) -> dict[str, Any]:
    configs = []
    for path in sorted(shard_root.glob("shard_*/run_config.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            configs.append(value)
    if not configs:
        return {"parallel_workers": num_shards, "num_shards": num_shards, "live": True}
    base = comparable_config(configs[0])
    base.update(
        parallel_workers=num_shards,
        gpus=[str(item.get("gpu") or "") for item in configs],
        shard_id=None,
        num_shards=num_shards,
        live=True,
    )
    return base


def load_completed(shard_root: Path) -> list[dict[str, Any]]:
    by_episode: dict[str, dict[str, Any]] = {}
    paths = sorted(shard_root.glob("shard_*/episode_reports/episode*.json"))
    for path in paths:
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A shard may be between opening and finishing a report write.
            continue
        if not isinstance(result, dict) or not result.get("episode"):
            continue
        by_episode[str(result["episode"])] = result
    return sorted(by_episode.values(), key=episode_key)


def result_signature(results: list[dict[str, Any]]) -> tuple[tuple[str, str], ...]:
    return tuple((str(item.get("episode")), str(item.get("status"))) for item in results)


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("interval must be positive")
    shard_root = args.shard_root.resolve()
    output_root = args.output_root.resolve()
    done_file = args.done_file.resolve()
    previous: tuple[tuple[str, str], ...] | None = None
    while True:
        results = load_completed(shard_root)
        signature = result_signature(results)
        if signature != previous:
            write_reports(output_root, results, load_config(shard_root, args.num_shards))
            print(f"[live-report] completed={len(results)} csv={output_root / 'report.csv'}", flush=True)
            previous = signature
        if done_file.exists():
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
