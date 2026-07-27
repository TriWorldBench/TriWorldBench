#!/usr/bin/env python3
"""Merge isolated VLM evaluation shard outputs into one report directory."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from run_yes_no_vlm_eval import write_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--started-at-epoch", type=float)
    return parser.parse_args()


def format_duration(total_seconds: float) -> str:
    rounded_seconds = int(round(total_seconds))
    hours, remainder = divmod(rounded_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def episode_key(result: dict[str, Any]) -> tuple[int, str]:
    episode = str(result.get("episode") or "")
    match = re.search(r"(\d+)$", episode)
    return (int(match.group(1)) if match else 10**12, episode)


def comparable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key not in {"gpu", "shard_id", "num_shards"}}


def copy_episode_artifacts(shard_dir: Path, output_root: Path, result: dict[str, Any]) -> None:
    episode = str(result["episode"])
    source_raw = shard_dir / "raw_responses" / f"{episode}.json"
    if source_raw.is_file():
        destination = output_root / "raw_responses" / source_raw.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_raw, destination)
        result["raw_response_path"] = str(destination)
    source_cot = shard_dir / "cot_logs" / f"{episode}.json"
    if source_cot.is_file():
        destination = output_root / "cot_logs" / source_cot.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_cot, destination)
        result["cot_log_path"] = str(destination)
    destination_report = output_root / "episode_reports" / f"{episode}.json"
    destination_report.parent.mkdir(parents=True, exist_ok=True)
    destination_report.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    shard_root = args.shard_root.resolve()
    output_root = args.output_root.resolve()
    report_paths = sorted(shard_root.glob("shard_*/report.json"))
    if not report_paths:
        raise FileNotFoundError(f"no shard reports found under {shard_root}")

    merged_results: list[dict[str, Any]] = []
    seen_episodes: set[str] = set()
    base_config: dict[str, Any] | None = None
    shard_gpus: list[str] = []
    for report_path in report_paths:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        config = report.get("config")
        episodes = report.get("episodes")
        if not isinstance(config, dict) or not isinstance(episodes, list):
            raise ValueError(f"invalid shard report: {report_path}")
        comparable = comparable_config(config)
        shard_gpus.append(str(config.get("gpu") or ""))
        if base_config is None:
            base_config = comparable
        elif comparable != base_config:
            raise ValueError(f"incompatible shard config: {report_path}")
        for result in episodes:
            if not isinstance(result, dict) or not result.get("episode"):
                raise ValueError(f"invalid episode result in {report_path}")
            episode = str(result["episode"])
            if episode in seen_episodes:
                raise ValueError(f"duplicate episode across shards: {episode}")
            seen_episodes.add(episode)
            copy_episode_artifacts(report_path.parent, output_root, result)
            merged_results.append(result)

    merged_results.sort(key=episode_key)
    merged_config = dict(base_config or {})
    merged_config.update(
        {
            "parallel_workers": len(report_paths),
            "gpus": shard_gpus,
            "shard_id": None,
            "num_shards": len(report_paths),
        }
    )
    if args.started_at_epoch is not None:
        elapsed_seconds = max(0.0, time.time() - args.started_at_epoch)
        merged_config["total_time_seconds"] = round(elapsed_seconds, 3)
        merged_config["total_time_hms"] = format_duration(elapsed_seconds)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_config.json").write_text(
        json.dumps(merged_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_reports(output_root, merged_results, merged_config)
    final_report = json.loads((output_root / "report.json").read_text(encoding="utf-8"))
    print(f"Merged {len(report_paths)} shards and {len(merged_results)} episodes into {output_root}")
    print(
        "Average score: "
        f"{final_report['average_score']} "
        f"({final_report['evaluated_episode_count']} complete episodes; "
        f"{final_report['excluded_partial_episode_count']} partial episodes excluded)"
    )
    if args.started_at_epoch is not None:
        print(
            "Total time: "
            f"{merged_config['total_time_hms']} "
            f"({merged_config['total_time_seconds']:.3f} seconds)"
        )


if __name__ == "__main__":
    main()
