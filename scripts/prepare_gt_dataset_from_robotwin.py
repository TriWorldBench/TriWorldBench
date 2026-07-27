#!/usr/bin/env python3
"""Prepare flat TriWorldBench gt_dataset from RoboTwin2.0 task-level outputs.

Expected RoboTwin2.0 source layout:

  <source-root>/
  └── <task_name>/
      └── <config_name>/
          ├── data/episode*.hdf5
          ├── _traj_data/
          ├── instructions/episode*.json
          └── video/

Output:

  <target-root>/
  └── episodeN/
      ├── episodeN.json
      ├── head/frames/frame_*.jpg
      ├── left/frames/frame_*.jpg
      └── right/frames/frame_*.jpg

The script also writes a mapping JSON that can be passed to generate_state.py so
STATE and gt_dataset use identical flat episode ids.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

VIEW_DATASETS = (
    ("head", "observation/head_camera/rgb"),
    ("left", "observation/left_camera/rgb"),
    ("right", "observation/right_camera/rgb"),
)
EP_RE = re.compile(r"^episode(\d+)$", re.IGNORECASE)
HDF5_RE = re.compile(r"^episode(\d+)\.hdf5$", re.IGNORECASE)
INSTRUCTION_KEYS = ("instruction", "prompt", "task_instruction", "text", "description")
INSTRUCTION_LIST_KEYS = ("seen", "unseen", "instructions")


@dataclass(frozen=True)
class SourceEpisode:
    task: str
    config_name: str
    source_episode: str
    hdf5_path: Path
    instruction_path: Path
    instruction: str


@dataclass(frozen=True)
class FlatEpisode:
    new_episode: str
    source: SourceEpisode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="RoboTwin2.0 root containing <task_name>/<config_name>/ directories.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=PROJECT_ROOT / "test_data" / "gt_dataset",
        help="Output flat gt_dataset root.",
    )
    parser.add_argument(
        "--mapping-output",
        type=Path,
        default=PROJECT_ROOT / "test_data" / "robotwin_episode_mapping.json",
        help="Output mapping JSON shared with generate_state.py.",
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help="Config/run directory below each task. If omitted, each task must have one usable config.",
    )
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional task-name filter.")
    parser.add_argument(
        "--episodes",
        nargs="*",
        default=None,
        help="Optional source episode filter, e.g. episode18 episode19. Default: all episode*.hdf5.",
    )
    parser.add_argument("--start-index", type=int, default=1, help="First flat episode id.")
    parser.add_argument("--limit", type=int, default=0, help="Prepare first N flattened episodes.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--no-bgr-swap",
        action="store_true",
        help="Do not swap B/R channels after decoding HDF5 images.",
    )
    parser.add_argument(
        "--allow-missing-instruction",
        action="store_true",
        help="Write an empty instruction if instructions/episodeN.json is missing or unreadable.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def episode_sort_key(name: str) -> tuple[int, str]:
    match = EP_RE.match(name)
    if match:
        return int(match.group(1)), name
    return 10**9, name


def hdf5_sort_key(path: Path) -> tuple[int, str]:
    match = HDF5_RE.match(path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_first_string(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list):
        for item in value:
            found = find_first_string(item)
            if found:
                return found
    if isinstance(value, dict):
        for key in INSTRUCTION_KEYS:
            found = find_first_string(value.get(key))
            if found:
                return found
        for key in INSTRUCTION_LIST_KEYS:
            found = find_first_string(value.get(key))
            if found:
                return found
        for item in value.values():
            found = find_first_string(item)
            if found:
                return found
    return ""


def load_instruction(path: Path, allow_missing: bool) -> str:
    if not path.is_file():
        if allow_missing:
            return ""
        raise FileNotFoundError(f"missing instruction JSON: {path}")
    instruction = find_first_string(read_json(path))
    if not instruction and not allow_missing:
        raise ValueError(f"no instruction string found in {path}")
    return instruction


def usable_configs(task_dir: Path, config_name: str | None) -> list[Path]:
    if config_name:
        candidate = task_dir / config_name
        if list((candidate / "data").glob("episode*.hdf5")):
            return [candidate]
        return []
    configs = []
    for child in sorted(task_dir.iterdir(), key=lambda p: p.name):
        if child.is_dir() and list((child / "data").glob("episode*.hdf5")):
            configs.append(child)
    if len(configs) > 1:
        raise RuntimeError(
            f"multiple usable configs under {task_dir}; pass --config-name "
            f"({', '.join(p.name for p in configs[:5])})"
        )
    return configs


def discover_source_episodes(
    source_root: Path,
    config_name: str | None,
    tasks: list[str] | None,
    episodes: list[str] | None,
    allow_missing_instruction: bool,
) -> list[SourceEpisode]:
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root not found: {source_root}")

    task_filter = set(tasks) if tasks else None
    episode_filter = set(episodes) if episodes else None
    records: list[SourceEpisode] = []

    for task_dir in sorted(source_root.iterdir(), key=lambda p: p.name):
        if not task_dir.is_dir() or task_dir.name.startswith(".") or task_dir.name == "STATE":
            continue
        if task_filter is not None and task_dir.name not in task_filter:
            continue
        configs = usable_configs(task_dir, config_name)
        for config_dir in configs:
            hdf5_files = sorted((config_dir / "data").glob("episode*.hdf5"), key=hdf5_sort_key)
            for hdf5_path in hdf5_files:
                source_episode = hdf5_path.stem
                if episode_filter is not None and source_episode not in episode_filter:
                    continue
                instruction_path = config_dir / "instructions" / f"{source_episode}.json"
                instruction = load_instruction(instruction_path, allow_missing_instruction)
                records.append(
                    SourceEpisode(
                        task=task_dir.name,
                        config_name=config_dir.name,
                        source_episode=source_episode,
                        hdf5_path=hdf5_path,
                        instruction_path=instruction_path,
                        instruction=instruction,
                    )
                )
    return records


def assign_flat_episodes(
    records: list[SourceEpisode],
    start_index: int,
    limit: int,
) -> list[FlatEpisode]:
    if start_index < 1:
        raise ValueError("--start-index must be >= 1")
    if limit > 0:
        records = records[:limit]
    return [
        FlatEpisode(new_episode=f"episode{start_index + index}", source=record)
        for index, record in enumerate(records)
    ]


def decode_hdf5_image(value: Any, swap_bgr: bool) -> Image.Image:
    if isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.ndim in (2, 3):
        image = Image.fromarray(value).convert("RGB")
    else:
        image = Image.open(io.BytesIO(bytes(value))).convert("RGB")
    if swap_bgr:
        r, g, b = image.split()
        image = Image.merge("RGB", (b, g, r))
    return image


def write_episode_json(path: Path, flat: FlatEpisode) -> None:
    payload = {
        "instruction": flat.source.instruction,
        "task": flat.source.task,
        "episode": flat.new_episode,
        "source_episode": flat.source.source_episode,
        "source_hdf5": str(flat.source.hdf5_path.resolve()),
        "source_instruction_json": str(flat.source.instruction_path.resolve()),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def extract_gt_episode(
    flat: FlatEpisode,
    target_root: Path,
    overwrite: bool,
    jpeg_quality: int,
    swap_bgr: bool,
) -> dict[str, Any]:
    out_episode = target_root / flat.new_episode
    if out_episode.exists():
        if not overwrite:
            raise FileExistsError(f"target episode exists, use --overwrite: {out_episode}")
        shutil.rmtree(out_episode)
    out_episode.mkdir(parents=True, exist_ok=True)

    view_counts: dict[str, int] = {}
    with h5py.File(flat.source.hdf5_path, "r") as handle:
        for view, dataset_key in VIEW_DATASETS:
            if dataset_key not in handle:
                raise KeyError(f"missing HDF5 dataset {dataset_key}: {flat.source.hdf5_path}")
            dataset = handle[dataset_key]
            frame_dir = out_episode / view / "frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            for frame_idx in range(dataset.shape[0]):
                image = decode_hdf5_image(dataset[frame_idx], swap_bgr=swap_bgr)
                image.save(
                    frame_dir / f"frame_{frame_idx:05d}.jpg",
                    "JPEG",
                    quality=jpeg_quality,
                )
            view_counts[view] = int(dataset.shape[0])

    if len(set(view_counts.values())) != 1:
        raise RuntimeError(f"frame count mismatch for {flat.new_episode}: {view_counts}")

    write_episode_json(out_episode / f"{flat.new_episode}.json", flat)
    return {
        "episode": flat.new_episode,
        "task": flat.source.task,
        "source_episode": flat.source.source_episode,
        "frame_count": min(view_counts.values()),
        "view_counts": view_counts,
    }


def mapping_entry(flat: FlatEpisode, result: dict[str, Any] | None = None) -> dict[str, Any]:
    entry = {
        "new_episode": flat.new_episode,
        "task": flat.source.task,
        "config_name": flat.source.config_name,
        "source_episode": flat.source.source_episode,
        "source_stem": f"{flat.source.task}_randomized_{flat.source.source_episode}",
        "source_hdf5": str(flat.source.hdf5_path.resolve()),
        "source_instruction_json": str(flat.source.instruction_path.resolve()),
        "instruction": flat.source.instruction,
    }
    if result:
        entry["frame_count"] = result.get("frame_count")
        entry["view_counts"] = result.get("view_counts")
    return entry


def write_mapping(
    path: Path,
    args: argparse.Namespace,
    flats: list[FlatEpisode],
    results: list[dict[str, Any]],
) -> None:
    result_by_episode = {item["episode"]: item for item in results}
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": str(args.source_root.resolve()),
        "target_root": str(args.target_root.resolve()),
        "config_name": args.config_name,
        "start_index": args.start_index,
        "entries": [
            mapping_entry(flat, result_by_episode.get(flat.new_episode))
            for flat in flats
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not args.overwrite:
        raise FileExistsError(f"mapping exists, use --overwrite: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    target_root = args.target_root.resolve()
    mapping_output = args.mapping_output.resolve()

    try:
        source_records = discover_source_episodes(
            source_root,
            args.config_name,
            args.tasks,
            args.episodes,
            args.allow_missing_instruction,
        )
        flats = assign_flat_episodes(source_records, args.start_index, args.limit)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    if not flats:
        print("[error] no source episodes found", file=sys.stderr)
        return 1

    print(f"planned_episodes={len(flats)}")
    for flat in flats[:10]:
        print(
            f"[plan] {flat.new_episode} <- "
            f"{flat.source.task}/{flat.source.config_name}/{flat.source.source_episode}"
        )
    if len(flats) > 10:
        print(f"[plan] ... {len(flats) - 10} more")

    if args.dry_run:
        print("dry_run=true")
        return 0

    target_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for flat in flats:
        try:
            result = extract_gt_episode(
                flat,
                target_root,
                args.overwrite,
                args.jpeg_quality,
                swap_bgr=not args.no_bgr_swap,
            )
            results.append(result)
            print(
                f"[ok] {flat.new_episode} <- "
                f"{flat.source.task}/{flat.source.source_episode} frames={result['frame_count']}"
            )
        except Exception as exc:
            failed.append({"episode": flat.new_episode, "error": str(exc)})
            print(f"[fail] {flat.new_episode}: {exc}", file=sys.stderr)

    if failed:
        print(f"[error] failed={len(failed)}", file=sys.stderr)
        return 1

    try:
        write_mapping(mapping_output, args, flats, results)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"gt_dataset: {target_root}")
    print(f"mapping: {mapping_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
