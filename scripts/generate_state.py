#!/usr/bin/env python3
"""Generate flat per-episode STATE JSON from HDF5 trajectories.

Backward-compatible flat input layout:

  input/<method>/
  └── episodeN/
        episode*.hdf5
        head.mp4, left.mp4, right.mp4   # mp4 ignored by this script

RoboTwin2.0 task-level input can be provided through --robotwin-root, or by
passing the mapping JSON written by prepare_gt_dataset_from_robotwin.py.

Output:

  test_data/STATE/episodeN.json

All logic lives inside Triworld (no sibling repo imports).
Requires: h5py, numpy (same as triworldbench env).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR / "lib"))

from segment_episode_phases import segment_hdf5  # noqa: E402


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

EP_RE = re.compile(r"^episode(\d+)$", re.IGNORECASE)
HDF5_RE = re.compile(r"^episode(\d+)\.hdf5$", re.IGNORECASE)
STEM_RE = re.compile(r"^(?P<task>.+)_(?P<mode>clean|randomized)_episode(?P<ep>\d+)$")
INSTRUCTION_KEYS = ("instruction", "prompt", "task_instruction", "text", "description")
INSTRUCTION_LIST_KEYS = ("seen", "unseen", "instructions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Flat episode root, e.g. input/test.",
    )
    parser.add_argument(
        "--robotwin-root",
        type=Path,
        default=None,
        help="RoboTwin2.0 root containing <task_name>/<config_name>/ directories.",
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help="Config/run directory below each task when using --robotwin-root.",
    )
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional task-name filter.")
    parser.add_argument(
        "--episodes",
        nargs="*",
        default=None,
        help="Optional source episode filter for --robotwin-root, e.g. episode18 episode19.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="First flat episode id when discovering --robotwin-root directly.",
    )
    parser.add_argument(
        "--mapping-json",
        type=Path,
        default=None,
        help="Optional shuffle_mapping.json / manifest.json for task names and source HDF5 paths.",
    )
    parser.add_argument(
        "--mapping-output",
        type=Path,
        default=None,
        help="Optional mapping JSON to write when using --robotwin-root without --mapping-json.",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=PROJECT_ROOT / "test_data" / "gt_dataset",
        help="Flat GT root; instruction read from episodeN/episodeN.json when present.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "test_data" / "STATE",
        help="Flat STATE output root (episodeN.json).",
    )
    parser.add_argument(
        "--scene-info-root",
        type=Path,
        default=None,
        help="Optional scene_info reference directory (<task>/episodeM.json).",
    )
    parser.add_argument("--limit", type=int, default=0)
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


def read_instruction_json(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return find_first_string(read_json(path))
    except (OSError, json.JSONDecodeError):
        return ""


def load_mapping(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    payload = read_json(path)
    items = payload.get("entries", payload.get("items", []))
    if not items:
        raise RuntimeError(f"No mapping entries in {path}")
    mapping: dict[str, dict[str, Any]] = {}
    for item in items:
        episode = str(item.get("new_episode") or "")
        if episode:
            mapping[episode] = item
    return mapping


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


def discover_robotwin_mapping(
    robotwin_root: Path,
    config_name: str | None,
    tasks: list[str] | None,
    episodes: list[str] | None,
    start_index: int,
    limit: int,
) -> dict[str, dict[str, Any]]:
    if start_index < 1:
        raise ValueError("--start-index must be >= 1")
    if not robotwin_root.is_dir():
        raise FileNotFoundError(f"robotwin root not found: {robotwin_root}")

    task_filter = set(tasks) if tasks else None
    episode_filter = set(episodes) if episodes else None
    entries: list[dict[str, Any]] = []

    for task_dir in sorted(robotwin_root.iterdir(), key=lambda p: p.name):
        if not task_dir.is_dir() or task_dir.name.startswith(".") or task_dir.name == "STATE":
            continue
        if task_filter is not None and task_dir.name not in task_filter:
            continue
        for config_dir in usable_configs(task_dir, config_name):
            hdf5_files = sorted((config_dir / "data").glob("episode*.hdf5"), key=hdf5_sort_key)
            for hdf5_path in hdf5_files:
                source_episode = hdf5_path.stem
                if episode_filter is not None and source_episode not in episode_filter:
                    continue
                instruction_path = config_dir / "instructions" / f"{source_episode}.json"
                entries.append(
                    {
                        "task": task_dir.name,
                        "config_name": config_dir.name,
                        "source_episode": source_episode,
                        "source_stem": f"{task_dir.name}_randomized_{source_episode}",
                        "source_hdf5": str(hdf5_path.resolve()),
                        "source_instruction_json": str(instruction_path.resolve()),
                        "instruction": read_instruction_json(instruction_path),
                    }
                )

    if limit > 0:
        entries = entries[:limit]

    mapping: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(entries):
        episode = f"episode{start_index + index}"
        item["new_episode"] = episode
        mapping[episode] = item
    return mapping


def write_mapping(path: Path, args: argparse.Namespace, mapping: dict[str, dict[str, Any]]) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": str(args.robotwin_root.resolve()) if args.robotwin_root else None,
        "config_name": args.config_name,
        "start_index": args.start_index,
        "entries": [mapping[ep] for ep in sorted(mapping, key=episode_sort_key)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_source_stem(stem: str) -> tuple[str, int] | None:
    match = STEM_RE.match(stem)
    if not match:
        return None
    return match.group("task"), int(match.group("ep"))


def discover_episodes(input_root: Path | None, mapping: dict[str, dict[str, Any]]) -> list[str]:
    if mapping:
        episodes = [ep for ep in mapping if EP_RE.match(ep)]
        return sorted(episodes, key=episode_sort_key)
    if input_root is None:
        return []
    episodes = [
        p.name
        for p in input_root.iterdir()
        if p.is_dir() and EP_RE.match(p.name)
    ]
    return sorted(episodes, key=episode_sort_key)


def list_hdf5(episode_dir: Path) -> list[Path]:
    return sorted(episode_dir.glob("*.hdf5"), key=lambda p: p.name)


def resolve_hdf5(
    episode_dir: Path | None,
    preferred_local_ep: int | None,
    mapping_item: dict[str, Any],
) -> tuple[Path, str]:
    source_hdf5 = mapping_item.get("source_hdf5") or mapping_item.get("hdf5_path")
    if isinstance(source_hdf5, str) and source_hdf5.strip():
        path = Path(source_hdf5).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"mapped HDF5 not found: {path}")
        local = str(mapping_item.get("source_episode") or "")
        if not local:
            match = HDF5_RE.match(path.name)
            local = f"episode{match.group(1)}" if match else path.stem
        return path, local

    if episode_dir is None:
        raise FileNotFoundError("no episode directory or mapped source_hdf5 available")

    candidates = list_hdf5(episode_dir)
    if not candidates:
        raise FileNotFoundError(f"No HDF5 under {episode_dir}")

    if preferred_local_ep is not None:
        preferred = episode_dir / f"episode{preferred_local_ep}.hdf5"
        if preferred.is_file():
            return preferred, f"episode{preferred_local_ep}"

    if len(candidates) == 1:
        path = candidates[0]
        match = HDF5_RE.match(path.name)
        local = f"episode{match.group(1)}" if match else path.stem
        return path, local

    raise FileNotFoundError(
        f"Multiple HDF5 files under {episode_dir}; pass --mapping-json or keep one file per episode"
    )


def read_episode_metadata(episode_dir: Path | None, episode: str) -> dict[str, Any]:
    if episode_dir is None:
        return {}
    for name in (f"{episode}.json", "metadata.json"):
        path = episode_dir / name
        if path.is_file():
            payload = read_json(path)
            if isinstance(payload, dict):
                return payload
    return {}


def resolve_task(episode: str, mapping: dict[str, dict[str, Any]], metadata: dict[str, Any]) -> tuple[str, int | None]:
    item = mapping.get(episode, {})
    task = item.get("task") or metadata.get("task")
    if isinstance(task, str) and task.strip():
        source_episode = str(item.get("source_episode") or "")
        match = EP_RE.match(source_episode)
        return task.strip(), int(match.group(1)) if match else None

    stem = str(item.get("source_stem") or "")
    parsed = parse_source_stem(stem)
    if parsed:
        return parsed[0], parsed[1]

    return "benchmark", None


def resolve_instruction(
    episode: str,
    metadata: dict[str, Any],
    mapping_item: dict[str, Any],
    gt_root: Path | None,
) -> str:
    instruction = mapping_item.get("instruction")
    if isinstance(instruction, str) and instruction.strip():
        return instruction.strip()

    source_instruction_json = mapping_item.get("source_instruction_json")
    if isinstance(source_instruction_json, str) and source_instruction_json.strip():
        instruction = read_instruction_json(Path(source_instruction_json).expanduser())
        if instruction:
            return instruction

    instruction = metadata.get("instruction")
    if isinstance(instruction, str) and instruction.strip():
        return instruction.strip()

    if gt_root is not None:
        gt_json = gt_root / episode / f"{episode}.json"
        if gt_json.is_file():
            payload = read_json(gt_json)
            instruction = payload.get("instruction")
            if isinstance(instruction, str) and instruction.strip():
                return instruction.strip()
    return ""


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve() if args.input_root else None
    gt_root = args.gt_root.resolve() if args.gt_root else None
    output_root = args.output_root.resolve()
    scene_info_root = args.scene_info_root.resolve() if args.scene_info_root else None

    if input_root is not None and not input_root.is_dir():
        print(f"[error] input root not found: {input_root}", file=sys.stderr)
        return 1

    try:
        mapping = load_mapping(args.mapping_json.resolve() if args.mapping_json else None)
        if not mapping and args.robotwin_root:
            mapping = discover_robotwin_mapping(
                args.robotwin_root.resolve(),
                args.config_name,
                args.tasks,
                args.episodes,
                args.start_index,
                args.limit,
            )
            if args.mapping_output and not args.dry_run:
                write_mapping(args.mapping_output.resolve(), args, mapping)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    if input_root is None and not mapping:
        print("[error] pass --input-root, --mapping-json, or --robotwin-root", file=sys.stderr)
        return 1

    episodes = discover_episodes(input_root, mapping)
    if args.limit > 0 and not args.robotwin_root:
        episodes = episodes[: args.limit]
    if not episodes:
        print("[error] no episodes found", file=sys.stderr)
        return 1

    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for episode in episodes:
        item = mapping.get(episode, {})
        episode_dir = input_root / episode if input_root is not None else None
        if episode_dir is not None and not episode_dir.is_dir() and not item.get("source_hdf5"):
            failed.append({"episode": episode, "error": f"missing directory: {episode_dir}"})
            print(f"[fail] {episode}: missing directory")
            continue

        metadata = read_episode_metadata(episode_dir, episode)
        task, local_ep_num = resolve_task(episode, mapping, metadata)
        output_json = output_root / f"{episode}.json"

        if output_json.is_file() and not args.overwrite:
            records.append({"episode": episode, "status": "exists"})
            continue

        try:
            hdf5_path, local_episode = resolve_hdf5(episode_dir, local_ep_num, item)
        except Exception as exc:
            failed.append({"episode": episode, "error": str(exc)})
            print(f"[fail] {episode}: {exc}")
            continue

        if args.dry_run:
            print(f"[dry-run] {episode} task={task} <- {hdf5_path}")
            continue

        instruction = resolve_instruction(episode, metadata, item, gt_root)
        try:
            record = segment_hdf5(
                hdf5_path,
                task,
                episode,
                local_episode,
                instruction,
                scene_info_root,
            )
            record["output_json"] = str(output_json.resolve())
            output_json.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            records.append(record)
            print(f"[ok] {episode} task={task} frames={record['num_frames']}")
        except Exception as exc:
            failed.append({"episode": episode, "error": str(exc)})
            print(f"[fail] {episode}: {exc}")

    if args.dry_run:
        print(f"total: {len(episodes)}")
        return 0

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_root": str(input_root) if input_root else None,
        "robotwin_root": str(args.robotwin_root.resolve()) if args.robotwin_root else None,
        "mapping_json": str(args.mapping_json) if args.mapping_json else None,
        "gt_root": str(gt_root) if gt_root else None,
        "output_root": str(output_root),
        "written": len(records),
        "failed": len(failed),
        "records": records,
        "failures": failed,
    }
    (output_root / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"STATE manifest: {output_root / 'manifest.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
