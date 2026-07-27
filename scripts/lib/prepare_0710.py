#!/usr/bin/env python3
"""Prepare 0710 metric inputs and run-local YAML configs from split_views."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metrics._common.episode_sort import EP_RE, VIEW_ORDER, episode_sort_key
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def relative_symlink(src: Path, dst: Path, directory: bool | None = None) -> None:
    src = src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    target = os.path.relpath(src, start=dst.parent.resolve())
    os.symlink(target, dst, target_is_directory=src.is_dir() if directory is None else directory)


def frames(path: Path) -> list[Path]:
    return sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_EXTS) if path.is_dir() else []


def uniform_indices(count: int, target: int) -> list[int]:
    if count <= 0 or target <= 0:
        return []
    if target == 1:
        return [0]
    if count == 1:
        return [0] * target
    return [round(index * (count - 1) / (target - 1)) for index in range(target)]


def link_selected(source_frames: list[Path], indices: list[int], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for output_index, source_index in enumerate(indices):
        src = source_frames[source_index]
        relative_symlink(src, destination / f"frame_{output_index:05d}{src.suffix.lower()}", directory=False)


def iter_episodes(split_root: Path, dataset: str) -> list[Path]:
    root = split_root / dataset
    if not root.is_dir():
        raise FileNotFoundError(f"{dataset} not found under split root: {root}")
    return sorted(
        (p for p in root.iterdir() if p.is_dir() and EP_RE.match(p.name)),
        key=lambda p: episode_sort_key(p.name),
    )


def build_head_data(split_root: Path, output_root: Path) -> int:
    source_gen = split_root / "generated_dataset"
    source_gt = split_root / "gt_dataset"
    target_gen = output_root / "head_data/generated_dataset/head"
    target_gt = output_root / "head_data/gt_dataset/head"
    count = 0
    for episode_dir in iter_episodes(split_root, "generated_dataset"):
        episode = episode_dir.name
        gen_video = episode_dir / "head" / "video"
        gt_video = source_gt / episode / "head" / "video"
        if not gen_video.is_dir() or not gt_video.is_dir():
            continue
        relative_symlink(gen_video, target_gen / episode / "1" / "video")
        relative_symlink(gt_video, target_gt / episode / "video")
        count += 1
    if count <= 0:
        raise RuntimeError(f"no head semantic inputs found under {source_gen}")
    return count


def build_full_data(split_root: Path, output_root: Path) -> int:
    source_gen = split_root / "generated_dataset"
    source_gt = split_root / "gt_dataset"
    target_gen = output_root / "full_data/generated_dataset"
    target_gt = output_root / "full_data/gt_dataset"
    count = 0
    for episode_dir in iter_episodes(split_root, "generated_dataset"):
        episode = episode_dir.name
        for view in VIEW_ORDER:
            gen_video = episode_dir / view / "video"
            gt_video = source_gt / episode / view / "video"
            if not gen_video.is_dir() or not gt_video.is_dir():
                continue
            relative_symlink(gen_video, target_gen / view / episode / "1" / "video")
            relative_symlink(gt_video, target_gt / view / episode / "video")
            count += 1
    if count <= 0:
        raise RuntimeError(f"no full-view inputs found under {source_gen}")
    return count


def build_trajectory_data(split_root: Path, output_root: Path, num_frames: int) -> list[dict[str, Any]]:
    source_gen = split_root / "generated_dataset"
    source_gt = split_root / "gt_dataset"
    target_gen = output_root / "trajectory_head16/generated_dataset/head"
    target_gt = output_root / "trajectory_head16/gt_dataset/head"
    records: list[dict[str, Any]] = []

    for episode_dir in iter_episodes(split_root, "generated_dataset"):
        episode = episode_dir.name
        gen_video = episode_dir / "head" / "video"
        gt_video = source_gt / episode / "head" / "video"
        if not gen_video.is_dir() or not gt_video.is_dir():
            continue
        gen_frames = frames(gen_video)
        gt_frames = frames(gt_video)
        paired_count = min(len(gen_frames), len(gt_frames))
        if paired_count <= 0:
            raise RuntimeError(f"empty trajectory pair: {episode}")
        indices = uniform_indices(paired_count, num_frames)
        link_selected(gen_frames, indices, target_gen / episode / "1" / "video")
        link_selected(gt_frames, indices, target_gt / episode / "video")
        records.append(
            {
                "episode": episode,
                "gid": "1",
                "source_gen_frames": len(gen_frames),
                "source_gt_frames": len(gt_frames),
                "paired_frames": paired_count,
                "sampled_indices": indices,
                "output_frames": len(indices),
            }
        )
    if not records:
        raise RuntimeError(f"no head trajectory inputs found under {source_gen}")
    return records


def run_inputs(args: argparse.Namespace) -> None:
    if args.clean and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    with args.hyperparameters.open("r", encoding="utf-8") as f:
        params = yaml.safe_load(f) or {}
    num_frames = int(params.get("trajectory", {}).get("num_frames", 16))

    head_episodes = build_head_data(args.split_root, args.output_root)
    full_view_records = build_full_data(args.split_root, args.output_root)
    trajectory_records = build_trajectory_data(args.split_root, args.output_root, num_frames)
    manifest = {
        "protocol_version": "0710",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "split_root": str(args.split_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "head_episodes": head_episodes,
        "full_view_records": full_view_records,
        "trajectory_num_frames": num_frames,
        "trajectory_records": trajectory_records,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"head episodes: {head_episodes}")
    print(f"full view records: {full_view_records}")
    print(f"trajectory episodes: {len(trajectory_records)} x {num_frames} frames")


def write_config(base: dict[str, Any], path: Path, generated: Path, gt: Path, save_path: Path) -> None:
    config = dict(base)
    config["data"] = dict(base.get("data", {}))
    config["data"]["val_base"] = str(generated)
    config["data"]["gt_path"] = str(gt)
    config["save_path"] = str(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def run_configs(args: argparse.Namespace) -> None:
    with args.base_config.open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}
    full_gen = args.prepared_root / "full_data/generated_dataset"
    full_gt = args.prepared_root / "full_data/gt_dataset"
    head_gen = args.prepared_root / "head_data/generated_dataset/head"
    head_gt = args.prepared_root / "head_data/gt_dataset/head"
    trajectory_gen = args.prepared_root / "trajectory_head16/generated_dataset/head"
    trajectory_gt = args.prepared_root / "trajectory_head16/gt_dataset/head"

    write_config(base, args.output_dir / "subject_consistency.yaml", full_gen, full_gt, args.work_root / "output_metrics/subject_consistency")
    write_config(base, args.output_dir / "background_consistency.yaml", full_gen, full_gt, args.work_root / "output_metrics/background_consistency")
    write_config(base, args.output_dir / "semantic_alignment.yaml", head_gen, head_gt, args.work_root / "output_metrics/semantic_alignment")
    write_config(base, args.output_dir / "trajectory_accuracy.yaml", trajectory_gen, trajectory_gt, args.work_root / "output_metrics/trajectory_accuracy")
    write_config(base, args.output_dir / "trajectory_detection.yaml", trajectory_gen, trajectory_gt, args.work_root / "output_metrics/trajectory_accuracy")
    print(f"configs: {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inputs_parser = subparsers.add_parser("inputs", help="prepare head/full/trajectory inputs")
    inputs_parser.add_argument("--split-root", type=Path, required=True)
    inputs_parser.add_argument("--output-root", type=Path, required=True)
    inputs_parser.add_argument("--hyperparameters", type=Path, required=True)
    inputs_parser.add_argument("--clean", action="store_true")

    configs_parser = subparsers.add_parser("configs", help="write run-local metric YAML configs")
    configs_parser.add_argument("--base-config", type=Path, required=True)
    configs_parser.add_argument("--prepared-root", type=Path, required=True)
    configs_parser.add_argument("--work-root", type=Path, required=True)
    configs_parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "inputs":
        run_inputs(args)
    elif args.command == "configs":
        run_configs(args)
    else:
        raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
