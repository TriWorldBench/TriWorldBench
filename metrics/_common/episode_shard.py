"""Episode-level shard helpers for parallel metric evaluation."""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from metrics._common.episode_sort import EP_RE, VIEW_ORDER
from metrics._common.episode_sort import episode_sort_key, natural_key, sort_episode_items

# Metrics split into per-episode jobs with fixed parallel_per_gpu workers.
EPISODE_PARALLEL_METRICS: frozenset[str] = frozenset({
    "aesthetic_quality",
    "image_quality",
    "background_consistency",
    "dynamic_degree",
    "subject_consistency",
    "flow_score",
    "photometric_smoothness",
    "psnr",
    "ssim",
    "trajectory_accuracy",
})


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def replace_symlink(src: Path, dst: Path) -> None:
    src = src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    os.symlink(src, dst, target_is_directory=src.is_dir())


@dataclass(frozen=True)
class EpisodeRef:
    episode_id: str

    @property
    def key(self) -> str:
        return self.episode_id

    @property
    def shard_key(self) -> str:
        return safe_name(self.episode_id)


def discover_episodes(gen_root: Path) -> list[EpisodeRef]:
    episodes: list[EpisodeRef] = []
    if not gen_root.is_dir():
        return episodes

    seen: set[str] = set()
    for view_dir in sorted(gen_root.iterdir(), key=lambda p: natural_key(p.name)):
        if view_dir.name not in VIEW_ORDER or not view_dir.is_dir():
            continue
        for episode_dir in sorted(view_dir.iterdir(), key=lambda p: episode_sort_key(p.name)):
            if not episode_dir.is_dir() or not EP_RE.match(episode_dir.name):
                continue
            if episode_dir.name in seen:
                continue
            seen.add(episode_dir.name)
            episodes.append(EpisodeRef(episode_id=episode_dir.name))
    return sort_episode_items(episodes, key="episode_id")


def _is_flat_episode_root(gen_root: Path) -> bool:
    if not gen_root.is_dir():
        return False
    children = [p for p in gen_root.iterdir() if p.is_dir()]
    return bool(children) and all(EP_RE.match(p.name) for p in children)


def discover_episodes_from_val_base(gen_root: Path) -> list[EpisodeRef]:
    """Discover episodes from val_base (flat head layout or view/episode tree)."""
    if _is_flat_episode_root(gen_root):
        refs = [
            EpisodeRef(episode_id=p.name)
            for p in sorted(gen_root.iterdir(), key=lambda p: episode_sort_key(p.name))
            if p.is_dir() and EP_RE.match(p.name)
        ]
        return sort_episode_items(refs, key="episode_id")
    return discover_episodes(gen_root)


def episode_shard_root(work_root: Path, metric: str, ref: EpisodeRef) -> Path:
    return work_root / "episode_shards" / metric / ref.shard_key


def shard_output_dir(work_root: Path, metric: str, ref: EpisodeRef) -> Path:
    return work_root / "output_metrics" / metric / "shards" / ref.shard_key


def episode_shard_done(work_root: Path, metric: str, ref: EpisodeRef) -> bool:
    shard_dir = shard_output_dir(work_root, metric, ref)
    return (shard_dir / "generated_results.json").is_file()


def episode_job_name(metric: str, ref: EpisodeRef) -> str:
    return f"{metric}__{ref.shard_key}"


def prepare_episode_shard(
    work_root: Path,
    config_dir: Path,
    metric: str,
    ref: EpisodeRef,
) -> tuple[Path, Path]:
    """Symlink single-episode dataset + config; return (config_path, output_dir)."""
    gen_root = work_root / "inputs/data/generated_dataset"
    gt_root = work_root / "inputs/data/gt_dataset"
    shard_root = episode_shard_root(work_root, metric, ref)
    shard_gen = shard_root / "generated_dataset"
    shard_gt = shard_root / "gt_dataset"

    if shard_root.exists():
        shutil.rmtree(shard_root)
    shard_gen.mkdir(parents=True, exist_ok=True)
    shard_gt.mkdir(parents=True, exist_ok=True)

    for view in VIEW_ORDER:
        src_gen = gen_root / view / ref.episode_id
        src_gt = gt_root / view / ref.episode_id
        if src_gen.is_dir():
            replace_symlink(src_gen, shard_gen / view / ref.episode_id)
        if src_gt.is_dir():
            replace_symlink(src_gt, shard_gt / view / ref.episode_id)

    config_name = metric
    if metric in ("psnr", "ssim"):
        config_name = "psnr_ssim"
    elif metric == "image_quality":
        config_name = "image_quality"

    base_cfg = load_yaml(config_dir / f"{config_name}.yaml")
    cfg = dict(base_cfg)
    cfg["data"] = dict(base_cfg.get("data", {}))
    cfg["data"]["val_base"] = str(shard_gen)
    cfg["data"]["gt_path"] = str(shard_gt)
    output_dir = shard_output_dir(work_root, metric, ref)
    cfg["save_path"] = str(output_dir)
    cfg_path = shard_root / "config.yaml"
    save_yaml(cfg_path, cfg)
    return cfg_path, output_dir


def prepare_config_episode_shard(
    work_root: Path,
    config_dir: Path,
    metric: str,
    ref: EpisodeRef,
    *,
    config_name: str | None = None,
) -> tuple[Path, Path]:
    """Shard one episode using metric YAML val_base/gt_path (0710 head or trajectory layouts)."""
    name = config_name or metric
    if metric in ("psnr", "ssim"):
        name = "psnr_ssim"
    base_cfg = load_yaml(config_dir / f"{name}.yaml")
    gen_root = Path(base_cfg["data"]["val_base"])
    gt_root = Path(base_cfg["data"]["gt_path"])

    if not _is_flat_episode_root(gen_root):
        return prepare_episode_shard(work_root, config_dir, metric, ref)

    shard_root = episode_shard_root(work_root, metric, ref)
    if shard_root.exists():
        shutil.rmtree(shard_root)

    view_name = gen_root.name if gen_root.name in VIEW_ORDER else "head"
    shard_gen = shard_root / "generated_dataset" / view_name
    shard_gt = shard_root / "gt_dataset" / view_name
    shard_gen.mkdir(parents=True, exist_ok=True)
    shard_gt.mkdir(parents=True, exist_ok=True)

    src_gen_ep = gen_root / ref.episode_id
    src_gt_ep = gt_root / ref.episode_id
    if src_gen_ep.is_dir():
        for gid_dir in sorted(src_gen_ep.iterdir(), key=lambda p: natural_key(p.name)):
            if gid_dir.is_dir():
                replace_symlink(gid_dir, shard_gen / ref.episode_id / gid_dir.name)
    if src_gt_ep.is_dir():
        replace_symlink(src_gt_ep, shard_gt / ref.episode_id)

    cfg = dict(base_cfg)
    cfg["data"] = dict(base_cfg.get("data", {}))
    cfg["data"]["val_base"] = str(shard_gen)
    cfg["data"]["gt_path"] = str(shard_gt)
    output_dir = shard_output_dir(work_root, metric, ref)
    cfg["save_path"] = str(output_dir)
    cfg_path = shard_root / "config.yaml"
    save_yaml(cfg_path, cfg)
    return cfg_path, output_dir


def prepare_metric_episode_shard(
    work_root: Path,
    config_dir: Path,
    metric: str,
    ref: EpisodeRef,
) -> tuple[Path, Path]:
    metric_cfg = config_dir / f"{metric}.yaml"
    if metric_cfg.is_file():
        sample = load_yaml(metric_cfg)
        val_base = Path(sample.get("data", {}).get("val_base", ""))
        if val_base.is_dir() and _is_flat_episode_root(val_base):
            return prepare_config_episode_shard(work_root, config_dir, metric, ref)
    return prepare_episode_shard(work_root, config_dir, metric, ref)


def _finite_mean(values: list[float]) -> float | None:
    finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def _merge_avg_list(entries: list[Any]) -> list[Any]:
    video_lists: list[list[dict[str, Any]]] = []
    for entry in entries:
        if not (isinstance(entry, list) and len(entry) >= 2 and isinstance(entry[1], list)):
            continue
        video_lists.append(entry[1])
    merged_videos: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for video_list in video_lists:
        for item in video_list:
            if not isinstance(item, dict):
                continue
            key = str(item.get("video_path") or item.get("video") or "")
            if key and key in seen_paths:
                continue
            if key:
                seen_paths.add(key)
            merged_videos.append(item)
    merged_videos.sort(key=lambda item: str(item.get("video_path") or item.get("video") or ""))
    values = [
        float(item["video_results"])
        for item in merged_videos
        if isinstance(item.get("video_results"), (int, float)) and math.isfinite(float(item["video_results"]))
    ]
    avg = _finite_mean(values)
    if avg is None and merged_videos:
        avg = 0.0
    return [avg, merged_videos]


def merge_metric_shard_results(metric: str, shard_dir: Path, output_path: Path) -> dict[str, Any]:
    """Merge per-episode shard JSON files into one generated_results.json."""
    shard_files = sorted(shard_dir.glob("*/generated_results.json"))
    if not shard_files:
        raise RuntimeError(f"no episode shards for metric={metric} under {shard_dir}")

    if metric in ("psnr", "ssim"):
        parts: list[Any] = []
        for path in shard_files:
            data = json.loads(path.read_text(encoding="utf-8"))
            payload = data.get(metric, data)
            parts.append(payload)
        merged = {metric: _merge_avg_list(parts)}
    else:
        parts = []
        for path in shard_files:
            data = json.loads(path.read_text(encoding="utf-8"))
            parts.append(data.get(metric, data))
        merged = {metric: _merge_avg_list(parts)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged
