"""Triworld single-dimension evaluation using metrics/_common compute modules."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml

from metrics._common.basic_metrics import compute_basic_metrics
from metrics._common.episode_sort import VIEW_ORDER, episode_sort_key, natural_key
from metrics._common.compute_registry import dimension_kwargs, get_compute_fn, get_module
from metrics._common.distributed import dist_init, get_rank, print0
from metrics._common.parallel_runner import is_episode_parallel, run_episode_parallel
from metrics._common.metric_scores import (
    cleanup_run_artifacts,
    extract_triworld_episode_scores,
    save_episode_scores,
    scores_path,
)
from metrics._common.utils import init_submodules, save_json


class NoCouplingDynamicDegree:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def infer(self, _video_path: str) -> float:
        return 1.0


def patch_raw_consistency(dimension: str) -> None:
    if dimension == "subject_consistency":
        get_module("subject_consistency").DynamicDegree = NoCouplingDynamicDegree
    elif dimension == "background_consistency":
        get_module("background_consistency").DynamicDegree = NoCouplingDynamicDegree


def patch_trajectory_accuracy(zero_distance_score: float) -> None:
    trajectory_accuracy = get_module("trajectory_accuracy")

    def safe_ndtw(traj_pred, traj_gt, invalid_pred_trajs, invalid_gt_trajs, max_distance_index):
        if invalid_pred_trajs[max_distance_index]:
            return 0.0
        distance, path = trajectory_accuracy.dtw_distance(
            traj_pred[:, max_distance_index], traj_gt[:, max_distance_index]
        )
        mean_distance = distance / len(path) if path else float("inf")
        if mean_distance <= 1e-12:
            return zero_distance_score
        return 1.0 / mean_distance

    trajectory_accuracy.NDTW = safe_ndtw

_EMPIRICAL_BOUNDS = {
    "photometric_smoothness": {"min": 0.1257, "max": 6.7899, "invert": False},
    "motion_smoothness": {"min": 0.0, "max": 2.6413, "invert": False},
    "trajectory_accuracy": {"min": 0.0, "max": 40.8540, "invert": False},
    "flow_score": {"min": 0.0531, "max": 8.9414, "invert": False},
    "depth_accuracy": {"min": 0.2228, "max": 4.3711, "invert": True},
}


def _normalize_value(metric: str, value: float) -> float:
    bounds = _EMPIRICAL_BOUNDS.get(metric)
    if bounds is None:
        return value
    vmin, vmax = bounds["min"], bounds["max"]
    if vmax <= vmin:
        return value
    norm = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    return 1.0 - norm if bounds.get("invert") else norm


def _add_normalized_scores(metric: str, results: Any) -> Any:
    if not (isinstance(results, (list, tuple)) and len(results) >= 2 and isinstance(results[1], list)):
        return results
    avg_val, video_list = results[0], results[1]
    norm_list = []
    for item in video_list:
        if not isinstance(item, dict) or "video_results" not in item:
            norm_list.append(item)
            continue
        norm_item = dict(item)
        norm_item["video_results_normalized"] = _normalize_value(metric, item["video_results"])
        norm_list.append(norm_item)
    return [avg_val, norm_list]


def _metric_video_path(data_base: str, view_id: str, episode_id: str, gid: str | None = None) -> str:
    """Build video_path for parse_video_path; 0710 flat head val_base already includes view."""
    base = Path(data_base)
    # val_base ends with head/left/right only in flat or per-episode shard layouts.
    if base.name in VIEW_ORDER:
        if gid is None:
            return os.path.join(data_base, episode_id)
        return os.path.join(data_base, episode_id, gid, "video")
    if gid is None:
        return os.path.join(data_base, view_id, episode_id)
    return os.path.join(data_base, view_id, episode_id, gid, "video")


def _to_standard_results(metric: str, results: Any, data_base: str) -> Any:
    if not isinstance(results, dict) or metric not in {"trajectory_accuracy", "semantic_alignment"}:
        return results
    video_entries = []
    for view_id, episodes in results.items():
        if not isinstance(episodes, dict):
            continue
        for episode_id, groups in episodes.items():
            if not isinstance(groups, dict):
                try:
                    val = float(groups)
                except (TypeError, ValueError):
                    val = 0.0
                video_entries.append({
                    "video_path": _metric_video_path(data_base, view_id, episode_id),
                    "video_results": val,
                })
                continue
            for gid, metrics in groups.items():
                if isinstance(metrics, dict):
                    raw_val = metrics.get("ndtw", 0.0) if metric == "trajectory_accuracy" else metrics.get("CLIPScore", 0.0)
                else:
                    raw_val = metrics
                try:
                    val = float(raw_val)
                except (TypeError, ValueError):
                    val = 0.0
                video_entries.append({
                    "video_path": _metric_video_path(data_base, view_id, episode_id, gid),
                    "video_results": val,
                })
    if not video_entries:
        return results
    avg_val = sum(item["video_results"] for item in video_entries) / len(video_entries)
    return [avg_val, video_entries]


def _build_full_info_json(output_path: Path, data_base: str, data_name: str, dimension_list: list[str]) -> str:
    cur_full_info_list = []
    for view_id in sorted(os.listdir(data_base)):
        view_path = os.path.join(data_base, view_id)
        for episode_id in sorted(os.listdir(view_path), key=episode_sort_key):
            if episode_id.endswith((".png", ".json")):
                continue
            episode_path = os.path.join(view_path, episode_id)
            for gid in sorted(os.listdir(episode_path), key=natural_key):
                video_path = os.path.join(episode_path, gid, "video")
                cur_full_info_list.append({"dimension": dimension_list, "video_list": [video_path]})
    cur_full_info_path = output_path / f"{data_name}_full_info.json"
    save_json(cur_full_info_list, str(cur_full_info_path))
    return str(cur_full_info_path)



def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _compute_dimension(
    dimension: str,
    data_base: str,
    data_name: str,
    gt_path: str,
    output_path: Path,
    kwargs: dict[str, Any],
) -> Any:
    submodules_dict = init_submodules([dimension], local=True, **kwargs)

    if dimension == "trajectory_accuracy":
        return get_compute_fn("trajectory_accuracy")(gt_path=gt_path, data_base=data_base)
    if dimension == "semantic_alignment":
        return get_compute_fn("semantic_alignment")(
            data_name=data_name,
            data_base=data_base,
            gt_path=gt_path,
            output_path=output_path,
            submodules_list=submodules_dict[dimension],
            **kwargs,
        )

    cur_full_info_path = _build_full_info_json(output_path, data_base, data_name, [dimension])
    if dimension == "psnr_ssim":
        return compute_basic_metrics(gt_path=gt_path, pd_path=data_base, metric_names=["psnr", "ssim"])
    if dimension == "psnr":
        return compute_basic_metrics(gt_path=gt_path, pd_path=data_base, metric_names=["psnr"])
    if dimension == "ssim":
        return compute_basic_metrics(gt_path=gt_path, pd_path=data_base, metric_names=["ssim"])

    submodules_list = submodules_dict[dimension]
    if dimension == "aesthetic_quality":
        results = get_compute_fn("aesthetic_quality")(cur_full_info_path, submodules_list, **kwargs)
    elif dimension == "background_consistency":
        results = get_compute_fn("background_consistency")(cur_full_info_path, submodules_list, **kwargs)
    elif dimension == "dynamic_degree":
        results = get_compute_fn("dynamic_degree")(cur_full_info_path, submodules_list, **kwargs)
    elif dimension in ("image_quality", "imaging_quality"):
        results = get_compute_fn("image_quality")(cur_full_info_path, submodules_list, **kwargs)
    elif dimension == "subject_consistency":
        results = get_compute_fn("subject_consistency")(cur_full_info_path, submodules_list, **kwargs)
    elif dimension == "flow_score":
        results = get_compute_fn("flow_score")(cur_full_info_path, submodules_list, **kwargs)
    elif dimension == "photometric_smoothness":
        results = get_compute_fn("photometric_smoothness")(cur_full_info_path, submodules_list, **kwargs)
    else:
        raise ValueError(f"unsupported dimension: {dimension}")

    results = _to_standard_results(dimension, results, data_base)
    return _add_normalized_scores(dimension, results)


def _resolve_config_path(ctx: Any, dimension: str, config_path: Path | None) -> Path:
    if config_path is not None:
        return config_path
    if dimension in ("psnr", "ssim"):
        return ctx.config_dir / "psnr_ssim.yaml"
    if dimension == "image_quality":
        return ctx.config_dir / "image_quality.yaml"
    return ctx.config_dir / f"{dimension}.yaml"


def _run_batch(
    ctx: Any,
    dimension: str,
    metric_index: int,
    metric_name: str,
    *,
    config_path: Path,
    apply_0710_patches: bool = False,
    trajectory_zero_score: float | None = None,
) -> None:
    """Full-dataset single-process evaluation for global metrics."""
    out_dir = ctx.work / "output_metrics" / dimension
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "generated_results.json"
    score_file = scores_path(ctx.temp_dir, metric_name)

    if score_file.is_file() and ctx.resume:
        print(f"[skip] {metric_name}: {score_file}")
        return

    if apply_0710_patches:
        patch_raw_consistency(dimension)
    if dimension == "trajectory_accuracy" and trajectory_zero_score is not None:
        patch_trajectory_accuracy(trajectory_zero_score)

    os.environ.setdefault("MATRIC_EVAL_ROOT", str(ctx.run_dir.parent.parent))
    dist_init()

    config = _load_yaml(config_path)
    data_base = config["data"]["val_base"]
    gt_path = config["data"]["gt_path"]
    data_name = os.path.basename(data_base).replace("_dataset", "")
    save_path = Path(config["save_path"])
    save_path.mkdir(parents=True, exist_ok=True)

    kwargs = dimension_kwargs(config, dimension)
    results = _compute_dimension(dimension, data_base, data_name, gt_path, save_path, kwargs)

    results_dict: dict[str, Any] = {dimension: results}
    if get_rank() == 0:
        marker.write_text(json.dumps(results_dict, ensure_ascii=False, indent=2), encoding="utf-8")
        scores = extract_triworld_episode_scores(dimension, results_dict)
        save_episode_scores(ctx.temp_dir, metric_name, metric_index, scores)
        cleanup_run_artifacts(ctx)
    print0(f"[done] {metric_name} -> {score_file}")


def run_dimension(
    ctx: Any,
    dimension: str,
    metric_index: int,
    metric_name: str,
    *,
    config_path: Path | None = None,
    apply_0710_patches: bool = False,
    trajectory_zero_score: float | None = None,
) -> None:
    """Run one metric dimension; episode-parallel metrics use fixed parallel_per_gpu queue."""
    resolved_config = _resolve_config_path(ctx, dimension, config_path)
    config_dir = resolved_config.parent

    if is_episode_parallel(dimension):
        run_episode_parallel(
            ctx,
            dimension,
            metric_index,
            metric_name,
            config_dir=config_dir,
            apply_0710_patches=apply_0710_patches,
            trajectory_zero_score=trajectory_zero_score,
        )
        return

    _run_batch(
        ctx,
        dimension,
        metric_index,
        metric_name,
        config_path=resolved_config,
        apply_0710_patches=apply_0710_patches,
        trajectory_zero_score=trajectory_zero_score,
    )
