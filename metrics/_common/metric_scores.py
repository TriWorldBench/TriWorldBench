"""Per-metric episode score files under temp/Metric_<name>/."""

from __future__ import annotations

import json
import math
import shutil
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from metrics._common.episode_sort import EP_RE, VIEW_ORDER, sort_episode_names


def metric_folder(temp_dir: Path, metric_name: str) -> Path:
    return temp_dir / f"Metric_{metric_name}"


def scores_path(temp_dir: Path, metric_name: str) -> Path:
    return metric_folder(temp_dir, metric_name) / "episode_scores.json"


def load_episode_scores(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def completed_episodes(temp_dir: Path, metric_name: str) -> set[str]:
    data = load_episode_scores(scores_path(temp_dir, metric_name))
    episodes = data.get("episodes")
    if not isinstance(episodes, dict):
        return set()
    return set(episodes.keys())


def save_episode_scores(
    temp_dir: Path,
    metric_name: str,
    metric_index: int,
    episodes: dict[str, float | None],
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = scores_path(temp_dir, metric_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {
        ep: round(float(value), 6)
        for ep, value in sorted(episodes.items(), key=lambda item: item[0])
        if value is not None and math.isfinite(float(value))
    }
    payload: dict[str, Any] = {
        "metric": metric_name,
        "index": metric_index,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "episodes": clean,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


NPSNR_SCALE_C = 20.0


def normalize_npsnr(value: Any, scale_c: float = NPSNR_SCALE_C) -> float | None:
    """Map PSNR (dB) to [0, 1): q = 1 - exp(-PSNR / c)."""
    try:
        psnr = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(psnr):
        return None
    if psnr == math.inf:
        return 1.0
    if psnr == -math.inf or psnr <= 0:
        return 0.0
    c = finite_float(scale_c)
    if c is None or c <= 0:
        raise ValueError("psnr scale_c must be a positive finite number")
    return clamp01(1.0 - math.exp(-psnr / c))


def parse_video_path(path_value: Any) -> tuple[str, str] | None:
    parts = Path(str(path_value or "")).parts
    marker_index = None
    for marker in ("generated_dataset", "gt_dataset"):
        if marker in parts:
            marker_index = parts.index(marker)
            break
    if marker_index is None or len(parts) <= marker_index + 2:
        return None

    token = parts[marker_index + 1]
    second = parts[marker_index + 2]
    if token in VIEW_ORDER and EP_RE.match(second):
        return second, token
    for view in VIEW_ORDER:
        suffix = f"__{view}"
        if token.endswith(suffix) and EP_RE.match(second):
            return second, view
    if len(parts) > marker_index + 3 and parts[marker_index + 3] in VIEW_ORDER and EP_RE.match(second):
        return second, parts[marker_index + 3]
    return None


def preferred_score(row: dict[str, Any], metric: str | None = None) -> float | None:
    if metric == "psnr":
        normalized = finite_float(row.get("video_results_normalized"))
        if normalized is not None:
            return clamp01(normalized)
        return normalize_npsnr(row.get("video_results"))
    normalized = finite_float(row.get("video_results_normalized"))
    if normalized is not None:
        return normalized
    try:
        out = float(row.get("video_results"))
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def extract_triworld_episode_scores(metric_name: str, payload: dict[str, Any]) -> dict[str, float]:
    entry = payload.get(metric_name)
    if not (isinstance(entry, list) and len(entry) >= 2 and isinstance(entry[1], list)):
        return {}

    per_episode: dict[str, list[float]] = {}
    for row in entry[1]:
        if not isinstance(row, dict):
            continue
        parsed = parse_video_path(row.get("video_path"))
        score = preferred_score(row, metric_name)
        if parsed is None or score is None:
            continue
        episode, _view = parsed
        per_episode.setdefault(episode, []).append(score)

    return {
        episode: sum(values) / len(values)
        for episode, values in per_episode.items()
        if values
    }


def _robust_view_aggregate(
    per_view: dict[str, float],
    config: dict[str, Any],
) -> float | None:
    ordered = [(view, per_view[view]) for view in VIEW_ORDER if view in per_view]
    if not ordered:
        return None
    if not config.get("enabled", True) or len(ordered) < 3:
        values = [score for _, score in ordered]
        return sum(values) / len(values)

    values = [score for _, score in ordered]
    median = statistics.median(values)
    deviations = [abs(score - median) for score in values]
    mad = statistics.median(deviations)
    absolute_limit = float(config.get("max_deviation_from_median", 0.25))
    mad_limit = float(config.get("mad_scale", 2.5)) * mad
    limit = max(absolute_limit, mad_limit)
    kept = [(view, score, dev) for (view, score), dev in zip(ordered, deviations) if dev <= limit]
    min_keep = max(1, int(config.get("min_views_after_filter", 2)))
    if len(kept) < min_keep:
        kept = sorted(
            [(view, score, dev) for (view, score), dev in zip(ordered, deviations)],
            key=lambda item: (item[2], VIEW_ORDER.index(item[0])),
        )[:min_keep]
    kept_scores = [score for _, score, _ in kept]
    return sum(kept_scores) / len(kept_scores) if kept_scores else None


def extract_vlm_episode_scores(
    vlm_data: list[dict[str, Any]],
    vlm_config: dict[str, Any],
) -> dict[str, dict[str, float]]:
    outlier_config = vlm_config.get("outlier_filter", {})
    multi_metrics = vlm_config.get("aggregate_multiview_metrics", ["Interaction_Quality", "Perspectivity"])
    result: dict[str, dict[str, float]] = {}

    for item in vlm_data:
        if not isinstance(item, dict):
            continue
        episode = str(item.get("episode") or "")
        if not episode:
            continue
        views = item.get("views", {}) if isinstance(item.get("views"), dict) else {}
        episode_scores: dict[str, float] = {}

        for metric in multi_metrics:
            per_view: dict[str, float] = {}
            for view in VIEW_ORDER:
                view_item = views.get(view, {}) if isinstance(views.get(view), dict) else {}
                metrics = view_item.get("metrics", {}) if isinstance(view_item.get("metrics"), dict) else {}
                metric_item = metrics.get(metric, {})
                if not isinstance(metric_item, dict):
                    continue
                score = finite_float(metric_item.get("score_normalized"))
                if score is not None:
                    per_view[view] = clamp01(score)
            agg = _robust_view_aggregate(per_view, outlier_config)
            if agg is not None:
                episode_scores[metric] = agg

        head_instruction = (
            views.get("head", {}).get("metrics", {}).get("Instruction_Following", {})
            if isinstance(views.get("head"), dict)
            else {}
        )
        if isinstance(head_instruction, dict):
            instruction_score = finite_float(head_instruction.get("score_normalized"))
            if instruction_score is not None:
                episode_scores["Instruction_Following"] = clamp01(instruction_score)

        if episode_scores:
            result[episode] = episode_scores

    return result


def save_vlm_episode_scores(
    temp_dir: Path,
    vlm_data: list[dict[str, Any]],
    vlm_config: dict[str, Any],
    metric_indices: dict[str, int],
) -> None:
    by_episode = extract_vlm_episode_scores(vlm_data, vlm_config)
    metric_episodes: dict[str, dict[str, float]] = {name: {} for name in metric_indices}
    for episode, scores in by_episode.items():
        for metric_name, value in scores.items():
            if metric_name in metric_episodes:
                metric_episodes[metric_name][episode] = value
    for metric_name, index in metric_indices.items():
        if metric_episodes[metric_name]:
            save_episode_scores(temp_dir, metric_name, index, metric_episodes[metric_name])


def remove_metric_dirs_without_scores(temp_dir: Path) -> None:
    if not temp_dir.is_dir():
        return
    for path in temp_dir.iterdir():
        if not path.is_dir() or not path.name.startswith("Metric_"):
            continue
        if not scores_path(temp_dir, path.name.removeprefix("Metric_")).is_file():
            shutil.rmtree(path, ignore_errors=True)


def cleanup_run_artifacts(ctx: Any, *, aggressive: bool = False) -> None:
    """Drop intermediate artifacts. Per-metric mode keeps data needed by finalize/resume."""
    temp_dir = ctx.temp_dir
    for rel in (
        "workspace/episode_shards",
        "aggregate/vlm_raw",
        "aggregate/vlm_shards",
    ):
        path = temp_dir / rel
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

    aggregate_dir = temp_dir / "aggregate"
    if aggregate_dir.is_dir():
        for child in aggregate_dir.iterdir():
            if child.name.endswith("_raw") and child.name.startswith("vlm_consistency"):
                shutil.rmtree(child, ignore_errors=True)

    remove_metric_dirs_without_scores(temp_dir)

    if aggressive:
        aggregate_dir = temp_dir / "aggregate"
        if aggregate_dir.is_dir():
            for child in aggregate_dir.iterdir():
                if child.name.endswith("_shards") and child.name.startswith("vlm_consistency"):
                    continue
                shutil.rmtree(child, ignore_errors=True)

        for rel in (
            "workspace",
            "split_views",
            "state_layout",
            "prepared",
            "metric_configs",
        ):
            path = temp_dir / rel
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
        for legacy in ("triworld16", "triworld0710", "prepared_0710", "configs_0710"):
            path = temp_dir / legacy
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)


def all_selected_scores_ready(temp_dir: Path, metric_names: list[str]) -> bool:
    return all(scores_path(temp_dir, name).is_file() for name in metric_names)


def discover_episode_columns(temp_dir: Path, metric_names: list[str]) -> list[str]:
    episodes: set[str] = set()
    for name in metric_names:
        data = load_episode_scores(scores_path(temp_dir, name))
        ep_map = data.get("episodes")
        if isinstance(ep_map, dict):
            episodes.update(ep_map.keys())
    return sort_episode_names(episodes)
