#!/usr/bin/env python3
"""Build state-aware tri-view Triworld metrics for the 0710 protocol."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from metrics._common.episode_sort import VIEW_ORDER, episode_sort_key, sort_episode_names, sort_episode_items
from metrics._common.metric_scores import clamp01, finite_float, normalize_npsnr, parse_video_path

CONSISTENCY_METRICS = (
    "subject_consistency",
    "background_consistency",
    "photometric_smoothness",
)

QUALITY_PENALTY_METRICS = frozenset({"aesthetic_quality", "image_quality"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triworld-results", type=Path, required=True)
    parser.add_argument(
        "--inputs-root",
        type=Path,
        required=True,
        help="Prepared Triworld inputs; used to align STATE to generated frame count.",
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        required=True,
        help="Flat STATE layout: <root>/episodeN.json",
    )
    parser.add_argument("--vlm", type=Path, required=True)
    parser.add_argument("--hyperparameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return data


def numeric_float(value: Any) -> float | None:
    """Return finite numbers and infinities, but reject NaN/invalid values."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    return sum(clean) / len(clean) if clean else None


def metric_rows(data: dict[str, Any], metric: str) -> list[dict[str, Any]]:
    entry = data.get(metric)
    if not isinstance(entry, list) or len(entry) < 2 or not isinstance(entry[1], list):
        return []
    return [row for row in entry[1] if isinstance(row, dict)]


def preferred_score(row: dict[str, Any], metric: str | None = None) -> float | None:
    if metric == "psnr":
        return numeric_float(row.get("video_results"))
    normalized = finite_float(row.get("video_results_normalized"))
    return normalized if normalized is not None else numeric_float(row.get("video_results"))


def normalize_psnr(value: Any, scale_c: Any = 20.0) -> float | None:
    c = 20.0 if scale_c is None else float(scale_c)
    return normalize_npsnr(value, scale_c=c)


def ingest_triworld_results(data: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    records: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for metric in data:
        for row in metric_rows(data, metric):
            parsed = parse_video_path(row.get("video_path"))
            score = preferred_score(row, metric)
            if parsed is None or score is None:
                continue
            episode, view = parsed
            records[episode][view][metric] = score
    return records


def bool_intervals(values: list[bool]) -> list[dict[str, Any]]:
    if not values:
        return []
    intervals: list[dict[str, Any]] = []
    start = 0
    current = values[0]
    for index in range(1, len(values)):
        if values[index] == current:
            continue
        intervals.append(
            {"start_frame": start, "end_frame": index - 1, "moving": current}
        )
        start = index
        current = values[index]
    intervals.append(
        {"start_frame": start, "end_frame": len(values) - 1, "moving": current}
    )
    return intervals


def state_view_motion(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    num_frames = max(0, int(record.get("num_frames") or 0))
    moving_phases = set(config.get("moving_phases", []))
    arm_motion = {
        "left": [False] * num_frames,
        "right": [False] * num_frames,
    }
    for phase in record.get("phases", []):
        if not isinstance(phase, dict) or phase.get("phase") not in moving_phases:
            continue
        arm = phase.get("arm")
        if arm not in arm_motion or num_frames == 0:
            continue
        start = max(0, int(phase.get("start_frame", 0)))
        end = min(num_frames - 1, int(phase.get("end_frame", start)))
        for index in range(start, end + 1):
            arm_motion[arm][index] = True

    if config.get("head_motion_rule", "any_arm") != "any_arm":
        raise ValueError("0710 currently supports only head_motion_rule=any_arm")
    view_motion = {
        "left": arm_motion["left"],
        "right": arm_motion["right"],
        "head": [left or right for left, right in zip(arm_motion["left"], arm_motion["right"])],
    }
    output: dict[str, Any] = {}
    for view in VIEW_ORDER:
        flags = view_motion[view]
        moving_frames = sum(flags)
        output[view] = {
            "moving_frames": moving_frames,
            "static_frames": num_frames - moving_frames,
            "total_frames": num_frames,
            "moving_fraction": moving_frames / num_frames if num_frames else None,
            "intervals": bool_intervals(flags),
        }
    return output


def generated_frame_count(prepared_head_root: Path, episode: str) -> int:
    video_dir = prepared_head_root / "generated_dataset/head" / episode / "1/video"
    if not video_dir.is_dir():
        return 0
    return sum(
        path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for path in video_dir.iterdir()
    )


def ingest_state_root(
    state_root: Path,
    inputs_root: Path,
    expected_episodes: set[str],
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not state_root.is_dir():
        raise FileNotFoundError(f"STATE root not found: {state_root}")
    records: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for episode in sort_episode_names(expected_episodes):
        path = state_root / f"{episode}.json"
        if not path.is_file():
            missing.append(episode)
            continue
        data = load_json(path)
        if not isinstance(data, dict):
            missing.append(f"{episode} (invalid JSON object)")
            continue
        source_num_frames = int(data.get("num_frames") or 0)
        evaluated_num_frames = generated_frame_count(inputs_root, episode)
        if evaluated_num_frames <= 0:
            missing.append(f"{episode} (no generated head frames)")
            continue
        aligned = dict(data)
        aligned["num_frames"] = min(source_num_frames, evaluated_num_frames)
        records[episode] = {
            "source": str(path),
            "source_num_frames": source_num_frames,
            "evaluated_num_frames": aligned["num_frames"],
            "active_arms": data.get("active_arms", []),
            "views": state_view_motion(aligned, config),
        }
    if missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(f"STATE/input alignment failed for {len(missing)} episode(s): {preview}")
    if not records:
        raise RuntimeError(f"no per-episode STATE JSON found under {state_root}")
    return records


def robust_view_aggregate(
    per_view: dict[str, float], config: dict[str, Any]
) -> tuple[float | None, list[str], list[str]]:
    ordered = [(view, per_view[view]) for view in VIEW_ORDER if view in per_view]
    if not ordered:
        return None, [], []
    if not config.get("enabled", True) or len(ordered) < 3:
        return mean([score for _, score in ordered]), [view for view, _ in ordered], []

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

    included = [view for view, _, _ in kept]
    excluded = [view for view, _ in ordered if view not in included]
    return mean([score for _, score, _ in kept]), included, excluded


def ingest_vlm(
    data: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    outlier_config = config.get("outlier_filter", {})
    multi_metrics = config.get("aggregate_multiview_metrics", ["Interaction_Quality", "Perspectivity"])

    for item in data:
        if not isinstance(item, dict):
            continue
        episode = str(item.get("episode") or "")
        if not episode:
            continue
        views = item.get("views", {}) if isinstance(item.get("views"), dict) else {}
        result: dict[str, Any] = {"per_view": {}, "metrics": {}}
        for view in VIEW_ORDER:
            view_item = views.get(view, {}) if isinstance(views.get(view), dict) else {}
            metrics = view_item.get("metrics", {}) if isinstance(view_item.get("metrics"), dict) else {}
            result["per_view"][view] = metrics

        for metric in multi_metrics:
            per_view: dict[str, float] = {}
            for view in VIEW_ORDER:
                metric_item = result["per_view"].get(view, {}).get(metric, {})
                if not isinstance(metric_item, dict):
                    continue
                score = finite_float(metric_item.get("score_normalized"))
                if score is not None:
                    per_view[view] = clamp01(score)
            score, included, excluded = robust_view_aggregate(per_view, outlier_config)
            result["metrics"][metric] = {
                "score_normalized": score,
                "per_view": per_view,
                "included_views": included,
                "excluded_views": excluded,
            }

        head_instruction = (
            result["per_view"].get("head", {}).get("Instruction_Following", {})
            if isinstance(result["per_view"].get("head"), dict)
            else {}
        )
        instruction_score = finite_float(head_instruction.get("score_normalized"))
        result["metrics"]["Instruction_Following"] = {
            "score_normalized": clamp01(instruction_score) if instruction_score is not None else None,
            "included_views": ["head"] if instruction_score is not None else [],
            "excluded_views": [],
        }
        out[episode] = result
    return out


def motion_factor(
    generated_dynamic: float,
    moving_fraction: float,
    threshold: float,
    config: dict[str, Any],
) -> float:
    under_motion = 1.0
    unexpected_motion = 1.0
    if generated_dynamic < threshold:
        under_motion = math.exp(
            float(config.get("under_motion_lambda", 1.0))
            * (generated_dynamic - threshold)
        )
    if generated_dynamic > threshold:
        unexpected_motion = math.exp(
            -float(config.get("unexpected_motion_lambda", 1.0))
            * generated_dynamic
        )
    return moving_fraction * under_motion + (1.0 - moving_fraction) * unexpected_motion


def dynamic_alignment(
    generated_dynamic: float,
    moving_fraction: float,
    threshold: float,
    config: dict[str, Any],
) -> float:
    moving_score = clamp01(generated_dynamic)
    if generated_dynamic < threshold:
        moving_score *= math.exp(
            float(config.get("under_motion_lambda", 1.0))
            * (generated_dynamic - threshold)
        )
    static_score = 1.0
    if generated_dynamic > threshold:
        static_score = math.exp(
            -float(config.get("unexpected_motion_lambda", 1.0))
            * generated_dynamic
        )
    return moving_fraction * moving_score + (1.0 - moving_fraction) * static_score


def weighted_mean(values: dict[str, float | None], weights: dict[str, Any]) -> float | None:
    pairs = []
    for key, weight_value in weights.items():
        value = values.get(key)
        weight = finite_float(weight_value)
        if value is not None and weight is not None and weight > 0:
            pairs.append((value, weight))
    total_weight = sum(weight for _, weight in pairs)
    return sum(value * weight for value, weight in pairs) / total_weight if total_weight else None


def compute_episode(
    episode: str,
    generated: dict[str, dict[str, float]],
    state_motion: dict[str, Any] | None,
    vlm: dict[str, Any] | None,
    params: dict[str, Any],
    *,
    vlm_consistency_mean: float | None = None,
    include_penalized_metrics: bool = False,
) -> dict[str, Any]:
    motion_config = params.get("motion_state", {})
    gen_thresholds = motion_config.get("generated_dynamic_thresholds", {})
    consistency_metrics = motion_config.get("consistency_metrics", list(CONSISTENCY_METRICS))
    state_views = state_motion.get("views", {}) if isinstance(state_motion, dict) else {}

    views: dict[str, Any] = {}
    for view in VIEW_ORDER:
        gen_metrics = generated.get(view, {})
        view_state = state_views.get(view, {}) if isinstance(state_views.get(view), dict) else {}
        gen_dynamic = finite_float(gen_metrics.get("dynamic_degree"))
        moving_fraction = finite_float(view_state.get("moving_fraction"))
        threshold = float(gen_thresholds.get(view, 0.1213))
        factor = (
            motion_factor(gen_dynamic, moving_fraction, threshold, motion_config)
            if gen_dynamic is not None and moving_fraction is not None
            else None
        )

        adjusted_consistency = {}
        for metric in consistency_metrics:
            base = finite_float(gen_metrics.get(metric))
            adjusted_consistency[metric] = base * factor if base is not None and factor is not None else None

        generated_flow = finite_float(gen_metrics.get("flow_score"))
        if moving_fraction is None:
            expected_motion = "unknown"
        elif moving_fraction == 0.0:
            expected_motion = "static"
        elif moving_fraction == 1.0:
            expected_motion = "moving"
        else:
            expected_motion = "mixed"
        views[view] = {
            "expected_motion": expected_motion,
            "state_moving_fraction": moving_fraction,
            "state_moving_frames": view_state.get("moving_frames"),
            "state_static_frames": view_state.get("static_frames"),
            "state_total_frames": view_state.get("total_frames"),
            "state_intervals": view_state.get("intervals", []),
            "generated_dynamic_threshold": threshold,
            "generated_dynamic_degree": gen_dynamic,
            "motion_factor": factor,
            "dynamic_state_alignment": (
                dynamic_alignment(gen_dynamic, moving_fraction, threshold, motion_config)
                if gen_dynamic is not None and factor is not None
                else None
            ),
            "generated_flow_score": generated_flow,
            "consistency_raw": {metric: finite_float(gen_metrics.get(metric)) for metric in consistency_metrics},
            "consistency_adjusted": adjusted_consistency,
        }

    aggregated: dict[str, Any] = {
        "dynamic_state_alignment": mean([views[v]["dynamic_state_alignment"] for v in VIEW_ORDER]),
        "flow_score": mean([views[v]["generated_flow_score"] for v in VIEW_ORDER]),
    }
    for metric in consistency_metrics:
        aggregated[metric] = mean([views[v]["consistency_adjusted"].get(metric) for v in VIEW_ORDER])

    for metric in ("image_quality", "aesthetic_quality", "psnr", "ssim", "jepa_similarity"):
        aggregated[metric] = mean([finite_float(generated.get(view, {}).get(metric)) for view in VIEW_ORDER])

    psnr_scale_c = params.get("psnr", {}).get("scale_c", 20.0)
    aggregated["psnr_normalized"] = mean(
        [normalize_psnr(generated.get(view, {}).get("psnr"), psnr_scale_c) for view in VIEW_ORDER]
    )

    aggregated["trajectory_accuracy"] = finite_float(generated.get("head", {}).get("trajectory_accuracy"))
    aggregated["semantic_alignment"] = finite_float(generated.get("head", {}).get("semantic_alignment"))

    vlm_metrics = vlm.get("metrics", {}) if isinstance(vlm, dict) else {}
    for metric in ("Interaction_Quality", "Perspectivity", "Instruction_Following"):
        metric_item = vlm_metrics.get(metric, {}) if isinstance(vlm_metrics.get(metric), dict) else {}
        aggregated[metric] = finite_float(metric_item.get("score_normalized"))

    quality_config = params.get("quality_penalties", {})
    vlm_consistency = vlm_consistency_mean

    image_quality = aggregated.get("image_quality")
    aesthetic_quality = aggregated.get("aesthetic_quality")
    trajectory = aggregated.get("trajectory_accuracy")
    if quality_config.get("clamp_inputs_to_unit_interval", True):
        image_quality = clamp01(image_quality) if image_quality is not None else None
        aesthetic_quality = clamp01(aesthetic_quality) if aesthetic_quality is not None else None
        trajectory = clamp01(trajectory) if trajectory is not None else None
        if vlm_consistency is not None:
            vlm_consistency = clamp01(vlm_consistency)

    if include_penalized_metrics:
        quality_inputs = {
            "image_quality": image_quality,
            "aesthetic_quality": aesthetic_quality,
        }
        for metric, quality in quality_inputs.items():
            output_key = f"{metric}_penalized"
            metric_config = quality_config.get(metric, {})
            if None in (quality, trajectory, vlm_consistency):
                aggregated[output_key] = None
                continue
            aggregated[output_key] = (
                quality
                * math.exp(
                    float(metric_config.get("trajectory_penalty_A", 2.0))
                    * (trajectory - 1.0)
                )
                * math.exp(
                    float(metric_config.get("vlm_penalty_B", 2.0))
                    * (vlm_consistency - 1.0)
                )
            )

    return {
        "episode": episode,
        "state": state_motion,
        "views": views,
        "vlm": vlm,
        "metrics": aggregated,
    }


def round_tree(value: Any, digits: int) -> Any:
    if isinstance(value, float):
        return round(value, digits) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: round_tree(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [round_tree(item, digits) for item in value]
    return value


from typing import Any


def _load_vlm_consistency_means(ctx: Any, selected_names: set[str]) -> dict[str, float | None]:
    from metrics._common.metric_scores import load_episode_scores, scores_path
    from metrics._common.vlm_judge_registry import VLM_CONSISTENCY_METRIC_NAMES

    per_metric: dict[str, dict[str, float | None]] = {}
    for name in VLM_CONSISTENCY_METRIC_NAMES:
        if name not in selected_names:
            continue
        score_data = load_episode_scores(scores_path(ctx.temp_dir, name))
        episodes_map = score_data.get("episodes")
        if not isinstance(episodes_map, dict):
            continue
        per_metric[name] = {str(ep): finite_float(value) for ep, value in episodes_map.items()}

    episodes = sort_episode_names({ep for scores in per_metric.values() for ep in scores})
    means: dict[str, float | None] = {}
    for episode in episodes:
        values = [
            scores[episode]
            for scores in per_metric.values()
            if episode in scores and scores[episode] is not None
        ]
        means[episode] = mean(values) if values else None
    return means


def _protocol_score_key(metric_name: str, *, include_penalized: bool) -> str:
    if include_penalized and metric_name in QUALITY_PENALTY_METRICS:
        return f"{metric_name}_penalized"
    return metric_name


def _append_quality_penalty_log(
    ctx: Any,
    spec: Any,
    raw_scores: dict[str, float | None],
    penalized_scores: dict[str, float | None],
) -> None:
    log_path = ctx.log_dir / f"metric_{spec.index:02d}_{spec.name}.log"
    lines = [
        "",
        "[quality_penalty] final CSV score uses penalized value; raw base score below",
        f"metric={spec.name} index={spec.index}",
    ]
    for episode in sort_episode_names(set(raw_scores) | set(penalized_scores)):
        lines.append(
            f"  {episode}: raw={raw_scores.get(episode)!r} penalized={penalized_scores.get(episode)!r}"
        )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def build_protocol_results(ctx: Any) -> None:
    """Aggregate base metric JSON + VLM; write selected metric episode scores."""
    from metrics._common.metric_scores import save_episode_scores

    results_path = ctx.work / "output" / "generated_results.json"
    vlm_path = ctx.aggregate_dir / "vlm_per_view_results.json"
    if not vlm_path.is_file():
        vlm_path.write_text("[]", encoding="utf-8")

    params = load_yaml(ctx.hyperparameters)
    triworld_data = load_json(results_path)
    vlm_data = load_json(vlm_path)
    if not isinstance(triworld_data, dict) or not isinstance(vlm_data, list):
        raise ValueError("invalid input result structure")

    selected_names = set(ctx.selected_metric_names)
    include_penalized = bool(selected_names & QUALITY_PENALTY_METRICS)

    generated = ingest_triworld_results(triworld_data)
    vlm = ingest_vlm(vlm_data, params.get("vlm", {}))
    vlm_consistency_by_episode = _load_vlm_consistency_means(ctx, selected_names)

    needs_state_motion = bool(generated) or include_penalized
    if needs_state_motion:
        state_motion = ingest_state_root(
            ctx.state_layout,
            ctx.prepared_dir / "head_data",
            set(generated),
            params.get("motion_state", {}),
        )
    else:
        state_motion = {}

    keys = sort_episode_names(set(generated) | set(vlm) | set(vlm_consistency_by_episode))
    episodes = sort_episode_items(
        [
            compute_episode(
                ep,
                generated.get(ep, {}),
                state_motion.get(ep),
                vlm.get(ep),
                params,
                vlm_consistency_mean=vlm_consistency_by_episode.get(ep),
                include_penalized_metrics=include_penalized,
            )
            for ep in keys
        ]
    )

    derived_names = {
        "dynamic_state_alignment",
        "aesthetic_quality",
        "image_quality",
        "psnr_normalized",
        "Instruction_Following",
        "Interaction_Quality",
        "Perspectivity",
    }
    for spec in ctx.selected_metrics:
        if spec.name not in derived_names:
            continue
        score_key = _protocol_score_key(spec.name, include_penalized=include_penalized)
        ep_scores = {
            item["episode"]: finite_float(item["metrics"].get(score_key))
            for item in episodes
        }
        extra: dict[str, Any] | None = None
        if include_penalized and spec.name in QUALITY_PENALTY_METRICS:
            raw_scores = {
                item["episode"]: finite_float(item["metrics"].get(spec.name))
                for item in episodes
            }
            extra = {
                "score_source": "penalized",
                "raw_scores": raw_scores,
            }
            _append_quality_penalty_log(ctx, spec, raw_scores, ep_scores)
        save_episode_scores(ctx.temp_dir, spec.name, spec.index, ep_scores, extra=extra)

    ctx.aggregate_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    params = load_yaml(args.hyperparameters)
    triworld_data = load_json(args.triworld_results)
    vlm_data = load_json(args.vlm)
    if not isinstance(triworld_data, dict) or not isinstance(vlm_data, list):
        raise ValueError("invalid input result structure")

    generated = ingest_triworld_results(triworld_data)
    state_motion = ingest_state_root(
        args.state_root,
        args.inputs_root,
        set(generated),
        params.get("motion_state", {}),
    )
    vlm = ingest_vlm(vlm_data, params.get("vlm", {}))
    keys = sort_episode_names(set(generated) | set(vlm))
    episodes = sort_episode_items(
        [
            compute_episode(
                ep,
                generated.get(ep, {}),
                state_motion.get(ep),
                vlm.get(ep),
                params,
            )
            for ep in keys
        ]
    )

    metric_names = sorted({name for item in episodes for name in item["metrics"]})
    summary = {
        metric: {
            "mean": mean([finite_float(item["metrics"].get(metric)) for item in episodes]),
            "count": sum(finite_float(item["metrics"].get(metric)) is not None for item in episodes),
        }
        for metric in metric_names
    }
    digits = int(params.get("aggregation", {}).get("output_round_digits", 6))
    output = round_tree(
        {
            "protocol_version": "0710",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "inputs": {
                "triworld_results": str(args.triworld_results),
                "inputs_root": str(args.inputs_root),
                "state_root": str(args.state_root),
                "vlm": str(args.vlm),
                "hyperparameters": str(args.hyperparameters),
            },
            "disabled_metrics": params.get("metric_selection", {}).get("disabled_metrics", []),
            "summary": summary,
            "episodes": episodes,
        },
        digits,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"0710 episodes: {len(episodes)}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
