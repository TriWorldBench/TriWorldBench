"""Shared helpers for metric execution dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from metrics._common.evaluate_one import run_dimension
from metrics._common.semantic_parallel import run_semantic_alignment_parallel
from metrics._common.trajectory_parallel import run_trajectory_parallel
from metrics._common.vlm_parallel import run_vlm
from metrics._common.vlm_consistency_parallel import run_vlm_consistency


def ensure_metric_configs(ctx: Any) -> None:
    from eval_common import ensure_prepared_inputs, metric_configs_need_rebuild

    if not metric_configs_need_rebuild(ctx):
        return
    ensure_prepared_inputs(ctx)


def _trajectory_zero_score(ctx: Any) -> float:
    with ctx.hyperparameters.open("r", encoding="utf-8") as f:
        params = yaml.safe_load(f) or {}
    return float(params.get("trajectory", {}).get("zero_distance_score", 40.854))


def run_dimension_0710(ctx: Any, dimension: str, metric_index: int, metric_name: str) -> None:
    ensure_metric_configs(ctx)
    apply_patches = dimension in {"subject_consistency", "background_consistency"}
    zero_score = _trajectory_zero_score(ctx) if dimension == "trajectory_accuracy" else None
    run_dimension(
        ctx,
        dimension,
        metric_index,
        metric_name,
        config_path=ctx.metric_config_dir / f"{dimension}.yaml",
        apply_0710_patches=apply_patches,
        trajectory_zero_score=zero_score,
    )


def run_semantic_alignment_metric(ctx: Any, metric_index: int, metric_name: str) -> None:
    run_semantic_alignment_parallel(ctx, metric_index, metric_name)


def run_trajectory(ctx: Any, metric_index: int, metric_name: str) -> None:
    run_trajectory_parallel(ctx, metric_index, metric_name)


def run_vlm_metric(ctx: Any, metric_index: int, metric_name: str) -> None:
    """Run per-view VLM judge once (metric 0); other VLM-derived metrics use mark_derived."""
    run_vlm(ctx, metric_index, metric_name)


def run_vlm_consistency_metric(ctx: Any, metric_index: int, metric_name: str) -> None:
    """Run state-based vlm_consistency judge for 01/02/03 prompt variants."""
    from metrics._common.vlm_judge_registry import VLM_CONSISTENCY_VARIANT_BY_METRIC

    variant = VLM_CONSISTENCY_VARIANT_BY_METRIC.get(metric_name)
    if variant is None:
        raise ValueError(f"unknown vlm_consistency metric {metric_name!r}")
    run_vlm_consistency(ctx, metric_index, metric_name, variant)


def mark_derived(ctx: Any, metric_index: int, metric_name: str, note: str = "") -> None:
    """Derived metrics get episode scores during finalize; no placeholder files."""
    _ = (ctx, metric_index, metric_name, note)
