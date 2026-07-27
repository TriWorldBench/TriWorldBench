#!/usr/bin/env python3
"""Unified metric execution entrypoints (replaces metrics/metricXX_*/run.py)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (
    _PROJECT_ROOT,
    _PROJECT_ROOT / "scripts",
    _PROJECT_ROOT / "scripts" / "lib",
):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from metrics._common.evaluate_one import run_dimension
from metrics._common.metric_helpers import (
    mark_derived,
    run_dimension_0710,
    run_semantic_alignment_metric,
    run_trajectory,
    run_vlm_consistency_metric,
    run_vlm_metric,
)
from metrics.jepa_similarity import run_jepa
from metrics.vqa import run_vqa

ExecuteFn = Callable[[Any], None]


def _run_dynamic_state_alignment(ctx: Any) -> None:
    run_dimension(ctx, "dynamic_degree", 4, "dynamic_state_alignment")
    mark_derived(ctx, 4, "dynamic_state_alignment", note="dynamic_state_alignment from aggregation")


# metric index -> execute(ctx)
METRIC_EXECUTORS: dict[int, ExecuteFn] = {
    0: lambda ctx: run_vlm_metric(ctx, 0, "Instruction_Following"),
    1: lambda ctx: mark_derived(ctx, 1, "Interaction_Quality", note="derived from VLM judge"),
    2: lambda ctx: mark_derived(ctx, 2, "Perspectivity", note="derived from VLM judge"),
    3: lambda ctx: run_dimension(ctx, "background_consistency", 3, "background_consistency"),
    4: _run_dynamic_state_alignment,
    5: lambda ctx: run_dimension(ctx, "flow_score", 5, "flow_score"),
    6: lambda ctx: run_jepa(ctx, 6, "jepa_similarity"),
    7: lambda ctx: run_dimension(ctx, "photometric_smoothness", 7, "photometric_smoothness"),
    8: lambda ctx: run_dimension(ctx, "psnr", 8, "psnr"),
    9: lambda ctx: run_semantic_alignment_metric(ctx, 9, "semantic_alignment"),
    10: lambda ctx: run_dimension(ctx, "ssim", 10, "ssim"),
    11: lambda ctx: run_dimension_0710(ctx, "subject_consistency", 11, "subject_consistency"),
    12: lambda ctx: run_trajectory(ctx, 12, "trajectory_accuracy"),
    13: lambda ctx: run_vlm_consistency_metric(ctx, 13, "vlm_consistency01"),
    14: lambda ctx: run_vlm_consistency_metric(ctx, 14, "vlm_consistency02"),
    15: lambda ctx: run_vlm_consistency_metric(ctx, 15, "vlm_consistency03"),
    16: lambda ctx: run_vqa(ctx, 16, "VQA"),
    17: lambda ctx: run_dimension(ctx, "aesthetic_quality", 17, "aesthetic_quality"),
    18: lambda ctx: run_dimension(ctx, "image_quality", 18, "image_quality"),
}


def run_metric(ctx: Any, metric_index: int, metric_name: str) -> None:
    executor = METRIC_EXECUTORS.get(metric_index)
    if executor is None:
        raise KeyError(f"no executor for metric {metric_index} ({metric_name})")
    executor(ctx)


def main() -> int:
    from eval_common import build_context, ensure_workspace, load_config  # noqa: WPS433
    from metric_registry import expand_dependencies, resolve_metrics  # noqa: WPS433

    parser = argparse.ArgumentParser(description="Run one benchmark metric by index.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--skip-prep", action="store_true")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Reuse orchestrator run directory (required when resume=false)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    selected = expand_dependencies(resolve_metrics(cfg.get("metrics", "all")), cfg)
    ctx = build_context(args.config, cfg, selected=selected, run_dir=args.run_dir)
    if not args.skip_prep:
        ensure_workspace(ctx)
    run_metric(ctx, args.index, args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
