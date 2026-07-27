#!/usr/bin/env python3
"""Triworld benchmark evaluation orchestrator.

Reads config/config.yaml, prepares shared workspace, sequentially invokes
metrics/runner.py for each selected metric, then aggregates and exports CSV.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LIB_DIR = SCRIPT_DIR / "lib"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(LIB_DIR))

from eval_common import (  # noqa: E402
    build_context,
    ensure_workspace,
    export_csv,
    finalize_results,
    load_config,
    pipeline_complete,
    run_cmd,
)
from metric_registry import expand_dependencies, resolve_metrics  # noqa: E402
from pipeline_ui import PipelineUI  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "config.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plain", action="store_true", help="Disable colors and icons")
    return parser.parse_args()


def count_pipeline_steps(selected: list, ctx) -> int:
    """Count total visible pipeline steps for progress bar and step badge."""
    # Prep steps always appear in UI (run or skip); total must match step_skip/step_start count.
    prep_steps = 4  # prepare_split_views, prepare_triworld_inputs, write_triworld_configs, prepare_metric_inputs
    return prep_steps + len(selected) + 2  # finalize + export


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    selected = expand_dependencies(resolve_metrics(cfg.get("metrics", "all")), cfg)
    ctx = build_context(config_path, cfg, selected=selected)
    ui = PipelineUI(enabled=False if args.plain else None)
    exit_code = 1
    try:
        ui.banner()
        ui.config_panel(ctx, [m.name for m in selected])

        if args.dry_run:
            items = [(m.name, PROJECT_ROOT / "metrics" / "runner.py") for m in selected]
            ui.dry_run_list(items)
            exit_code = 0
            return exit_code

        if pipeline_complete(ctx) and ctx.resume:
            ui.step_skip("full_pipeline", f"already complete: {ctx.final_dir / 'summary_metrics.csv'}")
            export_csv(ctx)
            ui.finish_summary(
                ctx,
                episode_csv=ctx.final_dir / "episode_metrics.csv",
                summary_csv=ctx.final_dir / "summary_metrics.csv",
            )
            exit_code = 0
            return exit_code

        ui.set_total_steps(count_pipeline_steps(selected, ctx))
        ui.pipeline_start()

        ensure_workspace(ctx, ui=ui)

        for spec in selected:
            label = f"metric_{spec.index:02d}_{spec.name}"
            progress_file = ctx.log_dir / f".progress_{spec.index:02d}_{spec.name}.json"
            run_cmd([
                ctx.python, str(PROJECT_ROOT / "metrics" / "runner.py"),
                "--config", str(config_path),
                "--run-dir", str(ctx.run_dir),
                "--index", str(spec.index),
                "--name", spec.name,
                "--skip-prep",
            ], env=ctx.env,
               log_path=ctx.log_dir / f"metric_{spec.index:02d}_{spec.name}.log",
               ui=ui,
               step_name=label,
               progress_file=progress_file)
            ui.pipeline_progress()

        ui.step_start("finalize_results", kind="prep")
        t0 = time.time()
        finalize_results(ctx)
        ui.step_done("finalize_results", log_path=ctx.log_dir / "finalize_results.log",
                     elapsed=time.time() - t0)

        t1 = time.time()
        ui.step_start("export_csv", kind="file")
        export_csv(ctx)
        ui.step_done("export_csv", elapsed=time.time() - t1)
        ui.pipeline_progress()

        ui.finish_summary(
            ctx,
            episode_csv=ctx.final_dir / "episode_metrics.csv",
            summary_csv=ctx.final_dir / "summary_metrics.csv",
        )
        exit_code = 0
        return exit_code
    finally:
        ui._restore_terminal()


if __name__ == "__main__":
    raise SystemExit(main())
