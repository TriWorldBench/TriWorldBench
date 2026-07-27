#!/usr/bin/env python3
"""Persistent per-GPU worker: load models once, process episode jobs from stdin."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from metrics._common.compute_registry import dimension_kwargs  # noqa: E402
from metrics._common.evaluate_one import (  # noqa: E402
    _add_normalized_scores,
    _compute_dimension,
    _load_yaml,
    _to_standard_results,
    patch_raw_consistency,
    patch_trajectory_accuracy,
)

@contextlib.contextmanager
def _redirect_stdout_to_stderr():
    """Keep worker JSON protocol on stdout; third-party prints go to stderr."""
    stdout_fd = sys.stdout.fileno()
    saved_fd = os.dup(stdout_fd)
    try:
        os.dup2(sys.stderr.fileno(), stdout_fd)
        yield
    finally:
        os.dup2(saved_fd, stdout_fd)
        os.close(saved_fd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=None, help="Override config (0710 shard config)")
    parser.add_argument("--apply-0710-patches", action="store_true")
    parser.add_argument("--trajectory-zero-score", type=float, default=None)
    return parser.parse_args()


def _write_shard_result(dimension: str, eval_dim: str, results: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / "generated_results.json"
    if eval_dim in ("psnr", "ssim") and isinstance(results, dict) and eval_dim in results:
        payload = {eval_dim: results[eval_dim]}
    else:
        payload = {dimension: results}
    marker.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_episode_job(
    dimension: str,
    config_path: Path,
) -> None:
    config = _load_yaml(config_path)
    data_base = config["data"]["val_base"]
    gt_path = config["data"]["gt_path"]
    data_name = os.path.basename(data_base).replace("_dataset", "")
    save_path = Path(config["save_path"])
    save_path.mkdir(parents=True, exist_ok=True)
    kwargs = dimension_kwargs(config, dimension)

    eval_dim = dimension
    if dimension in ("psnr", "ssim"):
        eval_dim = dimension

    results = _compute_dimension(eval_dim, data_base, data_name, gt_path, save_path, kwargs)
    if dimension == "trajectory_accuracy":
        results = _add_normalized_scores(
            dimension,
            _to_standard_results(dimension, results, data_base),
        )
    _write_shard_result(dimension, eval_dim, results, save_path)


def _release_gpu_memory() -> None:
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MATRIC_EVAL_ROOT", str(PROJECT_ROOT))

    if args.apply_0710_patches:
        patch_raw_consistency(args.dimension)
    if args.dimension == "trajectory_accuracy" and args.trajectory_zero_score is not None:
        patch_trajectory_accuracy(args.trajectory_zero_score)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "STOP":
            break
        job = json.loads(line)
        config_path = Path(job["config_path"])
        try:
            with _redirect_stdout_to_stderr():
                run_episode_job(args.dimension, config_path)
            print(json.dumps({"status": "success", "episode": job.get("episode")}), flush=True)
        except Exception as exc:
            print(json.dumps({
                "status": "failed",
                "episode": job.get("episode"),
                "error": str(exc),
            }), flush=True)
        finally:
            _release_gpu_memory()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
