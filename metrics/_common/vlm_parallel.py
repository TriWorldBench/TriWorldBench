"""Shard the per-view VLM judge across GPUs and merge its outputs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from metrics._common.episode_sort import episode_sort_key
from metrics._common.gpu_pool import split_round_robin
from metrics._common.metric_scores import (
    cleanup_run_artifacts,
    save_vlm_episode_scores,
    scores_path,
)


def _save_vlm_scores(ctx: Any, merged: list[dict[str, Any]]) -> None:
    with ctx.hyperparameters.open("r", encoding="utf-8") as f:
        params = yaml.safe_load(f) or {}
    selected = {spec.name: spec.index for spec in ctx.selected_metrics}
    vlm_metrics = {
        name: index
        for name, index in selected.items()
        if name in {"Instruction_Following", "Interaction_Quality", "Perspectivity"}
    }
    if not vlm_metrics:
        return
    save_vlm_episode_scores(ctx.temp_dir, merged, params.get("vlm", {}), vlm_metrics)


def run_vlm(ctx: Any, metric_index: int, metric_name: str) -> None:
    """Run per-view VLM judge (compute lives in metrics/_common/vlm_judge.py)."""
    output = ctx.aggregate_dir / "vlm_per_view_results.json"
    if output.is_file() and ctx.resume and scores_path(ctx.temp_dir, "Instruction_Following").is_file():
        merged = json.loads(output.read_text(encoding="utf-8"))
        if isinstance(merged, list):
            _save_vlm_scores(ctx, merged)
        print(f"[skip] VLM {metric_name}")
        return

    ctx.aggregate_dir.mkdir(parents=True, exist_ok=True)
    judge_script = Path(__file__).resolve().parent / "vlm_judge.py"
    work_dir = ctx.aggregate_dir / "vlm_shards"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    entries = json.loads((ctx.work / "inputs" / "vlm_summary.json").read_text(encoding="utf-8"))
    gpus = [g.strip() for g in str(ctx.cfg.get("gpu_list", "0")).split(",") if g.strip()]
    jobs = []
    for index, bucket in enumerate(split_round_robin(entries, len(gpus))):
        if not bucket:
            continue
        shard_dir = work_dir / f"shard{index:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        summary_path = shard_dir / "summary.json"
        output_path = shard_dir / "results.json"
        summary_path.write_text(json.dumps(bucket, ensure_ascii=False, indent=2), encoding="utf-8")
        command = [
            ctx.python,
            str(judge_script),
            "--summary-json", str(summary_path),
            "--output", str(output_path),
            "--raw-output-dir", str(ctx.aggregate_dir / "vlm_raw" / f"shard{index:02d}"),
            "--config-path", str(ctx.config_dir / "base.yaml"),
            "--hyperparameters", str(ctx.hyperparameters),
            "--overwrite",
        ]
        jobs.append((index, gpus[index % len(gpus)], command, output_path, shard_dir / "judge.log"))

    failures = []
    with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as pool:
        futures = [pool.submit(_run_shard, job) for job in jobs]
        for future in as_completed(futures):
            index, return_code, output_path, log_path = future.result()
            print(f"VLM shard {index}: return_code={return_code}, log={log_path}")
            if return_code != 0 or not output_path.exists():
                failures.append((index, return_code, str(log_path)))
    if failures:
        raise RuntimeError(f"VLM shard failures: {failures}")

    merged = []
    for _, _, _, output_path, _ in sorted(jobs):
        shard_data = json.loads(output_path.read_text(encoding="utf-8"))
        if isinstance(shard_data, list):
            merged.extend(shard_data)
    merged.sort(key=lambda item: episode_sort_key(str(item.get("episode", ""))))
    output.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    _save_vlm_scores(ctx, merged)
    cleanup_run_artifacts(ctx)


def _run_shard(job):
    index, gpu, command, output_path, log_path = job
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    return index, result.returncode, output_path, log_path
