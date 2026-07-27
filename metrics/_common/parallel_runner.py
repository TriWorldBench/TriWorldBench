"""Fixed parallel_per_gpu episode queue with persistent per-slot workers."""

from __future__ import annotations

import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import yaml

from metrics._common.subprocess_env import without_progress_sidecar
from metrics._common.subprocess_worker import spawn_json_worker
from metrics._common.episode_shard import (
    EPISODE_PARALLEL_METRICS,
    EpisodeRef,
    discover_episodes_from_val_base,
    episode_shard_done,
    merge_metric_shard_results,
    prepare_metric_episode_shard,
)
from metrics._common.metric_scores import (
    cleanup_run_artifacts,
    completed_episodes,
    extract_triworld_episode_scores,
    save_episode_scores,
    scores_path,
)
from metrics._common.gpu_pool import (
    _ProgressTracker,
    parse_gpus,
    read_worker_json_line,
    resolve_parallel_per_gpu,
)

WORKER_SCRIPT = Path(__file__).resolve().parent / "evaluate_worker.py"
_SCORE_REFRESH_LOCK = threading.Lock()


def _config_yaml_name(dimension: str) -> str:
    if dimension in ("psnr", "ssim"):
        return "psnr_ssim"
    if dimension == "image_quality":
        return "image_quality"
    return dimension


def _resolve_gen_root(config_dir: Path, dimension: str) -> Path:
    metric_cfg = config_dir / f"{_config_yaml_name(dimension)}.yaml"
    if metric_cfg.is_file():
        with metric_cfg.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        val_base = cfg.get("data", {}).get("val_base")
        if val_base:
            return Path(val_base)
    raise RuntimeError(f"cannot resolve val_base for {dimension} under {config_dir}")


def _pending_episodes(
    ctx: Any,
    dimension: str,
    gen_root: Path,
    metric_name: str,
) -> list[EpisodeRef]:
    all_eps = discover_episodes_from_val_base(gen_root)
    completed = completed_episodes(ctx.temp_dir, metric_name)
    pending: list[EpisodeRef] = []
    for ref in all_eps:
        if ref.key in completed:
            continue
        if episode_shard_done(ctx.work, dimension, ref):
            continue
        pending.append(ref)
    return pending


def _persist_metric_scores(
    ctx: Any,
    metric_name: str,
    metric_index: int,
    dimension: str,
    marker: Path,
) -> None:
    merged = json.loads(marker.read_text(encoding="utf-8"))
    scores = extract_triworld_episode_scores(dimension, merged)
    save_episode_scores(ctx.temp_dir, metric_name, metric_index, scores)


def _refresh_partial_scores(
    ctx: Any,
    metric_name: str,
    metric_index: int,
    dimension: str,
) -> None:
    shard_root = ctx.work / "output_metrics" / dimension / "shards"
    if not shard_root.is_dir():
        return
    episodes: dict[str, float | None] = {}
    for shard_file in sorted(shard_root.glob("*/generated_results.json")):
        try:
            merged = json.loads(shard_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        scores = extract_triworld_episode_scores(dimension, merged)
        episodes.update(scores)
    if episodes:
        with _SCORE_REFRESH_LOCK:
            save_episode_scores(ctx.temp_dir, metric_name, metric_index, episodes)


def _run_slot_worker(
    ctx: Any,
    gpu: str,
    dimension: str,
    config_dir: Path,
    jobs: list[tuple[EpisodeRef, Path]],
    worker_env: dict[str, str],
    apply_0710_patches: bool,
    trajectory_zero_score: float | None,
    *,
    on_job_start: Callable[[dict[str, Any]], None] | None = None,
    on_job_done: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    if not jobs:
        return []

    env = without_progress_sidecar(worker_env)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    cmd = [
        ctx.python,
        str(WORKER_SCRIPT),
        "--dimension", dimension,
        "--config-dir", str(config_dir),
    ]
    if apply_0710_patches:
        cmd.append("--apply-0710-patches")
    if trajectory_zero_score is not None:
        cmd.extend(["--trajectory-zero-score", str(trajectory_zero_score)])

    proc = spawn_json_worker(cmd, env=env, gpu=gpu)
    assert proc.stdin is not None
    assert proc.stdout is not None

    statuses: list[dict[str, Any]] = []
    for ref, cfg_path in jobs:
        job = {"episode": ref.key, "config_path": str(cfg_path)}
        if on_job_start:
            on_job_start(job)
        job_line = json.dumps(job) + "\n"
        proc.stdin.write(job_line)
        proc.stdin.flush()
        result = read_worker_json_line(proc)
        if result is None:
            failed = {"episode": ref.key, "status": "failed", "error": "worker died"}
            statuses.append(failed)
            if on_job_done:
                on_job_done(failed)
            break
        statuses.append(result)
        if on_job_done:
            on_job_done(result)

    proc.stdin.write("STOP\n")
    proc.stdin.flush()
    proc.stdin.close()
    proc.wait()
    return statuses


def run_episode_parallel(
    ctx: Any,
    dimension: str,
    metric_index: int,
    metric_name: str,
    *,
    config_dir: Path,
    apply_0710_patches: bool = False,
    trajectory_zero_score: float | None = None,
) -> None:
    """Run dimension via fixed parallel_per_gpu workers per GPU."""
    if dimension not in EPISODE_PARALLEL_METRICS:
        raise ValueError(f"dimension {dimension} is not episode-parallel")

    out_dir = ctx.work / "output_metrics" / dimension
    marker = out_dir / "generated_results.json"
    score_file = scores_path(ctx.temp_dir, metric_name)

    gen_root = _resolve_gen_root(config_dir, dimension)
    pending = _pending_episodes(ctx, dimension, gen_root, metric_name)

    if score_file.is_file() and ctx.resume and not pending:
        scored = completed_episodes(ctx.temp_dir, metric_name)
        expected = {ref.key for ref in discover_episodes_from_val_base(gen_root)}
        if scored >= expected:
            print(f"[skip] {metric_name}: all episodes scored")
            return
        print(
            f"[resume] {metric_name}: score file incomplete "
            f"({len(scored)}/{len(expected)} episodes), re-merging shards",
            flush=True,
        )

    if not pending:
        shard_root = out_dir / "shards"
        if marker.is_file():
            _persist_metric_scores(ctx, metric_name, metric_index, dimension, marker)
            if shard_root.exists():
                shutil.rmtree(shard_root)
            cleanup_run_artifacts(ctx)
            return
        if shard_root.is_dir() and any(shard_root.glob("*/generated_results.json")):
            merge_metric_shard_results(dimension, shard_root, marker)
            _persist_metric_scores(ctx, metric_name, metric_index, dimension, marker)
            shutil.rmtree(shard_root)
            cleanup_run_artifacts(ctx)
            print(f"[done] {metric_name} -> {score_file} (merged remaining shards)", flush=True)
            return
        raise RuntimeError(f"no episodes found for {metric_name}")

    gpus = parse_gpus(ctx)
    parallel_per_gpu = resolve_parallel_per_gpu(
        ctx.cfg, metric_name, metric_index=metric_index
    )
    slots = [(gpu, slot) for gpu in gpus for slot in range(parallel_per_gpu)]
    buckets: list[list[tuple[EpisodeRef, Path]]] = [[] for _ in slots]

    for idx, ref in enumerate(pending):
        cfg_path, _ = prepare_metric_episode_shard(ctx.work, config_dir, dimension, ref)
        buckets[idx % len(slots)].append((ref, cfg_path))

    print(
        f"[parallel] {metric_name}: pending={len(pending)} "
        f"gpus={len(gpus)} parallel_per_gpu={parallel_per_gpu}",
        flush=True,
    )

    failures: list[dict[str, Any]] = []
    total = len(pending)
    tracker = _ProgressTracker(total, phase="scoring")
    tracker.start()

    def _handle_job_done(item: dict[str, Any]) -> None:
        current = tracker.on_job_done(item)
        if current is not None:
            tracker._emit()
            _refresh_partial_scores(ctx, metric_name, metric_index, dimension)
            if current == total or current % max(1, total // 20) == 0:
                print(f"[scoring] {metric_name}: {current}/{total} episodes", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=len(slots)) as pool:
            futures = {
                pool.submit(
                    _run_slot_worker,
                    ctx,
                    gpu,
                    dimension,
                    config_dir,
                    bucket,
                    ctx.env,
                    apply_0710_patches,
                    trajectory_zero_score,
                    on_job_start=tracker.on_job_start,
                    on_job_done=_handle_job_done,
                ): (gpu, slot)
                for (gpu, slot), bucket in zip(slots, buckets)
                if bucket
            }
            for future in as_completed(futures):
                gpu, slot = futures[future]
                try:
                    statuses = future.result()
                    for item in statuses:
                        if item.get("status") != "success":
                            failures.append({**item, "gpu": gpu, "slot": slot})
                except Exception as exc:
                    failures.append({"gpu": gpu, "slot": slot, "status": "failed", "error": str(exc)})
    finally:
        tracker.stop()

    if failures:
        raise RuntimeError(f"{metric_name} episode failures: {failures[:3]}")

    shard_root = out_dir / "shards"
    merge_metric_shard_results(dimension, shard_root, marker)
    _persist_metric_scores(ctx, metric_name, metric_index, dimension, marker)
    if shard_root.exists():
        shutil.rmtree(shard_root)
    cleanup_run_artifacts(ctx)
    print(f"[done] {metric_name} -> {score_file}")


def is_episode_parallel(dimension: str) -> bool:
    return dimension in EPISODE_PARALLEL_METRICS
