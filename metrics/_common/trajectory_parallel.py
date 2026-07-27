"""Parallel SAM3 detection + episode-parallel trajectory scoring."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from metrics._common.episode_shard import EpisodeRef, discover_episodes_from_val_base
from metrics._common.evaluate_one import run_dimension
from metrics._common.gpu_pool import parse_gpus, resolve_parallel_per_gpu, run_gpu_pool

WORKER_SCRIPT = Path(__file__).resolve().parent / "detection_worker.py"


def _load_metric_config(ctx: Any) -> dict[str, Any]:
    with (ctx.metric_config_dir / "trajectory_detection.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _detection_state_path(ctx: Any) -> Path:
    return ctx.work / "output_metrics" / "trajectory_accuracy" / "detection_state.json"


def _write_detection_state(
    ctx: Any,
    *,
    phase: str,
    total: int,
    completed: int,
    pending_episodes: list[str] | None = None,
) -> None:
    path = _detection_state_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": phase,
        "total_episodes": total,
        "completed_episodes": completed,
        "pending_episodes": pending_episodes or [],
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "traj_root": str(_load_metric_config(ctx)["data"]["val_base"]),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _pending_detection_episodes(ctx: Any, data_base: Path, gt_base: Path) -> list[EpisodeRef]:
    all_eps = discover_episodes_from_val_base(data_base)
    pending: list[EpisodeRef] = []
    for ref in all_eps:
        episode_path = data_base / ref.episode_id
        if not episode_path.is_dir():
            pending.append(ref)
            continue
        ready = True
        for gid_dir in episode_path.iterdir():
            if gid_dir.is_dir() and not (gid_dir / "traj" / "traj.npy").is_file():
                ready = False
                break
        gt_traj = gt_base / ref.episode_id / "traj" / "traj.npy"
        if not gt_traj.is_file():
            ready = False
        if not ready:
            pending.append(ref)
    return pending


def run_detection_parallel(ctx: Any, metric_name: str, metric_index: int) -> None:
    config = _load_metric_config(ctx)
    data_base = Path(config["data"]["val_base"])
    gt_path = Path(config["data"]["gt_path"])
    model_path = str(config.get("ckpt", {}).get("sam3_model_ckpt", ""))

    all_eps = discover_episodes_from_val_base(data_base)
    pending = _pending_detection_episodes(ctx, data_base, gt_path)
    total = len(all_eps)
    completed = total - len(pending)

    if not pending:
        _write_detection_state(ctx, phase="done", total=total, completed=total)
        print(f"[skip] {metric_name} detection: all {total} episodes ready", flush=True)
        return

    _write_detection_state(
        ctx,
        phase="running",
        total=total,
        completed=completed,
        pending_episodes=[ref.episode_id for ref in pending],
    )

    gpus = parse_gpus(ctx)
    parallel_per_gpu = resolve_parallel_per_gpu(ctx.cfg, metric_name, metric_index=metric_index, default_override=1)
    print(
        f"[parallel] {metric_name} detection pending={len(pending)}/{total} "
        f"(resume {completed} done) gpus={len(gpus)} parallel_per_gpu={parallel_per_gpu}",
        flush=True,
    )

    jobs = [
        {
            "episode": ref.episode_id,
            "task": data_base.name if data_base.name in {"head", "left", "right"} else "head",
            "data_base": str(data_base),
            "gt_path": str(gt_path),
        }
        for ref in pending
    ]

    def _on_detection_done(item: dict[str, Any], done: int, job_total: int) -> None:
        _write_detection_state(
            ctx,
            phase="running",
            total=total,
            completed=completed + done,
            pending_episodes=[job["episode"] for job in jobs[done:]],
        )
        if done == job_total or done % max(1, job_total // 20) == 0:
            print(f"[detection] {metric_name}: {completed + done}/{total} episodes", flush=True)

    run_gpu_pool(
        ctx=ctx,
        metric_name=f"{metric_name}_detection",
        metric_index=metric_index,
        jobs=jobs,
        worker_script=WORKER_SCRIPT,
        worker_args=["--model-path", model_path, "--detect-gt"],
        job_encoder=lambda job: json.dumps(job, ensure_ascii=False) + "\n",
        parallel_per_gpu=parallel_per_gpu,
        progress_phase="detection",
        on_item_done=_on_detection_done,
    )
    _write_detection_state(ctx, phase="done", total=total, completed=total)
    print(f"[done] {metric_name} detection -> {data_base}", flush=True)


def run_trajectory_parallel(ctx: Any, metric_index: int, metric_name: str) -> None:
    from metrics._common.metric_helpers import ensure_metric_configs, _trajectory_zero_score

    ensure_metric_configs(ctx)
    run_detection_parallel(ctx, metric_name, metric_index)
    zero_score = _trajectory_zero_score(ctx)
    print(f"[parallel] {metric_name} scoring (episode-parallel, resume via shards)", flush=True)
    run_dimension(
        ctx,
        "trajectory_accuracy",
        metric_index,
        metric_name,
        config_path=ctx.metric_config_dir / "trajectory_accuracy.yaml",
        trajectory_zero_score=zero_score,
    )
