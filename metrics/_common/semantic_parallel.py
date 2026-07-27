"""8-GPU semantic_alignment: GT/gen Qwen caption workers + parallel CLIP scoring."""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from metrics._common.compute_registry import dimension_kwargs
from metrics._common.evaluate_one import _add_normalized_scores, _load_yaml, _to_standard_results
from metrics._common.episode_shard import EpisodeRef, discover_episodes_from_val_base
from metrics._common.gpu_pool import (
    parse_gpus,
    resolve_parallel_per_gpu,
    run_persistent_worker,
    split_round_robin,
)
from metrics._common.metric_scores import cleanup_run_artifacts, save_episode_scores, scores_path
from metrics.semantic_alignment import (
    build_clip_score_jobs,
    clip_scores_to_results_list,
    list_gen_episode_entries,
    list_gt_caption_entries,
    load_merged_caption_shards,
    merge_caption_shards,
    pending_caption_entries,
)

CAPTION_WORKER_SCRIPT = Path(__file__).resolve().parent / "semantic_caption_worker.py"
CLIP_WORKER_SCRIPT = Path(__file__).resolve().parent / "semantic_clip_worker.py"


def _load_metric_config(ctx: Any) -> dict[str, Any]:
    return _load_yaml(ctx.metric_config_dir / "semantic_alignment.yaml")


def _caption_shard_path(shard_dir: Path, gpu: str, slot: int) -> Path:
    return shard_dir / f"gpu{gpu}_s{slot}.json"


def _resolve_slots(ctx: Any, metric_name: str, metric_index: int) -> tuple[list[str], int, list[tuple[str, int]]]:
    gpus = parse_gpus(ctx)
    parallel_per_gpu = resolve_parallel_per_gpu(
        ctx.cfg, metric_name, metric_index=metric_index, default_override=1
    )
    slots = [(gpu, slot) for gpu in gpus for slot in range(parallel_per_gpu)]
    return gpus, parallel_per_gpu, slots


def _run_caption_pool(
    ctx: Any,
    *,
    caption_model: str,
    shard_dir: Path,
    jobs_by_slot: list[list[dict[str, Any]]],
    slots: list[tuple[str, int]],
    phase: str,
    metric_name: str,
) -> None:
    active = sum(1 for bucket in jobs_by_slot if bucket)
    if not active:
        return

    shard_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[parallel] {metric_name}: {phase} workers={active} "
        f"jobs={sum(len(bucket) for bucket in jobs_by_slot)}",
        flush=True,
    )

    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = {}
        for (gpu, slot), bucket in zip(slots, jobs_by_slot):
            if not bucket:
                continue
            shard_path = _caption_shard_path(shard_dir, gpu, slot)
            futures[pool.submit(
                run_persistent_worker,
                python=ctx.python,
                worker_script=CAPTION_WORKER_SCRIPT,
                worker_args=["--model-path", caption_model, "--shard-path", str(shard_path)],
                gpu=gpu,
                jobs=bucket,
                env_overrides=ctx.env,
                job_encoder=lambda job: json.dumps(job, ensure_ascii=False) + "\n",
            )] = (gpu, slot)
        for future in as_completed(futures):
            gpu, slot = futures[future]
            try:
                statuses = future.result()
                for item in statuses:
                    if item.get("status") != "success":
                        failures.append({**item, "gpu": gpu, "slot": slot})
            except Exception as exc:
                failures.append({"gpu": gpu, "slot": slot, "status": "failed", "error": str(exc)})

    if failures:
        raise RuntimeError(f"{metric_name} {phase} caption failures: {failures[:3]}")


def _run_clip_pool(
    ctx: Any,
    *,
    clip_model: str,
    jobs: list[dict[str, str]],
    slots: list[tuple[str, int]],
    metric_name: str,
) -> dict[str, float]:
    if not jobs:
        return {}

    jobs_by_slot = split_round_robin(jobs, len(slots))
    active = sum(1 for bucket in jobs_by_slot if bucket)
    print(
        f"[parallel] {metric_name}: clip workers={active} pairs={len(jobs)}",
        flush=True,
    )

    scores: dict[str, float] = {}
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = {}
        for (gpu, slot), bucket in zip(slots, jobs_by_slot):
            if not bucket:
                continue
            futures[pool.submit(
                run_persistent_worker,
                python=ctx.python,
                worker_script=CLIP_WORKER_SCRIPT,
                worker_args=["--model-path", clip_model],
                gpu=gpu,
                jobs=bucket,
                env_overrides=ctx.env,
                job_encoder=lambda job: json.dumps(job, ensure_ascii=False) + "\n",
            )] = (gpu, slot)
        for future in as_completed(futures):
            gpu, slot = futures[future]
            try:
                statuses = future.result()
                for item in statuses:
                    if item.get("status") != "success":
                        failures.append({**item, "gpu": gpu, "slot": slot})
                        continue
                    gen_id = str(item.get("gen_id") or "")
                    if gen_id:
                        scores[gen_id] = float(item["score"])
            except Exception as exc:
                failures.append({"gpu": gpu, "slot": slot, "status": "failed", "error": str(exc)})

    if failures:
        raise RuntimeError(f"{metric_name} clip failures: {failures[:3]}")
    if len(scores) != len(jobs):
        raise RuntimeError(
            f"{metric_name} clip incomplete: scored {len(scores)}/{len(jobs)} pairs"
        )
    return scores


def _pending_gen_episodes(gen_root: Path, shard_dir: Path, view: str) -> list[EpisodeRef]:
    all_eps = discover_episodes_from_val_base(gen_root)
    merged = load_merged_caption_shards(shard_dir)

    pending: list[EpisodeRef] = []
    for ref in all_eps:
        entries = list_gen_episode_entries(gen_root, ref.episode_id, view)
        if not entries:
            continue
        if all(entry["key"] in merged for entry in entries):
            continue
        pending.append(ref)
    return pending


def _build_gt_caption_jobs(
    pending_entries: list[dict[str, str]],
    slots: list[tuple[str, int]],
) -> list[list[dict[str, Any]]]:
    jobs_by_slot: list[list[dict[str, Any]]] = [[] for _ in slots]
    for idx, entry in enumerate(pending_entries):
        episode_id = str(entry["key"]).rsplit("_", 1)[-1]
        jobs_by_slot[idx % len(slots)].append({
            "episode": episode_id,
            "entries": [entry],
        })
    return jobs_by_slot


def _build_gen_caption_jobs(
    gen_root: Path,
    pending: list[EpisodeRef],
    view: str,
    slots: list[tuple[str, int]],
) -> list[list[dict[str, Any]]]:
    jobs_by_slot: list[list[dict[str, Any]]] = [[] for _ in slots]
    for idx, ref in enumerate(pending):
        jobs_by_slot[idx % len(slots)].append({
            "episode": ref.episode_id,
            "entries": list_gen_episode_entries(gen_root, ref.episode_id, view),
        })
    return jobs_by_slot


def _run_gt_captions_parallel(
    ctx: Any,
    *,
    caption_model: str,
    gt_root: Path,
    save_path: Path,
    view: str,
    slots: list[tuple[str, int]],
    metric_name: str,
    resume: bool,
) -> Path:
    gt_json = save_path / "gt_caption_responses.json"
    if gt_json.is_file() and resume:
        print(f"[skip] GT captions exist: {gt_json}", flush=True)
        return gt_json

    gt_shard_dir = save_path / "gt_caption_shards"
    all_entries = list_gt_caption_entries(gt_root, view)
    if not all_entries:
        raise RuntimeError(f"no GT caption entries under {gt_root}")

    pending_entries = pending_caption_entries(all_entries, gt_shard_dir)
    if pending_entries:
        jobs_by_slot = _build_gt_caption_jobs(pending_entries, slots)
        _run_caption_pool(
            ctx,
            caption_model=caption_model,
            shard_dir=gt_shard_dir,
            jobs_by_slot=jobs_by_slot,
            slots=slots,
            phase="gt_caption",
            metric_name=metric_name,
        )

    merge_caption_shards(gt_shard_dir, gt_json)
    if gt_shard_dir.exists():
        shutil.rmtree(gt_shard_dir)
    print(f"[done] GT captions -> {gt_json}", flush=True)
    return gt_json


def _run_gen_captions_parallel(
    ctx: Any,
    *,
    caption_model: str,
    gen_root: Path,
    save_path: Path,
    view: str,
    slots: list[tuple[str, int]],
    metric_name: str,
) -> Path:
    shard_dir = save_path / "caption_shards"
    gen_json = save_path / f"{view}_caption_responses.json"

    pending = _pending_gen_episodes(gen_root, shard_dir, view)
    if pending:
        jobs_by_slot = _build_gen_caption_jobs(gen_root, pending, view, slots)
        _run_caption_pool(
            ctx,
            caption_model=caption_model,
            shard_dir=shard_dir,
            jobs_by_slot=jobs_by_slot,
            slots=slots,
            phase="gen_caption",
            metric_name=metric_name,
        )

    merge_caption_shards(shard_dir, gen_json)
    return gen_json


def run_semantic_alignment_parallel(ctx: Any, metric_index: int, metric_name: str) -> None:
    from metrics._common.metric_helpers import ensure_metric_configs

    ensure_metric_configs(ctx)
    score_file = scores_path(ctx.temp_dir, metric_name)
    if score_file.is_file() and ctx.resume:
        print(f"[skip] {metric_name}")
        return

    config = _load_metric_config(ctx)
    gen_root = Path(config["data"]["val_base"])
    gt_root = Path(config["data"]["gt_path"])
    save_path = Path(config["save_path"])
    save_path.mkdir(parents=True, exist_ok=True)

    from metrics.semantic_alignment import resolve_semantic_root

    gen_root, view = resolve_semantic_root(gen_root)
    gt_root, _gt_view = resolve_semantic_root(gt_root)

    kwargs = dimension_kwargs(config, "semantic_alignment")
    caption_model = kwargs["semantic_alignment_caption_model_ckpt"]
    clip_model = kwargs["semantic_alignment_clip_model_ckpt"]

    gpus, parallel_per_gpu, slots = _resolve_slots(ctx, metric_name, metric_index)
    print(
        f"[parallel] {metric_name}: gpus={len(gpus)} parallel_per_gpu={parallel_per_gpu}",
        flush=True,
    )

    gt_json = _run_gt_captions_parallel(
        ctx,
        caption_model=caption_model,
        gt_root=gt_root,
        save_path=save_path,
        view=view,
        slots=slots,
        metric_name=metric_name,
        resume=ctx.resume,
    )
    gen_json = _run_gen_captions_parallel(
        ctx,
        caption_model=caption_model,
        gen_root=gen_root,
        save_path=save_path,
        view=view,
        slots=slots,
        metric_name=metric_name,
    )

    clip_jobs = build_clip_score_jobs(gen_json, gt_json)
    print(f"[parallel] {metric_name}: clip pairs={len(clip_jobs)}", flush=True)
    clip_scores = _run_clip_pool(
        ctx,
        clip_model=clip_model,
        jobs=clip_jobs,
        slots=slots,
        metric_name=metric_name,
    )
    raw_results = clip_scores_to_results_list(clip_scores)

    standard = _add_normalized_scores(
        "semantic_alignment",
        _to_standard_results("semantic_alignment", raw_results, str(gen_root)),
    )

    marker = save_path / "generated_results.json"
    marker.write_text(json.dumps({"semantic_alignment": standard}, ensure_ascii=False, indent=2), encoding="utf-8")

    from metrics._common.metric_scores import extract_triworld_episode_scores

    episode_scores = extract_triworld_episode_scores("semantic_alignment", {"semantic_alignment": standard})
    save_episode_scores(ctx.temp_dir, metric_name, metric_index, episode_scores)

    shard_dir = save_path / "caption_shards"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    cleanup_run_artifacts(ctx)
    print(f"[done] {metric_name} -> {score_file}")
