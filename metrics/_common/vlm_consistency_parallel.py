"""Shard and run state-based vlm_consistency judges (01/02/03)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from metrics._common.episode_sort import sort_episode_names
from metrics._common.progress import write_progress
from metrics._common.metric_scores import cleanup_run_artifacts, load_episode_scores, save_episode_scores, scores_path
from metrics._common.multiview_eval_common import load_phase_samples_from_state_root, sample_key
from metrics._common.vlm_judge_registry import (
    prompt_file_path,
    uses_state_consistency_judge,
)
from metrics._common.gpu_pool import split_round_robin
from metrics._common.subprocess_env import build_subprocess_env


def _episode_filter(ctx: Any) -> list[str] | None:
    manifest = ctx.split_root / "manifest.json"
    if not manifest.is_file():
        return None
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    episodes: list[str] = []
    for key in ("records", "entries"):
        rows = payload.get(key)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and row.get("episode"):
                    episodes.append(str(row["episode"]))
            if episodes:
                return sort_episode_names(set(episodes))
    return None


def ensure_state_layout(ctx: Any) -> None:
    if ctx.state_layout.is_dir() and any(ctx.state_layout.iterdir()):
        return
    project_root = Path(__file__).resolve().parents[2]
    cmd = [
        ctx.python,
        str(project_root / "scripts" / "prepare_state_layout.py"),
        "--state-root",
        str(ctx.state_root),
        "--output-root",
        str(ctx.state_layout),
        "--clean",
    ]
    manifest = ctx.split_root / "manifest.json"
    if manifest.is_file():
        cmd.extend(["--episodes-file", str(manifest)])
    subprocess.run(cmd, check=True, env=build_subprocess_env(ctx.env))


def _normalized_component_scores(parsed: dict[str, Any]) -> tuple[float | None, float | None]:
    state = parsed.get("state_consistency_score")
    obj = parsed.get("object_consistency_score")
    if state is None or obj is None:
        return None, None
    try:
        state_val = float(state)
        obj_val = float(obj)
    except (TypeError, ValueError):
        return None, None
    scale = 100.0 if max(state_val, obj_val) > 5.0 else 5.0
    return state_val / scale, obj_val / scale


_PROGRESS_POLL_S = 1.0


class _SampleProgressTracker:
    """Poll shard results.json and emit TRIWORLD_PROGRESS_FILE updates."""

    def __init__(self, total: int, shard_outputs: list[Path], *, phase: str = "vlm_judge") -> None:
        self.total = total
        self.shard_outputs = shard_outputs
        self.phase = phase
        self.active = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._heartbeat: threading.Thread | None = None

    def start(self) -> None:
        self._emit()
        self._heartbeat = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat.start()

    def stop(self) -> None:
        self._stop.set()
        if self._heartbeat is not None:
            self._heartbeat.join(timeout=1.0)
        write_progress(self.total, self.total, self.total, self.total, phase=self.phase)

    def on_shard_start(self) -> None:
        with self._lock:
            self.active += 1
        self._emit()

    def on_shard_done(self) -> None:
        with self._lock:
            self.active = max(0, self.active - 1)
        self._emit()

    def _count_done(self) -> int:
        count = 0
        for path in self.shard_outputs:
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, list):
                count += len(data)
        return count

    def _emit(self) -> None:
        done = min(self._count_done(), self.total)
        with self._lock:
            active = self.active
            total = self.total
            phase = self.phase
        write_progress(
            done,
            total,
            done,
            total,
            phase=phase,
            active_items=active if active else None,
        )

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(_PROGRESS_POLL_S):
            self._emit()


def _vlm_reason_entry(item: dict[str, Any], parsed: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "sample_id": item.get("sample_id"),
        "phase": item.get("phase"),
        "arm": item.get("arm"),
        "score_normalized": score,
        "state_consistency_reason": str(parsed.get("state_consistency_reason") or ""),
        "object_consistency_reason": str(parsed.get("object_consistency_reason") or ""),
    }


def aggregate_episode_scores(sample_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_episode: dict[str, list[float]] = {}
    by_state: dict[str, list[float]] = {}
    by_object: dict[str, list[float]] = {}
    details: dict[str, list[dict[str, Any]]] = {}
    reason_low: dict[str, dict[str, Any]] = {}
    reason_high: dict[str, dict[str, Any]] = {}
    state_reason_low: dict[str, dict[str, Any]] = {}
    state_reason_high: dict[str, dict[str, Any]] = {}
    object_reason_low: dict[str, dict[str, Any]] = {}
    object_reason_high: dict[str, dict[str, Any]] = {}
    for item in sample_results:
        episode = str(item.get("episode") or "")
        if not episode:
            continue
        score = item.get("score_normalized")
        state_norm = object_norm = None
        parsed = item.get("vlm_parsed") if isinstance(item.get("vlm_parsed"), dict) else None
        if parsed:
            state_norm, object_norm = _normalized_component_scores(parsed)
            if score is None and "overall_consistency_score" in parsed:
                score = float(parsed["overall_consistency_score"]) / 5.0
            elif score is None and state_norm is not None and object_norm is not None:
                score = (state_norm + object_norm) / 2.0
        if score is None:
            continue
        score = float(score)
        by_episode.setdefault(episode, []).append(score)
        if state_norm is not None:
            by_state.setdefault(episode, []).append(state_norm)
        if object_norm is not None:
            by_object.setdefault(episode, []).append(object_norm)
        details.setdefault(episode, []).append(
            {
                "sample_id": item.get("sample_id"),
                "phase": item.get("phase"),
                "arm": item.get("arm"),
                "score_normalized": score,
                "state_consistency_normalized": state_norm,
                "object_consistency_normalized": object_norm,
            }
        )
        if parsed:
            reason_entry = _vlm_reason_entry(item, parsed, score)
            prev_low = reason_low.get(episode)
            if prev_low is None or score < float(prev_low["score_normalized"]):
                reason_low[episode] = reason_entry
            prev_high = reason_high.get(episode)
            if prev_high is None or score > float(prev_high["score_normalized"]):
                reason_high[episode] = reason_entry
            if state_norm is not None:
                prev_state_low = state_reason_low.get(episode)
                if prev_state_low is None or state_norm < float(prev_state_low["state_consistency_normalized"]):
                    state_reason_low[episode] = {**reason_entry, "state_consistency_normalized": state_norm}
                prev_state_high = state_reason_high.get(episode)
                if prev_state_high is None or state_norm > float(prev_state_high["state_consistency_normalized"]):
                    state_reason_high[episode] = {**reason_entry, "state_consistency_normalized": state_norm}
            if object_norm is not None:
                prev_object_low = object_reason_low.get(episode)
                if prev_object_low is None or object_norm < float(prev_object_low["object_consistency_normalized"]):
                    object_reason_low[episode] = {**reason_entry, "object_consistency_normalized": object_norm}
                prev_object_high = object_reason_high.get(episode)
                if prev_object_high is None or object_norm > float(prev_object_high["object_consistency_normalized"]):
                    object_reason_high[episode] = {**reason_entry, "object_consistency_normalized": object_norm}

    aggregated: dict[str, dict[str, Any]] = {}
    for episode, scores in by_episode.items():
        aggregated[episode] = {
            "score_normalized": sum(scores) / len(scores),
            "state_consistency_normalized": (
                sum(by_state[episode]) / len(by_state[episode]) if episode in by_state else None
            ),
            "object_consistency_normalized": (
                sum(by_object[episode]) / len(by_object[episode]) if episode in by_object else None
            ),
            "num_samples": len(scores),
            "samples": details.get(episode, []),
            "vlm_reason_low": reason_low.get(episode),
            "vlm_reason_high": reason_high.get(episode),
            "vlm_state_reason_low": state_reason_low.get(episode),
            "vlm_state_reason_high": state_reason_high.get(episode),
            "vlm_object_reason_low": object_reason_low.get(episode),
            "vlm_object_reason_high": object_reason_high.get(episode),
        }
    return aggregated


def run_vlm_consistency(ctx: Any, metric_index: int, metric_name: str, variant: str) -> None:
    if not uses_state_consistency_judge(variant):
        raise ValueError(f"vlm_consistency metric {metric_name!r} requires state judge variant, got {variant!r}")

    score_file = scores_path(ctx.temp_dir, metric_name)
    if score_file.is_file() and ctx.resume:
        prior = load_episode_scores(score_file)
        prior_eps = prior.get("episodes") if isinstance(prior.get("episodes"), dict) else {}
        if prior_eps:
            print(f"[skip] VLM consistency ({variant}, {metric_name})")
            return

    ensure_state_layout(ctx)
    episodes = _episode_filter(ctx)
    samples = load_phase_samples_from_state_root(ctx.state_layout, episodes=episodes)
    for sample in samples:
        sample["sample_id"] = sample_key(sample)

    ctx.aggregate_dir.mkdir(parents=True, exist_ok=True)
    judge_script = Path(__file__).resolve().parent / "vlm_consistency_judge.py"
    work_dir = ctx.aggregate_dir / f"{metric_name}_shards"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    prompt_file = prompt_file_path(variant)
    reference_db = ctx.cfg.get("vlm_reference_db")
    reference_db_path = None
    if reference_db:
        path = Path(reference_db)
        if not path.is_absolute():
            path = (Path(__file__).resolve().parents[2] / path).resolve()
        if path.is_dir():
            reference_db_path = path

    max_new_tokens = ctx.cfg.get("vlm_max_new_tokens")
    if max_new_tokens is not None:
        max_new_tokens = int(max_new_tokens)

    gpus = [g.strip() for g in str(ctx.cfg.get("gpu_list", "0")).split(",") if g.strip()]
    jobs = []
    for index, bucket in enumerate(split_round_robin(samples, len(gpus))):
        if not bucket:
            continue
        shard_dir = work_dir / f"shard{index:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        summary_path = shard_dir / "samples.json"
        output_path = shard_dir / "results.json"
        summary_path.write_text(json.dumps(bucket, ensure_ascii=False, indent=2), encoding="utf-8")
        command = [
            ctx.python,
            str(judge_script),
            "--variant",
            variant,
            "--samples-json",
            str(summary_path),
            "--output",
            str(output_path),
            "--raw-output-dir",
            str(ctx.aggregate_dir / f"{metric_name}_raw" / f"shard{index:02d}"),
            "--state-root",
            str(ctx.state_layout),
            "--frames-root",
            str(ctx.eval_input),
            "--config-path",
            str(ctx.config_dir / "base.yaml"),
            "--overwrite",
        ]
        if prompt_file is not None:
            command.extend(["--prompt-file", str(prompt_file)])
        if reference_db_path is not None:
            command.extend(["--reference-db-root", str(reference_db_path)])
        if max_new_tokens is not None:
            command.extend(["--max-new-tokens", str(max_new_tokens)])
        jobs.append((index, gpus[index % len(gpus)], command, output_path, shard_dir / "judge.log"))

    failures = []
    shard_outputs = [output_path for _, _, _, output_path, _ in jobs]
    tracker = _SampleProgressTracker(len(samples), shard_outputs)
    if jobs:
        tracker.start()
    try:
        with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as pool:
            futures = {}
            for job in jobs:
                tracker.on_shard_start()
                futures[pool.submit(_run_shard, job)] = job
            for future in as_completed(futures):
                index, return_code, output_path, log_path = future.result()
                tracker.on_shard_done()
                print(f"VLM consistency shard {index}: return_code={return_code}, log={log_path}")
                if return_code != 0 or not output_path.exists():
                    failures.append((index, return_code, str(log_path)))
    finally:
        if jobs:
            tracker.stop()
    if failures:
        raise RuntimeError(f"VLM consistency shard failures: {failures}")

    merged: list[dict[str, Any]] = []
    for _, _, _, output_path, _ in sorted(jobs):
        shard_data = json.loads(output_path.read_text(encoding="utf-8"))
        if isinstance(shard_data, list):
            merged.extend(shard_data)

    aggregated = aggregate_episode_scores(merged)
    episode_scores = {
        episode: item.get("score_normalized")
        for episode, item in aggregated.items()
        if isinstance(item, dict) and item.get("score_normalized") is not None
    }
    episode_state_scores = {
        episode: item.get("state_consistency_normalized")
        for episode, item in aggregated.items()
        if isinstance(item, dict) and item.get("state_consistency_normalized") is not None
    }
    episode_object_scores = {
        episode: item.get("object_consistency_normalized")
        for episode, item in aggregated.items()
        if isinstance(item, dict) and item.get("object_consistency_normalized") is not None
    }
    episode_vlm_details = {
        episode: {
            "reason_low": item.get("vlm_reason_low"),
            "reason_high": item.get("vlm_reason_high"),
            "state_reason_low": item.get("vlm_state_reason_low"),
            "state_reason_high": item.get("vlm_state_reason_high"),
            "object_reason_low": item.get("vlm_object_reason_low"),
            "object_reason_high": item.get("vlm_object_reason_high"),
        }
        for episode, item in aggregated.items()
        if any(
            item.get(key)
            for key in (
                "vlm_reason_low",
                "vlm_reason_high",
                "vlm_state_reason_low",
                "vlm_state_reason_high",
                "vlm_object_reason_low",
                "vlm_object_reason_high",
            )
        )
    }
    save_episode_scores(
        ctx.temp_dir,
        metric_name,
        metric_index,
        episode_scores,
        extra={
            "judge": variant,
            "episodes_state_consistency": episode_state_scores,
            "episodes_object_consistency": episode_object_scores,
            "episode_vlm_details": episode_vlm_details,
        },
    )
    cleanup_run_artifacts(ctx)


def _run_shard(job):
    index, gpu, command, output_path, log_path = job
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    return index, result.returncode, output_path, log_path
