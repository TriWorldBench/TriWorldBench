"""Round-robin GPU worker pool helpers."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from metrics._common.progress import write_progress
from metrics._common.subprocess_env import without_progress_sidecar
from metrics._common.subprocess_worker import spawn_json_worker

HEARTBEAT_INTERVAL_S = 60.0


def parse_gpus(ctx: Any) -> list[str]:
    return [g.strip() for g in str(ctx.cfg.get("gpu_list", "0")).split(",") if g.strip()]


def resolve_parallel_per_gpu(
    cfg: dict[str, Any],
    metric_name: str,
    *,
    metric_index: int | None = None,
    default_override: int | None = None,
) -> int:
    """Resolve per-metric parallel cap; never exceeds global parallel_per_gpu.

    default_override only sets the fallback when no metric override exists.
    """
    global_cap = max(int(cfg.get("parallel_per_gpu", 2)), 1)
    default = default_override if default_override is not None else global_cap
    overrides = cfg.get("metric_parallel_per_gpu") or {}
    if not isinstance(overrides, dict):
        return min(default, global_cap)
    for key in (metric_name, str(metric_index) if metric_index is not None else None):
        if not key or key not in overrides:
            continue
        return min(max(int(overrides[key]), 1), global_cap)
    return min(default, global_cap)


def split_round_robin(items: list[Any], slot_count: int) -> list[list[Any]]:
    buckets: list[list[Any]] = [[] for _ in range(max(1, slot_count))]
    for index, item in enumerate(items):
        buckets[index % len(buckets)].append(item)
    return buckets


def read_worker_json_line(proc: subprocess.Popen[str]) -> dict[str, Any] | None:
    assert proc.stdout is not None
    while True:
        line = proc.stdout.readline()
        if not line:
            return None
        text = line.strip()
        if not text:
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            print(f"[worker] ignored stdout: {text[:200]}", file=sys.stderr, flush=True)


def run_persistent_worker(
    *,
    python: str,
    worker_script: Path,
    worker_args: list[str],
    gpu: str,
    jobs: list[dict[str, Any]],
    env_overrides: dict[str, str],
    job_encoder: Callable[[dict[str, Any]], str],
    on_job_start: Callable[[dict[str, Any]], None] | None = None,
    on_job_done: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    if not jobs:
        return []

    env = without_progress_sidecar(env_overrides)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    proc = spawn_json_worker(
        [python, str(worker_script), *worker_args],
        env=env,
        gpu=gpu,
    )
    assert proc.stdin is not None

    statuses: list[dict[str, Any]] = []
    for job in jobs:
        if on_job_start:
            on_job_start(job)
        proc.stdin.write(job_encoder(job))
        proc.stdin.flush()
        result = read_worker_json_line(proc)
        if result is None:
            failed = {"status": "failed", "error": "worker died", **job}
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


def _emit_pool_progress(done: int, total: int, *, phase: str | None = None) -> None:
    write_progress(done, total, done, total, phase=phase)


class _ProgressTracker:
    def __init__(self, total: int, *, phase: str | None = None) -> None:
        self.total = total
        self.phase = phase
        self.done = 0
        self.active = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._heartbeat: threading.Thread | None = None

    def start(self) -> None:
        self._emit(force=True)
        self._heartbeat = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat.start()

    def stop(self) -> None:
        self._stop.set()
        if self._heartbeat is not None:
            self._heartbeat.join(timeout=1.0)

    def on_job_start(self, _job: dict[str, Any]) -> None:
        with self._lock:
            self.active += 1
        self._emit()

    def on_job_done(self, item: dict[str, Any]) -> int | None:
        with self._lock:
            self.active = max(0, self.active - 1)
            if item.get("status") == "success":
                self.done += 1
                return self.done
        return None

    def _emit(self, *, force: bool = False) -> None:
        with self._lock:
            done = self.done
            active = self.active
            total = self.total
            phase = self.phase
        write_progress(done, total, done, total, phase=phase, active_items=active)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(HEARTBEAT_INTERVAL_S):
            self._emit()
            with self._lock:
                done = self.done
                active = self.active
                total = self.total
            print(
                f"[progress] {self.phase or 'pool'}: {done}/{total} done, {active} active",
                flush=True,
            )


def run_gpu_pool(
    *,
    ctx: Any,
    metric_name: str,
    metric_index: int,
    jobs: list[dict[str, Any]],
    worker_script: Path,
    worker_args: list[str],
    job_encoder: Callable[[dict[str, Any]], str],
    parallel_per_gpu: int | None = None,
    progress_phase: str | None = None,
    on_item_done: Callable[[dict[str, Any], int, int], None] | None = None,
) -> list[dict[str, Any]]:
    gpus = parse_gpus(ctx)
    per_gpu = parallel_per_gpu or resolve_parallel_per_gpu(
        ctx.cfg, metric_name, metric_index=metric_index
    )
    slots = [(gpu, slot) for gpu in gpus for slot in range(per_gpu)]
    buckets = split_round_robin(jobs, len(slots))
    failures: list[dict[str, Any]] = []
    total = len(jobs)
    tracker = _ProgressTracker(total, phase=progress_phase)

    if total:
        tracker.start()

    def _handle_job_done(item: dict[str, Any]) -> None:
        current = tracker.on_job_done(item)
        if current is not None:
            tracker._emit()
            if on_item_done:
                on_item_done(item, current, total)
        elif item.get("status") != "success":
            tracker._emit()

    try:
        with ThreadPoolExecutor(max_workers=len(slots)) as pool:
            futures = {
                pool.submit(
                    run_persistent_worker,
                    python=ctx.python,
                    worker_script=worker_script,
                    worker_args=worker_args,
                    gpu=gpu,
                    jobs=bucket,
                    env_overrides=ctx.env,
                    job_encoder=job_encoder,
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
        raise RuntimeError(f"{metric_name} worker failures: {failures[:3]}")
    return failures
