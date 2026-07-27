"""Helpers for persistent JSON-line worker subprocesses."""

from __future__ import annotations

import subprocess
import sys
import threading
from typing import IO


def start_stderr_drainer(
    stream: IO[str] | None,
    *,
    prefix: str = "",
) -> threading.Thread | None:
    """Drain worker stderr in background to avoid PIPE buffer deadlock."""
    if stream is None:
        return None

    def _drain() -> None:
        for line in stream:
            text = line.rstrip()
            if text:
                print(f"{prefix}{text}", file=sys.stderr, flush=True)

    thread = threading.Thread(target=_drain, daemon=True, name="worker-stderr-drainer")
    thread.start()
    return thread


def spawn_json_worker(
    cmd: list[str],
    *,
    env: dict[str, str],
    gpu: str | None = None,
) -> subprocess.Popen[str]:
    """Start worker with stdin/stdout/stderr pipes; stderr drained immediately."""
    prefix = f"[worker gpu={gpu}] " if gpu else "[worker] "
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    start_stderr_drainer(proc.stderr, prefix=prefix)
    return proc
