#!/usr/bin/env python3
"""Preflight import check for Triworld eval entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "lib"))

# eval_common must expose symbols used before any metric runs.
from eval_common import (  # noqa: E402
    build_context,
    build_subprocess_env,
    ensure_prepared_inputs,
    ensure_workspace,
    load_config,
)

# Worker/subprocess stack touched by recent refactors.
from metrics._common.gpu_pool import read_worker_json_line, run_persistent_worker  # noqa: E402
from metrics._common.parallel_runner import run_episode_parallel  # noqa: E402
from metrics._common.subprocess_worker import spawn_json_worker, start_stderr_drainer  # noqa: E402

__all__ = [
    "build_context",
    "build_subprocess_env",
    "ensure_prepared_inputs",
    "ensure_workspace",
    "load_config",
    "read_worker_json_line",
    "run_episode_parallel",
    "run_persistent_worker",
    "spawn_json_worker",
    "start_stderr_drainer",
]


def main() -> int:
    print("[smoke] imports ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
