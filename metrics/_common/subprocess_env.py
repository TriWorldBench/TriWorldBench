"""Build subprocess environments without dropping host PATH/LD_LIBRARY_PATH."""

from __future__ import annotations

import os
from typing import Mapping


def build_subprocess_env(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    """Copy the current process env, then apply metric-specific overrides."""
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if overrides:
        env.update(overrides)
    return env


def without_progress_sidecar(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build subprocess env without progress sidecar (for GPU workers).

    Must run after ``os.environ`` merge — parent metric process inherits
    ``TRIWORLD_PROGRESS_FILE`` from the orchestrator.
    """
    merged = build_subprocess_env(env)
    merged.pop("TRIWORLD_PROGRESS_FILE", None)
    merged.pop("WA_PROGRESS_FILE", None)
    return merged
