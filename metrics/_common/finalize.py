"""Merge metric outputs and build 0710 protocol results."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from metrics._common.subprocess_env import build_subprocess_env


def merge_metric_outputs(metric_root: Path, output: Path) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for result_path in sorted(metric_root.glob("*/generated_results.json")):
        data = json.loads(result_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            merged.update(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged


def finalize_all(ctx: Any) -> None:
    from metrics._common.protocol_0710 import build_protocol_results

    project_root = Path(__file__).resolve().parents[2]
    merge_metric_outputs(ctx.work / "output_metrics", ctx.work / "output" / "generated_results.json")
    ctx.aggregate_dir.mkdir(parents=True, exist_ok=True)

    split_manifest = ctx.split_root / "manifest.json"
    state_cmd = [
        ctx.python,
        str(project_root / "scripts" / "prepare_state_layout.py"),
        "--state-root", str(ctx.state_root),
        "--output-root", str(ctx.state_layout),
        "--clean",
    ]
    if split_manifest.is_file():
        state_cmd.extend(["--episodes-file", str(split_manifest)])

    if not ctx.state_layout.is_dir() or not any(ctx.state_layout.iterdir()):
        subprocess.run(state_cmd, check=True, env=build_subprocess_env(ctx.env))
    elif split_manifest.is_file():
        subprocess.run(state_cmd, check=True, env=build_subprocess_env(ctx.env))

    build_protocol_results(ctx)
