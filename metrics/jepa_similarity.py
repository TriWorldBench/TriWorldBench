"""Run JEDi JEPA similarity and write per-episode scores."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from metrics._common.episode_sort import episode_sort_key
from metrics._common.metric_scores import cleanup_run_artifacts, save_episode_scores, scores_path
from metrics._common.subprocess_env import build_subprocess_env

JEDI_ROOT = Path(__file__).resolve().parents[1] / "external" / "Triworld" / "video_quality" / "JEDi"


def _episode_scores(data_base: Path, score: float) -> dict[str, float]:
    rows: dict[str, float] = {}
    if not data_base.is_dir():
        return rows
    for view_dir in sorted(data_base.iterdir()):
        if not view_dir.is_dir():
            continue
        for episode_dir in sorted(view_dir.iterdir(), key=lambda p: episode_sort_key(p.name)):
            if episode_dir.is_dir():
                rows[episode_dir.name] = score
    return rows


def run_jepa(ctx: Any, metric_index: int, metric_name: str) -> None:
    score_file = scores_path(ctx.temp_dir, metric_name)
    if score_file.is_file() and ctx.resume:
        print(f"[skip] {metric_name}")
        return

    output_root = ctx.work / "output_JEDi"
    output_root.mkdir(parents=True, exist_ok=True)
    batch_script = JEDI_ROOT / "batch.py"
    config_path = JEDI_ROOT / "configs" / "vith16_ssv2_16x2x3.yaml"
    results_json = output_root / "results.json"

    env = build_subprocess_env(ctx.env)
    port = os.environ.get("MASTER_PORT", "12355")
    env.setdefault("WA_JEPA_MASTER_PORT", port)
    env.setdefault("MASTER_PORT", port)

    jepa_cmd = [
        ctx.python, str(batch_script),
        "--real_dir", str(ctx.work / "inputs" / "flat" / "gt"),
        "--gen_dir", str(ctx.work / "inputs" / "flat" / "generated"),
        "--model_dir", str(ctx.weights_root / "vjepa"),
        "--config_path", str(config_path),
        "--output_root", str(output_root),
        "--save_intersection", str(output_root / "intersection_names.json"),
        "--batch_size", "4",
        "--num_workers", "4",
    ]
    subprocess.run(jepa_cmd, check=True, env=env)

    payload = json.loads(results_json.read_text(encoding="utf-8"))
    score = float(payload["score"])
    data_base = ctx.work / "inputs" / "data" / "generated_dataset"
    episode_scores = _episode_scores(data_base, score)
    save_episode_scores(ctx.temp_dir, metric_name, metric_index, episode_scores, extra={"global_score": score})

    marker = ctx.work / "output_metrics" / "jepa_similarity" / "generated_results.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"video_path": str(data_base / view / ep / "1" / "video"), "video_results_normalized": val}
        for ep, val in episode_scores.items()
        for view in ("head", "left", "right")
        if (data_base / view / ep).is_dir()
    ]
    marker.write_text(
        json.dumps({"jepa_similarity": [score, rows]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    cleanup_run_artifacts(ctx)
    print(f"[done] {metric_name} -> {score_file}")
