"""Run VQA metric via metrics/VQA/run_parallel_eval.sh."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from metrics._common.metric_scores import cleanup_run_artifacts, load_episode_scores, save_episode_scores, scores_path
from metrics._common.subprocess_env import build_subprocess_env

VQA_ROOT = Path(__file__).resolve().parent / "VQA"
QA_ROOT = VQA_ROOT / "qa_val"


def _load_vqa_num_frames(ctx: Any) -> int:
    with ctx.hyperparameters.open("r", encoding="utf-8") as f:
        params = yaml.safe_load(f) or {}
    return int(params.get("vqa", {}).get("num_frames", 4))


def _episode_score(episode_result: dict[str, Any]) -> float | None:
    if episode_result.get("status") != "ok":
        return None
    score_ratio = episode_result.get("score_ratio")
    if score_ratio is None:
        return None
    ratio = float(score_ratio)
    view_eval = episode_result.get("view_evaluation") or episode_result.get("camera_view_evaluation")
    view_score = view_eval.get("score") if isinstance(view_eval, dict) else None
    if view_score == 1:
        return ratio
    return ratio / 2.0


def _extract_episode_scores(report: dict[str, Any]) -> dict[str, float]:
    episodes = report.get("episodes")
    if not isinstance(episodes, list):
        return {}
    scores: dict[str, float] = {}
    for item in episodes:
        if not isinstance(item, dict):
            continue
        episode = str(item.get("episode") or "")
        if not episode:
            continue
        score = _episode_score(item)
        if score is not None:
            scores[episode] = score
    return scores


def _vqa_parallel_shard_count(gpu_list: str) -> int:
    return len([part.strip() for part in gpu_list.split(",") if part.strip()])


def _vqa_run_resumable(output_root: Path, gpu_list: str) -> bool:
    shard_root = output_root / "shards"
    num_shards = _vqa_parallel_shard_count(gpu_list)
    if num_shards < 1 or not shard_root.is_dir():
        return False
    for shard_id in range(num_shards):
        if not (shard_root / f"shard_{shard_id}").is_dir():
            return False
    return True


def _vqa_model_path(ctx: Any) -> Path:
    return Path(ctx.weights_root) / "qwenvl3"


def _vqa_no_scores_message(report: dict[str, Any]) -> str:
    episodes = report.get("episodes")
    if not isinstance(episodes, list):
        return "VQA report has no episode results"
    details: list[str] = []
    for item in episodes:
        if not isinstance(item, dict):
            continue
        if item.get("status") == "ok":
            continue
        name = str(item.get("episode") or "?")
        err = item.get("error") or item.get("status") or "unknown"
        details.append(f"{name}: {err}")
    if details:
        return "VQA produced no scored episodes:\n  - " + "\n  - ".join(details)
    return "VQA produced no scored episodes"


def run_vqa(ctx: Any, metric_index: int, metric_name: str) -> None:
    score_file = scores_path(ctx.temp_dir, metric_name)
    if score_file.is_file() and ctx.resume:
        prior = load_episode_scores(score_file)
        prior_eps = prior.get("episodes") if isinstance(prior.get("episodes"), dict) else {}
        if prior_eps:
            print(f"[skip] {metric_name}")
            return

    metric_dir = score_file.parent
    metric_dir.mkdir(parents=True, exist_ok=True)
    output_root = metric_dir / "run"
    num_frames = _load_vqa_num_frames(ctx)
    gpu_list = str(ctx.cfg.get("gpu_list", "0"))
    model_path = _vqa_model_path(ctx)
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"VQA model not found: {model_path} (set config weights_root and install qwenvl3; see DOWNLOAD_LINKS.md)"
        )

    cmd = [
        "bash",
        str(VQA_ROOT / "run_parallel_eval.sh"),
        "--input-dir",
        str(ctx.eval_input),
        "--qa-root",
        str(QA_ROOT),
        "--model-path",
        str(model_path),
        "--gpus",
        gpu_list,
        "--num-frames",
        str(num_frames),
    ]
    use_resume = output_root.is_dir() and ctx.resume and _vqa_run_resumable(output_root, gpu_list)
    if output_root.is_dir() and ctx.resume and not use_resume:
        print(f"[vqa] dropping incomplete run dir (cannot resume): {output_root}")
        shutil.rmtree(output_root)
    if use_resume:
        cmd.extend(["--resume", str(output_root)])
    else:
        cmd.extend(["--output-root", str(output_root)])

    env = build_subprocess_env(ctx.env)
    env["PYTHON"] = ctx.python
    env["VLM_EVAL_MODEL_PATH"] = str(model_path)

    subprocess.run(cmd, check=True, cwd=str(VQA_ROOT), env=env)

    report_path = output_root / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"VQA report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    episode_scores = _extract_episode_scores(report)
    if not episode_scores:
        raise RuntimeError(_vqa_no_scores_message(report))
    save_episode_scores(ctx.temp_dir, metric_name, metric_index, episode_scores)
    cleanup_run_artifacts(ctx)
    print(f"[done] {metric_name} -> {score_file}")
