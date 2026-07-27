#!/usr/bin/env python3
"""Shared helpers for Triworld benchmark evaluation."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent
LIB_DIR = SCRIPT_DIR / "lib"
TRIWORLD_ROOT = PROJECT_ROOT / "external" / "Triworld" / "video_quality"
EP_RE = re.compile(r"^episode(\d+)$", re.IGNORECASE)

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
from metric_registry import MetricSpec  # noqa: E402
from metrics._common.episode_sort import sort_episode_names  # noqa: E402
from metrics._common.metric_scores import (  # noqa: E402
    all_selected_scores_ready,
    cleanup_run_artifacts,
    discover_episode_columns,
    load_episode_scores,
    scores_path,
)
from metrics._common.subprocess_env import build_subprocess_env  # noqa: E402
from eval_data_common import VIEW_NAMES, count_frames, resolve_frame_dir  # noqa: E402
from output_naming import (  # noqa: E402
    find_latest_output_dir,
    format_output_dir,
    now_run_timestamp,
    parse_stamped_name,
    resolve_latest_pointer,
    write_latest_pointer,
)

FRAME_WEIGHT_BUCKETS: tuple[tuple[str, float], ...] = (
    ("short", 0.32),   # < 200 frames
    ("medium", 0.33),  # 200–400 frames
    ("long", 0.35),    # > 400 frames
)


@dataclass
class EvalContext:
    cfg: dict[str, Any]
    config_path: Path
    python: str
    method: str
    input_ts: str | None
    eval_input: Path
    gt_root: Path
    state_root: Path
    hyperparameters: Path
    weights_root: Path
    run_dir: Path
    temp_dir: Path
    final_dir: Path
    split_root: Path
    work: Path
    state_layout: Path
    log_dir: Path
    config_dir: Path
    prepared_dir: Path
    metric_config_dir: Path
    aggregate_dir: Path
    selected_metrics: tuple[MetricSpec, ...]
    env: dict[str, str]
    parallel_per_gpu: int
    resume: bool

    @property
    def selected_metric_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.selected_metrics)

    @property
    def selected_metric_indices(self) -> set[int]:
        return {spec.index for spec in self.selected_metrics}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(value: str | Path, root: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (root / path).resolve()
    return resolve_latest_pointer(path)


def resolve_data_path(cfg: dict[str, Any], *keys: str, default: str) -> Path:
    for key in keys:
        if cfg.get(key):
            return resolve_path(cfg[key])
    return resolve_path(default)


def count_episode_dirs(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for p in root.iterdir() if p.is_dir() and EP_RE.match(p.name))


def split_views_needs_rebuild(ctx: EvalContext) -> bool:
    """Rebuild when missing or episode count mismatches eval input (GT superset mode)."""
    gen_split = ctx.split_root / "generated_dataset"
    if not gen_split.is_dir() or not any(gen_split.iterdir()):
        return True
    input_eps = count_episode_dirs(ctx.eval_input)
    split_eps = count_episode_dirs(gen_split)
    if input_eps > 0 and split_eps != input_eps:
        return True
    return False


def infer_method_name(eval_input: Path) -> str:
    method, _input_ts, _run_ts = parse_stamped_name(eval_input.name)
    return method


def infer_input_timestamp(eval_input: Path) -> str | None:
    from output_naming import infer_input_timestamp_from_name

    return infer_input_timestamp_from_name(eval_input.name)


def resolve_run_dir(
    output_root: Path,
    method: str,
    input_ts: str | None,
    resume: bool,
) -> Path:
    """Resume picks latest run for same method+input; else create a new stamped dir."""
    if resume:
        latest = find_latest_output_dir(output_root, method, input_ts)
        if latest is not None:
            return latest

    run_ts = now_run_timestamp()
    run_dir = output_root / format_output_dir(method, input_ts, run_ts)
    output_root.mkdir(parents=True, exist_ok=True)
    write_latest_pointer(output_root, method, input_ts, run_dir.name)
    return run_dir


def write_triworld_config(cfg_path: Path, save_path: Path, work: Path, weights: Path) -> None:
    root = TRIWORLD_ROOT
    content = f"""save_path: "{save_path}"
data:
  val_base: "{work}/inputs/data/generated_dataset"
  gt_path: "{work}/inputs/data/gt_dataset"
ckpt:
  semantic_alignment:
    caption: "{weights}/qwenvl3"
    CLIP: "{weights}/clip-vit-base-patch16"
  depth_accuracy: "{weights}/depth-anything"
  aesthetic_quality:
    clip: "{weights}/clip_model/ViT-L-14.pt"
    aesthetic_head: "{weights}/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth"
  background_consistency:
    clip: "{weights}/clip_model/ViT-B-32.pt"
    raft: "{weights}/raft_model/RAFT/models/raft-things.pth"
  dynamic_degree:
    raft: "{weights}/raft_model/RAFT/models/raft-things.pth"
  flow_score:
    raft: "{weights}/raft_model/RAFT/models/raft-things.pth"
  photometric_smoothness:
    cfg: "{root}/Triworld/third_party/SEA-RAFT/config/eval/spring-M.json"
    model: "{weights}/_hf_downloads/videogenevalkit-checkpoints/worldscore/Tartan-C-T-TSKH-spring540x960-M.pth"
  motion_smoothness:
    model: "{weights}/VFIMamba/VFIMamba.pkl"
  imaging_quality:
    musiq: "{weights}/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth"
  image_quality:
    musiq: "{weights}/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth"
  subject_consistency:
    repo: "{weights}/dino_model/facebookresearch_dino_minimal"
    weight: "{weights}/dino_model/dino_vitbase16_pretrain.pth"
    model: "dino_vitb16"
    raft: "{weights}/raft_model/RAFT/models/raft-things.pth"
  sam3_model_ckpt: "{weights}/sam3"
  vlm_model: "{weights}/qwenvl3"
"""
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(content, encoding="utf-8")


def setup_env(cfg: dict[str, Any]) -> dict[str, str]:
    weights = resolve_path(cfg["weights_root"])
    pythonpath = ":".join([
        str(PROJECT_ROOT),
        str(SCRIPT_DIR),
        str(LIB_DIR),
        str(PROJECT_ROOT / "external" / "mamba_ssm_shim"),
        str(PROJECT_ROOT / "external" / "vjepa_shim"),
        str(PROJECT_ROOT / "external" / "jepa" / "src"),
        str(TRIWORLD_ROOT),
        str(TRIWORLD_ROOT / "processing" / "sam3"),
        str(TRIWORLD_ROOT / "Triworld" / "third_party"),
        os.environ.get("PYTHONPATH", ""),
    ])
    env = {
        "CUDA_VISIBLE_DEVICES": str(cfg.get("gpu_list", "0")),
        "TORCH_HOME": str(weights / "torch_home"),
        "PYTHONPATH": pythonpath,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TRIWORLD_PARALLEL_PER_GPU": str(cfg.get("parallel_per_gpu", 2)),
    }
    return env


def run_cmd(
    cmd: list[str],
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
    *,
    ui: Any | None = None,
    step_name: str | None = None,
    progress_file: Path | None = None,
) -> None:
    if ui is not None:
        from pipeline_ui import run_cmd_with_ui

        run_cmd_with_ui(
            cmd,
            env=env,
            log_path=log_path,
            ui=ui,
            step_name=step_name,
            progress_file=progress_file,
        )
        return

    print(f"[run] {' '.join(cmd)}")
    merged_env = build_subprocess_env(env)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(cmd, check=True, env=merged_env, stdout=log, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, check=True, env=merged_env)


def build_context(
    config_path: Path,
    cfg: dict[str, Any] | None = None,
    selected: list[MetricSpec] | None = None,
    *,
    run_dir: Path | None = None,
) -> EvalContext:
    cfg = cfg or load_config(config_path)
    if selected is None:
        from metric_registry import expand_dependencies, resolve_metrics

        selected = expand_dependencies(resolve_metrics(cfg.get("metrics", "all")), cfg)
    python = str(cfg.get("python", sys.executable))
    eval_input = resolve_path(cfg["eval_input"])
    method = infer_method_name(eval_input)
    input_ts = infer_input_timestamp(eval_input)
    output_root = resolve_path(cfg.get("output_dir", "output"))
    resume = bool(cfg.get("resume", True))
    if run_dir is None:
        run_dir = resolve_run_dir(output_root, method, input_ts, resume)
    else:
        run_dir = run_dir.resolve()
    temp_dir = run_dir / "temp"
    work = temp_dir / "workspace"
    env = setup_env(cfg)
    return EvalContext(
        cfg=cfg,
        config_path=config_path,
        python=python,
        method=method,
        input_ts=input_ts,
        eval_input=eval_input,
        gt_root=resolve_data_path(cfg, "gt_dataset", "gt_root", default="test_data/gt_dataset"),
        state_root=resolve_data_path(cfg, "STATE", "STATE_all500", "state_root", default="test_data/STATE"),
        hyperparameters=resolve_path(cfg["hyperparameters"]),
        weights_root=resolve_path(cfg["weights_root"]),
        run_dir=run_dir,
        temp_dir=temp_dir,
        final_dir=run_dir / "final_results",
        split_root=temp_dir / "split_views",
        work=work,
        state_layout=temp_dir / "state_layout",
        log_dir=run_dir / "logs",
        config_dir=work / "configs",
        prepared_dir=temp_dir / "prepared",
        metric_config_dir=temp_dir / "metric_configs",
        aggregate_dir=temp_dir / "aggregate",
        selected_metrics=tuple(selected),
        env=env,
        parallel_per_gpu=int(cfg.get("parallel_per_gpu", 2)),
        resume=resume,
    )


def prepared_needs_rebuild(ctx: EvalContext) -> bool:
    """Rebuild when manifest missing, stale, or head GT tree incomplete."""
    manifest_path = ctx.prepared_dir / "manifest.json"
    if not manifest_path.is_file():
        return True
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    if manifest.get("split_root") != str(ctx.split_root.resolve()):
        return True
    if "inputs_root" in manifest and "split_root" not in manifest:
        return True
    head_gt = ctx.prepared_dir / "head_data/gt_dataset/head"
    if not head_gt.is_dir():
        return True
    sample = next((p for p in head_gt.iterdir() if p.is_dir()), None)
    if sample is None or not (sample / "video").is_dir():
        return True
    return False


def metric_configs_need_rebuild(ctx: EvalContext) -> bool:
    if prepared_needs_rebuild(ctx):
        return True
    if not ctx.metric_config_dir.is_dir() or not any(ctx.metric_config_dir.glob("*.yaml")):
        return True
    for yaml_file in ctx.metric_config_dir.glob("*.yaml"):
        text = yaml_file.read_text(encoding="utf-8")
        if "workspace/inputs" in text:
            return True
    return False


def ensure_prepared_inputs(ctx: EvalContext, *, clean: bool = False) -> None:
    """Build run-local metric inputs from split_views (not workspace/inputs)."""
    if not metric_configs_need_rebuild(ctx) and not clean:
        return
    lib = LIB_DIR
    env = build_subprocess_env(ctx.env)
    if prepared_needs_rebuild(ctx) or clean:
        run_cmd([
            ctx.python, str(lib / "prepare_0710.py"), "inputs",
            "--split-root", str(ctx.split_root),
            "--output-root", str(ctx.prepared_dir),
            "--hyperparameters", str(ctx.hyperparameters),
            "--clean",
        ], env=env, log_path=ctx.log_dir / "prepare_inputs_0710.log")
    run_cmd([
        ctx.python, str(lib / "prepare_0710.py"), "configs",
        "--base-config", str(ctx.config_dir / "base.yaml"),
        "--prepared-root", str(ctx.prepared_dir),
        "--work-root", str(ctx.work),
        "--output-dir", str(ctx.metric_config_dir),
    ], env=env, log_path=ctx.log_dir / "prepare_configs_0710.log")


def pipeline_complete(ctx: EvalContext) -> bool:
    summary_csv = ctx.final_dir / "summary_metrics.csv"
    if not summary_csv.is_file():
        return False
    return all_selected_scores_ready(ctx.temp_dir, list(ctx.selected_metric_names))


def ensure_workspace(ctx: EvalContext, ui: Any | None = None) -> None:
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    ctx.log_dir.mkdir(parents=True, exist_ok=True)
    ctx.work.mkdir(parents=True, exist_ok=True)
    ctx.config_dir.mkdir(parents=True, exist_ok=True)

    if not split_views_needs_rebuild(ctx):
        if ui:
            ui.step_skip("prepare_split_views", f"exists: {ctx.split_root}")
        else:
            print(f"[skip] split_views exists: {ctx.split_root}")
    else:
        split_cmd = [
            ctx.python, str(SCRIPT_DIR / "prepare_split_views.py"),
            "--gen-root", str(ctx.eval_input),
            "--gt-root", str(ctx.gt_root),
            "--out-root", str(ctx.split_root),
            "--clean",
        ]
        run_cmd(
            split_cmd,
            env=ctx.env,
            log_path=ctx.log_dir / "prepare_split_views.log",
            ui=ui,
            step_name="prepare_split_views",
        )

    inputs_ready = ctx.work / "inputs" / "data" / "generated_dataset"
    input_eps = count_episode_dirs(inputs_ready)
    split_eps = count_episode_dirs(ctx.split_root / "generated_dataset")
    inputs_stale = input_eps > 0 and split_eps > 0 and input_eps != split_eps
    if inputs_ready.is_dir() and any(inputs_ready.iterdir()) and not inputs_stale:
        if ui:
            ui.step_skip("prepare_triworld_inputs", f"exists: {inputs_ready}")
        else:
            print(f"[skip] triworld inputs exist: {inputs_ready}")
    else:
        run_cmd([
            ctx.python, str(LIB_DIR / "prepare_triworld_inputs.py"),
            "--split-root", str(ctx.split_root),
            "--output-root", str(ctx.work / "inputs"),
            "--dataset-name", ctx.method,
            "--clean",
        ], env=ctx.env, log_path=ctx.log_dir / "prepare_triworld_inputs.log",
           ui=ui, step_name="prepare_triworld_inputs")

    base_config = ctx.config_dir / "base.yaml"
    if not base_config.is_file():
        if ui:
            ui.step_start("write_triworld_configs", kind="prep")
        write_triworld_config(base_config, ctx.work / "output", ctx.work, ctx.weights_root)
        for dim in (
            "aesthetic_quality", "background_consistency", "dynamic_degree",
            "image_quality", "subject_consistency", "flow_score",
            "photometric_smoothness", "motion_smoothness", "semantic_alignment",
            "trajectory_accuracy", "psnr_ssim",
        ):
            write_triworld_config(
                ctx.config_dir / f"{dim}.yaml",
                ctx.work / "output_metrics" / dim,
                ctx.work,
                ctx.weights_root,
            )
        if ui:
            ui.step_done("write_triworld_configs", elapsed=0.0)
    elif ui:
        ui.step_skip("write_triworld_configs", f"exists: {base_config}")

    if not metric_configs_need_rebuild(ctx):
        if ui:
            ui.step_skip("prepare_metric_inputs", f"exists: {ctx.prepared_dir}")
        else:
            print(f"[skip] prepared metric inputs exist: {ctx.prepared_dir}")
    else:
        if ui:
            ui.step_start("prepare_metric_inputs", kind="prep")
        ensure_prepared_inputs(ctx)
        if ui:
            ui.step_done("prepare_metric_inputs", elapsed=0.0)


def _parse_episode_weights_flag(cfg: dict[str, Any]) -> bool:
    raw = cfg.get("episode_weights")
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    raise ValueError(
        "episode_weights must be a boolean (false=equal average, true=frame-bucket weights)"
    )


def _frame_weight_bucket(frame_count: int) -> str:
    if frame_count < 200:
        return "short"
    if frame_count <= 400:
        return "medium"
    return "long"


def _episode_frame_count(ctx: EvalContext, episode: str) -> int:
    for root in (ctx.eval_input, ctx.split_root / "generated_dataset"):
        ep_dir = root / episode
        if not ep_dir.is_dir():
            continue
        counts = []
        for view in VIEW_NAMES:
            frame_dir = resolve_frame_dir(ep_dir, view)
            if frame_dir is not None:
                counts.append(count_frames(frame_dir))
        if counts:
            return min(counts)

    state_path = ctx.state_root / f"{episode}.json"
    if state_path.is_file():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 0
        if isinstance(data, dict):
            return int(data.get("num_frames") or 0)
    return 0


def _resolve_episode_summary_weights(
    ctx: EvalContext,
    episode_names: list[str],
) -> list[float] | None:
    """Return per-episode weights, or None for equal average."""
    if not _parse_episode_weights_flag(ctx.cfg):
        return None

    buckets: dict[str, list[str]] = {name: [] for name, _ in FRAME_WEIGHT_BUCKETS}
    for episode in episode_names:
        frame_count = _episode_frame_count(ctx, episode)
        if frame_count <= 0:
            print(
                "[warn] episode_weights=true but frame count missing; "
                "falling back to equal average"
            )
            return None
        buckets[_frame_weight_bucket(frame_count)].append(episode)

    if not all(buckets[name] for name, _ in FRAME_WEIGHT_BUCKETS):
        print(
            "[warn] episode_weights=true but not all frame buckets present "
            "(<200, 200-400, >400); falling back to equal average"
        )
        return None

    weight_by_episode: dict[str, float] = {}
    for bucket_name, bucket_total in FRAME_WEIGHT_BUCKETS:
        episodes = buckets[bucket_name]
        per_episode = bucket_total / len(episodes)
        for episode in episodes:
            weight_by_episode[episode] = per_episode

    return [weight_by_episode[episode] for episode in episode_names]


def _weighted_mean(values: dict[str, float | None], weights: list[float], episode_names: list[str]) -> float | None:
    pairs = []
    for ep, weight in zip(episode_names, weights):
        value = values.get(ep)
        if value is not None:
            pairs.append((float(value), float(weight)))
    total_w = sum(weight for _, weight in pairs)
    return sum(value * weight for value, weight in pairs) / total_w if pairs and total_w else None


def export_episode_csv(ctx: EvalContext, output_csv: Path) -> None:
    metric_names = list(ctx.selected_metric_names)
    episode_names = discover_episode_columns(ctx.temp_dir, metric_names)
    if not episode_names:
        raise RuntimeError("no episode scores found for selected metrics")

    rows = []
    for metric in metric_names:
        data = load_episode_scores(scores_path(ctx.temp_dir, metric))
        episodes = data.get("episodes") if isinstance(data.get("episodes"), dict) else {}
        row: dict[str, Any] = {"metric": metric}
        for ep in episode_names:
            row[ep] = episodes.get(ep)
        rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", *episode_names])
        writer.writeheader()
        writer.writerows(rows)


def export_summary_csv(ctx: EvalContext, output_csv: Path) -> None:
    metric_names = list(ctx.selected_metric_names)
    episode_names = discover_episode_columns(ctx.temp_dir, metric_names)
    if not episode_names:
        raise RuntimeError("no episode scores found for selected metrics")

    weights = _resolve_episode_summary_weights(ctx, episode_names)

    row: dict[str, Any] = {
        "run_dir": str(ctx.run_dir),
        "dataset": ctx.method,
        "run_id": ctx.run_dir.name,
        "protocol_version": "triworld",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "episode_count": len(episode_names),
    }

    for metric in metric_names:
        data = load_episode_scores(scores_path(ctx.temp_dir, metric))
        episodes = data.get("episodes") if isinstance(data.get("episodes"), dict) else {}
        if weights:
            row[metric] = _weighted_mean(episodes, weights, episode_names)
        else:
            values = [float(v) for v in episodes.values() if v is not None]
            row[metric] = sum(values) / len(values) if values else None

    summary_values = [row[metric] for metric in metric_names if row.get(metric) is not None]
    row["TWB-Score"] = (
        sum(summary_values) / len(summary_values) if summary_values else None
    )

    columns = [
        "run_dir", "dataset", "run_id", "protocol_version", "generated_at", "episode_count",
        "TWB-Score",
        *metric_names,
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)


def finalize_results(ctx: EvalContext) -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from metrics._common.finalize import finalize_all

    finalize_all(ctx)


def export_csv(ctx: EvalContext, *, cleanup: bool = True) -> None:
    ctx.final_dir.mkdir(parents=True, exist_ok=True)
    export_episode_csv(ctx, ctx.final_dir / "episode_metrics.csv")
    export_summary_csv(ctx, ctx.final_dir / "summary_metrics.csv")
    if cleanup:
        cleanup_run_artifacts(ctx, aggressive=True)
