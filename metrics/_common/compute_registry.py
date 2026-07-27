"""Lazy registry: dimension -> compute module in metrics/*.py."""

from __future__ import annotations

import importlib
from typing import Any, Callable

_REGISTRY: dict[str, tuple[str, str]] = {
    "aesthetic_quality": ("aesthetic_quality", "compute_aesthetic_quality"),
    "background_consistency": ("background_consistency", "compute_background_consistency"),
    "dynamic_degree": ("dynamic_degree", "compute_dynamic_degree"),
    "flow_score": ("flow_score", "compute_flow_score"),
    "image_quality": ("image_quality", "compute_imaging_quality"),
    "imaging_quality": ("image_quality", "compute_imaging_quality"),
    "photometric_smoothness": ("photometric_smoothness", "compute_photometric_smoothness"),
    "semantic_alignment": ("semantic_alignment", "compute_semantic_alignment"),
    "subject_consistency": ("subject_consistency", "compute_subject_consistency"),
    "trajectory_accuracy": ("trajectory_accuracy", "compute_trajectory_accuracy"),
}


def get_compute_fn(dimension: str) -> Callable[..., Any]:
    if dimension not in _REGISTRY:
        raise KeyError(f"no compute registry entry for dimension: {dimension}")
    module_name, attr = _REGISTRY[dimension]
    mod = importlib.import_module(f"metrics.{module_name}")
    fn = getattr(mod, attr)
    if not callable(fn):
        raise TypeError(f"metrics.{module_name}.{attr} is not callable")
    return fn


def get_module(dimension: str):
    """Return compute module for patching (subject/background/trajectory)."""
    if dimension not in _REGISTRY:
        raise KeyError(f"no compute registry entry for dimension: {dimension}")
    module_name, _ = _REGISTRY[dimension]
    return importlib.import_module(f"metrics.{module_name}")


def dimension_kwargs(config: dict[str, Any], dimension: str) -> dict[str, Any]:
    """Build checkpoint kwargs from metric YAML config."""
    kwargs: dict[str, Any] = {}
    ckpt = config.get("ckpt", {})
    if dimension == "semantic_alignment":
        item = ckpt.get(dimension, {})
        kwargs[f"{dimension}_caption_model_ckpt"] = item.get("caption")
        kwargs[f"{dimension}_clip_model_ckpt"] = item.get("CLIP")
        kwargs[f"{dimension}_bleus_model_ckpt"] = config.get(dimension, {}).get("BLEUs")
    elif dimension == "aesthetic_quality":
        item = ckpt.get(dimension, {})
        kwargs[f"{dimension}_clip_ckpt"] = item.get("clip")
        kwargs[f"{dimension}_head_ckpt"] = item.get("aesthetic_head")
    elif dimension == "background_consistency":
        item = ckpt.get(dimension, {})
        kwargs[f"{dimension}_clip_ckpt"] = item.get("clip")
        kwargs[f"{dimension}_raft_ckpt"] = item.get("raft")
    elif dimension in ("dynamic_degree", "flow_score"):
        item = ckpt.get(dimension, {})
        kwargs[f"{dimension}_raft_ckpt"] = item.get("raft")
    elif dimension == "photometric_smoothness":
        item = ckpt.get(dimension, {})
        kwargs[f"{dimension}_cfg_ckpt"] = item.get("cfg")
        kwargs[f"{dimension}_model_ckpt"] = item.get("model")
    elif dimension == "motion_smoothness":
        item = ckpt.get(dimension, {})
        kwargs[f"{dimension}_model_ckpt"] = item.get("model")
    elif dimension == "subject_consistency":
        item = ckpt.get(dimension, {})
        kwargs[f"{dimension}_repo_ckpt"] = item.get("repo")
        kwargs[f"{dimension}_weight_ckpt"] = item.get("weight")
        kwargs[f"{dimension}_model_name"] = item.get("model", "dino_vitb16")
        kwargs[f"{dimension}_raft_ckpt"] = item.get("raft")
    elif dimension in ("imaging_quality", "image_quality"):
        key = dimension if dimension in ckpt else "imaging_quality"
        kwargs[f"{dimension}_musiq_ckpt"] = ckpt.get(key, {}).get("musiq")
    elif dimension not in ("psnr", "ssim", "psnr_ssim"):
        kwargs[f"{dimension}_model_ckpt"] = ckpt.get(dimension)
    return kwargs
