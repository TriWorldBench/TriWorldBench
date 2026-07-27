"""Metric registry for Triworld benchmark evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from metrics._common.vlm_judge_registry import VLM_CONSISTENCY_METRIC_NAMES

MetricPhase = Literal["base", "triworld_0710", "derived"]

TRAJECTORY_ACCURACY_INDEX = 12
VLM_CONSISTENCY_INDICES: tuple[int, ...] = (13, 14, 15)
QUALITY_PENALTY_METRICS = frozenset({"aesthetic_quality", "image_quality"})


@dataclass(frozen=True)
class MetricSpec:
    index: int
    name: str
    phase: MetricPhase
    base_key: str | None = None  # underlying Triworld metric key for derived metrics


METRIC_REGISTRY: tuple[MetricSpec, ...] = (
    MetricSpec(0, "Instruction_Following", "derived", "vlm_judge"),
    MetricSpec(1, "Interaction_Quality", "derived", "vlm_judge"),
    MetricSpec(2, "Perspectivity", "derived", "vlm_judge"),
    MetricSpec(3, "background_consistency", "base", "background_consistency"),
    MetricSpec(4, "dynamic_state_alignment", "derived", "dynamic_degree"),
    MetricSpec(5, "flow_score", "base", "flow_score"),
    MetricSpec(6, "jepa_similarity", "base", "jepa_similarity"),
    MetricSpec(7, "photometric_smoothness", "base", "photometric_smoothness"),
    MetricSpec(8, "psnr", "base", "psnr"),
    MetricSpec(9, "semantic_alignment", "base", "semantic_alignment"),
    MetricSpec(10, "ssim", "base", "ssim"),
    MetricSpec(11, "subject_consistency", "base", "subject_consistency"),
    MetricSpec(12, "trajectory_accuracy", "base", "trajectory_accuracy"),
    MetricSpec(13, "vlm_consistency01", "derived", "vlm_consistency01"),
    MetricSpec(14, "vlm_consistency02", "derived", "vlm_consistency02"),
    MetricSpec(15, "vlm_consistency03", "derived", "vlm_consistency03"),
    MetricSpec(16, "VQA", "base", "VQA"),
    MetricSpec(17, "aesthetic_quality", "base", "aesthetic_quality"),
    MetricSpec(18, "image_quality", "base", "image_quality"),
)

METRIC_BY_INDEX = {m.index: m for m in METRIC_REGISTRY}
METRIC_BY_NAME = {m.name: m for m in METRIC_REGISTRY}

SUMMARY_COLUMNS = tuple(m.name for m in METRIC_REGISTRY)


def resolve_metrics(selection: list[int] | str) -> list[MetricSpec]:
    if selection == "all" or selection == ["all"]:
        return list(METRIC_REGISTRY)
    indices = [int(i) for i in selection]
    return [METRIC_BY_INDEX[i] for i in sorted(indices) if i in METRIC_BY_INDEX]


def expand_dependencies(selection: list[MetricSpec], cfg: dict | None = None) -> list[MetricSpec]:
    """Ensure base metrics run before derived metrics that depend on them."""
    _ = cfg
    indices = {spec.index for spec in selection}
    extras: list[int] = []

    for spec in selection:
        if spec.name in QUALITY_PENALTY_METRICS:
            if TRAJECTORY_ACCURACY_INDEX not in indices and TRAJECTORY_ACCURACY_INDEX not in extras:
                extras.append(TRAJECTORY_ACCURACY_INDEX)
            for vlm_idx in VLM_CONSISTENCY_INDICES:
                if vlm_idx not in indices and vlm_idx not in extras:
                    extras.append(vlm_idx)
        if spec.phase != "derived" or not spec.base_key:
            continue
        if spec.base_key == "vlm_judge":
            if spec.index in (0, 1, 2) and 0 not in indices and 0 not in extras:
                extras.append(0)
            continue
        if spec.base_key == "dynamic_degree" and spec.index == 4:
            continue
        base = next(
            (m.index for m in METRIC_REGISTRY if m.phase == "base" and m.base_key == spec.base_key),
            None,
        )
        if base is not None and base not in indices and base not in extras:
            extras.append(base)

    ordered = sorted(indices | set(extras))
    return [METRIC_BY_INDEX[i] for i in ordered]
