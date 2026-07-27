"""Registry mapping vlm_consistency judge variants to scripts and prompts."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = Path(__file__).resolve().parent
PROMPT_DIR = PROJECT_ROOT / "metrics" / "vlm_prompt"

VLM_JUDGE_VARIANTS: dict[str, dict[str, str | None]] = {
    "xy": {
        "script": "vlm_judge.py",
        "prompt": None,
        "mode": "per_view",
    },
    "02": {
        "script": "vlm_judge_02.py",
        "prompt": "02.txt",
        "mode": "state_consistency",
    },
    "03": {
        "script": "vlm_judge_03.py",
        "prompt": "03.txt",
        "mode": "state_consistency",
    },
    "01": {
        "script": "vlm_judge_01.py",
        "prompt": "01.txt",
        "mode": "state_consistency_fewshot",
    },
}

DEFAULT_VLM_JUDGE = "xy"

VLM_CONSISTENCY_VARIANT_BY_METRIC: dict[str, str] = {
    "vlm_consistency01": "01",
    "vlm_consistency02": "02",
    "vlm_consistency03": "03",
}

VLM_CONSISTENCY_METRIC_NAMES: tuple[str, ...] = tuple(VLM_CONSISTENCY_VARIANT_BY_METRIC)


def resolve_vlm_judge(cfg: dict) -> str:
    variant = str(cfg.get("vlm_judge", DEFAULT_VLM_JUDGE)).strip().lower()
    if variant not in VLM_JUDGE_VARIANTS:
        raise ValueError(
            f"unknown vlm_judge={variant!r}; expected one of {sorted(VLM_JUDGE_VARIANTS)}"
        )
    return variant


def judge_script_path(variant: str) -> Path:
    spec = VLM_JUDGE_VARIANTS[variant]
    return COMMON_DIR / str(spec["script"])


def prompt_file_path(variant: str) -> Path | None:
    spec = VLM_JUDGE_VARIANTS[variant]
    prompt_name = spec.get("prompt")
    if not prompt_name:
        return None
    path = PROMPT_DIR / str(prompt_name)
    return path if path.is_file() else None


def uses_state_consistency_judge(variant: str) -> bool:
    return VLM_JUDGE_VARIANTS[variant]["mode"] != "per_view"
