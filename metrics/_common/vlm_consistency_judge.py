#!/usr/bin/env python3
"""Benchmark-compatible runner for state-based vlm_consistency judges (01/02/03)."""

from __future__ import annotations

import argparse
import importlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from metrics._common.multiview_eval_common import (
    ARM_TO_VIEW,
    VIEW_DIRS,
    load_phase_samples_from_state_root,
    load_reference_bundle,
    load_triplet_images,
    raw_response_filename,
    sample_key,
)
from metrics._common.vlm_judge_registry import VLM_JUDGE_VARIANTS

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(k for k in VLM_JUDGE_VARIANTS if k != "xy"), required=True)
    parser.add_argument("--samples-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output-dir", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--reference-db-root", type=Path, default=None)
    parser.add_argument("--use-reference-db", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def load_samples(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"samples must be a list: {path}")
    return [item for item in data if isinstance(item, dict)]


def load_prompt_templates(prompt_file: Path | None, variant: str, module: Any) -> dict[str, str | None]:
    if prompt_file is None or not prompt_file.is_file():
        return {"left": None, "right": None}

    text = prompt_file.read_text(encoding="utf-8")
    if "VLM_PROMPT_TEMPLATE" in text:
        namespace: dict[str, Any] = {}
        exec(text, namespace)  # noqa: S102
        if variant == "02":
            return {
                "left": namespace.get("VLM_PROMPT_TEMPLATE_LEFT_V3"),
                "right": namespace.get("VLM_PROMPT_TEMPLATE_RIGHT_V3"),
            }
        return {
            "left": namespace.get("VLM_PROMPT_TEMPLATE_LEFT"),
            "right": namespace.get("VLM_PROMPT_TEMPLATE_RIGHT") or text.strip(),
        }

    if variant == "03":
        return {
            "left": getattr(module, "VLM_PROMPT_TEMPLATE_LEFT", None),
            "right": text.strip(),
        }
    return {"left": None, "right": None}


def build_dual_view_prompt(
    sample: dict[str, Any],
    templates: dict[str, str | None],
    module: Any,
) -> str:
    arm = sample.get("arm", "")
    if arm == "left":
        template = templates.get("left") or getattr(module, "VLM_PROMPT_TEMPLATE_LEFT", None)
    else:
        template = templates.get("right") or getattr(module, "VLM_PROMPT_TEMPLATE_RIGHT", None)
    if not template:
        return module.build_vlm_prompt(sample)
    return template.format(
        instruction=sample.get("instruction") or sample.get("task", ""),
        phase=sample["phase"],
        description=sample["description"],
        arm=arm,
    )


def normalize_overall_score(parsed: dict[str, Any], variant: str) -> float | None:
    if "overall_consistency_score" in parsed:
        raw = float(parsed["overall_consistency_score"])
        if variant == "01":
            from metrics._common import vlm_judge_01 as judge_01_module

            component = judge_01_module.normalize_01_component_score(raw)
            if component is None:
                return None
            return round(max(0.0, min(1.0, component / 5.0)), 6)
        return max(0.0, min(1.0, raw / 5.0))

    state = parsed.get("state_consistency_score")
    obj = parsed.get("object_consistency_score")
    if state is None or obj is None:
        return None

    if variant == "01":
        from metrics._common import vlm_judge_01 as judge_01_module

        return judge_01_module.normalize_01_overall_score(state, obj)

    raw = (float(state) + float(obj)) / 2.0
    if float(state) > 5 or float(obj) > 5:
        return max(0.0, min(1.0, raw / 100.0))
    return max(0.0, min(1.0, raw / 5.0))


def evaluate_dual_view_sample(
    sample: dict[str, Any],
    module: Any,
    model: Any,
    processor: Any,
    frames_root: Path,
    dataset_root: Path | None,
    templates: dict[str, str | None],
    max_new_tokens: int,
    variant: str,
) -> dict[str, Any]:
    images, image_sources = load_triplet_images(sample, frames_root, dataset_root)
    views = module.vlm_view_order(sample)
    prompt = build_dual_view_prompt(sample, templates, module)
    vlm_images = [images[view] for view in views]
    raw_text = module.run_qwen(model, processor, prompt, vlm_images, max_new_tokens)
    parsed = module.parse_vlm_result(raw_text)
    return {
        "image_sources": image_sources,
        "vlm_views": list(views),
        "vlm_raw": raw_text,
        "vlm_parsed": parsed,
        "score_normalized": normalize_overall_score(parsed, variant) if parsed else None,
    }


def evaluate_01_sample(
    sample: dict[str, Any],
    module: Any,
    model: Any,
    processor: Any,
    frames_root: Path,
    dataset_root: Path | None,
    reference_db_root: Path | None,
    use_reference_db: bool,
    max_new_tokens: int,
    prompt_file: Path | None = None,
) -> dict[str, Any]:
    images, image_sources = load_triplet_images(sample, frames_root, dataset_root)
    prompt_config = module.load_prompt_config(prompt_file) if hasattr(module, "load_prompt_config") else None
    load_ref = getattr(module, "load_reference_bundle", load_reference_bundle)
    resolved_db = None
    if use_reference_db:
        if hasattr(module, "resolve_reference_db_root"):
            resolved_db = module.resolve_reference_db_root(reference_db_root, prompt_config)
        elif reference_db_root is not None and reference_db_root.is_dir():
            resolved_db = reference_db_root.resolve()

    reference_bundle = None
    if use_reference_db and resolved_db is not None and resolved_db.is_dir():
        reference_bundle = load_ref(resolved_db, sample)

    target_image = module.combine_views(images)
    vlm_content = module.build_vlm_content(sample, target_image, reference_bundle, prompt_config)

    raw_text = module.run_qwen(model, processor, vlm_content, max_new_tokens)
    parsed = module.parse_vlm_result(raw_text)
    return {
        "image_sources": image_sources,
        "vlm_reference": {
            "enabled": reference_bundle is not None,
            "entry_dir": reference_bundle.get("entry_dir") if reference_bundle else None,
            "num_negatives": len(reference_bundle["negatives"]) if reference_bundle else 0,
        },
        "vlm_raw": raw_text,
        "vlm_parsed": parsed,
        "score_normalized": normalize_overall_score(parsed, "01") if parsed else None,
    }


def main() -> None:
    args = parse_args()
    module = importlib.import_module(f"metrics._common.vlm_judge_{args.variant}")

    base_config = load_yaml(args.config_path)
    model_path = args.model_path or Path(base_config.get("ckpt", {}).get("vlm_model", ""))
    if not str(model_path):
        raise ValueError("VLM model path is missing")

    samples = load_samples(args.samples_json)
    existing: dict[str, dict[str, Any]] = {}
    if args.output.is_file() and not args.overwrite:
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        if isinstance(prior, list):
            existing = {
                str(item.get("sample_id")): item
                for item in prior
                if isinstance(item, dict) and item.get("sample_id")
            }

    templates = load_prompt_templates(args.prompt_file, args.variant, module)
    dataset_root = args.dataset_root.resolve() if args.dataset_root else None
    reference_db_root = args.reference_db_root.resolve() if args.reference_db_root else None

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path), torch_dtype="auto", device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(str(model_path))
    args.raw_output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for sample in tqdm(samples, desc=f"vlm-consistency-{args.variant}", unit="sample"):
        sid = sample.get("sample_id") or sample_key(sample)
        if sid in existing:
            results.append(existing[sid])
            continue

        item: dict[str, Any] = {
            "sample_id": sid,
            "episode": sample.get("episode"),
            **sample,
        }
        try:
            if args.variant == "01":
                payload = evaluate_01_sample(
                    sample,
                    module,
                    model,
                    processor,
                    args.frames_root.resolve(),
                    dataset_root,
                    reference_db_root,
                    args.use_reference_db,
                    args.max_new_tokens,
                    args.prompt_file,
                )
            else:
                payload = evaluate_dual_view_sample(
                    sample,
                    module,
                    model,
                    processor,
                    args.frames_root.resolve(),
                    dataset_root,
                    templates,
                    args.max_new_tokens,
                    args.variant,
                )
            item.update(payload)
            if item.get("vlm_parsed") is None:
                item["error"] = "failed_to_parse_vlm_json"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                raw_path = args.raw_output_dir / raw_response_filename(sid, timestamp)
                raw_path.write_text(
                    json.dumps({"raw_response": item.get("vlm_raw")}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                item["raw_response_file"] = str(raw_path)
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"

        results.append(item)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
