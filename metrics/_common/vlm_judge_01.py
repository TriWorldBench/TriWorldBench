#!/usr/bin/env python3
"""VLM multi-view consistency evaluation driven by STATE phase JSON files.

For each phase that is not idle/approach, load head / left-wrist / right-wrist
frames at sampled_frame, query a VLM for:
  - state consistency (action matches phase description)
  - object consistency (manipulated object color/shape across views)

Also runs a placeholder math check from HDF5 gripper state (formula TBD by user).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from metrics._common.multiview_eval_common import (
    ARM_TO_VIEW,
    VIEW_DIRS,
    collect_phase_samples,
    combine_views,
    discover_state_files,
    load_reference_bundle as load_legacy_reference_bundle,
    load_triplet_images,
    negative_mix_caption,
    reference_entry_dir,
    sample_key,
    sample_sort_key,
)
from metrics._common.vlm_epipolar import compute_math_check
PROJECT_ROOT = Path(os.environ.get("MATRIC_EVAL_ROOT", Path(__file__).resolve().parents[2]))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "test_data"))
WEIGHTS_ROOT = Path(os.environ.get("WEIGHTS_ROOT", PROJECT_ROOT / "weights"))
OUTPUTS_ROOT = Path(os.environ.get("OUTPUTS_ROOT", PROJECT_ROOT / "output"))
DEFAULT_PROMPT_FILE = PROJECT_ROOT / "metrics" / "vlm_prompt" / "01.txt"
DEFAULT_SHARED_REFERENCE_DB = PROJECT_ROOT / "test_data" / "vlm_data_base"
DEFAULT_MODEL_PATH = WEIGHTS_ROOT / "qwenvl3"
ARM_TO_CAMERA = {"left": "left_camera", "right": "right_camera", "head": "head_camera"}
EPIPOLAR_ERROR_THRESHOLD_PX = 5.0
EPIPOLAR_MIN_MATCHES = 5
VLM_PROMPT_VERSION = "01_shared_kb_v1"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def _sorted_images(folder: Path, pattern: str = "*") -> list[Path]:
    if not folder.is_dir():
        return []
    paths = [path for path in folder.glob(pattern) if path.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(paths, key=lambda path: path.name)


def _is_shared_reference_db(db_root: Path) -> bool:
    return (db_root / "positive").is_dir() or (db_root / "negative").is_dir()


def load_prompt_config(prompt_file: Path | None = None) -> dict[str, Any]:
    """Load few-shot prompt sections from metrics/vlm_prompt/01.txt."""
    path = (prompt_file or DEFAULT_PROMPT_FILE).resolve()
    defaults = _builtin_prompt_defaults()
    if not path.is_file():
        return defaults

    namespace: dict[str, Any] = {}
    exec(path.read_text(encoding="utf-8"), namespace)  # noqa: S102
    return {
        "prompt_file": str(path),
        "intro_template": str(namespace.get("VLM_INTRO_TEMPLATE") or defaults["intro_template"]),
        "positive_library_intro": str(
            namespace.get("VLM_POSITIVE_LIBRARY_INTRO") or defaults["positive_library_intro"]
        ),
        "negative_library_intro": str(
            namespace.get("VLM_NEGATIVE_LIBRARY_INTRO") or defaults["negative_library_intro"]
        ),
        "eval_template": str(namespace.get("VLM_EVAL_TEMPLATE") or defaults["eval_template"]),
        "legacy_template": str(
            namespace.get("VLM_PROMPT_TEMPLATE_LEGACY") or defaults["legacy_template"]
        ),
        "captions": dict(namespace.get("REFERENCE_CAPTIONS") or {}),
        "default_reference_db": namespace.get("DEFAULT_REFERENCE_DB"),
    }


def _builtin_prompt_defaults() -> dict[str, Any]:
    return {
        "prompt_file": str(DEFAULT_PROMPT_FILE),
        "intro_template": (
            "你是机器人三视角一致性评测专家。\n\n"
            "每张图均为横向拼接的三视角图像，从左到右依次为 head-view、left-wrist-view、right-wrist-view。\n\n"
            "## 参考样本库\n\n{positive_library_intro}\n\n{negative_library_intro}"
        ),
        "positive_library_intro": "**正样本库**：画质清晰、三视角一致且任务执行正确。",
        "negative_library_intro": "**负样本库**：包含画质模糊、视角不一致、任务执行错误等低分示例。",
        "eval_template": (
            "## 待评估样本（图{target_figure}）\n\n"
            "任务指令：{instruction}\n"
            "当前阶段类型：{phase}\n"
            "当前阶段描述：{description}\n"
            "标注执行手臂：{arm}\n\n"
            "评分：0-100 整数。\n"
            '只输出 JSON：{{"state_consistency_score": int, "state_consistency_reason": str, '
            '"object_consistency_score": int, "object_consistency_reason": str}}'
        ),
        "legacy_template": (
            "任务总指令：{instruction}\n"
            "当前阶段类型：{phase}\n"
            "当前阶段描述：{description}\n"
            "标注执行手臂：{arm}\n"
            "评分 0-100，只输出 JSON。"
        ),
        "captions": {},
        "default_reference_db": str(DEFAULT_SHARED_REFERENCE_DB.relative_to(PROJECT_ROOT)),
    }


def normalize_01_component_score(value: Any) -> float | None:
    """Map VLM score to 0-5; accept legacy 0-100 responses."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score > 5.0:
        score = score / 100.0 * 5.0
    return round(max(0.0, min(5.0, score)), 4)


def normalize_01_overall_score(state: Any, obj: Any) -> float | None:
    state_score = normalize_01_component_score(state)
    obj_score = normalize_01_component_score(obj)
    if state_score is None or obj_score is None:
        return None
    return round(max(0.0, min(1.0, (state_score + obj_score) / 10.0)), 6)


def resolve_reference_db_root(db_root: Path | None, prompt_config: dict[str, Any] | None = None) -> Path:
    if db_root is not None:
        resolved = db_root.resolve()
        if resolved.is_dir():
            return resolved
    rel = (prompt_config or {}).get("default_reference_db")
    if rel:
        path = Path(str(rel))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()
    return DEFAULT_SHARED_REFERENCE_DB.resolve()


def load_reference_bundle(
    db_root: Path,
    sample: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Load shared few-shot images from vlm_data_base; fallback to legacy per-task DB."""
    db_root = db_root.resolve()
    if _is_shared_reference_db(db_root):
        positive_paths = _sorted_images(db_root / "positive")
        negative_paths = _sorted_images(db_root / "negative", "negative_*")
        if not positive_paths and not negative_paths:
            return None

        meta_path = db_root / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        positives = [Image.open(path).convert("RGB") for path in positive_paths]
        negatives = [Image.open(path).convert("RGB") for path in negative_paths]
        return {
            "entry_dir": str(db_root),
            "mode": "shared",
            "positive": positives[0] if positives else None,
            "positives": positives,
            "positive_paths": [str(path) for path in positive_paths],
            "negatives": negatives,
            "negative_paths": [str(path) for path in negative_paths],
            "meta": meta,
        }

    if sample is not None and reference_entry_dir(db_root, sample).is_dir():
        return load_legacy_reference_bundle(db_root, sample)
    return None


def _reference_caption(
    stem: str,
    prompt_config: dict[str, Any],
    bundle_meta: dict[str, Any],
    fallback: str,
) -> str:
    captions = prompt_config.get("captions") or {}
    if stem in captions:
        return str(captions[stem])
    meta_caps = bundle_meta.get("captions") if isinstance(bundle_meta.get("captions"), dict) else {}
    if stem in meta_caps:
        return str(meta_caps[stem])
    return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STATE-driven tri-view VLM consistency evaluation.")
    parser.add_argument(
        "--state-root",
        type=Path,
        default=DATA_ROOT / "STATE",
    )
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=DATA_ROOT / "gt_dataset",
        help="Frames root: <frames-root>/<task>/<episode>/{head,left,right}/{frames|video}/",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "input",
        help="RobotWin dataset root (HDF5 + preview mp4 fallback).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUTS_ROOT / "state_vlm_multiview",
    )
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL_PATH))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--episodes", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of phase samples.")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index for parallel runs (0-based).")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards for parallel runs.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--combine-views", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_FILE,
        help="Few-shot prompt template (metrics/vlm_prompt/01.txt).",
    )
    parser.add_argument(
        "--reference-db-root",
        type=Path,
        default=DEFAULT_SHARED_REFERENCE_DB,
        help="Shared few-shot KB (test_data/vlm_data_base) or legacy per-task DB root.",
    )
    parser.add_argument(
        "--use-reference-db",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use 1 positive + 3 negative reference examples in VLM prompt.",
    )
    parser.add_argument("--skip-vlm", action="store_true", help="Only collect samples and math checks.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--epipolar-error-threshold",
        type=float,
        default=EPIPOLAR_ERROR_THRESHOLD_PX,
        help="Pass threshold for mean epipolar error in pixels.",
    )
    parser.add_argument(
        "--epipolar-min-matches",
        type=int,
        default=EPIPOLAR_MIN_MATCHES,
        help="Minimum feature matches required for epipolar math check.",
    )
    return parser.parse_args()


def build_vlm_prompt_legacy(
    sample: dict[str, Any],
    prompt_config: dict[str, Any] | None = None,
) -> str:
    cfg = prompt_config or load_prompt_config()
    template = cfg["legacy_template"]
    return template.format(
        instruction=sample.get("instruction") or sample["task"],
        phase=sample["phase"],
        description=sample["description"],
        arm=sample["arm"],
    )


def build_vlm_content(
    sample: dict[str, Any],
    target_image: Image.Image,
    reference_bundle: dict[str, Any] | None,
    prompt_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cfg = prompt_config or load_prompt_config()
    if reference_bundle is None:
        prompt = build_vlm_prompt_legacy(sample, cfg)
        return [{"type": "text", "text": prompt}, {"type": "image", "image": target_image}]

    intro = cfg["intro_template"].format(
        positive_library_intro=cfg["positive_library_intro"],
        negative_library_intro=cfg["negative_library_intro"],
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": intro}]
    figure_num = 1
    bundle_meta = reference_bundle.get("meta") or {}

    positive_images = reference_bundle.get("positives")
    positive_paths = reference_bundle.get("positive_paths") or []
    if not positive_images and reference_bundle.get("positive") is not None:
        positive_images = [reference_bundle["positive"]]
        positive_paths = positive_paths or ["positive"]
    for index, positive in enumerate(positive_images or []):
        stem = Path(positive_paths[index]).stem if index < len(positive_paths) else "positive"
        caption = _reference_caption(stem, cfg, bundle_meta, "正样本（一致）")
        content.append({"type": "text", "text": f"**图{figure_num} — {caption}**"})
        content.append({"type": "image", "image": positive})
        figure_num += 1

    neg_metas = bundle_meta.get("negatives", [])
    negative_paths = reference_bundle.get("negative_paths") or []
    for index, negative in enumerate(reference_bundle.get("negatives") or []):
        stem = Path(negative_paths[index]).stem if index < len(negative_paths) else f"negative_{index + 1}"
        fallback = (
            negative_mix_caption(neg_metas[index])
            if index < len(neg_metas)
            else "负样本（不一致）"
        )
        caption = _reference_caption(stem, cfg, bundle_meta, fallback)
        content.append({"type": "text", "text": f"**图{figure_num} — {caption}**"})
        content.append({"type": "image", "image": negative})
        figure_num += 1

    eval_text = cfg["eval_template"].format(
        instruction=sample.get("instruction") or sample["task"],
        phase=sample["phase"],
        description=sample["description"],
        arm=sample["arm"],
        target_figure=figure_num,
    )
    content.append({"type": "text", "text": eval_text})
    content.append({"type": "text", "text": f"**图{figure_num} — 待评估样本**"})
    content.append({"type": "image", "image": target_image})
    return content


def run_qwen(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    content: list[dict[str, Any]],
    max_new_tokens: int,
) -> str:
    images = [item["image"] for item in content if item.get("type") == "image"]
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if "pixel_values" not in inputs:
        image_inputs = processor(images=images, return_tensors="pt")
        inputs["pixel_values"] = image_inputs.pixel_values
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    input_len = inputs["input_ids"].shape[1]
    return processor.batch_decode(
        generated_ids[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None


def parse_vlm_result(raw_text: str) -> dict[str, Any] | None:
    obj = _extract_json_object(raw_text)
    if obj is None:
        return None
    out: dict[str, Any] = {}
    for key in (
        "state_consistency_score",
        "object_consistency_score",
    ):
        score = normalize_01_component_score(obj.get(key))
        if score is None:
            return None
        out[key] = score
    for key in (
        "state_consistency_reason",
        "object_consistency_reason",
    ):
        out[key] = str(obj.get(key, ""))
    return out


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [item for item in results if item.get("vlm_parsed")]
    math_ok = [item for item in results if item.get("math_check", {}).get("math_score") is not None]

    def mean_score(key: str) -> float | None:
        vals = [float(item["vlm_parsed"][key]) for item in ok if key in item["vlm_parsed"]]
        return round(sum(vals) / len(vals), 4) if vals else None

    by_task: dict[str, list[float]] = defaultdict(list)
    for item in ok:
        score = (
            float(item["vlm_parsed"]["state_consistency_score"])
            + float(item["vlm_parsed"]["object_consistency_score"])
        ) / 2.0
        by_task[item["task"]].append(score)

    return {
        "total_samples": len(results),
        "vlm_success": len(ok),
        "vlm_errors": len(results) - len(ok),
        "mean_state_consistency": mean_score("state_consistency_score"),
        "mean_object_consistency": mean_score("object_consistency_score"),
        "mean_math_score": round(
            sum(float(item["math_check"]["math_score"]) for item in math_ok) / len(math_ok), 4
        )
        if math_ok
        else None,
        "mean_epipolar_error": round(
            sum(float(item["math_check"]["epipolar_error_mean"]) for item in math_ok) / len(math_ok), 4
        )
        if math_ok
        else None,
        "mean_by_task": {
            task: round(sum(vals) / len(vals), 4) for task, vals in sorted(by_task.items())
        },
    }


def load_all_samples(
    state_root: Path,
    tasks: list[str] | None = None,
    episodes: list[str] | None = None,
    limit: int = 0,
    shard_id: int = 0,
    num_shards: int = 1,
) -> list[dict[str, Any]]:
    state_files = discover_state_files(state_root, tasks, episodes)
    all_samples: list[dict[str, Any]] = []
    for state_path in state_files:
        record = json.loads(state_path.read_text(encoding="utf-8"))
        all_samples.extend(collect_phase_samples(record))
    all_samples.sort(key=sample_sort_key)
    if limit:
        all_samples = all_samples[:limit]
    if num_shards > 1:
        all_samples = [
            sample for idx, sample in enumerate(all_samples) if idx % num_shards == shard_id
        ]
    return all_samples


def main() -> None:
    args = parse_args()
    if args.shard_id < 0 or args.num_shards < 1:
        raise ValueError("shard-id must be >= 0 and num-shards must be >= 1")
    if args.shard_id >= args.num_shards:
        raise ValueError(f"shard-id ({args.shard_id}) must be < num-shards ({args.num_shards})")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    state_root = args.state_root.resolve()
    frames_root = args.frames_root.resolve()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    prompt_config = load_prompt_config(args.prompt_file.resolve())
    reference_db_root = resolve_reference_db_root(args.reference_db_root, prompt_config)
    use_reference_db = args.use_reference_db and reference_db_root.is_dir()

    result_path = output_root / "state_vlm_multiview_results.json"
    summary_path = output_root / "state_vlm_multiview_summary.json"
    manifest_path = output_root / "eval_manifest.json"

    all_samples = load_all_samples(
        state_root,
        args.tasks,
        args.episodes,
        args.limit,
        args.shard_id,
        args.num_shards,
    )
    if args.num_shards > 1:
        print(f"[shard] {args.shard_id + 1}/{args.num_shards} -> {len(all_samples)} samples on gpu {args.gpu}")

    existing: list[dict[str, Any]] = []
    done: set[str] = set()
    if result_path.exists() and not args.overwrite:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        done = {item["sample_id"] for item in existing if item.get("vlm_parsed") or item.get("math_check")}

    model = processor = None
    if not args.skip_vlm:
        print(f"[model] {args.model_path}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(args.model_path),
            torch_dtype="auto",
            device_map="auto",
        ).eval()
        processor = AutoProcessor.from_pretrained(str(args.model_path))
    else:
        print("[skip-vlm] VLM inference disabled; vlm_raw / vlm_parsed will not be written.")

    results = list(existing)
    manifest_samples: list[dict[str, Any]] = []

    for sample in tqdm(all_samples, desc="state-vlm-multiview", ncols=100):
        sid = sample_key(sample)
        manifest_samples.append(
            {
                "sample_id": sid,
                "task": sample["task"],
                "episode": sample["episode"],
                "phase": sample["phase"],
                "arm": sample["arm"],
                "sampled_frame": sample["sampled_frame"],
                "description": sample["description"],
            }
        )
        if sid in done:
            continue

        item: dict[str, Any] = {
            "sample_id": sid,
            **sample,
        }
        try:
            images, image_sources = load_triplet_images(sample, frames_root, dataset_root)
            item["image_sources"] = image_sources
            active_view = ARM_TO_VIEW.get(sample["arm"])
            if active_view is None:
                raise ValueError(f"unsupported arm for epipolar check: {sample['arm']}")
            item["math_check"] = compute_math_check(
                sample,
                images["head"],
                images[active_view],
                args.epipolar_error_threshold,
                args.epipolar_min_matches,
            )

            if args.skip_vlm:
                results.append(item)
            else:
                reference_bundle = load_reference_bundle(reference_db_root, sample) if use_reference_db else None
                if reference_bundle is not None or args.combine_views:
                    target_image = combine_views(images)
                    vlm_content = build_vlm_content(sample, target_image, reference_bundle, prompt_config)
                else:
                    prompt = build_vlm_prompt_legacy(sample, prompt_config)
                    vlm_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
                    for view in VIEW_DIRS:
                        vlm_content.append({"type": "image", "image": images[view]})

                raw_text = run_qwen(model, processor, vlm_content, args.max_new_tokens)
                parsed = parse_vlm_result(raw_text)
                raw_obj = _extract_json_object(raw_text)
                item["vlm_prompt_version"] = VLM_PROMPT_VERSION if reference_bundle else "legacy_tri_view"
                item["vlm_reference"] = {
                    "enabled": reference_bundle is not None,
                    "entry_dir": reference_bundle.get("entry_dir") if reference_bundle else None,
                    "num_negatives": len(reference_bundle["negatives"]) if reference_bundle else 0,
                }
                item["vlm_raw"] = raw_obj if raw_obj is not None else raw_text
                item["vlm_parsed"] = parsed
                if parsed is None:
                    item["error"] = "failed_to_parse_vlm_json"
                results.append(item)
        except Exception as exc:
            item["error"] = str(exc)
            results.append(item)

        result_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "state_root": str(state_root),
        "frames_root": str(frames_root),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "skip_vlm": args.skip_vlm,
        "vlm_prompt_version": VLM_PROMPT_VERSION,
        "vlm_prompt_file": prompt_config.get("prompt_file"),
        "use_reference_db": use_reference_db,
        "reference_db_root": str(reference_db_root),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "gpu": args.gpu,
        "stats": summarize_results(results),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at": summary["generated_at"],
                "num_samples": len(manifest_samples),
                "samples": manifest_samples,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Samples: {len(all_samples)}")
    print(f"Results: {result_path}")
    print(f"Summary: {summary_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
