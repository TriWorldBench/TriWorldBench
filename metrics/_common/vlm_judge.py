#!/usr/bin/env python3
"""Per-view VLM judge for the 0710 tri-view evaluation protocol."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


VIEW_ORDER = ("head", "left", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output-dir", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--hyperparameters", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def load_entries(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"summary must contain a list: {path}")
    return [item for item in data if isinstance(item, dict)]


def image_paths(folder: str | Path) -> list[Path]:
    path = Path(folder)
    if not path.is_dir():
        return []
    return sorted(
        item for item in path.iterdir() if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )


def uniform_indices(count: int, target: int) -> list[int]:
    if count <= 0 or target <= 0:
        return []
    if target == 1:
        return [0]
    if count == 1:
        return [0] * target
    return [round(index * (count - 1) / (target - 1)) for index in range(target)]


def sample_view_frames(folder: str | Path, num_frames: int) -> list[Image.Image]:
    paths = image_paths(folder)
    frames: list[Image.Image] = []
    for index in uniform_indices(len(paths), num_frames):
        bgr = cv2.imread(str(paths[index]), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    return frames


def instruction_of(item: dict[str, Any]) -> str:
    prompt = item.get("prompt", "")
    if isinstance(prompt, list) and prompt:
        return str(prompt[0])
    return str(prompt) if isinstance(prompt, str) else ""


def build_prompt(view: str, instruction: str) -> str:
    metric_section = """1. Interaction_Quality
- 1: impossible contact, penetration, or no meaningful interaction
- 2: major physical errors
- 3: generally plausible with visible defects
- 4: realistic robot-object interaction
- 5: visually convincing interaction physics

2. Perspectivity
- 1: incoherent geometry or severe shape/scale changes
- 2: unstable 3D structure
- 3: mostly consistent with minor drift
- 4: stable perspective and depth relationships
- 5: fully coherent camera geometry and 3D structure
"""
    output_keys = ["Interaction_Quality", "Perspectivity"]
    if view == "head":
        metric_section += """
3. Instruction_Following
- 1: unrelated to the instruction
- 2: major action/object errors
- 3: follows the intent with execution errors
- 4: mostly correct with minor deviations
- 5: correctly executes the specified action and outcome
"""
        output_keys.append("Instruction_Following")

    schema = ",\n".join(
        f'  "{key}": {{"score": 1, "reason": "specific visual evidence"}}' for key in output_keys
    )
    instruction_block = instruction if instruction else "No instruction text is available."
    return f"""You are evaluating the {view} camera of a robot manipulation video.

Camera role:
- head: third-person overview and the authoritative view for task completion
- left: left wrist camera; close-up evidence may be partial or occluded
- right: right wrist camera; close-up evidence may be partial or occluded

Task instruction: {instruction_block}

Judge only evidence visible in this camera. Do not infer evidence from other cameras.
The agent must be a robot arm/end-effector, not a human hand.

{metric_section}
Use integer scores from 1 to 5. Output only one valid JSON object with exactly these keys:
{{
{schema}
}}
"""


def run_model(model, processor, prompt: str, images: list[Image.Image]) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": image} for image in images],
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if "pixel_values" not in inputs:
        inputs["pixel_values"] = processor(images=images, return_tensors="pt").pixel_values
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    generated = model.generate(
        **inputs,
        max_new_tokens=320,
        do_sample=False,
        temperature=0.1,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    input_length = inputs["input_ids"].shape[1]
    return processor.batch_decode(
        generated[:, input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def extract_json(raw_text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw_text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


def normalize_metrics(parsed: dict[str, Any], view: str) -> dict[str, Any]:
    keys = ["Interaction_Quality", "Perspectivity"]
    if view == "head":
        keys.append("Instruction_Following")
    output = {}
    for key in keys:
        item = parsed.get(key, {}) if isinstance(parsed.get(key), dict) else {}
        try:
            score = int(item.get("score"))
        except (TypeError, ValueError):
            score = 0
        if score < 1 or score > 5:
            score = 0
        output[key] = {
            "score": score if score else None,
            "score_normalized": round(score / 5.0, 4) if score else None,
            "reason": str(item.get("reason") or ""),
        }
    return output


def entry_id(item: dict[str, Any]) -> str:
    return str(item.get("episode") or "unknown")


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, list):
        return {}
    return {entry_id(item): item for item in data if isinstance(item, dict)}


def main() -> None:
    args = parse_args()
    params = load_yaml(args.hyperparameters)
    base_config = load_yaml(args.config_path)
    vlm_config = params.get("vlm", {})
    num_frames = int(vlm_config.get("num_frames", 16))
    model_path = args.model_path or Path(base_config.get("ckpt", {}).get("vlm_model", ""))
    if not str(model_path):
        raise ValueError("VLM model path is missing")

    entries = load_entries(args.summary_json)
    existing = {} if args.overwrite else load_existing(args.output)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path), torch_dtype="auto", device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(str(model_path))
    args.raw_output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for source in tqdm(entries, desc="vlm-0710", unit="episode"):
        key = entry_id(source)
        if key in existing:
            results.append(existing[key])
            continue
        result = {
            "episode": source.get("episode"),
            "video": source.get("video"),
            "views": {},
        }
        gen_views = source.get("gen_views", {}) if isinstance(source.get("gen_views"), dict) else {}
        instruction = instruction_of(source)
        for view in VIEW_ORDER:
            view_result: dict[str, Any] = {"metrics": {}, "error": None, "raw_response_file": None}
            frames = sample_view_frames(gen_views.get(view, ""), num_frames)
            if not frames:
                view_result["error"] = "no_frames"
                result["views"][view] = view_result
                continue
            try:
                raw_text = run_model(model, processor, build_prompt(view, instruction), frames)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                raw_path = args.raw_output_dir / f"{key}__{view}__{timestamp}.json"
                raw_path.write_text(json.dumps({"raw_response": raw_text}, ensure_ascii=False, indent=2), encoding="utf-8")
                view_result["raw_response_file"] = str(raw_path)
                parsed = extract_json(raw_text)
                if parsed is None:
                    view_result["error"] = "parse_failed"
                else:
                    view_result["metrics"] = normalize_metrics(parsed, view)
            except Exception as exc:
                view_result["error"] = f"{type(exc).__name__}: {exc}"
            result["views"][view] = view_result
        results.append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
