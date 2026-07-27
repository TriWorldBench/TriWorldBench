#!/usr/bin/env python3
"""VLM dual-view consistency evaluation driven by STATE phase JSON files.

For each phase that is not idle/approach, load head + active wrist frames at
sampled_frame (left arm -> head+left, right arm -> head+right), query a VLM for:
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

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from metrics._common.vlm_epipolar import compute_math_check
from metrics._common.frames import frame_filename
PROJECT_ROOT = Path(os.environ.get("MATRIC_EVAL_ROOT", Path(__file__).resolve().parents[2]))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "test_data"))
WEIGHTS_ROOT = Path(os.environ.get("WEIGHTS_ROOT", PROJECT_ROOT / "weights"))
OUTPUTS_ROOT = Path(os.environ.get("OUTPUTS_ROOT", PROJECT_ROOT / "output"))
DEFAULT_TESTSET_ROOT = Path(os.environ.get("TESTSET_ROOT", DATA_ROOT))
DEFAULT_MODEL_PATH = WEIGHTS_ROOT / "qwenvl3"
VIEW_DIRS = ("head", "left", "right")
VIEW_LABELS = {
    "head": "head-view",
    "left": "left-wrist-view",
    "right": "right-wrist-view",
}
PREVIEW_CAMERAS = {
    "head": "head_camera",
    "left": "left_camera",
    "right": "right_camera",
}
SKIP_PHASES = {"idle", "approach"}
ARM_TO_VIEW = {"left": "left", "right": "right"}
ARM_TO_CAMERA = {"left": "left_camera", "right": "right_camera", "head": "head_camera"}
EPIPOLAR_ERROR_THRESHOLD_PX = 5.0
EPIPOLAR_MIN_MATCHES = 5
VLM_SCORE_MIN = 1
VLM_SCORE_MAX = 5
VLM_PROMPT_VERSION = "v4_en_dual_view_gt_calibrated_1-5"

# --- Prompt version history (reference only; not used at runtime) ---
# v1: Chinese tri-view prompt, scores 0-100, inputs head + left + right (3 images).
VLM_PROMPT_TEMPLATE_V1_ZH_TRIVIEW = """你是机器人三视角一致性评测器。

输入为同一时刻的三张图片，按顺序分别为：
1. head-view（第三人称头部相机）
2. left-wrist-view（左腕相机；若该臂未参与可主要为背景）
3. right-wrist-view（右腕相机；若该臂未参与可主要为背景）

任务总指令：{instruction}
当前阶段类型：{phase}
当前阶段描述：{description}
标注执行手臂：{arm}

请分别评估：
1. **状态一致性 (state_consistency)**：三视角呈现的动作状态是否与「当前阶段描述」一致。
2. **物体一致性 (object_consistency)**：被移动/操作物体在三视角中的颜色、形状是否一致（允许遮挡与视角差异，但不应出现明显矛盾）。

评分规则：0-100 整数，100 表示完全一致，0 表示完全不一致。
只能输出合法 JSON，不要输出其他文字：
{{
  "state_consistency_score": 0到100之间的整数,
  "state_consistency_reason": "少于30字的简短理由",
  "object_consistency_score": 0到100之间的整数,
  "object_consistency_reason": "少于30字的简短理由"
}}
"""

# v4 (active): English dual-view prompt, scores 1-5, head + active wrist only.
VLM_PROMPT_TEMPLATE_LEFT = """You are an expert dual-view consistency evaluator for a robotic manipulation system.

You are given two images captured at the exact same timestep:
1. head-view (third-person head camera)
2. left-wrist-view (left wrist camera, often close-up, cropped, blurry, or occluded by the gripper)

Task instruction: {instruction}
Current phase type: {phase}
Current phase description: {description}
Active arm: left

Evaluate whether the two views are compatible observations of the same synchronized robot state.

Calibration target:
- Start from the assumption that the two views are consistent. Lower the score only when visible evidence clearly contradicts this.
- A score below 4 requires a concrete visual contradiction, not just missing evidence.

Important assumptions:
- The two images are synchronized and may look very different because the cameras have very different viewpoints.
- The wrist camera may show only gripper, robot arm, table texture, blur, a partial object, or no useful object view.
- Natural occlusion, cropping, motion blur, lighting differences, and close-up wrist perspective should NOT be treated as inconsistency.
- Do NOT require the manipulated object to be clearly visible in both views.
- Absence of the object in the wrist view is NOT a contradiction by itself.
- Phase descriptions are approximate helpers. Do not punish minor disagreement with the text if the two images remain visually compatible.

Scoring guidance:
5 = Consistent. Both views support the same state/object, OR one view is informative and the other is compatible/occluded/uninformative.
4 = Mostly consistent. There is some ambiguity from blur, crop, occlusion, viewpoint, or phase wording, but no clear contradiction.
3 = Uncertain. Use only when both views are too unclear to judge and there is no strong positive evidence of consistency.
2 = Likely inconsistent. There is a specific visible mismatch in timing, robot state, active arm, or object, but uncertainty remains.
1 = Clearly inconsistent. The views show an obvious contradiction, such as incompatible objects, impossible robot states, or clearly different action phases.

For state_consistency:
- Focus on whether the robot pose/action could match the current phase description.
- If either view supports the phase and the other view does not contradict it, score 5.
- If the phase is hard to see because of occlusion/cropping/blur but the views are still compatible, score 4.
- Use 1 or 2 only when the two views visibly cannot represent the same robot state.

For object_consistency:
- Focus only on clear contradictions in object identity, color, or shape.
- If the object is visible in only one view and the other view is occluded/cropped/blurred/uninformative, score 4 or 5.
- If the object is not clearly visible in either view but there is no contradiction, score 4.
- Use 1 or 2 only when both views show incompatible object evidence.

Output ONLY valid JSON, no other text or markdown formatting. Use this exact schema:
{{
  "state_consistency_score": <integer 1-5>,
  "state_consistency_reason": "<brief justification under 30 words>",
  "object_consistency_score": <integer 1-5>,
  "object_consistency_reason": "<brief justification under 30 words>"
}}
"""

VLM_PROMPT_TEMPLATE_RIGHT = """You are an expert dual-view consistency evaluator for a robotic manipulation system.

You are given two images captured at the exact same timestep:
1. head-view (third-person head camera)
2. right-wrist-view (right wrist camera, often close-up, cropped, blurry, or occluded by the gripper)

Task instruction: {instruction}
Current phase type: {phase}
Current phase description: {description}
Active arm: right

Evaluate whether the two views are compatible observations of the same synchronized robot state.

Calibration ta
- Start from the assumption that the two views are consistent. Lower the score only when visible evidence clearly contradicts this.
- A score below 4 requires a concrete visual contradiction, not just missing evidence.

Important assumptions:
- The two images are synchronized and may look very different because the cameras have very different viewpoints.
- The wrist camera may show only gripper, robot arm, table texture, blur, a partial object, or no useful object view.
- Natural occlusion, cropping, motion blur, lighting differences, and close-up wrist perspective should NOT be treated as inconsistency.
- Do NOT require the manipulated object to be clearly visible in both views.
- Absence of the object in the wrist view is NOT a contradiction by itself.
- Phase descriptions are approximate helpers. Do not punish minor disagreement with the text if the two images remain visually compatible.

Scoring guidance:
5 = Consistent. Both views support the same state/object, OR one view is informative and the other is compatible/occluded/uninformative.
4 = Mostly consistent. There is some ambiguity from blur, crop, occlusion, viewpoint, or phase wording, but no clear contradiction.
3 = Uncertain. Use only when both views are too unclear to judge and there is no strong positive evidence of consistency.
2 = Likely inconsistent. There is a specific visible mismatch in timing, robot state, active arm, or object, but uncertainty remains.
1 = Clearly inconsistent. The views show an obvious contradiction, such as incompatible objects, impossible robot states, or clearly different action phases.

For state_consistency:
- Focus on whether the robot pose/action could match the current phase description.
- If either view supports the phase and the other view does not contradict it, score 5.
- If the phase is hard to see because of occlusion/cropping/blur but the views are still compatible, score 4.
- Use 1 or 2 only when the two views visibly cannot represent the same robot state.

For object_consistency:
- Focus only on clear contradictions in object identity, color, or shape.
- If the object is visible in only one view and the other view is occluded/cropped/blurred/uninformative, score 4 or 5.
- If the object is not clearly visible in either view but there is no contradiction, score 4.
- Use 1 or 2 only when both views show incompatible object evidence.

Output ONLY valid JSON, no other text or markdown formatting. Use this exact schema:
{{
  "state_consistency_score": <integer 1-5>,
  "state_consistency_reason": "<brief justification under 30 words>",
  "object_consistency_score": <integer 1-5>,
  "object_consistency_reason": "<brief justification under 30 words>"
}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STATE-driven tri-view VLM consistency evaluation.")
    parser.add_argument(
        "--state-root",
        type=Path,
        default=DEFAULT_TESTSET_ROOT / "STATE",
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
    parser.add_argument("--combine-views", action=argparse.BooleanOptionalAction, default=False)
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


def discover_state_files(state_root: Path, tasks: list[str] | None, episodes: list[str] | None) -> list[Path]:
    files = sorted(state_root.glob("*/*.json"))
    files = [path for path in files if path.name != "summary.json"]
    if tasks:
        task_set = set(tasks)
        files = [path for path in files if path.parent.name in task_set]
    if episodes:
        episode_set = set(episodes)
        files = [path for path in files if path.stem in episode_set]
    return files


def collect_phase_samples(state_record: dict[str, Any]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for phase in state_record.get("phases", []):
        name = phase.get("phase", "")
        if name in SKIP_PHASES:
            continue
        if "sampled_frame" not in phase:
            continue
        samples.append(
            {
                "task": state_record["task"],
                "episode": state_record["episode"],
                "phase": name,
                "arm": phase.get("arm", ""),
                "description": phase.get("description", ""),
                "sampled_frame": int(phase["sampled_frame"]),
                "start_frame": int(phase.get("start_frame", phase["sampled_frame"])),
                "end_frame": int(phase.get("end_frame", phase["sampled_frame"])),
                "instruction": state_record.get("instruction", ""),
                "task_category": state_record.get("task_category", ""),
                "hdf5_path": state_record.get("hdf5_path", ""),
            }
        )
    return samples


def frame_path_from_gt(frames_root: Path, task: str, episode: str, view: str, frame_idx: int) -> Path:
    frame_name = frame_filename(frame_idx)
    for subdir in ("frames", "video"):
        candidate = frames_root / task / episode / view / subdir / frame_name
        if candidate.is_file():
            return candidate
    return frames_root / task / episode / view / "frames" / frame_name


def preview_mp4_path(dataset_root: Path, task: str, episode: str, view: str) -> Path:
    camera = PREVIEW_CAMERAS[view]
    return (
        dataset_root
        / task
        / "gl_clean_1000_0615"
        / "data"
        / f"{episode}_preview"
        / f"{camera}.mp4"
    )


def read_frame_from_mp4(mp4_path: Path, frame_idx: int) -> Image.Image | None:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def load_triplet_images(
    sample: dict[str, Any],
    frames_root: Path,
    dataset_root: Path,
) -> tuple[dict[str, Image.Image], dict[str, str]]:
    images: dict[str, Image.Image] = {}
    sources: dict[str, str] = {}
    task = sample["task"]
    episode = sample["episode"]
    frame_idx = sample["sampled_frame"]

    for view in VIEW_DIRS:
        gt_path = frame_path_from_gt(frames_root, task, episode, view, frame_idx)
        if gt_path.is_file():
            images[view] = Image.open(gt_path).convert("RGB")
            sources[view] = str(gt_path)
            continue
        mp4_path = preview_mp4_path(dataset_root, task, episode, view)
        image = read_frame_from_mp4(mp4_path, frame_idx)
        if image is None:
            raise FileNotFoundError(
                f"Missing frame {frame_idx} for {task}/{episode}/{view}: "
                f"no {gt_path} and no readable {mp4_path}"
            )
        images[view] = image
        sources[view] = str(mp4_path) + f"#frame={frame_idx}"
    return images, sources


def combine_views(images: dict[str, Image.Image], view_order: tuple[str, ...] = VIEW_DIRS) -> Image.Image:
    arrays = [np.asarray(images[view]) for view in view_order]
    heights = [arr.shape[0] for arr in arrays]
    target_h = min(heights)
    resized = []
    for arr in arrays:
        if arr.shape[0] != target_h:
            scale = target_h / arr.shape[0]
            width = max(1, int(arr.shape[1] * scale))
            arr = cv2.resize(arr, (width, target_h), interpolation=cv2.INTER_AREA)
        resized.append(arr)
    return Image.fromarray(cv2.hconcat(resized))


def active_view_for_arm(arm: str) -> str:
    active_view = ARM_TO_VIEW.get(arm)
    if active_view is None:
        raise ValueError(f"unsupported arm for VLM evaluation: {arm}")
    return active_view


def vlm_view_order(sample: dict[str, Any]) -> tuple[str, ...]:
    active_view = active_view_for_arm(sample["arm"])
    return ("head", active_view)


def build_vlm_prompt(sample: dict[str, Any]) -> str:
    arm = sample.get("arm", "")
    template = VLM_PROMPT_TEMPLATE_LEFT if arm == "left" else VLM_PROMPT_TEMPLATE_RIGHT
    if arm not in ARM_TO_VIEW:
        raise ValueError(f"unsupported arm for VLM prompt: {arm}")
    return template.format(
        instruction=sample.get("instruction") or sample["task"],
        phase=sample["phase"],
        description=sample["description"],
    )


def run_qwen(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    prompt: str,
    images: list[Image.Image],
    max_new_tokens: int,
) -> str:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image in images:
        content.append({"type": "image", "image": image})
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


def parse_vlm_score(value: Any) -> int | None:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(VLM_SCORE_MIN, min(VLM_SCORE_MAX, score))


def parse_vlm_result(raw_text: str) -> dict[str, Any] | None:
    obj = _extract_json_object(raw_text)
    if obj is None:
        return None
    out: dict[str, Any] = {}
    for key in (
        "state_consistency_score",
        "object_consistency_score",
    ):
        score = parse_vlm_score(obj.get(key))
        if score is None:
            return None
        out[key] = score
    for key in (
        "state_consistency_reason",
        "object_consistency_reason",
    ):
        out[key] = str(obj.get(key, ""))
    out["overall_consistency_score"] = round(
        (out["state_consistency_score"] + out["object_consistency_score"]) / 2.0,
        4,
    )
    return out


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [item for item in results if item.get("vlm_parsed")]
    math_ok = [item for item in results if item.get("math_check", {}).get("math_score") is not None]

    def mean_score(key: str) -> float | None:
        vals = [float(item["vlm_parsed"][key]) for item in ok if key in item["vlm_parsed"]]
        return round(sum(vals) / len(vals), 4) if vals else None

    by_task: dict[str, list[float]] = defaultdict(list)
    overall_scores: list[float] = []
    for item in ok:
        parsed = item["vlm_parsed"]
        overall = float(parsed["overall_consistency_score"])
        overall_scores.append(overall)
        by_task[item["task"]].append(overall)

    return {
        "total_samples": len(results),
        "vlm_success": len(ok),
        "vlm_errors": len(results) - len(ok),
        "score_scale": f"{VLM_SCORE_MIN}-{VLM_SCORE_MAX}",
        "mean_state_consistency": mean_score("state_consistency_score"),
        "mean_object_consistency": mean_score("object_consistency_score"),
        "mean_overall_consistency": round(sum(overall_scores) / len(overall_scores), 4)
        if overall_scores
        else None,
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


from metrics._common.episode_sort import sample_sort_key


def sample_key(sample: dict[str, Any]) -> str:
    return (
        f"{sample['task']}/{sample['episode']}/"
        f"{sample['phase']}/{sample['arm']}/f{sample['sampled_frame']}"
    )


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
                views = vlm_view_order(sample)
                item["vlm_views"] = list(views)
                item["vlm_pair"] = f"head-{views[1]}"
                prompt = build_vlm_prompt(sample)
                if args.combine_views:
                    vlm_images = [combine_views(images, views)]
                else:
                    vlm_images = [images[view] for view in views]
                raw_text = run_qwen(model, processor, prompt, vlm_images, args.max_new_tokens)
                parsed = parse_vlm_result(raw_text)
                raw_obj = _extract_json_object(raw_text)
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
    mean_overall = summary["stats"].get("mean_overall_consistency")
    if mean_overall is not None:
        print(f"Mean overall VLM consistency ({VLM_SCORE_MIN}-{VLM_SCORE_MAX}): {mean_overall}")


if __name__ == "__main__":
    main()
