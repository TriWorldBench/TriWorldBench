#!/usr/bin/env python3
"""Multiple-choice VLM evaluation over multiview episode frames.

The pipeline is intentionally simple:
1. Sample frames from an episode.
2. Show those frames to the VLM.
3. Ask multiple-choice questions loaded from the episode QA JSON.
4. Score each model answer by matching it to the reference answer.
5. Write per-episode reports plus aggregate JSON/CSV reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

_BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
if str(_BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_ROOT))

from metrics._common.episode_sort import episode_sort_key

VQA_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(os.environ.get("VLM_EVAL_DATA_ROOT", _BENCHMARK_ROOT / "generate"))
DEFAULT_QA_ROOT = Path(os.environ.get("VLM_EVAL_QA_ROOT", VQA_ROOT / "qa_val"))
DEFAULT_PHASE_WINDOW_ROOT = Path(os.environ.get("VLM_EVAL_PHASE_WINDOW_ROOT", VQA_ROOT / "phase_windows"))
DEFAULT_MODEL_PATH = Path(os.environ.get("VLM_EVAL_MODEL_PATH", _BENCHMARK_ROOT / "weights" / "qwenvl3"))
DEFAULT_GPU = os.environ.get("VLM_EVAL_GPU", "0,1,2,3,4,5,6,7")
DEFAULT_PYTORCH_ALLOC_CONF = "expandable_segments:True"
DEFAULT_VIEWS = ("head", "left", "right")
VIEW_LABELS = {
    "head": "Head view",
    "left": "Left wrist view",
    "right": "Right wrist view",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        "--input_dir",
        "--data-root",
        "--data_root",
        dest="data_root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory containing episode subdirectories (default: %(default)s).",
    )
    parser.add_argument("--qa-root", type=Path, default=DEFAULT_QA_ROOT)
    parser.add_argument("--phase-window-root", type=Path, default=DEFAULT_PHASE_WINDOW_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--cot-log-root",
        type=Path,
        default=None,
        help="Directory for per-episode reasoning logs (default: <output-root>/cot_logs).",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--gpu", default=DEFAULT_GPU)
    parser.add_argument("--views", nargs="+", default=list(DEFAULT_VIEWS))
    parser.add_argument("--episodes", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of episodes after filtering.")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help=(
            "Maximum frames sampled per view: uniformly across the whole video for "
            "view/general questions and within each aligned window for window questions "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument("--image-max-edge", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--combine-views", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-model", action="store_true", help="Write manifests/reports without VLM calls.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--init-qa", action="store_true", help="Create missing empty QA files and exit.")
    parser.add_argument("--overwrite-qa", action="store_true", help="Overwrite QA files during --init-qa.")
    return parser.parse_args()


def output_data_label(data_root: Path) -> str:
    import re

    label = re.sub(r"[^A-Za-z0-9._-]+", "_", data_root.name).strip("._-")
    return label or "dataset"


def discover_episodes(data_root: Path, episodes: list[str] | None, limit: int) -> list[Path]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root not found: {data_root}")
    episode_dirs = sorted(
        [item for item in data_root.iterdir() if item.is_dir()],
        key=lambda path: episode_sort_key(path.name),
    )
    if episodes:
        wanted = {ep if ep.startswith("episode") else f"episode{ep}" for ep in episodes}
        episode_dirs = [path for path in episode_dirs if path.name in wanted]
    if limit:
        episode_dirs = episode_dirs[:limit]
    return episode_dirs


def apply_shard(items: list[Path], shard_id: int, num_shards: int) -> list[Path]:
    if shard_id < 0 or num_shards < 1:
        raise ValueError("shard-id must be >= 0 and num-shards must be >= 1")
    if shard_id >= num_shards:
        raise ValueError("shard-id must be smaller than num-shards")
    if num_shards == 1:
        return items
    return [item for idx, item in enumerate(items) if idx % num_shards == shard_id]


def qa_template(episode: str) -> dict[str, Any]:
    return {
        "episode": episode,
        "Q&A": [],
    }


def init_qa_files(data_root: Path, qa_root: Path, episodes: list[str] | None, limit: int, overwrite: bool) -> None:
    episode_dirs = discover_episodes(data_root, episodes, limit)
    qa_root.mkdir(parents=True, exist_ok=True)
    created = overwritten = kept = 0
    for episode_dir in episode_dirs:
        qa_dir = qa_root / episode_dir.name
        qa_path = qa_dir / "qa.json"
        existed = qa_path.exists()
        if qa_path.exists() and not overwrite:
            kept += 1
            continue
        qa_dir.mkdir(parents=True, exist_ok=True)
        qa_path.write_text(json.dumps(qa_template(episode_dir.name), indent=2) + "\n", encoding="utf-8")
        if existed:
            overwritten += 1
        else:
            created += 1
    print(f"QA root: {qa_root}")
    print(f"Episodes: {len(episode_dirs)} created={created} overwritten={overwritten} kept={kept}")


def qa_path_for_episode(episode_dir: Path, qa_root: Path) -> Path | None:
    candidates = [
        qa_root / episode_dir.name / "qa.json",
        qa_root / f"{episode_dir.name}.json",
        episode_dir / "qa.json",
        episode_dir / f"{episode_dir.name}.json",
        episode_dir / "questions.json",
    ]
    return next((path for path in candidates if path.is_file()), None)


def load_json_or_empty(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {"Q&A": []}
    return json.loads(text)


def normalize_choice_code(value: Any) -> str | None:
    code = str(value).strip().upper() if value is not None else ""
    return code if re.fullmatch(r"[A-Z][A-Z0-9_-]*", code) else None


def normalize_selections(value: Any, question_index: int) -> dict[str, str]:
    if not isinstance(value, dict) or len(value) < 2:
        raise ValueError(f"question {question_index} selections must be an object with at least 2 options")
    selections: dict[str, str] = {}
    for raw_code, raw_label in value.items():
        code = normalize_choice_code(raw_code)
        label = str(raw_label).strip()
        if code is None or not label:
            raise ValueError(f"question {question_index} has an invalid selection")
        if code in selections:
            raise ValueError(f"question {question_index} has duplicate selection code {code}")
        selections[code] = label
    return selections


def normalize_questions(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        raise ValueError("QA JSON must be an object")
    groups = raw.get("Q&A")
    if not isinstance(groups, list):
        raise ValueError("QA JSON must contain a 'Q&A' array")
    if not groups:
        return []

    pairs: list[tuple[Any, Any, Any, Any]] = []
    for group_index, group in enumerate(groups, start=1):
        if not isinstance(group, dict):
            raise ValueError(f"Q&A entry {group_index} must be an object")
        items = group.get("questions")
        selection_sets = group.get("selections")
        answers = group.get("answers")
        if not isinstance(items, list) or not isinstance(selection_sets, list) or not isinstance(answers, list):
            raise ValueError(
                f"Q&A entry {group_index} must contain 'questions', 'selections', and 'answers' arrays"
            )
        if len(items) != len(selection_sets) or len(items) != len(answers):
            raise ValueError(f"Q&A entry {group_index} has unequal question, selection, and answer counts")
        frame_window = group.get("frame_window")
        question_type = group.get("question_type")
        gt_episode_length = group.get("gt_episode_length")
        pairs.extend(
            (item, selections, answer, frame_window, question_type, gt_episode_length)
            for item, selections, answer in zip(items, selection_sets, answers)
        )

    if not pairs:
        raise ValueError("a non-empty Q&A array must contain at least 1 question/answer pair")

    questions: list[dict[str, Any]] = []
    for index, (
        item,
        selection_item,
        answer_item,
        group_frame_window,
        group_question_type,
        group_gt_episode_length,
    ) in enumerate(pairs, start=1):
        if isinstance(item, str):
            question_text = item
            raw_item: dict[str, Any] = {}
        elif isinstance(item, dict):
            raw_item = item
            question_text = str(item.get("question") or item.get("q") or "").strip()
        else:
            continue
        if not question_text:
            raise ValueError(f"question {index} must not be empty")
        question_id = str(raw_item.get("id") or raw_item.get("qid") or f"q{index}")
        selections = normalize_selections(selection_item, index)
        if isinstance(answer_item, dict):
            answer_item = answer_item.get("answer")
        raw_expected = answer_item if isinstance(answer_item, list) else [answer_item]
        expected_answers: list[str] = []
        for value in raw_expected:
            code = normalize_choice_code(value)
            if code not in selections:
                raise ValueError(f"answer {index} must match one of its selection codes")
            if code not in expected_answers:
                expected_answers.append(code)
        if not expected_answers:
            raise ValueError(f"answer {index} must contain at least one selection code")
        expected: str | list[str] = (
            expected_answers[0] if len(expected_answers) == 1 else expected_answers
        )
        frame_window = raw_item.get("frame_window", group_frame_window)
        if frame_window is not None:
            if isinstance(frame_window, list) and len(frame_window) == 2:
                frame_window = {
                    "start_frame": frame_window[0],
                    "end_frame": frame_window[1],
                    "source_num_frames": group_gt_episode_length,
                }
            elif isinstance(frame_window, dict):
                frame_window = dict(frame_window)
            else:
                raise ValueError(f"question {index} frame_window must be [start, end] or an object")
            required = {"start_frame", "end_frame", "source_num_frames"}
            if not required.issubset(frame_window):
                raise ValueError(
                    f"question {index} frame_window requires start/end and gt_episode_length"
                )
            start = frame_window["start_frame"]
            end = frame_window["end_frame"]
            source_count = frame_window["source_num_frames"]
            if not all(isinstance(value, int) for value in (start, end, source_count)):
                raise ValueError(f"question {index} frame_window frame values must be integers")
            if source_count < 1 or start < 0 or end < start or end >= source_count:
                raise ValueError(f"question {index} has an invalid frame_window range")
        question_type = raw_item.get("question_type", group_question_type)
        if question_type is None:
            question_type = "window" if frame_window is not None else "general"
        question_type = str(question_type).strip().lower()
        if question_type not in {"view", "general", "window"}:
            raise ValueError(f"question {index} question_type must be view, general, or window")
        if question_type == "window" and frame_window is None:
            raise ValueError(f"window question {index} requires frame_window")
        if question_type in {"view", "general"} and frame_window is not None:
            raise ValueError(f"{question_type} question {index} must not have frame_window")
        questions.append(
            {
                "id": question_id,
                "question": question_text,
                "selections": selections,
                "expected_answer": expected,
                "expected_answers": expected_answers,
                "source": raw_item,
                "frame_window": frame_window,
                "question_type": question_type,
            }
        )
    return questions


def load_episode_questions(episode_dir: Path, qa_root: Path) -> tuple[Path | None, list[dict[str, Any]], str | None]:
    qa_path = qa_path_for_episode(episode_dir, qa_root)
    if qa_path is None:
        return None, [], "missing_qa_json"
    try:
        return qa_path, normalize_questions(load_json_or_empty(qa_path)), None
    except Exception as exc:
        return qa_path, [], f"invalid_qa_json: {exc}"


def validate_episode_questions(episode_dirs: list[Path], qa_root: Path) -> None:
    errors = []
    for episode_dir in episode_dirs:
        qa_path, questions, qa_error = load_episode_questions(episode_dir, qa_root)
        if qa_error:
            errors.append(f"{episode_dir.name}: {qa_error} ({qa_path or 'no qa path'})")
    if errors:
        preview = "\n".join(f"  - {item}" for item in errors[:20])
        remaining = len(errors) - min(len(errors), 20)
        if remaining:
            preview += f"\n  ... and {remaining} more"
        raise RuntimeError(
            "QA validation failed. Every selected episode must have a non-empty QA JSON before running.\n"
            f"{preview}"
        )


def frame_number(path: Path, fallback: int) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else fallback


def view_frame_map(episode_dir: Path, view: str) -> dict[int, Path]:
    roots = [episode_dir / view / "frames", episode_dir / view / "video", episode_dir / view]
    paths: list[Path] = []
    for root in roots:
        if root.is_dir():
            paths.extend(item for item in root.iterdir() if item.suffix.lower() in IMAGE_EXTENSIONS)
            if paths:
                break
    out: dict[int, Path] = {}
    for fallback, path in enumerate(sorted(paths)):
        out[frame_number(path, fallback)] = path
    return out


def uniform_sample(values: list[int], count: int) -> list[int]:
    if not values or count <= 0:
        return []
    if len(values) <= count:
        return values
    if count == 1:
        return [values[len(values) // 2]]
    return [values[round(idx * (len(values) - 1) / (count - 1))] for idx in range(count)]


def resize_image(image: Any, max_edge: int) -> Any:
    if max_edge <= 0:
        return image.convert("RGB")
    image = image.convert("RGB")
    width, height = image.size
    edge = max(width, height)
    if edge <= max_edge:
        return image
    scale = max_edge / edge
    resample = getattr(getattr(image, "Resampling", image), "LANCZOS", 1)
    return image.resize((max(1, int(width * scale)), max(1, int(height * scale))), resample)


def load_episode_frame_maps(episode_dir: Path, views: list[str]) -> dict[str, dict[int, Path]]:
    return {view: view_frame_map(episode_dir, view) for view in views}


def load_cached_frame(path: Path, image_max_edge: int, cache: dict[Path, Any]) -> Any:
    cached = cache.get(path)
    if cached is not None:
        return cached
    from PIL import Image

    with Image.open(path) as image:
        frame = resize_image(image, image_max_edge)
    cache[path] = frame
    return frame


def load_episode_videos(
    frame_maps: dict[str, dict[int, Path]],
    episode_dir: Path,
    views: list[str],
    num_frames: int,
    image_max_edge: int,
    frame_cache: dict[Path, Any],
) -> tuple[list[Any], dict[str, Any]]:
    available_sets = [set(mapping) for mapping in frame_maps.values() if mapping]
    if not available_sets:
        raise FileNotFoundError(f"no image frames found in {episode_dir}")
    common = sorted(set.intersection(*available_sets)) if len(available_sets) == len(views) else []
    frame_indices = uniform_sample(common or sorted(set.union(*available_sets)), num_frames)

    videos: list[dict[str, Any]] = []
    sources: dict[str, Any] = {"frame_indices": frame_indices, "views": {}}
    for view in views:
        paths = [frame_maps[view].get(frame_idx) for frame_idx in frame_indices]
        if any(path is None for path in paths):
            raise FileNotFoundError(f"view {view!r} is missing one or more synchronized frames in {episode_dir}")
        frames = [load_cached_frame(path, image_max_edge, frame_cache) for path in paths if path is not None]
        if not frames:
            raise FileNotFoundError(f"no sampled frames found for view {view!r} in {episode_dir}")
        label = VIEW_LABELS.get(view, f"{view} view")
        videos.append({"view": view, "label": label, "frames": frames})
        sources["views"][view] = {
            "label": label,
            "frames": [str(path) for path in paths],
        }
    return videos, sources


def load_phase_windows(
    episode: str,
    phase_window_root: Path,
    questions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    path = phase_window_root / f"{episode}.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    source_num_frames = int(manifest.get("source_num_frames") or 0)
    windows = manifest.get("windows")
    if source_num_frames < 1 or not isinstance(windows, list):
        raise ValueError(f"invalid phase window manifest: {path}")
    for window in windows:
        qa_index = int(window.get("qa_index", -1))
        if qa_index < 0 or qa_index >= len(questions):
            raise ValueError(f"phase window QA index is out of range: {path}")
        if window.get("question") != questions[qa_index]["question"]:
            raise ValueError(f"phase window question does not match QA index {qa_index}: {path}")
    return manifest


def align_phase_window(
    start_frame: int,
    end_frame: int,
    source_num_frames: int,
    target_num_frames: int,
) -> tuple[int, int]:
    if source_num_frames < 1 or target_num_frames < 1:
        raise ValueError("source and target frame counts must be positive")
    source_last = max(source_num_frames - 1, 1)
    target_last = target_num_frames - 1
    start = round(max(0, min(start_frame, source_num_frames - 1)) / source_last * target_last)
    end = round(max(0, min(end_frame, source_num_frames - 1)) / source_last * target_last)
    return min(start, end), max(start, end)


def load_phase_window_videos(
    frame_maps: dict[str, dict[int, Path]],
    episode_dir: Path,
    views: list[str],
    questions: list[dict[str, Any]],
    manifest: dict[str, Any] | None,
    num_frames: int,
    image_max_edge: int,
    frame_cache: dict[Path, Any],
) -> tuple[list[Any], list[dict[str, Any]]]:
    if manifest is None:
        return [], []
    source_num_frames = int(manifest["source_num_frames"])
    videos: list[Any] = []
    window_sources: list[dict[str, Any]] = []
    for window in manifest["windows"]:
        qa_index = int(window["qa_index"])
        source_start = int(window["start_frame"])
        source_end = int(window["end_frame"])
        source_record: dict[str, Any] = {
            "qa_index": qa_index,
            "question": questions[qa_index]["question"],
            "source_window": [source_start, source_end],
            "source_num_frames": source_num_frames,
            "views": {},
        }
        for view in views:
            ordered_paths = [frame_maps[view][key] for key in sorted(frame_maps[view])]
            if not ordered_paths:
                raise FileNotFoundError(f"no frames found for view {view!r} in {episode_dir}")
            aligned_start, aligned_end = align_phase_window(
                source_start, source_end, source_num_frames, len(ordered_paths)
            )
            window_positions = list(range(aligned_start, aligned_end + 1))
            sample_count = min(num_frames, len(window_positions))
            sampled_positions = uniform_sample(window_positions, sample_count)
            sampled_paths = [ordered_paths[position] for position in sampled_positions]
            frames = [load_cached_frame(path, image_max_edge, frame_cache) for path in sampled_paths]
            view_label = VIEW_LABELS.get(view, f"{view} view")
            videos.append(
                {
                    "view": view,
                    "label": f"Question {qa_index + 1} phase window - {view_label}",
                    "frames": frames,
                }
            )
            source_record["views"][view] = {
                "target_num_frames": len(ordered_paths),
                "aligned_window_positions": [aligned_start, aligned_end],
                "sampled_positions": sampled_positions,
                "frames": [str(path) for path in sampled_paths],
            }
        window_sources.append(source_record)
    return videos, window_sources


def load_embedded_window_videos(
    frame_maps: dict[str, dict[int, Path]],
    episode_dir: Path,
    views: list[str],
    questions: list[dict[str, Any]],
    num_frames: int,
    image_max_edge: int,
    frame_cache: dict[Path, Any],
) -> tuple[dict[int, list[Any]], list[dict[str, Any]]]:
    videos_by_question: dict[int, list[Any]] = {}
    window_sources: list[dict[str, Any]] = []
    for qa_index, question in enumerate(questions):
        if question.get("question_type") != "window":
            continue
        window = question.get("frame_window")
        if window is None:
            continue
        source_start = int(window["start_frame"])
        source_end = int(window["end_frame"])
        source_num_frames = int(window["source_num_frames"])
        source_record: dict[str, Any] = {
            "qa_index": qa_index,
            "question": question["question"],
            "source_window": [source_start, source_end],
            "source_num_frames": source_num_frames,
            "phase": window.get("phase"),
            "arm": window.get("arm"),
            "views": {},
        }
        for view in views:
            ordered_paths = [frame_maps[view][key] for key in sorted(frame_maps[view])]
            if not ordered_paths:
                raise FileNotFoundError(f"no frames found for view {view!r} in {episode_dir}")
            aligned_start, aligned_end = align_phase_window(
                source_start, source_end, source_num_frames, len(ordered_paths)
            )
            positions = list(range(aligned_start, aligned_end + 1))
            sampled_positions = uniform_sample(positions, min(num_frames, len(positions)))
            sampled_paths = [ordered_paths[position] for position in sampled_positions]
            frames = [load_cached_frame(path, image_max_edge, frame_cache) for path in sampled_paths]
            view_label = VIEW_LABELS.get(view, f"{view} view")
            videos_by_question.setdefault(qa_index, []).append(
                {
                    "view": view,
                    "label": f"Question {qa_index + 1} frame window - {view_label}",
                    "frames": frames,
                }
            )
            source_record["views"][view] = {
                "target_num_frames": len(ordered_paths),
                "aligned_window_positions": [aligned_start, aligned_end],
                "sampled_positions": sampled_positions,
                "frames": [str(path) for path in sampled_paths],
            }
        window_sources.append(source_record)
    return videos_by_question, window_sources


def build_watch_messages(videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    watch_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Watch the synchronized whole-video overview and question-specific frame-window clips carefully. "
                "Each video clip is explicitly labeled with its camera viewpoint and, when applicable, question number. "
                "Frames within every video are ordered from early to late. "
                "General questions use the whole-video overview. Each window question uses its matching labeled "
                "frame-window clip. Use evidence across all three viewpoints. Do not answer yet."
            ),
        }
    ]
    for video in videos:
        watch_content.append({"type": "text", "text": f'{video["label"]} video:'})
        watch_content.append(
            {
                "type": "video",
                "video": video["frames"],
                "sample_fps": 2.0,
            }
        )

    return [
        {
            "role": "system",
            "content": (
                "You are an expert robotic vision evaluator. You are provided with a video "
                "(or sequence of frames) showing three simultaneous camera views of a robot arm "
                "performing a task."
            ),
        },
        {"role": "user", "content": watch_content},
        {"role": "assistant", "content": [{"type": "text", "text": "I have watched the videos."}]},
    ]


def build_questions_message(
    questions: list[dict[str, Any]], question_numbers: list[int] | None = None
) -> dict[str, Any]:
    if question_numbers is None:
        question_numbers = list(range(1, len(questions) + 1))
    if len(question_numbers) != len(questions):
        raise ValueError("question_numbers must match questions")
    question_blocks = []
    for index, question in zip(question_numbers, questions):
        selection_lines = "\n".join(f"{code}. {label}" for code, label in question["selections"].items())
        question_blocks.append(f'{index}. {question["question"]}\n{selection_lines}')
    question_prompt = (
        "Answer all multiple-choice questions using visible evidence from the videos only. "
        "Use the whole-video overview for general questions. For each window question, base that answer on its "
        "matching question-numbered frame-window clip. "
        "For each question, briefly state the concrete visual observations supporting the answer, then give "
        "every correct selection code. Use a string when exactly one selection is correct and an array of strings "
        "when multiple selections are correct. Return ONLY one valid JSON array in the same order, using forms "
        'such as [{"reasoning":"brief visible evidence","answer":"A"}] or '
        '[{"reasoning":"brief visible evidence","answer":["B","D"]}]. '
        "Do not include markdown or text outside the JSON array. Do not invent evidence.\n"
        f"The response array must contain exactly {len(questions)} objects.\n\n"
        + "\n\n".join(question_blocks)
    )
    return {"role": "user", "content": [{"type": "text", "text": question_prompt}]}


def load_model(model_path: Path) -> tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype="auto",
        device_map="balanced",
    ).eval()
    processor = AutoProcessor.from_pretrained(str(model_path))
    return model, processor


def run_qwen_messages(model: Any, processor: Any, messages: list[dict[str, Any]], max_new_tokens: int) -> str:
    from qwen_vl_utils import process_vision_info

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, packed_video_inputs, _ = process_vision_info(
        [messages], return_video_kwargs=True, return_video_metadata=True
    )
    video_inputs = [item[0] for item in packed_video_inputs or []]
    video_metadata = [item[1] for item in packed_video_inputs or []]
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadata,
        do_sample_frames=False,
        padding=True,
        return_tensors="pt",
    )
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


def normalize_model_answer(value: Any, selections: dict[str, str]) -> str | list[str] | None:
    raw_codes = value if isinstance(value, list) else [value]
    if not raw_codes:
        return None
    codes: list[str] = []
    for raw_code in raw_codes:
        code = normalize_choice_code(raw_code)
        if code not in selections:
            return None
        if code not in codes:
            codes.append(code)
    return codes[0] if len(codes) == 1 else codes


def parse_model_responses(
    raw_text: str, questions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    try:
        values = json.loads(raw_text.strip())
    except json.JSONDecodeError:
        return [{"answer": None, "reasoning": None} for _ in questions]
    if not isinstance(values, list) or len(values) != len(questions):
        return [{"answer": None, "reasoning": None} for _ in questions]
    responses = []
    for value, question in zip(values, questions):
        if isinstance(value, dict):
            answer_value = value.get("answer", value.get("code"))
            reasoning_value = value.get("reasoning", value.get("rationale"))
            reasoning = str(reasoning_value).strip() if reasoning_value is not None else None
            if not reasoning:
                reasoning = None
        else:
            # Backward compatibility for older code-only model responses.
            answer_value = value
            reasoning = None
        answer = normalize_model_answer(answer_value, question["selections"])
        responses.append(
            {
                "answer": answer,
                "reasoning": reasoning,
            }
        )
    return responses


def parse_model_choices(raw_text: str, questions: list[dict[str, Any]]) -> list[str | list[str] | None]:
    return [item["answer"] for item in parse_model_responses(raw_text, questions)]


def score_answer(answer: str | list[str] | None, expected: str | list[str]) -> int:
    if answer is None:
        return 0
    actual_answers = answer if isinstance(answer, list) else [answer]
    expected_answers = expected if isinstance(expected, list) else [expected]
    return int(set(actual_answers) == set(expected_answers))


def evaluate_episode(
    episode_dir: Path,
    qa_root: Path,
    output_root: Path,
    args: argparse.Namespace,
    ensure_model: Callable[[], tuple[Any, Any]],
) -> dict[str, Any]:
    episode = episode_dir.name
    report_path = output_root / "episode_reports" / f"{episode}.json"
    if report_path.exists() and not args.overwrite:
        try:
            existing_result = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[resume] retry {episode}: unreadable existing report ({exc})")
        else:
            existing_status = str(existing_result.get("status") or "")
            if existing_status in {"ok", "no_questions"}:
                print(f"[resume] skip {episode}: status={existing_status}")
                return existing_result
            print(f"[resume] retry {episode}: status={existing_status or 'unknown'}")

    qa_path, questions, qa_error = load_episode_questions(episode_dir, qa_root)
    scored_question_count = sum(
        question.get("question_type") != "view" for question in questions
    )
    result: dict[str, Any] = {
        "episode": episode,
        "episode_dir": str(episode_dir),
        "qa_path": str(qa_path) if qa_path else None,
        "question_count": len(questions),
        "scored_question_count": scored_question_count,
        "score_mode": "match_expected",
        "total_score": 0,
        "max_score": scored_question_count,
        "score_ratio": None,
        "view_evaluation": None,
        "camera_view_evaluation": None,
        "answers": [],
        "status": "pending",
        "error": qa_error,
    }
    if qa_error:
        result["status"] = "qa_error"
    elif not questions:
        result["status"] = "no_questions"
    else:
        try:
            frame_maps = load_episode_frame_maps(episode_dir, args.views)
            frame_cache: dict[Path, Any] = {}
            videos, video_sources = load_episode_videos(
                frame_maps,
                episode_dir,
                args.views,
                args.num_frames,
                args.image_max_edge,
                frame_cache,
            )
            window_videos, phase_window_sources = load_embedded_window_videos(
                frame_maps,
                episode_dir,
                args.views,
                questions,
                args.num_frames,
                args.image_max_edge,
                frame_cache,
            )
            result["video_sources"] = video_sources
            result["phase_window_sources"] = phase_window_sources
            aligned_frame_windows = {
                int(source["qa_index"]): {
                    view: details["aligned_window_positions"]
                    for view, details in source["views"].items()
                }
                for source in phase_window_sources
            }
            if args.skip_model:
                result["status"] = "skip_model"
            else:
                model, processor = ensure_model()
                raw_dir = output_root / "raw_responses"
                raw_dir.mkdir(parents=True, exist_ok=True)
                raw_path = raw_dir / f"{episode}.json"
                result["raw_response_path"] = str(raw_path)
                cot_dir = args.cot_log_root
                cot_dir.mkdir(parents=True, exist_ok=True)
                cot_path = cot_dir / f"{episode}.json"
                result["cot_log_path"] = str(cot_path)
                total = 0
                answered = 0
                all_answered = 0
                answer_rows = []
                cot_rows = []
                model_responses: list[dict[str, str | None]] = [
                    {"answer": None, "reasoning": None} for _ in questions
                ]
                raw_calls: list[dict[str, Any]] = []

                overview_indices = [
                    index for index, question in enumerate(questions)
                    if question.get("question_type") in {"view", "general"}
                ]
                if overview_indices:
                    overview_questions = [questions[index] for index in overview_indices]
                    overview_numbers = [index + 1 for index in overview_indices]
                    messages = build_watch_messages(videos)
                    messages.append(
                        build_questions_message(overview_questions, overview_numbers)
                    )
                    token_budget = max(args.max_new_tokens, len(overview_questions) * 96 + 32)
                    raw_text = run_qwen_messages(model, processor, messages, token_budget)
                    parsed = parse_model_responses(raw_text, overview_questions)
                    for question_index, response in zip(overview_indices, parsed):
                        model_responses[question_index] = response
                    raw_calls.append(
                        {
                            "question_numbers": overview_numbers,
                            "input": "whole_video_overview",
                            "response": raw_text,
                            "parsed_responses": parsed,
                        }
                    )

                window_indices = [
                    index for index, question in enumerate(questions)
                    if question.get("question_type") == "window"
                ]
                for question_index in window_indices:
                    question_videos = window_videos.get(question_index)
                    if not question_videos:
                        raise ValueError(f"question {question_index + 1} has no loaded frame-window clips")
                    messages = build_watch_messages(question_videos)
                    messages.append(
                        build_questions_message(
                            [questions[question_index]], [question_index + 1]
                        )
                    )
                    token_budget = max(args.max_new_tokens, 512)
                    raw_text = run_qwen_messages(model, processor, messages, token_budget)
                    parsed = parse_model_responses(raw_text, [questions[question_index]])
                    model_responses[question_index] = parsed[0]
                    raw_calls.append(
                        {
                            "question_numbers": [question_index + 1],
                            "input": "embedded_frame_window",
                            "response": raw_text,
                            "parsed_responses": parsed,
                        }
                    )
                model_answers = [item["answer"] for item in model_responses]
                for question_index, (question, response) in enumerate(zip(questions, model_responses)):
                    answer = response["answer"]
                    score = score_answer(answer, question["expected_answer"])
                    if answer is not None:
                        all_answered += 1
                    answer_row = {
                        "id": question["id"],
                        "question_type": question["question_type"],
                        "question": question["question"],
                        "selections": question["selections"],
                        "expected_answer": question["expected_answer"],
                        "model_answer": answer,
                        "score": score,
                        "included_in_total": question["question_type"] != "view",
                    }
                    answer_rows.append(answer_row)
                    cot_rows.append(
                        {
                            **answer_row,
                            "frame_window": question["frame_window"],
                            "aligned_frame_window": aligned_frame_windows.get(question_index),
                            "reasoning": response["reasoning"],
                        }
                    )
                    if question["question_type"] == "view":
                        if result["view_evaluation"] is not None:
                            raise ValueError("each episode may contain only one view question")
                        result["view_evaluation"] = dict(answer_row)
                        result["camera_view_evaluation"] = dict(answer_row)
                    else:
                        total += score
                        if answer is not None:
                            answered += 1
                raw_path.write_text(
                    json.dumps(
                        {"calls": raw_calls, "parsed_responses": model_responses},
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                cot_path.write_text(
                    json.dumps(
                        {"episode": episode, "questions": cot_rows},
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                result["answers"] = answer_rows
                result["answered_count"] = answered
                result["all_answered_count"] = all_answered
                result["total_score"] = total
                scored_count = result["scored_question_count"]
                result["score_ratio"] = round(total / scored_count, 4) if scored_count else None
                result["status"] = "ok" if all_answered == len(questions) else "partial"
        except Exception as exc:
            result["status"] = "error"
            result["error"] = f"{type(exc).__name__}: {exc}"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def write_reports(output_root: Path, results: list[dict[str, Any]], config: dict[str, Any]) -> None:
    scored_results = [item for item in results if item.get("status") == "ok"]
    total_questions = sum(
        int(item.get("scored_question_count", item.get("question_count")) or 0)
        for item in scored_results
    )
    total_score = sum(int(item.get("total_score") or 0) for item in scored_results)
    answered = sum(int(item.get("answered_count") or 0) for item in scored_results)
    episode_score_ratios = [
        float(item["score_ratio"])
        for item in scored_results
        if item.get("score_ratio") is not None
    ]
    view_results = [
        item.get("view_evaluation") or item.get("camera_view_evaluation")
        for item in scored_results
        if isinstance(item.get("view_evaluation") or item.get("camera_view_evaluation"), dict)
    ]
    view_answered = sum(item.get("model_answer") is not None for item in view_results)
    view_score = sum(int(item.get("score") or 0) for item in view_results)
    partial_count = sum(item.get("status") == "partial" for item in results)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "episode_count": len(results),
        "evaluated_episode_count": len(scored_results),
        "excluded_partial_episode_count": partial_count,
        "total_questions": total_questions,
        "answered_questions": answered,
        "total_score": total_score,
        "score_ratio": round(total_score / total_questions, 4) if total_questions else None,
        "average_score": (
            round(sum(episode_score_ratios) / len(episode_score_ratios), 4)
            if episode_score_ratios
            else None
        ),
        "view_evaluation": {
            "question_count": len(view_results),
            "answered_questions": view_answered,
            "total_score": view_score,
            "score_ratio": (
                round(view_score / len(view_results), 4)
                if view_results
                else None
            ),
        },
        "status_counts": {},
        "episodes": results,
    }
    summary["camera_view_evaluation"] = dict(summary["view_evaluation"])
    for item in results:
        status = str(item.get("status") or "unknown")
        summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1

    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "report.json"
    json_temp = output_root / ".report.json.tmp"
    json_temp.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    json_temp.replace(json_path)

    csv_path = output_root / "report.csv"
    csv_temp = output_root / ".report.csv.tmp"
    with csv_temp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "status",
                "question_count",
                "scored_question_count",
                "answered_count",
                "total_score",
                "max_score",
                "score_ratio",
                "view_model_answer",
                "view_expected_answer",
                "view_score",
                "camera_view_model_answer",
                "camera_view_expected_answer",
                "camera_view_score",
                "qa_path",
                "error",
            ],
        )
        writer.writeheader()
        for item in results:
            row = {field: item.get(field) for field in writer.fieldnames}
            view = item.get("view_evaluation") or item.get("camera_view_evaluation")
            if isinstance(view, dict):
                row["view_model_answer"] = view.get("model_answer")
                row["view_expected_answer"] = view.get("expected_answer")
                row["view_score"] = view.get("score")
                row["camera_view_model_answer"] = view.get("model_answer")
                row["camera_view_expected_answer"] = view.get("expected_answer")
                row["camera_view_score"] = view.get("score")
            writer.writerow(row)
    csv_temp.replace(csv_path)


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.qa_root = args.qa_root.resolve()
    args.phase_window_root = args.phase_window_root.resolve()
    args.views = [str(view) for view in args.views]
    if args.num_frames < 1:
        raise ValueError("--num-frames must be at least 1")

    if args.init_qa:
        init_qa_files(args.data_root, args.qa_root, args.episodes, args.limit, args.overwrite_qa)
        return

    # Pin one physical GPU before any torch/CUDA init (parent may expose all GPUs).
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", DEFAULT_PYTORCH_ALLOC_CONF)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = VQA_ROOT / "outputs" / f"run_{output_data_label(args.data_root)}_{timestamp}"
    output_root = (args.output_root or default_output).resolve()
    args.cot_log_root = (args.cot_log_root or output_root / "cot_logs").resolve()

    episodes = discover_episodes(args.data_root, args.episodes, args.limit)
    episodes = apply_shard(episodes, args.shard_id, args.num_shards)
    validate_episode_questions(episodes, args.qa_root)
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[data] {args.data_root}")
    print(f"[qa] {args.qa_root}")
    print(f"[output] {output_root}")
    print(f"[episodes] {len(episodes)}")

    model_bundle: tuple[Any, Any] | None = None

    def ensure_model() -> tuple[Any, Any]:
        nonlocal model_bundle
        if model_bundle is None:
            print(f"[model] {args.model_path}")
            model_bundle = load_model(args.model_path)
        return model_bundle

    config = {
        "data_root": str(args.data_root),
        "qa_root": str(args.qa_root),
        "phase_window_root": str(args.phase_window_root),
        "model_path": str(args.model_path),
        "cot_log_root": str(args.cot_log_root),
        "gpu": args.gpu,
        "views": args.views,
        "num_frames": args.num_frames,
        "image_max_edge": args.image_max_edge,
        "max_new_tokens": args.max_new_tokens,
        "combine_views": args.combine_views,
        "score_mode": "match_expected",
        "skip_model": args.skip_model,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
    }
    (output_root / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    results: list[dict[str, Any]] = []
    for index, episode_dir in enumerate(episodes, start=1):
        print(f"[episode {index}/{len(episodes)}] {episode_dir.name}")
        results.append(evaluate_episode(episode_dir, args.qa_root, output_root, args, ensure_model))
        write_reports(output_root, results, config)

    final_report = load_json_or_empty(output_root / "report.json")
    print(
        "Average score: "
        f"{final_report.get('average_score')} "
        f"({final_report.get('evaluated_episode_count', 0)} complete episodes; "
        f"{final_report.get('excluded_partial_episode_count', 0)} partial episodes excluded)"
    )
    print(f"Report JSON: {output_root / 'report.json'}")
    print(f"Report CSV:  {output_root / 'report.csv'}")


if __name__ == "__main__":
    main()
