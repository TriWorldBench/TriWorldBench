"""Shared helpers for tri-view VLM consistency evaluation and reference DB."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from metrics._common.episode_sort import EP_RE, episode_sort_key, sample_sort_key
from metrics._common.frames import frame_filename

import cv2
import numpy as np
from PIL import Image

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
NEGATIVE_SAMPLES_PER_STATE = 3
MIN_DONOR_FRAME_GAP = 15


def sample_key(sample: dict[str, Any]) -> str:
    return (
        f"{sample['task']}/{sample['episode']}/"
        f"{sample['phase']}/{sample['arm']}/f{sample['sampled_frame']}"
    )


def raw_response_filename(sample_id: str, timestamp: str) -> str:
    """Flat filename for raw VLM dumps; sample_id may contain path separators."""
    safe_id = sample_id.replace("/", "__")
    return f"{safe_id}__{timestamp}.json"


def sample_dir_name(sample: dict[str, Any]) -> str:
    return f"{sample['phase']}_{sample['arm']}_f{sample['sampled_frame']}"


def reference_entry_dir(db_root: Path, sample: dict[str, Any]) -> Path:
    return db_root / sample["task"] / sample["episode"] / sample_dir_name(sample)


def discover_state_files(
    state_root: Path,
    tasks: list[str] | None = None,
    episodes: list[str] | None = None,
) -> list[Path]:
    """Discover STATE JSON files (nested task/episode or flat episodeN.json)."""
    flat_files = sorted(
        (path for path in state_root.glob("episode*.json") if EP_RE.match(path.stem)),
        key=lambda path: episode_sort_key(path.stem),
    )
    if flat_files:
        files = flat_files
    else:
        files = sorted(state_root.glob("*/*.json"))
        files = [path for path in files if path.name != "summary.json"]

    if tasks:
        task_set = set(tasks)
        files = [path for path in files if path.parent.name in task_set or path.stem in task_set]
    if episodes:
        episode_set = set(episodes)
        files = [path for path in files if path.stem in episode_set]
    return files


def collect_phase_samples(state_record: dict[str, Any]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for phase_index, phase in enumerate(state_record.get("phases", [])):
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
                "phase_index": phase_index,
                "instruction": state_record.get("instruction", ""),
                "task_category": state_record.get("task_category", ""),
                "hdf5_path": state_record.get("hdf5_path", ""),
            }
        )
    return samples


def frame_path_from_gt(
    frames_root: Path,
    task: str,
    episode: str,
    view: str,
    frame_idx: int,
) -> Path:
    frame_name = frame_filename(frame_idx)
    candidates = [
        frames_root / episode / view / subdir / frame_name
        for subdir in ("frames", "video")
    ]
    candidates.extend(
        frames_root / task / episode / view / subdir / frame_name
        for subdir in ("frames", "video")
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return frames_root / episode / view / "frames" / frame_name


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


def load_triplet_at_frame(
    task: str,
    episode: str,
    frame_idx: int,
    frames_root: Path,
    dataset_root: Path | None = None,
) -> tuple[dict[str, Image.Image], dict[str, str]]:
    images: dict[str, Image.Image] = {}
    sources: dict[str, str] = {}

    for view in VIEW_DIRS:
        gt_path = frame_path_from_gt(frames_root, task, episode, view, frame_idx)
        if gt_path.is_file():
            images[view] = Image.open(gt_path).convert("RGB")
            sources[view] = str(gt_path)
            continue
        if dataset_root is None:
            raise FileNotFoundError(f"Missing frame {frame_idx} for {episode}/{view}: {gt_path}")
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


def load_triplet_images(
    sample: dict[str, Any],
    frames_root: Path,
    dataset_root: Path | None = None,
) -> tuple[dict[str, Image.Image], dict[str, str]]:
    return load_triplet_at_frame(
        sample["task"],
        sample["episode"],
        int(sample["sampled_frame"]),
        frames_root,
        dataset_root,
    )


def combine_views(images: dict[str, Image.Image]) -> Image.Image:
    arrays = [np.asarray(images[view]) for view in VIEW_DIRS]
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


def active_view_for_sample(sample: dict[str, Any]) -> str:
    arm = sample.get("arm", "")
    return ARM_TO_VIEW.get(arm, "right")


def frame_gap(current: dict[str, Any], donor: dict[str, Any]) -> int:
    return abs(int(donor["sampled_frame"]) - int(current["sampled_frame"]))


def is_adjacent_phase(current: dict[str, Any], donor: dict[str, Any]) -> bool:
    cur_idx = current.get("phase_index")
    don_idx = donor.get("phase_index")
    if cur_idx is None or don_idx is None:
        return False
    return abs(int(cur_idx) - int(don_idx)) <= 1


def is_valid_negative_donor(
    current: dict[str, Any],
    donor: dict[str, Any],
    min_frame_gap: int,
) -> bool:
    if sample_key(current) == sample_key(donor):
        return False
    if donor["phase"] == current["phase"]:
        return False
    if is_adjacent_phase(current, donor):
        return False
    if frame_gap(current, donor) < min_frame_gap:
        return False
    return True


def donor_priority(current: dict[str, Any], donor: dict[str, Any]) -> tuple[float, float]:
    arm_diff = 1.0 if donor["arm"] != current["arm"] else 0.0
    return (arm_diff, float(frame_gap(current, donor)))


def select_negative_donors(
    current: dict[str, Any],
    episode_samples: list[dict[str, Any]],
    num_negatives: int = NEGATIVE_SAMPLES_PER_STATE,
    min_frame_gap: int = MIN_DONOR_FRAME_GAP,
) -> list[dict[str, Any]]:
    cur_key = sample_key(current)
    candidates = [item for item in episode_samples if sample_key(item) != cur_key]

    def pick_from(pool: list[dict[str, Any]], already: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen = {sample_key(item) for item in already}
        ranked = sorted(pool, key=lambda donor: donor_priority(current, donor), reverse=True)
        picked: list[dict[str, Any]] = []
        for donor in ranked:
            donor_key = sample_key(donor)
            if donor_key in seen:
                continue
            picked.append(donor)
            seen.add(donor_key)
            if len(already) + len(picked) >= num_negatives:
                break
        return picked

    strict_pool = [
        donor for donor in candidates if is_valid_negative_donor(current, donor, min_frame_gap)
    ]
    selected = pick_from(strict_pool, [])

    if len(selected) < num_negatives:
        relaxed_pool = [
            donor
            for donor in candidates
            if donor["phase"] != current["phase"]
            and not is_adjacent_phase(current, donor)
            and sample_key(donor) not in {sample_key(s) for s in selected}
        ]
        selected.extend(pick_from(relaxed_pool, selected))

    return selected[:num_negatives]


def make_first_frame_donor(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "donor_kind": "first_frame",
        "task": sample["task"],
        "episode": sample["episode"],
        "phase": "first_frame",
        "arm": "",
        "description": "episode initial frame at t=0",
        "sampled_frame": 0,
        "phase_index": -1,
    }


def select_negative_mix_plans(
    current: dict[str, Any],
    episode_samples: list[dict[str, Any]],
    num_negatives: int = NEGATIVE_SAMPLES_PER_STATE,
    min_frame_gap: int = MIN_DONOR_FRAME_GAP,
) -> list[dict[str, Any]]:
    active_view = active_view_for_sample(current)
    phase_donors = select_negative_donors(current, episode_samples, num_negatives, min_frame_gap)

    plans: list[dict[str, Any]] = []

    if phase_donors and len(plans) < num_negatives:
        plans.append(
            {
                "replace_view": active_view,
                "donor_kind": "phase",
                "donor": phase_donors[0],
            }
        )

    if phase_donors and len(plans) < num_negatives:
        head_donor = phase_donors[1] if len(phase_donors) > 1 else phase_donors[0]
        plans.append(
            {
                "replace_view": "head",
                "donor_kind": "phase",
                "donor": head_donor,
            }
        )

    if len(plans) < num_negatives and int(current["sampled_frame"]) >= min_frame_gap:
        replaced_views = {plan["replace_view"] for plan in plans}
        ff_view = "head" if "head" not in replaced_views else active_view
        plans.append(
            {
                "replace_view": ff_view,
                "donor_kind": "first_frame",
                "donor": make_first_frame_donor(current),
            }
        )
    elif len(plans) < num_negatives and len(phase_donors) > 2:
        plans.append(
            {
                "replace_view": active_view,
                "donor_kind": "phase",
                "donor": phase_donors[2],
            }
        )

    return plans[:num_negatives]


def load_donor_triplet(
    plan: dict[str, Any],
    frames_root: Path,
    dataset_root: Path | None,
) -> tuple[dict[str, Image.Image], dict[str, str], dict[str, Any]]:
    donor = plan["donor"]
    if plan["donor_kind"] == "first_frame":
        images, sources = load_triplet_at_frame(
            donor["task"],
            donor["episode"],
            0,
            frames_root,
            dataset_root,
        )
    else:
        images, sources = load_triplet_images(donor, frames_root, dataset_root)
    return images, sources, donor


def negative_mix_caption(plan_meta: dict[str, Any]) -> str:
    view_label = VIEW_LABELS.get(plan_meta["replaced_view"], plan_meta["replaced_view"])
    if plan_meta.get("donor_kind") == "first_frame":
        return f"不一致，{view_label} 来自 episode 首帧（t=0）"
    return f"不一致，{view_label} 来自其他动作阶段"


def build_mixed_triplet(
    current_images: dict[str, Image.Image],
    donor_images: dict[str, Image.Image],
    replace_view: str,
) -> dict[str, Image.Image]:
    mixed = dict(current_images)
    mixed[replace_view] = donor_images[replace_view]
    return mixed


def load_reference_bundle(db_root: Path, sample: dict[str, Any]) -> dict[str, Any] | None:
    entry_dir = reference_entry_dir(db_root, sample)
    positive_path = entry_dir / "positive.jpg"
    if not positive_path.is_file():
        return None

    negative_paths = sorted(entry_dir.glob("negative_*.jpg"))
    if not negative_paths:
        return None

    meta_path = entry_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}

    return {
        "entry_dir": str(entry_dir),
        "positive": Image.open(positive_path).convert("RGB"),
        "negatives": [Image.open(path).convert("RGB") for path in negative_paths[:NEGATIVE_SAMPLES_PER_STATE]],
        "meta": meta,
    }


def load_phase_samples_from_state_root(
    state_root: Path,
    episodes: list[str] | None = None,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for state_path in discover_state_files(state_root, episodes=episodes):
        record = json.loads(state_path.read_text(encoding="utf-8"))
        samples.extend(collect_phase_samples(record))
    samples.sort(key=sample_sort_key)
    return samples
