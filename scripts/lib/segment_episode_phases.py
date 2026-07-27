"""Segment robot manipulation episodes into coarse phases from HDF5 trajectories.

Vendored for Triworld so STATE generation runs without
external Benchmark_evaluation dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

GRIPPER_CLOSED = 0.25
GRIPPER_OPEN = 0.7
GRIPPER_OPEN_START = 0.3
MOTION_THRESHOLD = 0.002
GRIPPER_TREND_EPS = 0.01
ACTIVE_MOTION_EPS = 1e-6
RELEASE_OPEN_FRAMES = 6

TASK_CATEGORIES: dict[str, str] = {
    "adjust_bottle": "pick_place",
    "beat_block_hammer": "tool_use",
    "blocks_ranking_rgb": "arrange",
    "blocks_ranking_size": "arrange",
    "click_alarmclock": "click",
    "click_bell": "click",
    "dump_bin_bigbin": "dump",
    "grab_roller": "pick_place",
    "handover_block": "handover",
    "handover_mic": "handover",
    "hanging_mug": "pick_place",
    "lift_pot": "pick_place",
    "move_can_pot": "pick_place",
    "move_pillbottle_pad": "pick_place",
    "move_playingcard_away": "pick_place",
    "move_stapler_pad": "pick_place",
    "open_laptop": "articulate",
    "open_microwave": "articulate",
    "pick_diverse_bottles": "pick_place",
    "pick_dual_bottles": "pick_place",
    "place_a2b_left": "pick_place",
    "place_a2b_right": "pick_place",
    "place_bread_basket": "pick_place",
    "place_bread_skillet": "pick_place",
    "place_burger_fries": "pick_place",
    "place_can_basket": "pick_place",
    "place_cans_plasticbox": "pick_place",
    "place_container_plate": "pick_place",
    "place_dual_shoes": "pick_place",
    "place_empty_cup": "pick_place",
    "place_fan": "pick_place",
    "place_mouse_pad": "pick_place",
    "place_object_basket": "pick_place",
    "place_object_scale": "pick_place",
    "place_object_stand": "pick_place",
    "place_phone_stand": "pick_place",
    "place_shoe": "pick_place",
    "press_stapler": "pick_place",
    "put_bottles_dustbin": "pick_place",
    "put_object_cabinet": "pick_place",
    "rotate_qrcode": "pick_place",
    "scan_object": "pick_place",
    "shake_bottle": "pick_place",
    "shake_bottle_horizontally": "pick_place",
    "stack_blocks_three": "stack",
    "stack_blocks_two": "stack",
    "stack_bowls_three": "stack",
    "stack_bowls_two": "stack",
    "stamp_seal": "pick_place",
    "turn_switch": "pick_place",
}

PHASE_DESCRIPTIONS = {
    ("idle", "left"): "left arm idle with gripper open",
    ("idle", "right"): "right arm idle with gripper open",
    ("approach", "left"): "left arm approaches target with gripper open",
    ("approach", "right"): "right arm approaches target with gripper open",
    ("grasp_close", "left"): "left gripper closes to grasp/contact",
    ("grasp_close", "right"): "right gripper closes to grasp/contact",
    ("hold", "left"): "left arm holds object with gripper closed",
    ("hold", "right"): "right arm holds object with gripper closed",
    ("manipulate", "left"): "left arm manipulates object while grasping",
    ("manipulate", "right"): "right arm manipulates object while grasping",
    ("handover_move", "left"): "left arm moves object toward the other arm for handover",
    ("handover_move", "right"): "right arm moves object toward the other arm for handover",
    ("execute_action", "left"): "left arm executes task-specific action while grasping",
    ("execute_action", "right"): "right arm executes task-specific action while grasping",
    ("adjust", "left"): "left arm adjusts pose",
    ("adjust", "right"): "right arm adjusts pose",
    ("release_open", "left"): "left gripper opens to release object",
    ("release_open", "right"): "right gripper opens to release object",
    ("complete", "left"): "left arm reaches end state",
    ("complete", "right"): "right arm reaches end state",
}


def load_hdf5_arrays(hdf5_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(hdf5_path, "r") as handle:
        return {
            "left_gripper": np.asarray(handle["endpose/left_gripper"], dtype=np.float64),
            "right_gripper": np.asarray(handle["endpose/right_gripper"], dtype=np.float64),
            "left_endpose": np.asarray(handle["endpose/left_endpose"], dtype=np.float64)[:, :3],
            "right_endpose": np.asarray(handle["endpose/right_endpose"], dtype=np.float64)[:, :3],
        }


def xyz_motion(endpose_xyz: np.ndarray) -> np.ndarray:
    delta = np.linalg.norm(np.diff(endpose_xyz, axis=0), axis=1)
    return np.concatenate([[0.0], delta])


def is_arm_active(gripper: np.ndarray, motion: np.ndarray) -> bool:
    return float(np.min(gripper)) < 1.0 - 1e-6 or float(np.max(motion)) > ACTIVE_MOTION_EPS


def find_close_starts(gripper: np.ndarray) -> list[int]:
    starts: list[int] = []
    for idx in range(1, len(gripper)):
        if gripper[idx - 1] >= GRIPPER_OPEN and gripper[idx] < GRIPPER_OPEN:
            starts.append(idx)
            continue
        if (
            GRIPPER_CLOSED < gripper[idx - 1] < GRIPPER_OPEN
            and gripper[idx] < gripper[idx - 1] - GRIPPER_TREND_EPS
            and abs(gripper[idx - 1] - gripper[idx - 2]) <= GRIPPER_TREND_EPS
        ):
            starts.append(idx)
    return starts


def find_open_starts(gripper: np.ndarray) -> list[int]:
    starts: list[int] = []
    for idx in range(1, len(gripper)):
        if gripper[idx - 1] <= GRIPPER_CLOSED and gripper[idx] > GRIPPER_OPEN_START:
            starts.append(idx)
    return starts


def close_frames(gripper: np.ndarray) -> np.ndarray:
    return np.flatnonzero(gripper <= GRIPPER_CLOSED)


def close_frame_ranges(close_idx: np.ndarray) -> list[list[int]]:
    if close_idx.size == 0:
        return []
    ranges: list[list[int]] = []
    start = int(close_idx[0])
    prev = int(close_idx[0])
    for value in close_idx[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        ranges.append([start, prev])
        start = value
        prev = value
    ranges.append([start, prev])
    return ranges


def gripper_summary(gripper: np.ndarray) -> dict[str, Any]:
    closed_idx = close_frames(gripper)
    return {
        "min": float(np.min(gripper)),
        "max": float(np.max(gripper)),
        "close_frames": [int(v) for v in closed_idx.tolist()],
        "close_frame_ranges": close_frame_ranges(closed_idx),
        "num_close_frames": int(closed_idx.size),
    }


def gripper_trend(gripper: np.ndarray) -> np.ndarray:
    trend = np.zeros(len(gripper), dtype=np.int8)
    for idx in range(1, len(gripper)):
        delta = gripper[idx] - gripper[idx - 1]
        if delta < -GRIPPER_TREND_EPS:
            trend[idx] = -1
        elif delta > GRIPPER_TREND_EPS:
            trend[idx] = 1
    return trend


def grasp_close_end(gripper: np.ndarray, start: int) -> int:
    end = start
    while end + 1 < len(gripper) and gripper[end + 1] < gripper[end] - 1e-6:
        end += 1
    if GRIPPER_CLOSED < gripper[end] < GRIPPER_OPEN:
        return end
    for idx in range(start, len(gripper)):
        if gripper[idx] <= 0.35:
            return idx
    return end


def finishing_arm(close_starts: dict[str, list[int]], open_starts: dict[str, list[int]]) -> str | None:
    last_open: dict[str, int] = {}
    for arm, starts in open_starts.items():
        if starts:
            last_open[arm] = starts[-1]
    if last_open:
        return max(last_open, key=last_open.get)
    last_close: dict[str, int] = {}
    for arm, starts in close_starts.items():
        if starts:
            last_close[arm] = starts[-1]
    if last_close:
        return max(last_close, key=last_close.get)
    return None


def apply_complete_phase(
    labels: list[str],
    gripper: np.ndarray,
    motion: np.ndarray,
    arm: str,
    finishing: str | None,
) -> None:
    if finishing is None or arm != finishing:
        return

    last_approach_end = -1
    for idx, label in enumerate(labels):
        if label == "approach":
            last_approach_end = idx

    if last_approach_end >= 0 and last_approach_end < len(labels) - 1:
        tail_start = last_approach_end + 1
        if tail_start < len(labels) and len(labels) - tail_start >= 2:
            if np.max(motion[tail_start:]) < MOTION_THRESHOLD:
                for idx in range(tail_start, len(labels)):
                    if gripper[idx] >= GRIPPER_OPEN_START:
                        labels[idx] = "complete"
                return

    last_release_end = -1
    for idx, label in enumerate(labels):
        if label == "release_open":
            last_release_end = idx
    if last_release_end < 0 or last_release_end >= len(labels) - 1:
        return

    tail_start = last_release_end + 1
    if np.max(motion[tail_start:]) >= MOTION_THRESHOLD:
        if last_approach_end > last_release_end:
            tail_start = last_approach_end + 1
        else:
            return

    for idx in range(tail_start, len(labels)):
        if gripper[idx] >= GRIPPER_OPEN_START:
            labels[idx] = "complete"


def mark_grasp_close(labels: list[str], gripper: np.ndarray) -> list[tuple[int, float]]:
    events: list[tuple[int, float]] = []
    for start in find_close_starts(gripper):
        end = grasp_close_end(gripper, start)
        for idx in range(start, end + 1):
            labels[idx] = "grasp_close"
        events.append((start, float(gripper[start])))
    return events


def mark_release_open(labels: list[str], gripper: np.ndarray) -> list[tuple[int, float]]:
    events: list[tuple[int, float]] = []
    for start in find_open_starts(gripper):
        end = min(len(labels) - 1, start + RELEASE_OPEN_FRAMES - 1)
        for idx in range(start, end + 1):
            labels[idx] = "release_open"
        events.append((start, float(gripper[start])))
    return events


def closed_motion_phase(
    task_category: str,
    arm: str,
    delivering_arm: str | None,
    motion_value: float,
) -> str:
    if motion_value < MOTION_THRESHOLD:
        return "hold"
    if task_category == "tool_use":
        return "execute_action"
    if task_category == "handover" and delivering_arm == arm:
        return "handover_move"
    return "manipulate"


def open_motion_phase(motion_value: float) -> str:
    return "approach" if motion_value >= MOTION_THRESHOLD else "idle"


def label_arm_frames(
    gripper: np.ndarray,
    motion: np.ndarray,
    task_category: str,
    arm: str,
    delivering_arm: str | None,
    finishing: str | None,
) -> list[str]:
    labels = [""] * len(gripper)
    mark_grasp_close(labels, gripper)
    mark_release_open(labels, gripper)
    trend = gripper_trend(gripper)

    for idx, value in enumerate(gripper):
        if labels[idx]:
            continue
        if GRIPPER_CLOSED < value < GRIPPER_OPEN:
            if trend[idx] == 0 and (idx == 0 or trend[idx - 1] == 0):
                labels[idx] = "adjust"
            elif trend[idx] < 0:
                labels[idx] = "grasp_close"
            elif trend[idx] > 0:
                labels[idx] = "release_open"
            else:
                labels[idx] = "adjust"
            continue
        if value <= GRIPPER_CLOSED:
            labels[idx] = closed_motion_phase(task_category, arm, delivering_arm, motion[idx])
            continue
        labels[idx] = open_motion_phase(motion[idx])

    apply_complete_phase(labels, gripper, motion, arm, finishing)
    return labels


def build_phase_dict(
    phase: str,
    arm: str,
    start: int,
    end: int,
    gripper: np.ndarray,
    motion: np.ndarray,
) -> dict[str, Any]:
    result = {
        "phase": phase,
        "arm": arm,
        "start_frame": int(start),
        "end_frame": int(end),
        "duration_frames": int(end - start + 1),
        "gripper_start": float(gripper[start]),
        "gripper_end": float(gripper[end]),
        "mean_motion": float(np.mean(motion[start : end + 1])),
        "description": PHASE_DESCRIPTIONS.get((phase, arm), f"{arm} arm {phase}"),
    }
    if phase not in {"idle", "approach"}:
        result["sampled_frame"] = int(round((start + end) / 2))
    return result


def merge_phase_segments(
    labels: list[str],
    gripper: np.ndarray,
    motion: np.ndarray,
    arm: str,
) -> list[dict[str, Any]]:
    if not labels:
        return []
    phases: list[dict[str, Any]] = []
    start = 0
    current = labels[0]
    for idx in range(1, len(labels)):
        if labels[idx] == current:
            continue
        phases.append(build_phase_dict(current, arm, start, idx - 1, gripper, motion))
        start = idx
        current = labels[idx]
    phases.append(build_phase_dict(current, arm, start, len(labels) - 1, gripper, motion))
    return phases


def choose_delivering_arm(
    task_category: str,
    active_arms: list[str],
    close_starts: dict[str, list[int]],
) -> str | None:
    if task_category != "handover" or len(active_arms) != 2:
        return None
    first: dict[str, int] = {}
    for arm in active_arms:
        starts = close_starts.get(arm, [])
        if starts:
            first[arm] = starts[0]
    if len(first) != 2:
        return None
    return min(first, key=first.get)


def load_scene_info(scene_info_root: Path | None, task: str, episode: str) -> dict[str, Any]:
    default = {
        "cluttered_table_info": [],
        "texture_info": {"wall_texture": None, "table_texture": None},
        "info": {},
    }
    if scene_info_root is None:
        return default
    ref_path = scene_info_root / task / f"{episode}.json"
    if not ref_path.is_file():
        return default
    payload = json.loads(ref_path.read_text(encoding="utf-8"))
    return payload.get("scene_info", default)


def segment_hdf5(
    hdf5_path: Path,
    task: str,
    global_episode: str,
    local_episode: str,
    instruction: str,
    scene_info_root: Path | None = None,
) -> dict[str, Any]:
    """Build a STATE record dict from one HDF5 trajectory file."""
    arrays = load_hdf5_arrays(hdf5_path)
    num_frames = len(arrays["left_gripper"])
    task_category = TASK_CATEGORIES.get(task, "pick_place")

    arm_signals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    active_arms: list[str] = []
    close_starts: dict[str, list[int]] = {}
    open_starts: dict[str, list[int]] = {}
    for arm in ("left", "right"):
        gripper = arrays[f"{arm}_gripper"]
        motion = xyz_motion(arrays[f"{arm}_endpose"])
        arm_signals[arm] = (gripper, motion)
        close_starts[arm] = find_close_starts(gripper)
        open_starts[arm] = find_open_starts(gripper)
        if is_arm_active(gripper, motion):
            active_arms.append(arm)

    delivering_arm = choose_delivering_arm(task_category, active_arms, close_starts)
    finishing = finishing_arm(close_starts, open_starts)

    phases: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    gripper_summaries: dict[str, Any] = {}
    for arm in active_arms:
        gripper, motion = arm_signals[arm]
        labels = label_arm_frames(gripper, motion, task_category, arm, delivering_arm, finishing)
        phases.extend(merge_phase_segments(labels, gripper, motion, arm))
        gripper_summaries[arm] = gripper_summary(gripper)
        for frame, value in mark_grasp_close([""] * len(gripper), gripper):
            events.append(
                {
                    "frame": int(frame),
                    "event": "gripper_close_start",
                    "arm": arm,
                    "gripper_value": float(value),
                }
            )
        for frame, value in mark_release_open([""] * len(gripper), gripper):
            events.append(
                {
                    "frame": int(frame),
                    "event": "gripper_open_start",
                    "arm": arm,
                    "gripper_value": float(value),
                }
            )

    phases.sort(key=lambda item: (item["start_frame"], item["arm"]))
    events.sort(key=lambda item: (item["frame"], item["arm"], item["event"]))

    return {
        "task": task,
        "episode": global_episode,
        "local_episode": local_episode,
        "hdf5_path": str(hdf5_path.resolve()),
        "task_category": task_category,
        "active_arms": active_arms,
        "num_frames": int(num_frames),
        "scene_info": load_scene_info(scene_info_root, task, local_episode),
        "instruction": instruction,
        "gripper_summary": gripper_summaries,
        "events": events,
        "phases": phases,
    }
