"""Shared epipolar geometry helpers for VLM consistency judges."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
from PIL import Image

from metrics._common.multiview_eval_common import ARM_TO_VIEW

ARM_TO_CAMERA = {"left": "left_camera", "right": "right_camera", "head": "head_camera"}


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def skew_symmetric(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def load_camera_matrices(
    hdf5_path: str | Path,
    frame_idx: int,
    camera_name: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    path = Path(hdf5_path)
    if not path.is_file():
        return None
    intrinsic_key = f"observation/{camera_name}/intrinsic_cv"
    extrinsic_key = f"observation/{camera_name}/extrinsic_cv"
    with h5py.File(path, "r") as handle:
        if intrinsic_key not in handle or extrinsic_key not in handle:
            return None
        intrinsics = np.asarray(handle[intrinsic_key], dtype=np.float64)
        extrinsics = np.asarray(handle[extrinsic_key], dtype=np.float64)
    if frame_idx < 0 or frame_idx >= len(intrinsics):
        return None
    return intrinsics[frame_idx], extrinsics[frame_idx]


def compute_fundamental_matrix(
    intrinsic_head: np.ndarray,
    extrinsic_head: np.ndarray,
    intrinsic_active: np.ndarray,
    extrinsic_active: np.ndarray,
) -> np.ndarray:
    rotation_head, translation_head = extrinsic_head[:, :3], extrinsic_head[:, 3]
    rotation_active, translation_active = extrinsic_active[:, :3], extrinsic_active[:, 3]
    relative_rotation = rotation_active @ rotation_head.T
    relative_translation = translation_active - relative_rotation @ translation_head
    essential = skew_symmetric(relative_translation) @ relative_rotation
    fundamental = np.linalg.inv(intrinsic_active).T @ essential @ np.linalg.inv(intrinsic_head)
    return fundamental / fundamental[-1, -1]


def symmetric_epipolar_distance(
    fundamental: np.ndarray,
    point_head: np.ndarray,
    point_active: np.ndarray,
) -> float:
    point_head_h = np.array([point_head[0], point_head[1], 1.0], dtype=np.float64)
    point_active_h = np.array([point_active[0], point_active[1], 1.0], dtype=np.float64)
    line_active = fundamental @ point_head_h
    line_head = fundamental.T @ point_active_h
    dist_active = abs(point_active_h @ line_active) / np.hypot(line_active[0], line_active[1])
    dist_head = abs(point_head_h @ line_head) / np.hypot(line_head[0], line_head[1])
    return float(0.5 * (dist_head + dist_active))


def _extract_center_roi(image: np.ndarray, frac: float) -> tuple[np.ndarray, int, int]:
    height, width = image.shape[:2]
    roi_h = max(16, int(height * frac))
    roi_w = max(16, int(width * frac))
    offset_y = (height - roi_h) // 2
    offset_x = (width - roi_w) // 2
    return image[offset_y : offset_y + roi_h, offset_x : offset_x + roi_w], offset_x, offset_y


def _extract_bottom_center_roi(image: np.ndarray, frac: float) -> tuple[np.ndarray, int, int]:
    height, width = image.shape[:2]
    roi_h = max(16, int(height * frac))
    roi_w = max(16, int(width * frac))
    offset_y = height - roi_h
    offset_x = (width - roi_w) // 2
    return image[offset_y : offset_y + roi_h, offset_x : offset_x + roi_w], offset_x, offset_y


def compute_epipolar_errors(
    head_image: np.ndarray,
    active_image: np.ndarray,
    fundamental: np.ndarray,
    active_roi_frac: float = 0.75,
    head_roi_frac: float = 0.85,
    patch_size: int = 9,
    band_width_px: float = 8.0,
    ncc_threshold: float = 0.25,
    max_corners: int = 100,
) -> list[float]:
    active_roi, active_offset_x, active_offset_y = _extract_center_roi(active_image, active_roi_frac)
    head_roi, head_offset_x, head_offset_y = _extract_bottom_center_roi(head_image, head_roi_frac)
    active_gray = cv2.cvtColor(active_roi, cv2.COLOR_BGR2GRAY)
    head_gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(
        active_gray,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=5,
    )
    if corners is None:
        return []

    radius = patch_size // 2
    errors: list[float] = []
    head_x1 = head_offset_x + head_gray.shape[1] - 1
    head_y1 = head_offset_y + head_gray.shape[0] - 1

    for corner in corners:
        active_x, active_y = corner.ravel()
        active_point = np.array([active_x + active_offset_x, active_y + active_offset_y], dtype=np.float64)
        epipolar_line = fundamental.T @ np.array([active_point[0], active_point[1], 1.0], dtype=np.float64)
        a, b, c = epipolar_line
        line_norm = np.hypot(a, b)
        if line_norm < 1e-8:
            continue

        active_patch = active_gray[
            int(active_y) - radius : int(active_y) + radius + 1,
            int(active_x) - radius : int(active_x) + radius + 1,
        ]
        if active_patch.shape != (patch_size, patch_size):
            continue
        active_patch = active_patch.astype(np.float32)
        active_patch -= active_patch.mean()
        active_patch /= active_patch.std() + 1e-6

        best_head_point: np.ndarray | None = None
        best_ncc = -1.0
        for local_y in range(radius, head_gray.shape[0] - radius):
            for local_x in range(radius, head_gray.shape[1] - radius):
                head_x = local_x + head_offset_x
                head_y = local_y + head_offset_y
                if head_x < head_offset_x or head_y < head_offset_y or head_x > head_x1 or head_y > head_y1:
                    continue
                line_dist = abs(a * head_x + b * head_y + c) / line_norm
                if line_dist > band_width_px:
                    continue
                head_patch = head_gray[
                    local_y - radius : local_y + radius + 1,
                    local_x - radius : local_x + radius + 1,
                ]
                if head_patch.shape != (patch_size, patch_size):
                    continue
                head_patch = head_patch.astype(np.float32)
                head_patch -= head_patch.mean()
                head_patch /= head_patch.std() + 1e-6
                ncc = float(np.mean(head_patch * active_patch))
                if ncc > best_ncc:
                    best_ncc = ncc
                    best_head_point = np.array([head_x, head_y], dtype=np.float64)

        if best_head_point is None or best_ncc < ncc_threshold:
            continue
        errors.append(symmetric_epipolar_distance(fundamental, best_head_point, active_point))

    return errors


def compute_math_check(
    sample: dict[str, Any],
    head_image: Image.Image,
    active_image: Image.Image,
    error_threshold_px: float,
    min_matches: int,
    *,
    arm_to_view: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compute head <-> active-view epipolar error at sampled_frame."""
    mapping = arm_to_view or ARM_TO_VIEW
    arm = sample.get("arm", "")
    active_view = mapping.get(arm)
    frame_idx = int(sample["sampled_frame"])
    if active_view is None:
        return {
            "formula": "head <-> active_view epipolar error",
            "arm": arm,
            "active_view": None,
            "frame_id": frame_idx,
            "pair": None,
            "epipolar_error_mean": None,
            "epipolar_error_median": None,
            "epipolar_error_p90": None,
            "num_matches": 0,
            "error_threshold_px": error_threshold_px,
            "min_matches": min_matches,
            "math_score": None,
            "math_pass": None,
            "note": f"unsupported arm: {arm}",
        }

    camera_head = ARM_TO_CAMERA["head"]
    camera_active = ARM_TO_CAMERA[active_view]
    head_mats = load_camera_matrices(sample["hdf5_path"], frame_idx, camera_head)
    active_mats = load_camera_matrices(sample["hdf5_path"], frame_idx, camera_active)
    if head_mats is None or active_mats is None:
        return {
            "formula": "head <-> active_view epipolar error",
            "arm": arm,
            "active_view": active_view,
            "frame_id": frame_idx,
            "pair": f"head-{active_view}",
            "epipolar_error_mean": None,
            "epipolar_error_median": None,
            "epipolar_error_p90": None,
            "num_matches": 0,
            "error_threshold_px": error_threshold_px,
            "min_matches": min_matches,
            "math_score": None,
            "math_pass": None,
            "note": "missing camera intrinsics/extrinsics in HDF5",
        }

    intrinsic_head, extrinsic_head = head_mats
    intrinsic_active, extrinsic_active = active_mats
    fundamental = compute_fundamental_matrix(
        intrinsic_head,
        extrinsic_head,
        intrinsic_active,
        extrinsic_active,
    )
    errors = compute_epipolar_errors(
        pil_to_bgr(head_image),
        pil_to_bgr(active_image),
        fundamental,
    )
    if not errors:
        return {
            "formula": "head <-> active_view epipolar error (ROI near gripper)",
            "arm": arm,
            "active_view": active_view,
            "frame_id": frame_idx,
            "pair": f"head-{active_view}",
            "epipolar_error_mean": None,
            "epipolar_error_median": None,
            "epipolar_error_p90": None,
            "num_matches": 0,
            "error_threshold_px": error_threshold_px,
            "min_matches": min_matches,
            "math_score": None,
            "math_pass": False,
            "note": "no valid feature matches in gripper ROI",
        }

    error_mean = float(np.mean(errors))
    error_median = float(np.median(errors))
    error_p90 = float(np.percentile(errors, 90))
    enough_matches = len(errors) >= min_matches
    math_pass = bool(enough_matches and error_mean <= error_threshold_px)
    math_score = round(max(0.0, 1.0 - error_mean / error_threshold_px), 4) if enough_matches else None

    return {
        "formula": "head <-> active_view epipolar error (ROI near gripper)",
        "arm": arm,
        "active_view": active_view,
        "frame_id": frame_idx,
        "pair": f"head-{active_view}",
        "epipolar_error_mean": round(error_mean, 4),
        "epipolar_error_median": round(error_median, 4),
        "epipolar_error_p90": round(error_p90, 4),
        "num_matches": len(errors),
        "error_threshold_px": error_threshold_px,
        "min_matches": min_matches,
        "math_score": math_score,
        "math_pass": math_pass,
        "note": "active 视角中心 ROI 取角点，在 head 视角下半区极线邻域内匹配后计算对称极线距离",
    }
