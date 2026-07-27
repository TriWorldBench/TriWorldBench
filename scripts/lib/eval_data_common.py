"""Shared helpers for Triworld eval data prep and input adaptation."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from metrics._common.frames import FRAME_EXT, frame_filename

VIEW_NAMES = ("head", "left", "right")
PROMPT_KEYS = ("instruction", "prompt", "task_instruction", "text", "description")
GT_FRAME_SUBDIR = "frames"
GENERATED_FRAME_SUBDIR = "frames"
FRAME_DIR_NAMES = (GT_FRAME_SUBDIR, "video")
LEGACY_SPLIT_DIRS = ("gt_dataset", "all_infer")

def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_frame_dir(episode_dir: Path, view: str) -> Path | None:
    """Return frame directory for a view (prefers frames/ to match shared GT)."""
    for subdir in FRAME_DIR_NAMES:
        candidate = episode_dir / view / subdir
        if candidate.is_dir() and any(p.suffix.lower() == FRAME_EXT for p in candidate.iterdir()):
            return candidate
    return None


def generated_frame_dir(output_root: Path, episode: str, view: str) -> Path:
    return output_root / "generated_dataset" / episode / view / GENERATED_FRAME_SUBDIR


def gt_episode_ready(gt_root: Path, episode: str) -> bool:
    ep_dir = gt_root / episode
    if not ep_dir.is_dir():
        return False
    return all(resolve_frame_dir(ep_dir, view) is not None for view in VIEW_NAMES)


def cleanup_legacy_split_artifacts(output_root: Path) -> None:
    """Remove per-split gt_dataset/all_infer and stale generated video/ dirs."""
    for name in LEGACY_SPLIT_DIRS:
        path = output_root / name
        if path.exists():
            shutil.rmtree(path)
            print(f"[cleanup] removed legacy {path}")

    gen_root = output_root / "generated_dataset"
    if gen_root.is_dir():
        for video_dir in gen_root.glob("*/*/*/video"):
            if video_dir.is_dir():
                shutil.rmtree(video_dir)
                print(f"[cleanup] removed stale {video_dir}")


def uniform_frame_indices(source_count: int, target_count: int) -> list[int]:
    """Map target_count output frames to source indices in [0, source_count-1].

    Uses endpoint-aligned uniform sampling (linspace), independent of video fps.
    Upsampling repeats nearest source frames; downsampling skips frames evenly.
    """
    if target_count <= 0:
        return []
    if source_count <= 0:
        raise ValueError(f"source_count must be positive, got {source_count}")
    if target_count == 1:
        return [0]
    if source_count == 1:
        return [0] * target_count
    return [
        min(source_count - 1, round(i * (source_count - 1) / (target_count - 1)))
        for i in range(target_count)
    ]


def count_frames(video_dir: Path) -> int:
    if not video_dir.is_dir():
        return 0
    return sum(1 for p in video_dir.iterdir() if p.suffix.lower() == FRAME_EXT)


def read_episode_instruction(gt_root: Path, episode: str) -> str:
    """Read instruction/prompt from shared GT episode JSON."""
    for name in (f"{episode}.json", "episode.json"):
        path = gt_root / episode / name
        if not path.is_file():
            continue
        try:
            data = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            for key in PROMPT_KEYS:
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return ""


def run_cmd(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Command failed ({exc.returncode}): {' '.join(cmd)}\n{exc.stderr.strip()}"
        ) from exc


def ffprobe_path(ffmpeg: str = "ffmpeg") -> str:
    candidate = Path(ffmpeg).resolve().parent / "ffprobe"
    return str(candidate) if candidate.is_file() else "ffprobe"


def probe_video_frame_count(video_path: Path, ffmpeg: str = "ffmpeg") -> int:
    """Return decoded frame count; raises RuntimeError if video unreadable."""
    ffprobe = ffprobe_path(ffmpeg)
    for cmd in (
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(video_path),
        ],
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(video_path),
        ],
    ):
        try:
            result = subprocess.run(
                cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            count = int(result.stdout.strip())
            if count > 0:
                return count
        except (subprocess.CalledProcessError, ValueError):
            continue
    raise RuntimeError(f"Failed to probe frame count for {video_path}")


def write_lossless_mp4_from_frame_dir(
    frame_dir: Path,
    dst: Path,
    fps: int,
    frame_count: int,
    ffmpeg: str = "ffmpeg",
) -> None:
    """Encode frame sequence (frame_%05d + FRAME_EXT) to H.264 lossless MP4 (yuv444p)."""
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_cmd([
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / f"frame_%05d{FRAME_EXT}"),
        "-frames:v",
        str(frame_count),
        "-c:v",
        "libx264",
        "-crf",
        "0",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv444p",
        str(dst),
    ])


def probe_image_size(image_path: Path, ffmpeg: str = "ffmpeg") -> tuple[int, int]:
    ffprobe = ffprobe_path(ffmpeg)
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(image_path),
    ]
    result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    width, height = result.stdout.strip().split("x")
    return int(width), int(height)
