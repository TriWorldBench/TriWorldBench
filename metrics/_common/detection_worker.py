#!/usr/bin/env python3
"""Persistent per-GPU SAM3 detection worker for trajectory_accuracy."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

DETECTION_ROOT = Path(__file__).resolve().parents[2] / "external" / "Triworld" / "video_quality" / "processing"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DETECTION_ROOT))

from detection_tracking import GripperDetector, process_video_with_tracking  # noqa: E402

@contextlib.contextmanager
def _redirect_stdout_to_stderr():
    stdout_fd = sys.stdout.fileno()
    saved_fd = os.dup(stdout_fd)
    try:
        os.dup2(sys.stderr.fileno(), stdout_fd)
        yield
    finally:
        os.dup2(saved_fd, stdout_fd)
        os.close(saved_fd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--detect-gt", action="store_true")
    parser.add_argument("--force-reprocess", action="store_true")
    return parser.parse_args()


def _process_job(detector, job: dict, detect_gt: bool, force: bool) -> None:
    episode = str(job["episode"])
    task = str(job.get("task") or "head")
    data_base = Path(job["data_base"])
    gt_path = Path(job["gt_path"])

    episode_path = data_base / episode
    if detect_gt:
        gt_episode_path = gt_path / episode
        gt_video = gt_episode_path / "video"
        if gt_video.is_dir():
            with _redirect_stdout_to_stderr():
                process_video_with_tracking(
                    input_path=str(gt_video),
                    output_path=str(gt_episode_path),
                    detector=detector,
                    gid=None,
                    data_type="gt",
                    force_reprocess=force,
                )

    if not episode_path.is_dir():
        return
    for gid_dir in sorted(episode_path.iterdir(), key=lambda p: p.name):
        if not gid_dir.is_dir():
            continue
        gen_video = gid_dir / "video"
        if not gen_video.is_dir():
            continue
        with _redirect_stdout_to_stderr():
            process_video_with_tracking(
                input_path=str(gen_video),
                output_path=str(episode_path),
                detector=detector,
                gid=gid_dir.name,
                data_type="val",
                force_reprocess=force,
            )


def main() -> int:
    args = parse_args()
    detector = GripperDetector(model_path=args.model_path)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "STOP":
            break
        job = json.loads(line)
        episode = str(job.get("episode") or "")
        try:
            _process_job(detector, job, args.detect_gt, args.force_reprocess)
            print(json.dumps({"status": "success", "episode": episode}), flush=True)
        except Exception as exc:
            print(json.dumps({
                "status": "failed",
                "episode": episode,
                "error": str(exc),
            }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
