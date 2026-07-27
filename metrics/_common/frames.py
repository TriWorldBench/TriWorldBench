"""Episode frame naming for preprocess, split views, and metrics."""

FRAME_EXT = ".jpg"


def frame_filename(frame_idx: int) -> str:
    return f"frame_{frame_idx:05d}{FRAME_EXT}"
