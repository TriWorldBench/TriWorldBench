"""Shared output directory naming: {method}_{input_ts}_{run_ts}."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

RUN_TS_FMT = "%Y%m%d_%H%M%S"
_TS_PART = r"\d{8}_\d{6}"
_INPUT_RUN_SUFFIX_RE = re.compile(rf"^(?P<prefix>.+)_(?P<input>{_TS_PART})_(?P<run>{_TS_PART})$")
_TWO_PART_SUFFIX_RE = re.compile(rf"^(?P<prefix>.+)_(?P<run>{_TS_PART})$")
_LEGACY_GENERATED_RE = re.compile(rf"^(?P<method>.+)_generated_(?P<ts>{_TS_PART})$")
_LEGACY_OUTPUT_RE = re.compile(rf"^(?P<method>.+)_output_(?P<ts>{_TS_PART})$")


def is_run_timestamp(value: str) -> bool:
    try:
        datetime.strptime(value, RUN_TS_FMT)
        return True
    except ValueError:
        return False


def now_run_timestamp() -> str:
    return datetime.now().strftime(RUN_TS_FMT)


def effective_input_ts(input_ts: str | None, run_ts: str | None) -> str | None:
    """Batch id used to group runs; falls back to run_ts for two-part / legacy names."""
    return input_ts or run_ts


def parse_stamped_name(name: str) -> tuple[str, str | None, str | None]:
    """Parse method, input_ts, run_ts from generate/output folder names (new + legacy)."""
    match = _INPUT_RUN_SUFFIX_RE.match(name)
    if match and is_run_timestamp(match.group("input")) and is_run_timestamp(match.group("run")):
        return match.group("prefix"), match.group("input"), match.group("run")

    legacy = _LEGACY_GENERATED_RE.match(name)
    if legacy and is_run_timestamp(legacy.group("ts")):
        ts = legacy.group("ts")
        return legacy.group("method"), ts, ts

    legacy = _LEGACY_OUTPUT_RE.match(name)
    if legacy and is_run_timestamp(legacy.group("ts")):
        ts = legacy.group("ts")
        return legacy.group("method"), ts, ts

    two_part = _TWO_PART_SUFFIX_RE.match(name)
    if two_part and is_run_timestamp(two_part.group("run")):
        ts = two_part.group("run")
        return two_part.group("prefix"), ts, ts

    return name, None, None


def infer_input_timestamp_from_name(name: str) -> str | None:
    _method, input_ts, run_ts = parse_stamped_name(name)
    return effective_input_ts(input_ts, run_ts)


def format_output_dir(method: str, input_ts: str | None, run_ts: str) -> str:
    if input_ts and input_ts != run_ts:
        return f"{method}_{input_ts}_{run_ts}"
    return f"{method}_{run_ts}"


def latest_pointer_path(parent: Path, method: str, input_ts: str | None) -> Path:
    base = f"{method}_{input_ts}" if input_ts else method
    return parent / f"{base}_latest.path"


def write_latest_pointer(parent: Path, method: str, input_ts: str | None, dir_name: str) -> Path:
    pointer = latest_pointer_path(parent, method, input_ts)
    pointer.write_text(dir_name + "\n", encoding="utf-8")
    return pointer


def resolve_latest_pointer(path: Path) -> Path:
    """Resolve *_latest placeholder via sibling *.path file."""
    if path.name.endswith("_latest") and not path.is_dir():
        pointer = path.parent / f"{path.name}.path"
        if pointer.is_file():
            return (path.parent / pointer.read_text(encoding="utf-8").strip()).resolve()
    return path.resolve()


def output_dir_sort_key(path: Path) -> tuple[int, float]:
    _method, _input_ts, run_ts = parse_stamped_name(path.name)
    if run_ts and is_run_timestamp(run_ts):
        return (1, datetime.strptime(run_ts, RUN_TS_FMT).timestamp())
    return (0, path.stat().st_mtime)


def list_output_dirs(output_root: Path, method: str, input_ts: str | None) -> list[Path]:
    if not output_root.is_dir():
        return []
    matches: list[Path] = []
    for path in output_root.iterdir():
        if not path.is_dir():
            continue
        parsed_method, parsed_input_ts, run_ts = parse_stamped_name(path.name)
        if parsed_method != method or run_ts is None:
            continue
        if input_ts is not None:
            if effective_input_ts(parsed_input_ts, run_ts) != input_ts:
                continue
        matches.append(path)
    return matches


def find_latest_output_dir(output_root: Path, method: str, input_ts: str | None) -> Path | None:
    candidates = list_output_dirs(output_root, method, input_ts)
    if not candidates:
        return None
    return max(candidates, key=output_dir_sort_key)
