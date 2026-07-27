#!/usr/bin/env python3
"""
Triworld dataset validation and packaging tool (single-file edition).

Requirements: Python 3.10+, ffprobe available on PATH.

Usage:
    python validate_dataset.py /path/to/dataset
    python validate_dataset.py /path/to/dataset --output ./out

On success, packs the source dataset directory into <original_name>.zip
and writes <original_name>.sha256.txt (SHA256) next to this script (or under --output).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from zipfile import ZipFile

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WORKERS = 4

# ---------------------------------------------------------------------------
# Built-in GT (episode1..episode500, resolution 320x240)
# ---------------------------------------------------------------------------
GT_DEFAULT_WIDTH = 320
GT_DEFAULT_HEIGHT = 240
FRAME_ERROR_THRESHOLD = 0.5
RESOLUTION_MISMATCH_ERROR = (
    "Some videos do not match the expected GT resolution; please fix them before packaging."
)
FRAME_COUNT_WARNING_EN = (
    "To properly evaluate your submission, we will process data whose frame counts "
    "differ from the GT, which may affect your score!"
)
VIEW_FRAME_MISMATCH_WARNING = (
    "Inconsistent multi-view frame counts detected for some tasks (head/left/right MP4s)."
)
VIEW_NAMES = ("head", "left", "right")
VIEW_ALIASES = {
    "head": ("head",),
    "left": ("left", "left_hand", "lefthand", "left-hand"),
    "right": ("right", "right_hand", "righthand", "right-hand"),
}

GT_EPISODE_FRAMES: dict[str, int] = {
    "episode1": 482,
    "episode2": 138,
    "episode3": 288,
    "episode4": 231,
    "episode5": 113,
    "episode6": 477,
    "episode7": 311,
    "episode8": 146,
    "episode9": 159,
    "episode10": 170,
    "episode11": 116,
    "episode12": 279,
    "episode13": 120,
    "episode14": 281,
    "episode15": 239,
    "episode16": 226,
    "episode17": 659,
    "episode18": 117,
    "episode19": 266,
    "episode20": 117,
    "episode21": 131,
    "episode22": 156,
    "episode23": 176,
    "episode24": 115,
    "episode25": 96,
    "episode26": 227,
    "episode27": 167,
    "episode28": 137,
    "episode29": 175,
    "episode30": 126,
    "episode31": 114,
    "episode32": 289,
    "episode33": 160,
    "episode34": 701,
    "episode35": 138,
    "episode36": 150,
    "episode37": 249,
    "episode38": 162,
    "episode39": 514,
    "episode40": 137,
    "episode41": 146,
    "episode42": 119,
    "episode43": 163,
    "episode44": 102,
    "episode45": 258,
    "episode46": 360,
    "episode47": 141,
    "episode48": 337,
    "episode49": 241,
    "episode50": 141,
    "episode51": 115,
    "episode52": 116,
    "episode53": 103,
    "episode54": 156,
    "episode55": 157,
    "episode56": 123,
    "episode57": 330,
    "episode58": 483,
    "episode59": 317,
    "episode60": 159,
    "episode61": 272,
    "episode62": 231,
    "episode63": 159,
    "episode64": 658,
    "episode65": 158,
    "episode66": 240,
    "episode67": 127,
    "episode68": 253,
    "episode69": 171,
    "episode70": 173,
    "episode71": 445,
    "episode72": 84,
    "episode73": 136,
    "episode74": 325,
    "episode75": 113,
    "episode76": 143,
    "episode77": 160,
    "episode78": 531,
    "episode79": 248,
    "episode80": 106,
    "episode81": 247,
    "episode82": 144,
    "episode83": 162,
    "episode84": 153,
    "episode85": 137,
    "episode86": 243,
    "episode87": 138,
    "episode88": 261,
    "episode89": 169,
    "episode90": 98,
    "episode91": 141,
    "episode92": 141,
    "episode93": 146,
    "episode94": 117,
    "episode95": 80,
    "episode96": 147,
    "episode97": 452,
    "episode98": 81,
    "episode99": 504,
    "episode100": 165,
    "episode101": 485,
    "episode102": 166,
    "episode103": 472,
    "episode104": 276,
    "episode105": 115,
    "episode106": 467,
    "episode107": 135,
    "episode108": 113,
    "episode109": 226,
    "episode110": 163,
    "episode111": 154,
    "episode112": 244,
    "episode113": 113,
    "episode114": 488,
    "episode115": 480,
    "episode116": 286,
    "episode117": 132,
    "episode118": 660,
    "episode119": 485,
    "episode120": 113,
    "episode121": 99,
    "episode122": 168,
    "episode123": 478,
    "episode124": 112,
    "episode125": 331,
    "episode126": 227,
    "episode127": 286,
    "episode128": 153,
    "episode129": 157,
    "episode130": 183,
    "episode131": 138,
    "episode132": 174,
    "episode133": 334,
    "episode134": 295,
    "episode135": 269,
    "episode136": 121,
    "episode137": 154,
    "episode138": 460,
    "episode139": 223,
    "episode140": 162,
    "episode141": 148,
    "episode142": 245,
    "episode143": 325,
    "episode144": 121,
    "episode145": 251,
    "episode146": 103,
    "episode147": 446,
    "episode148": 121,
    "episode149": 148,
    "episode150": 243,
    "episode151": 477,
    "episode152": 142,
    "episode153": 125,
    "episode154": 220,
    "episode155": 248,
    "episode156": 140,
    "episode157": 159,
    "episode158": 176,
    "episode159": 234,
    "episode160": 275,
    "episode161": 162,
    "episode162": 313,
    "episode163": 142,
    "episode164": 147,
    "episode165": 154,
    "episode166": 480,
    "episode167": 306,
    "episode168": 160,
    "episode169": 92,
    "episode170": 274,
    "episode171": 145,
    "episode172": 448,
    "episode173": 347,
    "episode174": 657,
    "episode175": 145,
    "episode176": 231,
    "episode177": 328,
    "episode178": 126,
    "episode179": 79,
    "episode180": 146,
    "episode181": 145,
    "episode182": 300,
    "episode183": 670,
    "episode184": 120,
    "episode185": 253,
    "episode186": 243,
    "episode187": 118,
    "episode188": 274,
    "episode189": 271,
    "episode190": 221,
    "episode191": 144,
    "episode192": 144,
    "episode193": 148,
    "episode194": 117,
    "episode195": 287,
    "episode196": 78,
    "episode197": 137,
    "episode198": 159,
    "episode199": 153,
    "episode200": 118,
    "episode201": 463,
    "episode202": 284,
    "episode203": 94,
    "episode204": 334,
    "episode205": 461,
    "episode206": 281,
    "episode207": 276,
    "episode208": 310,
    "episode209": 76,
    "episode210": 112,
    "episode211": 148,
    "episode212": 226,
    "episode213": 117,
    "episode214": 190,
    "episode215": 219,
    "episode216": 450,
    "episode217": 117,
    "episode218": 167,
    "episode219": 1214,
    "episode220": 226,
    "episode221": 306,
    "episode222": 478,
    "episode223": 160,
    "episode224": 156,
    "episode225": 166,
    "episode226": 278,
    "episode227": 242,
    "episode228": 143,
    "episode229": 129,
    "episode230": 284,
    "episode231": 143,
    "episode232": 169,
    "episode233": 138,
    "episode234": 115,
    "episode235": 199,
    "episode236": 273,
    "episode237": 242,
    "episode238": 147,
    "episode239": 164,
    "episode240": 156,
    "episode241": 119,
    "episode242": 160,
    "episode243": 92,
    "episode244": 165,
    "episode245": 663,
    "episode246": 319,
    "episode247": 128,
    "episode248": 333,
    "episode249": 143,
    "episode250": 173,
    "episode251": 247,
    "episode252": 228,
    "episode253": 462,
    "episode254": 162,
    "episode255": 449,
    "episode256": 82,
    "episode257": 222,
    "episode258": 169,
    "episode259": 139,
    "episode260": 162,
    "episode261": 454,
    "episode262": 277,
    "episode263": 115,
    "episode264": 77,
    "episode265": 126,
    "episode266": 163,
    "episode267": 243,
    "episode268": 116,
    "episode269": 464,
    "episode270": 223,
    "episode271": 102,
    "episode272": 110,
    "episode273": 152,
    "episode274": 230,
    "episode275": 472,
    "episode276": 461,
    "episode277": 145,
    "episode278": 159,
    "episode279": 424,
    "episode280": 113,
    "episode281": 516,
    "episode282": 78,
    "episode283": 125,
    "episode284": 123,
    "episode285": 434,
    "episode286": 109,
    "episode287": 145,
    "episode288": 166,
    "episode289": 150,
    "episode290": 128,
    "episode291": 243,
    "episode292": 255,
    "episode293": 314,
    "episode294": 168,
    "episode295": 157,
    "episode296": 220,
    "episode297": 227,
    "episode298": 118,
    "episode299": 117,
    "episode300": 228,
    "episode301": 333,
    "episode302": 159,
    "episode303": 140,
    "episode304": 175,
    "episode305": 424,
    "episode306": 115,
    "episode307": 284,
    "episode308": 187,
    "episode309": 184,
    "episode310": 277,
    "episode311": 97,
    "episode312": 138,
    "episode313": 136,
    "episode314": 126,
    "episode315": 129,
    "episode316": 246,
    "episode317": 306,
    "episode318": 291,
    "episode319": 326,
    "episode320": 475,
    "episode321": 321,
    "episode322": 79,
    "episode323": 148,
    "episode324": 467,
    "episode325": 336,
    "episode326": 119,
    "episode327": 140,
    "episode328": 241,
    "episode329": 142,
    "episode330": 178,
    "episode331": 101,
    "episode332": 242,
    "episode333": 451,
    "episode334": 457,
    "episode335": 306,
    "episode336": 168,
    "episode337": 320,
    "episode338": 164,
    "episode339": 152,
    "episode340": 140,
    "episode341": 126,
    "episode342": 292,
    "episode343": 165,
    "episode344": 165,
    "episode345": 323,
    "episode346": 104,
    "episode347": 273,
    "episode348": 288,
    "episode349": 242,
    "episode350": 238,
    "episode351": 128,
    "episode352": 155,
    "episode353": 517,
    "episode354": 98,
    "episode355": 165,
    "episode356": 122,
    "episode357": 335,
    "episode358": 244,
    "episode359": 279,
    "episode360": 148,
    "episode361": 235,
    "episode362": 162,
    "episode363": 191,
    "episode364": 141,
    "episode365": 82,
    "episode366": 477,
    "episode367": 672,
    "episode368": 155,
    "episode369": 290,
    "episode370": 120,
    "episode371": 168,
    "episode372": 146,
    "episode373": 161,
    "episode374": 457,
    "episode375": 118,
    "episode376": 116,
    "episode377": 285,
    "episode378": 163,
    "episode379": 140,
    "episode380": 83,
    "episode381": 119,
    "episode382": 172,
    "episode383": 155,
    "episode384": 463,
    "episode385": 173,
    "episode386": 159,
    "episode387": 141,
    "episode388": 222,
    "episode389": 441,
    "episode390": 244,
    "episode391": 287,
    "episode392": 140,
    "episode393": 172,
    "episode394": 321,
    "episode395": 488,
    "episode396": 169,
    "episode397": 140,
    "episode398": 131,
    "episode399": 246,
    "episode400": 135,
    "episode401": 329,
    "episode402": 278,
    "episode403": 222,
    "episode404": 147,
    "episode405": 157,
    "episode406": 286,
    "episode407": 136,
    "episode408": 154,
    "episode409": 159,
    "episode410": 148,
    "episode411": 79,
    "episode412": 253,
    "episode413": 117,
    "episode414": 262,
    "episode415": 102,
    "episode416": 172,
    "episode417": 77,
    "episode418": 149,
    "episode419": 139,
    "episode420": 158,
    "episode421": 143,
    "episode422": 96,
    "episode423": 273,
    "episode424": 284,
    "episode425": 286,
    "episode426": 76,
    "episode427": 223,
    "episode428": 95,
    "episode429": 119,
    "episode430": 428,
    "episode431": 309,
    "episode432": 157,
    "episode433": 141,
    "episode434": 465,
    "episode435": 151,
    "episode436": 166,
    "episode437": 159,
    "episode438": 140,
    "episode439": 221,
    "episode440": 166,
    "episode441": 242,
    "episode442": 159,
    "episode443": 243,
    "episode444": 136,
    "episode445": 150,
    "episode446": 156,
    "episode447": 111,
    "episode448": 151,
    "episode449": 154,
    "episode450": 146,
    "episode451": 77,
    "episode452": 169,
    "episode453": 271,
    "episode454": 464,
    "episode455": 176,
    "episode456": 652,
    "episode457": 134,
    "episode458": 149,
    "episode459": 131,
    "episode460": 225,
    "episode461": 268,
    "episode462": 244,
    "episode463": 467,
    "episode464": 463,
    "episode465": 170,
    "episode466": 81,
    "episode467": 248,
    "episode468": 152,
    "episode469": 148,
    "episode470": 332,
    "episode471": 255,
    "episode472": 119,
    "episode473": 460,
    "episode474": 138,
    "episode475": 113,
    "episode476": 164,
    "episode477": 221,
    "episode478": 393,
    "episode479": 134,
    "episode480": 283,
    "episode481": 142,
    "episode482": 334,
    "episode483": 123,
    "episode484": 168,
    "episode485": 226,
    "episode486": 187,
    "episode487": 96,
    "episode488": 245,
    "episode489": 123,
    "episode490": 223,
    "episode491": 310,
    "episode492": 340,
    "episode493": 252,
    "episode494": 109,
    "episode495": 163,
    "episode496": 340,
    "episode497": 151,
    "episode498": 142,
    "episode499": 117,
    "episode500": 232
}

EPISODE_LIST = list(GT_EPISODE_FRAMES.keys())
EPISODE_DIR_RE = re.compile(r"^episode\d+$", re.I)
OUTPUT_RGB_RE = re.compile(r"^(?P<task>.+)_(?P<episode>episode\d+)_outputs_rgb\.mp4$", re.I)
EXECUTABLE_SUFFIXES = {".exe", ".dll", ".so", ".dylib", ".bat", ".cmd", ".com", ".msi", ".sh", ".bin"}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Finding:
    level: str
    code: str
    message: str


@dataclass
class ScanResult:
    dataset_root: Path
    found_episodes: list[str]
    missing_episodes: list[str]
    incomplete_episodes: list[str]
    extra_episodes: list[str]


@dataclass
class ProcessReport:
    processed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    frame_errors: list[str] = field(default_factory=list)
    frame_warnings: list[str] = field(default_factory=list)
    view_frame_warnings: list[str] = field(default_factory=list)
    resolution_errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ffprobe
# ---------------------------------------------------------------------------
def ffprobe_bin() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        sibling = Path(ffmpeg).resolve().parent / ("ffprobe.exe" if os.name == "nt" else "ffprobe")
        if sibling.is_file():
            return str(sibling)
    return shutil.which("ffprobe") or "ffprobe"

def probe_video_size(video_path: Path) -> tuple[int, int]:
    cmd = [
        ffprobe_bin(), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", str(video_path),
    ]
    result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def probe_video_frame_count(video_path: Path) -> int:
    ffprobe = ffprobe_bin()
    for cmd in (
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1", str(video_path)],
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames",
         "-of", "default=nokey=1:noprint_wrappers=1", str(video_path)],
    ):
        try:
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            count = int(result.stdout.strip())
            if count > 0:
                return count
        except (subprocess.CalledProcessError, ValueError):
            continue
    raise RuntimeError(f"Unable to read frame count: {video_path}")


# ---------------------------------------------------------------------------
# MP4 layout detection
# ---------------------------------------------------------------------------
def _match_view_from_name(name: str) -> str | None:
    lower = name.lower()
    stem = Path(name).stem.lower()
    for view, aliases in VIEW_ALIASES.items():
        for alias in aliases:
            if alias in lower or alias in stem:
                return view
    return None


def _view_folder_mp4s(episode_dir: Path) -> dict[str, Path] | None:
    views: dict[str, Path] = {}
    for view in VIEW_NAMES:
        view_dir = episode_dir / view
        if not view_dir.is_dir():
            continue
        mp4s = sorted(view_dir.glob("*.mp4"))
        if mp4s:
            views[view] = mp4s[0]
    return views if len(views) == 3 else None


def _named_view_mp4s(episode_dir: Path) -> dict[str, Path] | None:
    views: dict[str, Path] = {}
    for mp4 in sorted(episode_dir.glob("*.mp4")):
        view = _match_view_from_name(mp4.name)
        if view and view not in views:
            views[view] = mp4
    return views if len(views) == 3 else None


def _single_stitched_mp4(episode_dir: Path) -> Path | None:
    mp4s = sorted(episode_dir.glob("*.mp4"))
    if len(mp4s) == 1:
        return mp4s[0]
    rgb = sorted(episode_dir.glob("*_outputs_rgb.mp4"))
    return rgb[0] if len(rgb) == 1 else None


def parse_episode(episode_dir: Path) -> tuple[str, dict[str, Path]]:
    views = _view_folder_mp4s(episode_dir)
    if views:
        return "view_folders", views
    views = _named_view_mp4s(episode_dir)
    if views:
        return "named_mp4", views
    stitched = _single_stitched_mp4(episode_dir)
    if stitched is not None:
        return "stitched", {"__stitched__": stitched}
    raise RuntimeError(
        "Unrecognized MP4 layout (expected head/left/right folders, named MP4s, or a single stitched MP4)"
    )


def missing_views(episode_dir: Path) -> list[str]:
    missing: list[str] = []
    for view in VIEW_NAMES:
        if (episode_dir / f"{view}.mp4").is_file():
            continue
        if (episode_dir / view).is_dir() and any((episode_dir / view).glob("*.mp4")):
            continue
        if list(episode_dir.glob(f"*{view}*.mp4")):
            continue
        missing.append(view)
    return missing


def episode_has_all_views(episode_dir: Path) -> bool:
    return not missing_views(episode_dir)


# ---------------------------------------------------------------------------
# Scanning and security checks
# ---------------------------------------------------------------------------
def _episode_hit(directory: Path, episode_names: set[str]) -> bool:
    try:
        children = [p.name for p in directory.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))]
    except OSError:
        return False
    return any(EPISODE_DIR_RE.match(name) and name in episode_names for name in children)


def find_dataset_root(root: Path, episode_names: set[str]) -> Path | None:
    if not root.is_dir():
        return None
    queue = [root]
    seen: set[Path] = set()
    while queue:
        current = queue.pop(0)
        resolved = current.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _episode_hit(current, episode_names):
            return current
        try:
            subdirs = sorted(p for p in current.iterdir() if p.is_dir() and not p.name.startswith((".", "_")))
        except OSError:
            continue
        queue.extend(subdirs)
    return None


def scan_dataset(inspect_root: Path, episode_list: list[str]) -> ScanResult:
    episode_names = set(episode_list)
    dataset_root = find_dataset_root(inspect_root, episode_names)
    if dataset_root is None:
        raise RuntimeError("No dataset root matching the official episode list was found; check directory layout.")
    found, incomplete, extra = [], [], []
    for child in sorted(dataset_root.iterdir()):
        if not child.is_dir() or child.name.startswith((".", "_")):
            continue
        if not EPISODE_DIR_RE.match(child.name):
            continue
        if child.name not in episode_names:
            extra.append(child.name)
            continue
        (found if episode_has_all_views(child) else incomplete).append(child.name)
    missing = sorted(episode_names - set(found) - set(incomplete))
    return ScanResult(
        dataset_root=dataset_root,
        found_episodes=sorted(found),
        missing_episodes=missing,
        incomplete_episodes=sorted(incomplete),
        extra_episodes=sorted(extra),
    )


def scan_security(dataset_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in dataset_root.rglob("*"):
        try:
            rel = str(path.relative_to(dataset_root))
        except ValueError:
            rel = str(path)
        if path.name.startswith("."):
            findings.append(Finding("ERROR", "hidden", f"Hidden file or directory not allowed: {rel}"))
            continue
        if path.is_symlink():
            findings.append(Finding("ERROR", "symlink", f"Symlink not allowed: {rel}"))
            continue
        if path.is_file():
            if path.suffix.lower() in EXECUTABLE_SUFFIXES:
                findings.append(Finding("ERROR", "executable", f"Executable file not allowed: {rel}"))
            elif os.name != "nt":
                mode = path.stat().st_mode
                if mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
                    findings.append(Finding("ERROR", "executable", f"Executable file not allowed: {rel}"))
    return findings


# ---------------------------------------------------------------------------
# Processing and validation
# ---------------------------------------------------------------------------
def validate_episode(episode: str, episode_dir: Path) -> ProcessReport:
    report = ProcessReport()
    gt_count = GT_EPISODE_FRAMES.get(episode, 0)
    missing = missing_views(episode_dir)
    if missing:
        report.failed.append(f"{episode}: missing head/left/right MP4 views ({', '.join(missing)})")
        return report
    try:
        fmt, views = parse_episode(episode_dir)
        per_view_counts: dict[str, int] = {}

        if fmt == "stitched":
            video = views["__stitched__"]
            width, height = probe_video_size(video)
            if width % 3 != 0 and abs(width / 3 - round(width / 3)) > 1:
                raise RuntimeError(f"Stitched video width {width} cannot be split into three equal panels")
            frame_count = probe_video_frame_count(video)
            for view in VIEW_NAMES:
                per_view_counts[view] = frame_count
            target_w, target_h = GT_DEFAULT_WIDTH * 3, GT_DEFAULT_HEIGHT
            if width != target_w or height != target_h:
                report.resolution_errors.append(
                    f"{episode} stitched {width}x{height} (expected {target_w}x{target_h})"
                )
        else:
            for view in VIEW_NAMES:
                video = views[view]
                frame_count = probe_video_frame_count(video)
                width, height = probe_video_size(video)
                per_view_counts[view] = frame_count
                if width != GT_DEFAULT_WIDTH or height != GT_DEFAULT_HEIGHT:
                    report.resolution_errors.append(
                        f"{episode}/{view} {width}x{height} "
                        f"(expected {GT_DEFAULT_WIDTH}x{GT_DEFAULT_HEIGHT})"
                    )

        if fmt != "stitched":
            counts = list(per_view_counts.values())
            if len(set(counts)) > 1:
                detail = ", ".join(f"{view}={per_view_counts[view]}" for view in VIEW_NAMES)
                report.view_frame_warnings.append(f"{episode}: {detail}")

        if gt_count > 0:
            ref = max(per_view_counts.values())
            diff = abs(ref - gt_count)
            if diff == 0:
                pass
            elif diff / gt_count > FRAME_ERROR_THRESHOLD:
                report.frame_errors.append(
                    f"{episode} frame count mismatch {ref} vs GT {gt_count} (exceeds 50% threshold)"
                )
            else:
                report.frame_warnings.append(
                    f"{episode} frame count mismatch {ref} vs GT {gt_count}"
                )

        report.processed.append(episode)
    except Exception as exc:
        report.failed.append(f"{episode}: {exc}")
    return report


def print_progress(done: int, total: int, width: int = 40) -> None:
    if total <= 0:
        return
    ratio = done / total
    filled = int(width * ratio)
    bar = "=" * filled + "-" * (width - filled)
    print(f"\rProgress: [{bar}] {done}/{total} ({ratio * 100:.1f}%)", end="", flush=True)


def output_basename(input_path: Path) -> str:
    return input_path.stem if input_path.is_file() else input_path.name


def prompt_continue_on_warnings() -> bool:
    while True:
        answer = input("Warnings were reported. Continue to generate validated output files? (Y/N): ").strip().upper()
        if answer == "Y":
            return True
        if answer == "N":
            return False
        print("Please enter Y or N.")


def validate_dataset(
    input_path: Path, *, workers: int = DEFAULT_WORKERS,
) -> tuple[str, list[Finding], Path | None]:
    findings: list[Finding] = []

    if not input_path.is_dir():
        findings.append(Finding("ERROR", "input", "Input must be a dataset directory"))
        return "ERROR", findings, None

    try:
        scan = scan_dataset(input_path, EPISODE_LIST)
    except Exception as exc:
        findings.append(Finding("ERROR", "structure", str(exc)))
        return "ERROR", findings, None

    findings.extend(scan_security(scan.dataset_root))
    for ep in scan.extra_episodes:
        findings.append(Finding("ERROR", "episode_id", f"Non-official episode ID: {ep}"))
    for ep in scan.incomplete_episodes:
        missing = missing_views(scan.dataset_root / ep)
        detail = ", ".join(missing) if missing else "head/left/right"
        findings.append(
            Finding("ERROR", "missing_view", f"{ep} missing complete head/left/right MP4 views ({detail})")
        )
    if scan.missing_episodes:
        preview = ", ".join(scan.missing_episodes[:10])
        suffix = f" ... ({len(scan.missing_episodes)} total)" if len(scan.missing_episodes) > 10 else ""
        findings.append(Finding(
            "ERROR", "missing_episode",
            f"Missing official episodes ({len(scan.missing_episodes)}): {preview}{suffix}",
        ))
    if len(scan.found_episodes) != len(EPISODE_LIST):
        findings.append(Finding(
            "ERROR", "episode_count",
            f"Episode count mismatch (expected {len(EPISODE_LIST)}, complete {len(scan.found_episodes)})",
        ))

    if any(f.level == "ERROR" for f in findings):
        return "ERROR", findings, None

    merged = ProcessReport()
    total = len(scan.found_episodes)
    progress_lock = threading.Lock()
    completed = 0

    def validate_one(episode: str) -> ProcessReport:
        nonlocal completed
        rep = validate_episode(episode, scan.dataset_root / episode)
        with progress_lock:
            completed += 1
            print_progress(completed, total)
        return rep

    worker_count = max(1, min(workers, total))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(validate_one, episode) for episode in scan.found_episodes]
        for future in as_completed(futures):
            rep = future.result()
            merged.processed.extend(rep.processed)
            merged.failed.extend(rep.failed)
            merged.frame_errors.extend(rep.frame_errors)
            merged.frame_warnings.extend(rep.frame_warnings)
            merged.view_frame_warnings.extend(rep.view_frame_warnings)
            merged.resolution_errors.extend(rep.resolution_errors)
            for fail in rep.failed:
                findings.append(Finding("ERROR", "corrupt", fail))

    print()

    for msg in merged.resolution_errors:
        findings.append(Finding("ERROR", "resolution", msg))
    if merged.resolution_errors:
        findings.append(Finding("ERROR", "resolution", RESOLUTION_MISMATCH_ERROR))
    for msg in merged.frame_errors:
        findings.append(Finding("ERROR", "frame", msg))
    for msg in merged.frame_warnings:
        findings.append(Finding("WARNING", "frame", msg))
    if merged.frame_warnings:
        findings.append(Finding("WARNING", "frame", FRAME_COUNT_WARNING_EN))
    for msg in merged.view_frame_warnings:
        findings.append(Finding("WARNING", "view_frame", msg))
    if merged.view_frame_warnings:
        findings.append(Finding("WARNING", "view_frame", VIEW_FRAME_MISMATCH_WARNING))

    if merged.failed or merged.frame_errors or merged.resolution_errors:
        return "ERROR", findings, None
    level = "WARNING" if any(f.level == "WARNING" for f in findings) else "PASS"
    return level, findings, scan.dataset_root


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zip_directory(source_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w") as archive:
        for file in source_dir.rglob("*"):
            if file.is_file():
                archive.write(file, file.relative_to(source_dir))


def main() -> int:
    parser = argparse.ArgumentParser(description="Triworld dataset validation (episodes 1..500)")
    parser.add_argument("input", type=Path, help="Dataset directory")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory for <name>.zip and <name>.sha256.txt (default: next to this script)",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Number of episodes to validate in parallel (default: {DEFAULT_WORKERS})",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        print(f"ERROR: Input path does not exist: {input_path}", file=sys.stderr)
        return 2
    if not input_path.is_dir():
        print(f"ERROR: Input must be a dataset directory: {input_path}", file=sys.stderr)
        return 2

    base_name = output_basename(input_path)
    pack_dir = (args.output or SCRIPT_DIR / base_name).resolve()

    print(f"ffprobe: {ffprobe_bin()}")
    print(
        f"Criteria: {len(EPISODE_LIST)} episodes, "
        f"{GT_DEFAULT_WIDTH}x{GT_DEFAULT_HEIGHT} (validation only; videos are not modified)"
    )
    print(f"Input: {input_path}")
    print(f"Workers: {max(1, args.workers)}")
    print("---")

    level, findings, dataset_root = validate_dataset(
        input_path, workers=max(1, args.workers),
    )

    for item in findings:
        print(f"[{item.level}] {item.message}")
    print("---")
    print(f"Result: {level}")

    pack_ok = level == "PASS"
    if level == "WARNING" and dataset_root is not None:
        pack_ok = prompt_continue_on_warnings()

    if pack_ok and level in ("PASS", "WARNING") and dataset_root is not None:
        if pack_dir.exists():
            shutil.rmtree(pack_dir)
        pack_dir.mkdir(parents=True, exist_ok=True)
        zip_path = pack_dir / f"{base_name}.zip"
        zip_directory(dataset_root, zip_path)
        txt_path = pack_dir / f"{base_name}.sha256.txt"
        txt_path.write_text(sha256_file(zip_path) + "\n", encoding="utf-8")
        print(f"Please submit both {zip_path.name} and {txt_path.name} by email.")
        print(f"Output directory: {pack_dir}")
        print(f"Archive: {zip_path}")
        print(f"Checksum: {txt_path}")
    elif level == "WARNING" and not pack_ok:
        print("Cancelled: validated output files were not generated.")

    if level == "ERROR":
        return 1
    if level == "WARNING" and not pack_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
