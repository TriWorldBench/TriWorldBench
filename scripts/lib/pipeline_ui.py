"""Terminal UI for Triworld eval pipeline — icons, progress bars, step tracker."""

from __future__ import annotations

import atexit
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from metrics._common.subprocess_env import build_subprocess_env

# ── ANSI palette ──────────────────────────────────────────────────────────────
_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"
_RED = "\033[31m"
_WHITE = "\033[97m"

_ICON = {
    "launch": "⚡",
    "metric": "◈",
    "prep": "⚙",
    "done": "✓",
    "skip": "⊘",
    "fail": "✗",
    "run": "▸",
    "info": "◉",
    "gpu": "⬡",
    "file": "⎘",
    "clock": "⏱",
    "bar": "█",
    "bar_empty": "░",
}

# Hollow outline ASCII — short-dash stitched style
_WELCOME_ART_FULL = r"""

██╗    ██╗███████╗██╗      ██████╗ ██████╗ ███╗   ███╗███████╗
██║    ██║██╔════╝██║     ██╔════╝██╔═══██╗████╗ ████║██╔════╝
██║ █╗ ██║█████╗  ██║     ██║     ██║   ██║██╔████╔██║█████╗
██║███╗██║██╔══╝  ██║     ██║     ██║   ██║██║╚██╔╝██║██╔══╝
╚███╔███╔╝███████╗███████╗╚██████╗╚██████╔╝██║ ╚═╝ ██║███████╗
 ╚══╝╚══╝ ╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝

████████╗██████╗ ██╗██╗    ██╗ ██████╗ ██████╗ ██╗     ██████╗
╚══██╔══╝██╔══██╗██║██║    ██║██╔═══██╗██╔══██╗██║     ██╔══██╗
   ██║   ██████╔╝██║██║ █╗ ██║██║   ██║██████╔╝██║     ██║  ██║
   ██║   ██╔══██╗██║██║███╗██║██║   ██║██╔══██╗██║     ██║  ██║
   ██║   ██║  ██║██║╚███╔███╔╝╚██████╔╝██║  ██║███████╗██████╔╝
   ╚═╝   ╚═╝  ╚═╝╚═╝ ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═════╝

""".strip("\n").split("\n")

_WELCOME_ART_COMPACT = r"""

██╗    ██╗███████╗██╗      ██████╗ ██████╗ ███╗   ███╗███████╗
██║    ██║██╔════╝██║     ██╔════╝██╔═══██╗████╗ ████║██╔════╝
██║ █╗ ██║█████╗  ██║     ██║     ██║   ██║██╔████╔██║█████╗
██║███╗██║██╔══╝  ██║     ██║     ██║   ██║██║╚██╔╝██║██╔══╝
╚███╔███╔╝███████╗███████╗╚██████╗╚██████╔╝██║ ╚═╝ ██║███████╗
 ╚══╝╚══╝ ╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝

████████╗██████╗ ██╗██╗    ██╗ ██████╗ ██████╗ ██╗     ██████╗
╚══██╔══╝██╔══██╗██║██║    ██║██╔═══██╗██╔══██╗██║     ██╔══██╗
   ██║   ██████╔╝██║██║ █╗ ██║██║   ██║██████╔╝██║     ██║  ██║
   ██║   ██╔══██╗██║██║███╗██║██║   ██║██╔══██╗██║     ██║  ██║
   ██║   ██║  ██║██║╚███╔███╔╝╚██████╔╝██║  ██║███████╗██████╔╝
   ╚═╝   ╚═╝  ╚═╝╚═╝ ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═════╝

""".strip("\n").split("\n")



_METRIC_ICONS = {
    "Instruction_Following": "📋",
    "Interaction_Quality": "🤝",
    "Perspectivity": "📐",
    "aesthetic_quality": "🎨",
    "background_consistency": "🌄",
    "dynamic_state_alignment": "🔄",
    "flow_score": "🌊",
    "image_quality": "🖼",
    "jepa_similarity": "🧬",
    "photometric_smoothness": "✨",
    "psnr": "📊",
    "semantic_alignment": "🔤",
    "ssim": "🔍",
    "subject_consistency": "👤",
    "trajectory_accuracy": "🎯",
    "vlm_consistency01": "🤖",
    "vlm_consistency02": "🤖",
    "vlm_consistency03": "🤖",
    "VQA": "❓",
}


_TERMINAL_RESET_REGISTERED = False


def reset_terminal() -> None:
    """Reset echo, cursor, and ANSI attributes on the controlling terminal."""
    if sys.stdin.isatty():
        try:
            subprocess.run(["stty", "sane"], stdin=sys.stdin, check=False)
        except OSError:
            pass
    if sys.stdout.isatty():
        sys.stdout.write("\r\033[K\033[0m\033[?25h\n")
        sys.stdout.flush()


def register_terminal_reset() -> None:
    global _TERMINAL_RESET_REGISTERED
    if _TERMINAL_RESET_REGISTERED:
        return
    atexit.register(reset_terminal)
    _TERMINAL_RESET_REGISTERED = True


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TRIWORLD_PLAIN_UI") == "1":
        return False
    return sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"


def _c(code: str, text: str, color: bool) -> str:
    return f"{code}{text}{_RESET}" if color else text


def _rgb_fg(r: int, g: int, b: int) -> str:
    return f"\033[38;2;{r};{g};{b}m"


def _hsv_to_rgb_bytes(h: float, s: float, v: float) -> tuple[int, int, int]:
    h = h % 1.0
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return int(r * 255), int(g * 255), int(b * 255)


def _rainbow_line(line: str, *, line_idx: int, frame: int, char_stride: float = 0.11) -> str:
    """Per-char RGB with sine flicker; spaces stay uncolored."""
    parts: list[str] = []
    col = 0
    for ch in line:
        if ch == " ":
            parts.append(ch)
            continue
        phase = frame * 0.17 + line_idx * 0.09 + col * char_stride
        hue = (phase * 0.14) % 1.0
        flicker = 0.50 + 0.50 * math.sin(phase * math.tau)
        r, g, b = _hsv_to_rgb_bytes(hue, 0.92, 0.35 + 0.65 * flicker)
        parts.append(f"{_rgb_fg(r, g, b)}{_BOLD}{ch}{_RESET}")
        col += 1
    return "".join(parts)


def _can_animate_banner_in_place(art: list[str]) -> bool:
    """In-place CSI cursor-up breaks when lines wrap; fall back to one static frame."""
    if not sys.stdout.isatty() or os.environ.get("TERM", "") == "dumb":
        return False
    if os.environ.get("TRIWORLD_RGB_BANNER", "1") == "0":
        return False
    if not art:
        return False
    width = shutil.get_terminal_size((80, 24)).columns
    max_line_len = max(len(line) for line in art)
    return width >= max_line_len


def _print_rgb_banner_frame(art: list[str], *, frame: int = 0) -> None:
    for i, line in enumerate(art):
        print(_rainbow_line(line, line_idx=i, frame=frame), flush=True)


def _animate_rgb_banner(art: list[str], *, frames: int = 30, fps: float = 20.0) -> None:
    """Play RGB per-char flicker animation in-place, leave last frame."""
    try:
        if frames <= 1 or not _can_animate_banner_in_place(art):
            _print_rgb_banner_frame(art, frame=0)
            return

        interval = 1.0 / fps
        for frame in range(frames):
            if frame > 0:
                sys.stdout.write(f"\033[{len(art)}A")
            for i, line in enumerate(art):
                sys.stdout.write("\033[K" + _rainbow_line(line, line_idx=i, frame=frame) + "\n")
            sys.stdout.flush()
            if frame + 1 < frames:
                time.sleep(interval)
    finally:
        reset_terminal()


@dataclass
class StepRecord:
    name: str
    status: str  # running | done | skip | fail
    elapsed: float = 0.0
    log_path: Path | None = None
    detail: str = ""


class PipelineUI:
    """Rich terminal display for the eval orchestrator."""

    def __init__(self, enabled: bool | None = None) -> None:
        self.color = _supports_color() if enabled is None else bool(enabled)
        self._t0 = time.time()
        self._steps: list[StepRecord] = []
        self._total = 0
        self._current = 0
        self._live_line = ""
        if sys.stdout.isatty():
            atexit.register(self._restore_terminal)

    def _restore_terminal(self) -> None:
        self._clear_live()
        reset_terminal()

    # ── formatting helpers ────────────────────────────────────────────────────

    def _icon(self, key: str) -> str:
        return _ICON.get(key, "•")

    def _metric_icon(self, name: str) -> str:
        return _METRIC_ICONS.get(name, self._icon("metric"))

    def _elapsed(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        mins, secs = divmod(int(seconds), 60)
        if mins < 60:
            return f"{mins}m{secs:02d}s"
        hours, mins = divmod(mins, 60)
        return f"{hours}h{mins:02d}m"

    def _progress_bar(self, current: int, total: int, width: int = 28) -> str:
        if total <= 0:
            pct = 0.0
            filled = 0
        else:
            pct = min(100.0, current / total * 100)
            filled = int(width * current / total)
        bar = self._icon("bar") * filled + self._icon("bar_empty") * (width - filled)
        return f"[{bar}] {pct:5.1f}%"

    def _step_badge(self) -> str:
        if self._total <= 0:
            return ""
        return _c(_DIM, f"[{self._current:>2}/{self._total:<2}]", self.color)

    def _clear_live(self) -> None:
        if self._live_line and sys.stdout.isatty():
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
        self._live_line = ""

    def _println(self, text: str = "") -> None:
        self._clear_live()
        print(text, flush=True)

    # ── public API ────────────────────────────────────────────────────────────

    def banner(self) -> None:
        if os.environ.get("TRIWORLD_SKIP_BANNER") == "1":
            return

        w = shutil.get_terminal_size((80, 24)).columns
        art = _WELCOME_ART_FULL if w >= 72 else _WELCOME_ART_COMPACT

        self._println()
        rgb_on = (
            self.color
            and sys.stdout.isatty()
            and os.environ.get("TRIWORLD_RGB_BANNER", "1") != "0"
        )
        if rgb_on:
            frames = int(os.environ.get("TRIWORLD_BANNER_FRAMES", "30"))
            fps = float(os.environ.get("TRIWORLD_BANNER_FPS", "20"))
            _animate_rgb_banner(art, frames=frames, fps=fps)
        else:
            palette = [_CYAN, _CYAN + _BOLD, _BLUE + _BOLD, _MAGENTA, _MAGENTA + _BOLD]
            for i, line in enumerate(art):
                color = palette[min(i * len(palette) // max(len(art), 1), len(palette) - 1)]
                self._println(_c(color, line, self.color))

        # decorative frame + subtitle
        subtitle = " ◈  neural metric orchestrator  ·  protocol 0710  ·  Triworld  "
        frame_w = min(w - 2, max(len(subtitle) + 4, 58))
        inner = frame_w - 2
        pad = max(0, inner - len(subtitle))
        left = pad // 2
        mid_plain = "│" + " " * left + subtitle + " " * (pad - left) + "│"
        top = "╭" + "─" * inner + "╮"
        bot = "╰" + "─" * inner + "╯"
        self._println(_c(_CYAN + _DIM, top, self.color))
        self._println(_c(_CYAN + _DIM, _c(_DIM, mid_plain, self.color), self.color))
        self._println(_c(_CYAN + _DIM, bot, self.color))
        self._println()

    def config_panel(self, ctx: Any, metrics: list[str]) -> None:
        gpu = ctx.env.get("CUDA_VISIBLE_DEVICES", "?")
        rows = [
            ("method", str(ctx.method)),
            ("input_ts", str(ctx.input_ts or "-")),
            ("eval_input", str(ctx.eval_input)),
            ("run_dir", str(ctx.run_dir)),
            ("gpu_list", gpu),
            ("parallel/gpu", str(ctx.parallel_per_gpu)),
            ("metrics", f"{len(metrics)} selected"),
            ("resume", "on" if ctx.resume else "off"),
        ]
        self._println()
        self._println(_c(_CYAN + _BOLD, f"  {self._icon('info')} RUN CONFIG", self.color))
        self._println(_c(_DIM, "  " + "─" * 58, self.color))
        for key, val in rows:
            label = _c(_BLUE, f"  {key:<14}", self.color)
            self._println(f"{label} {_c(_WHITE, val, self.color)}")
        self._println(_c(_DIM, "  " + "─" * 58, self.color))
        if len(metrics) <= 8:
            metric_line = "  ".join(
                f"{self._metric_icon(m)}{_c(_DIM, m, self.color)}" for m in metrics
            )
            self._println(f"  {metric_line}")
        self._println()

    def set_total_steps(self, total: int) -> None:
        self._total = total
        self._current = 0

    def _pretty_step_name(self, name: str) -> str:
        if name.startswith("metric_"):
            parts = name.split("_", 2)
            if len(parts) == 3:
                idx, metric = parts[1], parts[2]
                return f"M{idx} {self._metric_icon(metric)} {metric}"
        return name

    def step_start(self, name: str, *, kind: str = "run", icon: str | None = None) -> None:
        self._current += 1
        sym = icon or self._icon(kind)
        badge = self._step_badge()
        label = _c(_MAGENTA + _BOLD, self._pretty_step_name(name), self.color)
        self._println(f"  {badge} {sym} {label} {_c(_DIM, '…', self.color)}")
        self._steps.append(StepRecord(name=name, status="running"))

    def step_done(self, name: str, *, log_path: Path | None = None, elapsed: float | None = None) -> None:
        dt = elapsed if elapsed is not None else 0.0
        for rec in reversed(self._steps):
            if rec.name == name and rec.status == "running":
                rec.status = "done"
                rec.elapsed = dt
                rec.log_path = log_path
                break
        sym = _c(_GREEN, self._icon("done"), self.color)
        badge = self._step_badge()
        time_str = _c(_DIM, f"{self._icon('clock')} {self._elapsed(dt)}", self.color)
        log_str = ""
        if log_path:
            log_detail = f"{self._icon('file')} {log_path}"
            log_str = f"  {_c(_DIM, log_detail, self.color)}"
        self._println(
            f"  {badge} {sym} {_c(_BOLD, self._pretty_step_name(name), self.color)}  {time_str}{log_str}"
        )

    def step_skip(self, name: str, reason: str = "") -> None:
        self._current += 1
        sym = _c(_YELLOW, self._icon("skip"), self.color)
        badge = self._step_badge()
        detail = f"  {_c(_DIM, reason, self.color)}" if reason else ""
        self._println(f"  {badge} {sym} {_c(_DIM, self._pretty_step_name(name), self.color)}{detail}")
        self._steps.append(StepRecord(name=name, status="skip", detail=reason))

    def step_fail(self, name: str, error: str = "") -> None:
        for rec in reversed(self._steps):
            if rec.name == name and rec.status == "running":
                rec.status = "fail"
                rec.detail = error
                break
        sym = _c(_RED, self._icon("fail"), self.color)
        badge = self._step_badge()
        self._println(f"  {badge} {sym} {_c(_RED + _BOLD, name, self.color)}  {_c(_RED, error, self.color)}")

    def pipeline_progress(self) -> None:
        """Overall pipeline progress bar."""
        if self._total <= 0:
            return
        done = sum(1 for s in self._steps if s.status in ("done", "skip"))
        bar = self._progress_bar(done, self._total)
        elapsed = _c(_DIM, self._elapsed(time.time() - self._t0), self.color)
        self._println(_c(_DIM, f"  pipeline {bar}  {elapsed}", self.color))

    def live_metric_progress(
        self,
        label: str,
        processed: int,
        total: int,
        *,
        items: tuple[int, int] | None = None,
        active: int | None = None,
    ) -> None:
        """In-place progress update (\\r) for long-running metric jobs."""
        if not sys.stdout.isatty():
            return
        bar = self._progress_bar(processed, total, width=24)
        extra = ""
        if active:
            extra = f"  run {active}"
        line = (
            f"\r  {_c(_CYAN, '◉', self.color)} {_c(_BOLD, self._pretty_step_name(label), self.color)}"
            f"  {bar}{extra}  {_c(_DIM, self._elapsed(time.time() - self._t0), self.color)}"
        )
        sys.stdout.write(line)
        sys.stdout.flush()
        self._live_line = line

    def pipeline_start(self) -> None:
        self._println(
            _c(_CYAN + _BOLD, f"  {self._icon('launch')} PIPELINE START", self.color)
        )
        self._println()

    def finish_summary(self, ctx: Any, *, episode_csv: Path, summary_csv: Path) -> None:
        self._clear_live()
        total_elapsed = time.time() - self._t0
        done = sum(1 for s in self._steps if s.status == "done")
        skipped = sum(1 for s in self._steps if s.status == "skip")
        failed = sum(1 for s in self._steps if s.status == "fail")

        self._println()
        self._println(_c(_GREEN + _BOLD, f"  {self._icon('done')} PIPELINE COMPLETE", self.color))
        self._println(_c(_DIM, "  " + "─" * 58, self.color))
        self._println(
            f"  {self._icon('clock')} total {_c(_BOLD, self._elapsed(total_elapsed), self.color)}"
            f"   │ done {done}  skip {skipped}  fail {failed}"
        )
        self._println(f"  {self._icon('file')} episode  → {_c(_CYAN, str(episode_csv), self.color)}")
        self._println(f"  {self._icon('file')} summary  → {_c(_CYAN, str(summary_csv), self.color)}")
        self._println(f"  {self._icon('info')} metrics  → {_c(_CYAN, str(ctx.final_dir), self.color)}")
        self._println()
        self._restore_terminal()

    def dry_run_list(self, items: list[tuple[str, Path]]) -> None:
        self._println(_c(_YELLOW, f"  {self._icon('info')} DRY RUN — no commands executed", self.color))
        for i, (name, path) in enumerate(items, 1):
            self._println(f"  {_c(_DIM, f'{i:>2}.', self.color)} {self._metric_icon(name)} {name}")
            self._println(f"      {_c(_DIM, str(path), self.color)}")


def _read_progress(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def run_cmd_with_ui(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
    ui: PipelineUI | None = None,
    step_name: str | None = None,
    progress_file: Path | None = None,
    poll_interval: float = 0.8,
) -> float:
    """Run subprocess, optionally streaming progress to UI. Returns elapsed seconds."""
    t0 = time.time()
    if ui and step_name:
        ui.step_start(step_name)

    merged_env = build_subprocess_env(env)
    if progress_file:
        merged_env["TRIWORLD_PROGRESS_FILE"] = str(progress_file)
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        if progress_file.exists():
            progress_file.unlink()

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = log_path.open("w", encoding="utf-8", buffering=1)
        proc = subprocess.Popen(
            cmd,
            env=merged_env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
    else:
        log_fh = None
        proc = subprocess.Popen(cmd, env=merged_env)

    last_emit = 0.0
    while proc.poll() is None:
        time.sleep(0.2)
        now = time.time()
        if ui and progress_file and step_name and now - last_emit >= poll_interval:
            payload = _read_progress(progress_file)
            if payload:
                phase = payload.get("phase")
                label = step_name
                if phase:
                    label = f"{step_name} ({phase})"
                ui.live_metric_progress(
                    label,
                    int(payload.get("processed_frames", 0)),
                    int(payload.get("total_frames", 0)),
                    items=(
                        int(payload.get("processed_items", 0)),
                        int(payload.get("total_items", 0)),
                    ),
                    active=int(payload["active_items"]) if payload.get("active_items") else None,
                )
            last_emit = now

    if log_fh:
        log_fh.close()

    elapsed = time.time() - t0
    if ui:
        ui._clear_live()
    if proc.returncode != 0:
        if ui and step_name:
            ui.step_fail(step_name, f"exit {proc.returncode}")
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    if ui and step_name:
        ui.step_done(step_name, log_path=log_path, elapsed=elapsed)
    return elapsed
