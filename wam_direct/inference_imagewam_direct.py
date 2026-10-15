#!/usr/bin/env python3
"""Interactive, direct ImageWAM rollout for the Leo-700h checkpoints.

This is intentionally a thin control layer over the maintained unified-v2.2
Leo client in ``Liberai_Deploy``.  The base client owns camera/state assembly,
the canonical 326-D wire contract, MANO-to-q20 conversion, chest-frame action
decoding, and ROS publishing.  This module adds only:

* numeric model modes (0..10, excluding unused mode 4 when desired);
* free-form two-stage Mode-2 prompts with automatic ``and`` splitting;
* a chunk-boundary ``P`` control for replacing the active subtask; and
* checkpoint/adapter validation so a mode cannot silently use the wrong server.

Modes 0..4 use direct WAM policies. Modes 5..10 use the remote hierarchical
policy server for automatic planning and subtask transitions.
"""

from __future__ import annotations

import argparse
import atexit
from collections import deque
import json
import os
import re
import select
import subprocess
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence


@dataclass(frozen=True)
class ModeSpec:
    number: int
    label: str
    checkpoint_marker: str
    prompt_kind: str


MODE_SPECS = {
    0: ModeSpec(
        0,
        "L3-only baseline",
        "iwam-leo-700h-h30-joint-bs1024-6566",
        "l3",
    ),
    1: ModeSpec(
        1,
        "L2-only",
        "iwam-leo-700h-h30-joint-bs1024-l2-canonical-7092",
        "l2",
    ),
    2: ModeSpec(
        2,
        "L3+L2",
        "iwam-leo-700h-h30-joint-bs1024-l3-l2-canonical-labeled-7114",
        "l3_l2",
    ),
    3: ModeSpec(
        3,
        "L3+L2+Progress (detach)",
        "iwam-leo-700h-h30-joint-bs1024-l3-l2-canonical-labeled-progress-a1-7117",
        "l3_l2",
    ),
    4: ModeSpec(
        4,
        "L3+L2+Progress (backprop)",
        "iwam-leo-700h-h30-joint-bs1024-l3-l2-canonical-labeled-progress-a1-backprop-7489",
        "l3_l2",
    ),
    5: ModeSpec(
        5,
        "V3 mix 200k memory",
        "iwam-leo-700h-h30-joint-bs1024-l3-l2-canonical-labeled-7114",
        "l3",
    ),
    6: ModeSpec(
        6,
        "V3 only 200k memory",
        "iwam-leo-700h-h30-joint-bs1024-l3-l2-canonical-labeled-7114",
        "l3",
    ),
    7: ModeSpec(
        7,
        "V2 mix 200k no-memory",
        "iwam-leo-700h-h30-joint-bs1024-l3-l2-canonical-labeled-7114",
        "l3",
    ),
    8: ModeSpec(
        8,
        "V3 mix 300k memory",
        "iwam-leo-700h-h30-joint-bs1024-l3-l2-canonical-labeled-7114",
        "l3",
    ),
    9: ModeSpec(
        9,
        "V3 mix first-L2 200k memory",
        "iwam-leo-700h-h30-joint-bs1024-l3-l2-canonical-labeled-7114",
        "l3",
    ),
    10: ModeSpec(
        10,
        "V5 mix 200k memory with L3-start and 1s execution history",
        "iwam-leo-700h-h30-joint-bs1024-l3-l2-canonical-labeled-7114",
        "l3",
    ),
}

BENCHMARK_TASKS: dict[int, tuple[str, tuple[str, ...]]] = {
    1: (
        "Pick up the red pepper and place it in the bowl.",
        ("pick up the red pepper", "place the red pepper in the bowl"),
    ),
    2: (
        "Pick up the cup and place it on the plate.",
        ("pick up the cup", "place the cup on the plate"),
    ),
    3: (
        "Pick up the cup and place it in the bowl.",
        ("pick up the cup", "place the cup in the bowl"),
    ),
    4: (
        "Pick up the red pepper and place it on the plate.",
        ("pick up the red pepper", "place the red pepper on the plate"),
    ),
    5: (
        "Pick up the banana and place it on the plate.",
        ("pick up the banana", "place the banana on the plate"),
    ),
    6: (
        "Pick up the green apple and place it in the bowl.",
        ("pick up the green apple", "place the green apple in the bowl"),
    ),
    7: (
        "Pick up the red pepper and place it in the bowl, then pick up the cup and place it on the plate.",
        (
            "pick up the red pepper",
            "place the red pepper in the bowl",
            "pick up the cup",
            "place the cup on the plate",
        ),
    ),
    8: (
        "Pick up the green apple and place it in the bowl, then pick up the red pepper and place it on the plate.",
        (
            "pick up the green apple",
            "place the green apple in the bowl",
            "pick up the red pepper",
            "place the red pepper on the plate",
        ),
    ),
    9: (
        "Pick up the red pepper and place it in the bowl, pick up the green apple and place it in the bowl, then pick up the cup and place it on the plate.",
        (
            "pick up the red pepper",
            "place the red pepper in the bowl",
            "pick up the green apple",
            "place the green apple in the bowl",
            "pick up the cup",
            "place the cup on the plate",
        ),
    ),
    10: (
        "Pick up the red pepper and place it in the bowl, pick up the banana and place it in the bowl, then pick up the cup and place it on the plate, and pick up the green apple and place it on the plate.",
        (
            "pick up the red pepper",
            "place the red pepper in the bowl",
            "pick up the banana",
            "place the banana in the bowl",
            "pick up the cup",
            "place the cup on the plate",
            "pick up the green apple",
            "place the green apple on the plate",
        ),
    ),
}


def _normalize_task(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).rstrip(".").casefold()


def _task_id_from_l3(l3: str) -> int | None:
    normalized = _normalize_task(l3)
    for task_id, (task, _) in BENCHMARK_TASKS.items():
        if _normalize_task(task) == normalized:
            return task_id
    return None


def _task_id_from_video_name(video_name: str | None) -> int | None:
    if not video_name:
        return None
    match = re.match(r"^task(?P<task_id>\d+)(?:_|$)", video_name, re.IGNORECASE)
    if match is None:
        return None
    task_id = int(match.group("task_id"))
    return task_id if task_id in BENCHMARK_TASKS else None


RECORD_ROOT = Path("/home/liberai/wzn/vla_inference/rollout")


class ProgressPlanComplete(Exception):
    """Raised when the final planned subtask reaches the progress threshold."""


_L3_L2_PATTERN = re.compile(
    r"^Task:\s*(?P<task>.+?)\s+Subtask:\s*(?P<subtask>.+?)\s*$",
    re.DOTALL,
)
_TWO_STAGE_AND_PATTERN = re.compile(r"\s+\band\b\s+", re.IGNORECASE)


def _split_two_stage_l3(value: str) -> tuple[str, list[str]]:
    """Split a plain two-stage L3 at its first standalone ``and`` token."""
    task = str(value).strip()
    parts = _TWO_STAGE_AND_PATTERN.split(task, maxsplit=1)
    if len(parts) != 2 or any(not part.strip() for part in parts):
        raise ValueError(
            "mode 2 plain L3 must be '<first L2> and <second L2>'; "
            "alternatively pass the exact structured prompt "
            "'Task: <L3> Subtask: <L2>'"
        )
    return task, [part.strip() for part in parts]


class InteractivePrompt:
    """Own the current model prompt and one-key chunk-boundary switching."""

    def __init__(self, mode: ModeSpec, prompt: str, plan: list[str] | None = None) -> None:
        self.mode = mode
        self._task: str | None = None
        self._plan = plan
        self._plan_index = 0
        self._prompt = self._normalize_initial_prompt(prompt)
        self._pause_requested = threading.Event()
        self._reader_suspended = threading.Event()
        self._stop = threading.Event()
        self._reader: threading.Thread | None = None
        self._fd: int | None = None
        self._terminal_attrs = None

    @property
    def current(self) -> str:
        return self._prompt

    @property
    def current_subtask(self) -> str:
        if self.mode.prompt_kind in ("l2", "l3"):
            return self._prompt
        match = _L3_L2_PATTERN.fullmatch(self._prompt)
        if match is None:
            raise RuntimeError(f"invalid live prompt: {self._prompt!r}")
        return match.group("subtask").strip()

    def _normalize_initial_prompt(self, prompt: str) -> str:
        value = str(prompt).strip()
        if not value:
            raise ValueError("prompt must not be empty")
        if self.mode.prompt_kind in ("l2", "l3"):
            return value
        match = _L3_L2_PATTERN.fullmatch(value)
        if match is None:
            raise ValueError(
                "mode 2/3/4 prompt must be exactly: "
                "Task: <L3> Subtask: <L2>"
            )
        self._task = match.group("task").strip()
        subtask = match.group("subtask").strip()
        if not self._task or not subtask:
            raise ValueError("Task and Subtask must both be non-empty")
        return self._format_l3_l2(subtask)

    def _format_l3_l2(self, subtask: str) -> str:
        if self._task is None:
            raise RuntimeError("L3 task is unavailable for this mode")
        return f"Task: {self._task} Subtask: {subtask}"

    def replace_subtask(self, subtask: str) -> None:
        value = str(subtask).strip()
        if not value:
            raise ValueError("subtask must not be empty")
        self._prompt = (
            value
            if self.mode.prompt_kind in ("l2", "l3")
            else self._format_l3_l2(value)
        )

    def start_keyboard(self) -> None:
        if self._reader is not None:
            return
        if not sys.stdin.isatty():
            print("[control] stdin is not a TTY; P switching is unavailable", flush=True)
            return
        self._fd = sys.stdin.fileno()
        self._terminal_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        atexit.register(self.stop_keyboard)
        self._reader = threading.Thread(
            target=self._keyboard_loop,
            name="wam-prompt-keyboard",
            daemon=True,
        )
        self._reader.start()
        if self.mode.number in (1, 2):
            print("[control] P=pause after current chunk and show the next planned subtask", flush=True)
            print("[control] Press Enter to confirm the switch; Ctrl+C=exit rollout", flush=True)
        else:
            print("[control] P=pause after current chunk and paste the next subtask", flush=True)
            print("[control] Ctrl+C=exit rollout", flush=True)

    def _keyboard_loop(self) -> None:
        assert self._fd is not None
        while not self._stop.is_set():
            if self._reader_suspended.is_set():
                self._stop.wait(0.05)
                continue
            readable, _, _ = select.select([self._fd], [], [], 0.10)
            if not readable:
                continue
            key = os.read(self._fd, 1).decode("utf-8", "ignore")
            if key in ("p", "P"):
                self._reader_suspended.set()
                self._pause_requested.set()
                print(
                    "\n[control] pause requested; waiting for the current chunk boundary ...",
                    flush=True,
                )

    def apply_pending_switch(self) -> tuple[str, str] | None:
        """Switch at a chunk boundary using a preloaded plan or manual input."""
        if not self._pause_requested.is_set():
            return None
        previous_subtask = self.current_subtask
        self._restore_terminal()
        try:
            if self.mode.number in (1, 2):
                if self._plan is None:
                    print("[control] no plan supplied; press Ctrl+C to exit", flush=True)
                    return None
                if self._plan_index >= len(self._plan) - 1:
                    print("[control] no next planned subtask; press Ctrl+C to exit", flush=True)
                    return None
                next_subtask = self._plan[self._plan_index + 1]
                print(f"[control] next subtask: {next_subtask}", flush=True)
                input("[control] press Enter to switch (Ctrl+C to exit): ")
                self._plan_index += 1
                self.replace_subtask(next_subtask)
            else:
                while True:
                    try:
                        value = input("Paste next subtask: ")
                        self.replace_subtask(value)
                    except ValueError as error:
                        print(f"[control] {error}", flush=True)
                        continue
                    break
            print(f"[control] resumed with prompt: {self.current!r}", flush=True)
        finally:
            self._pause_requested.clear()
            if self._fd is not None and not self._stop.is_set():
                self._terminal_attrs = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
            self._reader_suspended.clear()
        return previous_subtask, self.current_subtask

    def _restore_terminal(self) -> None:
        if self._fd is not None and self._terminal_attrs is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._terminal_attrs)

    def stop_keyboard(self) -> None:
        self._stop.set()
        self._reader_suspended.clear()
        self._restore_terminal()


class RolloutEventLog:
    """Append-only timing metadata paired with one rollout video."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._start_monotonic = time.monotonic()
        self._video_start_monotonic: float | None = None
        self._video_fps: float | None = None
        self._frame_index = -1
        self._lock = threading.Lock()
        self._file = path.open("x", encoding="utf-8", buffering=1)

    def set_video_fps(self, fps: float) -> None:
        with self._lock:
            self._video_fps = float(fps)

    def mark_frame(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._video_start_monotonic is None:
                self._video_start_monotonic = now
            self._frame_index += 1

    def write(self, event: str, **payload) -> None:
        with self._lock:
            now = time.monotonic()
            record = {
                "event": event,
                "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "unix_time_s": time.time(),
                "elapsed_s": now - self._start_monotonic,
                **payload,
            }
            if self._video_start_monotonic is not None:
                record["video_elapsed_s"] = now - self._video_start_monotonic
                record["frame_index"] = self._frame_index
                if self._video_fps is not None:
                    record["video_time_s"] = self._frame_index / self._video_fps
            self._file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()


class H264FrameRecorder:
    """Write every received head-camera frame directly to an H.264 MP4."""

    def __init__(
        self,
        path: Path,
        fps: float,
        event_log: RolloutEventLog,
        subtask_provider: Callable[[], str],
    ) -> None:
        self.path = path
        self.fps = float(fps)
        self.event_log = event_log
        self.subtask_provider = subtask_provider
        self._process: subprocess.Popen | None = None
        self._frame_size: tuple[int, int] | None = None
        self._closed = False
        self._lock = threading.Lock()
        self.event_log.set_video_fps(self.fps)

    def _start(self, width: int, height: int) -> None:
        self._frame_size = (width, height)
        self._process = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "bgr24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                f"{self.fps:g}",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.path),
            ],
            stdin=subprocess.PIPE,
            # Keep terminal Ctrl+C on the rollout process.  Python closes this
            # pipe during cleanup so ffmpeg can flush and finalize the MP4.
            start_new_session=True,
        )

    def _annotate(self, image):
        import cv2

        frame = image.copy()
        text = f"Subtask: {self.subtask_provider()}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, scale, thickness
        )
        max_width = frame.shape[1] - 24
        if text_width > max_width:
            scale *= max_width / text_width
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, scale, thickness
            )
        cv2.rectangle(
            frame,
            (0, 0),
            (text_width + 24, text_height + baseline + 20),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame,
            text,
            (12, text_height + 10),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        return frame

    def write(self, image) -> None:
        with self._lock:
            if self._closed:
                return
            frame = self._annotate(image)
            height, width = frame.shape[:2]
            if self._process is None:
                self._start(width, height)
            elif self._frame_size != (width, height):
                raise RuntimeError(
                    f"head-camera size changed from {self._frame_size} to {(width, height)}"
                )
            assert self._process is not None and self._process.stdin is not None
            self._process.stdin.write(frame.tobytes())
            self.event_log.mark_frame()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            process = self._process
            if process is None:
                return
            assert process.stdin is not None
            process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with code {return_code}")
        print(f"[record] H.264 video finalized -> {self.path}", flush=True)


def _recording_paths(mode: ModeSpec, video_name: str | None) -> tuple[Path, Path]:
    name = (video_name or "").strip()
    if not name:
        name = f"wam_mode{mode.number}_{datetime.now():%Y%m%d_%H%M%S}"
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("video name must be a file name, not a path")
    if not name.lower().endswith(".mp4"):
        name += ".mp4"

    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    video_path = RECORD_ROOT / name
    event_path = video_path.with_suffix(".events.jsonl")
    if video_path.exists() or event_path.exists():
        raise ValueError(f"recording already exists: {video_path.stem}")
    return video_path, event_path


def _parse_wrapper_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    if "-h" in argv or "--help" in argv:
        print(
            "Usage: inference_imagewam_direct.py --mode {0,1,2,3,4,5,6,7,8,9,10} "
            "--prompt TEXT [--port PORT] [--task-id N] [--plan JSON] [--progress-threshold FLOAT] "
            "[--record] [--video-name NAME] "
            "[base-client options]\n\n"
            "Mode 0 prompt: <L3>\n"
            "Mode 1 prompt: <L2>\n"
            "Mode 2 prompt: <first L2> and <second L2> (auto-split; P switches)\n"
            "  Legacy structured prompt is also accepted: Task: <L3> Subtask: <L2>\n"
            "Mode 2 plain prompts auto-split; legacy benchmark/task-ID plans remain supported\n"
            "Mode 3/4: same prompt plus ordered --plan (progress switching)\n"
            "Mode 5/6/7/8/9/10 prompt: <L3> (automatic hierarchical inference)\n"
            "Recording root: /home/liberai/wzn/vla_inference/rollout\n"
            "During rollout press P to replace the subtask; press Ctrl+C to exit."
        )
        raise SystemExit(0)

    parser = argparse.ArgumentParser(
        description="Direct Leo ImageWAM inference with manual subtask switching",
        add_help=False,
    )
    parser.add_argument("--mode", type=int, choices=sorted(MODE_SPECS), required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--port", type=int)
    parser.add_argument("--task-id", type=int, choices=sorted(BENCHMARK_TASKS))
    parser.add_argument(
        "--plan",
        help='Ordered JSON array of L2 subtasks; enables automatic P switching for modes 1/2 and is required for 3/4.',
    )
    parser.add_argument("--progress-threshold", type=float, default=0.9)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--video-name")
    args, forwarded = parser.parse_known_args(argv)
    if args.video_name and not args.record:
        parser.error("--video-name requires --record")
    if "--record_head" in forwarded:
        parser.error("use --record and --video-name instead of --record_head")
    if args.port is not None:
        forwarded = ["--policy_port", str(args.port), *forwarded]
    if args.plan:
        try:
            plan = json.loads(args.plan)
        except json.JSONDecodeError as error:
            parser.error(f"--plan must be a JSON array: {error}")
        if (
            not isinstance(plan, list)
            or not plan
            or any(not isinstance(item, str) or not item.strip() for item in plan)
        ):
            parser.error("--plan must be a non-empty JSON array of L2 strings")
        args.plan = [item.strip() for item in plan]
        if len(set(args.plan)) != len(args.plan):
            parser.error("--plan must not contain duplicate L2 subtasks")

    # A Mode-2 evaluator may provide the same plain L3 used by Modes 0 and 7.
    # For the supported two-stage grammar, derive the exact training prompt and
    # the manual-switch plan automatically. Explicit structured prompts and
    # explicit plans remain supported for backward compatibility.
    if args.mode == 2 and _L3_L2_PATTERN.fullmatch(args.prompt.strip()) is None:
        if args.plan:
            parser.error(
                "plain Mode-2 L3 is auto-split at `and`; do not also pass --plan. "
                "Use the legacy structured prompt to supply an explicit plan"
            )
        if args.task_id is not None:
            parser.error(
                "plain Mode-2 L3 is auto-split at `and`; do not also pass --task-id"
            )
        try:
            task, auto_plan = _split_two_stage_l3(args.prompt)
        except ValueError as error:
            parser.error(str(error))
        args.prompt = f"Task: {task} Subtask: {auto_plan[0]}"
        args.plan = auto_plan
        print(
            f"[plan] mode 2 auto-split: "
            f"{json.dumps(args.plan, ensure_ascii=False)}",
            flush=True,
        )

    if args.task_id is not None and args.mode not in (1, 2):
        parser.error("--task-id is only valid for mode 1/2")
    if args.task_id is not None and args.plan:
        parser.error("use either the built-in --task-id plan or --plan, not both")
    if args.mode in (1, 2) and not args.plan:
        task_id = args.task_id
        prompt_match = _L3_L2_PATTERN.fullmatch(args.prompt.strip())
        if task_id is None and args.mode == 2 and prompt_match is not None:
            task_id = _task_id_from_l3(prompt_match.group("task"))
        if task_id is None:
            task_id = _task_id_from_video_name(args.video_name)
        if task_id is None:
            if args.mode == 2 and prompt_match is None:
                parser.error("mode 2 prompt must be exactly: Task: <L3> Subtask: <L2>")
            return args, forwarded

        expected_task, subtasks = BENCHMARK_TASKS[task_id]
        if args.mode == 2:
            if prompt_match is None:
                parser.error("mode 2 prompt must be exactly: Task: <L3> Subtask: <L2>")
            if _normalize_task(prompt_match.group("task")) != _normalize_task(expected_task):
                parser.error(
                    f"--task-id {task_id} does not match prompt Task: {expected_task}"
                )
        args.task_id = task_id
        args.plan = list(subtasks)
        print(f"[plan] benchmark task {task_id}: {json.dumps(args.plan)}", flush=True)
    if args.mode in (3, 4):
        if not args.plan:
            parser.error("mode 3/4 requires --plan for automatic progress switching")
        if not 0.0 < args.progress_threshold <= 1.0:
            parser.error("--progress-threshold must be in (0, 1]")
    return args, forwarded


def main(argv: Sequence[str] | None = None) -> int:
    wrapper_args, forwarded = _parse_wrapper_args(
        sys.argv[1:] if argv is None else list(argv)
    )
    mode = MODE_SPECS[wrapper_args.mode]
    try:
        prompt = InteractivePrompt(
            mode,
            wrapper_args.prompt,
            wrapper_args.plan if mode.number in (1, 2) else None,
        )
    except ValueError as error:
        raise SystemExit(f"ERROR: {error}") from None

    progress_plan = wrapper_args.plan if mode.number in (3, 4) else None
    selected_plan = wrapper_args.plan if mode.number in (1, 2, 3, 4) else None
    if selected_plan is not None and prompt.current_subtask != selected_plan[0]:
        raise SystemExit(
            "ERROR: the initial prompt must use the first subtask in --plan"
        )

    event_log = None
    if wrapper_args.record:
        try:
            video_path, event_path = _recording_paths(mode, wrapper_args.video_name)
        except ValueError as error:
            raise SystemExit(f"ERROR: {error}") from None
        forwarded = ["--record_head", str(video_path), *forwarded]
        event_log = RolloutEventLog(event_path)
        event_log.write(
            "rollout_start",
            mode=mode.number,
            mode_label=mode.label,
            video_path=str(video_path),
            prompt=prompt.current,
        )
        print(f"[record] video -> {video_path}", flush=True)
        print(f"[record] events -> {event_path}", flush=True)
    recording_state = {
        "subtask": prompt.current_subtask,
        "memory": "",
        "overlay": f"{prompt.current_subtask} | Memory: (none)",
    }


    # Import only after wrapper parsing so --help does not require ROS/native IK.
    from scripts.inference_leo import inference_multiview_leo_sync as base

    original_build_request = base.build_request
    original_send_request = base.DPClient.send_request
    original_validate_server = base.validate_server_inference_profile
    recorder_state: dict[str, H264FrameRecorder] = {}
    mode10_history = {
        "lock": threading.Lock(),
        "tracking": False,
        "pending_frames": 0,
        "execution_start": 0.0,
        "execution_end": 0.0,
        "executed_heads": deque(maxlen=64),
        "l3_start_wire": None,
    }
    if event_log is not None or mode.number == 10:
        original_env = base.LeoWujiInferEnv

        class RuntimeLeoWujiInferEnv(original_env):
            def _camera_worker(self, camera_name):
                cfg = self.camera_configs[camera_name]
                raw_queue = self.camera_raw_queues[camera_name]
                camera_deque = self.camera_deques[camera_name]

                while True:
                    msg = raw_queue.get()
                    receive_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    adjusted_time = receive_time - cfg["latency"]
                    try:
                        image = self.cv_bridge.compressed_imgmsg_to_cv2(
                            msg, desired_encoding="bgr8"
                        )
                        if cfg.get("rotate_180", False):
                            image = base.cv2.rotate(image, base.cv2.ROTATE_180)
                    except Exception as error:
                        self._node.get_logger().error(
                            f"Failed to convert {camera_name} image: {error}"
                        )
                        continue

                    camera_deque.append(
                        {
                            "timestamp": adjusted_time,
                            "image": image,
                            "raw_recv_timestamp": receive_time,
                        }
                    )
                    if camera_name == "image" and "head" in recorder_state:
                        recorder_state["head"].write(image)
                    if camera_name == "image" and mode.number == 10:
                        with mode10_history["lock"]:
                            if (
                                mode10_history["tracking"]
                                and mode10_history["pending_frames"] > 0
                                and mode10_history["execution_start"] <= adjusted_time
                                <= mode10_history["execution_end"]
                            ):
                                mode10_history["executed_heads"].append(image.copy())
                                mode10_history["pending_frames"] -= 1

            def publish_arm_chunk(
                self,
                arm_chunk,
                action_timestamps,
                cancel_previous=False,
                *,
                start_stamp=None,
            ):
                stamp = start_stamp
                if mode.number == 10 and stamp is None:
                    stamp = self._node.get_clock().now().to_msg()
                result = super().publish_arm_chunk(
                    arm_chunk,
                    action_timestamps,
                    cancel_previous=cancel_previous,
                    start_stamp=stamp,
                )
                if mode.number == 10 and mode10_history["tracking"] and not self._resetting:
                    timestamps = base.np.asarray(action_timestamps, dtype=float)
                    scheduled = timestamps[timestamps >= self.arm_exec_latency]
                    origin = stamp.sec + stamp.nanosec * 1e-9
                    with mode10_history["lock"]:
                        mode10_history["pending_frames"] = len(scheduled)
                        if len(scheduled):
                            mode10_history["execution_start"] = origin
                            mode10_history["execution_end"] = (
                                origin + float(scheduled[-1]) - self.arm_exec_latency
                                + 1.0 / self.fps
                            )
                return result

        class ReceivedFrameHeadRecorder:
            def __init__(self, path, fps):
                self._recorder = H264FrameRecorder(
                    Path(path), fps, event_log, lambda: recording_state["overlay"]
                )
                recorder_state["head"] = self._recorder
                print(f"[record] head camera -> {path} @ received-frame rate")

            def write(self, image):
                # Frames are written by the camera receive worker, not the policy loop.
                return None

            def close(self):
                self._recorder.close()

        base.LeoWujiInferEnv = RuntimeLeoWujiInferEnv
        if event_log is not None:
            base._HeadCameraRecorder = ReceivedFrameHeadRecorder

    keyboard_started = False
    subtask_started = False

    def handle_prompt_boundary() -> None:
        nonlocal keyboard_started
        if not keyboard_started:
            if mode.number in (0, 5, 6, 7, 8, 9, 10):
                print("[control] Ctrl+C=exit rollout", flush=True)
            else:
                prompt.start_keyboard()
            keyboard_started = True
        transition = prompt.apply_pending_switch()
        if transition is not None and event_log is not None:
            previous_subtask, current_subtask = transition
            recording_state["subtask"] = current_subtask
            recording_state["memory"] = ""
            recording_state["overlay"] = f"{current_subtask} | Memory: (none)"
            event_log.write(
                "subtask_switch",
                previous_subtask=previous_subtask,
                subtask=current_subtask,
                prompt=prompt.current,
            )

    original_get_obs = base.LeoWujiInferEnv.get_obs

    def get_obs_with_live_prompt(environment, *args, **kwargs):
        # Handle a manual switch before capturing the observation used for the
        # next request, so the new L2 is paired with the current robot state.
        handle_prompt_boundary()
        return original_get_obs(environment, *args, **kwargs)

    base.LeoWujiInferEnv.get_obs = get_obs_with_live_prompt

    def prepare_mode10_head(image):
        from utils.imagewam_contract import _DATASET_FRAME_RESIZERS

        bgr = base.np.asarray(image)
        rgb = base.np.ascontiguousarray(bgr[..., ::-1])
        return _DATASET_FRAME_RESIZERS["head"](rgb)

    def build_request_with_live_prompt(
        observation,
        _startup_prompt,
        *request_args,
        **request_kwargs,
    ):
        nonlocal keyboard_started, subtask_started
        if not keyboard_started:
            if mode.number in (0, 5, 6, 7, 8, 9, 10):
                print("[control] Ctrl+C=exit rollout", flush=True)
            else:
                prompt.start_keyboard()
            keyboard_started = True
        if event_log is not None:
            if not subtask_started:
                event_log.write(
                    "subtask_start",
                    subtask=prompt.current_subtask,
                    prompt=prompt.current,
                )
                subtask_started = True
        print(f"[prompt] {prompt.current}", flush=True)
        request, context = original_build_request(
            observation,
            prompt.current,
            *request_args,
            **request_kwargs,
        )
        if mode.number == 10:
            current_head = base.np.asarray(observation["image"])
            wire_observation = request["observation"]
            with mode10_history["lock"]:
                mode10_history["pending_frames"] = 0
                if mode10_history["l3_start_wire"] is None:
                    mode10_history["l3_start_wire"] = wire_observation[
                        "observation.images.image_ego_0"
                    ].copy()
                    mode10_history["executed_heads"].clear()
                    mode10_history["executed_heads"].append(current_head.copy())
                    mode10_history["tracking"] = True
                start_head = mode10_history["l3_start_wire"].copy()
                history_head = (
                    mode10_history["executed_heads"][-31].copy()
                    if len(mode10_history["executed_heads"]) >= 31
                    else None
                )
            wire_observation["high_level.image_ego_l3_start"] = start_head
            wire_observation["high_level.image_ego_history_1s"] = (
                prepare_mode10_head(history_head)
                if history_head is not None
                else start_head.copy()
            )
        return request, context

    def validate_selected_server(server_info, **kwargs):
        original_validate_server(server_info, **kwargs)
        checkpoint = str(server_info.get("checkpoint") or "")
        adapter_name = server_info.get("adapter_name")

        # The legacy mode-0 server predates server_info metadata.  Validate any
        # metadata it does provide, while accepting the original empty profile.
        if mode.number == 0:
            if checkpoint and mode.checkpoint_marker not in checkpoint:
                raise RuntimeError(
                    f"mode 0 reached the wrong checkpoint: {checkpoint!r}"
                )
            if adapter_name not in (None, "teleop_mano"):
                raise RuntimeError(
                    f"mode 0 requires adapter_name='teleop_mano', "
                    f"got {adapter_name!r}"
                )
            return

        if mode.number in (5, 6, 7, 8, 9, 10):
            actual_hl_mode = server_info.get("hl_mode")
            if actual_hl_mode != mode.number:
                raise RuntimeError(
                    f"mode {mode.number} reached hierarchical mode {actual_hl_mode!r}"
                )

        if mode.checkpoint_marker not in checkpoint:
            raise RuntimeError(
                f"mode {mode.number} expects checkpoint containing "
                f"{mode.checkpoint_marker!r}, got {checkpoint!r}"
            )
        if adapter_name != "teleop_mano":
            raise RuntimeError(
                f"mode {mode.number} requires adapter_name='teleop_mano', "
                f"got {adapter_name!r}"
            )

    def send_with_progress_switching(client, request):
        response = original_send_request(client, request)
        if not response.get("success", False):
            return response

        server_info = response.get("server_info") or {}
        validate_selected_server(
            server_info,
            expected_num_inference_steps=None,
            expected_image_num_inference_steps=None,
        )
        latest_l2 = str(server_info.get("latest_l2") or prompt.current_subtask)
        memory = server_info.get("memory")
        recording_state["subtask"] = latest_l2
        recording_state["memory"] = str(memory or "")
        recording_state["overlay"] = (
            f"{latest_l2} | Memory: {recording_state['memory'] or '(none)'}"
        )
        if event_log is not None and mode.number in (5, 6, 7, 8, 9, 10):
            event_log.write(
                "high_level",
                l3=server_info.get("l3"),
                plan=server_info.get("plan"),
                subtask=latest_l2,
                memory=memory,
                high_level_raw=server_info.get("high_level_raw"),
                wam_instruction=server_info.get("wam_instruction"),
            )
        if progress_plan is None:
            return response
        if "progress" not in server_info:
            raise RuntimeError(
                f"mode {mode.number} server did not return progress; restart it with "
                "the updated ImageWAM policy server"
            )

        progress = float(server_info["progress"])
        current_subtask = prompt.current_subtask
        print(
            f"[progress] {current_subtask}: {progress:.1%} "
            f"(switch at {wrapper_args.progress_threshold:.0%})",
            flush=True,
        )
        if event_log is not None:
            event_log.write(
                "progress",
                subtask=current_subtask,
                progress=progress,
                threshold=wrapper_args.progress_threshold,
            )
        if progress < wrapper_args.progress_threshold:
            return response

        assert progress_plan is not None
        try:
            current_index = progress_plan.index(current_subtask)
        except ValueError:
            raise RuntimeError(
                f"current subtask is not present in the ordered plan: {current_subtask!r}"
            ) from None
        if current_index == len(progress_plan) - 1:
            if event_log is not None:
                event_log.write(
                    "task_complete",
                    subtask=current_subtask,
                    progress=progress,
                )
            raise ProgressPlanComplete

        next_subtask = progress_plan[current_index + 1]
        prompt.replace_subtask(next_subtask)
        recording_state["subtask"] = next_subtask
        recording_state["memory"] = ""
        recording_state["overlay"] = f"{next_subtask} | Memory: (none)"
        print(
            f"[control] progress switch: {current_subtask!r} -> {next_subtask!r}",
            flush=True,
        )
        if event_log is not None:
            event_log.write(
                "subtask_switch",
                trigger="progress",
                previous_subtask=current_subtask,
                subtask=next_subtask,
                progress=progress,
                prompt=prompt.current,
            )

        next_request = dict(request)
        next_request["instruction"] = prompt.current
        return original_send_request(client, next_request)

    base.build_request = build_request_with_live_prompt
    base.validate_server_inference_profile = validate_selected_server
    if mode.number in (5, 6, 7, 8, 9, 10) or progress_plan is not None:
        base.DPClient.send_request = send_with_progress_switching

    # The maintained client still owns all runtime flags.  Supply the initial
    # prompt through its normal CLI while the patched request builder swaps in
    # the live prompt at every chunk boundary.
    original_argv = sys.argv
    sys.argv = [
        original_argv[0],
        "--instruction",
        prompt.current,
        *forwarded,
    ]
    print(f"[mode] {mode.number}: {mode.label}", flush=True)
    print(f"[mode] expected checkpoint: {mode.checkpoint_marker}", flush=True)
    if mode.number in (3, 4):
        print(
            f"[progress] automatic switch threshold: "
            f"{wrapper_args.progress_threshold:.1%}",
            flush=True,
        )
    exit_reason = "completed"
    try:
        base.main()
    except ProgressPlanComplete:
        exit_reason = "task_complete"
        print("\n[control] final subtask completed; ending rollout", flush=True)
        return 0
    except KeyboardInterrupt:
        exit_reason = "keyboard_interrupt"
        print("\n[control] Ctrl+C received; exiting rollout", flush=True)
        return 130
    except BaseException:
        exit_reason = "error"
        raise
    finally:
        prompt.stop_keyboard()
        sys.argv = original_argv
        if event_log is not None:
            event_log.write("rollout_end", reason=exit_reason)
            recorder = recorder_state.get("head")
            if recorder is not None:
                recorder.close()
            event_log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
