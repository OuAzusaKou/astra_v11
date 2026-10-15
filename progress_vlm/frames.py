"""Timestamp-based five-image assembly and diagnostic progress trends."""
from collections import deque
from dataclasses import dataclass
import math
import threading


@dataclass(frozen=True)
class Frame:
    stamp: float
    received: float
    data: bytes


class Frames:
    def __init__(self, max_age=1., sync_tolerance=.15, history_tolerance=.15):
        self.cameras = {k: deque(maxlen=180) for k in ("main", "left", "right")}
        self.lock = threading.Lock()
        self.max_age, self.sync_tolerance, self.history_tolerance = max_age, sync_tolerance, history_tolerance

    def add(self, camera, frame):
        if not math.isfinite(frame.stamp) or frame.stamp <= 0 or not frame.data:
            return
        with self.lock:
            frames = self.cameras[camera]
            if frames and frame.stamp <= frames[-1].stamp:
                return
            frames.append(frame)

    def latest(self):
        with self.lock:
            return self.cameras["main"][-1] if self.cameras["main"] else None

    def assemble(self, start, monotonic_now, ros_now):
        with self.lock:
            cameras = {k: list(v) for k, v in self.cameras.items()}
        if any(not v for v in cameras.values()):
            raise ValueError("waiting for all three camera topics")
        main = cameras["main"][-1]
        history = min(cameras["main"], key=lambda f: abs(f.stamp - (main.stamp - 1.)))
        if abs(history.stamp - (main.stamp - 1.)) > self.history_tolerance:
            raise ValueError("waiting for one second of main-camera history")
        current = [main] + [min(cameras[k], key=lambda f: abs(f.stamp - main.stamp)) for k in ("left", "right")]
        for f in current:
            if not 0 <= monotonic_now - f.received <= self.max_age or not -.1 <= ros_now - f.stamp <= self.max_age:
                raise ValueError("camera stale or timestamp mismatch")
            if abs(f.stamp - main.stamp) > self.sync_tolerance:
                raise ValueError("three-camera synchronization outside tolerance")
        if start is None or start.stamp > main.stamp:
            raise ValueError("subtask start image unavailable; press Enter")
        return [history, start, *current]


class Trend:
    """Diagnostics only. 'Replay' is a training label, never a robot retry verdict."""
    def __init__(self, seconds=8., min_samples=5, min_gain=10, max_gap=3.):
        self.seconds, self.min_samples, self.min_gain, self.max_gap = seconds, min_samples, min_gain, max_gap
        self.reset()

    def reset(self):
        self.values = deque()
        self.replay_streak = 0

    def add(self, stamp, progress, replay):
        if self.values and stamp <= self.values[-1][0]:
            raise ValueError("duplicate or reversed observation timestamp")
        if self.values and stamp - self.values[-1][0] > self.max_gap:
            self.reset()
        self.values.append((stamp, progress))
        # Retain the sample just before the time-window boundary.
        while len(self.values) > 1 and self.values[1][0] <= stamp - self.seconds:
            self.values.popleft()
        self.replay_streak = self.replay_streak + 1 if replay else 0
        span = stamp - self.values[0][0]
        values = [v for _, v in self.values]
        plateau = len(values) >= self.min_samples and span >= self.seconds and max(values) - min(values) < self.min_gain
        return {"plateau_candidate": plateau, "window_seconds": span, "samples": len(values),
                "replay_streak": self.replay_streak, "diagnostic_only": True}
