"""Record the rendered observer canvas at wall-clock speed in short MP4 segments."""
import json
from pathlib import Path
import time


class WindowRecorder:
    def __init__(self, directory, fps=10, segment_seconds=60):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.fps = fps
        self.segment_frames = round(fps * segment_seconds)
        self.writer = None
        self.previous = None
        self.started = None
        self.frames = 0
        self.segment = 0
        self.closed = False
        self.index = (self.directory / "segments.jsonl").open("x", buffering=1)

    def _write(self, frame):
        import cv2
        if self.frames % self.segment_frames == 0:
            self._finish_segment()
            self.segment += 1
            self.path = self.directory / f"window_{self.segment:04d}.mp4"
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                                          self.fps, (frame.shape[1], frame.shape[0]))
            if not self.writer.isOpened():
                self.writer.release()
                self.writer = None
                raise RuntimeError(f"Cannot open video encoder: {self.path}")
            self.segment_start = self.frames
            self.index.write(json.dumps({"event": "opened", "file": self.path.name,
                "time": time.time(), "offset_seconds": self.frames / self.fps, "fps": self.fps}) + "\n")
        self.writer.write(frame)
        self.frames += 1

    def write(self, canvas, now=None):
        if self.closed:
            raise RuntimeError("Recorder is closed")
        now = time.monotonic() if now is None else now
        if self.started is None:
            self.started = now
        target = int((now - self.started) * self.fps + 1e-6)
        # Fill missed display ticks with the previous visible frame, preserving playback duration.
        while self.frames < target:
            self._write(self.previous)
        if self.frames == target:
            self._write(canvas)
        self.previous = canvas.copy()

    def _finish_segment(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self.index.write(json.dumps({"event": "closed", "file": self.path.name,
                "frames": self.frames - self.segment_start,
                "duration_seconds": (self.frames - self.segment_start) / self.fps}) + "\n")

    def close(self):
        if not self.closed:
            self._finish_segment()
            self.index.close()
            self.closed = True
