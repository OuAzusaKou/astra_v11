"""V4 reads progress evidence; only the runner can select/start the next L2."""
import json
import math
from pathlib import Path
import time

from progress_vlm.protocol import CHECKPOINT_SHA256, PROTOCOL


class ProgressGate:
    allow_final = False
    retain_arrival_valid_history = False
    def __init__(self, path, threshold=100, count=2, min_span=.8,
                 max_gap=2.5, max_age=3., min_l2_seconds=3.):
        if type(threshold) is not int or threshold not in (90, 100):
            raise ValueError("V4 threshold must be 90 or 100")
        if type(count) is not int or not 2 <= count <= 10:
            raise ValueError("V4 needs 2–10 independent predictions")
        self.path = Path(path)
        self.threshold, self.count = threshold, count
        self.min_span, self.max_gap = min_span, max_gap
        self.max_age, self.min_l2_seconds = max_age, min_l2_seconds
        self.offset, self.file_identity = 0, None
        self.key, self.since, self.barrier = None, 0., 0.
        self.highs = []
        self.last_stamp = None
        self.last_request = None
        self.reason = "Waiting for fresh VLM evidence"

    def reset(self, reason):
        self.highs.clear()
        self.reason = reason

    def suspend(self, reason):
        self.since = self.barrier = time.time()
        self.reset(reason)

    @property
    def status(self):
        return {"threshold": self.threshold, "required_count": self.count,
                "consecutive": len(self.highs), "reason": self.reason}

    def _rows(self):
        info = self.path.stat()
        identity = (info.st_dev, info.st_ino)
        if self.file_identity != identity or info.st_size < self.offset:
            self.offset = 0
            self.file_identity = identity
            self.reset("Progress log opened/replaced")
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            for _ in range(128):
                line = handle.readline(65537)
                if not line:
                    return
                if len(line) > 65536:
                    raise ValueError("oversized progress log record")
                if not line.endswith(b"\n"):
                    return  # Wait for the writer to finish the record.
                self.offset = handle.tell()
                yield json.loads(line)
            # Never act on a prefix with unread newer evidence.
            if handle.read(1):
                raise ValueError("progress log backlog; waiting to catch up")

    def poll(self, controller, now=None):
        now = time.time() if now is None else now
        key = (controller.session.id, controller.condition_version, controller.phase)
        if key != self.key:
            self.key, self.since, self.barrier = key, now, now
            self.last_stamp = self.last_request = None
            self.reset("New L2/control phase; waiting for fresh evidence")
        eligible = (controller.settings.version in (4, 5, 6, 7, 8, 10) and controller.mode == "WAM"
                    and controller.state == "AUTO" and not controller.switch_pending()
                    and not controller.cancelled())
        if not eligible:
            self.since = self.barrier = now
            self.reset("Inactive: manual start/pause/takeover has priority")
            return None
        if not self.allow_final and controller.subtask_index + 1 >= len(controller.subtasks):
            self.reset("Final L2: no automatic completion or wraparound")
            return None
        try:
            for row in self._rows():
                self._accept(row, controller, now)
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.reset("VLM unavailable: " + str(error)[:100])
            return None
        return self._decision(controller, now)

    def _decision(self, controller, now):
        if self.highs and now - self.highs[-1]["capture_stamp"] > self.max_age:
            self.reset("VLM evidence expired")
        if len(self.highs) < self.count:
            return None
        span = self.highs[-1]["capture_stamp"] - self.highs[0]["capture_stamp"]
        if span < self.min_span or now - self.since < self.min_l2_seconds:
            self.reason = "High progress; waiting for persistence/minimum L2 time"
            return None
        self.reason = "Confirmed; switching at the runner batch boundary"
        return {"threshold": self.threshold, "request_ids": [r["request_id"] for r in self.highs],
                "progress": [r["progress"] for r in self.highs], "capture_span_seconds": span,
                "latest_capture_stamp": self.highs[-1]["capture_stamp"],
                "condition_version": controller.condition_version, "phase": controller.phase}

    def _accept(self, row, c, now):
        if not isinstance(row, dict):
            raise ValueError("progress record is not an object")
        event = row.get("event")
        if event == "runner_context":
            if row.get("active") is not True or row.get("control_mode") != "WAM":
                self.reset("VLM observer inactive")
            return
        if event != "prediction":
            if event in ("discarded_result", "discarded_old_subtask", "observation_error"):
                self.reset("Invalid/gapped prediction")
            return
        ctx, model = row.get("runner_context") or {}, row.get("model") or {}
        if not isinstance(ctx, dict) or not isinstance(model, dict):
            raise ValueError("invalid progress context")
        numeric = (row.get("time"), row.get("capture_stamp"), row.get("age_on_arrival"))
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in numeric):
            raise ValueError("invalid progress timestamps")
        received, stamp, age = numeric
        bound = (ctx.get("session_id") == c.session.id
                 and type(ctx.get("condition_version")) is int and ctx["condition_version"] == c.condition_version
                 and ctx.get("phase") == c.phase and ctx.get("control_mode") == "WAM"
                 and ctx.get("active") is True and ctx.get("state") in ("AUTO", "EXECUTING", "INFERRING")
                 and row.get("l3") == c.l3 and row.get("l2") == c.l2)
        if not (bound and row.get("valid") is True and row.get("success") is True
                and row.get("protocol") == PROTOCOL and model.get("step") == 40000
                and model.get("checkpoint_sha256") == CHECKPOINT_SHA256
                and self.barrier <= stamp <= received <= now + .05
                and 0 <= now - received
                and (self.retain_arrival_valid_history or now - stamp <= self.max_age)
                and 0 <= received - stamp <= 2.1 and 0 <= age <= 2.):
            self.reset("Rejected stale, mismatched or invalid evidence")
            return
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("missing prediction identity")
        if request_id == self.last_request or (self.last_stamp is not None and stamp <= self.last_stamp):
            self.reset("Repeated/out-of-order camera evidence")
            return
        if self.last_stamp is not None and stamp - self.last_stamp > self.max_gap:
            self.reset("Gap in consecutive evidence")
        self.last_stamp, self.last_request = stamp, request_id
        progress = row.get("progress")
        if type(progress) is not int or progress not in range(10, 101, 10) or type(row.get("anomalous_replay")) is not bool:
            self.reset("Invalid progress/replay value")
            return
        self._prediction(row)

    def _prediction(self, row):
        progress, stamp, request_id = row["progress"], row["capture_stamp"], row["request_id"]
        if progress < self.threshold or row["anomalous_replay"]:
            self.reset("Below threshold or anomalous replay")
            return
        self.highs.append({"request_id": request_id, "progress": progress, "capture_stamp": stamp})
        self.highs = self.highs[-self.count:]
        self.reason = f"Confirming high progress {len(self.highs)}/{self.count}"


class TakeoverGate(ProgressGate):
    """Separate evidence cursor; final L2 remains eligible for recovery."""
    allow_final = True
    retain_arrival_valid_history = True

    def __init__(self, path):
        super().__init__(path)
        self.samples = []
        self.peak = None
        self.peak_stamp = None
        self.cooldown_until = 0.
        self.was_astra = False

    def reset(self, reason):
        super().reset(reason)
        # Lose immediate confirmation on invalid/expired data, not the known L2 high.
        self.samples = []

    @property
    def status(self):
        return {"enabled": True, "replay_count": sum(r["anomalous_replay"] for r in self.samples),
                "sample_count": len(self.samples), "required_replays": 0, "replay_required": False, "window_count": 8,
                "peak": self.peak, "reason": self.reason}

    def poll(self, controller, now=None):
        now = time.time() if now is None else now
        key = (controller.session.id, controller.condition_version, controller.phase)
        if key != self.key:
            # Only a new conditioning/control phase creates a new progress baseline.
            self.peak = self.peak_stamp = None
        if controller.mode == "ASTRA":
            self.was_astra = True
        elif controller.mode == "WAM" and self.was_astra:
            self.was_astra = False
            self.cooldown_until = now + 15.
            self.suspend("Astra hand-back: 15s cooldown")
        return super().poll(controller, now)

    def _prediction(self, row):
        if self.peak is None or row["progress"] > self.peak:
            self.peak, self.peak_stamp = row["progress"], row["capture_stamp"]
        self.samples.append({k: row[k] for k in ("progress", "capture_stamp", "request_id", "anomalous_replay")})
        self.samples = self.samples[-8:]

    def _decision(self, controller, now):
        if self.samples and now-self.samples[-1]["capture_stamp"] > self.max_age:
            self.reset("Takeover evidence expired")
        if not self.samples:
            return None
        last = self.samples[-1]
        stagnant = last["capture_stamp"] - self.peak_stamp
        replays = sum(r["anomalous_replay"] for r in self.samples)
        self.reason = f"No new high {stagnant:.1f}/12s; replay {replays}/8 (diagnostic only)"
        if now < self.cooldown_until:
            self.reason = f"Hand-back cooldown {self.cooldown_until-now:.1f}s"
            return None
        if (now-self.since < 10 or stagnant < 12 or len(self.samples) < 8
                or last["capture_stamp"]-self.samples[0]["capture_stamp"] < 6
                or last["progress"] >= 90):
            return None
        self.reason = "Confirmed: automatic Astra takeover at batch boundary"
        return {"operation": "takeover", "no_new_high_seconds": stagnant, "peak": self.peak,
                "progress": last["progress"], "replay_count": replays, "sample_count": 8,
                "request_ids": [r["request_id"] for r in self.samples],
                "condition_version": controller.condition_version, "phase": controller.phase}
