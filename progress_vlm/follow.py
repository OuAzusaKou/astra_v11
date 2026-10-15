"""Read-only follower of an existing Astra runner's task and saved observations."""
import json
from pathlib import Path
import time

from .frames import Frame


class SessionFollower:
    def __init__(self, path, max_status_age=15.):
        self.path = Path(path).expanduser().resolve()
        self.session_id = None
        self.offset = 0
        self.first_observations = {}
        self.frames = {}
        self.max_status_age = max_status_age

    def _read_events(self):
        # Read only appended complete lines; bounded work per poll. Never write runner files.
        path = self.path / "events.jsonl"
        if not path.exists():
            return
        with path.open("rb") as handle:
            handle.seek(self.offset)
            budget = 4 * 1024 * 1024
            while budget > 0:
                line = handle.readline(2 * 1024 * 1024)
                if not line or not line.endswith(b"\n"):
                    break
                self.offset = handle.tell()
                budget -= len(line)
                event = json.loads(line)
                if event.get("kind") == "inference":
                    version = event.get("conditioning_version")
                    if type(version) is int:
                        self.first_observations.setdefault(version, event["observation"])

    def poll(self, now=None):
        now = time.time() if now is None else now
        manifest = json.loads((self.path / "manifest.json").read_text())
        status = json.loads((self.path / "status.json").read_text())
        identity = manifest["session_id"]
        if status.get("session_id") != identity or (self.session_id and self.session_id != identity):
            raise ValueError("runner session identity changed or mismatched")
        self.session_id = identity
        l3 = manifest["instruction"]
        l2 = status.get("l2_instruction") or manifest["initial_l2_instruction"]
        version = status.get("condition_version", 0)
        context = {"key": (identity, version), "session_id": identity, "condition_version": version,
                   "l3": l3, "l2": l2, "subtask_index": status.get("subtask_index", 0),
                   "state": status["state"], "control_mode": status.get("control_mode"),
                   "phase": status.get("phase"), "version": status.get("version"),
                   "vlm_auto_l2": status.get("vlm_auto_l2"),
                   "vlm_auto_takeover": status.get("vlm_auto_takeover"),
                   "active": False, "anchor": None, "reason": ""}
        if not 0 <= now - status["updated_at"] <= self.max_status_age:
            context["reason"] = "Waiting: runner status is stale"
            return context
        if status["state"] in ("STARTING", "WAITING_FOR_START", "RESETTING", "PAUSED_FOR_OPERATOR", "FINISHED", "EXITED"):
            context["reason"] = "Runner " + status["state"] + "; use its main terminal"
            return context
        if status["state"] not in ("AUTO", "INFERRING", "EXECUTING", "WAITING"):
            context["reason"] = "Waiting: unknown runner state"
            return context
        self._read_events()
        if version not in self.first_observations:
            context["reason"] = "Waiting for this L2's first WAM observation"
            return context
        if version not in self.frames:
            observation = self.first_observations[version]
            image = Path(observation["images"]["image"]["path"]).resolve()
            image.relative_to(self.path)  # Only read this session's saved anchor.
            self.frames[version] = Frame(float(observation["state"]["timestamp"]), 0., image.read_bytes())
        context.update(active=True, anchor=self.frames[version],
                       reason="Following runner; V4 auto next L2 enabled" if status.get("version") in (4, 5, 6, 7) else "Following runner; diagnostic only")
        return context
