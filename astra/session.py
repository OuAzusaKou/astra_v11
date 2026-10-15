"""Atomic, owner-bound supervision exchange with immutable observation files."""
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np


def atomic_json(path, value, *, exclusive=False):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as out:
            json.dump(value, out, ensure_ascii=False, allow_nan=False, indent=2)
            out.flush()
            os.fsync(out.fileno())
        if exclusive:
            os.link(tmp, path)
        else:
            os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class Session:
    def __init__(self, path, owner, manifest):
        self.path = Path(path).expanduser().resolve()
        self.path.mkdir(parents=True, mode=0o700, exist_ok=False)
        self.owner, self.id = owner, uuid.uuid4().hex
        self.sequence = 0
        self.pending = None
        self.reference_video = None
        self.depth_source = None
        atomic_json(self.path / "manifest.json", {
            **manifest, "session_id": self.id, "owner": owner,
            "created_at": time.time(), "mode": 2,
            "step_semantics": "waypoint_budget_charged_before_publish_not_hardware_execution_ack",
        })

    def event(self, kind, **data):
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps({"kind": kind, "time": time.time(), **data},
                                 ensure_ascii=False, allow_nan=False) + "\n")
            out.flush()
            os.fsync(out.fileno())

    def snapshot(self, obs, label, *, include_depth=False):
        from PIL import Image
        started_at = time.time()
        self.sequence += 1
        directory = self.path / f"obs_{self.sequence:06d}_{label}"
        directory.mkdir()
        depth, depth_images = None, {}
        if include_depth and self.depth_source is not None:
            from .depth_camera import save_depth_observation
            # Select and validate the contemporaneous frame before PNG encoding.
            # The continuous camera keeps advancing while full-size RGBs save.
            try:
                depth_images, depth = save_depth_observation(
                    directory, self.depth_source.latest(), obs["timestamp"])
            except Exception as error:
                depth = {"status": "unavailable", "reason": str(error),
                         "purpose": "optical_depth_judgment_only_no_xyz_or_robot_targets",
                         **getattr(error, "timing", {})}
                self.event("depth_unavailable", reason=str(error), observation=str(directory),
                           timing=getattr(error, "timing", {}))
        images = {}
        for key in ("image", "image_wrist_left", "image_wrist_right"):
            arr = np.asarray(obs[key])
            if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
                raise ValueError(f"invalid BGR image: {key}")
            target = directory / (key + ".png")
            Image.fromarray(arr[..., ::-1]).save(target)
            images[key] = {"path": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        state = {k: np.asarray(obs[k]).tolist() for k in
                 ("timestamp", "left_ee_pose", "right_ee_pose", "left_hand_joint", "right_hand_joint")}
        atomic_json(directory / "state.json", state)
        snapshot = {"images": images, "state": state, "state_path": str(directory / "state.json")}
        if depth is not None:
            images.update(depth_images)
            depth["snapshot_processing_s"] = time.time() - started_at
            depth["metadata_path"] = str((directory / "depth_metadata.json").resolve())
            atomic_json(depth["metadata_path"], depth)
            snapshot["depth"] = depth
            self.event("depth_snapshot", observation=str(directory), depth=depth)
            if depth["status"] == "available":
                print(f"[DEPTH] available frame={depth['frame_number']} "
                      f"skew_ms={depth['robot_observation_time_delta_s'] * 1000:.1f} "
                      f"age_ms={depth['age_at_snapshot_s'] * 1000:.1f}", flush=True)
            else:
                print(f"[DEPTH] unavailable: {depth['reason']}", flush=True)
        return snapshot

    def request(self, **data):
        self.pending = {**data, "session_id": self.id, "owner": self.owner,
                        "request_id": uuid.uuid4().hex, "created_at": time.time()}
        if self.reference_video is not None:
            self.pending["reference_video"] = self.reference_video
        atomic_json(self.path / "request.json", self.pending)
        self.event("request", **self.pending)
        return self.pending

    def status(self, **data):
        atomic_json(self.path / "status.json", {
            **data, "session_id": self.id, "owner": self.owner,
            "request_id": self.pending["request_id"] if self.pending else None,
            "updated_at": time.time(),
        })

    def consume(self, filename):
        path = self.path / filename
        if not path.exists():
            return None
        archive = self.path / f"consumed_{uuid.uuid4().hex}_{filename}"
        os.rename(path, archive)
        if archive.stat().st_size > 2_000_000:
            raise ValueError("response too large")
        return json.loads(archive.read_text(encoding="utf-8"))

    def validate_identity(self, response):
        if not isinstance(response, dict) or self.pending is None:
            raise ValueError("no pending review")
        for key in ("session_id", "owner", "request_id"):
            if response.get(key) != self.pending[key]:
                raise ValueError(f"stale/foreign {key}")


def submit(path, response, *, operator=False):
    directory = Path(path).expanduser().resolve()
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    if status.get("state") in ("FINISHED", "EXITED"):
        raise ValueError("runner is no longer accepting decisions")
    if operator:
        if response.get("session_id") != status["session_id"]:
            raise ValueError("foreign session")
        if response.get("pause_id") != status.get("pause_id"):
            raise ValueError("stale pause")
        name = "operator.json"
    else:
        if status.get("state") != "WAITING":
            raise ValueError("runner is not awaiting a review")
        request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
        for key in ("session_id", "owner", "request_id"):
            if response.get(key) != request[key]:
                raise ValueError(f"response must explicitly match current {key}")
        name = "response.json"
    atomic_json(directory / name, response, exclusive=True)
