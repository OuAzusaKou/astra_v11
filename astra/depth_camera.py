"""Optional D435i observations for optical-depth judgment only; no robot control/XYZ."""
import hashlib
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image


def save_image(directory, name, array):
    path = Path(directory) / name
    Image.fromarray(array).save(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class DepthTimingError(ValueError):
    def __init__(self, reason, timing):
        super().__init__(reason)
        self.timing = {key: value if np.isfinite(value) else None for key, value in timing.items()}


def save_depth_observation(directory, frame, observation_timestamp, *, now=None):
    """Preserve aligned Z16 data losslessly; the preview uses one fixed scale."""
    now = time.time() if now is None else now
    raw = np.asarray(frame["raw"])
    rgb = np.asarray(frame["rgb"])
    scale = float(frame["depth_scale_m"])
    if raw.dtype != np.uint16 or raw.ndim != 2:
        raise ValueError("depth must be a two-dimensional uint16 image")
    if rgb.dtype != np.uint8 or rgb.shape != (*raw.shape, 3):
        raise ValueError("depth and RealSense RGB dimensions must match")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid depth scale")
    received = float(frame["host_received_at"])
    age = now - received
    stamp = float(np.asarray(observation_timestamp).reshape(-1)[0])
    skew = received - stamp
    sensor_skew = float(frame["color_depth_skew_ms"])
    timing = {"host_received_at": received, "selected_at": now,
              "robot_observation_timestamp": stamp, "age_at_snapshot_s": age,
              "robot_observation_time_delta_s": skew, "color_depth_skew_ms": sensor_skew}
    if not np.isfinite(age) or not -.1 <= age <= .5:
        raise DepthTimingError(f"RealSense frame stale or host clock changed (age_s={age:.3f})", timing)
    if not np.isfinite(skew) or abs(skew) > .5:
        raise DepthTimingError(f"RealSense and robot observations are over 0.5 s apart (delta_s={skew:.3f})", timing)
    if not np.isfinite(sensor_skew) or abs(sensor_skew) > 50:
        raise DepthTimingError(f"RealSense color/depth timestamps are not paired (delta_ms={sensor_skew:.1f})", timing)
    valid = raw > 0
    # Black is invalid; red is near and blue is far, fixed at 0.15 to 2.0 m.
    value = np.clip((raw.astype(np.float64) * scale - .15) / 1.85, 0., 1.)
    preview = np.stack((255 * (1 - value), 255 * (1 - abs(2 * value - 1)),
                        255 * value), axis=-1).astype(np.uint8)
    preview[~valid] = 0
    images = {
        "realsense_rgb": save_image(directory, "realsense_rgb.png", rgb),
        "realsense_depth_preview": save_image(directory, "realsense_depth_preview.png", preview),
    }
    metadata = {
        "status": "available", "purpose": "optical_depth_judgment_only_no_xyz_or_robot_targets",
        "camera_serial": frame["camera_serial"], "frame_number": int(frame["frame_number"]),
        "view": "realsense_rgb", "aligned_to": "realsense_rgb",
        "width": raw.shape[1], "height": raw.shape[0], "depth_scale_m": scale,
        **timing,
        "synchronization": "approximate_host_receive_time_not_hardware_sync",
        "sensor_timestamp_ms": float(frame["sensor_timestamp_ms"]),
        "sensor_timestamp_domain": frame["sensor_timestamp_domain"],
        "color_depth_skew_ms": sensor_skew,
        "valid_fraction": float(valid.mean()), "invalid_raw_value": 0,
        "raw": save_image(directory, "realsense_depth_z16.png", raw),
        "valid_mask": save_image(directory, "realsense_depth_valid.png", valid.astype(np.uint8) * 255),
        "preview": {"near_m": .15, "far_m": 2., "near_color": "red", "far_color": "blue",
                    "invalid_color": "black", "clipped_for_display_only": True},
        "robot_extrinsics": "not_used", "camera_pose_stability": "not_verified",
        "pixel_convention": "x right, y down, origin top-left of realsense_rgb; ROI x0 y0 x1 y1, end exclusive",
        "limitations": "Optical-axis depth is not Euclidean clearance, robot displacement or proof of contact. Zero/occluded pixels are unknown. Cross-frame changes require stable camera and matched visible surfaces.",
    }
    metadata["saved_at"] = time.time()
    return images, metadata


class DepthCamera:
    """One persistent SDK owner; reads recent paired frames without touching ROS."""
    def __init__(self, serial="auto"):
        import pyrealsense2 as rs
        self.rs = rs
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.lock = threading.Lock()
        self.frame = None
        self.error = None
        context = rs.context()
        devices = list(context.query_devices())
        serials = [d.get_info(rs.camera_info.serial_number) for d in devices
                   if "435" in d.get_info(rs.camera_info.name)]
        if serial == "auto":
            if len(serials) != 1:
                raise RuntimeError("--depth-camera auto requires exactly one D435/D435i; specify SDK serial")
            serial = serials[0]
        elif serial not in serials:
            raise RuntimeError("requested D435/D435i SDK serial was not found")
        self.serial = serial
        self.pipeline = rs.pipeline(context)
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        profile = self.pipeline.start(config)
        self.scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.worker = threading.Thread(target=self._read, name="astra-depth-camera", daemon=True)
        self.worker.start()
        if not self.ready.wait(8) or self.error:
            self.close()
            raise RuntimeError(self.error or "RealSense did not provide paired RGB/depth frames within 8 s")

    def _read(self):
        rs = self.rs
        align = rs.align(rs.stream.color)
        try:
            while not self.stop.is_set():
                frames = self.pipeline.wait_for_frames(1500)
                received, monotonic = time.time(), time.monotonic()
                frames = align.process(frames)
                depth, color = frames.get_depth_frame(), frames.get_color_frame()
                if not depth or not color:
                    continue
                if depth.get_frame_timestamp_domain() != color.get_frame_timestamp_domain():
                    raise RuntimeError("RealSense depth/color timestamp domains differ")
                sensor_skew = color.get_timestamp() - depth.get_timestamp()
                # The first SDK frameset can contain color/depth from different
                # instants. Do not announce READY or replace a good frame with it.
                if not np.isfinite(sensor_skew) or abs(sensor_skew) > 50:
                    continue
                frame = {"raw": np.asanyarray(depth.get_data()).copy(),
                         "rgb": np.asanyarray(color.get_data()).copy(),
                         "depth_scale_m": self.scale, "camera_serial": self.serial,
                         "frame_number": depth.get_frame_number(), "host_received_at": received,
                         "received_monotonic": monotonic, "sensor_timestamp_ms": depth.get_timestamp(),
                         "sensor_timestamp_domain": str(depth.get_frame_timestamp_domain()),
                         "color_depth_skew_ms": sensor_skew}
                with self.lock:
                    self.frame = frame
                self.ready.set()
        except Exception as error:
            if not self.stop.is_set():
                self.error = str(error)
            self.ready.set()

    def latest(self):
        with self.lock:
            if self.error:
                raise RuntimeError("RealSense acquisition failed: " + self.error)
            if self.frame is None or time.monotonic() - self.frame["received_monotonic"] > .5:
                raise RuntimeError("RealSense frame is unavailable or stale")
            return self.frame.copy()  # Arrays are immutable after publication.

    def close(self):
        self.stop.set()
        self.worker.join(timeout=2)
        self.pipeline.stop()
