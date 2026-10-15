"""ROS bridge using existing publishers; never starts/resets/stops ROS hardware."""
from pathlib import Path
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np


HW_NAMES = ("leo_head_control", "left_leo_arm_control", "right_leo_arm_control",
            "left_hand_control", "right_hand_control")


class ModeMonitor:
    """Latch hardware interruption even if hardware recovers during inference."""
    def __init__(self, max_age=3.):
        self.max_age = max_age
        self.values = {}
        self.lock = threading.Lock()
        self.armed = False
        self.incident = None

    def update(self, name, value):
        with self.lock:
            self.values[name] = (value, time.monotonic())
            if self.armed and value != "RUNNING":
                self.incident = f"{name} entered {value}"

    def status(self):
        with self.lock:
            return {"modes": {k: v[0] for k, v in self.values.items()}, "incident": self.incident}

    def check(self, *, rearm=False):
        with self.lock:
            now = time.monotonic()
            bad = [name for name in HW_NAMES if name not in self.values
                   or self.values[name][0] != "RUNNING"
                   or now - self.values[name][1] > self.max_age]
            if bad:
                if self.armed:
                    self.incident = f"hardware unavailable/stale: {bad}"
                raise RuntimeError(f"hardware not ready: {bad}")
            if rearm:
                self.incident = None
            if self.incident:
                raise RuntimeError(self.incident + "; operator resume required")
            self.armed = True


def hand_limits():
    root = Path(__file__).resolve().parents[1] / "mano/assets/robots/wuji"
    lower, upper = [], []
    for side in ("left", "right"):
        joints = {j.attrib["name"]: j.find("limit") for j in ET.parse(root / f"wuji_{side}_hand.urdf").getroot().findall("joint")}
        for finger in range(1, 6):
            for joint in range(1, 5):
                limit = joints[f"{side}_hand_finger{finger}_joint{joint}"]
                lower.append(float(limit.attrib["lower"]))
                upper.append(float(limit.attrib["upper"]))
    return np.array(lower), np.array(upper)


def validate_trajectory(batch, current, lower, upper, *, max_step_m, max_step_rad, enforce_hand_limits=True):
    from scipy.spatial.transform import Rotation
    if batch.ndim != 2 or batch.shape[1] != 52 or not np.isfinite(batch).all():
        raise ValueError("invalid absolute action trajectory")
    if enforce_hand_limits and (np.any(batch[:, 12:] < lower) or np.any(batch[:, 12:] > upper)):
        raise ValueError("hand target outside bundled URDF limits")
    path = np.concatenate([current[None, :], batch], axis=0)
    for offset in (0, 6):
        if np.any(np.linalg.norm(np.diff(path[:, offset:offset+3], axis=0), axis=1) > max_step_m):
            raise ValueError("EE translation jump exceeds configured guard")
        rotations = Rotation.from_rotvec(path[:, offset+3:offset+6])
        delta = rotations[:-1].inv() * rotations[1:]
        if np.any(delta.magnitude() > max_step_rad):
            raise ValueError("EE rotation jump exceeds configured guard")


class LeoRobot:
    def __init__(self, env, monitor, stop, *, fps=20., history=(0, 10, 30),
                 max_age=1., max_step_m=.10, max_step_rad=.75, lead_time=.1,
                 settle_time=.3, enforce_hand_limits=False, enforce_review_state=False):
        self.env, self.monitor, self.stop = env, monitor, stop
        self.fps, self.history, self.max_age = fps, history, max_age
        self.max_step_m, self.max_step_rad = max_step_m, max_step_rad
        self.lead_time, self.settle_time = lead_time, settle_time
        self.reset_target = None
        self.reset_path = None
        self.motion_deadline = 0.
        self.lower, self.upper = hand_limits()
        self.enforce_hand_limits = enforce_hand_limits
        self.enforce_review_state = enforce_review_state
        self.review_hand_tolerance_rad = .05

    def status(self):
        return self.monitor.status()

    def check(self):
        if self.stop.is_set():
            raise InterruptedError("runner stopping")
        self.monitor.check()
        for topic in ("/leo/arm_action", "/leo/left_hand_action", "/leo/right_hand_action"):
            if len(self.env._node.get_publishers_info_by_topic(topic)) != 1:
                raise RuntimeError(f"competing or missing publisher on {topic}")

    def reset_to_task_pose(self):
        """Explicit B action: same task targets/timings as the legacy startup.

        No hardware reset service: it can clear modes and reports completion
        before checking success. Keep the existing pedal interlock active.
        """
        self.check()
        if time.monotonic() < self.motion_deadline:
            raise RuntimeError("previous trajectory has not drained")
        target = self.reset_target
        if target is None:
            raise RuntimeError("task reset pose unavailable")
        if np.any(target[12:] < self.lower) or np.any(target[12:] > self.upper):
            raise ValueError("reset hand target outside URDF limits")
        stamp = self.env._node.get_clock().now().to_msg()
        self.motion_deadline = time.monotonic() + 5.3
        self.env.publish_hand_chunk(target[None, 12:], [1.0], cancel_previous=True, start_stamp=stamp)
        self.check()
        self.env.publish_arm_chunk(target[None, :12], [5.0], cancel_previous=True, start_stamp=stamp)
        self.drain(self.stop.is_set)
        from scipy.spatial.transform import Rotation
        deadline = time.monotonic() + 3.
        while True:
            obs = self.observe()
            current = np.concatenate([obs[k][0] for k in
                ("left_ee_pose", "right_ee_pose", "left_hand_joint", "right_hand_joint")])
            good = np.max(np.abs(current[12:] - target[12:])) <= .15
            for offset in (0, 6):
                good = good and np.linalg.norm(current[offset:offset+3] - target[offset:offset+3]) <= .03
                delta = Rotation.from_rotvec(current[offset+3:offset+6]).inv() * Rotation.from_rotvec(target[offset+3:offset+6])
                good = good and delta.magnitude() <= .15
            if good:
                return {"reset_state_path": self.reset_path, "state_verified": True,
                        "position_tolerance_m": .03, "rotation_tolerance_rad": .15,
                        "hand_tolerance_rad": .15}
            if time.monotonic() >= deadline:
                raise TimeoutError("reset target not reached; remain paused")
            if self.stop.wait(.1):
                raise InterruptedError("reset interrupted")

    def rearm(self):
        # Never reset hardware or approve cached targets after pedal intervention.
        if time.monotonic() < self.motion_deadline:
            raise RuntimeError("previous issued batch may still be active")
        self.monitor.check(rearm=True)

    def observe(self):
        self.check()
        obs = self.env.get_obs(self.history, timeout=3., cancel_event=self.stop)
        now = self.env._node.get_clock().now().nanoseconds / 1e9
        for stamp in (obs["timestamp"], obs["timestamp"] - obs["wrist_left_sync_error"],
                      obs["timestamp"] - obs["wrist_right_sync_error"]):
            if not np.isfinite(stamp) or not -.1 <= now - stamp <= self.max_age:
                raise RuntimeError("camera observation stale or clock mismatch")
        if (not self.env.hand_joint_deque
                or not -.1 <= now - self.env.hand_joint_deque[-1]["timestamp"] <= self.max_age):
            raise RuntimeError("hand state stale")
        # Existing EE buffer timestamps sampling time, so inspect source TF too.
        import rclpy.time
        for frame in ("left_link_ee", "right_link_ee"):
            tf = self.env._tf_buffer.lookup_transform("chest_link", frame, rclpy.time.Time())
            stamp = tf.header.stamp.sec + tf.header.stamp.nanosec / 1e9
            if not -.1 <= now - stamp <= self.max_age:
                raise RuntimeError("arm source TF stale")
        for key, width in (("left_ee_pose", 6), ("right_ee_pose", 6),
                           ("left_hand_joint", 20), ("right_hand_joint", 20)):
            arr = np.asarray(obs[key])
            if arr.ndim != 2 or arr.shape[1] != width or not len(arr) or not np.isfinite(arr).all():
                raise ValueError(f"invalid state: {key}")
        return obs

    def validate(self, batch, reviewed_observation):
        current_obs = self.observe()
        # Always validate trajectory jumps against fresh measured state.
        current = np.concatenate([current_obs[k][0] for k in
            ("left_ee_pose", "right_ee_pose", "left_hand_joint", "right_hand_joint")])
        reviewed = np.concatenate([reviewed_observation[k][0] for k in
            ("left_ee_pose", "right_ee_pose", "left_hand_joint", "right_hand_joint")])
        delta = np.abs(current - reviewed)
        limits = np.full(52, .05)
        limits[12:] = self.review_hand_tolerance_rad
        if np.any(delta > limits):
            index = int(np.argmax(delta / limits))
            names = [f"{side}_ee_{axis}" for side in ("left", "right")
                     for axis in ("x", "y", "z", "rx", "ry", "rz")]
            names += [f"{side}_hand_finger{finger}_joint{joint}"
                      for side in ("left", "right") for finger in range(1, 6) for joint in range(1, 5)]
            detail = (f"{names[index]} reviewed={reviewed[index]:.6f} "
                      f"current={current[index]:.6f} delta={delta[index]:.6f} limit={limits[index]:.6f}")
            if self.enforce_review_state:
                raise RuntimeError("robot state changed during review; observe/replan required: " + detail)
            print("[CLIENT] Review-state change (log only): " + detail, flush=True)
        validate_trajectory(batch, current, self.lower, self.upper,
                            max_step_m=self.max_step_m, max_step_rad=self.max_step_rad,
                            enforce_hand_limits=self.enforce_hand_limits)

    def execution_camera_frames(self):
        """Read decoded camera buffers without consuming the next control observation."""
        return {name: list(self.env.camera_deques[name])
                for name in ('image', 'image_wrist_left', 'image_wrist_right')}

    def publish(self, batch):
        self.check()
        if self.env._resetting:
            raise RuntimeError("ROS environment is resetting")
        # Review waits invalidate observation-clock deadlines. Start a NEW short
        # synchronous schedule, sharing one ROS origin across arms and hands.
        offsets = self.lead_time + (np.arange(len(batch)) + 1) / self.fps
        stamp = self.env._node.get_clock().now().to_msg()
        self.motion_deadline = time.monotonic() + offsets[-1] + self.settle_time
        self.env.publish_arm_chunk(batch[:, :12], offsets, cancel_previous=True, start_stamp=stamp)
        self.check()
        self.env.publish_hand_chunk(batch[:, 12:], offsets, cancel_previous=True, start_stamp=stamp)
        return {"published_rows": len(batch), "start_stamp": [stamp.sec, stamp.nanosec],
                "time_from_start": offsets.tolist(), "hardware_execution_confirmed": False}

    def drain(self, cancelled):
        while time.monotonic() < self.motion_deadline:
            # Do not send an invented zero/empty cancellation trajectory. The
            # ROS ESTOP pipeline handles hardware faults; this process stops issuing.
            self.monitor.check()
            time.sleep(min(.02, max(0., self.motion_deadline - time.monotonic())))


def create_robot(args, config, stop):
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.signals import SignalHandlerOptions
    from std_msgs.msg import String
    from envs.leo_wuji_env_infer import LeoWujiInferEnv

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Node("leo_astra_runner")
    executor = SingleThreadedExecutor()
    env = LeoWujiInferEnv(node, fps=args.fps,
        camera_latency=float(config["runtime"].get("camera_latency", .100)),
        wrist_camera_latency=float(config["runtime"].get("wrist_camera_latency", .040)))
    monitor = ModeMonitor()
    for name in HW_NAMES:
        node.create_subscription(String, f"/leo/{name}/mode",
                                 lambda msg, n=name: monitor.update(n, msg.data), 10)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()

    def cleanup():
        executor.shutdown()
        thread.join(timeout=3.)
        env.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    try:
        deadline = time.monotonic() + args.sensor_timeout
        while not stop.is_set():
            if env.is_ready():
                try:
                    monitor.check()
                    break
                except RuntimeError:
                    pass
            if time.monotonic() >= deadline:
                raise TimeoutError("ROS cameras/state/hardware not ready; no reset attempted")
            stop.wait(.1)
        if stop.is_set():
            raise InterruptedError("stopped waiting for ROS")
        robot = LeoRobot(env, monitor, stop, fps=args.fps,
                        history=config["runtime"]["history_offsets"],
                        max_step_m=args.max_step_m, max_step_rad=args.max_step_rad,
                        enforce_hand_limits=args.client_hand_limits,
                        enforce_review_state=args.review_state_guard)
        print(f"[CLIENT] Hand target URDF rejection: {'ON' if args.client_hand_limits else 'OFF'}; ROS driver limits unchanged", flush=True)
        print(f"[CLIENT] Review-state rejection: {'ON' if args.review_state_guard else 'OFF (log only)'}", flush=True)
        from .reset_pose import load_reset_pose
        robot.reset_path, robot.reset_target = load_reset_pose(config["startup"]["reset_state_path"])
        return robot, cleanup
    except BaseException:
        cleanup()
        raise
