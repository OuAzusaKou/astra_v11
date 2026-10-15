"""Inference environment for Leo humanoid robot.

Aligned with the planner pipeline (arm_planner_node / hand_planner_node):
  - Arm actions:  /leo/arm_action (Action — chunk of poses with time_from_start)
  - Hand actions: /leo/{left,right}_hand_action (Trajectory — chunk of JointTrajectoryPoints)
  - Reset:        /arm_ik/reset, /hand_planner/{left,right}/reset services

Observation:
  - Camera images from /cam/{chest,hand_left,hand_right}/image_raw/compressed
  - Arm EE poses via TF (chest_link → {right,left}_link_ee)
  - Hand joints from /joint_states
"""

import cv2
cv2.setNumThreads(2)
import rclpy
from rclpy.node import Node
from collections import deque
import time
import threading
import queue
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import CompressedImage, JointState
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectoryPoint
from leo_interfaces.msg import JointCommand, Action, Trajectory
from std_srvs.srv import Trigger
import tf2_ros

from utils.pose_utils import xyzquat_to_pose, pose6d_to_pose7d
from utils.interpolation_util import PoseInterpolator, get_interp1d

LEFT_HAND_JOINTS = [
    f'left_hand_finger{f}_joint{j}'
    for f in range(1, 6) for j in range(1, 5)
]
RIGHT_HAND_JOINTS = [
    f'right_hand_finger{f}_joint{j}'
    for f in range(1, 6) for j in range(1, 5)
]

_RESET_SERVICES = [
    '/arm_ik/reset',
    '/hand_ik/left/reset',
    '/hand_ik/right/reset',
]

class LeoWujiInferEnv:
    def __init__(self, node, fps=30.0, camera_latency=0.100,
                 wrist_camera_latency=0.040, arm_exec_latency=0.0, hand_exec_latency=0.0):
        self._node = node
        self.fps = fps
        self.camera_latency = camera_latency
        self.wrist_camera_latency = wrist_camera_latency
        self.arm_exec_latency = arm_exec_latency
        self.hand_exec_latency = hand_exec_latency

        self.cv_bridge = CvBridge()

        # ── Subscriptions ───────────────────────────────────────────────────
        self.leo_ee_deque = deque(maxlen=100)

        # ── Multi-camera setup (aligned with leo_wuji_env_depoly.py) ──────
        self.camera_deques = {
            'image': deque(maxlen=10),
            'image_wrist_left': deque(maxlen=10),
            'image_wrist_right': deque(maxlen=10),
        }
        self.last_camera_timestamp = None
        self.camera_configs = {
            'image': {
                'topic': '/cam/chest/image_raw/compressed',
                'msg_type': CompressedImage,
                'latency': self.camera_latency,
                'rotate_180': False,
            },
            'image_wrist_left': {
                'topic': '/cam/hand_left/image_raw/compressed',
                'msg_type': CompressedImage,
                'latency': self.wrist_camera_latency,
                'rotate_180': True,
            },
            'image_wrist_right': {
                'topic': '/cam/hand_right/image_raw/compressed',
                'msg_type': CompressedImage,
                'latency': self.wrist_camera_latency,
                'rotate_180': True,
            },
        }
        self.camera_raw_queues = {
            name: queue.Queue(maxsize=4) for name in self.camera_configs
        }
        camera_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.camera_threads = []
        for camera_name, cfg in self.camera_configs.items():
            self._node.create_subscription(
                cfg['msg_type'],
                cfg['topic'],
                lambda msg, name=camera_name: self._camera_cb(name, msg),
                camera_qos,
            )
            worker = threading.Thread(
                target=self._camera_worker,
                args=(camera_name,),
                daemon=True,
            )
            self.camera_threads.append(worker)
            worker.start()

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._node)
        self._node.create_timer(1 / 100, self._ee_timer_cb)

        wuji_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.hand_joint_deque = deque(maxlen=100)
        self._node.create_subscription(
            JointState, '/joint_states', self._hand_joint_cb, wuji_qos)

        # ── Action publishers (aligned with arm_planner_node / hand_planner_node)
        self._arm_pub = self._node.create_publisher(Action, '/leo/arm_action', 10)
        self._left_hand_pub = self._node.create_publisher(Trajectory, '/leo/left_hand_action', 10)
        self._right_hand_pub = self._node.create_publisher(Trajectory, '/leo/right_hand_action', 10)

        # ── Reset service clients (aligned with IK nodes) ────────────────────
        self._reset_clients = []
        for srv_name in _RESET_SERVICES:
            self._reset_clients.append(self._node.create_client(Trigger, srv_name))

        # ── Reset publishers (fallback, matching IK node output format)
        _reset_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._reset_left_arm_pub = self._node.create_publisher(JointCommand, '/leo/left_arm_commands', _reset_qos)
        self._reset_right_arm_pub = self._node.create_publisher(JointCommand, '/leo/right_arm_commands', _reset_qos)
        self._reset_head_pub = self._node.create_publisher(JointCommand, '/leo/head_commands', _reset_qos)
        self._reset_left_hand_pub = self._node.create_publisher(JointCommand, '/leo/left_hand_commands', _reset_qos)
        self._reset_right_hand_pub = self._node.create_publisher(JointCommand, '/leo/right_hand_commands', _reset_qos)

        self._resetting = False
        self._node.get_logger().info('LeoWujiInferEnv ready.')

    # ── Callbacks ───────────────────────────────────────────────────────────

    def _camera_cb(self, camera_name, msg):
        """Enqueue camera messages so image decoding stays off the executor thread."""
        raw_queue = self.camera_raw_queues[camera_name]
        if raw_queue.full():
            try:
                raw_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            raw_queue.put_nowait(msg)
        except queue.Full:
            pass

    def _camera_worker(self, camera_name):
        """Background thread to decode camera images."""
        cfg = self.camera_configs[camera_name]
        raw_queue = self.camera_raw_queues[camera_name]
        camera_deque = self.camera_deques[camera_name]

        while True:
            msg = raw_queue.get()
            if msg is None:
                return
            ts_recv = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            adjusted_timestamp = ts_recv - cfg['latency']

            try:
                cv_image = self.cv_bridge.compressed_imgmsg_to_cv2(
                    msg, desired_encoding='bgr8'
                )
                if cfg.get('rotate_180', False):
                    cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
            except Exception as e:
                self._node.get_logger().error(
                    f'Failed to convert {camera_name} image: {e}'
                )
                continue

            camera_deque.append({
                'timestamp': adjusted_timestamp,
                'image': cv_image,
                'raw_recv_timestamp': ts_recv,
            })

    def _get_nearest_camera_frame(self, camera_name, ref_timestamp):
        """Find the frame in camera_name's deque closest to ref_timestamp."""
        camera_data = list(self.camera_deques[camera_name])
        if len(camera_data) == 0:
            raise RuntimeError(f"No {camera_name} data available")

        timestamps = np.array([frame['timestamp'] for frame in camera_data], dtype=np.float64)
        nearest_idx = int(np.argmin(np.abs(timestamps - ref_timestamp)))
        nearest_frame = camera_data[nearest_idx]
        sync_error = abs(float(nearest_frame['timestamp']) - float(ref_timestamp))
        return nearest_frame, sync_error

    def _lookup_ee_pose(self, ee_frame: str):
        try:
            tf = self._tf_buffer.lookup_transform('chest_link', ee_frame, rclpy.time.Time())
            pose = Pose()
            pose.position.x = tf.transform.translation.x
            pose.position.y = tf.transform.translation.y
            pose.position.z = tf.transform.translation.z
            pose.orientation.x = tf.transform.rotation.x
            pose.orientation.y = tf.transform.rotation.y
            pose.orientation.z = tf.transform.rotation.z
            pose.orientation.w = tf.transform.rotation.w
            return pose
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None

    def _ee_timer_cb(self):
        right_ee_pose = self._lookup_ee_pose('right_link_ee')
        left_ee_pose = self._lookup_ee_pose('left_link_ee')
        if right_ee_pose is not None and left_ee_pose is not None:
            now = self._node.get_clock().now().to_msg()
            timestamp = now.sec + now.nanosec * 1e-9
            self.leo_ee_deque.append({
                'timestamp': timestamp,
                'right_ee_pose': xyzquat_to_pose(right_ee_pose),
                'left_ee_pose': xyzquat_to_pose(left_ee_pose),
            })

    def _hand_joint_cb(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.hand_joint_deque.append({
            'timestamp': timestamp,
            'left_hand_joint': msg.position[2:22],
            'right_hand_joint': msg.position[29:49],
        })

    # ── Observation ─────────────────────────────────────────────────────────

    def get_obs(self, his = [0], *, timeout=None, cancel_event=None):
        """Get camera-aligned observations (multiview images, arm EE, hand joints).

        The head camera is the master reference. Wrist cameras are soft-
        synchronized by nearest-neighbor timestamp matching.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("observation cancelled")
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("waiting for a new head camera frame")
            if len(self.camera_deques['image']) == 0:
                raise RuntimeError("No main camera data available")
            latest_camera = self.camera_deques['image'][-1]
            camera_timestamp = latest_camera['timestamp']
            if self.last_camera_timestamp is None or camera_timestamp != self.last_camera_timestamp:
                self.last_camera_timestamp = camera_timestamp
                break
            time.sleep(0.001)

        if len(self.camera_deques['image_wrist_left']) == 0:
            raise RuntimeError("No left wrist camera data available")
        if len(self.camera_deques['image_wrist_right']) == 0:
            raise RuntimeError("No right wrist camera data available")

        latest_wrist_left, wrist_left_sync_error = self._get_nearest_camera_frame(
            'image_wrist_left', camera_timestamp
        )
        latest_wrist_right, wrist_right_sync_error = self._get_nearest_camera_frame(
            'image_wrist_right', camera_timestamp
        )

        if len(self.leo_ee_deque) < 2:
            raise RuntimeError("Not enough arm data for interpolation")

        ee_data = list(self.leo_ee_deque)
        ee_timestamps = np.array([d['timestamp'] for d in ee_data])
        right_ee_poses = np.array([d['right_ee_pose'] for d in ee_data])
        left_ee_poses = np.array([d['left_ee_pose'] for d in ee_data])

        camera_timestamps = camera_timestamp - np.array(his) * (1.0 / self.fps)
        right_ee_pose = PoseInterpolator(t=ee_timestamps, x=right_ee_poses)(camera_timestamps)
        left_ee_pose = PoseInterpolator(t=ee_timestamps, x=left_ee_poses)(camera_timestamps)

        if len(self.hand_joint_deque) < 2:
            raise RuntimeError("Not enough hand data for interpolation")

        hand_data = list(self.hand_joint_deque)
        hand_timestamps = np.array([d['timestamp'] for d in hand_data])
        right_hand_joints = np.array([d['right_hand_joint'] for d in hand_data])
        left_hand_joints = np.array([d['left_hand_joint'] for d in hand_data])

        right_hand_joint = get_interp1d(t=hand_timestamps, x=right_hand_joints)(camera_timestamps)
        left_hand_joint = get_interp1d(t=hand_timestamps, x=left_hand_joints)(camera_timestamps)

        return {
            'image': latest_camera['image'],
            'image_wrist_left': latest_wrist_left['image'],
            'image_wrist_right': latest_wrist_right['image'],
            'right_ee_pose': np.array(right_ee_pose, dtype=np.float64),
            'left_ee_pose': np.array(left_ee_pose, dtype=np.float64),
            'right_hand_joint': np.array(right_hand_joint, dtype=np.float64),
            'left_hand_joint': np.array(left_hand_joint, dtype=np.float64),
            'timestamp': camera_timestamp,
            'raw_camera_recv_timestamp': latest_camera['raw_recv_timestamp'],
            'wrist_left_sync_error': wrist_left_sync_error,
            'wrist_right_sync_error': wrist_right_sync_error,
        }

    def is_ready(self):
        camera_ready = len(self.camera_deques['image']) > 0
        wrist_left_ready = len(self.camera_deques['image_wrist_left']) > 0
        wrist_right_ready = len(self.camera_deques['image_wrist_right']) > 0
        arm_ready = len(self.leo_ee_deque) >= 2
        hand_ready = len(self.hand_joint_deque) >= 2

        parts = []
        for name, ready in [
            ("camera", camera_ready),
            ("wrist L", wrist_left_ready),
            ("wrist R", wrist_right_ready),
            ("arm", arm_ready),
            ("hand", hand_ready),
        ]:
            color = "\033[32m" if ready else "\033[31m"
            parts.append(f"{color}{name}: {ready}\033[0m")
        print(f"\r{' | '.join(parts)}", end="", flush=True)

        return camera_ready and wrist_left_ready and wrist_right_ready and arm_ready and hand_ready

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _pose6d_to_ros_pose(pose6d) -> Pose:
        """Convert 6D pose array (xyz + rotvec) to ROS Pose (quaternion)."""
        pose7d = pose6d_to_pose7d(np.asarray(pose6d, dtype=np.float64))
        p = Pose()
        p.position.x, p.position.y, p.position.z = float(pose7d[0]), float(pose7d[1]), float(pose7d[2])
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (
            float(pose7d[3]), float(pose7d[4]), float(pose7d[5]), float(pose7d[6]))
        return p

    @staticmethod
    def _float_to_duration(t: float) -> Duration:
        """Convert a float (seconds) to a builtin_interfaces/Duration."""
        d = Duration()
        d.sec = int(t)
        d.nanosec = int((t - int(t)) * 1e9)
        return d

    # ── Action publishing (aligned with arm_planner_node / hand_planner_node)

    def publish_arm_chunk(self, arm_chunk, action_timestamps, cancel_previous=False, *, start_stamp=None):
        """Publish arm EE targets as a single Action message to /leo/arm_action.

        arm_chunk: (T, 12) — col 0:6 = left arm, col 6:12 = right arm (pose6d).
        Each waypoint appends two poses (left, right) and one time_from_start.
        The arm_planner_node interpolates and executes the full trajectory.
        start_stamp optionally shares one ROS time origin with the hand chunk.
        """
        if self._resetting:
            return
        arm_array = np.asarray(arm_chunk, dtype=np.float64)
        num_steps = arm_array.shape[0]
        now_stamp = start_stamp if start_stamp is not None else self._node.get_clock().now().to_msg()

        arm_msg = Action()
        arm_msg.header.stamp = now_stamp
        arm_msg.cancel_previous = cancel_previous

        for i in range(num_steps):
            if action_timestamps[i] < self.arm_exec_latency:
                continue
            arm_msg.poses.append(self._pose6d_to_ros_pose(arm_array[i, :6]))
            arm_msg.poses.append(self._pose6d_to_ros_pose(arm_array[i, 6:12]))
            arm_msg.time_from_start.append(self._float_to_duration(action_timestamps[i] - self.arm_exec_latency))

        self._arm_pub.publish(arm_msg)

    def publish_hand_chunk(self, hand_chunk, action_timestamps, cancel_previous=False, *, start_stamp=None):
        """Publish hand joint targets as Trajectory messages to /leo/{left,right}_hand_action.

        hand_chunk: (T, 40) — col 0:20 = left hand, col 20:40 = right hand.
        Each waypoint is a JointTrajectoryPoint with time_from_start.
        The hand_planner_node interpolates and executes the full trajectory.
        start_stamp optionally shares one ROS time origin with the arm chunk.
        """
        if self._resetting:
            return
        hand_array = np.asarray(hand_chunk, dtype=np.float64)
        num_steps = hand_array.shape[0]
        now_stamp = start_stamp if start_stamp is not None else self._node.get_clock().now().to_msg()

        left_msg = Trajectory()
        left_msg.header.stamp = now_stamp
        left_msg.joint_names = LEFT_HAND_JOINTS
        left_msg.cancel_previous = cancel_previous

        right_msg = Trajectory()
        right_msg.header.stamp = now_stamp
        right_msg.joint_names = RIGHT_HAND_JOINTS
        right_msg.cancel_previous = cancel_previous

        for i in range(num_steps):
            t = action_timestamps[i] - self.hand_exec_latency
            if t < 0:
                continue
            left_pt = JointTrajectoryPoint()
            left_pt.positions = hand_array[i, :20].tolist()
            left_pt.time_from_start = self._float_to_duration(t)
            left_msg.points.append(left_pt)

            right_pt = JointTrajectoryPoint()
            right_pt.positions = hand_array[i, 20:40].tolist()
            right_pt.time_from_start = self._float_to_duration(t)
            right_msg.points.append(right_pt)

        self._left_hand_pub.publish(left_msg)
        self._right_hand_pub.publish(right_msg)

    # ── Reset ───────────────────────────────────────────────────────────────

    def reset(self):
        """Reset arm + hand via service calls (like replay_node), with trajectory fallback."""
        self._resetting = True

        # Try service-based reset first (aligned with replay_node)
        services_available = False
        for client in self._reset_clients:
            if client.service_is_ready():
                services_available = True
                future = client.call_async(Trigger.Request())
                future.add_done_callback(self._reset_done_cb)

        if services_available:
            self._node.get_logger().info("Reset via service calls, waiting 5s...")
            time.sleep(5.0)
        else:
            # Fallback: direct trajectory commands to joint controllers
            self._node.get_logger().info("Reset services not available, using trajectory fallback...")
            self._reset_via_trajectories()

        self._resetting = False
        self._node.get_logger().info("Environment reset complete.")

    def _reset_done_cb(self, future):
        try:
            result = future.result()
            self._node.get_logger().info(f'Reset response: {result.message}')
        except Exception as e:
            self._node.get_logger().error(f'Reset call failed: {e}')

    def _reset_via_trajectories(self):
        stamp = self._node.get_clock().now().to_msg()

        left_hand_targets = [0.0] * 20
        left_hand_targets[0], left_hand_targets[1] = 0.5, 0.6
        right_hand_targets = [0.0] * 20
        right_hand_targets[0], right_hand_targets[1] = 0.5, 0.6

        configs = [
            (self._reset_left_arm_pub,
             ["left_joint1", "left_joint2", "left_joint3", "left_joint4",
              "left_wrist_yaw", "left_wrist_pitch", "left_wrist_roll"],
             [0, 0, -0.5236, 1.5708, 0, 0, 0]),
            (self._reset_right_arm_pub,
             ["right_joint1", "right_joint2", "right_joint3", "right_joint4",
              "right_wrist_yaw", "right_wrist_pitch", "right_wrist_roll"],
             [0, 0, 0.5236, -1.5708, 0, 0, 0]),
            (self._reset_head_pub,
             ["head_joint1", "head_joint2"],
             [0.0, 0.0]),
            (self._reset_left_hand_pub,
             [f"left_hand_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)],
             left_hand_targets),
            (self._reset_right_hand_pub,
             [f"right_hand_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)],
             right_hand_targets),
        ]

        for pub, names, targets in configs:
            msg = JointCommand()
            msg.header.stamp = stamp
            msg.joint_names = names
            msg.positions = [float(v) for v in targets]
            pub.publish(msg)

        self._node.get_logger().info("Reset commands published, waiting 5s...")
        time.sleep(5.0)

    def close(self):
        """Stop image decoders after the ROS executor has stopped callbacks."""
        for raw_queue in self.camera_raw_queues.values():
            raw_queue.put(None)
        for worker in self.camera_threads:
            worker.join()
