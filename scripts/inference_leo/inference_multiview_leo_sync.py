"""Run synchronous Liber_Teleop v2.2 policy inference on Leo.

The robot environment remains responsible only for acquiring observations and
publishing absolute arm/hand chunks.  After ``get_obs()``, the client adapter:

* cover-resizes the three BGR cameras to dataset RGB 480x640;
* standardizes raw flange orientation in ``chest_link``;
* creates the dataset's MANO palm/theta/beta fields;
* sends the exact ``observation.*`` field names used by training; and
* decodes named actions or unified ``[T,326]`` actions back to the existing
  ``[T,52] = arm12 + hand40`` absolute-control boundary.

The adapter also accepts transitional legacy ``[T,58]`` responses.  ROS
publishing remains compatible; sync-no-sleep uses observation-aligned deadlines.
"""

import time
import threading
import argparse
import json
import os
import socket
import atexit
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from envs.leo_wuji_env_infer import LeoWujiInferEnv
from policy_server.msgpack_numpy import packb, unpackb
from utils.client_config import DEFAULT_CONFIG_PATH, load_client_config
from utils.action_perturbation import (
    RandomEEActionPerturber,
    format_ee_perturbation,
)
from utils.liber_teleop_v22_client import (
    add_client_timing,
    build_request,
    decode_action,
    prepare_hand_action_decoder,
    validate_server_inference_profile,
)
from utils.action_timing import publish_observation_aligned_chunk
from utils.temporal_ensemble import TemporalActionEnsembler

ACTION_EXEC_LATENCY = 0
CHUNK_EXEC_MIN_STEP = 8
# ── Action decoding (defaults; override with --arm_action_type/--hand_action_type) ─
# Arm representation:  'relative' | 'absolute'
# Hand representation: 'relative' | 'absolute'
#   relative arm + relative hand -> normal relative checkpoint
#   absolute arm + absolute hand -> full-absolute checkpoint
#   relative arm + absolute hand -> absHand checkpoint (arm delta, hand abs angle)
ARM_ACTION_TYPE = 'relative'
HAND_ACTION_TYPE = 'relative'

# ── Persistent DP Client ─────────────────────────────────────────────────────

class DPClient:
    """Persistent-connection client for the policy server.

    The DP server keeps the TCP connection alive between requests (same as
    RDT2). This client mirrors that: connect() once, send_request() in a
    loop, disconnect() when done.

    The payload is assembled by ``utils.liber_teleop_v22_client``. Responses
    may contain dataset-named action fields or a canonical ``[T,326]`` chunk.
    """

    def __init__(self, host: str = "localhost", port: int = 8888):
        self.host = host
        self.port = port
        self.socket: socket.socket | None = None

    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        print(f"Connected to DP policy server at {self.host}:{self.port}")

    def disconnect(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def send_request(self, request: dict) -> dict:
        """Send one request and return the unpacked server response."""
        assert self.socket is not None, "Call connect() first"

        round_trip_started = time.perf_counter()
        data = packb(request)
        self.socket.sendall(len(data).to_bytes(4, byteorder="big"))
        self.socket.sendall(data)

        length_data = self._recv_exact(4)
        resp_len = int.from_bytes(length_data, byteorder="big")
        resp_data = self._recv_exact(resp_len)

        response = unpackb(resp_data)
        round_trip_ms = (time.perf_counter() - round_trip_started) * 1000.0
        if isinstance(response, dict):
            add_client_timing(response, round_trip_ms)
        return response

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.socket.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Server closed the connection")
            buf += chunk
        return buf


# ── Main ──────────────────────────────────────────────────────────────────────

def _load_init_hand(path, side):
    """Load the initial (20,) hand pose for `side` ('left'/'right').

    Three inputs are accepted:
      * a replay collect DIR (e.g. .../collect_1781315360_3): read frame 0 of
        `state.npz['{side}_hand_state']` (T,20).
      * a `*_mean_initial_state.npy` dict (0-d object array): the per-episode
        mean initial state. We read `mean_by_key['{side}_hand_state']` (20,).
      * a plain `(T, 20)` hand-state npy: we take frame 0 (`arr[0]`).
    Falls back to neutral zeros (hands open) with a loud WARN if the file is
    missing or lacks the side key, so a stale path can never crash the client
    at startup on the real robot."""
    # replay collect dir -> state.npz['{side}_hand_state'][0]
    if path and os.path.isdir(path):
        npz_path = os.path.join(path, "state.npz")
        key = f"{side}_hand_state"
        if os.path.exists(npz_path):
            with np.load(npz_path, allow_pickle=True) as s:
                if key in s.files:
                    pose = np.asarray(s[key][0], dtype=np.float32).reshape(-1)
                    print(f"[policy] init {side} hand <- {npz_path}#{key}[0]  {pose[:4]}...")
                    return pose
            print(f"[policy] WARN: {npz_path!r} has no {key!r} -> neutral zeros(20).")
        else:
            print(f"[policy] WARN: {npz_path!r} missing -> neutral zeros(20).")
        return np.zeros(20, dtype=np.float32)
    if path and os.path.exists(path):
        arr = np.load(path, allow_pickle=True)
        # mean_initial_state.npy -> 0-d object array wrapping a dict.
        if arr.dtype == object and arr.ndim == 0:
            d = arr.item()
            mean_by_key = d.get("mean_by_key", {}) if isinstance(d, dict) else {}
            key = f"{side}_hand_state"
            if key in mean_by_key:
                pose = np.asarray(mean_by_key[key], dtype=np.float32).reshape(-1)
                print(f"[policy] init {side} hand <- {path}#mean_by_key[{key!r}]  {pose[:4]}...")
                return pose
            print(f"[policy] WARN: {path!r} has no mean_by_key[{key!r}] -> neutral zeros(20).")
            return np.zeros(20, dtype=np.float32)
        # plain (T, 20) hand-state array.
        pose = np.asarray(arr[0], dtype=np.float32).reshape(-1)
        print(f"[policy] init {side} hand <- {path}  {pose[:4]}...")
        return pose
    print(f"[policy] WARN: init {side} hand npy {path!r} missing -> neutral zeros(20). "
          f"Pass --init_hand_npy_{side} to set a real start pose.")
    return np.zeros(20, dtype=np.float32)


def _load_start_pose(path):
    """Load frame-0 arm/hand reset targets from a replay dir or mean-state file.

    Replay captures store the live state in ``state.npz``.  The older deploy
    profile stores the same four fields inside a ``*_mean_initial_state.npy``
    object/dict.  JSON with the same ``mean_by_key`` mapping is supported for
    portable, reviewable one-command evaluation profiles.
    """
    if not path:
        raise ValueError("initial state path must not be empty")

    state_path = os.path.join(path, "state.npz") if os.path.isdir(path) else path
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"initial state does not exist: {state_path}")

    required = {
        "left_ee_pose": 6,
        "right_ee_pose": 6,
        "left_hand_state": 20,
        "right_hand_state": 20,
    }
    values = {}
    if state_path.endswith(".npz"):
        with np.load(state_path, allow_pickle=False) as state:
            missing = [key for key in required if key not in state.files]
            if missing:
                raise ValueError(f"initial state {state_path} is missing keys: {missing}")
            for key in required:
                values[key] = np.asarray(state[key][0], dtype=np.float32).reshape(-1)
    elif state_path.endswith(".json"):
        with open(state_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        values = metadata.get("mean_by_key", metadata) if isinstance(metadata, dict) else {}
    else:
        metadata = np.load(state_path, allow_pickle=True).item()
        values = metadata.get("mean_by_key", metadata) if isinstance(metadata, dict) else {}

    if isinstance(values, np.ndarray) and values.ndim == 0:
        values = values.item()
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"initial state {state_path} is missing keys: {missing}")
    values = {
        key: np.asarray(values[key], dtype=np.float32).reshape(-1)
        for key in required
    }

    for key, size in required.items():
        if values[key].shape != (size,) or not np.isfinite(values[key]).all():
            raise ValueError(
                f"initial state {state_path} {key} must be finite shape ({size},), "
                f"got {values[key].shape}"
            )

    pose = np.concatenate([values["left_ee_pose"], values["right_ee_pose"]])[None]
    hands = np.concatenate(
        [values["left_hand_state"], values["right_hand_state"]]
    )[None]
    return state_path, pose, hands


class _HeadCameraRecorder:
    """Lazy raw head-camera mp4 writer.

    The writer opens on the first frame so its size matches the live camera
    stream. One sync-loop observation writes one output frame.
    """

    def __init__(self, path, fps):
        self.path = path
        self.fps = fps
        self.writer = None
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
            print(f"[record] head camera -> {self.path} @ {self.fps}fps")

    def write(self, image):
        if not self.path:
            return
        if self.writer is None:
            h, w = image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.path, fourcc, self.fps, (w, h))
        self.writer.write(image)

    def close(self):
        if self.writer is not None and self.writer.isOpened():
            self.writer.release()
            print(f"[record] head camera saved -> {self.path}")


def _print_chunk_arm_movement(action_arm, observation):
    """Print maximum XYZ displacement in cm for each arm in one action chunk."""
    left_start = np.asarray(observation["left_ee_pose"], dtype=np.float64).reshape(-1, 6)[0, :3]
    right_start = np.asarray(observation["right_ee_pose"], dtype=np.float64).reshape(-1, 6)[0, :3]
    action = np.asarray(action_arm, dtype=np.float64).reshape(-1, 12)
    left_max_cm = np.linalg.norm(action[:, :3] - left_start, axis=1).max() * 100.0
    right_max_cm = np.linalg.norm(action[:, 6:9] - right_start, axis=1).max() * 100.0
    reset = "\033[0m"
    print(
        f"\033[95m[chunk] left arm max displacement: {left_max_cm:.2f} cm{reset}"
    )
    print(
        f"\033[94m[chunk] right arm max displacement: {right_max_cm:.2f} cm{reset}"
    )


def _load_instruction_sequence(path):
    """Load a non-empty JSON list of prompts used by staged long-horizon evals."""
    sequence_path = Path(path).expanduser().resolve()
    with sequence_path.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, list) or not values:
        raise ValueError(
            f"instruction sequence must be a non-empty JSON list: {sequence_path}"
        )
    prompts = [str(value).strip() for value in values]
    if any(not value for value in prompts):
        raise ValueError(f"instruction sequence contains an empty prompt: {sequence_path}")
    return prompts


class _InstructionController:
    """Thread-safe current prompt with optional operator-driven stage changes."""

    def __init__(self, prompts):
        self._prompts = tuple(prompts)
        self._index = 0
        self._lock = threading.Lock()

    @property
    def count(self):
        return len(self._prompts)

    def current(self):
        with self._lock:
            return self._index, self._prompts[self._index]

    def advance(self):
        with self._lock:
            if self._index + 1 >= len(self._prompts):
                return None
            self._index += 1
            return self._index, self._prompts[self._index]


def _start_instruction_advancer(controller):
    if controller.count <= 1:
        return None

    def advance_on_enter():
        for _ in range(1, controller.count):
            try:
                input("[instruction] 当前子任务完成后按回车切换到下一条指令")
            except EOFError:
                return
            current = controller.advance()
            if current is None:
                return
            index, prompt = current
            print(f"[imagewam] instruction {index + 1}/{controller.count} -> {prompt!r}")
        print("[instruction] 已切换到最后一条指令；完成后按 Ctrl-C 停止")

    thread = threading.Thread(target=advance_on_enter, daemon=True)
    thread.start()
    return thread


def main(argv=None, *, use_mix_client=False):
    global ARM_ACTION_TYPE, HAND_ACTION_TYPE
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    config_args, _ = config_parser.parse_known_args(argv)
    try:
        config = load_client_config(config_args.config)
    except (OSError, ValueError) as error:
        config_parser.error(str(error))

    policy_cfg = config["policy"]
    runtime_cfg = config["runtime"]
    startup_cfg = config["startup"]
    action_cfg = config["action"]
    parser = argparse.ArgumentParser(
        description="Leo Diffusion Policy inference (arm + hand), chest-aligned"
    )
    parser.add_argument("--config", default=config_args.config,
                        help="Human-editable client YAML configuration.")
    parser.add_argument("--policy_host", "--policy-host", dest="policy_host", type=str,
                        default=str(policy_cfg["host"]))
    parser.add_argument("--policy_port", "--policy-port", dest="policy_port", type=int,
                        default=int(policy_cfg["port"]))
    parser.add_argument(
        "--dataset-name",
        default=policy_cfg.get("dataset_name"),
        help="Checkpoint dataset binding used to select the runtime adapter and normalization.",
    )
    parser.add_argument("--fps", type=float, default=float(runtime_cfg["fps"]),
                        help="Observation history / recording FPS; also the default action FPS.")
    parser.add_argument("--action-fps", type=float, default=runtime_cfg.get("action_fps"),
                        help="Published action waypoint frequency; defaults to --fps.")
    parser.add_argument("--chunk_size", type=int, default=int(runtime_cfg["chunk_size"]),
                        help="Execute only the first N steps of each returned action chunk.")
    parser.add_argument("--discard-steps", type=int,
                        default=int(runtime_cfg["discard_steps"]))
    parser.add_argument(
        "--post-chunk-sleep",
        type=float,
        default=float(runtime_cfg.get("post_chunk_sleep", 1.5)),
        help="Seconds to wait after publishing each action chunk in sync mode.",
    )
    parser.add_argument(
        "--async-sleep-steps", type=int,
        default=int(runtime_cfg.get("async_sleep_steps", 0)),
        help="In sync-no-sleep mode, wait N / action_fps seconds after publishing before taking the next observation.",
    )
    parser.add_argument(
        "--temporal-ensemble", action=argparse.BooleanOptionalAction,
        default=bool(runtime_cfg.get("temporal_ensemble", False)),
        help="Blend overlapping absolute predictions at matching deadlines in sync-no-sleep mode.",
    )
    parser.add_argument(
        "--temporal-ensemble-max-chunks", type=int,
        default=int(runtime_cfg.get("temporal_ensemble_max_chunks", 0)),
        help="Maximum recent chunks to blend, including the current one; 0 = unlimited, 1 = current only.",
    )
    parser.add_argument(
        "--inference-mode", metavar="{sync,sync-no-sleep}",
        default=runtime_cfg.get("inference_mode", "sync"),
        help="sync-no-sleep publishes future observation-aligned actions; optional wait via --async-sleep-steps.",
    )
    parser.add_argument("--camera-latency", type=float,
                        default=float(runtime_cfg.get("camera_latency", 0.100)))
    parser.add_argument("--wrist-camera-latency", type=float,
                        default=float(runtime_cfg.get("wrist_camera_latency", 0.040)))
    parser.add_argument("--history-offsets", type=int, nargs="+",
                        default=list(runtime_cfg["history_offsets"]))
    parser.add_argument("--crop", type=int, nargs=3,
                        default=list(runtime_cfg["crop_percent"]),
                        metavar=("HEAD", "WRIST_L", "WRIST_R"),
                        help="Center crop %% per view.")
    parser.add_argument("--record_head", type=str,
                        default=runtime_cfg.get("record_head"), metavar="PATH",
                        help="Record the raw head-camera stream to this .mp4 file "
                             "(written at --fps). Omit to disable recording.")
    parser.add_argument(
        "--start-image-path",
        type=str,
        default=None,
        metavar="PATH",
        help="Save the first head-camera observation as a PNG at this path.",
    )
    parser.add_argument("--no_display", action="store_true",
                        default=not bool(runtime_cfg["display"]),
                        help="Disable camera preview windows.")
    # Language instruction forwarded to the policy server so it rebuilds the
    # SAME prompt the model trained on. teleop_v21 trains with
    # task_prompt_strategy=episode_random (per-episode natural-language tasks from
    # meta/tasks.jsonl), so this model is language-conditioned: pass the English
    # instruction for the task you are running. If omitted (None), the server
    # falls back to the checkpoint's baked-in task_prompt (generic "pick place"),
    # which will NOT match a specific teleop_v21 task.
    instruction_group = parser.add_mutually_exclusive_group(required=True)
    instruction_group.add_argument(
        "--task_prompt", "--instruction", dest="task_prompt", type=str,
        help="ImageWAM task instruction.",
    )
    instruction_group.add_argument(
        "--instruction-sequence-file",
        help=(
            "JSON list of staged instructions. After inference starts, press Enter "
            "whenever the current physical subtask is complete to select the next one."
        ),
    )
    parser.add_argument("--init_state_path", "--init-state-path", dest="init_state_path",
                        default=startup_cfg.get("state_path"),
                        help="Replay directory/state.npz or mean initial-state .npy "
                             "used for reset and startup pose.")
    # [INIT-HAND] initial hand state to move the hands to before the loop.
    # Default is a replay collect dir; _load_init_hand reads frame 0 of
    # state.npz['{left,right}_hand_state'] (T,20). Also accepts a
    # `*_mean_initial_state.npy` dict (mean_by_key['{side}_hand_state']) or a
    # plain (T,20) hand-state npy ([0]). Missing files / keys fall back to
    # neutral zeros (hands open) with a WARN rather than crash.
    parser.add_argument("--reset-state-path", default=startup_cfg.get("reset_state_path"),
                        help="Replay directory whose frame 0 is published during startup reset.")
    parser.add_argument("--no-wait-for-enter", dest="wait_for_enter", action="store_false",
                        default=bool(startup_cfg["wait_for_enter"]),
                        help="Start without the final interactive confirmation.")
    parser.add_argument("--init_hand_npy_left", type=str,
                        default=None,
                        help="replay dir (state.npz['left_hand_state'][0]), a "
                             "mean_initial_state.npy dict, or a (T,20) npy ([0]) "
                             "for the initial LEFT hand pose.")
    parser.add_argument("--init_hand_npy_right", type=str,
                        default=None,
                        help="replay dir (state.npz['right_hand_state'][0]), a "
                             "mean_initial_state.npy dict, or a (T,20) npy ([0]) "
                             "for the initial RIGHT hand pose.")
    parser.add_argument("--arm_action_type", type=str,
                        choices=["relative", "absolute"], default=str(action_cfg["arm_type"]),
                        help="Arm action representation of the checkpoint.")
    parser.add_argument("--hand_action_type", type=str,
                        choices=["relative", "absolute"], default=str(action_cfg["hand_type"]),
                        help="Hand action representation of the checkpoint. Use "
                             "'absolute' for absHand checkpoints (arm relative + "
                             "hand absolute).")
    parser.add_argument(
        "--hand-action-source",
        "--hand_action_source",
        choices=["q20", "mano"],
        default=str(action_cfg.get("hand_source", "mano")),
        help="Execute the model q20 head directly, or convert its MANO90 head to q20 locally.",
    )
    parser.add_argument(
        "--baseline-position-cm",
        dest="action_perturb_position_cm",
        type=float,
        default=0.0,
        help="Baseline EE position offset magnitude in cm.",
    )
    parser.add_argument(
        "--baseline-rotation-deg",
        dest="action_perturb_rotation_deg",
        type=float,
        default=0.0,
        help="Baseline EE orientation offset magnitude in degrees.",
    )
    parser.add_argument(
        "--baseline-seed",
        dest="action_perturb_seed",
        type=int,
        default=None,
        help="Optional random seed for the baseline.",
    )
    args = parser.parse_args(argv)

    if args.chunk_size < 1:
        parser.error("--chunk_size must be at least 1")
    if args.discard_steps < 0 or args.discard_steps >= args.chunk_size:
        parser.error("--discard-steps must satisfy 0 <= discard < chunk_size")
    if args.post_chunk_sleep < 0:
        parser.error("--post-chunk-sleep must be non-negative")
    if args.action_perturb_position_cm < 0:
        parser.error("--baseline-position-cm must be non-negative")
    if args.action_perturb_rotation_deg < 0:
        parser.error("--baseline-rotation-deg must be non-negative")
    if not args.init_state_path:
        parser.error("set startup.state_path in config/client.yaml or pass --init-state-path")
    if not args.reset_state_path:
        parser.error(
            "set startup.reset_state_path in config/client.yaml or pass --reset-state-path"
        )
    try:
        instruction_prompts = (
            _load_instruction_sequence(args.instruction_sequence_file)
            if args.instruction_sequence_file
            else [str(args.task_prompt).strip()]
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if not instruction_prompts[0]:
        parser.error("--instruction must not be empty")
    instruction_controller = _InstructionController(instruction_prompts)

    ARM_ACTION_TYPE = args.arm_action_type
    HAND_ACTION_TYPE = args.hand_action_type
    print(
        f"[imagewam] action decode: arm={ARM_ACTION_TYPE} "
        f"hand={HAND_ACTION_TYPE} source={args.hand_action_source}"
    )
    if not use_mix_client or policy_cfg.get("protocol") == "liber_teleop_v22":
        prepare_hand_action_decoder(args.hand_action_source)
    action_fps = args.fps if args.action_fps is None else args.action_fps
    print(f"[timing] mode={args.inference_mode}, camera_latency={args.camera_latency:.3f}s, "
          f"action_fps={action_fps:g}, action_dt={1.0 / action_fps:.6f}s")
    if args.inference_mode == "sync-no-sleep":
        print(f"[timing] async_sleep_steps={args.async_sleep_steps}, "
              f"async_sleep={args.async_sleep_steps / action_fps:.3f}s, "
              f"temporal_ensemble={args.temporal_ensemble}, "
              f"temporal_ensemble_max_chunks={args.temporal_ensemble_max_chunks}")
    action_perturber = None
    if args.action_perturb_position_cm or args.action_perturb_rotation_deg:
        action_perturber = RandomEEActionPerturber(
            position_cm=args.action_perturb_position_cm,
            rotation_deg=args.action_perturb_rotation_deg,
            seed=args.action_perturb_seed,
        )
        print(
            f"[baseline] enabled: seed={action_perturber.seed}"
        )

    fps = args.fps
    crop_percents = args.crop  # [head, wrist_left, wrist_right]
    waypoint_dt = 1.0 / action_fps
    _, initial_instruction = instruction_controller.current()
    print(
        f"[imagewam] instruction 1/{instruction_controller.count} "
        f"-> {initial_instruction!r}"
    )
    init_state_path, init_arm, init_hand = _load_start_pose(args.init_state_path)
    print(f"[startup] initial state -> {init_state_path}")

    reset_state_path, reset_arm, reset_hand = _load_start_pose(args.reset_state_path)
    print(f"[startup] reset state -> {reset_state_path}")


    # ── ROS2 init ─────────────────────────────────────────────────────────────
    rclpy.init()
    node = Node('leo_dp_inference_node')
    env = LeoWujiInferEnv(node, fps=fps, camera_latency=args.camera_latency,
                          wrist_camera_latency=args.wrist_camera_latency)

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    # ── Policy client ───────────────────────────────────────────────────────
    client = DPClient(host=args.policy_host, port=args.policy_port)
    if use_mix_client:
        from utils.mix_policy_client import MixPolicyClient
        client = MixPolicyClient(client, policy_cfg, hand_source=args.hand_action_source,
                                 crop=args.crop)
    client.connect()

    # ── Wait until all data sources are ready ─────────────────────────────────
    print("Waiting for camera, arm, and hand data...")
    while rclpy.ok():
        if env.is_ready():
            break
        time.sleep(0.1)
    print("\nAll data sources ready!")

    print("[startup] resetting robot...")
    env.reset()
    time.sleep(2.0)

    now_time = env._node.get_clock().now().to_msg()
    now_time = now_time.sec + now_time.nanosec * 1e-9
    env.publish_hand_chunk(reset_hand, action_timestamps=[1.0], cancel_previous=True)

    now_time = env._node.get_clock().now().to_msg()
    now_time = now_time.sec + now_time.nanosec * 1e-9
    env.publish_arm_chunk(reset_arm, action_timestamps=[5.0], cancel_previous=True)

    # env.publish_hand_chunk(init_hand, action_timestamps=[1.0], cancel_previous=True)
    # env.publish_arm_chunk(init_arm, action_timestamps=[5.0], cancel_previous=True)
    time.sleep(5.0)
    print(f"Ready — fps={fps}\n")

    if args.wait_for_enter:
        input("按下回车以继续")
    _start_instruction_advancer(instruction_controller)

    head_recorder = _HeadCameraRecorder(args.record_head, fps)
    if args.record_head:
        atexit.register(head_recorder.close)

    ensemble = TemporalActionEnsembler(args.temporal_ensemble_max_chunks) if args.temporal_ensemble else None
    ensemble_instruction = initial_instruction
    start_image_saved = False
    while rclpy.ok():
        loop_start = time.monotonic()

        start_time = time.monotonic()
        # ── Observation ───────────────────────────────────────────────────────
        obs = env.get_obs(args.history_offsets)
        obs_timestamp = obs["timestamp"]          # ROS wall-clock (CLOCK_REALTIME)
        print(f"Observation latency: {time.monotonic() - start_time:.4f}s")

        image = obs["image"]                      # HWC (H, W, 3) BGR uint8
        image_wrist_left = obs["image_wrist_left"]
        image_wrist_right = obs["image_wrist_right"]

        if args.start_image_path and not start_image_saved:
            start_image_path = Path(args.start_image_path).expanduser().resolve()
            start_image_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(start_image_path), image):
                raise OSError(f"failed to save start image: {start_image_path}")
            print(f"[start-image] saved -> {start_image_path}")
            start_image_saved = True

        head_recorder.write(image)

        # ── Visualize observation images ───────────────────────────────────────
        if not args.no_display:
            cv2.imshow("Head Camera", image)
            cv2.imshow("Wrist Left", image_wrist_left)
            cv2.imshow("Wrist Right", image_wrist_right)
            cv2.waitKey(1)

        # ── Policy inference ──────────────────────────────────────────────────
        _, current_instruction = instruction_controller.current()
        request, action_decode_context = (client.build_request if use_mix_client else build_request)(
            obs,
            current_instruction,
            dataset_name=args.dataset_name,
        )
        resp = client.send_request(request)
        if not resp.get("success", False):
            print(f"Policy error: {resp.get('error')}")
            break
        response_action = resp["action"]
        action_shape = (
            "named dataset fields"
            if isinstance(response_action, dict)
            else np.asarray(response_action).shape
        )
        timing = resp.get("timing") or {}
        print(
            "Policy latency: "
            f"network RTT={float(timing.get('network_round_trip_ms', float('nan'))):.2f}ms, "
            f"model inference={float(timing.get('inference_ms', float('nan'))):.2f}ms, "
            f"server processing={float(timing.get('server_processing_ms', float('nan'))):.2f}ms, "
            f"total RTT={float(timing.get('round_trip_ms', float('nan'))):.2f}ms, "
            f"chunk shape {action_shape}"
        )

        if use_mix_client:
            abs_action = client.decode_response(resp, action_decode_context)
            action_row_offset = client.action_row_offset
        else:
            server_info = resp.get("server_info") or {}
            if server_info.get("infer_kind") not in (None, "imagewam", "fastwam"):
                raise RuntimeError(f"unexpected policy server: {server_info}")
            validate_server_inference_profile(
                server_info,
                expected_num_inference_steps=policy_cfg.get(
                    "expected_num_inference_steps"
                ),
                expected_image_num_inference_steps=policy_cfg.get(
                    "expected_image_num_inference_steps"
                ),
            )
            adapter_name = server_info.get("adapter_name")
            if adapter_name == "teleop_q20" and (
                args.hand_action_source != "q20" or HAND_ACTION_TYPE != "absolute"
            ):
                raise RuntimeError(
                    "teleop_q20 checkpoints require hand_source=q20 and hand_type=absolute"
                )
            abs_action = decode_action(
                response_action,
                action_decode_context,
                action_type=ARM_ACTION_TYPE,
                hand_action_source=args.hand_action_source,
                adapter_name=adapter_name,
            )
            action_row_offset = server_info.get("action_row_offset", policy_cfg.get("action_row_offset", 0))
        abs_action = abs_action[:args.chunk_size]
        action_arm = abs_action[:, :12]    # (T, 12): left_ee(6) + right_ee(6)
        if action_perturber is not None:
            action_arm, perturbation = action_perturber.apply(action_arm)
            print(
                "[baseline] "
                + format_ee_perturbation(perturbation)
            )
        action_arm = action_arm[args.discard_steps:]
        action_hand = abs_action[args.discard_steps:, 12:]  # (T, 40): hands
        if len(action_arm):
            _print_chunk_arm_movement(action_arm, obs)

        if args.inference_mode == "sync-no-sleep":
            # Keep the original row indices after discard; do not rebase time.
            action_timestamps = (
                np.arange(args.discard_steps, len(abs_action)) + action_row_offset
            ) * waypoint_dt
            if ensemble is not None:
                if current_instruction != ensemble_instruction:
                    ensemble = TemporalActionEnsembler(args.temporal_ensemble_max_chunks)
                    ensemble_instruction = current_instruction
                blended = ensemble.apply(
                    np.concatenate([action_arm, action_hand], axis=1),
                    obs_timestamp + action_timestamps,
                )
                action_arm, action_hand = blended[:, :12], blended[:, 12:]
            count, age = publish_observation_aligned_chunk(
                env, action_arm, action_hand, action_timestamps, obs_timestamp
            )
            print(f"[timing] observation_age={age:.3f}s, published={count}/{len(action_arm)}, "
                  f"expired={len(action_arm) - count}")
            if count and args.async_sleep_steps:
                time.sleep(args.async_sleep_steps * waypoint_dt)
        elif len(action_arm):
            action_timestamps = (
                np.arange(args.discard_steps, len(abs_action))
                + (action_row_offset if use_mix_client else 0)
            ) * waypoint_dt
            env.publish_arm_chunk(action_arm, action_timestamps, cancel_previous=True)
            env.publish_hand_chunk(action_hand, action_timestamps, cancel_previous=True)
            time.sleep(args.post_chunk_sleep)
        print(f"Loop time: {time.monotonic() - loop_start:.4f}s\n")

    head_recorder.close()
    cv2.destroyAllWindows()
    client.disconnect()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
