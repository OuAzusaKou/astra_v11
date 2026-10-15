"""ImageWAM data contract for the Leo live client.

This module contains representation and preprocessing only.  It deliberately
does not own connection management, timing, chunk scheduling, or ROS control.
The contract matches ``Liberai-Model/policy_server/imagewam`` for Leo models:

* RGB HWC camera observations;
* an exact float32 ImageWAM mosaic in ``[-1, 1]``;
* 58D interleaved state/action layout
  ``[left_arm9, left_hand20, right_arm9, right_hand20]``;
* relative arm transforms with absolute hand targets.
"""

from __future__ import annotations

from fractions import Fraction
import threading
import uuid

import av
import numpy as np
import scipy.spatial.transform as st

from utils.pose_utils import mat_to_pose, pose_to_mat


# Top-level request marker used by the deployed live WAM client.  The exact
# float mosaic still travels in observation.model_input_image.
IMAGE_SOURCE = "raw_camera"
IMAGE_LAYOUT = "leo_native_320"
IMAGE_SIZES = ((320, 176), (160, 96), (160, 96))  # (width, height)
DATASET_FRAME_SIZE = (640, 480)  # (width, height), teleop v2.1 converter output
STATE_DIM = 58
ACTION_DIM = 58

# These are the deployed Leo/WAM child-frame orientation remaps.  Positions
# stay unchanged; only the EE orientation is expressed in the model frame.
R_RIGHT_HAND_TO_CHEST = np.array(
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64
)
R_LEFT_HAND_TO_CHEST = np.array(
    [[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64
)


def _pose6_to_chest(pose6, rotation):
    value = np.asarray(pose6)
    flat = value.astype(np.float64).reshape(-1, 6)
    rotation_ee = st.Rotation.from_rotvec(flat[:, 3:]).as_matrix()
    rotation_model = rotation_ee @ rotation.T
    output = np.concatenate(
        [flat[:, :3], st.Rotation.from_matrix(rotation_model).as_rotvec()], axis=1
    )
    return output.reshape(value.shape).astype(value.dtype, copy=False)


def _pose6_from_chest(pose6, rotation):
    value = np.asarray(pose6)
    flat = value.astype(np.float64).reshape(-1, 6)
    rotation_model = st.Rotation.from_rotvec(flat[:, 3:]).as_matrix()
    rotation_ee = rotation_model @ rotation
    output = np.concatenate(
        [flat[:, :3], st.Rotation.from_matrix(rotation_ee).as_rotvec()], axis=1
    )
    return output.reshape(value.shape).astype(value.dtype, copy=False)


def _model_pose9_from_matrix(matrix):
    # ImageWAM uses the first two rotation *columns* for its 6D rotation.
    return np.concatenate(
        [matrix[..., :3, 3], matrix[..., :3, 0], matrix[..., :3, 1]],
        axis=-1,
    )


def _rotation6d_to_matrix(rotation6d):
    first = rotation6d[..., :3]
    second_seed = rotation6d[..., 3:]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    second = second_seed - np.sum(first * second_seed, axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
    third = np.cross(first, second, axis=-1)
    return np.stack((first, second, third), axis=-1)


def _model_pose9_to_matrix(pose9):
    value = np.asarray(pose9)
    matrix = np.zeros(value.shape[:-1] + (4, 4), dtype=value.dtype)
    matrix[..., :3, :3] = _rotation6d_to_matrix(value[..., 3:])
    matrix[..., :3, 3] = value[..., :3]
    matrix[..., 3, 3] = 1
    return matrix


def _as_rgb_hwc(image, name):
    value = np.asarray(image)
    if value.ndim != 3:
        raise ValueError(f"{name} must be an HWC/CHW image, got {value.shape}")
    if value.shape[-1] != 3 and value.shape[0] == 3:
        value = value.transpose(1, 2, 0)
    if value.shape[-1] != 3:
        raise ValueError(f"{name} must have three channels, got {value.shape}")
    if value.dtype != np.uint8:
        raise TypeError(f"{name} must be uint8, got {value.dtype}")
    height, width = value.shape[:2]
    aspect = width / max(height, 1)
    if abs(aspect - 16.0 / 9.0) / (16.0 / 9.0) > 0.02:
        raise ValueError(f"{name} must preserve 16:9 aspect ratio, got {width}x{height}")
    return np.ascontiguousarray(value[..., ::-1])  # local BGR -> contract RGB


class _PyAVCoverResize:
    """Persistent libavfilter cover-resize graph for one live camera.

    This reproduces the geometry of the teleop-v2.1 conversion filter:

        scale=640:480:force_original_aspect_ratio=increase,crop=640:480

    A graph's buffer source is tied to its input dimensions/pixel format, so it
    is rebuilt only when a camera changes resolution. Calls are serialized
    because libavfilter graph instances are not safe for concurrent push/pull.
    """

    def __init__(self, width=DATASET_FRAME_SIZE[0], height=DATASET_FRAME_SIZE[1]):
        self.width = int(width)
        self.height = int(height)
        self._source_shape = None
        self._graph = None
        self._pts = 0
        self._lock = threading.Lock()

    @staticmethod
    def _frame(image_rgb, pts):
        frame = av.VideoFrame.from_ndarray(
            np.ascontiguousarray(image_rgb), format="rgb24"
        )
        frame.pts = int(pts)
        frame.time_base = Fraction(1, 30)
        return frame

    def _build_graph(self, template):
        graph = av.filter.Graph()
        source = graph.add_buffer(template=template)
        scale = graph.add(
            "scale",
            f"{self.width}:{self.height}:force_original_aspect_ratio=increase",
        )
        crop = graph.add("crop", f"{self.width}:{self.height}")
        pixel_format = graph.add("format", "pix_fmts=rgb24")
        sink = graph.add("buffersink")
        source.link_to(scale)
        scale.link_to(crop)
        crop.link_to(pixel_format)
        pixel_format.link_to(sink)
        graph.configure()
        self._graph = graph

    def __call__(self, image_rgb):
        value = np.asarray(image_rgb)
        if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
            raise ValueError(
                "PyAV cover-resize input must be uint8 RGB HWC, got "
                f"shape={value.shape} dtype={value.dtype}"
            )
        source_shape = tuple(int(v) for v in value.shape)
        with self._lock:
            frame = self._frame(value, self._pts)
            if self._graph is None or source_shape != self._source_shape:
                self._build_graph(frame)
                self._source_shape = source_shape
                self._pts = 0
                frame.pts = 0
            self._graph.push(frame)
            output = self._graph.pull().to_ndarray(format="rgb24")
            self._pts += 1
        expected = (self.height, self.width, 3)
        if output.shape != expected or output.dtype != np.uint8:
            raise RuntimeError(
                f"PyAV cover-resize produced {output.shape}/{output.dtype}, "
                f"expected {expected}/uint8"
            )
        return np.ascontiguousarray(output)


_DATASET_FRAME_RESIZERS = {
    "head": _PyAVCoverResize(),
    "left_wrist": _PyAVCoverResize(),
    "right_wrist": _PyAVCoverResize(),
}


def _resize_antialias(image_rgb, width, height):
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:
        raise RuntimeError("ImageWAM preprocessing requires PyTorch on the client") from error
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor = functional.interpolate(
        tensor, size=(height, width), mode="bilinear", align_corners=False, antialias=True
    )
    return tensor


def build_observation_payload(observation):
    """Build the ImageWAM request observation and return the model arm anchor."""
    head = _as_rgb_hwc(observation["image"], "head camera")
    left = _as_rgb_hwc(observation["image_wrist_left"], "left wrist camera")
    right = _as_rgb_hwc(observation["image_wrist_right"], "right wrist camera")
    # Reproduce the dataset conversion stage before constructing the ImageWAM
    # mosaic. The subsequent tile resize remains a separate operation, exactly
    # as it is during offline VAE-cache generation.
    head = _DATASET_FRAME_RESIZERS["head"](head)
    left = _DATASET_FRAME_RESIZERS["left_wrist"](left)
    right = _DATASET_FRAME_RESIZERS["right_wrist"](right)

    left_ee = np.asarray(observation["left_ee_pose"], dtype=np.float64)[0]
    right_ee = np.asarray(observation["right_ee_pose"], dtype=np.float64)[0]
    left_hand = np.asarray(observation["left_hand_joint"], dtype=np.float64)[0]
    right_hand = np.asarray(observation["right_hand_joint"], dtype=np.float64)[0]
    model_arm12 = np.concatenate(
        [
            _pose6_to_chest(left_ee, R_LEFT_HAND_TO_CHEST),
            _pose6_to_chest(right_ee, R_RIGHT_HAND_TO_CHEST),
        ]
    )
    model_pose18 = np.concatenate(
        [
            _model_pose9_from_matrix(pose_to_mat(model_arm12[:6])),
            _model_pose9_from_matrix(pose_to_mat(model_arm12[6:])),
        ]
    )
    state = np.concatenate(
        [model_pose18[:9], left_hand, model_pose18[9:], right_hand]
    ).astype(np.float32)
    if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
        raise ValueError(f"ImageWAM state must be finite {STATE_DIM}D, got {state.shape}")

    resized = [_resize_antialias(image, *size) for image, size in zip(
        (head, left, right), IMAGE_SIZES
    )]
    mosaic = np.concatenate(
        [resized[0].numpy(), np.concatenate([resized[1].numpy(), resized[2].numpy()], axis=-1)],
        axis=-2,
    )
    model_input = np.clip(mosaic * 2.0 - 1.0, -1.0, 1.0)[0].astype(np.float32)

    # The raw RGB fields are compact debug tiles; model_input_image is the
    # authoritative exact float input consumed by ImageWAM.
    debug_rgb = np.rint((model_input.transpose(1, 2, 0) + 1.0) * 127.5)
    debug_rgb = np.clip(debug_rgb, 0, 255).astype(np.uint8)
    payload = {
        "head_camera_rgb": np.ascontiguousarray(debug_rgb[:176, :]),
        "left_camera_rgb": np.ascontiguousarray(debug_rgb[176:, :160]),
        "right_camera_rgb": np.ascontiguousarray(debug_rgb[176:, 160:]),
        "model_input_image": np.ascontiguousarray(model_input),
        "joint_action_vector": np.ascontiguousarray(state),
        "image_source": IMAGE_SOURCE,
    }
    return payload, model_arm12


def build_request(observation, instruction, client_id="leo-inference"):
    payload, model_arm12 = build_observation_payload(observation)
    # Keep the legacy WAM location as well. ImageWAM reads the top-level field;
    # older WAM-compatible servers read obs.task_prompt.
    payload["task_prompt"] = str(instruction)
    image_source = payload.pop("image_source")
    return {
        "cmd": "infer_action_chunk",
        "request_id": uuid.uuid4().hex,
        "client_id": str(client_id),
        "observation": payload,
        "instruction": str(instruction),
        "image_source": image_source,
        "return_policy_image": False,
    }, model_arm12


def decode_action(action, current_model_arm12, action_type="relative"):
    """Decode ImageWAM [arm,hand,arm,hand] into robot-frame absolute targets."""
    value = np.asarray(action, dtype=np.float32)
    if value.ndim == 1:
        value = value[None, :]
    if value.ndim != 2 or value.shape[1] != ACTION_DIM:
        raise ValueError(f"ImageWAM action must have shape [T, {ACTION_DIM}], got {value.shape}")
    current = np.asarray(current_model_arm12, dtype=np.float64).reshape(12)
    left_target = _model_pose9_to_matrix(value[:, :9])
    right_target = _model_pose9_to_matrix(value[:, 29:38])
    if action_type == "relative":
        left_target = pose_to_mat(current[:6]) @ left_target
        right_target = pose_to_mat(current[6:]) @ right_target
    elif action_type != "absolute":
        raise ValueError(f"unsupported ImageWAM action_type={action_type!r}")
    left_arm = _pose6_from_chest(
        mat_to_pose(left_target), R_LEFT_HAND_TO_CHEST
    )
    right_arm = _pose6_from_chest(
        mat_to_pose(right_target), R_RIGHT_HAND_TO_CHEST
    )
    arm = np.concatenate([left_arm, right_arm], axis=-1)
    hand = np.concatenate([value[:, 9:29], value[:, 38:58]], axis=-1)
    return np.concatenate([arm, hand], axis=-1)
