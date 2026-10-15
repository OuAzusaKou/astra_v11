"""One client for standardized ImageWAM (58D) and LiberTeleop v2.2 (326D).

This 'mix' selects a wire protocol; it does not change the robot coordinate
frame to the unrelated historical MIX policy's camera/palm frame.
"""

from dataclasses import dataclass
import uuid

import numpy as np

from utils import liber_teleop_v22_client as v22


@dataclass(frozen=True)
class StandardizedContext:
    chest_arm12: np.ndarray
    hand40: np.ndarray


def build_standardized_request(observation, instruction, crop=(100, 100, 100), image_preprocess="server"):
    """Match the legacy WAM client: BGR CHW + LA9,RA9,LH20,RH20.

    Camera orientation is already corrected by LeoWujiInferEnv. The server
    owns MANO conversion and IK. With image_preprocess='client', the client
    sends the training-compatible float32 image mosaic to the optimized server.
    """
    arms = np.concatenate([
        v22._pose6_to_chest(
            np.asarray(observation[key], dtype=np.float32)[0], rotation
        )
        for key, rotation in (
            ("left_ee_pose", v22.R_LEFT_HAND_TO_CHEST),
            ("right_ee_pose", v22.R_RIGHT_HAND_TO_CHEST),
        )
    ]).astype(np.float32)
    hands = np.concatenate([
        np.asarray(observation[key], dtype=np.float32)[0]
        for key in ("left_hand_joint", "right_hand_joint")
    ]).astype(np.float32)
    poses = v22._pose6_to_matrix(arms.reshape(2, 6))
    # Standardized WAM uses the first two ROWS of R, unlike the native
    # interleaved ImageWAM contract's first two columns.
    pose18 = np.concatenate([poses[:, :3, 3], poses[:, :2, :3].reshape(2, 6)], axis=1)
    state = np.concatenate([pose18.reshape(18), hands]).astype(np.float32)
    payload = {"state": state, "image_source": "raw_camera"}
    sizes = []
    for index, key in enumerate(("image", "image_wrist_left", "image_wrist_right")):
        frame = np.asarray(observation[key])
        if image_preprocess == "server":
            payload[f"camera_{index}"] = np.ascontiguousarray(frame.transpose(2, 0, 1))
        sizes.append([frame.shape[1], frame.shape[0]])
    payload["camera_preprocess"] = {
        "schema": "liberteleop_v21_raw_v2",
        "camera_order": ["head", "wrist_left", "wrist_right"],
        "wire_layout": "CHW", "wire_dtype": "uint8", "wire_color": "bgr",
        "wire_orientation": "training_v21",
        "live_ros_to_wire_rotation_degrees": [0, 180, 180],
        "crop_percent": list(crop), "output_size_wh": [640, 480],
        "geometry": "percent_center_crop_even_then_cover_resize",
        "preprocess": "v21_encoded",
        "encoding": {"codec": "libx264", "pix_fmt": "yuv420p", "crf": 20, "gop": 2},
        "source_size_wh": sizes,
    }
    if image_preprocess == "client":
        from utils.imagewam_camera import prepare_model_input
        frames = [observation[key] for key in ("image", "image_wrist_left", "image_wrist_right")]
        payload = {
            "state": state, "image_source": "v21_model_input",
            "model_input_image": prepare_model_input(frames, crop),
            "camera_trace": {"raw_size_wh": sizes, "processed_size_wh": [[640, 480]] * 3},
            "camera_preprocess": dict(payload["camera_preprocess"],
                                      schema="imagewam_v21_model_input_v1", wire_dtype="float32",
                                      wire_color="rgb", model_input_size_wh=[288, 336]),
        }
    return {
        "cmd": "infer_action_chunk", "request_id": uuid.uuid4().hex,
        "obs": payload, "instruction": str(instruction),
    }, StandardizedContext(arms, hands)


def decode_standardized_action(action, context, action_type):
    """Decode standardized arms-first action58, including relative Q20 hands."""
    value = np.asarray(action, dtype=np.float32)
    arms = []
    for start, anchor, rotation in ((0, context.chest_arm12[:6], v22.R_LEFT_HAND_TO_CHEST),
                                    (9, context.chest_arm12[6:], v22.R_RIGHT_HAND_TO_CHEST)):
        target = v22._pose9_row_to_matrix(value[:, start:start + 9])
        if action_type != "absolute":
            target = v22._pose6_to_matrix(anchor) @ target
        arms.append(v22._pose6_from_chest(v22._matrix_to_pose6(target), rotation))
    hands = value[:, 18:].copy()
    if action_type == "relative":
        hands += context.hand40
    return np.ascontiguousarray(np.concatenate([*arms, hands], axis=1), dtype=np.float32)


class MixPolicyClient:
    """Persistent transport with explicit per-model request and decode adapters."""

    def __init__(self, transport, policy_config, *, hand_source="mano", crop=(100, 100, 100)):
        self.transport = transport
        self.config = policy_config
        self.protocol = policy_config["protocol"]
        self.hand_source = hand_source
        self.crop = crop
        self.action_row_offset = None

    def connect(self):
        self.transport.connect()

    def disconnect(self):
        self.transport.disconnect()

    def build_request(self, observation, instruction, **kwargs):
        if self.protocol == "imagewam_standardized":
            return build_standardized_request(observation, instruction, self.crop,
                                              self.config.get("image_preprocess", "server"))
        return v22.build_request(observation, instruction, **kwargs)

    def send_request(self, request):
        return self.transport.send_request(request)

    def decode_response(self, response, context):
        info = response["server_info"]
        # Older v2.2 servers omit the row offset; their task configs supply it.
        self.action_row_offset = info.get("action_row_offset", self.config.get("action_row_offset"))
        if self.protocol == "imagewam_standardized":
            return decode_standardized_action(response["action"], context, response["action_type"])
        return v22.decode_action(
            response["action"], context, hand_action_source=self.hand_source,
            adapter_name=info["adapter_name"],
        )
