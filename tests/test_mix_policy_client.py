import unittest
from unittest import mock

import numpy as np

from utils import liber_teleop_v22_client as v22
from utils.mix_policy_client import (
    MixPolicyClient, build_standardized_request, decode_standardized_action,
)
from utils.client_config import REPO_ROOT, load_client_config
from scripts.inference_leo.inference_multiview_leo_mix import TASKS, parse_task_args


IDENTITY6 = [1, 0, 0, 0, 1, 0]


def observation():
    return {
        "image": np.full((18, 32, 3), [1, 2, 3], dtype=np.uint8),
        "image_wrist_left": np.full((18, 32, 3), [4, 5, 6], dtype=np.uint8),
        "image_wrist_right": np.full((18, 32, 3), [7, 8, 9], dtype=np.uint8),
        "left_ee_pose": np.array([[0.2, 0.3, 0.4, 0.1, -0.2, 0.3]], np.float32),
        "right_ee_pose": np.array([[-0.3, 0.2, 0.5, -0.2, 0.3, 0.1]], np.float32),
        "left_hand_joint": np.full((1, 20), 0.2, np.float32),
        "right_hand_joint": np.full((1, 20), 0.4, np.float32),
    }


def standardized_action():
    return np.tile([0, 0, 0] + IDENTITY6 + [0, 0, 0] + IDENTITY6 + [0.05] * 40, (32, 1))


class MixPolicyClientTest(unittest.TestCase):
    def test_client_preprocessing_transmits_mosaic_and_preserves_state(self):
        obs = observation()
        raw, raw_context = build_standardized_request(obs, "do task")
        mosaic = np.zeros((3, 336, 288), np.float32)
        with mock.patch("utils.imagewam_camera.prepare_model_input", return_value=mosaic) as build_image:
            request, context = build_standardized_request(obs, "do task", image_preprocess="client")
        self.assertEqual(request["obs"]["image_source"], "v21_model_input")
        self.assertEqual(request["obs"]["camera_preprocess"]["wire_dtype"], "float32")
        self.assertIs(request["obs"]["model_input_image"], mosaic)
        for index in range(3):
            self.assertNotIn(f"camera_{index}", request["obs"])
        self.assertIs(build_image.call_args.args[0][0], obs["image"])
        np.testing.assert_array_equal(request["obs"]["state"], raw["obs"]["state"])
        np.testing.assert_array_equal(context.chest_arm12, raw_context.chest_arm12)
        np.testing.assert_array_equal(context.hand40, raw_context.hand40)

    def test_standardized_wire_preserves_bgr_and_arms_first_layout(self):
        obs = observation()
        request, context = build_standardized_request(obs, "do task")
        self.assertNotIn("observation", request)
        self.assertEqual(request["instruction"], "do task")
        wire = request["obs"]
        np.testing.assert_array_equal(wire["camera_0"][:, 0, 0], [1, 2, 3])
        self.assertEqual(wire["camera_0"].dtype, np.uint8)
        np.testing.assert_allclose(wire["state"][:3], context.chest_arm12[:3])
        np.testing.assert_allclose(wire["state"][9:12], context.chest_arm12[6:9])
        np.testing.assert_array_equal(wire["state"][18:], context.hand40)
        self.assertEqual(wire["camera_preprocess"]["source_size_wh"], [[32, 18]] * 3)
        self.assertEqual(wire["camera_preprocess"]["live_ros_to_wire_rotation_degrees"], [0, 180, 180])

    def test_identity_relative_arms_round_trip_and_hand_semantics(self):
        obs = observation()
        _, context = build_standardized_request(obs, "task")
        raw = standardized_action()
        expected_arm = np.concatenate([obs["left_ee_pose"], obs["right_ee_pose"]], axis=1)
        for mode, hands in (("relative_arm_absolute_hand", np.full(40, 0.05)),
                            ("relative", context.hand40 + 0.05)):
            decoded = decode_standardized_action(raw, context, mode)
            np.testing.assert_allclose(decoded[:, :12], np.repeat(expected_arm, 32, axis=0), atol=1e-6)
            np.testing.assert_allclose(decoded[:, 12:], np.tile(hands, (32, 1)), atol=1e-6)

    def test_relative_arm_translation_is_in_moving_frame(self):
        from scipy.spatial.transform import Rotation
        _, context = build_standardized_request(observation(), "task")
        raw = standardized_action()
        raw[:, :3] = [0.01, 0.02, -0.03]
        decoded = decode_standardized_action(raw, context, "relative_arm_absolute_hand")
        expected_xyz = context.chest_arm12[:3] + Rotation.from_rotvec(context.chest_arm12[3:6]).apply(raw[0, :3])
        np.testing.assert_allclose(decoded[0, :3], expected_xyz, atol=1e-6)

    def test_standardized_rotation6d_uses_rows_for_request_and_response(self):
        from scipy.spatial.transform import Rotation
        request, context = build_standardized_request(observation(), "task")
        rotation = Rotation.from_rotvec(context.chest_arm12[3:6]).as_matrix()
        np.testing.assert_allclose(request["obs"]["state"][3:9], rotation[:2].reshape(6), atol=1e-7)
        target_rotation = Rotation.from_euler("xyz", [0.3, -0.5, 0.7]).as_matrix()
        raw = standardized_action()
        raw[:, 3:9] = target_rotation[:2].reshape(6)
        decoded = decode_standardized_action(raw, context, "absolute")
        actual_rotation = Rotation.from_rotvec(decoded[0, 3:6]).as_matrix()
        np.testing.assert_allclose(actual_rotation, target_rotation @ v22.R_LEFT_HAND_TO_CHEST, atol=1e-6)

    def test_both_protocols_decode_to_52d(self):
        for task in TASKS:
            with self.subTest(task=task):
                cfg = load_client_config(REPO_ROOT / "config" / f"client_dex_bench_mix_{task}.yaml")
                client = MixPolicyClient(mock.Mock(), cfg["policy"])
                if task in ("1", "1-q20", "2"):
                    _, context = client.build_request(observation(), cfg["instruction"])
                    response = {
                        "success": True, "action": standardized_action(),
                        "action_type": "relative_arm_absolute_hand",
                        "server_info": {"infer_kind": "imagewam_standardized", "action_row_offset": 1,
                                        "response_action_order": "LA9,RA9,LH20,RH20", "instruction": cfg["instruction"]},
                    }
                else:
                    context = v22.ActionDecodeContext(
                        np.array(([0, 0, 0] + IDENTITY6) * 2, np.float32),
                        np.array(IDENTITY6 * 30, np.float32), np.zeros(40, np.float32), np.zeros(12, np.float32),
                    )
                    raw = np.zeros((32, 326), np.float32)
                    raw[:, 3:9] = IDENTITY6
                    raw[:, 128:134] = IDENTITY6
                    raw[:, 10:100] = IDENTITY6 * 15
                    raw[:, 135:225] = IDENTITY6 * 15
                    response = {"success": True, "action": raw,
                                "server_info": {"infer_kind": "imagewam", "protocol": "liber_teleop_v22",
                                                "adapter_name": "teleop_mano", "num_inference_steps": 10}}
                with mock.patch.object(v22, "_decode_mano_hand_chunk", return_value=np.zeros((32, 40), np.float32)):
                    self.assertEqual(client.decode_response(response, context).shape, (32, 52))
                self.assertEqual(client.action_row_offset, 1 if task in ("1", "1-q20", "2") else 0)

    def test_task_profiles_select_expected_endpoints_and_existing_reset_states(self):
        import json
        endpoints = {"1": (19213, 18213), "1-q20": (19225, 18225), "2": (19221, 18221),
                     "3": (29555, 29566), "4": (29557, 29569)}
        for task in TASKS:
            args, path, rest = parse_task_args(["--task", task, "--camera-latency", "0.2"])
            self.assertEqual(rest, ["--camera-latency", "0.2"])
            cfg = load_client_config(path)
            self.assertEqual((cfg["policy"]["port"], cfg["tunnel"]["remote_port"]), endpoints[task])
            self.assertEqual(cfg["runtime"]["inference_mode"], "sync-no-sleep")
            self.assertEqual(cfg["startup"]["state_path"], cfg["startup"]["reset_state_path"])
            with open(cfg["startup"]["state_path"]) as handle:
                state = json.load(handle)["mean_by_key"]
            for key, size in (("left_ee_pose", 6), ("right_ee_pose", 6),
                              ("left_hand_state", 20), ("right_hand_state", 20)):
                self.assertEqual(np.asarray(state[key]).shape, (size,))
                self.assertTrue(np.isfinite(state[key]).all())


if __name__ == "__main__":
    unittest.main()
