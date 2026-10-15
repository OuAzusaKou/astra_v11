import unittest
from unittest import mock

import numpy as np

from utils import liber_teleop_v22_client as contract


IDENTITY_ROW6 = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)


class _FakeManoConverter:
    def convert(self, fields):
        horizon = fields["observation.left_arm_ee_state"].shape[0]

        def pose9(x):
            value = np.zeros((horizon, 9), dtype=np.float32)
            value[:, :3] = x
            value[:, 3:] = IDENTITY_ROW6
            return value

        theta = np.tile(IDENTITY_ROW6, (horizon, 15))
        return {
            "observation.mano.left_ee_state": pose9([1, 2, 3]),
            "observation.mano.right_ee_state": pose9([4, 5, 6]),
            "observation.mano.left_hand_theta": theta.copy(),
            "observation.mano.right_hand_theta": theta.copy(),
            "observation.mano.left_hand_beta": np.full((horizon, 10), 0.1, np.float32),
            "observation.mano.right_hand_beta": np.full((horizon, 10), 0.2, np.float32),
        }


class _FakeManoChunkDecoder:
    def decode(self, rotations, anchors):
        horizon = np.asarray(rotations).shape[1]
        return np.repeat(np.asarray(anchors, np.float32)[:, None, :], horizon, axis=1)


def _pose6_to_matrix(pose6):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = contract.Rotation.from_rotvec(pose6[3:]).as_matrix()
    matrix[:3, 3] = pose6[:3]
    return matrix


def _camera_palm_pose9(chest_flange_pose6):
    matrix = (
        contract.CAMERA_STANDARD_T_CHEST_LINK
        @ _pose6_to_matrix(chest_flange_pose6)
        @ contract.CHEST_FLANGE_T_PALM
    )
    return np.concatenate([matrix[:3, 3], matrix[:2, :3].reshape(6)]).astype(np.float32)


class LiberTeleopV22ClientTest(unittest.TestCase):
    def _observation(self):
        return {
            "image": np.zeros((72, 128, 3), dtype=np.uint8),
            "image_wrist_left": np.zeros((72, 128, 3), dtype=np.uint8),
            "image_wrist_right": np.zeros((72, 128, 3), dtype=np.uint8),
            "left_ee_pose": np.asarray([[0.1, 0.2, 0.3, 0.2, -0.1, 0.3]], np.float32),
            "right_ee_pose": np.asarray([[-0.1, 0.4, 0.2, -0.2, 0.3, 0.1]], np.float32),
            "left_hand_joint": np.arange(20, dtype=np.float32)[None, :] / 10,
            "right_hand_joint": np.arange(20, dtype=np.float32)[None, :] / 20,
        }

    def test_build_request_uses_only_dataset_named_fields(self):
        fake_images = {
            "observation.images.image_ego_0": np.zeros((480, 640, 3), np.uint8),
            "observation.images.image_wrist_left": np.zeros((480, 640, 3), np.uint8),
            "observation.images.image_wrist_right": np.zeros((480, 640, 3), np.uint8),
        }
        with mock.patch.object(contract, "_prepare_images", return_value=fake_images):
            request, context = contract.build_request(
                self._observation(), "pick up the object", converter=_FakeManoConverter()
            )

        self.assertEqual(request["protocol"], "liber_teleop_v22")
        self.assertEqual(request["instruction"], "pick up the object")
        wire = request["observation"]
        self.assertNotIn("joint_action_vector", wire)
        self.assertNotIn("state_valid_mask", wire)
        self.assertNotIn("action_component_mask", wire)
        self.assertNotIn("normalizer_id", wire)
        expected_widths = {
            "observation.left_arm_ee_state": 6,
            "observation.right_arm_ee_state": 6,
            "observation.left_hand_joint_state": 20,
            "observation.right_hand_joint_state": 20,
            "observation.mano.left_ee_state": 9,
            "observation.mano.right_ee_state": 9,
            "observation.mano.left_hand_theta": 90,
            "observation.mano.right_hand_theta": 90,
            "observation.mano.left_hand_beta": 10,
            "observation.mano.right_hand_beta": 10,
        }
        for key, width in expected_widths.items():
            self.assertEqual(wire[key].shape, (width,), key)
            self.assertEqual(wire[key].dtype, np.float32, key)
        for key in fake_images:
            self.assertEqual(wire[key].shape, (480, 640, 3))
            self.assertEqual(wire[key].dtype, np.uint8)

        self.assertEqual(context.current_mano_ee18.shape, (18,))
        self.assertEqual(context.current_mano_theta180.shape, (180,))
        self.assertEqual(context.current_q20_40.shape, (40,))
        self.assertEqual(context.current_chest_arm12.shape, (12,))

    def test_build_request_includes_dataset_name_when_selected(self):
        fake_images = {
            "observation.images.image_ego_0": np.zeros((480, 640, 3), np.uint8),
            "observation.images.image_wrist_left": np.zeros((480, 640, 3), np.uint8),
            "observation.images.image_wrist_right": np.zeros((480, 640, 3), np.uint8),
        }
        with mock.patch.object(contract, "_prepare_images", return_value=fake_images):
            request, _ = contract.build_request(
                self._observation(),
                "pick up the object",
                dataset_name="pickplace_green_apple_to_bowl",
                converter=_FakeManoConverter(),
            )

        self.assertEqual(request["dataset_name"], "pickplace_green_apple_to_bowl")

    def test_client_timing_separates_server_processing_from_network_rtt(self):
        response = {
            "timing": {"inference_ms": 80.0, "server_processing_ms": 95.0}
        }

        contract.add_client_timing(response, round_trip_ms=112.5)

        self.assertEqual(response["timing"]["inference_ms"], 80.0)
        self.assertEqual(response["timing"]["server_processing_ms"], 95.0)
        self.assertEqual(response["timing"]["network_round_trip_ms"], 17.5)
        self.assertEqual(response["timing"]["round_trip_ms"], 112.5)

    def test_server_inference_profile_fails_closed_on_wrong_variant(self):
        contract.validate_server_inference_profile(
            {"num_inference_steps": 10, "image_num_inference_steps": 5},
            expected_num_inference_steps=10,
            expected_image_num_inference_steps=5,
        )

        with self.assertRaisesRegex(RuntimeError, "wrong ImageWAM inference profile"):
            contract.validate_server_inference_profile(
                {"num_inference_steps": 10, "image_num_inference_steps": 10},
                expected_num_inference_steps=10,
                expected_image_num_inference_steps=5,
            )

    def test_identity_326d_delta_round_trips_current_robot_arm(self):
        left_chest = np.asarray([0.2, 0.1, 0.4, 0.2, -0.3, 0.1], np.float32)
        right_chest = np.asarray([-0.2, 0.3, 0.5, -0.1, 0.25, -0.2], np.float32)
        context = contract.ActionDecodeContext(
            current_mano_ee18=np.concatenate(
                [_camera_palm_pose9(left_chest), _camera_palm_pose9(right_chest)]
            ),
            current_mano_theta180=np.tile(IDENTITY_ROW6, 30),
            current_q20_40=np.concatenate(
                [np.full(20, 0.25, np.float32), np.full(20, 0.75, np.float32)]
            ),
            current_chest_arm12=np.concatenate([left_chest, right_chest]),
        )
        action = np.zeros((2, 326), dtype=np.float32)
        action[:, 3:9] = IDENTITY_ROW6
        action[:, 128:134] = IDENTITY_ROW6
        action[:, 10:100] = np.tile(IDENTITY_ROW6, 15)
        action[:, 135:225] = np.tile(IDENTITY_ROW6, 15)
        action[:, 100:120] = -10.0
        action[:, 225:245] = 10.0

        with mock.patch.object(
            contract, "_mano_chunk_decoder", return_value=_FakeManoChunkDecoder()
        ):
            decoded = contract.decode_action(action, context, hand_action_source="mano")
        expected_arm = np.concatenate(
            [
                contract._pose6_from_chest(left_chest, contract.R_LEFT_HAND_TO_CHEST),
                contract._pose6_from_chest(right_chest, contract.R_RIGHT_HAND_TO_CHEST),
            ]
        )
        np.testing.assert_allclose(
            decoded[:, :12], np.repeat(expected_arm[None, :], 2, axis=0), atol=2e-6
        )
        np.testing.assert_array_equal(decoded[:, 12:32], 0.25)
        np.testing.assert_array_equal(decoded[:, 32:52], 0.75)

    def test_named_absolute_action_decodes_to_existing_52d_control_layout(self):
        left = np.asarray([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0]], np.float32)
        right = np.asarray([[0.4, 0.5, 0.6, 0.1, -0.2, 0.3]], np.float32)
        named = {
            "action.left_arm_ee_action": left,
            "action.right_arm_ee_action": right,
            "action.left_hand_joint_action": np.full((1, 20), 1.0, np.float32),
            "action.right_hand_joint_action": np.full((1, 20), 2.0, np.float32),
        }
        unused_context = contract.ActionDecodeContext(
            current_mano_ee18=np.zeros(18, np.float32),
            current_mano_theta180=np.tile(IDENTITY_ROW6, 30),
            current_q20_40=np.zeros(40, np.float32),
            current_chest_arm12=np.zeros(12, np.float32),
        )
        decoded = contract.decode_action(
            named, unused_context, hand_action_source="q20"
        )
        self.assertEqual(decoded.shape, (1, 52))
        np.testing.assert_allclose(
            decoded[0, :6],
            contract._pose6_from_chest(left[0], contract.R_LEFT_HAND_TO_CHEST),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            decoded[0, 6:12],
            contract._pose6_from_chest(right[0], contract.R_RIGHT_HAND_TO_CHEST),
            atol=1e-6,
        )
        np.testing.assert_array_equal(decoded[0, 12:32], 1.0)
        np.testing.assert_array_equal(decoded[0, 32:52], 2.0)

    def test_named_mano_ee_action_also_decodes_to_control_layout(self):
        left_chest = np.asarray([0.2, 0.1, 0.4, 0.2, -0.3, 0.1], np.float32)
        right_chest = np.asarray([-0.2, 0.3, 0.5, -0.1, 0.25, -0.2], np.float32)
        named = {
            "action.mano.left_ee_action": _camera_palm_pose9(left_chest),
            "action.mano.right_ee_action": _camera_palm_pose9(right_chest),
            "action.mano.left_hand_theta": np.tile(IDENTITY_ROW6, 15),
            "action.mano.right_hand_theta": np.tile(IDENTITY_ROW6, 15),
        }
        unused_context = contract.ActionDecodeContext(
            current_mano_ee18=np.zeros(18, np.float32),
            current_mano_theta180=np.tile(IDENTITY_ROW6, 30),
            current_q20_40=np.concatenate(
                [np.full(20, 1.0, np.float32), np.full(20, 2.0, np.float32)]
            ),
            current_chest_arm12=np.zeros(12, np.float32),
        )
        with mock.patch.object(
            contract, "_mano_chunk_decoder", return_value=_FakeManoChunkDecoder()
        ):
            decoded = contract.decode_action(
                named, unused_context, hand_action_source="mano"
            )
        expected = np.concatenate(
            [
                contract._pose6_from_chest(left_chest, contract.R_LEFT_HAND_TO_CHEST),
                contract._pose6_from_chest(right_chest, contract.R_RIGHT_HAND_TO_CHEST),
            ]
        )
        self.assertEqual(decoded.shape, (1, 52))
        np.testing.assert_allclose(decoded[0, :12], expected, atol=2e-6)
        np.testing.assert_array_equal(decoded[0, 12:32], 1.0)
        np.testing.assert_array_equal(decoded[0, 32:52], 2.0)

    def test_q20_source_uses_q20_head_without_mano_ik(self):
        context = contract.ActionDecodeContext(
            current_mano_ee18=np.concatenate(
                [_camera_palm_pose9(np.zeros(6)), _camera_palm_pose9(np.zeros(6))]
            ),
            current_mano_theta180=np.tile(IDENTITY_ROW6, 30),
            current_q20_40=np.zeros(40, np.float32),
            current_chest_arm12=np.zeros(12, np.float32),
        )
        action = np.zeros((2, 326), dtype=np.float32)
        action[:, 3:9] = IDENTITY_ROW6
        action[:, 128:134] = IDENTITY_ROW6
        action[:, 100:120] = 0.25
        action[:, 225:245] = 0.75
        with mock.patch.object(
            contract, "_mano_chunk_decoder", side_effect=AssertionError("IK must not run")
        ):
            decoded = contract.decode_action(
                action, context, hand_action_source="q20"
            )
        np.testing.assert_array_equal(decoded[:, 12:32], 0.25)
        np.testing.assert_array_equal(decoded[:, 32:52], 0.75)

    def test_teleop_q20_delta_uses_chest_flange_anchor(self):
        left_chest = np.asarray([0.2, 0.1, 0.4, 0.2, -0.3, 0.1], np.float32)
        right_chest = np.asarray([-0.2, 0.3, 0.5, -0.1, 0.25, -0.2], np.float32)
        context = contract.ActionDecodeContext(
            # Deliberately unrelated MANO anchors: this adapter must not use them.
            current_mano_ee18=np.concatenate(
                [_camera_palm_pose9(np.ones(6)), _camera_palm_pose9(-np.ones(6))]
            ),
            current_mano_theta180=np.tile(IDENTITY_ROW6, 30),
            current_q20_40=np.zeros(40, np.float32),
            current_chest_arm12=np.concatenate([left_chest, right_chest]),
        )
        action = np.zeros((2, 326), dtype=np.float32)
        action[:, 3:9] = IDENTITY_ROW6
        action[:, 128:134] = IDENTITY_ROW6
        action[:, 100:120] = 0.25
        action[:, 225:245] = 0.75

        decoded = contract.decode_action(
            action,
            context,
            hand_action_source="q20",
            adapter_name="teleop_q20",
        )

        expected_arm = np.concatenate(
            [
                contract._pose6_from_chest(left_chest, contract.R_LEFT_HAND_TO_CHEST),
                contract._pose6_from_chest(right_chest, contract.R_RIGHT_HAND_TO_CHEST),
            ]
        )
        np.testing.assert_allclose(
            decoded[:, :12], np.repeat(expected_arm[None, :], 2, axis=0), atol=2e-6
        )
        np.testing.assert_array_equal(decoded[:, 12:32], 0.25)
        np.testing.assert_array_equal(decoded[:, 32:52], 0.75)


if __name__ == "__main__":
    unittest.main()
