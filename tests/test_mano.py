import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import mano
from mano import TeleopManoFieldConverter, verify_bundled_assets
from mano.coordinates import (
    CAMERA_STANDARD_T_CHEST_LINK,
    PALM_OFFSET_M,
    chest_flange_pose6_to_camera_palm_pose9,
)


class VendoredManoFieldsTest(unittest.TestCase):
    def test_ee_conversion_matches_full_homogeneous_transform(self):
        pose6 = np.asarray(
            [
                [0.1, -0.2, 0.3, 0.0, np.pi / 2, 0.0],
                [
                    -0.4,
                    0.5,
                    0.2,
                    np.pi / np.sqrt(3),
                    np.pi / np.sqrt(3),
                    np.pi / np.sqrt(3),
                ],
            ],
            dtype=np.float64,
        )
        chest_flange = np.broadcast_to(np.eye(4), (len(pose6), 4, 4)).copy()
        chest_flange[:, :3, :3] = Rotation.from_rotvec(pose6[:, 3:]).as_matrix()
        chest_flange[:, :3, 3] = pose6[:, :3]
        flange_palm = np.eye(4)
        flange_palm[0, 3] = PALM_OFFSET_M

        camera_palm = (
            CAMERA_STANDARD_T_CHEST_LINK[None, :, :]
            @ chest_flange
            @ flange_palm[None, :, :]
        )
        expected = np.concatenate(
            [
                camera_palm[:, :3, 3],
                camera_palm[:, :2, :3].reshape(-1, 6),
            ],
            axis=1,
        ).astype(np.float32)

        np.testing.assert_allclose(
            chest_flange_pose6_to_camera_palm_pose9(pose6),
            expected,
            rtol=0.0,
            atol=1e-7,
        )

    def test_package_and_all_assets_are_inside_deploy_checkout(self):
        repository = Path(__file__).resolve().parents[1]
        package_root = Path(mano.__file__).resolve().parent
        self.assertEqual(package_root, repository / "mano")
        self.assertFalse((repository / "mano_bridge").exists())

        verified = verify_bundled_assets()
        self.assertEqual(len(verified), 8)
        self.assertEqual(
            {path for path in verified if path.endswith(".urdf")},
            {
                "assets/robots/wuji/wuji_left_hand.urdf",
                "assets/robots/wuji/wuji_right_hand.urdf",
                "assets/robots/wuji/wuji_left_hand_mano45_v4.urdf",
                "assets/robots/wuji/wuji_right_hand_mano45_v4.urdf",
            },
        )

    def test_converter_produces_complete_finite_mano_contract(self):
        zeros6 = np.zeros((1, 6), dtype=np.float32)
        zeros20 = np.zeros((1, 20), dtype=np.float32)
        fields = {
            "observation.left_arm_ee_state": zeros6,
            "observation.right_arm_ee_state": zeros6,
            "observation.left_hand_joint_state": zeros20,
            "observation.right_hand_joint_state": zeros20,
            "action.left_arm_ee_action": zeros6,
            "action.right_arm_ee_action": zeros6,
            "action.left_hand_joint_action": zeros20,
            "action.right_hand_joint_action": zeros20,
        }

        converted = TeleopManoFieldConverter().convert(fields)
        expected_widths = {
            "observation.mano.left_ee_state": 9,
            "observation.mano.right_ee_state": 9,
            "observation.mano.left_hand_theta": 90,
            "observation.mano.right_hand_theta": 90,
            "observation.mano.left_hand_beta": 10,
            "observation.mano.right_hand_beta": 10,
            "action.mano.left_ee_action": 9,
            "action.mano.right_ee_action": 9,
            "action.mano.left_hand_theta": 90,
            "action.mano.right_hand_theta": 90,
        }
        self.assertEqual(set(converted), set(expected_widths))
        for key, width in expected_widths.items():
            self.assertEqual(converted[key].shape, (1, width), key)
            self.assertEqual(converted[key].dtype, np.float32, key)
            self.assertTrue(np.isfinite(converted[key]).all(), key)


if __name__ == "__main__":
    unittest.main()
