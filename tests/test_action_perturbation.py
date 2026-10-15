import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from utils.action_perturbation import RandomEEActionPerturber


class RandomEEActionPerturberTest(unittest.TestCase):
    def test_ramps_to_exact_bounded_offsets_without_mutating_input(self):
        action = np.zeros((32, 12), dtype=np.float32)
        action[:, 3:6] = np.array([0.2, -0.3, 0.4], dtype=np.float32)
        action[:, 9:12] = np.array([-0.5, 0.1, 0.2], dtype=np.float32)
        original = action.copy()

        perturbed, sample = RandomEEActionPerturber(5.0, 10.0, seed=17).apply(action)

        np.testing.assert_array_equal(action, original)
        self.assertEqual(perturbed.shape, (32, 12))
        self.assertEqual(perturbed.dtype, np.float32)
        expected_scale = (np.arange(32, dtype=np.float64) + 1.0) / 32.0
        for xyz_start, rot_start in ((0, 3), (6, 9)):
            translation = perturbed[:, xyz_start : xyz_start + 3] - action[
                :, xyz_start : xyz_start + 3
            ]
            translation_norm = np.linalg.norm(translation, axis=1)
            np.testing.assert_allclose(
                translation_norm, 0.05 * expected_scale, atol=1e-7
            )

            old_rotation = Rotation.from_rotvec(action[:, rot_start : rot_start + 3])
            new_rotation = Rotation.from_rotvec(perturbed[:, rot_start : rot_start + 3])
            angle = (new_rotation * old_rotation.inv()).magnitude()
            np.testing.assert_allclose(
                angle, np.deg2rad(10.0) * expected_scale, atol=1e-6
            )

        self.assertAlmostEqual(np.linalg.norm(sample.left_translation_m), 0.05)
        self.assertAlmostEqual(np.linalg.norm(sample.right_translation_m), 0.05)

    def test_seed_reproduces_chunk_sequence(self):
        action = np.zeros((3, 12), dtype=np.float64)
        first = RandomEEActionPerturber(5.0, 10.0, seed=99)
        second = RandomEEActionPerturber(5.0, 10.0, seed=99)

        for _ in range(3):
            np.testing.assert_allclose(first.apply(action)[0], second.apply(action)[0])

    def test_rejects_malformed_action(self):
        perturber = RandomEEActionPerturber(5.0, 10.0, seed=1)
        with self.assertRaisesRegex(ValueError, "shape"):
            perturber.apply(np.zeros((32, 11)))


if __name__ == "__main__":
    unittest.main()
