from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from mano import ManoWujiConverter
from mano.native import NativeChunkDecoder, NativeMcpBatchSolver
from mano.native.runtime import native_extension_path


class ManoNativeSolverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.converters = [
            ManoWujiConverter(side) for side in ("left", "right")
        ]

    def test_native_sources_live_inside_unified_mano_package(self):
        repository = Path(__file__).resolve().parents[1]
        self.assertTrue((repository / "mano/native/_native_mcp_ik.cpp").is_file())
        self.assertFalse((repository / "utils/native_mano_ik").exists())

    def test_batch_model_accepts_both_calibrated_hands(self):
        solver = NativeMcpBatchSolver(self.converters)
        self.assertEqual(solver.hands, 2)
        self.assertEqual(solver.model["lower"].shape, (2, 20))
        self.assertEqual(solver.model["tip_points"].shape, (2, 5, 6, 3))

    @unittest.skipIf(native_extension_path() is None, "native extension is not built")
    def test_full_h32_chunk_recovers_every_representable_frame(self):
        horizon = 32
        expected = []
        theta = []
        anchors = []
        for hand, converter in enumerate(self.converters):
            limits = converter.wuji.joint_limits.astype(np.float64)
            center = limits.mean(axis=1)
            frame = np.arange(horizon, dtype=np.float64)[:, None]
            target = center + 0.06 * np.sin(
                frame * 0.13 + hand + np.arange(20) * 0.19
            )
            target = np.clip(target, limits[:, 0], limits[:, 1])
            expected.append(target)
            theta.append(converter.q20_to_theta(target))
            anchors.append(target[0])

        rotations = Rotation.from_rotvec(
            np.asarray(theta).reshape(-1, 3)
        ).as_matrix().reshape(2, horizon, 15, 3, 3)
        decoded = NativeChunkDecoder(
            NativeMcpBatchSolver(self.converters),
            max_iterations=6,
            num_threads=8,
        ).decode(rotations, np.asarray(anchors))

        self.assertEqual(decoded.shape, (2, horizon, 20))
        error = decoded - np.asarray(expected)
        self.assertLess(float(np.sqrt(np.mean(error**2))), 1e-4)
        self.assertLess(float(np.max(np.abs(error))), 1e-3)


if __name__ == "__main__":
    unittest.main()
