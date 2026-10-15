import numpy as np
import unittest

from mano import ManoWujiConverter


class ManoInverseConverterTest(unittest.TestCase):
    def test_v4_calibration_is_zero_mano(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                converter = ManoWujiConverter(side)
                theta = converter.q20_to_theta(converter.calibration_q0)
                np.testing.assert_allclose(theta, 0.0, atol=1e-7, rtol=0.0)
                self.assertEqual(converter.q20_to_rot6d(converter.calibration_q0).shape, (90,))


    def test_calibration_roundtrip(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                converter = ManoWujiConverter(side)
                q20 = converter.calibration_q0.astype(np.float32)
                recovered = converter.theta_to_q20(
                    converter.q20_to_theta(q20), initial_q20=q20, iterations=5
                )
                np.testing.assert_allclose(recovered, q20, atol=2e-4, rtol=0.0)
