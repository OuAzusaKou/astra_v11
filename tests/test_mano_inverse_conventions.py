import numpy as np
import unittest

from mano.conventions import (
    mano_to_thumb_first,
    rot6d_to_theta,
    theta_to_rot6d,
    thumb_first_to_mano,
)


class ManoInverseConventionsTest(unittest.TestCase):
    def test_joint_reorder_is_invertible(self):
        native = np.arange(45).reshape(15, 3)
        np.testing.assert_array_equal(
            thumb_first_to_mano(mano_to_thumb_first(native)), native
        )

    def test_theta_rot6d_roundtrip_preserves_rotations(self):
        rng = np.random.default_rng(7)
        theta = rng.normal(0.0, 0.25, size=(4, 15, 3)).astype(np.float32)
        recovered = rot6d_to_theta(theta_to_rot6d(theta))
        np.testing.assert_allclose(recovered, theta, atol=1e-6, rtol=0.0)
