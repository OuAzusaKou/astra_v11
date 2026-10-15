import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

from utils.action_timing import publish_observation_aligned_chunk
from utils.temporal_ensemble import TemporalActionEnsembler


def actions(values):
    result = np.zeros((len(values), 52))
    result[:, 0] = values
    result[:, 12:] = np.asarray(values)[:, None]
    return result


class TemporalEnsembleTest(unittest.TestCase):
    def test_chunk_limit_includes_current_and_keeps_latest_predictions(self):
        for limit, expected in [(0, 15), (1, 30), (2, 25), (3, 20), (8, 15)]:
            with self.subTest(limit=limit):
                ensemble = TemporalActionEnsembler(max_chunks=limit)
                for i in range(4):
                    output = ensemble.apply(actions([i * 10, i * 10]), [i * .1, 10 + i * .1])
                # All four predictions overlap at the first deadline; only the
                # current prediction covers the last deadline.
                np.testing.assert_allclose(output[:, 0], [expected, 30])
                np.testing.assert_allclose(output[:, 12], [expected, 30])
                self.assertEqual(len(ensemble.history), min(limit or 4, 4))

    def test_interpolates_matching_times_without_extrapolating(self):
        ensemble = TemporalActionEnsembler()
        ensemble.apply(actions([0, 10, 20]), [100, 101, 102])
        output = ensemble.apply(actions([100, 100, 100]), [100.5, 101.5, 102.5])
        np.testing.assert_allclose(output[:, 0], [52.5, 57.5, 100])
        np.testing.assert_allclose(output[:, 12], output[:, 0])

    def test_stores_raw_predictions_instead_of_averaged_outputs(self):
        ensemble = TemporalActionEnsembler()
        ensemble.apply(actions([0, 0, 0]), [0, 1, 2])
        ensemble.apply(actions([10, 10, 10]), [.5, 1.5, 2.5])
        output = ensemble.apply(actions([20, 20, 20]), [1, 2, 3])
        np.testing.assert_allclose(output[:, 0], [10, 10, 20])

    def test_rotation_wrap_and_slerp(self):
        ensemble = TemporalActionEnsembler()
        past, current = actions([0, 0]), actions([0])
        past[:, 5] = np.deg2rad([170, 180])
        current[:, 5] = np.deg2rad(-175)
        past[:, 9:12] = past[:, 3:6]
        current[:, 9:12] = current[:, 3:6]
        ensemble.apply(past, [100, 101])
        output = ensemble.apply(current, [100.5])
        expected = Rotation.from_rotvec([0, 0, np.pi]).as_matrix()
        for part in [slice(3, 6), slice(9, 12)]:
            np.testing.assert_allclose(Rotation.from_rotvec(output[0, part]).as_matrix(), expected, atol=1e-10)

    def test_expired_history_and_repeated_observation(self):
        ensemble = TemporalActionEnsembler()
        ensemble.apply(actions([0, 0]), [0, 1])
        latest = actions([10, 10])
        np.testing.assert_allclose(ensemble.apply(latest, [2, 3]), latest)
        self.assertEqual(len(ensemble.history), 1)
        np.testing.assert_allclose(ensemble.apply(actions([20, 20]), [2, 3]), actions([20, 20]))
        self.assertEqual(len(ensemble.history), 1)
        self.assertEqual(ensemble.apply(np.empty((0, 52)), []).shape, (0, 52))

    def test_publish_still_filters_latency_and_preserves_deadlines(self):
        ensemble = TemporalActionEnsembler()
        ensemble.apply(actions([0, 0, 0]), [100, 100.5, 101])
        deadlines = np.array([100.2, 100.7, 101.2])
        blended = ensemble.apply(actions([10, 10, 10]), deadlines)
        env = mock.Mock(arm_exec_latency=0.1, hand_exec_latency=0.)
        env._node.get_clock.return_value.now.return_value.to_msg.return_value = SimpleNamespace(sec=100, nanosec=300000000)
        count, _ = publish_observation_aligned_chunk(env, blended[:, :12], blended[:, 12:], deadlines - 100.2, 100.2)
        self.assertEqual(count, 2)
        arm, offsets = env.publish_arm_chunk.call_args.args
        np.testing.assert_allclose(arm[:, 0], [5, 10])
        np.testing.assert_allclose(offsets + 100.3, deadlines[1:])


if __name__ == "__main__":
    unittest.main()
