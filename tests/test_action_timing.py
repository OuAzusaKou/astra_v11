from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from utils.action_timing import publish_observation_aligned_chunk


class ActionTimingTest(unittest.TestCase):
    def env(self, seconds):
        env = mock.Mock(arm_exec_latency=0.0, hand_exec_latency=0.0)
        stamp = SimpleNamespace(sec=int(seconds), nanosec=round((seconds % 1) * 1e9))
        env._node.get_clock.return_value.now.return_value.to_msg.return_value = stamp
        return env

    def test_camera_plus_inference_latency_preserves_deadlines_and_row_pairing(self):
        # Camera correction already happened: capture=100, receive=100.160,
        # publish=100.410 (including inference/transport/decode).
        env = self.env(100.410)
        arm = np.repeat(np.arange(32)[:, None], 12, axis=1)
        hand = np.repeat(np.arange(32)[:, None], 40, axis=1)
        count, age = publish_observation_aligned_chunk(env, arm, hand, np.arange(32) / 30, 100.0)
        self.assertEqual(count, 19)
        self.assertAlmostEqual(age, 0.410)
        arms, times = env.publish_arm_chunk.call_args.args
        hands, hand_times = env.publish_hand_chunk.call_args.args
        np.testing.assert_array_equal(arms[:, 0], np.arange(13, 32))
        np.testing.assert_array_equal(hands[:, 0], arms[:, 0])
        np.testing.assert_allclose(times + 100.410, 100.0 + np.arange(13, 32) / 30)
        np.testing.assert_array_equal(times, hand_times)
        self.assertEqual(env.publish_arm_chunk.call_args.kwargs, env.publish_hand_chunk.call_args.kwargs)
        env._node.get_clock.return_value.now.assert_called_once()

    def test_entirely_expired_chunk_does_not_publish_or_cancel(self):
        env = self.env(102.0)
        count, _ = publish_observation_aligned_chunk(env, np.zeros((32, 12)), np.zeros((32, 40)), np.arange(32) / 30, 100.)
        self.assertEqual(count, 0)
        env.publish_arm_chunk.assert_not_called()
        env.publish_hand_chunk.assert_not_called()

    def test_discard_offset_and_strict_expiry_boundary(self):
        env = self.env(10.5)
        # Original rows 2..5 at 4Hz, with dataset row offset 0.
        count, _ = publish_observation_aligned_chunk(env, np.zeros((4, 12)), np.zeros((4, 40)), np.arange(2, 6) / 4, 10.)
        self.assertEqual(count, 3)
        np.testing.assert_array_equal(env.publish_arm_chunk.call_args.args[1], [0.25, 0.5, 0.75])

    def test_execution_latency_keeps_arms_and_hands_on_same_rows(self):
        env = self.env(10.)
        env.hand_exec_latency = 0.1
        count, _ = publish_observation_aligned_chunk(env, np.zeros((3, 12)), np.zeros((3, 40)), np.array([.05, .1, .2]), 10.)
        self.assertEqual(count, 1)

if __name__ == "__main__":
    unittest.main()
