"""Exercise the real inference loop with a fake robot and policy transport."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import cv2  # Load native bindings before the temporary sys.modules ROS stubs.
import numpy as np

from test_mix_policy_client import observation, standardized_action


ROOT = Path(__file__).resolve().parents[1]


class MixInferenceLoopTest(unittest.TestCase):
    def run_loop(self, mode, publish_time=100.410, sleep_steps=0, fps=30, cycles=1,
                 temporal_ensemble=False, ensemble_max_chunks=0, action_fps=None):
        robot = mock.Mock(arm_exec_latency=0., hand_exec_latency=0.)
        stamp = SimpleNamespace(sec=int(publish_time), nanosec=round(publish_time % 1 * 1e9))
        robot._node.get_clock.return_value.now.return_value.to_msg.return_value = stamp
        robot.get_obs.return_value = dict(observation(), timestamp=100.)
        ros = mock.Mock()
        ros.ok.side_effect = [True] + [True] * cycles + [False, False]
        modules = {
            "rclpy": ros,
            "rclpy.node": mock.Mock(),
            "rclpy.executors": mock.Mock(),
            "envs.leo_wuji_env_infer": SimpleNamespace(LeoWujiInferEnv=mock.Mock(return_value=robot)),
        }
        spec = importlib.util.spec_from_file_location(
            "_mix_loop_under_test", ROOT / "scripts/inference_leo/inference_multiview_leo_sync.py"
        )
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)
        transport = mock.Mock()
        transport.send_request.side_effect = lambda request: {
            "request_id": request["request_id"], "success": True,
            "action": standardized_action(), "action_type": "relative_arm_absolute_hand",
            "server_info": {"infer_kind": "imagewam_standardized", "instruction": "task",
                            "response_action_order": "LA9,RA9,LH20,RH20", "action_row_offset": 0},
        }
        with mock.patch.object(module, "DPClient", return_value=transport), \
             mock.patch.object(module.threading, "Thread"), \
             mock.patch.object(module.time, "sleep") as sleep, \
             mock.patch.object(module.cv2, "destroyAllWindows"), \
             mock.patch.object(module, "TemporalActionEnsembler", wraps=module.TemporalActionEnsembler) as make_ensemble, \
             mock.patch("builtins.print"):
            events = mock.Mock()
            events.attach_mock(robot.get_obs, "observe")
            events.attach_mock(robot.publish_arm_chunk, "arm")
            events.attach_mock(robot.publish_hand_chunk, "hand")
            events.attach_mock(sleep, "sleep")
            module.main([
                "--config", str(ROOT / "config/client_dex_bench_mix_1.yaml"),
                "--instruction", "task", "--no-wait-for-enter", "--inference-mode", mode,
                "--async-sleep-steps", str(sleep_steps), "--fps", str(fps),
                "--temporal-ensemble" if temporal_ensemble else "--no-temporal-ensemble",
                "--temporal-ensemble-max-chunks", str(ensemble_max_chunks),
            ] + (["--action-fps", str(action_fps)] if action_fps is not None else []), use_mix_client=True)
            if temporal_ensemble:
                make_ensemble.assert_called_once_with(ensemble_max_chunks)
        self.assertEqual(transport.send_request.call_count, cycles)
        self.assertEqual(module.LeoWujiInferEnv.call_args.kwargs["fps"], fps)
        transport.disconnect.assert_called_once()
        return robot, sleep, events

    def test_no_sleep_mode_only_sleeps_during_startup(self):
        robot, sleep, _ = self.run_loop("sync-no-sleep")
        self.assertEqual(sleep.call_args_list, [mock.call(2.), mock.call(5.)])
        # Reset chunk plus the inference chunk.
        self.assertEqual(robot.publish_arm_chunk.call_count, 2)
        self.assertEqual(robot.publish_hand_chunk.call_count, 2)
        arm, times = robot.publish_arm_chunk.call_args.args
        self.assertEqual(arm.shape, (19, 12))
        np.testing.assert_allclose(times, np.arange(13, 32) / 30 - .410)

    def test_sync_mode_retains_configured_post_chunk_sleep(self):
        robot, sleep, _ = self.run_loop("sync", sleep_steps=10)
        self.assertEqual(sleep.call_args_list, [mock.call(2.), mock.call(5.), mock.call(1.5)])
        self.assertEqual(robot.publish_arm_chunk.call_args.args[0].shape, (32, 12))

    def test_entirely_expired_inference_does_not_cancel_reset_chunk(self):
        robot, sleep, _ = self.run_loop("sync-no-sleep", publish_time=102., sleep_steps=10)
        self.assertEqual(robot.publish_arm_chunk.call_count, 1)
        self.assertEqual(robot.publish_hand_chunk.call_count, 1)
        self.assertEqual(sleep.call_args_list, [mock.call(2.), mock.call(5.)])

    def test_async_waits_after_both_publishes_before_next_observation(self):
        robot, sleep, events = self.run_loop("sync-no-sleep", sleep_steps=10, cycles=2)
        np.testing.assert_allclose(
            [call.args[0] for call in sleep.call_args_list], [2., 5., 10 / 30, 10 / 30]
        )
        order = [call[0] for call in events.mock_calls]
        self.assertEqual(order[order.index("observe"):],
                         ["observe", "arm", "hand", "sleep"] * 2)
        # Waiting must not rebase the observation-aligned deadlines.
        arm, times = robot.publish_arm_chunk.call_args.args
        self.assertEqual(arm.shape, (19, 12))
        np.testing.assert_allclose(times, np.arange(13, 32) / 30 - .410)

    def test_async_wait_uses_configured_fps(self):
        _, sleep, _ = self.run_loop("sync-no-sleep", sleep_steps=10, fps=60)
        sleep.assert_called_with(10 / 60)

    def test_ensemble_and_sleep_keep_observation_deadlines(self):
        robot, sleep, _ = self.run_loop("sync-no-sleep", sleep_steps=10,
                                        temporal_ensemble=True, ensemble_max_chunks=3, cycles=2)
        np.testing.assert_allclose(robot.publish_arm_chunk.call_args.args[1],
                                   np.arange(13, 32) / 30 - .410)
        sleep.assert_called_with(10 / 30)

    def test_action_fps_changes_async_deadlines_filtering_and_wait(self):
        robot, sleep, _ = self.run_loop("sync-no-sleep", sleep_steps=10, action_fps=60,
                                        temporal_ensemble=True, ensemble_max_chunks=3)
        arm, times = robot.publish_arm_chunk.call_args.args
        self.assertEqual(arm.shape, (7, 12))
        np.testing.assert_allclose(times, np.arange(25, 32) / 60 - .410)
        np.testing.assert_allclose(robot.publish_hand_chunk.call_args.args[1], times)
        sleep.assert_called_with(10 / 60)

    def test_action_fps_changes_sync_spacing_without_changing_fixed_wait(self):
        robot, sleep, _ = self.run_loop("sync", action_fps=60)
        arm, times = robot.publish_arm_chunk.call_args.args
        self.assertEqual(arm.shape, (32, 12))
        np.testing.assert_allclose(times, np.arange(32) / 60)
        np.testing.assert_allclose(robot.publish_hand_chunk.call_args.args[1], times)
        sleep.assert_called_with(1.5)


if __name__ == "__main__":
    unittest.main()
