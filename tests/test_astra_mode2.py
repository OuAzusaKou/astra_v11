"""Offline only: fake sensors/publishers/server. Never connects to ROS or TCP."""
import json
import ast
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from astra.control import Controller, Settings
from astra.policy import MODE2_CHECKPOINT, Mode2Policy
from astra.robot import HW_NAMES, ModeMonitor, LeoRobot, hand_limits, validate_trajectory
from astra.run import serve, parse_mode2_prompt
from astra.session import Session, submit


def observation():
    return {"image": np.full((12, 16, 3), [10, 20, 30], np.uint8),
            "image_wrist_left": np.zeros((12, 16, 3), np.uint8),
            "image_wrist_right": np.zeros((12, 16, 3), np.uint8),
            "timestamp": 1., "left_ee_pose": np.zeros((1, 6)),
            "right_ee_pose": np.zeros((1, 6)), "left_hand_joint": np.zeros((1, 20)),
            "right_hand_joint": np.zeros((1, 20))}


def subgoal():
    return {"l2_instruction": "pick up the red pepper", "demo_context": {
        "video_path": "synthetic-fixture-only", "viewed_frames": [0, 5, 10],
        "review_evidence": "synthetic test, not a real demonstration review",
        "scene_differences": "test fixture"}}


class Robot:
    def __init__(self):
        self.published = []
        self.fault = False
        self.reject = False

    def check(self):
        if self.fault:
            raise RuntimeError("ESTOP")

    def status(self):
        return {"fault": self.fault}

    def observe(self):
        self.check()
        return observation()

    def validate(self, batch, obs):
        self.check()
        if self.reject:
            raise ValueError("test limit")

    def publish(self, batch):
        self.published.append(batch.copy())
        return {"published_rows": len(batch), "hardware_execution_confirmed": False}

    def drain(self, cancelled):
        self.check()

    def rearm(self):
        self.check()


class Policy:
    def __init__(self):
        self.calls = []
        self.error = None

    def infer(self, obs, instruction):
        self.calls.append(instruction)
        if self.error:
            raise self.error
        return np.full((30, 52), .01), {"request_id": str(len(self.calls))}


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.session = Session(Path(self.temp.name) / "run", "astra", {})
        self.robot, self.policy = Robot(), Policy()
        self.c = Controller(self.robot, self.policy, self.session, Settings(100),
                            "Pick up the red pepper and place it in the bowl.",
                            initial_l2="pick up the red pepper")
        self.c.start()

    def d(self, operation, **kwargs):
        return {**{k: self.session.pending[k] for k in ("request_id", "owner", "session_id")},
                "operation": operation, "reason": "test", "evidence": "synthetic reviewed frame", **kwargs}

    def ready(self):
        self.c.decide(self.d("condition", **subgoal()))

    def test_initial_l2_cannot_be_replaced_by_supervisor(self):
        self.assertEqual(self.session.pending["initial_l2_instruction"], "pick up the red pepper")
        changed = subgoal()
        changed["l2_instruction"] = "place the red pepper in the bowl"
        with self.assertRaisesRegex(ValueError, "operator-supplied"):
            self.c.decide(self.d("condition", **changed))
        self.assertEqual(self.policy.calls, [])
        self.assertEqual(self.robot.published, [])
        self.ready()
        self.assertIn("Subtask: pick up the red pepper", self.policy.calls[0])

    def test_mode2_prompt_requires_user_l3_and_l2(self):
        self.assertEqual(parse_mode2_prompt("Task: pick and place Subtask: pick"), ("pick and place", "pick"))
        for value in ("pick and place", "Task: task Subtask: ", "Task: Subtask: pick",
                      "Task: task Subtask: pick Subtask: place"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_mode2_prompt(value)

    def test_first_inference_waits_for_subgoal_and_review(self):
        self.assertEqual(self.policy.calls, [])
        with self.assertRaises(ValueError):
            self.c.decide(self.d("approve"))
        self.ready()
        self.assertEqual(self.policy.calls, ["Task: Pick up the red pepper and place it in the bowl. Subtask: pick up the red pepper"])
        self.assertEqual(self.robot.published, [])

    def test_15_step_reviews_and_next_inference_is_not_auto_executed(self):
        self.ready()
        for _ in range(2):
            self.c.decide(self.d("approve"))
        self.assertEqual([len(a) for a in self.robot.published], [15, 15])
        self.c.decide(self.d("approve"))
        self.assertEqual(len(self.policy.calls), 2)
        self.assertEqual(len(self.robot.published), 2)

    def test_30_step_takeover_review_and_handback(self):
        self.ready()
        self.c.decide(self.d("takeover", actions=np.full((30, 52), .02).tolist(),
                             new_attempt=True, change_reason="different approach",
                             execute_steps=15, short_batch_reason="partial batch test"))
        self.assertEqual(self.c.mode, "ASTRA")
        self.assertEqual(len(self.c.queue), 15)
        with self.assertRaises(ValueError):
            self.c.decide(self.d("replan", conditioning_version=1))
        self.c.decide(self.d("approve"))
        self.assertEqual(len(self.policy.calls), 1)
        self.c.decide(self.d("takeover", actions=np.full((30, 52), .03).tolist(),
                             new_attempt=False))
        self.assertEqual((self.c.takeovers, self.c.attempts), (1, 1))
        self.c.decide(self.d("hand_back", conditioning_version=1, recovery_evidence="stable grasp"))
        self.assertEqual(self.c.mode, "WAM")
        self.assertTrue(np.all(self.c.queue == .01))
        self.assertEqual(len(self.policy.calls), 2)

    def test_attempt_count_separate_from_event_count(self):
        self.ready()
        for _ in range(2):
            self.c.decide(self.d("takeover", actions=np.zeros((30, 52)).tolist(),
                                 new_attempt=True, change_reason="new evidence"))
        self.assertEqual((self.c.takeovers, self.c.attempts), (1, 2))

    def test_invalid_or_foreign_decision_never_publishes(self):
        self.ready()
        for key in ("owner", "request_id", "session_id"):
            with self.assertRaises(ValueError):
                self.c.decide({**self.d("approve"), key: "foreign"})
        for actions in ([[0]*14], [[float("nan")]*52], [[0]*52]*31):
            with self.assertRaises(ValueError):
                self.c.decide(self.d("takeover", actions=actions, new_attempt=True, change_reason="test"))
        for op in ("abort", "reset"):
            with self.assertRaises(ValueError):
                self.c.decide(self.d(op))
        self.assertEqual(self.robot.published, [])

    def test_replay_response_rejected(self):
        self.ready()
        d = self.d("approve")
        self.c.decide(d)
        with self.assertRaises(ValueError):
            self.c.decide(d)

    def test_model_fault_and_resume_keep_session(self):
        self.policy.error = TimeoutError("model timeout")
        self.ready()
        self.assertEqual(self.c.state, "PAUSED_FOR_OPERATOR")
        pause_id = self.c.pause_id
        self.policy.error = None
        self.c.operator({"session_id": self.session.id, "pause_id": pause_id,
                         "operation": "resume", "reason": "model restored"})
        self.assertEqual(self.c.state, "WAITING")
        self.c.decide(self.d("replan", conditioning_version=1))
        self.assertEqual(self.robot.published, [])

    def test_estop_during_review_discards_all_cached_actions(self):
        self.ready()
        self.robot.fault = True
        self.c.tick()
        self.assertEqual(self.c.state, "PAUSED_FOR_OPERATOR")
        self.assertEqual(len(self.c.queue), 0)
        self.assertEqual(self.robot.published, [])

    def test_invalid_target_pauses_without_charging_budget(self):
        self.ready()
        self.robot.reject = True
        self.c.decide(self.d("approve"))
        self.assertEqual(self.c.state, "PAUSED_FOR_OPERATOR")
        self.assertEqual(self.c.steps, 0)

    def test_partial_publication_charges_budget_without_false_ack(self):
        self.ready()
        self.robot.publish = Mock(side_effect=RuntimeError("hand publisher failed"))
        self.c.decide(self.d("approve"))
        self.assertEqual((self.c.steps, self.c.published), (15, 0))
        self.assertEqual(self.c.state, "PAUSED_FOR_OPERATOR")

    def test_stagnation_cannot_be_reset_by_renaming(self):
        self.ready()
        self.c.steps = 128
        with self.assertRaises(ValueError):
            self.c.decide(self.d("approve"))
        self.c.decide(self.d("replan", **subgoal()))
        self.assertEqual(self.c.last_progress, 0)
        self.assertIsNone(self.c.result)

    def test_budget_and_visual_completion_are_distinct(self):
        self.c.settings = Settings(2)
        self.ready()
        self.c.decide(self.d("approve"))
        self.assertEqual(len(self.robot.published[0]), 6)
        self.assertEqual(self.c.result["outcome"], "budget_exhausted")
        self.assertFalse(self.c.result["hardware_execution_confirmed"])

    def test_astra_visual_and_human_completion(self):
        self.c.decide(self.d("complete"))
        self.assertEqual(self.c.result["completion_source"], "astra_visual")
        self.assertEqual(self.robot.published, [])

    def test_human_can_complete_while_paused(self):
        self.c.pause("human intervention")
        self.c.operator({"session_id": self.session.id, "pause_id": self.c.pause_id,
                         "operation": "complete", "reason": "operator checked placement"})
        self.assertEqual(self.c.result["completion_source"], "human")

    def test_timeout_does_not_approve(self):
        self.c.wait_started -= 1000
        self.c.tick()
        self.assertEqual(self.c.state, "PAUSED_FOR_OPERATOR")
        self.assertEqual(self.policy.calls, [])

    def test_file_exchange_end_to_end(self):
        stop = threading.Event()
        self.session.pending = None
        thread = threading.Thread(target=serve, args=(self.c, stop))
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while not self.session.pending:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.01)
            time.sleep(.03)
            response = self.d("condition", **subgoal())
            submit(self.session.path, response)
            while not self.session.pending or self.session.pending["stage"] != "REVIEW_ACTIONS":
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.01)
            time.sleep(.03)
            submit(self.session.path, self.d("complete"))
            thread.join(3.)
            self.assertFalse(thread.is_alive())
            self.assertEqual(self.c.result["completion_source"], "astra_visual")
            self.assertTrue((self.session.path / "summary.json").exists())
        finally:
            stop.set()
            thread.join(2.)


class AdapterTests(unittest.TestCase):
    def test_observation_wait_is_bounded_and_cancellable(self):
        # Compile only this ROS-independent wait method from the actual source;
        # importing/constructing the environment would require ROS hardware.
        source = Path(__file__).resolve().parents[1] / "envs/leo_wuji_env_infer.py"
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LeoWujiInferEnv")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "get_obs")
        scope = {"time": time}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
        dummy = SimpleNamespace(camera_deques={"image": [{"timestamp": 1.}]}, last_camera_timestamp=1.)
        with self.assertRaises(TimeoutError):
            scope["get_obs"](dummy, timeout=.002)
        stop = threading.Event()
        stop.set()
        with self.assertRaises(InterruptedError):
            scope["get_obs"](dummy, cancel_event=stop)

    def test_competing_publisher_blocks_motion(self):
        env = SimpleNamespace(_node=SimpleNamespace(get_publishers_info_by_topic=lambda topic: [1, 2]))
        robot = LeoRobot(env, Mock(), threading.Event())
        with self.assertRaisesRegex(RuntimeError, "competing"):
            robot.check()

    def test_hardware_fault_latches_even_after_running_returns(self):
        monitor = ModeMonitor()
        for name in HW_NAMES:
            monitor.update(name, "RUNNING")
        monitor.check()
        monitor.update(HW_NAMES[0], "ESTOP")
        monitor.update(HW_NAMES[0], "RUNNING")
        with self.assertRaises(RuntimeError):
            monitor.check()
        monitor.check(rearm=True)
        with patch("astra.robot.time.monotonic", return_value=time.monotonic()+10):
            with self.assertRaises(RuntimeError):
                monitor.check()

    def test_shared_positive_schedule_and_no_reset(self):
        monitor = Mock()
        clock = SimpleNamespace(to_msg=lambda: SimpleNamespace(sec=10, nanosec=0))
        env = SimpleNamespace(_resetting=False, _node=SimpleNamespace(
            get_clock=lambda: SimpleNamespace(now=lambda: clock),
            get_publishers_info_by_topic=lambda topic: [object()]),
            publish_arm_chunk=Mock(), publish_hand_chunk=Mock())
        robot = LeoRobot(env, monitor, threading.Event())
        result = robot.publish(np.zeros((15, 52)))
        arm = env.publish_arm_chunk.call_args
        hand = env.publish_hand_chunk.call_args
        self.assertIs(arm.kwargs["start_stamp"], hand.kwargs["start_stamp"])
        self.assertTrue(np.all(arm.args[1] > 0))
        np.testing.assert_allclose(np.diff(arm.args[1]), .05)
        self.assertAlmostEqual(arm.args[1][-1], .85)
        np.testing.assert_array_equal(arm.args[1], hand.args[1])
        self.assertEqual(result["published_rows"], 15)
        self.assertFalse(result["hardware_execution_confirmed"])

    def test_native_decoder_accepts_real_326_layout(self):
        from utils import liber_teleop_v22_client as codec
        identity9 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], np.float32)
        identity90 = np.tile([1, 0, 0, 0, 1, 0], 15).astype(np.float32)
        context = codec.ActionDecodeContext(np.tile(identity9, 2), np.tile(identity90, 2),
                                            np.zeros(40), np.zeros(12))
        raw = np.zeros((30, 326), np.float32)
        raw[:, :9] = identity9
        raw[:, 125:134] = identity9
        raw[:, 10:100] = identity90
        raw[:, 135:225] = identity90
        # Native hand IK is hardware-environment dependent; test the real
        # canonical/coordinate decoder with just that calibrated backend stubbed.
        with patch.object(codec, "_decode_mano_hand_chunk", return_value=np.zeros((30, 40))):
            decoded = codec.decode_action(raw, context, adapter_name="teleop_mano")
        self.assertEqual(decoded.shape, (30, 52))
        self.assertTrue(np.isfinite(decoded).all())

    def test_mode2_wire_uses_only_infer_and_checks_checkpoint(self):
        codec = SimpleNamespace(build_request=Mock(return_value=({"request_id": "r",
            "cmd": "infer_action_chunk", "observation": {}}, None)),
            decode_action=Mock(return_value=np.zeros((30, 52))))
        response = {"request_id": "r", "success": True, "action_type": "lerobot_canonical",
                    "action": np.zeros((30, 326)), "server_info": {
                        "checkpoint": MODE2_CHECKPOINT+"/step_100000.pt", "adapter_name": "teleop_mano",
                        "protocol": "liber_teleop_v22", "action_dim": 326, "action_horizon": 30}}
        connection = SimpleNamespace(send=Mock(return_value=response))
        policy = Mode2Policy(connection, "session", codec=codec)
        actions, meta = policy.infer(observation(), "Task: task Subtask: subgoal")
        self.assertEqual(actions.shape, (30, 52))
        self.assertFalse(meta["conditioning_receipt_available"])
        self.assertEqual(connection.send.call_args.args[0]["cmd"], "infer_action_chunk")
        # Explicit migrated checkpoint path is accepted only when configured.
        response["server_info"]["checkpoint"] = "/data/trans/astra_wam_mode2_20260927/mode2-7114/checkpoints/weights/step_100000.pt"
        with self.assertRaises(RuntimeError):
            policy.infer(observation(), "test")
        migrated = Mode2Policy(connection, "session", codec=codec,
                               expected_checkpoint="/mode2-7114/checkpoints/weights/step_100000.pt")
        self.assertEqual(migrated.infer(observation(), "test")[0].shape, (30, 52))
        response["server_info"]["checkpoint"] = "mode0"
        with self.assertRaises(RuntimeError):
            policy.infer(observation(), "test")

    def test_review_drift_log_only_and_optional_rejection(self):
        keys = ('left_ee_pose', 'right_ee_pose', 'left_hand_joint', 'right_hand_joint')
        obs = {k: np.zeros((1, n)) for k, n in zip(keys, (6, 6, 20, 20))}
        now = {k: v.copy() for k, v in obs.items()}
        now['left_hand_joint'][0, 1] = -.193
        robot = LeoRobot(Mock(), Mock(), threading.Event())
        robot.observe = Mock(return_value=now)
        batch = np.zeros((6, 52))
        with patch('builtins.print') as log:
            robot.validate(batch, obs)
        self.assertIn('left_hand_finger1_joint2', log.call_args.args[0])
        robot.enforce_review_state = True
        with self.assertRaisesRegex(RuntimeError, 'state changed'):
            robot.validate(batch, obs)
        robot.enforce_review_state = False
        now['left_ee_pose'][0, 0] = .2
        with self.assertRaisesRegex(ValueError, 'translation jump'):
            robot.validate(batch, obs)

    def test_v5_relaxed_hand_drift_preserves_arm_guard(self):
        keys = ('left_ee_pose', 'right_ee_pose', 'left_hand_joint', 'right_hand_joint')
        obs = {k: np.zeros((1, n)) for k, n in zip(keys, (6, 6, 20, 20))}
        now = {k: v.copy() for k, v in obs.items()}
        robot = LeoRobot(Mock(), Mock(), threading.Event(), enforce_review_state=True)
        robot.review_hand_tolerance_rad = .15
        robot.observe = Mock(return_value=now)
        with patch('astra.robot.validate_trajectory'):
            now['left_hand_joint'][0, 1] = .054219
            robot.validate(np.zeros((6,52)),obs)
            now['left_hand_joint'][0, 1] = .151
            with self.assertRaisesRegex(RuntimeError,'left_hand_finger1_joint2'):
                robot.validate(np.zeros((6,52)),obs)
            now['left_hand_joint'][0, 1] = .14
            now['left_ee_pose'][0, 0] = .051
            with self.assertRaisesRegex(RuntimeError,'left_ee_x'):
                robot.validate(np.zeros((6,52)),obs)

    def test_disabled_client_hand_limits_preserves_other_guards(self):
        lower, upper = hand_limits()
        current = np.zeros(52)
        current[12:] = (lower + upper) / 2
        current[13] = -.193
        batch = np.tile(current, (8, 1))
        batch[:, 0] += np.linspace(.002, .016, 8)
        options = dict(max_step_m=.1, max_step_rad=.75)
        with self.assertRaisesRegex(ValueError, "hand target"):
            validate_trajectory(batch, current, lower, upper, **options)
        original = batch.copy()
        validate_trajectory(batch, current, lower, upper, enforce_hand_limits=False, **options)
        np.testing.assert_array_equal(batch, original)
        batch[0, 0] = 1.
        with self.assertRaisesRegex(ValueError, "translation"):
            validate_trajectory(batch, current, lower, upper, enforce_hand_limits=False, **options)
        batch[0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, "invalid absolute"):
            validate_trajectory(batch, current, lower, upper, enforce_hand_limits=False, **options)

    def test_trajectory_guards_reject_jumps_and_invalid_hands(self):
        lower, upper = hand_limits()
        current = np.zeros(52)
        current[12:] = (lower + upper) / 2
        batch = np.tile(current, (3, 1))
        validate_trajectory(batch, current, lower, upper, max_step_m=.1, max_step_rad=.75)
        batch[0, 0] = 1
        with self.assertRaises(ValueError):
            validate_trajectory(batch, current, lower, upper, max_step_m=.1, max_step_rad=.75)
        batch = np.tile(current, (3, 1))
        batch[0, 12] = upper[0] + 1
        with self.assertRaises(ValueError):
            validate_trajectory(batch, current, lower, upper, max_step_m=.1, max_step_rad=.75)


if __name__ == "__main__":
    unittest.main()
