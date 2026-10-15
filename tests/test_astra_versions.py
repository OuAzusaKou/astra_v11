"""Offline ownership and keyboard-boundary tests; never imports ROS."""
import json
import os
from pathlib import Path
import pty
import tempfile
import termios
import threading
import time
import unittest
from unittest.mock import Mock

import numpy as np
from astra.control import Controller, Settings
from astra.keyboard import ToggleKeyboard
from astra.run import load_control_config, serve
from astra.session import Session
from test_astra_mode2 import Robot, Policy


class VersionsTests(unittest.TestCase):
    control_version = 2
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.session = Session(Path(self.temp.name) / 'run', 'astra', {})
        self.robot, self.policy = Robot(), Policy()
        self.c = Controller(self.robot, self.policy, self.session, Settings(100, version=self.control_version),
                            'put pepper in bowl', initial_l2='pick pepper')
        self.pending = threading.Event()
        self.c.switch_pending = self.pending.is_set
        self.c.start()

    def decision(self, operation, **kwargs):
        return {**{k: self.session.pending[k] for k in ('request_id', 'owner', 'session_id')},
                'operation': operation, 'reason': 'test', 'evidence': 'synthetic test', **kwargs}

    def toggle(self):
        self.pending.clear()
        self.c.keyboard_toggle()

    def test_independent_takeover_without_wam_and_reobserve(self):
        self.toggle()
        self.assertEqual(self.session.pending['wam_context']['trajectories'], [])
        self.assertEqual(self.session.pending['wam_context']['purpose'],
                         'control_learning_history_not_future_plan')
        self.c.decide(self.decision('takeover', actions=np.full((30, 52), .02).tolist(),
                                   new_attempt=True, change_reason='independent plan',
                                   execute_steps=15, short_batch_reason='partial batch test'))
        self.assertEqual(len(self.c.queue), 15)
        stale = self.decision('approve')
        self.c.decide(self.decision('observe'))
        self.assertEqual(len(self.c.queue), 0)
        self.assertEqual(self.c.mode, 'ASTRA')
        self.assertEqual(self.c.state, 'WAITING')
        self.assertEqual(self.session.pending['stage'], 'OBSERVE')
        with self.assertRaises(ValueError):
            self.c.decide(stale)
        self.c.decide(self.decision('observe'))
        self.assertEqual(len(self.robot.published), 1)
        self.c.decide(self.decision('takeover', actions=np.full((30, 52), .03).tolist(),
                                   new_attempt=False))
        self.assertEqual(len(self.robot.published), 2)
        self.assertEqual(self.policy.calls, [])
        self.toggle()
        self.assertEqual(self.c.mode, 'WAM')
        self.assertEqual(len(self.c.queue), 0)

    def test_explicit_risk_pause_preserved_during_takeover(self):
        self.toggle()
        self.c.decide(self.decision('pause', reason='unresolved collision clearance'))
        self.assertEqual(self.c.state, 'PAUSED_FOR_OPERATOR')
        self.assertEqual(self.robot.published, [])
        self.assertEqual(self.policy.calls, [])

    def test_full_30_point_takeover_executes_before_review(self):
        self.toggle()
        self.assertEqual(self.session.pending['execution_timing']['max_execute_steps'], 30)
        d = self.decision('takeover', actions=np.zeros((30, 52)).tolist(),
                          new_attempt=True, change_reason='full endpoint path')
        with self.assertRaises(ValueError):
            self.c.decide({**d, 'execute_steps': 31})
        self.c.decide(d)
        self.assertEqual([len(a) for a in self.robot.published], [30])
        self.assertEqual(len(self.c.queue), 0)
        self.assertEqual(self.session.pending['stage'], 'VERIFY_RECOVERY')
        self.assertEqual(self.c.steps, 30)
        self.assertEqual(self.policy.calls, [])

    def test_retry_paused_astra_refreshes_identity_without_motion(self):
        self.toggle()
        old = self.decision('observe')
        self.c.pause('uncertain final placement')
        steps, condition = self.c.steps, self.c.condition_version
        self.assertTrue(self.c.retry_astra())
        self.assertEqual(self.c.mode, 'ASTRA')
        self.assertEqual(self.c.state, 'WAITING')
        self.assertIsNone(self.c.pause_id)
        request = self.session.pending
        self.assertEqual(request['stage'], 'ASTRA_RETRY')
        self.assertEqual(request['retry_context']['previous_pause_reason'], 'uncertain final placement')
        self.assertNotEqual(request['request_id'], old['request_id'])
        with self.assertRaises(ValueError):
            self.c.decide(old)
        self.assertEqual((self.c.steps, self.c.condition_version), (steps, condition))
        self.assertEqual(self.robot.published, [])
        self.assertEqual(self.policy.calls, [])
        self.c.decide(self.decision('observe'))
        self.assertIsNone(self.session.pending['retry_context'])

    def test_retry_cannot_clear_hardware_fault_or_start_wam(self):
        self.assertFalse(self.c.retry_astra())
        self.toggle()
        self.c.pause('hardware fault')
        self.robot.fault = True
        pause_id = self.c.pause_id
        with self.assertRaises(RuntimeError):
            self.c.retry_astra()
        self.assertEqual(self.c.pause_id, pause_id)
        self.assertEqual(self.c.state, 'PAUSED_FOR_OPERATOR')
        self.assertEqual(self.robot.published, [])

    def test_runner_routes_r_to_fresh_supervisor_request(self):
        from types import SimpleNamespace
        keyboard = SimpleNamespace(**{name: threading.Event() for name in
            ('pending', 'reset_pending', 'subtask_pending', 'retry_pending', 'closed', 'start_requested')})
        keyboard.arm_start = lambda: keyboard.start_requested.set()
        start = self.c.start
        def paused_start():
            start()
            self.c.keyboard_toggle()
            self.c.pause('uncertain placement')
            keyboard.retry_pending.set()
        self.c.start = paused_start
        stop = threading.Event()
        def poll(request):
            if request and request['stage'] == 'ASTRA_RETRY':
                stop.set()
            return None
        supervisor = Mock(poll=poll)
        timer = threading.Timer(2, stop.set)
        timer.start()
        try:
            serve(self.c, stop, keyboard, supervisor)
        finally:
            timer.cancel()
        events = [json.loads(line) for line in (self.session.path/'events.jsonl').read_text().splitlines()]
        self.assertTrue(any(e['kind'] == 'operator_astra_retry' for e in events))
        self.assertEqual(self.policy.calls, [])
        self.assertEqual(self.robot.published, [])
        self.assertTrue(supervisor.invalidate.called)

    def test_model_runs_without_any_astra_request(self):
        self.assertEqual(self.c.state, 'AUTO')
        self.assertIsNone(self.session.pending)
        self.c.auto_step()
        self.c.auto_step()
        self.c.auto_step()
        self.assertEqual([len(x) for x in self.robot.published], [15, 15])
        self.assertIsNone(self.session.pending)
        self.assertFalse((self.session.path / 'request.json').exists())
        self.assertEqual(self.policy.calls, ['Task: put pepper in bowl Subtask: pick pepper'])

    def test_key_during_inference_discards_result_before_publish(self):
        infer = self.policy.infer
        def delayed(*args):
            self.pending.set()
            return infer(*args)
        self.policy.infer = delayed
        self.c.auto_step()
        self.c.auto_step()
        self.assertEqual(self.robot.published, [])
        self.toggle()
        self.assertEqual(self.c.mode, 'ASTRA')
        self.assertEqual(len(self.c.queue), 0)
        self.assertEqual(self.session.pending['stage'], 'ASTRA_TAKEOVER')

    def test_key_during_batch_waits_for_drain_then_exclusive_astra(self):
        self.c.auto_step()
        self.robot.drain = lambda cancelled: self.pending.set()
        self.c.auto_step()
        self.assertEqual(len(self.robot.published), 1)
        self.toggle()
        self.c.auto_step()
        self.assertEqual(len(self.robot.published), 1)
        self.assertEqual(len(self.policy.calls), 1)

    def test_switch_back_reinfers_and_rejects_late_astra_response(self):
        self.c.auto_step()
        self.toggle()
        d = self.decision('takeover', actions=np.zeros((30, 52)).tolist(),
                          new_attempt=True, change_reason='test',
                          execute_steps=15, short_batch_reason='partial batch test')
        self.c.decide(d)
        self.assertEqual(self.c.mode, 'ASTRA')
        self.assertEqual(len(self.robot.published), 1)
        self.assertEqual(len(self.c.queue), 15)
        stale = self.decision('approve')
        self.toggle()
        self.assertEqual(len(self.c.queue), 0)
        with self.assertRaises(ValueError):
            self.c.decide(stale)
        self.c.auto_step()
        self.assertEqual(len(self.policy.calls), 2)
        self.assertEqual(len(self.robot.published), 1)
        self.c.auto_step()
        self.assertEqual(len(self.robot.published), 2)
        events = [json.loads(x) for x in (self.session.path / 'events.jsonl').read_text().splitlines()]
        transitions = [e for e in events if e['kind'] == 'control_transition']
        self.assertEqual([e['current'] for e in transitions], ['WAM', 'ASTRA', 'WAM'])
        self.assertEqual([e['phase'] for e in transitions], [1, 2, 3])

    def test_astra_cannot_self_handback_in_version2(self):
        self.toggle()
        for op in ('hand_back', 'condition', 'replan'):
            with self.assertRaises(ValueError):
                self.c.decide(self.decision(op))
        self.assertEqual(self.policy.calls, [])

    def test_key_cannot_clear_fault(self):
        self.c.pause('hardware ESTOP')
        old = self.c.pause_id
        self.toggle()
        self.assertEqual(self.c.state, 'PAUSED_FOR_OPERATOR')
        self.assertEqual(self.c.pause_id, old)
        self.assertEqual(self.robot.published, [])

    def test_key_arriving_during_validation_blocks_publish(self):
        self.c.auto_step()
        self.robot.validate = lambda *args: self.pending.set()
        self.c.auto_step()
        self.assertEqual(self.robot.published, [])
        self.assertEqual(self.c.steps, 0)

    def test_auto_budget_is_enforced_without_supervisor(self):
        self.c.settings = Settings(2, version=2)
        self.c.auto_step()
        self.c.auto_step()
        self.assertEqual(self.c.state, 'FINISHED')
        self.assertEqual(self.c.result['outcome'], 'budget_exhausted')
        self.assertEqual(len(self.robot.published[0]), 6)

    def test_auto_invalid_action_pauses_in_runner_loop(self):
        stop = threading.Event()
        self.robot.reject = True
        thread = threading.Thread(target=serve, args=(self.c, stop))
        thread.start()
        try:
            deadline = time.monotonic() + 2
            while self.c.state != "PAUSED_FOR_OPERATOR":
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.01)
            self.assertEqual(self.robot.published, [])
        finally:
            stop.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())

    def test_runner_waits_for_enter_without_inference_or_actions(self):
        from types import SimpleNamespace
        stop = threading.Event()
        armed = threading.Event()
        keyboard = SimpleNamespace(pending=threading.Event(), reset_pending=threading.Event(),subtask_pending=threading.Event(), closed=threading.Event(),
                                   start_requested=threading.Event(), arm_start=armed.set)
        thread = threading.Thread(target=serve, args=(self.c, stop, keyboard))
        thread.start()
        try:
            self.assertTrue(armed.wait(1))
            time.sleep(.15)
            self.assertEqual(self.c.state, "WAITING_FOR_START")
            self.assertEqual(self.policy.calls, [])
            self.assertEqual(self.robot.published, [])
            keyboard.start_requested.set()
            deadline = time.monotonic() + 2
            while not self.robot.published:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.01)
        finally:
            stop.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())

    def test_version1_enter_then_astra_review_still_required(self):
        from types import SimpleNamespace
        self.c.settings = Settings(100, version=1)
        self.c.l2 = None
        stop = threading.Event()
        armed = threading.Event()
        keyboard = SimpleNamespace(pending=threading.Event(), reset_pending=threading.Event(),subtask_pending=threading.Event(), closed=threading.Event(),
                                   start_requested=threading.Event(), arm_start=armed.set)
        thread = threading.Thread(target=serve, args=(self.c, stop, keyboard))
        thread.start()
        try:
            self.assertTrue(armed.wait(1))
            self.assertEqual(self.c.state, "WAITING_FOR_START")
            self.assertIsNone(self.session.pending)
            keyboard.start_requested.set()
            deadline = time.monotonic() + 2
            while self.session.pending is None:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.01)
            self.assertEqual(self.session.pending["stage"], "PREPARE_SUBGOAL")
            self.assertEqual(self.policy.calls, [])
            self.assertEqual(self.robot.published, [])
        finally:
            stop.set()
            thread.join(2)

    def test_configs_target_5090_and_keep_versions_distinct(self):
        root = Path(__file__).resolve().parents[1]
        for name, version in [('version1_review', 1), ('version2_keyboard', 2), ('version3_auto_handback', 3)]:
            config = load_control_config(root / f'config/astra/{name}.json')
            self.assertEqual(config['host'], '10.128.40.203')
            self.assertEqual(config['port'], 29572)
            self.assertEqual(config['version'], version)


class Version3Tests(VersionsTests):
    # Reuse V2's startup, keyboard, reset, retry and batch-boundary contracts.
    control_version = 3

    def test_astra_cannot_self_handback_in_version2(self):
        # The one intentional difference from the inherited V2 contracts.
        self.toggle()
        for op in ('condition', 'replan'):
            with self.assertRaises(ValueError):
                self.c.decide(self.decision(op))

    def test_auto_handback_requires_evidence_and_fresh_inference(self):
        self.c.auto_step()  # WAM queue before manual takeover
        self.toggle()
        self.c.decide(self.decision('takeover', actions=np.full((30, 52), .02).tolist(),
                                   new_attempt=True, change_reason='recover', execute_steps=15,
                                   short_batch_reason='test remaining queue discard'))
        self.assertEqual(len(self.c.queue), 15)
        stale = self.decision('observe')
        with self.assertRaises(ValueError):
            self.c.decide(self.decision('hand_back'))
        self.assertEqual(self.c.mode, 'ASTRA')
        before = (self.c.steps, self.c.l2, self.c.condition_version, len(self.robot.published))
        calls = len(self.policy.calls)
        self.c.decide(self.decision('hand_back', recovery_evidence='fresh feedback shows recovered grasp'))
        self.assertEqual((self.c.mode, self.c.state), ('WAM', 'AUTO'))
        self.assertIsNone(self.session.pending)
        self.assertEqual(len(self.c.queue), 0)
        self.assertEqual(before, (self.c.steps, self.c.l2, self.c.condition_version, len(self.robot.published)))
        with self.assertRaises(ValueError):
            self.c.decide(stale)
        self.c.auto_step()
        self.assertEqual(len(self.policy.calls), calls + 1)
        self.assertIsNone(self.session.pending)
        self.toggle()
        self.assertEqual(self.session.pending['version'], 3)
        self.assertEqual(self.session.pending['stage'], 'ASTRA_TAKEOVER')

    def test_handback_cannot_change_l2_or_publish_actions(self):
        self.toggle()
        for extra in ({'l2_instruction': 'new subtask'}, {'actions': [[0.] * 52]}, {'demo_context': {}}):
            with self.assertRaises(ValueError):
                self.c.decide(self.decision('hand_back', recovery_evidence='test', **extra))
            self.assertEqual(self.c.mode, 'ASTRA')
        self.assertEqual(self.robot.published, [])



class KeyboardTests(unittest.TestCase):
    def test_real_pty_key_latches_and_restores_terminal(self):
        master, slave = pty.openpty()
        stream = os.fdopen(slave, 'r')
        original = termios.tcgetattr(slave)
        keyboard = ToggleKeyboard(stream=stream)
        try:
            keyboard.start()
            # Keystrokes during loading cannot arm motion or takeover.
            os.write(master, b'\nt')
            time.sleep(.15)
            self.assertFalse(keyboard.start_requested.is_set())
            self.assertFalse(keyboard.pending.is_set())
            keyboard.arm_start()
            os.write(master, b't')
            time.sleep(.15)
            self.assertFalse(keyboard.pending.is_set())
            os.write(master, b'\n')
            self.assertTrue(keyboard.start_requested.wait(1))
            os.write(master, b'rR')
            self.assertTrue(keyboard.retry_pending.wait(1))
            keyboard.retry_pending.clear()
            os.write(master, b'pP')
            time.sleep(.15)
            self.assertFalse(keyboard.pending.is_set())
            os.write(master, b't')
            self.assertTrue(keyboard.pending.wait(1))
            os.write(master, b'Ttt')
            time.sleep(.15)
            self.assertTrue(keyboard.pending.is_set())
            keyboard.pending.clear()
            time.sleep(.15)
            self.assertFalse(keyboard.pending.is_set())
            os.write(master, b't')
            self.assertTrue(keyboard.pending.wait(1))
        finally:
            keyboard.close()
            restored = termios.tcgetattr(slave)
            stream.close()
            os.close(master)
            # macOS can add the transient PENDIN bit while restoring a PTY.
            flags = termios.ICANON | termios.ECHO | termios.ISIG
            self.assertEqual(restored[3] & flags, original[3] & flags)
            self.assertEqual(restored[6], original[6])

    def test_version1_keyboard_accepts_enter_but_not_takeover_key(self):
        master, slave = pty.openpty()
        stream = os.fdopen(slave, 'r')
        keyboard = ToggleKeyboard(stream=stream, enable_toggle=False)
        try:
            keyboard.start()
            keyboard.arm_start()
            os.write(master, b'\n')
            self.assertTrue(keyboard.start_requested.wait(1))
            os.write(master, b'tT')
            time.sleep(.15)
            self.assertFalse(keyboard.pending.is_set())
        finally:
            keyboard.close()
            stream.close()
            os.close(master)

    def test_noninteractive_terminal_rejected(self):
        keyboard = ToggleKeyboard(stream=Mock(isatty=lambda: False))
        with self.assertRaises(ValueError):
            keyboard.start()
