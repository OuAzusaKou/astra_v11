"""No model calls, GUI or ROS: race/cancellation and physical reset contracts."""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np

from astra.supervisor import CodexSupervisor
from astra.control import Controller, Settings
from astra.session import Session
from astra.reset_pose import load_reset_pose
from astra.keyboard import ToggleKeyboard
from astra.robot import LeoRobot
from astra.run import serve
from test_astra_mode2 import Robot, Policy, observation


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.session = Session(Path(self.temp.name) / 'run', 'test', {})
        self.bridge = CodexSupervisor(self.session, binary='/bin/echo')
        self.bridge.thread_id = '00000000-0000-0000-0000-000000000001'
        self.addCleanup(self.bridge.close)
        self.robot = Robot()
        self.controller = Controller(self.robot, Policy(), self.session, Settings(100, version=2), 'task', 'subtask')
        self.controller.start()
        self.controller.keyboard_toggle()

    def wait_result(self):
        for _ in range(100):
            result = self.bridge.poll(self.session.pending)
            if result is not None:
                return result
            time.sleep(.01)
        self.fail('bridge produced no result')

    def test_fast_medium_command_and_error_visibility(self):
        def launch(command, **kwargs):
            self.assertIn('service_tier="fast"', command)
            self.assertIn('default_permissions="astra_observe"', command)
            self.assertIn('permissions.astra_observe.extends=":read-only"', command)
            self.assertIn('permissions.astra_observe.network.enabled=true', command)
            self.assertNotIn('-s', command)
            import platform
            if platform.system() == 'Darwin':
                self.assertIn('model_providers.astra_https.supports_websockets=false', command)
            else:
                self.assertNotIn('model_providers.astra_https.supports_websockets=false', command)
            self.assertIn('features.fast_mode=true', command)
            self.assertIn('model_reasoning_effort="medium"', command)
            events = kwargs['stdout']
            events.write(json.dumps({'type':'error', 'message':'Reconnecting... 2/5'}) + '\n')
            events.write(json.dumps({'type':'thread.started', 'thread_id':self.bridge.thread_id}) + '\n')
            events.flush()
            (self.bridge.path / 'fixture.json').write_text('{"ready":true}')
            return SimpleNamespace(stdin=Mock(), poll=lambda: 0, returncode=0)
        with patch('astra.supervisor.subprocess.Popen', side_effect=launch):
            result = self.bridge._call('fixture', [], threading.Event(), 'fixture')
        self.assertEqual(result, {'ready': True})
        self.assertIn('CALL_ERROR: Reconnecting... 2/5', (self.session.path/'supervisor.log').read_text())

    def test_compaction_advisory_is_warning_but_failures_remain_errors(self):
        advisory = ('Heads up: Long threads and multiple compactions can cause the model '
                    'to be less accurate. Start a new thread when possible to keep threads small and targeted.')
        events = self.bridge.path / 'warnings.events.jsonl'
        messages = [advisory, 'Compaction failed: connection closed',
                    'Reconnecting... 2/5 (stream disconnected before completion)']
        events.write_text(''.join(json.dumps({'type': 'item.completed',
            'item': {'type': 'error', 'message': m}}) + '\n' for m in messages))
        with patch.object(self.bridge, 'log') as log:
            offset = self.bridge._report_errors(events, 0)
            self.bridge._report_errors(events, offset)
            self.assertEqual([call.args[0] for call in log.call_args_list],
                ['WARNING: ' + advisory] + ['CALL_ERROR: ' + m for m in messages[1:]])
        self.assertIsNone(self.bridge.error)

    def test_partial_error_record_waits_and_is_logged_once(self):
        events = self.bridge.path / 'partial.events.jsonl'
        events.write_text('{"type":"error","message":"retry"}')
        with patch.object(self.bridge, 'log') as log:
            offset = self.bridge._report_errors(events, 0)
            self.assertEqual(offset, 0)
            log.assert_not_called()
            with events.open('a') as f:
                f.write('\n')
            offset = self.bridge._report_errors(events, offset)
            self.bridge._report_errors(events, offset)
            log.assert_called_once_with('CALL_ERROR: retry')

    def test_takeover_response_routes_to_runner(self):
        request = self.session.pending
        decision = {k:request[k] for k in ('session_id','owner','request_id')}
        decision.update(operation='takeover', actions=np.zeros((15,52)).tolist(),
                        new_attempt=True, change_reason='synthetic', short_batch_reason='test',
                        reason='fixture', evidence='fixture')
        with patch.object(self.bridge, '_call', return_value=decision):
            d = self.wait_result()
        self.controller.decide(d)
        self.assertEqual(len(self.robot.published), 1)
        self.assertEqual(self.controller.mode, 'ASTRA')
        self.assertEqual(self.session.pending['stage'], 'VERIFY_RECOVERY')

    def test_compact_request_preserves_all_data_and_attached_images(self):
        request = self.session.pending
        request['wam_context'] = {'trajectories': [{
            'trajectory_id': 'history-1',
            'actions': [[0.123456789012345, -0.9] * 26] * 32,
            'publication': {'published_rows': [0, 15], 'reason': '中文 unchanged'},
            'observation': {'images': {'front': {'path': '/fixture/history.png'}}}}]}
        decision = {k: request[k] for k in ('session_id', 'owner', 'request_id')}
        decision.update(operation='observe', reason='fixture', evidence='fixture')
        with patch.object(self.bridge, '_call', return_value=decision) as call:
            self.wait_result()
        prompt, images = call.call_args.args[:2]
        payload = prompt.split('\nCURRENT_REQUEST:\n', 1)[1].split('\nATTACHED IMAGE ORDER:\n', 1)[0]
        self.assertEqual(json.loads(payload), request)
        self.assertEqual(images, [v['path'] for v in request['observation']['images'].values()]
                         + ['/fixture/history.png'])

    def test_handback_discards_late_response(self):
        gate = threading.Event()
        request = self.session.pending.copy()
        def delayed(*args):
            gate.wait(2)
            return {**{k: request[k] for k in ('session_id','owner','request_id')}, 'operation':'takeover'}
        with patch.object(self.bridge, '_call', side_effect=delayed):
            self.bridge.poll(request)
            self.controller.keyboard_toggle()
            self.bridge.poll(None)
            gate.set()
            self.bridge.worker.join(2)
            self.assertIsNone(self.bridge.poll(None))
        self.assertEqual(self.robot.published, [])
        self.assertEqual(self.controller.mode, 'WAM')

    def test_failure_pauses_without_actions(self):
        with patch.object(self.bridge, '_call', side_effect=RuntimeError('provider unavailable')):
            self.bridge.poll(self.session.pending)
            self.bridge.worker.join(2)
            with self.assertRaisesRegex(RuntimeError, 'provider unavailable'):
                self.bridge.poll(self.session.pending)
        self.assertEqual(self.robot.published, [])

    def test_t_returns_from_supervisor_pause_but_not_hardware_fault(self):
        self.controller.pause('no grounded plan')
        self.controller.keyboard_toggle()
        self.assertEqual(self.controller.mode, 'WAM')
        self.assertEqual(self.controller.state, 'AUTO')
        self.controller.keyboard_toggle()
        self.controller.pause('hardware fault')
        self.robot.fault = True
        with self.assertRaises(RuntimeError):
            self.controller.keyboard_toggle()
        self.assertEqual(self.controller.mode, 'ASTRA')
        self.assertEqual(self.robot.published, [])

    def test_identity_mismatch_rejected(self):
        with patch.object(self.bridge, '_call', return_value={'request_id':'old'}):
            self.bridge.poll(self.session.pending)
            self.bridge.worker.join(2)
            with self.assertRaisesRegex(RuntimeError, 'mismatched'):
                self.bridge.poll(self.session.pending)


class ResetTests(unittest.TestCase):
    def test_b_cannot_be_toggle(self):
        with self.assertRaises(ValueError):
            ToggleKeyboard('B')

    def test_reset_pose_matches_legacy_frame_zero(self):
        with tempfile.TemporaryDirectory() as d:
            data={k:np.full((2,n), i) for i,(k,n) in enumerate(
                [('left_ee_pose',6),('right_ee_pose',6),('left_hand_state',20),('right_hand_state',20)])}
            np.savez(Path(d)/'state.npz', **data)
            _,target=load_reset_pose(d)
            np.testing.assert_array_equal(target,np.concatenate([v[0] for v in data.values()]))

    def test_reset_clears_queue_and_waits_without_resetting_budget(self):
        with tempfile.TemporaryDirectory() as d:
            session=Session(Path(d)/'run','test',{})
            robot=Robot(); robot.reset_to_task_pose=Mock(return_value={'state_verified':True})
            c=Controller(robot,Policy(),session,Settings(100,version=2),'task','subtask')
            c.start(); c.auto_step(); c.steps=15
            c.operator_reset()
            self.assertEqual(c.state,'WAITING_FOR_START')
            self.assertEqual(c.steps,15)
            self.assertEqual(len(c.queue),0)
            self.assertIsNone(session.pending)
            c.keyboard_toggle()
            self.assertEqual(c.state,'WAITING_FOR_START')
            self.assertEqual(robot.published,[])

    def test_reset_fault_cannot_resume(self):
        with tempfile.TemporaryDirectory() as d:
            robot=Robot(); robot.reset_to_task_pose=Mock(side_effect=RuntimeError('ESTOP'))
            c=Controller(robot,Policy(),Session(Path(d)/'r','x',{}),Settings(100,version=2),'task','sub')
            c.operator_reset()
            self.assertEqual(c.state,'PAUSED_FOR_OPERATOR')
            self.assertEqual(robot.published,[])

    def test_reset_publishes_configured_pose_then_checks_feedback(self):
        robot=LeoRobot.__new__(LeoRobot)
        robot.check=Mock(); robot.motion_deadline=0
        robot.reset_target=np.zeros(52);robot.lower=np.full(40,-1.);robot.upper=np.ones(40)
        robot.reset_path='fixture'; robot.stop=threading.Event()
        robot.env=SimpleNamespace(_node=SimpleNamespace(get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(to_msg=lambda:'stamp'))),publish_arm_chunk=Mock(),publish_hand_chunk=Mock())
        robot.drain=Mock(); robot.observe=Mock(return_value=observation())
        receipt=robot.reset_to_task_pose()
        self.assertTrue(receipt['state_verified'])
        self.assertEqual(robot.env.publish_arm_chunk.call_args.args[1],[5.0])
        self.assertEqual(robot.env.publish_hand_chunk.call_args.args[1],[1.0])
        robot.check.side_effect=RuntimeError('ESTOP')
        robot.env.publish_arm_chunk.reset_mock()
        with self.assertRaises(RuntimeError): robot.reset_to_task_pose()
        robot.env.publish_arm_chunk.assert_not_called()

class ResetKeyboardTests(unittest.TestCase):
    def test_b_before_start_and_rearm_discard_early_enter(self):
        import os, pty
        master, slave = pty.openpty()
        stream = os.fdopen(slave, 'r')
        keyboard = ToggleKeyboard(stream=stream)
        try:
            keyboard.start(); keyboard.arm_start()
            os.write(master,b'b')
            self.assertTrue(keyboard.reset_pending.wait(1))
            keyboard.arm_start()
            self.assertFalse(keyboard.reset_pending.is_set())
            self.assertFalse(keyboard.start_requested.is_set())
            os.write(master,b'\n')
            self.assertTrue(keyboard.start_requested.wait(1))
        finally:
            keyboard.close();stream.close();os.close(master)

    def test_reset_in_runner_requires_new_enter(self):
        with tempfile.TemporaryDirectory() as d:
            stop=threading.Event(); armed=threading.Event()
            keyboard=SimpleNamespace(pending=threading.Event(),reset_pending=threading.Event(),subtask_pending=threading.Event(),
                                     start_requested=threading.Event(),closed=threading.Event())
            def arm():
                keyboard.start_requested.clear();armed.set()
            keyboard.arm_start=arm
            robot=Robot();robot.reset_to_task_pose=Mock(return_value={'state_verified':True})
            policy=Policy();c=Controller(robot,policy,Session(Path(d)/'r','x',{}),Settings(100,version=2),'task','sub')
            worker=threading.Thread(target=serve,args=(c,stop,keyboard))
            worker.start()
            try:
                self.assertTrue(armed.wait(1));armed.clear()
                keyboard.reset_pending.set()
                self.assertTrue(armed.wait(1))
                time.sleep(.1)
                self.assertEqual(c.state,'WAITING_FOR_START')
                self.assertFalse(keyboard.start_requested.is_set())
                self.assertEqual(policy.calls,[])
                robot.reset_to_task_pose.assert_called_once()
            finally:
                stop.set();worker.join(2)
